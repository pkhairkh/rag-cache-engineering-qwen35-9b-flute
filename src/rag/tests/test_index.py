"""test_index.py — W6.3: acceptance gates for src/rag/index.py (W6.1+W6.2:
the §8 IVFADC build + §6 preselect/rerank, PROPOSAL D5 streaming).

Gates (the W6 definition of done):
  1.  PARAM FIDELITY: IndexConfig() defaults are EXACTLY the spec
      constants — d=13,631,488 (§4 arithmetic 24*524288 + 2*524288),
      nlist=224, m=64, nbits=8, nprobe=8 (§8), preselect_k=100,
      rerank_k=3 (§6 steps 3-4).  Asserted literally.
  2.  REDUCED-DIM BUILD + RECALL MECHANICS: d=128/nlist=8/m=16/nbits=8,
      800 random unit vectors (faiss trains fine at this scale — the box
      cannot hold full-d indexes), min_train_warn=100.  20 queries =
      (source vector + small noise): the exact source id is inside every
      top-100 preselect, and inside the rerank top-3 for >= 18/20
      (tolerance for PQ/ADC approximation noise; measured on this box:
      20/20, mean rank 1.0).
  3.  ChunkVectorLoader vs REAL W5 ARTIFACTS: a stub-model corpus built
      exactly like test_ingest.py's (the house StubModel pattern); the
      loader's vector(i) is fp32, length == manifest vector_dims, and
      equals the ingest-time record.cache_vector within rel-MSE < 0.10 —
      the DOUBLE-ROUND budget (disk: dequant(sys)+dequant(delta), ingest:
      dequant(absolute codes); one extra quant round on the delta;
      measured max ~0.013).  iter_vectors yields chunk-id order.
  4.  SIDE-METADATA ROUND-TRIP: build_index(path=...) -> load_index ->
      config fields + seeds + codebook hashes + extra_metadata survive;
      is_trained/ntotal/nprobe preserved; a search still works.
  5.  LAZY RERANK: a counting mock loader proves rerank touches EXACTLY
      the candidates it was given, once each (duplicates coalesced, no
      non-candidate loads, no extra loads) and matches brute-force
      cosine top-3.
  6.  LOUD REFUSALS: stream/config dim mismatch; m not dividing d (at
      construction AND on a mutated config at build); ChunkVectorLoader
      on a dir without system_state.npz; without a manifest; unknown
      chunk id.
  7.  MINI END-TO-END (W6 capstone): a 320-chunk stub corpus (NOT the
      mission sketch's 60: faiss's PQ stage at nbits=8 needs >= 256 train
      vectors — Clustering.cpp asserts nx >= k=256 — and 320 also clears
      the 39*nlist=312 coarse rule of thumb, so the build is warning-
      free), index built from loader.iter_vectors() with path=, reloaded
      from disk, queried with (held-out chunk vector + noise) ->
      preselect -> rerank: the source chunk is in the top-3 (measured
      10/10; gate >= 9/10).  The toy §12's 100%-retrieval is the ceiling;
      this is the mechanics-level version.

Determinism: every random draw is pinned (numpy default_rng seeds /
torch.Generator seeds); the stub keys its per-input RNG on zlib.crc32 of
the token ids (Python's hash() is process-salted — the test_ingest.py
lesson).  faiss's clustering is seeded, so the whole suite is
reproducible; the recall gates carry 18/20 + 9/10 tolerances anyway.
"""
from __future__ import annotations

import json
import os
import zlib
from collections import Counter

import numpy as np
import pytest
import torch

from index import (
    ChunkVectorLoader,
    IndexConfig,
    build_index,
    codebook_sha256,
    load_index,
    preselect,
    rerank,
)
from ingest import MANIFEST_NAME, IngestDriver
from tq_cache import TQCache
from turboquant import SEEDS

# ------------------------------------------------------------- geometry ---
# The test_ingest.py stub geometry (house pattern): power-of-two FHT units
# at the committed codebook scale d=128; non-contiguous linear indices
# {0, 2, 3} exercise the reseed/reconstruction loops; M1/M2 (2, 4, 16).
LIN, FULL = "linear_attention", "full_attention"
LAYER_TYPES = [LIN, FULL, LIN, LIN, FULL]
LINEARS = [i for i, lt in enumerate(LAYER_TYPES) if lt == LIN]
assert LINEARS == [0, 2, 3]

S_SHAPE = (1, 8, 16)
S_D = 128
CONV_D = 32
KERNEL = 4
M_SHAPE = (2, 4, 16)
M_D = 128
BITS = 3.5
VECTOR_DIMS = len(LINEARS) * S_D + 2 * M_D          # 640, §4 order

# The W6.3 documented double-round budget (cf. W5.3's 0.10 gate for the
# D4 delta reconstruction; measured here ~0.013).
DOUBLE_ROUND_GATE = 0.10


def _rel_mse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(((a - b) ** 2).sum() / max(((b ** 2).sum()), 1e-30))


def _tokens(seed: int, n: int) -> torch.Tensor:
    return torch.randint(0, 100_000, (1, n),
                         generator=torch.Generator().manual_seed(seed))


def _key_hash(ids: torch.Tensor) -> int:
    """DETERMINISTIC per-input hash (Python's hash() is process-salted)."""
    key = tuple(int(t) for t in ids.flatten().tolist())
    return zlib.crc32(np.asarray(key, dtype=np.int64).tobytes())


def _make_cache() -> TQCache:
    return TQCache(layer_types=LAYER_TYPES, bits=BITS)


# ------------------------------------------------------------- the stub ---
class StubModel:
    """test_ingest.py's StubModel pattern, compact form: per linear layer
    read recurrent_states[0] (None -> zeros), add a deterministic
    increment, write back (quantize-on-write); write a random conv input
    through update_conv_state; M1/M2 read -> gated increment -> write."""

    def __init__(self, s_noise: float = 0.3, m1_noise: float = 0.1,
                 m2_noise: float = 0.1):
        self.linears = LINEARS
        self.s_noise = s_noise
        self.m1_noise = m1_noise
        self.m2_noise = m2_noise

    def __call__(self, input_ids, past_key_values, use_cache=True):
        kh = _key_hash(input_ids)
        n_tok = int(input_ids.shape[-1])
        for L in self.linears:
            cur = past_key_values.layers[L].recurrent_states[0]
            cur = torch.zeros(S_D) if cur is None else cur.reshape(-1).float()
            g = torch.Generator().manual_seed(1_000_003 * L + kh)
            new = cur + self.s_noise * torch.randn(S_D, generator=g)
            past_key_values.update_recurrent_state(
                new.reshape(S_SHAPE).half(), L)
            g = torch.Generator().manual_seed(1_000_003 * L + 17 * kh)
            conv_in = torch.randn(1, CONV_D, n_tok, generator=g)
            past_key_values.update_conv_state(
                conv_in.half(), L, conv_kernel_size=KERNEL)
        for which, noise, m_shape in (("m1", self.m1_noise, M_SHAPE),
                                      ("m2", self.m2_noise, M_SHAPE)):
            cur = getattr(past_key_values, f"read_{which}")()
            if cur is None:
                cur = torch.zeros(*m_shape, dtype=torch.float16)
            g = torch.Generator().manual_seed(9001 + kh if which == "m1"
                                              else 9002 + kh)
            new = cur + noise * torch.randn(*m_shape, generator=g)
            getattr(past_key_values, f"update_{which}")(new.half())
        return None


def _ingest_stub(out_dir, n_chunks: int, seed0: int = 100,
                 return_records: bool = False):
    """Ingest a deterministic stub corpus; return the driver."""
    chunks = [_tokens(seed0 + i, 64) for i in range(n_chunks)]
    drv = IngestDriver(StubModel(), _tokens(7, 16), chunks, str(out_dir),
                       cache_factory=_make_cache, bits=BITS,
                       return_records=return_records)
    drv.run()
    return drv


# ==================================================== 1. param fidelity ====
def test_param_fidelity():
    """Gate 1: IndexConfig() defaults are EXACTLY the spec constants."""
    cfg = IndexConfig()
    assert cfg.d == 13_631_488                      # spec §8 (verbatim)
    assert cfg.nlist == 224                         # spec §8
    assert cfg.m == 64                              # spec §8
    assert cfg.nbits == 8                           # spec §8
    assert cfg.nprobe == 8                          # spec §8
    assert cfg.preselect_k == 100                   # spec §6 step 3
    assert cfg.rerank_k == 3                        # spec §6 step 4
    # the §4/§8 arithmetic the spec spells out
    assert cfg.d == 24 * 524_288 + 2 * 524_288 == 26 * 524_288
    assert cfg.d % cfg.m == 0                       # m=64 divides d
    assert cfg.d // cfg.m == 212_992
    # spec §8's own construction shape is legal at these constants
    assert IndexConfig(d=cfg.d, nlist=cfg.nlist, m=cfg.m, nbits=cfg.nbits)


# ================================= 2. reduced-dim build + recall mechanics =
def test_recall_mechanics(tmp_path):
    """Gate 2: d=128 build, 800 unit vectors, 20 (source + noise) queries.

    Tolerance: >= 18/20 sources in the rerank top-3 (PQ/ADC approximation
    noise; measured 20/20 with mean rank 1.0 on this box).
    """
    rng = np.random.default_rng(20260506)
    d = 128
    X = rng.standard_normal((800, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cfg = IndexConfig(d=d, nlist=8, m=16, nbits=8)

    index = build_index(iter(X), cfg, train_sample=4096,
                        min_train_warn=100, extra_metadata={"corpus":
                                                            "synthetic-800"})
    assert index.is_trained
    assert index.ntotal == 800
    assert index.nprobe == cfg.nprobe == 8
    assert int(index.metric_type) == 0  # faiss.METRIC_INNER_PRODUCT

    class ArrayLoader:                    # the mock the rerank consumes
        def __init__(self, mat):
            self.mat = mat

        def vector(self, cid):
            return self.mat[int(cid)]

    hits, ranks = 0, []
    for _ in range(20):
        src = int(rng.integers(0, 800))
        query = X[src] + 0.03 * rng.standard_normal(d)
        ids = preselect(index, query, k=100)
        assert ids.dtype == np.int64
        assert len(ids) <= 100 and -1 not in ids
        assert len(set(ids.tolist())) == len(ids)   # IVF: one list each
        assert src in ids.tolist()                  # preselect holds it
        top, scores = rerank(ArrayLoader(X), query, ids, k=3)
        assert top.shape == (3,) and scores.shape == (3,)
        assert np.all(scores[:-1] >= scores[1:])    # descending
        if src in top.tolist():
            hits += 1
            ranks.append(1 + top.tolist().index(src))
        else:
            ranks.append(4)
    assert hits >= 18, f"recall too low: {hits}/20, mean rank {np.mean(ranks):.2f}"


# ================================ 3. ChunkVectorLoader vs real artifacts ==
@pytest.fixture(scope="module")
def stub6(tmp_path_factory):
    """A 6-chunk stub corpus + its ingest records (module-shared)."""
    out = tmp_path_factory.mktemp("stub6")
    drv = _ingest_stub(out, 6, return_records=True)
    return str(out), drv


def test_loader_real_artifacts(stub6):
    """Gate 3: the loader reconstructs the §4 vector from the W5 delta
    codes within the documented double-round budget."""
    out_dir, drv = stub6
    loader = ChunkVectorLoader(out_dir)
    with open(os.path.join(out_dir, MANIFEST_NAME)) as f:
        manifest = json.load(f)

    assert loader.chunk_ids() == [0, 1, 2, 3, 4, 5]
    assert loader.dims == manifest["vector_dims"] == VECTOR_DIMS == 640

    v0 = loader.vector(0)
    assert v0.dtype == np.float32
    assert v0.ndim == 1 and v0.shape == (VECTOR_DIMS,)

    # the double-round gate: disk (dequant(sys)+dequant(delta)) vs the
    # ingest-time single-round vector (dequant of absolute codes)
    worst = 0.0
    for rec in drv.records:
        v = loader.vector(rec.chunk_idx)
        worst = max(worst, _rel_mse(v, rec.cache_vector))
    assert worst < DOUBLE_ROUND_GATE, f"worst rel-MSE {worst:.4f}"

    # iter_vectors: chunk-id order, values identical to vector()
    yielded = [v for v in loader.iter_vectors()]
    assert len(yielded) == 6
    for cid, v in zip(loader.chunk_ids(), yielded):
        assert np.array_equal(v, loader.vector(cid))
    # explicit id list: the given order
    sel = [v for v in loader.iter_vectors([3, 1])]
    assert np.array_equal(sel[0], loader.vector(3))
    assert np.array_equal(sel[1], loader.vector(1))


# ===================================== 4. side-metadata round-trip ========
def test_side_metadata_roundtrip(tmp_path):
    """Gate 4: build_index(path=...) -> load_index -> everything survives."""
    rng = np.random.default_rng(20260507)
    d = 64
    X = rng.standard_normal((300, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cfg = IndexConfig(d=d, nlist=4, m=8, nbits=8, nprobe=8)
    extra = {"corpus": "meta-roundtrip", "unit_dims": [128], "trial": 1}
    path = str(tmp_path / "mini.index")

    built = build_index(iter(X), cfg, train_sample=4096, min_train_warn=50,
                        path=path, extra_metadata=extra)
    assert built.is_trained and built.ntotal == 300 and built.nprobe == 8
    assert os.path.isfile(path) and os.path.isfile(path + ".meta.json")

    index, meta = load_index(path)
    assert index.is_trained
    assert index.ntotal == 300
    assert index.nprobe == 8                      # restored from metadata

    for field in ("d", "nlist", "m", "nbits", "nprobe", "preselect_k",
                  "rerank_k"):
        assert meta[field] == getattr(cfg, field), field
    assert meta["metric"] == "inner_product"
    assert meta["seeds"] == {k: int(v) for k, v in SEEDS.items()}
    assert meta["extra_metadata"] == extra
    assert meta["counts"] == {"train_vectors": 300, "indexed_vectors": 300}

    # codebook hashes: the pinned production frame (D2/D3), recomputable
    assert set(meta["codebook_sha256"]) == {"b3_d524288", "b4_d524288"}
    assert meta["codebook_sha256"]["b3_d524288"] == codebook_sha256(3, 524_288)
    assert meta["codebook_sha256"]["b4_d524288"] == codebook_sha256(4, 524_288)
    assert all(len(h) == 64 for h in meta["codebook_sha256"].values())

    # the reloaded index still serves §6 step 3
    ids = preselect(index, X[7], k=10)
    assert 7 in ids.tolist()


# ================================================= 5. lazy rerank =========
class CountingLoader:
    """Mock loader that counts .vector() calls (the laziness probe)."""

    def __init__(self, vectors: dict):
        self.vectors = {int(k): v for k, v in vectors.items()}
        self.calls = []

    def vector(self, cid):
        self.calls.append(int(cid))
        return self.vectors[int(cid)]


def test_lazy_rerank():
    """Gate 5: rerank touches EXACTLY the candidates, once each, and is
    brute-force-correct."""
    rng = np.random.default_rng(20260508)
    d = 64
    vectors = {i: rng.standard_normal(d) for i in range(120)}
    loader = CountingLoader(vectors)

    candidates = list(range(10, 110))            # 100 ids; 20 non-candidates
    query = rng.standard_normal(d)

    top, scores = rerank(loader, query, candidates + [candidates[0]], k=3)
    # the duplicated id was coalesced: 101 entries -> exactly 100 loads
    assert len(loader.calls) == 100
    assert Counter(loader.calls) == Counter(candidates)
    assert not set(loader.calls) - set(candidates)     # no non-candidates
    assert loader.calls == candidates                  # in arrival order

    # brute-force check: exact cosine, descending
    qn = float(np.linalg.norm(query))
    expected = []
    for c in candidates:
        v = vectors[c]
        vn = float(np.linalg.norm(v))
        expected.append(float(np.dot(query, v) / (qn * vn)))
    order = sorted(range(len(candidates)), key=lambda i: (-expected[i], i))
    assert top.tolist() == [candidates[i] for i in order[:3]]
    assert np.allclose(scores, np.array(expected)[order[:3]], atol=1e-6)


# ================================================ 6. loud refusals =========
def test_loud_refusals(tmp_path, stub6):
    """Gate 6: dim mismatch, m not dividing d, missing artifacts."""
    rng = np.random.default_rng(20260509)

    # (a) config.d != the streamed vectors' dim
    wrong = [rng.standard_normal(64) for _ in range(10)]
    with pytest.raises(ValueError, match="config.d"):
        build_index(iter(wrong), IndexConfig(d=128, nlist=4, m=16))

    # (b) m does not divide d — refused at construction...
    with pytest.raises(ValueError, match="must divide"):
        IndexConfig(d=128, nlist=8, m=7)
    # ... AND at build (dataclasses are mutable — re-validation)
    cfg = IndexConfig(d=128, nlist=8, m=16)
    cfg.m = 7
    with pytest.raises(ValueError, match="must divide"):
        build_index(iter(wrong[:0] or [np.ones(128)]), cfg)

    # (c) ChunkVectorLoader on a dir without system_state.npz
    empty = tmp_path / "no_state"
    empty.mkdir()
    with pytest.raises(ValueError, match="system_state"):
        ChunkVectorLoader(str(empty))

    # (c') system_state present but the manifest is missing
    out_dir, _ = stub6
    half = tmp_path / "no_manifest"
    half.mkdir()
    with open(os.path.join(out_dir, "system_state.npz"), "rb") as f:
        blob = f.read()
    with open(os.path.join(half, "system_state.npz"), "wb") as f:
        f.write(blob)
    with pytest.raises(ValueError, match="manifest"):
        ChunkVectorLoader(str(half))

    # (c'') unknown chunk id
    loader = ChunkVectorLoader(out_dir)
    with pytest.raises(ValueError, match="not in the manifest"):
        loader.vector(999)

    # (d) an empty stream is refused loudly
    with pytest.raises(ValueError, match="empty"):
        build_index(iter([]), IndexConfig(d=64, nlist=2, m=8),
                    min_train_warn=4)

    # (e) zero-norm corpus vectors cannot enter the cosine frame
    with pytest.raises(ValueError, match="zero-norm"):
        build_index(iter([np.zeros(64)]), IndexConfig(d=64, nlist=2, m=8),
                    min_train_warn=4)


# ================================= 7. mini end-to-end (W6 capstone) =======
def test_end_to_end_mini(tmp_path):
    """Gate 7: stub corpus -> streamed build -> disk -> preselect -> rerank.

    320 chunks (faiss PQ at nbits=8 requires >= 256 train vectors and 320
    clears 39*nlist=312, keeping the coarse stage warning-free); 10
    (held-out chunk vector + noise) queries through the RELOADED index;
    the source chunk must be in the rerank top-3 (measured 10/10, gate
    >= 9/10 — the PQ/ADC tolerance).
    """
    out_dir = str(tmp_path / "corpus")
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = 320
    _ingest_stub(out_dir, n_chunks)
    loader = ChunkVectorLoader(out_dir)
    assert loader.chunk_ids() == list(range(n_chunks))

    cfg = IndexConfig(d=loader.dims, nlist=8, m=8, nbits=8)
    path = str(tmp_path / "ivfadc.index")
    build_index(loader.iter_vectors(), cfg, train_sample=4096,
                path=path, extra_metadata={"corpus": "stub-320"})
    index, meta = load_index(path)
    assert index.ntotal == n_chunks
    assert meta["d"] == loader.dims == 640
    assert meta["counts"]["indexed_vectors"] == n_chunks

    rng = np.random.default_rng(20260510)
    hits = 0
    for _ in range(10):
        src = int(rng.integers(0, n_chunks))
        v = loader.vector(src)
        query = v + 0.1 * float(np.linalg.norm(v)) * rng.standard_normal(len(v))
        ids = preselect(index, query, k=100)
        assert src in ids.tolist()
        top, scores = rerank(loader, query, ids, k=3)
        assert np.all(scores[:-1] >= scores[1:])
        if src in top.tolist():
            hits += 1
    assert hits >= 9, f"end-to-end recall {hits}/10"
