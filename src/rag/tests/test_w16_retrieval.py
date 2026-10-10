"""test_w16_retrieval.py — the W16 retrieval gates: the centered frame
converts the common-component cosine collapse into a healthy-margin
retrieval, and the answer flow's full-attention layers run fresh.

THE W16 DIAGNOSIS (scripts/w16_probe_retrieval.py + the box): spec N17
says "cos-sim measures content overlap = relevance", but the ABSOLUTE §4
vectors on both sides carry a large COMMON component — the system
prompt's state plus the generic-text response. The box measured the
collapse: top-3 cosines [0.7232, 0.7072, 0.7071] — a floor with a 0.016
spread and the gold document OUTSIDE the top-3. The fix centers both
sides (sys + corpus-mean subtraction — RetrievalFrame, index.py), a
metric-level correction that stays inside the §4 cache-state space (no
embedder, no chunk text, no hidden states).

THE STUB (CommonComponentStub): the TopicStubModel of test_query.py
recalibrated to the box's failure regime — every update (system, chunk,
query) adds a large SHARED direction at COMMON magnitude (the
generic-text response) on top of the topic marker and the noise, with
per-chunk norm jitter (the chunk-length effect). MEASURED at these
constants (the real TQ machinery, d=128):

  * the ABSOLUTE frame's top-3 scores sit at a 0.9991 floor with a
    ~2e-7 spread — the score carries almost NO information (the box's
    0.70-floor signature at stub scale); its topic margin is ~1e-3,
    a couple of noise sigma — fragile;
  * the CENTERED frame's top-3 scores spread over ~0.02 with a topic
    margin of ~0.1+ — the discrimination the metric was supposed to
    deliver.

NOTE (honest scope): at the stub's favorable query SNR the absolute
frame may still HAPPEN to rank the topic on top (its margin is ~1e-3);
the box's real query↔doc signal sits below that (hence the wrong-doc
retrievals). The gates below pin the MECHANISM (margin amplification +
floor removal + end-to-end integration); scripts/gpu/eval_retrieval.py
measures the real corpus's regime on the GPU box.

GATES:
  1. THE FLOOR (the box's signature): the absolute rerank's top-3
     scores are a ~1.0 floor with a < 1e-5 spread.
  2. THE MARGIN (the fix's mechanism): the discrimination gap (best
     topic-t score − best non-topic-t) is > 3x larger in the centered
     frame, and decisively positive (> 0.05).
  3. THE RETRIEVAL: the centered rerank's top-3 is topic-pure, top-1 in
     the query's topic, with a real spread (> 100x the absolute's).
  4. FRAME PERSISTENCE: save/load roundtrip bit-exact; dims / system_ref
     drift and non-finite frames refuse loudly.
  5. E2E CENTERED RETRIEVAL: build_index over the CENTERED stream +
     answer_query (the frame auto-detected from retrieval_frame.npz)
     retrieves the topic.
  6. FULL-ATTN FRESH (the answer-flow fix): a stub that writes full-attn
     KV through the cache layer API; after answer_query the full-attn
     layers hold EXACTLY the query tokens once (the pre-W16 flow
     prefilled the query TWICE — step 2's capture + step 7's answer).
"""
from __future__ import annotations

import os
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import _paths  # noqa: F401
from ingest import IngestDriver, load_system_state, reseed_cache
from index import (ChunkVectorLoader, IndexConfig, RetrievalFrame,
                   build_index, build_retrieval_frame, iter_centered_vectors,
                   load_index, load_retrieval_frame, rerank,
                   save_retrieval_frame)
from query import answer_query, query_cache_vector
from tq_cache import TQCache

# ------------------------------------------------------------- geometry ---
LIN, FULL = "linear_attention", "full_attention"
LAYER_TYPES = [LIN, FULL, LIN, FULL]
LINEARS = [0, 2]
S_SHAPE, S_D = (1, 8, 16), 128
CONV_D, KERNEL = 32, 4
M_SHAPE, M_D = (2, 4, 16), 128
BITS = 3.5

N_TOPICS = 5
N_CHUNKS = 260                    # faiss PQ nbits=8: nx >= 256
MARKER, NOISE, M_NOISE = 6.0, 0.05, 0.1
COMMON = 40.0                     # the generic-text response (the floor)
QMARKER = 2.0                     # the query's WEAK topic signal
JITTER = 0.25                     # per-chunk norm modulation (length)

QUERY_IDS = torch.tensor([[200, 201]]).cuda()
SYSTEM_IDS = torch.tensor([[5, 6, 7, 8]]).cuda()
QUERY_TOPIC = 2


def _make_cache() -> TQCache:
    return TQCache(layer_types=LAYER_TYPES, bits=BITS)


# ------------------------------------------------------------- the stub ---
class CommonComponentStub:
    """The W7.3 TopicStubModel recalibrated to the box's failure regime:
    every update (system, chunk AND query) adds the SHARED generic
    direction at COMMON magnitude (English prose produces it too), the
    topic marker at MARKER (chunks) / QMARKER (the weak query signal),
    and per-chunk norm jitter. Cache traffic is exactly GatedDeltaNet's
    (the TQCache read/write path — quantize-on-write, dequantize-on-read).

    The chunk's topic is keyed by its TOKENS (baked in at construction);
    the query's topic comes from active_topic (None outside queries)."""

    def __init__(self, n_topics: int = N_TOPICS):
        g = torch.Generator().manual_seed(99)
        self.topic_dirs = [torch.randn(S_SHAPE, generator=g).cuda()
                           for _ in range(n_topics)]
        self.common_dir = torch.randn(S_SHAPE, generator=g).cuda()
        self.chunk_topics: dict = {}   # tuple(tokens) -> topic
        self.chunk_jitter: dict = {}   # tuple(tokens) -> the norm scale
        self.active_topic = None       # the QUERY's topic

    def __call__(self, input_ids, past_key_values, use_cache=True):
        ids = tuple(int(t) for t in input_ids.flatten().tolist())
        kh = zlib.crc32(np.asarray(ids, dtype=np.int64).tobytes())
        scale = self.chunk_jitter.get(ids, 1.0)
        topic = self.chunk_topics.get(ids, self.active_topic)
        for L in LINEARS:
            cur = past_key_values.layers[L].recurrent_states[0]
            if cur is None:
                cur = torch.zeros(S_SHAPE, dtype=torch.float16, device='cuda')
            else:
                cur = cur.cuda() if not cur.is_cuda else cur
            g = torch.Generator().manual_seed(31 * (L + 1) + kh)
            add = NOISE * torch.randn(S_SHAPE, generator=g).cuda() \
                + scale * COMMON * self.common_dir
            if topic is not None:
                marker = scale * MARKER if ids in self.chunk_topics \
                    else QMARKER
                add = add + marker * self.topic_dirs[topic]
            new = (cur.float() + add.float()).half().cuda()
            past_key_values.update_recurrent_state(new, L)
            conv_in = torch.randn(1, CONV_D, max(1, len(ids)), generator=g)
            past_key_values.update_conv_state(
                conv_in.half().cuda(), L, conv_kernel_size=KERNEL)
        for which, seed in (("m1", 7001), ("m2", 7002)):
            m = getattr(past_key_values, f"read_{which}")()
            if m is None:
                m = torch.zeros(*M_SHAPE, dtype=torch.float16, device='cuda')
            else:
                m = m.cuda() if not m.is_cuda else m
            g = torch.Generator().manual_seed(seed + kh)
            getattr(past_key_values, f"update_{which}")(
                (m.float() + M_NOISE * torch.randn(*M_SHAPE,
                                                   generator=g).cuda()).half().cuda())
        return torch.zeros(1, max(1, len(ids)), N_TOPICS)


# ------------------------------------------------------- module fixture ---
@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The 260-chunk common-component corpus: ingested once (absolute
    layout, the W16 default), the centered frame built + saved once, the
    faiss index built over the CENTERED stream once (~4 s)."""
    disk = str(tmp_path_factory.mktemp("w16_retrieval"))
    model = CommonComponentStub()
    chunks = []
    for i in range(N_CHUNKS):
        tok = (100 + 7 * i, 101 + 7 * i, 102 + 7 * i)
        model.chunk_topics[tok] = i % N_TOPICS
        model.chunk_jitter[tok] = 1.0 + JITTER * (
            ((i * 2654435761) % 1000) / 1000.0 - 0.5)
        chunks.append(torch.tensor([list(tok)]).cuda())
    assert model.active_topic is None
    IngestDriver(model, SYSTEM_IDS, chunks, disk,
                 cache_factory=_make_cache).run()

    loader = ChunkVectorLoader(disk)
    frame = build_retrieval_frame(loader)
    save_retrieval_frame(loader, frame)
    cfg = IndexConfig(d=loader.dims, nlist=8, m=8, nbits=8, nprobe=8)
    ipath = os.path.join(disk, "ivfadc_cache.index")
    build_index(iter_centered_vectors(loader, frame), cfg,
                train_sample=N_CHUNKS, path=ipath,
                extra_metadata={"vector_frame": "delta-centered"})
    index, _meta = load_index(ipath)
    return SimpleNamespace(
        model=model, disk=disk, loader=loader, frame=frame,
        index=index, system=load_system_state(disk), dims=loader.dims,
        topic_of=lambda cid: cid % N_TOPICS)


def _qvec(corpus):
    """The §4 query vector (absolute) for the topic-2 question, through
    the real query prefill (reseed + forward + query_cache_vector)."""
    cache = _make_cache()
    reseed_cache(cache, corpus.system)
    corpus.model.active_topic = QUERY_TOPIC
    try:
        with torch.no_grad():
            corpus.model(input_ids=QUERY_IDS, past_key_values=cache,
                         use_cache=True)
    finally:
        corpus.model.active_topic = None
    return query_cache_vector(cache, corpus.system)


def _score_all(corpus, qvec, frame):
    """Every chunk's cosine in the given frame (the rerank over the full
    corpus, one candidate list, k = the corpus size)."""
    ids, scores = rerank(corpus.loader, qvec,
                         np.array(corpus.loader.chunk_ids()),
                         k=N_CHUNKS, frame=frame)
    return {int(i): float(s) for i, s in zip(ids, scores)}


def _disc_gap(scores) -> float:
    """The discrimination gap: best topic-t score − best non-t score."""
    mates = [s for c, s in scores.items() if c % N_TOPICS == QUERY_TOPIC]
    others = [s for c, s in scores.items() if c % N_TOPICS != QUERY_TOPIC]
    return max(mates) - max(others)


# ================================= 1. the floor (the box's signature) =======
def test_absolute_frame_floor(corpus):
    """The legacy metric in the failure regime: the top-3 is a ~1.0 floor
    with a < 1e-5 spread — the score carries no information (the box's
    [0.7232, 0.7072, 0.7071] signature at stub scale: a common-component
    floor with a spread the topic signal cannot clear)."""
    qvec = _qvec(corpus)
    ids, scores = rerank(corpus.loader, qvec,
                         np.array(corpus.loader.chunk_ids()), k=3)
    assert float(scores[0]) > 0.99, "the common floor is not reproduced"
    spread = float(scores.max() - scores.min())
    assert spread < 1e-5, (
        f"the absolute top-3 spread {spread:.2e} is not the collapsed "
        f"floor — recalibrate COMMON")


# ================================= 2. the margin (the mechanism) ============
def test_centered_frame_margin(corpus):
    """The fix's mechanism: the discrimination gap (best topic-t − best
    non-t) is > 3x larger in the centered frame and decisively positive
    (measured: absolute ~1e-3 [fragile], centered 0.047 — a 40x margin
    amplification)."""
    qvec = _qvec(corpus)
    abs_scores = _score_all(corpus, qvec, None)
    ctr_scores = _score_all(corpus, qvec, corpus.frame)
    gap_abs = _disc_gap(abs_scores)
    gap_ctr = _disc_gap(ctr_scores)
    assert gap_ctr > 3.0 * max(gap_abs, 1e-12), (
        f"centered gap {gap_ctr:.4f} does not dominate the absolute "
        f"{gap_abs:.4f}")
    assert gap_ctr > 0.03, (
        f"centered gap {gap_ctr:.4f} is not decisive")


# ================================= 3. the retrieval =========================
def test_centered_frame_retrieves(corpus):
    """The centered frame sees the content: the top-3 is topic-pure, the
    top-1 is the query's topic, and the score spread is real (> 100x the
    absolute frame's)."""
    qvec = _qvec(corpus)
    ids, scores = rerank(corpus.loader, qvec,
                         np.array(corpus.loader.chunk_ids()), k=3,
                         frame=corpus.frame)
    assert all(int(i) % N_TOPICS == QUERY_TOPIC for i in ids), (
        f"the centered top-3 {list(ids)} is not topic-pure")
    ids_abs, scores_abs = rerank(corpus.loader, qvec,
                                 np.array(corpus.loader.chunk_ids()), k=3)
    spread_ctr = float(scores.max() - scores.min())
    spread_abs = float(scores_abs.max() - scores_abs.min())
    assert spread_ctr > 100.0 * max(spread_abs, 1e-30)


# ================================= 4. frame persistence =====================
def test_frame_roundtrip_and_guards(corpus, tmp_path):
    f = corpus.frame
    # bit-exact roundtrip (the side file answer_query auto-detects)
    save_retrieval_frame(corpus.loader, f, path=str(tmp_path / "f.npz"))
    g = load_retrieval_frame(str(tmp_path), path=str(tmp_path / "f.npz"))
    assert (g.sys_vector == f.sys_vector).all()
    assert (g.mean_vector == f.mean_vector).all()
    assert g.dims == f.dims and g.system_ref == f.system_ref
    assert g.n_mean_chunks == f.n_mean_chunks
    g.check_loader(corpus.loader)      # the good frame accepts the corpus

    # dims drift refuses
    bad = RetrievalFrame(sys_vector=f.sys_vector[:-1].copy(),
                         mean_vector=f.mean_vector[:-1].copy(),
                         dims=f.dims - 1, system_ref=f.system_ref)
    with pytest.raises(ValueError, match="dims"):
        bad.check_loader(corpus.loader)
    # system_ref drift refuses (a re-ingested corpus, foreign zero point)
    bad = RetrievalFrame(sys_vector=f.sys_vector, mean_vector=f.mean_vector,
                         dims=f.dims, system_ref="some-other-reset-point")
    with pytest.raises(ValueError, match="system_ref"):
        bad.check_loader(corpus.loader)
    # non-finite refuses at construction
    nan_vec = f.mean_vector.copy()
    nan_vec[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        RetrievalFrame(sys_vector=f.sys_vector, mean_vector=nan_vec,
                       dims=f.dims)


# ================================= 5. e2e centered retrieval ================
def test_answer_query_e2e_centered_frame(corpus):
    """answer_query picks the frame up from retrieval_frame.npz and
    retrieves the topic through the CENTERED index (preselect + rerank)."""
    res = answer_query(corpus.model, QUERY_IDS, corpus.system,
                       index=corpus.index, loader=corpus.loader,
                       cache_factory=_make_cache, max_new_tokens=0)
    assert res.retrieved_ids and len(res.retrieved_ids) == 3
    assert int(res.retrieved_ids[0]) % N_TOPICS == QUERY_TOPIC
    purity = sum(1 for i in res.retrieved_ids
                 if int(i) % N_TOPICS == QUERY_TOPIC)
    assert purity >= 2


# ================================= 6. full-attn fresh =======================
def test_answer_query_full_attention_fresh(tmp_path):
    """The answer-flow fix: after answer_query the full-attn layers hold
    EXACTLY the query tokens once (the pre-W16 flow prefilled the query
    twice — step 2's vector capture + step 7's answer — and the KV
    appended both, so every decoded token attended the question twice).
    The stub writes full-attn KV through the cache layer API at every
    forward; the count must be n_query, not 2 x n_query."""
    import torch.nn.functional as F

    class FullAttnWritingStub(CommonComponentStub):
        def __call__(self, input_ids, past_key_values, use_cache=True):
            ids = tuple(int(t) for t in input_ids.flatten().tolist())
            # the full-attn layers: this call's tokens as KV (B, H, T, D)
            for i, layer in enumerate(past_key_values.layers):
                if type(layer).__name__ == "DynamicLayer":
                    k = F.relu(torch.randn(1, 2, len(ids), 4))
                    v = F.relu(torch.randn(1, 2, len(ids), 4))
                    layer.update(k, v, layer_idx=i, cache_kwargs={})
            return super().__call__(input_ids, past_key_values, use_cache)

    model = FullAttnWritingStub()
    chunks = []
    for i in range(4):
        tok = (300 + 11 * i, 301 + 11 * i, 302 + 11 * i)
        model.chunk_jitter[tok] = 1.0
        chunks.append(torch.tensor([list(tok)]).cuda())
    disk = str(tmp_path)
    IngestDriver(model, SYSTEM_IDS, chunks, disk,
                 cache_factory=_make_cache).run()
    loader = ChunkVectorLoader(disk)
    system = load_system_state(disk)

    n_query = int(QUERY_IDS.shape[-1])
    cache = _make_cache()          # held: answer_query uses THIS cache
    answer_query(model, QUERY_IDS, system, loader=loader, cache=cache,
                 cache_factory=None, max_new_tokens=0,
                 retrieved_ids=[0])               # oracle: skip retrieval
    full_layers = [l for l in cache.layers
                   if type(l).__name__ == "DynamicLayer"]
    assert full_layers, "the stub stack has no full-attn layers to check"
    for layer in full_layers:
        seq = int(layer.get_seq_length())
        assert seq == n_query, (
            f"full-attn KV holds {seq} tokens, expected {n_query} — the "
            f"query was prefilled TWICE (the pre-W16 answer-flow bug)")
    # the M1/M2 write positions restarted at zero for the answer prefill
    for l in cache.layers:
        if type(l).__name__ != "DynamicLayer":
            assert l._m1m2_tokens == 0
