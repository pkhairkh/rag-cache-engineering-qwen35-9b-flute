"""test_query.py — W7.3: acceptance gates for src/rag/install.py (W7.1, the
§6 install math) on top of src/rag/query.py (W7.2, the §6 eight-step query
flow) — the W7 definition of done.

Gates (all deterministic; plain asserts; dirs under pytest tmp):
  1.  INSTALL MATH, BIT-EXACT: sum_turboquant_codes(sys, [d1..dn]) is
      BIT-identical (idx streams + norm + frame metadata) to
      q.quant(q.dequant(sys) + Σ q.dequant(d_i)) — the D4 "ONE requant"
      contract; per-addend (two-round) requantization provably differs;
      the empty-deltas edge is exactly one requant of the system.
  2.  NO CODE-PLUS-CODE OP: neither the turboquant module API nor TQCodes
      offers an add/sum of raw codes — Lloyd-Max is not additive
      (Q(a+b) != Q(a)+Q(b), SPECIFICATION §6/§3); the module docstring
      cites it (grep-level assertion).
  3.  CONV LAST-CHUNK RULE: install over [A, B, C] leaves the cache's conv
      codes BIT-identical to C's (identity, never a sum); a layer missing
      from C falls back to B's (spec §6 "use the last retrieved chunk's").
  4.  FRAME GUARDS: a delta (or system) quantized under a foreign D3 seed,
      a d-mismatched delta, and install_from_disk at the wrong bits all
      raise loudly (ValueError) — never silently corrupt the sum.
  5.  M1/M2 (+S) INSTALL BUDGET: install_snapshot over 3 corpus snapshots
      reconstructs dequant(sys) + Σ dequant(deltas) within the house
      SINGLE-quant-round budget rel-MSE < 0.06 for S, M1 AND M2 (measured
      0.003/0.025/0.022/0.021); install_from_disk over the same ids is
      bit-identical to install_snapshot from the loaded snapshots.
  6.  MINI END-TO-END RAG (THE W7 DoD) — 260-chunk topic corpus, ingested
      once + indexed once (module fixture, ~3.5 s):
        (i)   a topic-t query retrieves >= 1 chunk = t (mod 5) in the
              top-3 (measured 15/15 across all 5 topics, top-1 always t);
        (ii)  timings carries the §9 keys (prefill_snapshot,
              preselect_rerank, load_codes, install, answer_prefill, and
              decode iff max_new_tokens > 0); oracle mode carries exactly
              the 4 non-retrieval keys;
        (iii) oracle mode (retrieved_ids=[...]) runs, reports oracle=True
              and skips retrieval (no preselect_rerank, cos_scores empty);
        (iv)  THE ANSWER READS THE INSTALLED CACHE: with a topic head, the
              answer logits' argmax = t when a topic-t chunk is installed
              (oracle) vs a DIFFERENT argmax (= the wrong chunk's topic)
              when a wrong-topic chunk is installed — the §12
              "correct > wrong" pattern at mechanics level; the greedy
              first decoded token IS that argmax.
  7.  QUERY VECTOR (§4): query_cache_vector length = Σ S dims + M1 + M2
      (conv excluded), fp32, S-ascending/M1/M2 order with every segment
      BIT-equal to the cache's own dequant; the topic-t query vector has
      cos ~ 0.996 with a topic-t chunk vector and ~ 0 with a topic-w one
      (§4 "same space").
  8.  LOUD REFUSALS: answer_query with neither cache nor cache_factory,
      with a loader that has no .disk_dir, and with a retrieved id whose
      npz is not on disk, all raise ValueError with the culprit named.

THE STUB (TopicStubModel, calibrated after the ORCH's smoke_query.py):
  * Topic-marker S update (STRONG marker 6.0*topic_dir + LOW noise 0.05).
    On the real model this query/chunk separation is the §7 fine-tune's
    job; here it is baked in so the retrieval + answer gates test the
    CACHE MECHANICS, not the model.  The chunk's topic is keyed by its
    TOKENS (chunk i carries topic i%5 — the "document content"); the
    query's topic comes from model.active_topic (the "question").
    NOTE (W7.3 finding, see the worklog): the ORCH smoke script sets
    model.active_topic per chunk BEFORE IngestDriver.run() resets it to
    None — the ingested corpus there carries NO topic markers (its
    retrieval assert passed on a ~49% coin flip; retrieved scores ~0.19
    vs the 0.999 measured here).  This suite bakes the topics into the
    model's token->topic map instead, so the corpus is genuinely
    topic-structured and the retrieval gate is deterministic.
  * LOGITS: a matched-filter topic head (head = [dir_0..dir_4]^T, so
    logits[i] = S0 . dir_i) applied to the layer-0 S state the model
    READS at the START of the forward — at answer-prefill time that is
    exactly the INSTALLED codes (spec §6 step 6 -> step 7), so the answer
    provably reads the installed cache and the correct-vs-wrong argmax
    gate is deterministic at a ~5x logit margin (585 vs 115 measured).
  * DETERMINISM: every random tensor comes from a torch.Generator pinned
    by zlib.crc32 of the int64 token bytes (Python's hash() is
    process-salted — NEVER used here, the W5.3 lesson).
  * Corpus: 260 chunks (faiss IVFPQ nbits=8 needs >= 256 train vectors),
    topics i % 5, unique per-chunk token triples; LAYER_TYPES mirrors the
    ORCH smoke ([linear, full, linear, full] -> LINEARS {0, 2}); S/conv/M
    units all d=128 (FHT power-of-two contract, committed d128 codebooks
    — this run creates NO new npz).
"""
from __future__ import annotations

import os
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import turboquant
from turboquant import TurboQuant
from tq_cache import TQCache, resolve_quantizer
from ingest import IngestDriver, load_system_state, reseed_cache
from snapshot import ChunkSnapshot, load_chunk
from install import install_from_disk, install_snapshot, sum_turboquant_codes
from query import answer_query, query_cache_vector
from index import ChunkVectorLoader, IndexConfig, build_index, load_index

# ------------------------------------------------------------- geometry ---
LIN, FULL = "linear_attention", "full_attention"
LAYER_TYPES = [LIN, FULL, LIN, FULL]          # the ORCH smoke's stack
LINEARS = [0, 2]                              # non-contiguous: reseed loops
S_SHAPE, S_D = (1, 8, 16), 128
CONV_D, KERNEL = 32, 4
M_SHAPE, M_D = (2, 4, 16), 128
BITS = 3.5

N_TOPICS = 5
N_CHUNKS = 260                                # faiss PQ nbits=8: nx >= 256
MARKER, NOISE, M_NOISE = 6.0, 0.05, 0.1

# the house single-quant-round budget (test_hooks.py / test_ingest.py)
SINGLE_ROUND_GATE = 0.06

# corpus fixtures: chunk ids used by the install gates (all topic 0 — the
# math gates are topic-agnostic) and by the oracle answer gate
SNAP_IDS = (10, 15, 20)
CORRECT_CHUNK, CORRECT_TOPIC = 7, 2           # 7 % 5 == 2
WRONG_CHUNK, WRONG_TOPIC = 9, 4               # 9 % 5 == 4

QUERY_IDS = torch.tensor([[200, 201]])        # distinct from every corpus key
SYSTEM_IDS = torch.tensor([[5, 6, 7, 8]])


def _rel_mse(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-30).item()


def _make_cache() -> TQCache:
    return TQCache(layer_types=LAYER_TYPES, bits=BITS)


def _codes_equal(a, b) -> bool:
    """Bit-exact TQCodes comparison (idx streams + norm — house style)."""
    return (np.array_equal(a.idx_lo, b.idx_lo)
            and np.array_equal(a.idx_hi, b.idx_hi)
            and float(a.norm) == float(b.norm))


# ------------------------------------------------------------- the stub ---
class TopicStubModel:
    """Model-shaped topic-marker stub (see the module docstring).

    Cache traffic is exactly GatedDeltaNet's: READ
    layers[L].recurrent_states[0] / read_m1/read_m2 (dequantized, shaped),
    WRITE update_recurrent_state / update_conv_state / update_m1 /
    update_m2 (quantize-on-write).  logits (1, 1, n_topics) = the topic
    head applied to the layer-0 S state READ at the start of the forward
    — the installed codes at answer time, so the answer provably reads
    the installed cache.
    """

    def __init__(self, n_topics: int = N_TOPICS):
        g = torch.Generator().manual_seed(99)
        self.topic_dirs = [torch.randn(S_SHAPE, generator=g)
                           for _ in range(n_topics)]
        # matched-filter head: logits[i] = S0 . dir_i
        self.head = torch.stack(
            [d.reshape(-1) for d in self.topic_dirs]).t().contiguous()
        self.chunk_topics: dict = {}   # tuple(token ids) -> the chunk's topic
        self.active_topic = None       # the QUERY's topic (None outside)

    def _topic(self, ids: tuple):
        return self.chunk_topics.get(ids, self.active_topic)

    def __call__(self, input_ids, past_key_values, use_cache=True):
        ids = tuple(int(t) for t in input_ids.flatten().tolist())
        kh = zlib.crc32(np.asarray(ids, dtype=np.int64).tobytes())
        # the S state this forward READS (the installed codes at answer
        # time; the system codes at query-prefill time)
        cur0 = past_key_values.layers[0].recurrent_states[0]
        read_s0 = torch.zeros(S_D) if cur0 is None else cur0.reshape(-1).float()
        topic = self._topic(ids)
        for L in LINEARS:
            cur = past_key_values.layers[L].recurrent_states[0]
            if cur is None:
                cur = torch.zeros(S_SHAPE, dtype=torch.float16)
            g = torch.Generator().manual_seed(31 * (L + 1) + kh)
            add = NOISE * torch.randn(S_SHAPE, generator=g)
            if L == 0 and topic is not None:
                add = add + MARKER * self.topic_dirs[topic]
            new = (cur.float() + add.float()).half()
            past_key_values.update_recurrent_state(new, L)
            conv_in = torch.randn(1, CONV_D, max(1, len(ids)), generator=g)
            past_key_values.update_conv_state(
                conv_in.half(), L, conv_kernel_size=KERNEL)
        m1 = past_key_values.read_m1()
        if m1 is None:
            m1 = torch.zeros(*M_SHAPE, dtype=torch.float16)
        g = torch.Generator().manual_seed(7001 + kh)
        past_key_values.update_m1(
            (m1.float() + M_NOISE * torch.randn(*M_SHAPE, generator=g)).half())
        m2 = past_key_values.read_m2()
        if m2 is None:
            m2 = torch.zeros(*M_SHAPE, dtype=torch.float16)
        g = torch.Generator().manual_seed(7002 + kh)
        past_key_values.update_m2(
            (m2.float() + M_NOISE * torch.randn(*M_SHAPE, generator=g)).half())
        logits = (read_s0 @ self.head).reshape(1, 1, -1)
        return logits


# ------------------------------------------------------- module fixture ---
@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The 260-chunk topic corpus: ingest ONCE, build+load the index ONCE
    (~3.5 s measured); every e2e gate reuses it.  Returns the model, the
    loaded SystemState, the ChunkVectorLoader, the loaded faiss index.
    """
    disk = str(tmp_path_factory.mktemp("q_corpus"))
    model = TopicStubModel()
    chunks = []
    for i in range(N_CHUNKS):
        tok = (100 + 7 * i, 101 + 7 * i, 102 + 7 * i)
        model.chunk_topics[tok] = i % N_TOPICS   # bake the topic in
        chunks.append(torch.tensor([list(tok)]))
    assert model.active_topic is None            # clean state for ingest
    drv = IngestDriver(model, SYSTEM_IDS, chunks, disk,
                       cache_factory=_make_cache)
    stats = drv.run()
    assert stats["ingested"] == N_CHUNKS and stats["skipped"] == 0

    loader = ChunkVectorLoader(disk)
    cfg = IndexConfig(d=loader.dims, nlist=8, m=8, nbits=8, nprobe=8)
    ipath = os.path.join(disk, "ivfadc_cache.index")
    build_index(loader.iter_vectors(), cfg, train_sample=N_CHUNKS, path=ipath)
    index, _meta = load_index(ipath)
    assert index.ntotal == N_CHUNKS

    return SimpleNamespace(
        model=model, disk=disk, loader=loader, index=index,
        system=load_system_state(disk), dims=loader.dims,
        topic_of=lambda cid: cid % N_TOPICS)


def _snap(corpus, cid: int) -> ChunkSnapshot:
    return load_chunk(os.path.join(
        corpus.disk, "snapshots", f"chunk_{cid:05d}.npz"))


# ================================= 1. install math, bit-exact ================
def test_install_one_requant_bit_exact():
    """sum_turboquant_codes(sys, [d1, d2, d3]) is BIT-identical to the
    direct one-requant formula — the D4 contract; per-addend requantizing
    (two rounds) provably differs; empty deltas = one requant of sys."""
    q = resolve_quantizer("S", S_D, BITS)
    g = torch.Generator().manual_seed(11)
    sys_codes = q.quant(3.0 * torch.randn(S_D, generator=g))
    d_codes = [q.quant(0.5 * torch.randn(S_D, generator=g))
               for _ in range(3)]

    summed = sum_turboquant_codes(sys_codes, d_codes, kind="S", bits=BITS)
    direct = q.quant(q.dequant(sys_codes) + q.dequant(d_codes[0])
                     + q.dequant(d_codes[1]) + q.dequant(d_codes[2]))

    assert _codes_equal(summed, direct)                       # bit-exact
    assert (summed.d, summed.seed, summed.bits_lo, summed.bits_hi,
            summed.n_lo, summed.n_hi) == (direct.d, direct.seed,
                                          direct.bits_lo, direct.bits_hi,
                                          direct.n_lo, direct.n_hi)
    assert summed.seed == q.seed                              # the S D3 seed

    # per-addend (two-round) requantization is a DIFFERENT code — the one
    # requant is load-bearing, not an implementation detail
    two_round = q.quant(q.dequant(
        q.quant(q.dequant(sys_codes) + q.dequant(d_codes[0])))
        + q.dequant(d_codes[1]))
    assert not np.array_equal(two_round.idx_lo, summed.idx_lo)
    assert not _codes_equal(two_round, summed)

    # edge: no deltas -> exactly ONE requant round of the system codes
    empty = sum_turboquant_codes(sys_codes, [], kind="S", bits=BITS)
    assert _codes_equal(empty, q.quant(q.dequant(sys_codes)))


# ================================= 2. no code-plus-code op ===================
def test_no_code_plus_code_api():
    """The turboquant public API offers NO add/sum of raw codes —
    Lloyd-Max is not additive (Q(a+b) != Q(a)+Q(b), spec §3/§6), which is
    WHY install.py dequant-sum-requants; the module docstring cites it."""
    for name in dir(TurboQuant):
        assert not any(w in name.lower() for w in ("add", "sum", "plus")), name
    for name in dir(turboquant.TQCodes):
        assert not any(w in name.lower() for w in ("add", "sum", "plus")), name
    for name in turboquant.__all__:
        assert not any(w in name.lower() for w in ("add", "sum", "plus")), name
    for forbidden in ("add_codes", "sum_codes", "add", "sum",
                      "plus", "iadd", "combine", "accumulate"):
        assert not hasattr(turboquant, forbidden)
        assert not hasattr(TurboQuant, forbidden)
        assert not hasattr(turboquant.TQCodes, forbidden)
    # the docstring cites the reason (grep-level, W7.3 wording)
    doc = turboquant.__doc__ or ""
    assert "Q(a+b) ≠ Q(a)+Q(b)" in doc
    assert "no code-plus-code operation" in doc


# ================================= 3. conv last-chunk rule ===================
def test_conv_last_chunk_rule(corpus):
    """install over [A, B, C]: conv codes ARE C's (bit identity, never a
    sum, and NOT A's); a layer missing from C falls back to B's; the
    report records 'sum+last-conv' per layer."""
    snaps = [_snap(corpus, cid) for cid in SNAP_IDS]
    cache = _make_cache()
    report = install_snapshot(cache, corpus.system, snaps)

    for L in LINEARS:
        conv = cache.layers[L].conv_codes
        assert conv is not None
        # the LAST chunk's codes, verbatim (identity — a sum could not be)
        assert _codes_equal(conv, snaps[-1].conv_codes[L])
        # not the FIRST chunk's (last-chunk-wins, not first)
        assert not _codes_equal(conv, snaps[0].conv_codes[L])
        assert report[L]["mode"] == "sum+last-conv"
        assert report[L]["n_deltas"] == len(SNAP_IDS)
    # distinct conv codes across chunks -> the identity assert is
    # non-vacuous
    assert not _codes_equal(snaps[0].conv_codes[0], snaps[1].conv_codes[0])

    # a layer missing from the LAST snapshot falls back to the previous
    # one (spec §6 "the last retrieved chunk's" per layer)
    last = snaps[-1]
    trimmed = ChunkSnapshot(
        chunk_id=last.chunk_id, protocol=last.protocol,
        s_codes=last.s_codes, conv_codes={0: last.conv_codes[0]},
        m1_codes=last.m1_codes, m2_codes=last.m2_codes,
        system_ref=last.system_ref, extra=last.extra)
    cache2 = _make_cache()
    report2 = install_snapshot(cache2, corpus.system,
                               snaps[:-1] + [trimmed])
    assert _codes_equal(cache2.layers[0].conv_codes, trimmed.conv_codes[0])
    assert _codes_equal(cache2.layers[2].conv_codes, snaps[1].conv_codes[2])
    assert report2[2]["mode"] == "sum+last-conv"


# ================================= 4. frame guards ===========================
def test_install_frame_guards(corpus):
    """A delta quantized under a foreign D3 seed raises; so do a
    d-mismatched delta, a foreign-seed SYSTEM, and install_from_disk at
    the wrong bits — all loudly (ValueError, culprit named)."""
    q = resolve_quantizer("S", S_D, BITS)
    g = torch.Generator().manual_seed(5)
    sys_codes = q.quant(torch.randn(S_D, generator=g))
    rogue = TurboQuant(kind="custom", bits=BITS, d=S_D, seed=999)
    rogue_delta = rogue.quant(torch.randn(S_D, generator=g))
    ok_delta = q.quant(torch.randn(S_D, generator=g))

    # (a) delta frame drift — the mission's gate
    with pytest.raises(ValueError, match="frame drift"):
        sum_turboquant_codes(sys_codes, [rogue_delta, ok_delta], kind="S")
    # (b) unit mismatch: delta quantized at d=256 vs system d=128
    big = resolve_quantizer("S", 256, BITS)
    with pytest.raises(ValueError, match="unit mismatch"):
        sum_turboquant_codes(
            sys_codes, [big.quant(torch.randn(256, generator=g))], kind="S")
    # (c) the SYSTEM codes themselves in a foreign frame
    with pytest.raises(ValueError, match="frame drift"):
        sum_turboquant_codes(rogue_delta, [ok_delta], kind="S")

    # (d) install_from_disk refuses the wrong bits for the reset point
    cache = _make_cache()
    with pytest.raises(ValueError, match="bits"):
        install_from_disk(cache, corpus.disk, [SNAP_IDS[0]], bits=2.0)


# ================================= 5. M1/M2 (+ S) install budget =============
def test_install_m1m2_sum_budget(corpus):
    """install_snapshot over 3 corpus snapshots: S (both layers), M1 AND
    M2 all reconstruct dequant(sys) + Σ dequant(deltas) within ONE
    quant-round budget rel-MSE < 0.06 (measured 0.003-0.025); the disk
    convenience path is bit-identical to the in-memory path."""
    snaps = [_snap(corpus, cid) for cid in SNAP_IDS]
    cache = _make_cache()
    report = install_snapshot(cache, corpus.system, snaps)

    for L in LINEARS:
        q = resolve_quantizer("S", S_D, BITS)
        expect = q.dequant(corpus.system.s_codes[L]) \
            + sum(q.dequant(s.s_codes[L]) for s in snaps)
        got = q.dequant(cache.layers[L].s_codes)
        assert _rel_mse(got, expect) < SINGLE_ROUND_GATE
    for kind, attr in (("M1", "m1_codes"), ("M2", "m2_codes")):
        q = resolve_quantizer(kind, M_D, BITS)
        sys_c = getattr(corpus.system, attr)
        expect = q.dequant(sys_c) \
            + sum(q.dequant(getattr(s, attr)) for s in snaps)
        got = q.dequant(getattr(cache, attr))
        assert _rel_mse(got, expect) < SINGLE_ROUND_GATE
        assert report[kind] == {"mode": "sum", "n_deltas": len(SNAP_IDS)}
        assert getattr(cache, attr) is not None

    # the disk path over the same ids: bit-identical codes per unit
    cache2 = _make_cache()
    install_from_disk(cache2, corpus.disk, list(SNAP_IDS))
    for L in LINEARS:
        assert _codes_equal(cache.layers[L].s_codes, cache2.layers[L].s_codes)
        assert _codes_equal(cache.layers[L].conv_codes,
                            cache2.layers[L].conv_codes)
    assert _codes_equal(cache.m1_codes, cache2.m1_codes)
    assert _codes_equal(cache.m2_codes, cache2.m2_codes)


# ================================= 7. query vector (§4) ======================
def test_query_cache_vector_layout(corpus):
    """The §4 query vector: length = Σ S dims + M1 + M2 (conv EXCLUDED),
    fp32, S-ascending / M1 / M2 order, every segment BIT-equal to the
    cache's own dequant; the topic-3 query vector aligns with a topic-3
    chunk vector (cos ~ 0.996) and not a topic-0 one (§4 same-space)."""
    cache = _make_cache()
    reseed_cache(cache, corpus.system)
    corpus.model.active_topic = 3
    try:
        with torch.no_grad():
            corpus.model(QUERY_IDS, past_key_values=cache, use_cache=True)
    finally:
        corpus.model.active_topic = None

    v = query_cache_vector(cache, corpus.system)
    assert v.shape == (corpus.dims,)                 # 2*128 + 128 + 128
    assert v.dtype == np.float32
    assert corpus.dims == 2 * S_D + M_D + M_D        # conv excluded

    q_s = resolve_quantizer("S", S_D, BITS)
    q_m1 = resolve_quantizer("M1", M_D, BITS)
    q_m2 = resolve_quantizer("M2", M_D, BITS)
    assert np.array_equal(v[0:S_D],
                          q_s.dequant(cache.layers[0].s_codes).numpy())
    assert np.array_equal(v[S_D:2 * S_D],
                          q_s.dequant(cache.layers[2].s_codes).numpy())
    assert np.array_equal(v[2 * S_D:2 * S_D + M_D],
                          q_m1.dequant(cache.m1_codes).numpy())
    assert np.array_equal(v[2 * S_D + M_D:],
                          q_m2.dequant(cache.m2_codes).numpy())
    # non-vacuous: the prefill moved the vector off the system reset point
    assert not np.allclose(
        v[:S_D], q_s.dequant(corpus.system.s_codes[0]).numpy())

    # §4 same-space check: the topic-3 query vs topic-3 / topic-0 chunks
    vq = v / np.linalg.norm(v)
    c3 = corpus.loader.vector(13)                    # 13 % 5 == 3
    c0 = corpus.loader.vector(10)                    # 10 % 5 == 0
    cos3 = float(vq @ (c3 / np.linalg.norm(c3)))
    cos0 = float(vq @ (c0 / np.linalg.norm(c0)))
    assert cos3 > 0.95 and cos0 < 0.2 and cos3 - cos0 > 0.7


# ================================= 6a. mini e2e: flow + §9 ===================
def test_e2e_query_flow(corpus):
    """THE W7 DoD, one full flow: topic-2 query -> retrieve top-3 (all
    topic-2, top-1 topic-2) -> install -> answer.  §9 timings carry all
    SIX step keys (decode included at max_new_tokens=2); the greedy first
    decoded token is the answer argmax = 2 (the answer reads the
    installed cache); oracle flag off; install report well-formed."""
    corpus.model.active_topic = 2
    try:
        res = answer_query(corpus.model, QUERY_IDS, corpus.system,
                           index=corpus.index, loader=corpus.loader,
                           cache_factory=_make_cache, max_new_tokens=2)
    finally:
        corpus.model.active_topic = None

    # (i) retrieval: rerank_k=3 topic-2 chunks (measured 3/3, top-1 topic-2)
    assert len(res.retrieved_ids) == 3
    assert len(res.cos_scores) == 3
    assert all(res.cos_scores[i] >= res.cos_scores[i + 1]
               for i in range(2))                    # score-descending
    assert res.retrieved_ids[0] % N_TOPICS == 2
    assert sum(1 for i in res.retrieved_ids if i % N_TOPICS == 2) >= 2
    assert res.cos_scores[0] > 0.95
    assert not res.oracle

    # (ii) the §9 ledger: every step key present and positive
    assert set(res.timings) == {"prefill_snapshot", "preselect_rerank",
                                "load_codes", "install", "answer_prefill",
                                "decode"}
    assert all(isinstance(t, float) and t > 0.0
               for t in res.timings.values())

    # install report: both S layers + both memories, 3 deltas each
    assert set(map(str, res.install_report)) == {"0", "2", "M1", "M2"}
    for L in LINEARS:
        assert res.install_report[L]["mode"] == "sum+last-conv"
        assert res.install_report[L]["n_deltas"] == 3

    # (iv-a) the greedy decode: 2 tokens, first = the answer argmax = 2
    # (three topic-2 deltas installed -> the matched filter fires on t=2)
    assert len(res.new_token_ids) == 2
    assert all(0 <= t < N_TOPICS for t in res.new_token_ids)
    assert res.new_token_ids[0] == 2


# ================================= 6b. mini e2e: all topics ==================
def test_e2e_retrieval_all_topics(corpus):
    """Hit-rate gate over every topic: each of the 5 topic-t queries
    retrieves >= 1 topic-t chunk in the top-3 (measured 15/15 with top-1
    always topic-t); no-decode flow carries exactly the 5 non-decode §9
    keys."""
    hits_total = 0
    for t in range(N_TOPICS):
        corpus.model.active_topic = t
        try:
            res = answer_query(corpus.model, QUERY_IDS, corpus.system,
                               index=corpus.index, loader=corpus.loader,
                               cache_factory=_make_cache)  # no decode
        finally:
            corpus.model.active_topic = None
        hits = sum(1 for i in res.retrieved_ids if i % N_TOPICS == t)
        assert hits >= 1, (t, res.retrieved_ids, res.cos_scores)
        assert res.retrieved_ids[0] % N_TOPICS == t
        assert set(res.timings) == {"prefill_snapshot", "preselect_rerank",
                                    "load_codes", "install", "answer_prefill"}
        assert res.new_token_ids == []
        hits_total += hits
    assert hits_total >= 10                          # measured 15/15


# ================================= 6c. oracle mode ===========================
def test_oracle_mode(corpus):
    """retrieved_ids=[...] skips retrieval (steps 3-4): oracle=True, ids
    preserved verbatim, no rerank scores, exactly the 4 non-retrieval §9
    keys, and the install report carries n_deltas=1 for the oracle chunk."""
    corpus.model.active_topic = CORRECT_TOPIC
    try:
        res = answer_query(corpus.model, QUERY_IDS, corpus.system,
                           loader=corpus.loader, cache_factory=_make_cache,
                           retrieved_ids=[CORRECT_CHUNK])
    finally:
        corpus.model.active_topic = None

    assert res.oracle is True
    assert res.retrieved_ids == [CORRECT_CHUNK]
    assert res.cos_scores == []
    assert set(res.timings) == {"prefill_snapshot", "load_codes", "install",
                                "answer_prefill"}
    assert all(isinstance(v, float) and v > 0.0
               for v in res.timings.values())
    assert set(map(str, res.install_report)) == {"0", "2", "M1", "M2"}
    for key in ("M1", "M2"):
        assert res.install_report[key]["n_deltas"] == 1
    assert res.new_token_ids == []                    # max_new_tokens=0


# ================================= 6d. the answer reads the cache ============
def test_answer_reads_installed_cache(corpus):
    """THE §12 'correct > wrong' pattern at mechanics level (oracle mode,
    topic-2 query): installing chunk 7 (topic 2) -> the answer logits'
    argmax = 2 and the greedy first token = 2; installing chunk 9 (topic
    4) instead -> argmax = 4 (DIFFERENT), the correct-topic logit drops
    below the wrong-topic one.  Captured via decode_fn (which receives
    the answer-prefill logits) — the answer prefill reads the INSTALLED
    codes, so the argmax flips with the installed chunk."""
    logits_seen = {}

    def make_capture(label):
        def capture_decode(model, cache, logits, n):
            logits_seen[label] = logits.detach().clone()
            return [int(logits[0, -1, :].argmax())] * n
        return capture_decode

    results = {}
    for cid, label in ((CORRECT_CHUNK, "correct"), (WRONG_CHUNK, "wrong")):
        corpus.model.active_topic = CORRECT_TOPIC
        try:
            results[label] = answer_query(
                corpus.model, QUERY_IDS, corpus.system, loader=corpus.loader,
                cache_factory=_make_cache, retrieved_ids=[cid],
                max_new_tokens=1)                     # default greedy
            corpus.model.active_topic = CORRECT_TOPIC
            results[label + "_cap"] = answer_query(
                corpus.model, QUERY_IDS, corpus.system, loader=corpus.loader,
                cache_factory=_make_cache, retrieved_ids=[cid],
                max_new_tokens=1, decode_fn=make_capture(label))
        finally:
            corpus.model.active_topic = None

    # greedy first token IS the answer argmax (both paths agree)
    assert results["correct"].new_token_ids[0] == CORRECT_TOPIC
    assert results["wrong"].new_token_ids[0] == WRONG_TOPIC
    assert results["wrong"].new_token_ids[0] != CORRECT_TOPIC

    # the captured answer-prefill logits (decode_fn receives them):
    assert results["correct_cap"].new_token_ids == [CORRECT_TOPIC]
    lg_c = logits_seen["correct"][0, -1, :]
    assert int(lg_c.argmax()) == CORRECT_TOPIC
    others_c = [float(v) for j, v in enumerate(lg_c) if j != CORRECT_TOPIC]
    assert float(lg_c[CORRECT_TOPIC]) > 2.0 * max(others_c)

    assert results["wrong_cap"].new_token_ids == [WRONG_TOPIC]
    lg_w = logits_seen["wrong"][0, -1, :]
    assert int(lg_w.argmax()) == WRONG_TOPIC
    assert WRONG_TOPIC != CORRECT_TOPIC
    others_w = [float(v) for j, v in enumerate(lg_w) if j != WRONG_TOPIC]
    assert float(lg_w[WRONG_TOPIC]) > 2.0 * max(others_w)

    # the cross-run §12 pattern: the answer's topic logit is HIGH when the
    # matching chunk is installed, LOW when the wrong one is (measured
    # 585.3 vs -46.3 on the correct dim; 641.8 vs -26.7 on the wrong dim)
    assert float(lg_c[CORRECT_TOPIC]) > float(lg_w[CORRECT_TOPIC])
    assert float(lg_w[WRONG_TOPIC]) > float(lg_c[WRONG_TOPIC])
    assert float(lg_c[CORRECT_TOPIC]) - float(lg_w[CORRECT_TOPIC]) > 100.0


# ================================= 8. loud refusals ==========================
def test_loud_refusals(corpus):
    """answer_query refuses loudly: no cache and no cache_factory; a
    loader without .disk_dir; a retrieved id with no npz on disk."""
    # (a) neither cache nor cache_factory — refused before any model call
    with pytest.raises(ValueError, match="cache_factory"):
        answer_query(corpus.model, QUERY_IDS, corpus.system,
                     index=corpus.index, loader=corpus.loader)

    # (b) loader without .disk_dir (oracle mode: no index needed)
    with pytest.raises(ValueError, match="disk_dir"):
        answer_query(corpus.model, QUERY_IDS, corpus.system,
                     cache=_make_cache(), loader=object(),
                     retrieved_ids=[CORRECT_CHUNK])

    # (c) retrieved id with no snapshot on disk
    with pytest.raises(ValueError, match="chunk_09999.npz not found"):
        answer_query(corpus.model, QUERY_IDS, corpus.system,
                     loader=corpus.loader, cache_factory=_make_cache,
                     retrieved_ids=[9999])
