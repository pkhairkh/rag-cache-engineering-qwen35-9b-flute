"""test_ingest.py — W5.3: acceptance gates for src/rag/ingest.py (W5.2, the
D4 delta-protocol ingestion + the restartable driver) on top of
src/rag/snapshot.py (W5.1, the codes-only npz codec).

Gates (the W5 definition of done):
  1.  FULL-SCALE SIZE GATE (@slow, ~1 s measured): a synthetic
      production-dim ChunkSnapshot — 24 S codes (d=524,288) + 24 conv codes
      (d=32,768) + M1 + M2 (d=524,288) at 3.5 bits — saves to a file in
      [5.9, 6.2] MiB whose CODE bytes are exactly the spec §5 table
      arithmetic (6,308,040 B = 26×229,380 + 24×14,340), loads back
      bit-identically (spot-checked idx streams + norms) and passes
      verify_chunk.  The ~145 KB npz container overhead of the 551 flat
      members is the W5.1-documented format cost; the FILE measures
      6.154 MiB, inside the gate.
  2.  THE W5 DoD — D4 delta reconstruction on the quantized stub: system
      prefill + 3 chunks through the driver (return_records); for every
      (chunk, L): dequant(delta) + dequant(sys codes) reconstructs the
      stub's known final state rel-MSE < 0.10 — the DOUBLE-quant-round
      budget (BOTH operands of the reconstruction are quantized: the
      chunk-state storage round PLUS the delta requant round, which acts
      on a delta whose norm² is ~0.5× the state's; a single round would
      be the house 0.06).  Measured max: 0.034 (S) / 0.038 (M1/M2).
  3.  EXACT-DELTA check: a stub whose update writes KNOWN tensors
      (new = cur + n_tokens × fixed_L, no randomness between the read
      and the write) reconstructs rel-MSE < 0.06 — the SINGLE-round
      budget.  Measured 0.006: the only error term that scales with the
      state is the chunk-storage round on the 5/11-of-norm increment;
      the system segment is a lattice point that quantizes cleanly, so
      the delta requant contributes ~0.001.
  4.  §4 retrieval vector: fp32; length = Σ S dims + M1 dim + M2 dim
      (conv is NOT part of it); ORDER = S layers ascending, then M1,
      then M2 — pinned BIT-EXACTLY against an independent fresh-cache
      reseed + forward rerun, and every S segment equals
      dequant(delta_L) + dequant(sys_L) within the single-round budget
      (measured 0.011).
  5.  M1 zero-gate no-op THROUGH the delta path: a stub whose M1 write
      gate is 0 (m1 stays at its zero init — the W3 no-op semantics)
      leaves the vector's M1 segment EXACTLY zero and the delta M1 codes
      zero-norm, on disk as well.
  6.  Driver resume: 3 chunks, then a 4th, re-run -> {ingested: 1,
      skipped: 3}; manifest done keys {0,1,2,3}; the first 3 files are
      byte-identical (not rewritten — sha256-compared); exactly 4 files;
      verify_chunk ok on all 4; no leftover .tmp manifest.
  7.  Resume drift guard: re-running with a DIFFERENT system prompt
      raises ValueError and leaves the manifest + every file untouched
      (never silently corrupts); the SAME prompt resumes cleanly.
  8.  prefill_system fresh-cache contract: a cache that already holds S
      codes (via prefill or a plain forward) is loudly refused — the
      system point is the delta protocol's zero.
  9.  D5 codes-only disk: no 'vector' npz member, no vector field on the
      loaded snapshot; s_codes on disk ARE the W16 ABSOLUTE end codes
      (the default, zero extra rounds) and the legacy delta-v1 layout
      still round-trips (chunk_protocol="delta-v1"); conv codes are
      ABSOLUTE (bit-equal the cache's post-prefill conv codes of an
      independent rerun, chunk-specific, not the system's).
  10. Manifest invariants: protocol "delta-v1"; bits; system_ref equals
      the SystemState reference; vector_dims == the actual vector length;
      done ids -> existing relative paths whose meta matches the id.

Contract extras: reseed_cache fully neutralizes a used cache (ONE reused
cache yields bit-identical records to fresh-cache-per-chunk); the driver
accepts (idx, tensor) pairs with non-contiguous ids; no cache_factory +
a model without .config is a loud ValueError.

THE STUB (house pattern, cf. test_hooks.py / test_m1m2.py): a model-shaped
callable mimicking GatedDeltaNet's cache traffic — READ
layers[L].recurrent_states[0] / read_m1() (dequantized, shaped), WRITE
update_recurrent_state / update_conv_state / update_m1 / update_m2
(quantize-on-write) — while remembering the true final fp16 states for
the reconstruction gates.  Determinism: every random tensor comes from a
pinned torch.Generator, and the per-input seed is zlib.crc32 of the token
ids — Python's hash() is salted PER PROCESS (PYTHONHASHSEED) and would
make the stub's states non-reproducible across runs.
"""
from __future__ import annotations

import json
import os
import zlib
from hashlib import sha256
from itertools import combinations

import numpy as np
import pytest
import torch

from ingest import (
    MANIFEST_NAME,
    IngestDriver,
    ingest_chunk,
    prefill_system,
    reseed_cache,
)
from snapshot import (
    ChunkSnapshot,
    chunk_nbytes,
    load_chunk,
    save_chunk,
    snapshot_path,
    verify_chunk,
)
from tq_cache import TQCache, resolve_quantizer
from turboquant import get_quantizer

LIN, FULL = "linear_attention", "full_attention"

# ------------------------------------------------------------- geometry ---
# Stub sizes: power-of-two FHT units (PROPOSAL D2) at the committed
# codebook scales (d128): S (1, 8, 16) = 128 dims; the conv input is
# prefill-shaped (1, 32, n_tokens) so the windowing contract is exercised
# (the stored window is its last KERNEL columns); M1/M2 (2, 4, 16) = 128.
# Non-contiguous linear indices (0, 2, 3) exercise the reseed loops.
LAYER_TYPES = [LIN, FULL, LIN, LIN, FULL]
LINEARS = [i for i, lt in enumerate(LAYER_TYPES) if lt == LIN]
assert LINEARS == [0, 2, 3]

S_SHAPE = (1, 8, 16)
S_D = 128
CONV_D = 32
KERNEL = 4
CONV_WINDOW = (1, CONV_D, KERNEL)
M_SHAPE = (2, 4, 16)
M_D = 128
BITS = 3.5

# The house single-quant-round budget (test_hooks.py), and the D4
# double-round budget: the reconstruction dequant(delta) + dequant(sys)
# carries TWO TurboQuant rounds (chunk-state storage + delta requant);
# the delta requant error is relative to the DELTA norm, which for this
# stub is ~0.7x the state norm (~0.5x in MSE) -> expected worst ~1.5x the
# single-round error; 0.10 leaves ~2.6x measured headroom.
SINGLE_ROUND_GATE = 0.06
DOUBLE_ROUND_GATE = 0.10

# Production geometry (spec §1/§5): 24 linear indices among 32 layers.
LIN24 = [i for i in range(32) if i % 4 != 3]
assert len(LIN24) == 24
D_S = 524_288
D_CONV = 32_768
# spec §5 code-byte arithmetic at 3.5 bits (W1.4 gate 6):
#   S/M1/M2 unit: ceil(262144*3/8) + 262144*4/8 + 4 = 229,380 B
#   conv unit:    ceil(16384*3/8)  +  16384*4/8 + 4 =  14,340 B
S_UNIT_BYTES = 229_380
CONV_UNIT_BYTES = 14_340
CODE_BYTES_TOTAL = 26 * S_UNIT_BYTES + 24 * CONV_UNIT_BYTES  # 6,308,040
MiB = 1024 * 1024


def _rel_mse(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-30).item()


def _tokens(seed: int, n: int) -> torch.Tensor:
    return torch.randint(0, 100_000, (1, n),
                         generator=torch.Generator().manual_seed(seed))


def _key_of(ids: torch.Tensor) -> tuple:
    return tuple(int(t) for t in ids.flatten().tolist())


def _key_hash(key: tuple) -> int:
    """DETERMINISTIC per-input hash (Python's hash() is process-salted)."""
    return zlib.crc32(np.asarray(key, dtype=np.int64).tobytes())


def _make_cache() -> TQCache:
    return TQCache(layer_types=LAYER_TYPES, bits=BITS)


# ------------------------------------------------------------- the stub ---
class StubModel:
    """Model-shaped stand-in (the mission's verified pattern):

    * per linear layer L: read recurrent_states[0] (None on a fresh
      cache -> zero state), add a deterministic increment, write it back
      (quantize-on-write), remember the true final state; write a random
      conv input through update_conv_state (the windowing contract).
    * M1/M2: read_m1() (None -> zeros), add a gated increment, write
      back.  m_noise=0.0 is the W3 zero-gate no-op: the memory stays at
      its zero init, update still runs (codes exist, norm 0).
    * exact=True switches the S/M increments to KNOWN fixed tensors
      scaled by the token count (cur + n_tok * fixed) — no randomness
      between the read and the write, so the final state is analytic.
    """

    def __init__(self, layer_types=LAYER_TYPES, s_noise: float = 0.3,
                 m1_noise: float = 0.1, m2_noise: float = 0.1,
                 exact: bool = False):
        self.layer_types = list(layer_types)
        self.linears = [i for i, lt in enumerate(self.layer_types)
                        if lt == LIN]
        self.s_noise = s_noise
        self.m1_noise = m1_noise
        self.m2_noise = m2_noise
        self.exact = exact
        self.true_states = {}          # (key, L) -> final S as written
        self.true_m1, self.true_m2 = {}, {}
        # the exact variant's fixed increments (per layer / per memory)
        self.fixed = {L: torch.randn(S_D, generator=torch.Generator()
                                     .manual_seed(7000 + L))
                      for L in self.linears}
        self.fixed_m1 = torch.randn(*M_SHAPE, generator=torch.Generator()
                                    .manual_seed(8001))
        self.fixed_m2 = torch.randn(*M_SHAPE, generator=torch.Generator()
                                    .manual_seed(8002))

    def __call__(self, input_ids, past_key_values, use_cache=True):
        key = _key_of(input_ids)
        n_tok = int(input_ids.shape[-1])
        kh = _key_hash(key)
        for L in self.linears:
            cur = past_key_values.layers[L].recurrent_states[0]
            if cur is None:
                cur = torch.zeros(S_D, dtype=torch.float16)
            cur = cur.reshape(-1).float()
            if self.exact:
                new = cur + n_tok * self.fixed[L]
            else:
                g = torch.Generator().manual_seed(1_000_003 * L + kh)
                new = cur + self.s_noise * torch.randn(S_D, generator=g)
            past_key_values.update_recurrent_state(
                new.reshape(S_SHAPE).half(), L)
            self.true_states[(key, L)] = new.reshape(S_SHAPE).half().clone()
            g = torch.Generator().manual_seed(1_000_003 * L + 17 * kh)
            conv_in = torch.randn(1, CONV_D, n_tok, generator=g)
            past_key_values.update_conv_state(
                conv_in.half(), L, conv_kernel_size=KERNEL)
        m1 = past_key_values.read_m1()
        if m1 is None:
            m1 = torch.zeros(*M_SHAPE, dtype=torch.float16)
        if self.exact:
            new_m1 = m1 + n_tok * self.fixed_m1
        elif self.m1_noise:
            g = torch.Generator().manual_seed(9001 + kh)
            new_m1 = m1 + self.m1_noise * torch.randn(*M_SHAPE, generator=g)
        else:  # the zero-gate no-op: the memory stays at its current value
            new_m1 = m1
        past_key_values.update_m1(new_m1.half())
        self.true_m1[key] = new_m1.half().clone()
        m2 = past_key_values.read_m2()
        if m2 is None:
            m2 = torch.zeros(*M_SHAPE, dtype=torch.float16)
        if self.exact:
            new_m2 = m2 + n_tok * self.fixed_m2
        else:
            g = torch.Generator().manual_seed(9002 + kh)
            new_m2 = m2 + self.m2_noise * torch.randn(*M_SHAPE, generator=g)
        past_key_values.update_m2(new_m2.half())
        self.true_m2[key] = new_m2.half().clone()
        return None


def _run_driver(model, sys_tok, chunks, out_dir, **kw):
    kw.setdefault("cache_factory", _make_cache)
    kw.setdefault("return_records", True)
    return IngestDriver(model, sys_tok, chunks, str(out_dir), **kw)


# =============================================== 1. full-scale size gate ====
@pytest.mark.slow
def test_full_scale_size_gate(tmp_path):
    """The production-dim chunk: ~6 MiB on disk, bit-exact round-trip.

    No model — a synthetic ChunkSnapshot of full-scale TurboQuant codes
    (the ingest driver's exact output geometry), ~1 s on this box.
    """
    q = {k: get_quantizer(k, BITS) for k in ("S", "conv", "M1", "M2")}
    g = torch.Generator().manual_seed(20260503)

    def unit(d: int):
        x = torch.randn(d, generator=g, dtype=torch.float32)
        return x / x.norm()

    s_codes = {L: q["S"].quant(unit(D_S)) for L in LIN24}
    conv_codes = {L: q["conv"].quant(unit(D_CONV)) for L in LIN24}
    m1 = q["M1"].quant(unit(D_S))
    m2 = q["M2"].quant(unit(D_S))

    # code bytes: EXACTLY the spec §5 table arithmetic
    code_bytes = (sum(c.nbytes() for c in s_codes.values())
                  + sum(c.nbytes() for c in conv_codes.values())
                  + m1.nbytes() + m2.nbytes())
    assert all(c.nbytes() == S_UNIT_BYTES for c in s_codes.values())
    assert all(c.nbytes() == CONV_UNIT_BYTES for c in conv_codes.values())
    assert code_bytes == CODE_BYTES_TOTAL == 6_308_040

    snap = ChunkSnapshot(
        chunk_id=0, protocol="delta-v1", s_codes=s_codes,
        conv_codes=conv_codes, m1_codes=m1, m2_codes=m2,
        system_ref="w5.3-full-scale", extra={"n_tokens": 512})
    path = save_chunk(str(tmp_path), snap)
    assert path == snapshot_path(str(tmp_path), 0)

    size = chunk_nbytes(path)
    # FILE gate [5.9, 6.2] MiB: code bytes 6.016 MiB + the 551-member npz
    # container (~145 KB, the W5.1-documented format cost) = 6.154 MiB
    # measured (50,000 x this = 300.5 GiB, spec §11's own total).
    assert 5.9 * MiB <= size <= 6.2 * MiB, f"file size {size} B"

    with np.load(path) as z:
        assert len(z.files) == 551          # meta + 50 units x 11 fields
        assert "vector" not in z.files      # D5: codes-only by default

    back = load_chunk(path)
    # bit-identical round-trip: 3 S layers, 2 conv layers, M1, M2
    for L in (0, 12, 30):
        assert np.array_equal(back.s_codes[L].idx_lo, s_codes[L].idx_lo)
        assert np.array_equal(back.s_codes[L].idx_hi, s_codes[L].idx_hi)
        assert float(back.s_codes[L].norm) == float(s_codes[L].norm)
        assert back.s_codes[L].seed == s_codes[L].seed == 101  # D3
    for L in (2, 17):
        assert np.array_equal(back.conv_codes[L].idx_lo, conv_codes[L].idx_lo)
        assert np.array_equal(back.conv_codes[L].idx_hi, conv_codes[L].idx_hi)
        assert float(back.conv_codes[L].norm) == float(conv_codes[L].norm)
    assert np.array_equal(back.m1_codes.idx_lo, m1.idx_lo)
    assert np.array_equal(back.m2_codes.idx_hi, m2.idx_hi)
    assert float(back.m1_codes.norm) == float(m1.norm)
    assert back.protocol == "delta-v1" and back.chunk_id == 0
    assert back.system_ref == "w5.3-full-scale"

    v = verify_chunk(path)
    assert v["ok"] and v["sha256_ok"]
    assert v["n_s"] == 24 and v["n_conv"] == 24
    assert v["has_m1"] and v["has_m2"]
    assert v["nbytes"] == size


# ================================= 2. D4 delta reconstruction (the DoD) ====
def test_delta_protocol_reconstruction_quantized(tmp_path):
    """dequant(delta) + dequant(sys) reconstructs the stub's final state.

    DOUBLE-quant-round budget: the reconstruction's two operands are both
    TurboQuant codes (the chunk-state storage round + the delta requant
    round); measured max 0.034 (S) / 0.038 (M1/M2) vs the 0.10 gate.
    """
    model = StubModel()
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101 + i, 5) for i in range(3)]
    drv = _run_driver(model, sys_tok, chunks, tmp_path)
    stats = drv.run()
    assert stats["ingested"] == 3 and stats["skipped"] == 0
    assert stats["done_before"] == 0 and stats["n_chunks"] == 3
    assert [r.chunk_idx for r in drv.records] == [0, 1, 2]

    # independent system rebuild — the prefill is deterministic (crc32
    # keys), so a second prefill pins the SAME reset point
    system = prefill_system(model, sys_tok, _make_cache())
    system_b = prefill_system(model, sys_tok, _make_cache())
    assert system_b.reference() == system.reference()

    q_s = resolve_quantizer("S", S_D)
    q_m1 = resolve_quantizer("M1", M_D)
    q_m2 = resolve_quantizer("M2", M_D)
    for rec, ct in zip(drv.records, chunks):
        key = _key_of(ct)
        for L in LINEARS:
            recon = q_s.dequant(rec.delta_s[L]) + q_s.dequant(
                system.s_codes[L])
            rel = _rel_mse(recon, model.true_states[(key, L)])
            assert rel < DOUBLE_ROUND_GATE, f"chunk {rec.chunk_idx} L {L}: {rel}"
        rel1 = _rel_mse(q_m1.dequant(rec.delta_m1)
                        + q_m1.dequant(system.m1_codes), model.true_m1[key])
        rel2 = _rel_mse(q_m2.dequant(rec.delta_m2)
                        + q_m2.dequant(system.m2_codes), model.true_m2[key])
        assert rel1 < DOUBLE_ROUND_GATE and rel2 < DOUBLE_ROUND_GATE
        assert rec.n_tokens == 5
        assert rec.cache_vector.shape[0] == len(LINEARS) * S_D + 2 * M_D


# ================================= 3. exact-delta check (single round) =====
def test_delta_protocol_exact_stub(tmp_path):
    """A stub whose update writes KNOWN tensors (cur + n_tok*fixed):

    the reconstruction error is dominated by the single chunk-storage
    round (the increment is 5/11 of the state norm -> the delta requant
    contributes ~0.001); measured max 0.006 vs the single-round 0.06.
    """
    model = StubModel(exact=True)
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101 + i, 5) for i in range(3)]
    drv = _run_driver(model, sys_tok, chunks, tmp_path)
    assert drv.run()["ingested"] == 3

    system = prefill_system(model, sys_tok, _make_cache())
    q_s = resolve_quantizer("S", S_D)
    q_m1 = resolve_quantizer("M1", M_D)
    q_m2 = resolve_quantizer("M2", M_D)
    for rec, ct in zip(drv.records, chunks):
        key = _key_of(ct)
        for L in LINEARS:
            recon = q_s.dequant(rec.delta_s[L]) + q_s.dequant(
                system.s_codes[L])
            rel = _rel_mse(recon, model.true_states[(key, L)])
            assert rel < SINGLE_ROUND_GATE, f"chunk {rec.chunk_idx} L {L}: {rel}"
        assert _rel_mse(q_m1.dequant(rec.delta_m1)
                        + q_m1.dequant(system.m1_codes),
                        model.true_m1[key]) < SINGLE_ROUND_GATE
        assert _rel_mse(q_m2.dequant(rec.delta_m2)
                        + q_m2.dequant(system.m2_codes),
                        model.true_m2[key]) < SINGLE_ROUND_GATE


# =============================== 4. §4 retrieval vector: order + budget ====
def test_retrieval_vector_order_and_delta_consistency(tmp_path):
    """fp32, length = Σ S + M1 + M2 dims; S ascending, then M1, then M2.

    The order is pinned BIT-EXACTLY against an independent fresh-cache
    reseed + forward rerun (determinism), and each S segment equals the
    DELTA reconstruction dequant(delta_L) + dequant(sys_L) within the
    single-round budget (measured 0.011 — only the delta requant round
    separates the vector's absolute segment from the delta path).
    """
    model = StubModel()
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101 + i, 5) for i in range(3)]
    drv = _run_driver(model, sys_tok, chunks, tmp_path)
    drv.run()
    system = prefill_system(model, sys_tok, _make_cache())

    rec = drv.records[0]
    v = rec.cache_vector
    assert v.dtype == np.float32 and v.ndim == 1
    # length = Σ S dims + M1 dim + M2 dim (conv is NOT part of §4)
    assert v.shape[0] == len(LINEARS) * S_D + M_D + M_D == 640

    # independent rerun of chunk 0: fresh cache, reseed, forward
    ind = _make_cache()
    reseed_cache(ind, system)
    with torch.no_grad():
        model(input_ids=chunks[0], past_key_values=ind, use_cache=True)
    ind_codes = ind.snapshot_codes()

    q_s = resolve_quantizer("S", S_D)
    q_m1 = resolve_quantizer("M1", M_D)
    q_m2 = resolve_quantizer("M2", M_D)
    seg = lambda a, b: torch.from_numpy(np.ascontiguousarray(v[a:b]))

    # ORDER, bit-exact: S layers ascending, then M1, then M2
    for k, L in enumerate(sorted(LINEARS)):
        assert torch.equal(seg(k * S_D, (k + 1) * S_D),
                           q_s.dequant(ind_codes["s"][L])), (k, L)
    assert torch.equal(seg(len(LINEARS) * S_D, -M_D),
                       q_m1.dequant(ind_codes["m1"]))
    assert torch.equal(seg(-M_D, None), q_m2.dequant(ind_codes["m2"]))

    # every S segment == the DELTA reconstruction within the budget
    for k, L in enumerate(sorted(LINEARS)):
        approx = q_s.dequant(rec.delta_s[L]) + q_s.dequant(system.s_codes[L])
        rel = _rel_mse(seg(k * S_D, (k + 1) * S_D), approx)
        assert rel < SINGLE_ROUND_GATE, f"L {L}: {rel}"

    # the ordering check is non-vacuous: all 5 segments are distinct
    segs = [v[i * S_D:(i + 1) * S_D] for i in range(len(LINEARS) + 2)]
    for a, b in combinations(range(len(segs)), 2):
        assert not np.array_equal(segs[a], segs[b]), (a, b)


# ========================= 5. M1 zero-gate through the delta path ==========
def test_retrieval_vector_m1_zero_gate(tmp_path):
    """m1 write gate 0 (the W3 no-op): the M1 vector segment is EXACTLY
    zero, the delta M1 codes are zero-norm — on disk as well."""
    model = StubModel(m1_noise=0.0)
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101, 5)]
    drv = _run_driver(model, sys_tok, chunks, tmp_path)
    drv.run()
    rec = drv.records[0]
    v = rec.cache_vector

    # the M1 segment is PRESENT (codes exist, norm 0) and exactly zero
    assert v.shape[0] == len(LINEARS) * S_D + M_D + M_D
    m1_seg = v[len(LINEARS) * S_D: len(LINEARS) * S_D + M_D]
    assert np.all(m1_seg == 0.0)
    assert float(rec.delta_m1.norm) == 0.0

    system = prefill_system(model, sys_tok, _make_cache())
    assert float(system.m1_codes.norm) == 0.0

    # through the disk round-trip: the saved delta M1 codes are zero too
    snap = load_chunk(snapshot_path(str(tmp_path), 0))
    assert float(snap.m1_codes.norm) == 0.0
    assert not snap.m1_codes.idx_lo.any() and not snap.m1_codes.idx_hi.any()
    # M2 is unaffected by the M1 gate (the zero-gate is M1-specific)
    assert float(snap.m2_codes.norm) > 0.0
    assert not np.all(v[-M_D:] == 0.0)


# ================================================ 6. driver resume ==========
def test_driver_resume(tmp_path):
    model = StubModel()
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101 + i, 5) for i in range(3)]
    out = str(tmp_path)
    snaps = os.path.join(out, "snapshots")

    stats1 = _run_driver(model, sys_tok, chunks, out).run()
    assert stats1["ingested"] == 3 and stats1["skipped"] == 0
    files1 = sorted(os.listdir(snaps))
    assert files1 == [f"chunk_{i:05d}.npz" for i in range(3)]
    digest1 = {f: sha256(open(os.path.join(snaps, f), "rb").read()).hexdigest()
               for f in files1}

    # a 4th chunk arrives; a NEW driver instance re-runs (the restart
    # contract — nothing survives from the first process but the disk)
    chunks4 = chunks + [_tokens(104, 5)]
    stats2 = _run_driver(model, sys_tok, chunks4, out).run()
    assert stats2["ingested"] == 1 and stats2["skipped"] == 3
    assert stats2["done_before"] == 3 and stats2["n_chunks"] == 4

    files2 = sorted(os.listdir(snaps))
    assert files2 == [f"chunk_{i:05d}.npz" for i in range(4)]  # no duplicates
    # the 3 skipped files were NOT rewritten (byte-identical)
    for f in files1:
        d = sha256(open(os.path.join(snaps, f), "rb").read()).hexdigest()
        assert d == digest1[f], f"{f} was rewritten on resume"

    man = json.load(open(os.path.join(out, MANIFEST_NAME)))
    assert set(man["done"]) == {"0", "1", "2", "3"}
    for cid, rel in man["done"].items():
        assert os.path.isfile(os.path.join(out, rel))
    for f in files2:
        assert verify_chunk(os.path.join(snaps, f))["ok"]
    # the atomic manifest rewrite leaves no .tmp behind
    assert not os.path.exists(os.path.join(out, MANIFEST_NAME + ".tmp"))


# ============================================ 7. resume drift guard =========
def test_resume_drift_guard(tmp_path):
    model = StubModel()
    sys_a = _tokens(11, 6)
    sys_b = _tokens(999, 6)            # a DIFFERENT system prompt
    chunks = [_tokens(101 + i, 5) for i in range(2)]
    out = str(tmp_path)
    snaps = os.path.join(out, "snapshots")

    _run_driver(model, sys_a, chunks, out).run()
    man_before = json.load(open(os.path.join(out, MANIFEST_NAME)))
    files_before = sorted(os.listdir(snaps))
    digest_before = {
        f: sha256(open(os.path.join(snaps, f), "rb").read()).hexdigest()
        for f in files_before}

    # the reset point drifted: loud refusal, never silent corruption
    with pytest.raises(ValueError, match="drift"):
        _run_driver(StubModel(), sys_b, chunks, out).run()

    assert json.load(open(os.path.join(out, MANIFEST_NAME))) == man_before
    assert sorted(os.listdir(snaps)) == files_before
    for f in files_before:
        p = os.path.join(snaps, f)
        assert sha256(open(p, "rb").read()).hexdigest() == digest_before[f]
        assert verify_chunk(p)["ok"]

    # the SAME system prompt resumes cleanly (the guard is content-keyed)
    stats = _run_driver(model, sys_a, chunks, out).run()
    assert stats["ingested"] == 0 and stats["skipped"] == 2


# ================================ 8. prefill fresh-cache contract ===========
def test_prefill_system_fresh_cache_contract():
    model = StubModel()
    sys_tok = _tokens(11, 6)

    cache = _make_cache()
    system = prefill_system(model, sys_tok, cache)     # fresh: fine
    assert sorted(system.s_codes) == LINEARS
    assert sorted(system.conv_codes) == LINEARS
    assert system.m1_codes is not None and system.m2_codes is not None
    assert system.s_shapes == {L: S_SHAPE for L in LINEARS}
    assert system.conv_shapes == {L: CONV_WINDOW for L in LINEARS}
    assert system.m1_shape == M_SHAPE and system.m2_shape == M_SHAPE
    assert system.s_dtype == "float16" and system.conv_dtype == "float16"
    assert system.bits == BITS
    assert system.reference().startswith("sysstate-")

    # a second prefill on the now-dirty cache: loudly refused
    with pytest.raises(ValueError, match="already holds S codes"):
        prefill_system(model, sys_tok, cache)

    # a cache that merely ran a forward is equally refused
    cache2 = _make_cache()
    with torch.no_grad():
        model(input_ids=sys_tok, past_key_values=cache2, use_cache=True)
    with pytest.raises(ValueError, match="already holds S codes"):
        prefill_system(model, sys_tok, cache2)


# ============================ 9. D5 codes-only disk + absolute conv ========
def test_codes_only_disk_and_absolute_conv(tmp_path):
    """W16: the DEFAULT driver stores the ABSOLUTE end codes (the noise
    fix — the verbatim install); the delta codes remain available on the
    record, and the legacy chunk_protocol="delta-v1" layout still
    round-trips."""
    model = StubModel()
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101 + i, 5) for i in range(2)]
    drv = _run_driver(model, sys_tok, chunks, tmp_path)
    drv.run()
    system = prefill_system(model, sys_tok, _make_cache())

    for cid in (0, 1):
        path = snapshot_path(str(tmp_path), cid)
        with np.load(path) as z:
            assert "vector" not in z.files          # D5: codes-only npz
        snap = load_chunk(path)
        assert not hasattr(snap, "vector")          # no vector field at all
        assert snap.protocol == "absolute"          # W16: the default layout

        # s_codes on disk ARE the cache's own end codes (ZERO extra rounds
        # — the record's abs codes, bit-identical)
        rec = drv.records[cid]
        for L in LINEARS:
            assert np.array_equal(snap.s_codes[L].idx_lo, rec.abs_s[L].idx_lo)
            assert np.array_equal(snap.s_codes[L].idx_hi, rec.abs_s[L].idx_hi)
            assert float(snap.s_codes[L].norm) == float(rec.abs_s[L].norm)

        # conv codes are ABSOLUTE: bit-equal the cache's post-prefill conv
        # codes of an independent fresh-cache reseed + forward rerun
        ind = _make_cache()
        reseed_cache(ind, system)
        with torch.no_grad():
            model(input_ids=chunks[cid], past_key_values=ind, use_cache=True)
        ind_codes = ind.snapshot_codes()
        for L in LINEARS:
            assert np.array_equal(snap.conv_codes[L].idx_lo,
                                  ind_codes["conv"][L].idx_lo)
            assert np.array_equal(snap.conv_codes[L].idx_hi,
                                  ind_codes["conv"][L].idx_hi)
            assert float(snap.conv_codes[L].norm) \
                == float(ind_codes["conv"][L].norm)
            # chunk-specific absolutes, NOT the system's window (if conv
            # were stored as a window-delta, these streams would differ)
            assert not np.array_equal(snap.conv_codes[L].idx_lo,
                                      system.conv_codes[L].idx_lo)

    # the two chunks' conv codes differ from each other (per-chunk windows)
    snap0, snap1 = (load_chunk(snapshot_path(str(tmp_path), c)) for c in (0, 1))
    for L in LINEARS:
        assert not np.array_equal(snap0.conv_codes[L].idx_lo,
                                  snap1.conv_codes[L].idx_lo)

    # ---- the LEGACY layout still round-trips (chunk_protocol=delta-v1) ----
    legacy = tmp_path / "legacy"
    drv1 = _run_driver(model, sys_tok, chunks, legacy,
                       chunk_protocol="delta-v1")
    drv1.run()
    for cid in (0, 1):
        snap = load_chunk(snapshot_path(str(legacy), cid))
        assert snap.protocol == "delta-v1"
        rec = drv1.records[cid]
        for L in LINEARS:
            assert np.array_equal(snap.s_codes[L].idx_lo,
                                  rec.delta_s[L].idx_lo)
            assert np.array_equal(snap.s_codes[L].idx_hi,
                                  rec.delta_s[L].idx_hi)
            assert float(snap.s_codes[L].norm) == float(rec.delta_s[L].norm)


# =========================================== 10. manifest invariants ========
def test_manifest_invariants(tmp_path):
    model = StubModel()
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101 + i, 5) for i in range(3)]
    out = str(tmp_path)
    drv = _run_driver(model, sys_tok, chunks, out)
    drv.run()
    system = prefill_system(model, sys_tok, _make_cache())

    man = json.load(open(os.path.join(out, MANIFEST_NAME)))
    assert man["protocol"] == "delta-v1"
    assert man["chunk_protocol"] == "absolute"    # W16: the storage layout
    assert man["bits"] == BITS
    assert isinstance(man["system_ref"], str) and man["system_ref"]
    assert man["system_ref"] == system.reference()
    # vector_dims recorded == the ACTUAL vector length (Σ S + M1 + M2 dims)
    assert man["vector_dims"] == drv.records[-1].cache_vector.shape[0]
    assert man["vector_dims"] == len(LINEARS) * S_D + 2 * M_D

    # files dict: ids -> existing relative paths, matching meta
    assert set(man["done"]) == {"0", "1", "2"}
    for cid, rel in man["done"].items():
        p = os.path.join(out, rel)
        assert os.path.isfile(p)
        assert os.path.basename(p) == f"chunk_{int(cid):05d}.npz"
        s = load_chunk(p)
        assert s.chunk_id == int(cid)
        assert s.system_ref == man["system_ref"]
        # W16: the manifest's chunk_protocol pins the snapshot layout (the
        # family-level 'protocol' field stays the D4 'delta-v1' label)
        assert s.protocol == man["chunk_protocol"]
        assert s.extra == {"n_tokens": 5}


# ==================== contract extras: reseed neutralizes a used cache =====
def test_reseed_one_reused_cache_identical_records(tmp_path):
    """The D4 per-chunk contract: fresh-cache + reseed == one REUSED cache
    (reseed_cache must fully neutralize any prior chunk's state)."""
    model = StubModel()
    sys_tok = _tokens(11, 6)
    chunks = [_tokens(101 + i, 5) for i in range(3)]
    drv = _run_driver(model, sys_tok, chunks, tmp_path)
    drv.run()
    system = prefill_system(model, sys_tok, _make_cache())

    shared = _make_cache()
    for rec_fresh, ct in zip(drv.records, chunks):
        rec_reused = ingest_chunk(model, ct, shared, system)
        assert rec_reused.chunk_idx == -1   # set by the driver afterwards
        for L in LINEARS:
            assert np.array_equal(rec_fresh.delta_s[L].idx_lo,
                                  rec_reused.delta_s[L].idx_lo)
            assert np.array_equal(rec_fresh.delta_s[L].idx_hi,
                                  rec_reused.delta_s[L].idx_hi)
            assert float(rec_fresh.delta_s[L].norm) \
                == float(rec_reused.delta_s[L].norm)
            assert np.array_equal(rec_fresh.conv_codes[L].idx_lo,
                                  rec_reused.conv_codes[L].idx_lo)
        for m in ("delta_m1", "delta_m2"):
            a, b = getattr(rec_fresh, m), getattr(rec_reused, m)
            assert np.array_equal(a.idx_lo, b.idx_lo)
            assert float(a.norm) == float(b.norm)
        assert np.array_equal(rec_fresh.cache_vector, rec_reused.cache_vector)
        assert rec_fresh.n_tokens == rec_reused.n_tokens


# ==================== contract extras: (idx, tensor) pairs + factory ========
class _NoConfig:
    """A model with no .config and no usable cache route."""


def test_driver_chunk_pairs_and_no_factory_refusal(tmp_path):
    model = StubModel()
    sys_tok = _tokens(11, 6)
    pairs = [(7, _tokens(301, 5)), (2, _tokens(302, 5))]
    out = str(tmp_path)

    # (idx, tensor) pairs, non-contiguous ids: files keyed by the GIVEN id
    drv = _run_driver(model, sys_tok, pairs, out)
    stats = drv.run()
    assert stats["ingested"] == 2 and stats["skipped"] == 0
    assert [r.chunk_idx for r in drv.records] == [7, 2]
    assert sorted(os.listdir(os.path.join(out, "snapshots"))) == \
        ["chunk_00002.npz", "chunk_00007.npz"]
    man = json.load(open(os.path.join(out, MANIFEST_NAME)))
    assert set(man["done"]) == {"2", "7"}
    assert load_chunk(snapshot_path(out, 7)).chunk_id == 7

    # no cache_factory and a model without .config: loud refusal
    with pytest.raises(ValueError, match="cache_factory"):
        IngestDriver(_NoConfig(), sys_tok, pairs[:1],
                     os.path.join(out, "x"), resume=False).run()
