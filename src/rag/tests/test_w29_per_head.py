"""test_w29_per_head.py — the per-head S quantization gates (W29, the F2b
repair; FINDINGS W28 item 9, the box matrix's worst-block 7.65 at 38x).

The repair under test: a 4-D recurrent state (B, H, K, V) quantizes as
B*H PER-HEAD units of d=K*V — each unit its OWN fp32 norm (the paper's
per-vector norm recipe, arXiv:2504.19874 Theorem 1 applied per unit) —
through TurboQuant.quant_units' ONE vectorized pass, in the PERMUTED
rotation frame y = FHT(P x) (P from seed+4545: the 2^14 Hadamard's
Kronecker alignment with the 128x128 matrix layout mismatches the
codebook on low-rank heads without it — measured 0.087/0.056 vs the
0.022 floor; scripts/w28_tq_sim.py + the Kronecker experiment).

AUTHORED on the CPU dev box; RUNS anywhere torch runs (the stub route —
no model, no config, the test_tq_cache.py conventions).

Geometry & codebook hygiene: the per-unit d's hit ONLY committed
codebook npz files — d=128 (cb_b{3,4}_d128), d=1,024 (cb_b{3,4}_d1024),
d=8,192 (cb_b{3,4}_d8192, the flat A/B at 8x1024), d=16,384
(cb_b{3,4}_d16384), d=524,288 (cb_b{3,4}_d524288, the flat A/B at
production). No test writes a NEW (b, d) combination.

The gates, in file order:

  A. quant_units == the per-unit reference loop (the SAME perm frame),
     BIT-EXACT on every codes field — the vectorization pin; the decode
     side likewise (the flat (M*d,) layout).
  B. The dynamic-range immunity (THE repair property): at a 100x
     per-unit norm spread the WORST-unit rel-MSE stays at the Gaussian
     floor while the FLAT single-unit regime on the same data measures
     the spread-amplified floor (the W28 defect, reproduced). Measured
     at d=1,024 AND at the production geometry (32 x 16,384 vs the flat
     524,288) — the mini-repro of the box matrix's [0] per-block row.
  C. The codes contract: partition="units", the (M,) norms vector,
     stream lengths M*L; the type cops; to_arrays/from_arrays
     round-trip; the legacy no-units-key edge loads as "half".
  D. The perm frame is SEED-ADDRESSED: a fresh quantizer at the same
     (kind, d, seed) decodes units codes exactly; the draw is
     deterministic; a different seed draws a different P.
  E. The layer wiring (stub route): update_recurrent_state stores units
     codes on a 4-D state; the read path round-trips shape+dtype;
     s_flat=True restores the flat regime; the s_codes setter accepts
     units codes and refuses regime/unit-count mixes; non-4-D states
     fall back to flat.
  F. Zero-norm units decode to the EXACT zero vector (branchless).
  G. The install sums: sum_turboquant_codes on units codes routes the
     requant through quant_units (per-unit norms) and the flat decode
     of the sum matches the reference sum within one requant round;
     the mixed-regime refusals; sum_absolute_codes' n==1 verbatim
     hand-off.
  H. The snapshot disk edge: to_arrays-prefixed npz -> _read_unit
     round-trip; the corruption refusals; the digest covers the units
     count.

Determinism: every random draw goes through a torch.Generator pinned to
a fixed seed; the rel-MSE gates carry measured margins in comments.

rel-MSE gates: 0.08 per-unit at d >= 1,024 (the W2.2 0.06 gate's fat
tail is a d=128 effect; at d >= 1,024 the ratio concentrates — measured
~0.022-0.035, >= 2x margin); the FLAT-regime worst-unit gates assert
the HAZARD (>= 0.30 at 100x, >= 1.0 at the production 38x) — those
numbers are the tests' point, not budgets (the sim measured 10.7 at 55x
and 297 at 525x; the box matrix 7.65 at 38x).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import turboquant as tq
from turboquant import (KINDS, TQCodes, TurboQuant, UNITS_PERM_OFFSET,
                        pack_bits_t, unpack_bits_t)
import fht
from tq_cache import TQCache, TQLinearAttentionLayer, resolve_quantizer
import install as tq_install
import snapshot as snap

DEVICE = torch.device("cpu")   # the dev box path; the box runs the same


# ---- geometries (codebook-committed d's only) --------------------------------
D_UNIT = 128                    # (1, 4, 8, 16): 4 units x 128
D_UNIT_LG = 1024                # 8 units x 1024 (the fast A/B)
D_PROD = 16_384                 # the production per-head unit
N_PROD = 32                     # ... x 32 heads
D_FLAT_PROD = 524_288           # the W28 flat unit (the production A/B)
S_KIND_SEED = KINDS["S"][1]     # 101 — the D3 frame seed at every scale

REL_GATE = 0.08                 # the per-unit floor gate at d >= 1024
FLAT_HAZARD_GATE = 0.30         # the W28 defect's signature (>= 100x spread)


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-30).item()


def _per_unit_rel(xhat: torch.Tensor, x2: torch.Tensor) -> list:
    """Per-unit rel-MSE of a flat reconstruction vs the (M, d) truth."""
    xf = xhat.reshape(-1).float()
    out = []
    m, d = x2.shape
    for i in range(m):
        sl = slice(i * d, (i + 1) * d)
        out.append(_rel_mse(xf[sl], x2[i]))
    return out


def _frame_perm(d: int, seed: int) -> torch.Tensor:
    """The units frame's permutation, drawn exactly as turboquant does."""
    g = torch.Generator().manual_seed(int(seed) + UNITS_PERM_OFFSET)
    return torch.randperm(int(d), generator=g)


def _reference_quant_units(q: TurboQuant, x2: torch.Tensor) -> TQCodes:
    """The per-unit reference loop in the SAME perm frame (test A's twin):
    row i: permute -> rotate -> searchsorted(norm_i-scaled boundaries) ->
    pack; the streams concatenate row-major. Built ONLY from the module's
    own primitives — pins the VECTORIZATION, not the frame."""
    m, d = x2.shape
    x32 = x2.detach().to(torch.float32)
    perm = _frame_perm(d, q.seed)
    cb_lo, cb_hi = q._device_codebooks(x32.device)
    norms, streams_lo, streams_hi = [], [], []
    for i in range(m):
        xi = x32[i]
        norm = xi.norm().reshape(())
        norms.append(norm)
        y = fht.fht_apply(xi[perm].reshape(1, d), q._signs).reshape(d)
        idx_lo = torch.searchsorted(cb_lo.boundaries * norm,
                                    y[: q.n_lo], right=False)
        idx_hi = torch.searchsorted(cb_hi.boundaries * norm,
                                    y[q.n_lo:], right=False)
        streams_lo.append(pack_bits_t(idx_lo.to(torch.uint8), q.bits_lo))
        streams_hi.append(pack_bits_t(idx_hi.to(torch.uint8), q.bits_hi))
    return TQCodes(
        q.kind, d, torch.stack(norms),
        q.bits_lo, q.bits_hi, q.n_lo, q.n_hi,
        torch.cat(streams_lo), torch.cat(streams_hi),
        q.seed, partition="units", units=m)


# ============================================================= A: the pin === #
def test_quant_units_equals_reference_loop_bit_exact():
    """A: the vectorized pass == the per-unit loop, BIT-EXACT (norms,
    both streams, every scalar field). d=1,024 x 8 units — small enough
    to loop, big enough to concentrate."""
    q = resolve_quantizer("S", D_UNIT_LG, 3.5)
    assert q.seed == S_KIND_SEED          # the D3 frame at every scale
    g = torch.Generator().manual_seed(11)
    x2 = torch.randn(8, D_UNIT_LG, generator=g)
    codes = q.quant_units(x2)
    ref = _reference_quant_units(q, x2)
    assert codes.partition == "units" and codes.units == 8
    assert codes.d == D_UNIT_LG
    assert torch.equal(codes.norm, ref.norm)
    assert torch.equal(codes.idx_lo, ref.idx_lo)
    assert torch.equal(codes.idx_hi, ref.idx_hi)
    assert (codes.bits_lo, codes.bits_hi, codes.n_lo, codes.n_hi,
            codes.seed) == (ref.bits_lo, ref.bits_hi, ref.n_lo,
                            ref.n_hi, ref.seed)


def test_dequant_units_equals_reference_dequant():
    """A (decode side): dequant(quant_units(x2)) == the per-unit loop's
    dequant concatenated — the flat (M*d,) layout, bit-exact."""
    q = resolve_quantizer("S", D_UNIT_LG, 3.5)
    g = torch.Generator().manual_seed(12)
    x2 = torch.randn(6, D_UNIT_LG, generator=g)
    codes = q.quant_units(x2)
    out = q.dequant(codes, dtype=torch.float32)
    assert out.shape == (6 * D_UNIT_LG,)
    m = 6
    l_lo = codes.idx_lo.numel() // m     # byte-aligned: exact division
    l_hi = codes.idx_hi.numel() // m
    perm_inv = torch.argsort(_frame_perm(D_UNIT_LG, q.seed))
    cb_lo, cb_hi = q._device_codebooks(DEVICE)
    rows = []
    for i in range(m):
        idx_lo = unpack_bits_t(codes.idx_lo[i * l_lo:(i + 1) * l_lo],
                               q.bits_lo, q.n_lo)
        idx_hi = unpack_bits_t(codes.idx_hi[i * l_hi:(i + 1) * l_hi],
                               q.bits_hi, q.n_hi)
        yhat = torch.cat([cb_lo.centroids[idx_lo.long()],
                          cb_hi.centroids[idx_hi.long()]])
        row = fht.fht_adjoint(yhat.reshape(1, D_UNIT_LG),
                              q._signs).reshape(D_UNIT_LG)
        rows.append(row[perm_inv] * codes.norm[i])
    ref = torch.cat(rows)
    assert torch.equal(out, ref)


# ================================== B: the dynamic-range immunity (THE fix) === #
def _spread_state(m: int, d: int, spread: float, seed: int) -> torch.Tensor:
    """(M, d) units with log-uniform per-unit norm spread (the GDN-like
    per-head hazard): unit i scaled by spread^(u_i), u_i ~ U[0, 1]."""
    g = torch.Generator().manual_seed(seed)
    x2 = torch.randn(m, d, generator=g)
    u = torch.rand(m, generator=g)
    scales = torch.pow(torch.tensor(spread), u)
    return x2 * scales.reshape(m, 1)


def test_per_head_units_are_spread_immune_d1024():
    """B: at a 100x norm spread the worst-unit rel-MSE stays at the floor
    (< REL_GATE) — AND the flat single-unit regime on the SAME data shows
    the W28 hazard (worst-unit >= FLAT_HAZARD_GATE): the A/B in one test
    (d=1,024 x 8 units; the flat A/B at d=8,192)."""
    m, d, spread = 8, D_UNIT_LG, 100.0
    x2 = _spread_state(m, d, spread, 21)
    q_units = resolve_quantizer("S", d, 3.5)
    xhat = q_units.dequant(q_units.quant_units(x2))
    worst_units = max(_per_unit_rel(xhat, x2))
    assert worst_units < REL_GATE, (
        f"per-head worst-unit rel-MSE {worst_units:.4f} >= {REL_GATE} — "
        f"the dynamic-range immunity is broken (the repair regressed?)")
    # the flat regime on the same data: ONE (m*d,)-unit, ONE norm
    q_flat = resolve_quantizer("S", m * d, 3.5)
    xflat = x2.reshape(-1)
    xfhat = q_flat.dequant(q_flat.quant(xflat))
    worst_flat = max(_per_unit_rel(xfhat, x2))
    assert worst_flat >= FLAT_HAZARD_GATE, (
        f"the flat regime's worst-unit rel-MSE {worst_flat:.4f} fell below "
        f"{FLAT_HAZARD_GATE} — the W28 hazard fixture drifted (re-measure; "
        f"the hazard IS the point of this comparison)")


def test_per_head_production_geometry_the_matrix_row():
    """B (production geometry — the mini-repro of the box matrix's [0]
    per-head-spread row): 32 units of d=16,384 at a 38x spread (the box's
    measured spread) — worst-unit < 0.05; the flat 524,288-unit regime on
    the same state reproduces the soup-severity floor (worst >= 1.0).
    FINDINGS W28 item 9 predicted this (the sim: 10.7 at 55x); the box
    measured 7.65; the repair collapses it to the floor."""
    m, d, spread = N_PROD, D_PROD, 38.0
    x2 = _spread_state(m, d, spread, 22)
    q_units = resolve_quantizer("S", d, 3.5)
    xhat = q_units.dequant(q_units.quant_units(x2))
    worst_units = max(_per_unit_rel(xhat, x2))
    assert worst_units < 0.05, (
        f"production-geometry worst-unit rel-MSE {worst_units:.4f} — the "
        f"per-head floor gate is 0.05 (the sim measured 0.023)")
    q_flat = resolve_quantizer("S", D_FLAT_PROD, 3.5)
    xflat = x2.reshape(-1)
    xfhat = q_flat.dequant(q_flat.quant(xflat))
    worst_flat = max(_per_unit_rel(xfhat, x2))
    assert worst_flat >= 1.0, (
        f"the flat regime's production worst-unit rel-MSE {worst_flat:.4f} "
        f"< 1.0 — the hazard fixture drifted (the box measured 7.65; the "
        f"sim 10.7 at 55x — re-pin the seed/spread)")


# ================================================= C: the codes contract ====== #
def _mk_units_codes(m: int = 4, d: int = D_UNIT) -> tuple:
    q = resolve_quantizer("S", d, 3.5)
    g = torch.Generator().manual_seed(31)
    x2 = torch.randn(m, d, generator=g)
    return q, x2, q.quant_units(x2)


def test_units_codes_contract_fields_and_lengths():
    """C: partition/units/norm vector/stream lengths (M*L each); the
    norms are the TRUE row norms (the paper's per-vector recipe)."""
    q, x2, codes = _mk_units_codes()
    m, d = x2.shape
    assert codes.partition == "units" and codes.units == m
    assert codes.norm.dtype == torch.float32 and codes.norm.shape == (m,)
    assert codes.idx_lo.dtype == torch.uint8 and codes.idx_hi.dtype == torch.uint8
    l_lo = (codes.n_lo * codes.bits_lo + 7) // 8
    l_hi = (codes.n_hi * codes.bits_hi + 7) // 8
    assert codes.idx_lo.shape == (m * l_lo,)
    assert codes.idx_hi.shape == (m * l_hi,)
    assert torch.allclose(codes.norm, x2.float().norm(dim=1), atol=1e-5)
    assert codes.nbytes() == codes.idx_lo.numel() + codes.idx_hi.numel() + 4 * m


def test_units_codes_type_cops_refuse_loudly():
    """C: a 0-dim norm on a units unit, a missing units count, a stray
    units count on a half unit, a norm-length mismatch — all TypeError
    at construction."""
    q, x2, codes = _mk_units_codes()
    with pytest.raises(TypeError, match="norm"):
        # a TRUE 0-dim norm (1 element) — reshape(()) on the 4-element
        # norms vector itself raises RuntimeError in torch BEFORE the
        # constructor's type cops can fire (the authored-test bug the
        # box's Gate Zero caught, W29.2)
        TQCodes(codes.kind, codes.d, codes.norm[:1].reshape(()),
                codes.bits_lo, codes.bits_hi, codes.n_lo, codes.n_hi,
                codes.idx_lo, codes.idx_hi, codes.seed,
                partition="units", units=codes.units)
    with pytest.raises(TypeError, match="units"):
        TQCodes(codes.kind, codes.d, codes.norm,
                codes.bits_lo, codes.bits_hi, codes.n_lo, codes.n_hi,
                codes.idx_lo, codes.idx_hi, codes.seed,
                partition="units", units=None)
    with pytest.raises(TypeError, match="stray units|units"):
        # 0-dim norm is the CORRECT type for a half unit — the stray
        # units count is the corruption under test here
        TQCodes(codes.kind, codes.d, codes.norm[:1].reshape(()),
                codes.bits_lo, codes.bits_hi, codes.n_lo, codes.n_hi,
                codes.idx_lo, codes.idx_hi, codes.seed,
                partition="half", units=4)
    with pytest.raises(TypeError, match=r"\(4,\)"):
        TQCodes(codes.kind, codes.d, codes.norm[:3].clone(),
                codes.bits_lo, codes.bits_hi, codes.n_lo, codes.n_hi,
                codes.idx_lo, codes.idx_hi, codes.seed,
                partition="units", units=4)


def test_units_codes_serialization_roundtrip():
    """C: to_arrays -> from_arrays is the identity on every field; a
    LEGACY array dict (no units key, 0-dim norm) loads as 'half'."""
    q, x2, codes = _mk_units_codes()
    a = codes.to_arrays()
    assert a["units"].dtype == np.int64 and int(a["units"]) == codes.units
    assert np.asarray(a["norm"]).shape == (codes.units,)
    codes2 = TQCodes.from_arrays(a)
    assert codes2.partition == "units" and codes2.units == codes.units
    assert torch.equal(codes2.norm, codes.norm)
    assert torch.equal(codes2.idx_lo, codes.idx_lo)
    assert torch.equal(codes2.idx_hi, codes.idx_hi)
    assert codes.to(DEVICE).nbytes() == codes.nbytes()
    legacy = dict(a)
    legacy.pop("units")
    legacy["partition"] = np.array("half")
    legacy["norm"] = np.float32(1.0)
    lg = TQCodes.from_arrays(legacy)
    assert lg.partition == "half" and lg.units is None
    assert lg.norm.dim() == 0


# ================================= D: the seed-addressed perm frame =========== #
def test_perm_frame_is_seed_addressed_and_registry_independent():
    """D: a FRESH quantizer at the same (kind, d, seed) decodes units
    codes exactly (the frame is seed-addressed — no shared state); the
    draw is deterministic; a different seed draws a different P."""
    q, x2, codes = _mk_units_codes(m=4, d=D_UNIT)
    q_fresh = TurboQuant(kind="custom", bits=3.5, d=D_UNIT,
                         seed=S_KIND_SEED)
    assert torch.equal(q._units_perm(DEVICE), q_fresh._units_perm(DEVICE))
    assert torch.equal(q._units_perm(DEVICE),
                       _frame_perm(D_UNIT, S_KIND_SEED))
    assert torch.equal(q_fresh.dequant(codes, dtype=torch.float32),
                       q.dequant(codes, dtype=torch.float32))
    q_other = TurboQuant(kind="custom", bits=3.5, d=D_UNIT, seed=999)
    assert not torch.equal(q._units_perm(DEVICE),
                           q_other._units_perm(DEVICE))


# ================================================= E: the layer wiring ======== #
S_STUB = (1, 4, 8, 16)   # 4 heads x 128 dims — the 4-D per-head state


def _stub_layer(s_flat: bool = False) -> TQLinearAttentionLayer:
    cache = TQCache(layer_types=["linear_attention"],
                    bits=3.5, online=True, s_flat=s_flat)
    return cache.layers[0]


def test_layer_update_recurrent_state_stores_units_codes():
    """E: the 4-D write path stores partition='units' codes at the
    per-head geometry (units=B*H, d=K*V, the kind's seed)."""
    layer = _stub_layer()
    g = torch.Generator().manual_seed(41)
    s = torch.randn(*S_STUB, generator=g, dtype=torch.float16)
    out = layer.update_recurrent_state(s)
    codes = layer.s_codes
    assert codes is not None and codes.partition == "units"
    assert codes.units == 4 and codes.d == 128
    assert codes.seed == S_KIND_SEED
    assert layer._s_units == 4 and layer._s_unit_d == 128
    assert out.shape == s.shape and out.dtype == s.dtype
    assert _rel_mse(out.float(), s.float()) < REL_GATE
    assert layer.writes["s"] == 1


def test_layer_read_path_roundtrips_shape_and_dtype():
    """E: recurrent_states[0] (the model's read) dequantizes to the
    stored state's shape/dtype at the floor."""
    layer = _stub_layer()
    g = torch.Generator().manual_seed(42)
    s = torch.randn(*S_STUB, generator=g, dtype=torch.float16)
    layer.update_recurrent_state(s)
    t = layer.recurrent_states[0]
    assert t.shape == s.shape and t.dtype == torch.float16
    assert _rel_mse(t.float(), s.float()) < REL_GATE
    assert layer.reads["s"] == 1


def test_layer_s_flat_restores_the_w28_flat_regime():
    """E: s_flat=True -> partition='half', d=numel, 0-dim norm (the
    legacy regime; the A/B flag)."""
    layer = _stub_layer(s_flat=True)
    g = torch.Generator().manual_seed(43)
    s = torch.randn(*S_STUB, generator=g, dtype=torch.float16)
    out = layer.update_recurrent_state(s)
    codes = layer.s_codes
    assert codes.partition == "half" and codes.d == 4 * 128
    assert codes.units is None and codes.norm.dim() == 0
    assert layer._s_units is None
    assert out.shape == s.shape


def test_layer_s_codes_setter_accepts_units_and_refuses_mixes():
    """E: the setter resolves the frame from the codes' own cardinals;
    a units-count drift and a flat-installed-into-per-head mix refuse."""
    layer = _stub_layer()   # not yet initialized: the codes teach it
    q = resolve_quantizer("S", D_UNIT, 3.5)
    g = torch.Generator().manual_seed(44)
    x2 = torch.randn(4, D_UNIT, generator=g)
    codes = q.quant_units(x2)
    layer.s_codes = codes
    assert layer._tq_s.d == D_UNIT and layer._s_units == 4
    assert layer._s_shape == (4 * D_UNIT,)
    layer.s_codes = q.quant_units(x2 * 0.5)      # matching: accepted
    codes_8 = q.quant_units(torch.randn(
        8, D_UNIT, generator=torch.Generator().manual_seed(45)))
    with pytest.raises(ValueError, match="units=8|per-head geometry"):
        layer.s_codes = codes_8
    flat = resolve_quantizer("S", 4 * D_UNIT, 3.5).quant(
        torch.randn(4 * D_UNIT, generator=torch.Generator().manual_seed(46)))
    with pytest.raises(ValueError, match="regime|frame"):
        layer.s_codes = flat


def test_layer_non_4d_state_falls_back_to_flat():
    """E: a non-4-D state (e.g. a 2-D install-time tensor) resolves the
    flat regime even with s_flat=False — the per-head split needs the
    (B, H, K, V) layout to derive the units."""
    layer = _stub_layer()
    g = torch.Generator().manual_seed(47)
    s2 = torch.randn(4 * D_UNIT, generator=g, dtype=torch.float16)
    layer.update_recurrent_state(s2)
    codes = layer.s_codes
    assert codes.partition == "half" and codes.d == 4 * D_UNIT


# ================================================= F: zero-norm units ========= #
def test_zero_norm_units_decode_to_exact_zeros():
    """F: a zero row in the batch reconstructs to the EXACT zero vector
    (0 x finite, branchless) and its neighbors are unaffected."""
    q = resolve_quantizer("S", D_UNIT_LG, 3.5)
    g = torch.Generator().manual_seed(51)
    x2 = torch.randn(5, D_UNIT_LG, generator=g)
    x2[2] = 0.0
    codes = q.quant_units(x2)
    assert float(codes.norm[2]) == 0.0
    out = q.dequant(codes, dtype=torch.float32)
    zrow = out[2 * D_UNIT_LG:(3 * D_UNIT_LG)]
    assert torch.count_nonzero(zrow) == 0
    rels = _per_unit_rel(out, x2)
    for i in (0, 1, 3, 4):
        assert rels[i] < REL_GATE


# ================================================= G: the install sums ======== #
def _sys_and_deltas(m: int = 4, d: int = D_UNIT):
    q = resolve_quantizer("S", d, 3.5)
    g = torch.Generator().manual_seed(61)
    sys_x = torch.randn(m, d, generator=g)
    d1 = torch.randn(m, d, generator=torch.Generator().manual_seed(62))
    d2 = torch.randn(m, d, generator=torch.Generator().manual_seed(63))
    return q, sys_x, q.quant_units(sys_x), [q.quant_units(d1),
                                            q.quant_units(d2)]


def test_sum_turboquant_codes_units_regime():
    """G: the delta sum on units codes routes the requant through
    quant_units (units codes out) and the flat decode of the sum matches
    the reference sum within one requant round (each dequant ~2.2%/unit,
    three addends + one requant: gate 0.12)."""
    q, sys_x, system, deltas = _sys_and_deltas()
    summed = tq_install.sum_turboquant_codes(system, deltas, "S", 3.5)
    assert summed.partition == "units" and summed.units == system.units
    acc = q.dequant(system)
    for dcod in deltas:
        acc = acc + q.dequant(dcod)
    assert _rel_mse(q.dequant(summed, dtype=torch.float32), acc) < 0.12
    # the direct reference requant of the same accumulator: bit-identical
    ref = q.quant_units(acc.reshape(system.units, system.d))
    assert torch.equal(summed.norm, ref.norm)
    assert torch.equal(summed.idx_lo, ref.idx_lo)
    assert torch.equal(summed.idx_hi, ref.idx_hi)


def test_sum_turboquant_codes_refuses_regime_mix():
    """G: a FLAT delta into a units frame (and vice versa) is refused
    loudly — the re-ingest hint, never a silent cross-regime sum."""
    q, sys_x, system, deltas = _sys_and_deltas()
    flat_q = resolve_quantizer("S", 4 * D_UNIT, 3.5)
    flat_delta = flat_q.quant(
        torch.randn(4 * D_UNIT, generator=torch.Generator().manual_seed(64)))
    with pytest.raises(ValueError, match="regime mix"):
        tq_install.sum_turboquant_codes(system, [flat_delta], "S", 3.5)
    flat_sys = flat_q.quant(
        torch.randn(4 * D_UNIT, generator=torch.Generator().manual_seed(65)))
    with pytest.raises(ValueError, match="regime mix"):
        tq_install.sum_turboquant_codes(flat_sys, deltas, "S", 3.5)


def test_sum_absolute_codes_units_n1_verbatim():
    """G: the absolute protocol's n==1 path hands the codes off VERBATIM
    (zero requant rounds — object identity, the W16 contract on units
    codes)."""
    q, sys_x, system, _ = _sys_and_deltas()
    g = torch.Generator().manual_seed(66)
    abs_x = torch.randn(4, D_UNIT, generator=g)
    a0 = q.quant_units(abs_x)
    out = tq_install.sum_absolute_codes(system, [a0], "S", 3.5)
    assert out is a0


# ================================================= H: the snapshot edge ======= #
def _write_units_npz(path: str, codes: TQCodes, prefix: str = "s_0_",
                     drop_units: bool = False, flat_norm: bool = False,
                     stray_units: bool = False, truncate: bool = False):
    a = codes.to_arrays()
    out = {}
    for k, v in a.items():
        if k == "units" and drop_units:
            continue
        out[f"{prefix}{k}"] = v
    if flat_norm:
        out[f"{prefix}norm"] = np.float32(1.0)
    if stray_units:
        out[f"{prefix}units"] = np.int64(codes.units)
    if truncate:
        out[f"{prefix}idx_lo"] = out[f"{prefix}idx_lo"][:-1]
    np.savez(path, **out)


def test_snapshot_read_unit_units_roundtrip(tmp_path):
    """H: a to_arrays-prefixed npz -> _read_unit rebuilds the codes
    exactly (the disk edge); the digest covers the units count; the
    corruption refusals fire loudly."""
    q = resolve_quantizer("S", D_UNIT, 3.5)
    g = torch.Generator().manual_seed(71)
    codes = q.quant_units(torch.randn(4, D_UNIT, generator=g))
    p = str(tmp_path / "u.npz")
    _write_units_npz(p, codes)
    z = np.load(p)
    back = snap._read_unit(z, "s_0_")
    assert back.partition == "units" and back.units == codes.units
    assert torch.equal(back.norm, codes.norm)
    assert torch.equal(back.idx_lo, codes.idx_lo)
    assert torch.equal(back.idx_hi, codes.idx_hi)
    # the digest covers the units count: two codes differing only in
    # (units, the norms vector's length) digest differently
    a = codes.to_arrays()
    a2 = dict(a)
    a2["units"] = np.int64(int(a["units"]) + 1)
    a2["norm"] = np.concatenate(
        [np.asarray(a["norm"], dtype=np.float32), np.float32(0.5)])
    d1 = snap._unit_digest_bytes("s:0", codes)
    d2 = snap._unit_digest_bytes("s:0", TQCodes.from_arrays(a2))
    assert d1 != d2
    # corruption refusals
    p2 = str(tmp_path / "drop.npz")
    _write_units_npz(p2, codes, drop_units=True)
    with pytest.raises(ValueError, match="units"):
        snap._read_unit(np.load(p2), "s_0_")
    p3 = str(tmp_path / "stray.npz")
    flat = resolve_quantizer("S", 4 * D_UNIT, 3.5).quant(
        torch.randn(4 * D_UNIT, generator=torch.Generator().manual_seed(72)))
    _write_units_npz(p3, flat, stray_units=True)
    with pytest.raises(ValueError, match="stray"):
        snap._read_unit(np.load(p3), "s_0_")
    p4 = str(tmp_path / "flatnorm.npz")
    _write_units_npz(p4, codes, flat_norm=True)
    with pytest.raises(ValueError, match=r"norm"):
        snap._read_unit(np.load(p4), "s_0_")
    p5 = str(tmp_path / "short.npz")
    _write_units_npz(p5, codes, truncate=True)
    with pytest.raises(ValueError, match="bytes|expected"):
        snap._read_unit(np.load(p5), "s_0_")


# ---- the unsupported-combination refusals (loud, never silent) =============== #
def test_quant_units_refuses_split_qjl_and_misalignment():
    """The W15 split, the W9.2 qjl flag, and non-byte-aligned splits all
    refuse LOUDLY on quant_units (never a silent wrong path)."""
    # the qjl flag
    q_qjl = TurboQuant(kind="custom", bits=3.5, d=D_UNIT,
                       seed=S_KIND_SEED, qjl=True)
    with pytest.raises(ValueError, match="qjl"):
        q_qjl.quant_units(torch.randn(2, D_UNIT))
    # the W15 split (channel-grouped, non-integer bits)
    q_split = TurboQuant(kind="custom", bits=3.5, d=D_UNIT,
                         seed=S_KIND_SEED, group=4)
    with pytest.raises(ValueError, match="outlier split"):
        q_split.quant_units(torch.randn(2, D_UNIT))
    # a mis-aligned split: hand-set n_lo on a FRESH instance (65*3=195
    # bits — NOT byte-aligned; the guard is pure cardinals arithmetic)
    q = TurboQuant(kind="custom", bits=3.5, d=D_UNIT, seed=S_KIND_SEED)
    q.n_lo = 65
    with pytest.raises(ValueError, match="byte-aligned"):
        q.quant_units(torch.randn(2, D_UNIT))
