"""test_w20_qjl_l2norm.py — W20.9.3: the QJL/l2norm CUDA-gated suite.

The W20.9 kernel surface's pytest legs — flute_extended's tq_qjl_quant /
tq_qjl_compensate (W20.9.1) and tq_l2norm_rows (W20.9.2) — over TASKS.md
W20.9.3's brief: the QJL kernel-vs-torch parity (allclose 1e-5 on the
compensated reconstruction and the residual chain's x_mse legs, gamma
<= 1 ulp, the int8 sketch under the tie budget), the l2norm differential
vs torch's x.norm(dim=-1) (<= 1 ulp, the zero row exactly 0.0), the
run-to-run determinism (torch.equal), and the flag-off non-invasiveness
(the dispatcher's default qjl-off path unchanged when the extension is
present). The QJL leg folds into G20.P3's scope per the ledger (TASKS.md
§4's G20.P3 row: "K3+K4: TQ kernel parity (quant+dequant+qjl) ...
authored in W20.8 (qjl folds in W20.9)"), and the l2norm leg is the
serving index frame's normalize step G20.5's INDEX leg consumes
(TASKS.md:1443). Thresholds mirror TASKS.md §4 1:1 (the registry in
scripts/gpu/w20_gates.py THRESHOLDS is the source of truth; this file
defines NO thresholds beyond §4's):

    G20.P1/P3 (§4): "codes equal except ties <= 0.01%; norm <= 1 ulp;
        recon allclose <= 1e-5 rel; run-to-run torch.equal (CUDA)"
        -> TIES_MAX_FRACTION = 0.0001, NORM_MAX_ULPS = 1,
           RECON_ALLCLOSE_RTOL/ATOL = 1e-5

THE TWIN (the W20.8.3 suite's documented choice, narrowed to the qjl
bodies): the torch leg of every QJL differential is turboquant's OWN
reference body, driven directly — _attach_qjl (turboquant.py:860-901)
and _qjl_compensation_term (turboquant.py:1155-1193) on TQCodes rebuilt
from the KERNEL's own batched outputs (the _codes_from_batched_row
convention), so both legs see the SAME x and the SAME codes: the
differential is exactly kernel-chain vs torch-body with ZERO test-local
math and zero convention drift. No dispatcher monkeypatch is needed —
the qjl bodies are the torch reference by construction (the dispatcher
routes the qjl flag to them; the butterflies both legs ride are the same
committed kernels the twin's fht_apply / fht_adjoint route to at
eligible K, the 9.1 banner's parity claim, kernel_tq.cu:3633-3641).

PROMPT.md §1: nothing here is RUN on a CPU box and nothing here is
CPU-runnable — importing w20._gpu_guard hard-skips this module loudly
on a CPU box (the guard's import-time require_cuda() raises
pytest.skip(..., allow_module_level=True)), and an extension built
without the W20.9 surface skips the module loudly too (the same
allow_module_level=True module skip: a skip, never a collection
error). Everything is deterministic: every input is rebuilt from a
seed recorded in this file; no unrecorded randomness anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Path-robust guard import. pytest's default prepend import mode resolves
# this module as w20.test_w20_qjl_l2norm with basedir src/rag/tests on
# sys.path (src/rag/tests has no __init__.py; the w20/ package does), and
# importing the src/rag/tests/conftest.py already put that directory on
# sys.path before this module is collected — so the plain
# `from w20 import _gpu_guard` resolves on the box (the guard docstring's
# `src.rag.tests.w20` form would NOT: conftest/_paths anchor src/rag,
# src/scripts, src/flute_extended — never the repo root). The explicit
# insert below keeps the import resolving under any import mode.
_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from w20 import _gpu_guard  # noqa: E402

_gpu_guard.require_cuda()  # module top: CPU boxes skip loudly here

# The kernel surface under test — conftest.py anchors src/rag on sys.path
# (house convention, the same import test_w20_tq_kernels.py uses); the
# lazy probe is the dispatcher's own (turboquant._load_tq_kernel ->
# fht._load_kernel + the TQ-name hasattr checks, W20.8.2). The module
# skip covers BOTH absences: the extension not built at all, and an
# older build without the W20.9 names (qjl pair + l2norm).
import turboquant  # noqa: E402
from turboquant import TQCodes, TurboQuant  # noqa: E402

_C = turboquant._load_tq_kernel()
if (_C is None or not hasattr(_C, "tq_qjl_quant")
        or not hasattr(_C, "tq_qjl_compensate")
        or not hasattr(_C, "tq_l2norm_rows")):
    pytest.skip(
        "flute_extended not built with the W20.9 qjl/l2norm surface — "
        "build on the A10G box (scripts/gpu/w20_provision.sh)",
        allow_module_level=True,
    )

# --- thresholds: TASKS.md §4, mirrored 1:1 from w20_gates.py THRESHOLDS -- #
# §4 G20.P3: "same as G20.P1 thresholds on kernel outputs" (G20.P1: "codes
# equal except ties <= 0.01%; norm <= 1 ulp; recon allclose <= 1e-5 rel;
# run-to-run torch.equal (CUDA)") — the qjl leg folds into that row per
# the ledger. No thresholds beyond §4's live in this file.
TIES_MAX_FRACTION = 0.0001      # THRESHOLDS["G20.P3"]["ties_max_fraction"]
NORM_MAX_ULPS = 1               # THRESHOLDS["G20.P3"]["norm_max_ulps"]
RECON_ALLCLOSE_RTOL = 1e-5      # THRESHOLDS["G20.P3"]["recon_allclose_rtol"]
RECON_ALLCLOSE_ATOL = 1e-5      # THRESHOLDS["G20.P3"]["recon_allclose_atol"]

# --- the QJL rung (a small seeded batch; every seed recorded here) ------ #
# d = 2^15 is the smallest kernel-eligible width (strictly above one
# 64 KiB butterfly tile): the whole QJL chain exercises at prefill-row
# scale while the batch stays small. bits 3.5 = the b3/b4 production
# split (n_lo = n_hi = d/2, a 16-multiple — the pack-slice alignment).
QJL_RUNG = (2**15, 4, 915011, 915101)   # (d, units, x_seed, frame_seed)

# --- the l2norm ladder (the serving frame's widths; seeds recorded) ----- #
# N = 1,024 (a typical embedding width, power of two), N = 768 (NON-power-
# of-two — the kernel's any-N envelope: no partial axis, no pow2 clause),
# and the fp16 width (the kernel's second input surface; the reference is
# the WIDENED twin — the kernel's fp16->fp32 cast is exact, so torch's
# association on the same widened values is the reference).
L2_LADDER = (
    # (M, N, dtype, x_seed)
    (8, 1024, torch.float32, 902101),
    (5, 768, torch.float32, 902202),
    (8, 1024, torch.float16, 902303),
)
_L2_IDS = [f"M{M}-N{N}-{str(dtype).split('.')[-1]}"
           for M, N, dtype, _ in L2_LADDER]

# --- the flag-off rung ---------------------------------------------------- #
FLAG_RUNG = (2**15, 916001, 916101)      # (d, x_seed, frame_seed)


def _ulp_distance_fp32(a: torch.Tensor, b: torch.Tensor) -> int:
    """view(torch.int32) ulp distance, -0.0-safe — mirrors
    scripts/gpu/w20_gates.py::_ulp_distance_fp32 (the G20.P1/P3 (b)
    check). Norms are non-negative, where the signed int32 bit view is
    monotonic: |ia - ib| IS the ulp distance; a -0.0 is normalized to
    +0.0 first so a degenerate zero norm never reads as 2^31 ulps."""
    a = a.detach().to("cpu", torch.float32).reshape(-1)
    b = b.detach().to("cpu", torch.float32).reshape(-1)
    a = torch.where(a == 0, torch.zeros_like(a), a)
    b = torch.where(b == 0, torch.zeros_like(b), b)
    if a.numel() == 0 or b.numel() == 0:
        return 0
    ia = a.contiguous().view(torch.int32).to(torch.int64)
    ib = b.contiguous().view(torch.int32).to(torch.int64)
    return int((ia - ib).abs().max().item())


def _seeded_rows(M: int, N: int, seed: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """(M, N) rows from a recorded seed — drawn fp32 on the CPU generator
    (bit-exact across boxes) and moved to the device in `dtype` (the
    fp16 rung's values are the same draws narrowed once — the single
    deterministic cast; the G20.P2 surface's own recipe)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(M, N, generator=gen,
                       dtype=torch.float32).to("cuda", dtype)


def _quantizer(d: int, frame_seed: int, qjl: bool = False) -> TurboQuant:
    """The rung's quantizer: bits 3.5 (the b3/b4 split, the production
    recipe), flat partition, custom kind with the recorded frame seed
    (the W20.8.3 suite's recipe); `qjl` only flips the W9.2 flag — the
    frame is IDENTICAL either way (same first-draw signs, same codebook
    mirrors, same second draw, turboquant.py:599-606)."""
    return TurboQuant(kind="custom", bits=3.5, d=d, seed=frame_seed,
                      qjl=qjl)


def _codes_from_batched_row(q: TurboQuant, codes_lo: torch.Tensor,
                            codes_hi: torch.Tensor,
                            norms: torch.Tensor, u: int) -> TQCodes:
    """The dispatcher's own per-unit packing of the batched outputs
    (_quant_kernel's reshape, mirrored): row u of the (M, L) packed
    streams as the 1-D idx streams, norms[u] as the 0-dim fp32 norm,
    the quantizer's own bookkeeping — the twin bodies' codes input."""
    return TQCodes(q.kind, q.d, norms[u].reshape(()),
                   q.bits_lo, q.bits_hi, q.n_lo, q.n_hi,
                   codes_lo[u], codes_hi[u], q.seed)


# --- (a) the QJL kernel pair vs the torch twin ---------------------------- #
def test_qjl_kernel_vs_torch():
    """The QJL parity, both legs on the SAME seeded batch and the SAME
    codes on CUDA: the kernel leg (tq_qjl_quant + tq_qjl_compensate over
    the batched codes) vs the torch twin (_attach_qjl +
    _qjl_compensation_term over per-unit TQCodes rebuilt from the SAME
    kernel outputs — apples-to-apples: only the compute body differs).
    §4's G20.P3 (= G20.P1) thresholds: (a) the residual chain's x_mse
    legs — the kernel decode vs the twin decode of the SAME codes —
    allclose 1e-5/1e-5 (r = x - x_mse with x identical: the x_mse legs
    ARE the residual chain); (b) gamma <= 1 ulp (view-int32, -0.0-safe);
    (c) the int8 sketch under the tie budget (<= 0.01% — a sign flip
    needs a near-zero projection, the boundary-exact class); (d) the
    compensated reconstruction allclose 1e-5/1e-5."""
    d, units, x_seed, frame_seed = QJL_RUNG
    q = _quantizer(d, frame_seed)          # qjl OFF: the base MSE layer
    q_j = _quantizer(d, frame_seed, qjl=True)   # the twin (same frame)
    x = _seeded_rows(units, d, x_seed, torch.float32).contiguous()
    dev = x.device
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)
    # the SECOND draw (the quantizer's own unconditional field,
    # turboquant.py:605-606: seed + QJL_SEED_OFFSET), mirrored to the
    # device — the same vector both legs' butterflies rotate through.
    qjl_rot = q._qjl_signs_vec.detach().to(dev, torch.float32).contiguous()

    # the base codes: ONE batched kernel encode — deterministic, and the
    # SAME bits feed both legs (the codes themselves are the 8.3 suite's
    # differential; here they are the shared input).
    codes_lo, codes_hi, norms = _C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)

    # ---- the kernel leg: tq_qjl_quant + tq_qjl_compensate, batched ----
    k_signs, k_gamma = _C.tq_qjl_quant(
        x, signs, codes_lo, codes_hi, norms,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, qjl_rot)
    g = _C.tq_dequant_batched(
        codes_lo, codes_hi, norms, signs,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
    k_out = _C.tq_qjl_compensate(g, k_signs, k_gamma, qjl_rot)

    assert k_signs.shape == (units, d) and k_signs.dtype == torch.int8
    assert k_gamma.shape == (units,) and k_gamma.dtype == torch.float32
    assert k_out.shape == (units, d) and k_out.dtype == torch.float32

    # ---- the torch twin leg: _attach_qjl + _qjl_compensation_term, per
    # unit, on TQCodes rows rebuilt from the SAME kernel outputs ------
    t_signs, t_gamma, t_mse, t_out = [], [], [], []
    for u in range(units):
        c_u = _codes_from_batched_row(q, codes_lo, codes_hi, norms, u)
        q_j._attach_qjl(c_u, x[u])                # the twin's sketch+gamma
        m_u = q_j._dequant_mse(c_u, dtype=torch.float32)   # the twin x_mse
        t_signs.append(c_u.qjl_signs)
        t_gamma.append(c_u.gamma.reshape(()))
        t_mse.append(m_u)
        t_out.append(q_j._qjl_compensation_term(c_u, m_u))
    t_signs = torch.stack(t_signs)     # (units, d) int8
    t_gamma = torch.stack(t_gamma)     # (units,) fp32
    t_mse = torch.stack(t_mse)         # (units, d) fp32
    t_out = torch.stack(t_out)         # (units, d) fp32

    # (a) the residual chain's x_mse legs (the kernel decode vs the twin
    # decode of the SAME codes — x identical, so r parity rides this)
    assert torch.allclose(g, t_mse, rtol=RECON_ALLCLOSE_RTOL,
                          atol=RECON_ALLCLOSE_ATOL), (
        "the QJL residual chain drifts beyond §4's 1e-5: the kernel "
        "decode vs the torch decode of the SAME codes")
    # (b) gamma <= 1 ulp (§4's norm threshold, the 7.1-chain class)
    assert _ulp_distance_fp32(k_gamma, t_gamma) <= NORM_MAX_ULPS, (
        f"gamma {_ulp_distance_fp32(k_gamma, t_gamma)} ulp > "
        f"{NORM_MAX_ULPS} vs the twin's torch.norm")
    # (c) the sketch: the tie budget (a flip needs a near-zero projection)
    mismatches = int((k_signs != t_signs).sum().item())
    slots = int(k_signs.numel())
    tie_fraction = (mismatches / slots) if slots else 0.0
    assert tie_fraction <= TIES_MAX_FRACTION, (
        f"the QJL sketch tie fraction {tie_fraction:.2e} > "
        f"{TIES_MAX_FRACTION} ({mismatches}/{slots} slots)")
    # (d) the compensated reconstruction
    assert torch.allclose(k_out, t_out, rtol=RECON_ALLCLOSE_RTOL,
                          atol=RECON_ALLCLOSE_ATOL), (
        "the QJL compensated reconstruction drifts beyond §4's 1e-5 vs "
        "the twin's _qjl_compensation_term")


# --- (b) the l2norm differential ------------------------------------------- #
@pytest.mark.parametrize("M,N,dtype,x_seed", L2_LADDER, ids=_L2_IDS)
def test_l2norm_vs_torch(M, N, dtype, x_seed):
    """tq_l2norm_rows vs torch's x.norm(dim=-1): <= 1 ulp (§4's G20.P1/P3
    norm threshold, view-int32 -0.0-safe — the only divergence is the
    sum's association, the 7.1 class; the __fsqrt_rn epilogue rounds
    identically to torch's final sqrt). The fp16 rung's reference is the
    WIDENED twin x.float().norm(dim=-1) — the kernel's fp16->fp32 cast
    is exact (the x32 discipline), so torch's association runs on the
    SAME widened values. Row 0 is zeroed deterministically before the
    call: the zero row lands EXACTLY 0.0 (branchless — the chain sums
    zeros and sqrt.rn(+0.0) = +0.0), the -0.0-safe ulp helper proves it
    is a true zero, and the rung doubles as the determinism pair (two
    calls, torch.equal — §4's level-1 contract)."""
    x = _seeded_rows(M, N, x_seed, dtype).contiguous()
    x[0].zero_()                      # the zero row (row 0, deterministic)
    norms = _C.tq_l2norm_rows(x)
    assert norms.shape == (M,) and norms.dtype == torch.float32
    assert norms.is_cuda and norms.device == x.device

    if dtype == torch.float32:
        ref = x.norm(dim=-1)
    else:
        # the widened twin: identical widened values, torch's association
        ref = x.to(torch.float32).norm(dim=-1)
    assert _ulp_distance_fp32(norms, ref) <= NORM_MAX_ULPS, (
        f"l2norm {_ulp_distance_fp32(norms, ref)} ulp > {NORM_MAX_ULPS} "
        f"vs torch's x.norm(dim=-1) at (M={M}, N={N}, {dtype})")
    # the zero row: exactly +0.0, not a -0.0 and not an epsilon
    assert norms[0].item() == 0.0, "the zero row must land exactly 0.0"
    # the determinism pair (§4 level-1: two calls, equal bits)
    assert torch.equal(norms, _C.tq_l2norm_rows(x)), (
        "tq_l2norm_rows run-to-run drift on the same input")


# --- (c) the QJL run-to-run determinism ------------------------------------ #
def test_qjl_run_to_run_determinism():
    """§4's level-1 hard contract on the QJL chain: two full kernel-chain
    passes on the same seeded batch — the base encode, tq_qjl_quant and
    tq_qjl_compensate each called twice — torch.equal on EVERY output
    (the packed code streams, the norms, the sketch, gamma and the
    compensated reconstruction; no atomics, fixed tile/warp/lane
    ownership, fixed-order folds — the 9.1 chain's contract)."""
    d, units, x_seed, frame_seed = QJL_RUNG
    q = _quantizer(d, frame_seed)
    x = _seeded_rows(units, d, x_seed, torch.float32).contiguous()
    dev = x.device
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)
    qjl_rot = q._qjl_signs_vec.detach().to(dev, torch.float32).contiguous()

    # the chain head: the batched encode, twice
    a_lo, a_hi, a_norms = _C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    b_lo, b_hi, b_norms = _C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    assert torch.equal(a_lo, b_lo), "batched idx_lo run-to-run drift"
    assert torch.equal(a_hi, b_hi), "batched idx_hi run-to-run drift"
    assert torch.equal(a_norms, b_norms), "batched norms run-to-run drift"

    # the sketch + gamma: tq_qjl_quant, twice
    s1, g1 = _C.tq_qjl_quant(
        x, signs, a_lo, a_hi, a_norms,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, qjl_rot)
    s2, g2 = _C.tq_qjl_quant(
        x, signs, a_lo, a_hi, a_norms,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, qjl_rot)
    assert torch.equal(s1, s2), "qjl_signs run-to-run drift"
    assert torch.equal(g1, g2), "gamma run-to-run drift"

    # the compensation: tq_qjl_compensate, twice on the same inputs
    mse = _C.tq_dequant_batched(
        a_lo, a_hi, a_norms, signs,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
    o1 = _C.tq_qjl_compensate(mse, s1, g1, qjl_rot)
    o2 = _C.tq_qjl_compensate(mse, s1, g1, qjl_rot)
    assert torch.equal(o1, o2), "the compensated output run-to-run drift"


# --- (d) the flag-off non-invasiveness ------------------------------------- #
def test_flag_off_non_invasiveness():
    """The dispatcher's default qjl-off path is UNCHANGED when the
    extension (now carrying the W20.9 qjl/l2norm entries) is present:
    the routing predicates still select the kernel path for a plain
    CUDA unit (quant AND dequant — the no-sketch row), quant without
    the flag produces codes WITHOUT the qjl fields (qjl_signs and
    gamma are None — TQCodes' defaults; the A/B is purely additive,
    turboquant.py:631-633), and the round trip stays finite. The
    default path takes zero new code."""
    d, x_seed, frame_seed = FLAG_RUNG
    q = _quantizer(d, frame_seed)          # qjl defaults OFF
    assert q.qjl is False, "the W9.2 flag must default OFF"
    x = _seeded_rows(1, d, x_seed, torch.float32)[0]

    # the routing pins: the extension present -> the kernel path for the
    # plain unit (the qjl entries changed nothing about the predicates)
    assert turboquant._tq_kernel_available(x), (
        "the TQ kernel pair must be importable on this box")
    assert q._quant_kernel_eligible(x), (
        "the dispatcher must route the qjl-off CUDA unit to the kernel")

    codes = q.quant(x)
    assert codes.qjl_signs is None, (
        "quant without the qjl flag must NOT attach qjl_signs")
    assert codes.gamma is None, (
        "quant without the qjl flag must NOT attach gamma")

    # the decode side: the no-sketch codes still route to the kernel
    # (the dispatcher's qjl routing is absent for flag-off codes)
    assert q._dequant_kernel_eligible(codes, torch.float32), (
        "the plain codes must still decode through the kernel path")
    recon = q.dequant(codes)
    assert recon.shape == (d,) and recon.is_cuda
    assert torch.isfinite(recon).all(), (
        "the default round trip must stay finite")


# --- the gate device ------------------------------------------------------- #
def test_device_report():
    """The guard's device_report() reports an Ampere device — capability
    major 8 (the A10G target is (8, 6) per PROPOSAL.md §4).
    device_report() itself raises outside the Ampere family, so this
    passing is the device-not-CPU proof for every test in this module."""
    rep = _gpu_guard.device_report()
    assert rep["capability"][0] == 8
