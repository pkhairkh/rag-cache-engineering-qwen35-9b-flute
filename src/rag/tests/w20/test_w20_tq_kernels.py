"""test_w20_tq_kernels.py — W20.8.3: the G20.P3 + G20.4 pytest surface (K3).

The CUDA-gated suite of the TQ kernel pair — flute_extended's
tq_quant_batched / tq_dequant_batched (W20.7.4 / W20.8.1) behind the
W20.8.2 kernel-first dispatcher — over TASKS.md W20.8.3's brief: parity
vs the torch twin (both on CUDA), run-to-run determinism (G20.4
level-1), the batched-grid shapes (120-192 units), the tie counter,
geometry coverage (2^19, 24,576, 2^22), and the refusal paths (bad K,
bad dtype). Thresholds mirror TASKS.md §4 1:1 (the registry in
scripts/gpu/w20_gates.py THRESHOLDS is the source of truth; this file
defines NO thresholds beyond §4's):

    G20.P3 (§4): "same as G20.P1 thresholds on kernel outputs;
        batched-grid shapes 120-192 units" — with G20.P1's row being
        "codes equal except ties <= 0.01%; norm <= 1 ulp; recon
        allclose <= 1e-5 rel; run-to-run torch.equal (CUDA)"
        -> TIES_MAX_FRACTION = 0.0001, NORM_MAX_ULPS = 1,
           RECON_ALLCLOSE_RTOL/ATOL = 1e-5, RUN_TO_RUN_TORCH_EQUAL,
           BATCHED_GRID_UNITS_MIN/MAX = 120/192
    G20.4 (§4): "level 1: run-to-run torch.equal (codes + recon,
        kernel); level 2: ties <= 0.01%, norm <= 1 ulp, allclose
        <= 1e-5" — level 1 is test_run_to_run_kernel_determinism +
        the batched run-to-run pairs; level 2 IS the G20.P3 parity
        differential below (the same counters, the same thresholds).

THE TWIN (documented choice): the torch leg of every differential here
is turboquant's OWN reference body, forced by monkeypatching the
dispatcher's ONE probe — turboquant._tq_kernel_available -> False (the
fht.py kernel_available signature discipline applied at the
TurboQuant level, W20.8.2). The SAME quantizer instance serves both
legs (same D3 rotation signs, same codebook mirrors, same frame seed);
only the routing flips, so the differential is exactly kernel-chain vs
torch-body with ZERO test-local math and zero convention drift. The
twin's rotation itself rides fht.py's auto backend (fht_large on CUDA
at these d's) — the repo's reference path, not a re-implementation.
Chosen over a device-forced twin (CPU tensors would leave the CUDA
differential) and over calling the underlying torch math directly
(_dequant_mse etc. would fork the dispatcher's own composition).

PROMPT.md §1: nothing here is RUN on a CPU box and nothing here is
CPU-runnable — importing w20._gpu_guard hard-skips this module loudly
on a CPU box (the guard's import-time require_cuda() raises
pytest.skip(..., allow_module_level=True)), and an extension built
without the TQ pair skips the module loudly too (the same
allow_module_level=True module skip: a skip, never a collection
error). Everything is deterministic: every input batch is rebuilt from
a seed recorded in this file; no unrecorded randomness anywhere.

THE GEOMETRY LADDER mirrors the gate's (scripts/gpu/w20_gates.py
G20_P3_LADDER): 2^19 and 2^22 are the kernel-eligible rungs (M stays
1-2 rows — the twin's host-side staging stays bounded, the
test_w20_fht_large.py convention); 24,576 — the conv full-window (two
<=16,384 FHT segments; non-power-of-two) — is the ROUTING rung: the
dispatcher serves it on the torch twin on CUDA (the W20.8.2 REV-3
table's ineligible row) and the direct binding refuses it loudly (the
belt). The batched-grid rungs (M in {120, 192}) drive the DIRECT
tq_quant_batched / tq_dequant_batched entries through the same
_device_signs / _device_codebooks mirrors the dispatcher feeds them —
the per-unit public API serves M=1 this wave; the batched surface is
the prefill-scale contract (PROPOSAL.md §3.6).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest
import torch

# Path-robust guard import. pytest's default prepend import mode resolves
# this module as w20.test_w20_tq_kernels with basedir src/rag/tests on
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
# (house convention, the same import test_turboquant.py uses); the lazy
# probe is the dispatcher's own (turboquant._tq_kernel_available ->
# fht._load_kernel + the TQ-name hasattr check, W20.8.2).
import turboquant  # noqa: E402
from turboquant import TQCodes, TurboQuant, _packed_len, unpack_bits_t  # noqa: E402

_C = turboquant._load_tq_kernel()
if (_C is None or not hasattr(_C, "tq_quant_batched")
        or not hasattr(_C, "tq_dequant_batched")):
    pytest.skip(
        "flute_extended not built with the TQ kernel pair — build on "
        "the A10G box (scripts/gpu/w20_provision.sh)",
        allow_module_level=True,
    )

# --- thresholds: TASKS.md §4, mirrored 1:1 from w20_gates.py THRESHOLDS -- #
# §4 G20.P3: "same as G20.P1 thresholds on kernel outputs; batched-grid
# shapes 120-192 units" (G20.P1: "codes equal except ties <= 0.01%; norm
# <= 1 ulp; recon allclose <= 1e-5 rel; run-to-run torch.equal (CUDA)").
TIES_MAX_FRACTION = 0.0001      # THRESHOLDS["G20.P3"]["ties_max_fraction"]
NORM_MAX_ULPS = 1               # THRESHOLDS["G20.P3"]["norm_max_ulps"]
RECON_ALLCLOSE_RTOL = 1e-5      # THRESHOLDS["G20.P3"]["recon_allclose_rtol"]
RECON_ALLCLOSE_ATOL = 1e-5      # THRESHOLDS["G20.P3"]["recon_allclose_atol"]
RUN_TO_RUN_TORCH_EQUAL = True   # §4 G20.P3/G20.4: the level-1 hard contract
BATCHED_GRID_UNITS_MIN = 120    # THRESHOLDS["G20.P3"]["batched_grid_units_min"]
BATCHED_GRID_UNITS_MAX = 192    # THRESHOLDS["G20.P3"]["batched_grid_units_max"]
# §4 G20.4 level 2's thresholds are the SAME three counters (ties
# <= 0.01%, norm <= 1 ulp, allclose <= 1e-5) — the parity tests below are
# that level; no thresholds beyond §4's live in this file.

# --- the geometry ladder (§4 G20.P3; the gate's G20_P3_LADDER twin) ----- #
# Every seed is recorded right here. M stays 1-2 rows (the twin's
# host-side staging stays bounded, the test_w20_fht_large.py convention).
KERNEL_LADDER = (
    # (d, units, x_seed, frame_seed)
    (2**19, 2, 819011, 819101),
    (2**22, 1, 822014, 822104),
)
_LADDER_IDS = [f"d2p{d.bit_length() - 1}-u{units}"
               for d, units, *_ in KERNEL_LADDER]

# The torch-only rung (the REV-3 routing table's ineligible row): d=24,576
# — the conv full-window (two <=16,384 FHT segments; non-power-of-two) —
# quantizes on the torch twin on CUDA; the kernel pair refuses it by
# design (the geometry envelope: d a power of two above one 64 KiB tile).
TORCH_ONLY_RUNG = (24576, 2, 807601, 807701)

# The batched-grid rungs (§4 G20.P3's units band, 120-192): the DIRECT
# batched entries at the S/M1/M2 geometry (d=2^19).
BATCHED_GRID_LADDER = (
    # (M, d, x_seed, frame_seed)
    (120, 2**19, 811201, 811301),
    (192, 2**19, 819204, 819304),
)
_BATCHED_IDS = [f"M{M}" for M, *_ in BATCHED_GRID_LADDER]


# --- the twin: the dispatcher's ONE probe, patched off -------------------- #
@contextlib.contextmanager
def _torch_twin():
    """Force the torch twin ON CUDA: monkeypatch the dispatcher's ONE
    probe (turboquant._tq_kernel_available) to return False for any
    argument, so quant/dequant take the torch reference body for CUDA
    inputs (the W20.8.2 REV-3 table's kernel-absent row — the
    performance fallback that is NEVER a correctness fallback). The
    quantizer instance, its rotation signs, its codebook mirrors and its
    frame seed are untouched: only the routing flips, which is exactly
    the kernel-vs-torch differential (restored on exit, exception-safe).
    """
    original = turboquant._tq_kernel_available
    turboquant._tq_kernel_available = lambda x=None: False
    try:
        yield
    finally:
        turboquant._tq_kernel_available = original


def _seeded_units(units: int, d: int, seed: int) -> torch.Tensor:
    """(units, d) fp32 CUDA rows from a recorded seed — drawn on the CPU
    generator (bit-exact across boxes) and moved to the device, the
    G20.P2 surface's own recipe."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(units, d, generator=gen,
                       dtype=torch.float32).to("cuda")


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


def _quantizer(d: int, frame_seed: int) -> TurboQuant:
    """The ladder's quantizer: bits 3.5 (the b3/b4 split, the production
    recipe), flat partition, custom kind with the recorded frame seed
    (the constructor's tests-only channel — the geometry is the ladder's
    variable, not the kind)."""
    return TurboQuant(kind="custom", bits=3.5, d=d, seed=frame_seed)


def _unpack(codes: TQCodes, side: str) -> torch.Tensor:
    """One packed stream unpacked with the torch decode (the canonical
    comparator — G20.P1's (a) discipline: codes unpacked on BOTH sides
    with torch, the byte layout pinned)."""
    bits = int(codes.bits_lo if side == "lo" else codes.bits_hi)
    n = int(codes.n_lo if side == "lo" else codes.n_hi)
    return unpack_bits_t(getattr(codes, "idx_" + side), bits,
                         n).to(torch.int64)


def _codes_from_batched_row(q: TurboQuant, codes_lo: torch.Tensor,
                            codes_hi: torch.Tensor,
                            norms: torch.Tensor, u: int) -> TQCodes:
    """The dispatcher's own per-unit packing of the batched outputs
    (_quant_kernel's reshape, mirrored): row u of the (M, L) packed
    streams as the 1-D idx streams, norms[u] as the 0-dim fp32 norm,
    the quantizer's own bookkeeping — the twin's dequant input."""
    return TQCodes(q.kind, q.d, norms[u].reshape(()),
                   q.bits_lo, q.bits_hi, q.n_lo, q.n_hi,
                   codes_lo[u], codes_hi[u], q.seed)


# --- (a)/(b)/(c): the geometry ladder, kernel vs torch twin -------------- #
@pytest.mark.parametrize("d,units,x_seed,frame_seed",
                         KERNEL_LADDER, ids=_LADDER_IDS)
def test_geometry_ladder_kernel_vs_torch_twin(d, units, x_seed,
                                              frame_seed):
    """One kernel-eligible rung through G20.P3's parity contract: a
    seeded (units, d) batch quantized BOTH ways on CUDA — the kernel
    path (the dispatcher: probe present + geometry eligible -> the
    kernel, ALWAYS) and the torch twin (the probe patched off) — with
    (a) codes equal except ties (unpack both sides with torch, the tie
    fraction <= 0.0001), (b) norm ulp distance <= 1 (view-int32,
    -0.0-safe), (c) recon allclose at §4's 1e-5/1e-5 on TWO legs: the
    end-to-end round trips (kernel-quant->kernel-dequant vs
    twin-quant->twin-dequant) and the isolated decode differential
    (the SAME kernel codes decoded by the kernel and by the twin)."""
    q = _quantizer(d, frame_seed)
    x = _seeded_units(units, d, x_seed)

    # the routing pin: rows 1-3 of the REV-3 table evaluated in order —
    # the dispatcher MUST take the kernel path for this rung
    assert turboquant._tq_kernel_available(x[0]), (
        f"d={d}: the TQ kernel pair is absent — the differential would "
        "degenerate to twin-vs-twin")
    assert q._quant_kernel_eligible(x[0]), (
        f"d={d}: the dispatcher would NOT route this rung to the kernel")

    mismatches = 0
    slots = 0
    norm_max_ulps = 0
    for u in range(units):
        c_kernel = q.quant(x[u])                # the kernel path
        r_kernel = q.dequant(c_kernel)          # the kernel decode
        with _torch_twin():
            c_twin = q.quant(x[u])              # the torch twin encode
            r_round_twin = q.dequant(c_twin)    # the twin round trip
            r_decode_twin = q.dequant(c_kernel)  # the twin decode

        # (a) codes: the tie counter (unpacked on both sides with torch)
        for side in ("lo", "hi"):
            ku, tu = _unpack(c_kernel, side), _unpack(c_twin, side)
            mismatches += int((ku != tu).sum().item())
            slots += int(ku.numel())
        # (b) norm: view-int32 ulps
        norm_max_ulps = max(norm_max_ulps, _ulp_distance_fp32(
            c_kernel.norm, c_twin.norm))
        # (c) recon: the round trip (the §4 cross-backend leg) and the
        # isolated decode differential (the kernel codes, both decoders)
        assert torch.allclose(r_kernel, r_round_twin,
                              rtol=RECON_ALLCLOSE_RTOL,
                              atol=RECON_ALLCLOSE_ATOL), (
            f"d={d} unit {u}: kernel round trip drifts vs the twin round "
            "trip beyond §4's 1e-5")
        assert torch.allclose(r_kernel, r_decode_twin,
                              rtol=RECON_ALLCLOSE_RTOL,
                              atol=RECON_ALLCLOSE_ATOL), (
            f"d={d} unit {u}: the kernel decode drifts vs the torch "
            "decode of the SAME codes beyond §4's 1e-5")

    tie_fraction = (mismatches / slots) if slots else 0.0
    assert tie_fraction <= TIES_MAX_FRACTION, (
        f"d={d}: tie fraction {tie_fraction:.2e} > {TIES_MAX_FRACTION} "
        f"({mismatches}/{slots} slots — boundary-exact flips, §5 "
        "level-2 accepted only under the budget)")
    assert norm_max_ulps <= NORM_MAX_ULPS, (
        f"d={d}: norm {norm_max_ulps} ulp > {NORM_MAX_ULPS}")


# --- the routing rung: 24,576 stays on the torch twin -------------------- #
def test_torch_only_geometry_routes_to_the_twin():
    """The REV-3 routing table's ineligible row, verified end to end:
    d=24,576 (the conv full-window — two <=16,384 FHT segments;
    non-power-of-two) is NOT kernel-eligible, so the dispatcher serves
    it on the torch twin on CUDA (quant + dequant, the W20.3 body
    verbatim), and the DIRECT binding refuses it loudly (the geometry
    envelope's own TORCH_CHECK — the belt behind the routing table) on
    BOTH orientations."""
    d, units, x_seed, frame_seed = TORCH_ONLY_RUNG
    q = _quantizer(d, frame_seed)
    x = _seeded_units(units, d, x_seed)

    # the routing decision: ineligible, by the dispatcher's own predicate
    assert not q._kernel_geometry_eligible(), (
        "d=24,576 must be kernel-INeligible (non-power-of-two)")
    assert not q._quant_kernel_eligible(x[0]), (
        "the dispatcher must route d=24,576 to the torch twin")

    # the twin serves it on CUDA (the reference body, device-parametric)
    codes = q.quant(x[0])
    recon = q.dequant(codes)
    assert isinstance(codes, TQCodes) and codes.d == d
    assert recon.shape == (d,) and recon.is_cuda

    # the direct binding refuses it loudly on both orientations (the
    # belt: the kernel's own geometry envelope, never silently reached)
    dev = x.device
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)
    with pytest.raises(RuntimeError, match="power of two"):
        _C.tq_quant_batched(x[:1].contiguous(), d, signs,
                            cb_lo.boundaries, cb_hi.boundaries,
                            q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    codes_lo = torch.zeros(1, _packed_len(q.n_lo, q.bits_lo),
                           dtype=torch.uint8, device=dev)
    codes_hi = torch.zeros(1, _packed_len(q.n_hi, q.bits_hi),
                           dtype=torch.uint8, device=dev)
    norms = torch.zeros(1, dtype=torch.float32, device=dev)
    with pytest.raises(RuntimeError, match="power of two"):
        _C.tq_dequant_batched(codes_lo, codes_hi, norms, signs,
                              cb_lo.centroids, cb_hi.centroids,
                              q.n_lo, q.bits_lo, q.n_hi, q.bits_hi,
                              torch.float32)


# --- the batched-grid shapes: the direct entries vs the twin ------------- #
@pytest.mark.parametrize("M,d,x_seed,frame_seed",
                         BATCHED_GRID_LADDER, ids=_BATCHED_IDS)
def test_batched_grid_shapes_kernel_vs_twin(M, d, x_seed, frame_seed):
    """One batched-grid rung (§4's 120-192 units band) through G20.P3's
    contract at the DIRECT batched surface: ONE tq_quant_batched call
    over the seeded (M, d) prefill batch vs the torch twin quantized
    per unit (the probe patched off); then ONE tq_dequant_batched call
    over the kernel's own batched codes vs the twin's per-unit decode
    of the SAME codes (the isolated decode differential at batched
    scale). The run-to-run pairs of both batched entries (torch.equal,
    codes AND recon) are G20.4's level-1 hard contract at this surface.
    """
    assert BATCHED_GRID_UNITS_MIN <= M <= BATCHED_GRID_UNITS_MAX, (
        f"M={M} is outside §4's batched-grid units band "
        f"[{BATCHED_GRID_UNITS_MIN}, {BATCHED_GRID_UNITS_MAX}]")

    q = _quantizer(d, frame_seed)
    x = _seeded_units(M, d, x_seed).contiguous()
    dev = x.device
    # the dispatcher's own mirrors (the same tensors _quant_kernel feeds
    # the binding — no test-local re-derivation of signs or codebooks)
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)

    codes_lo, codes_hi, norms = _C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    codes_lo2, codes_hi2, norms2 = _C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    assert torch.equal(codes_lo, codes_lo2), "batched idx_lo run-to-run drift"
    assert torch.equal(codes_hi, codes_hi2), "batched idx_hi run-to-run drift"
    assert torch.equal(norms, norms2), "batched norms run-to-run drift"

    out = _C.tq_dequant_batched(
        codes_lo, codes_hi, norms, signs,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
    out2 = _C.tq_dequant_batched(
        codes_lo, codes_hi, norms, signs,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
    assert torch.equal(out, out2), "batched recon run-to-run drift"

    # the twin, per unit: encode parity (the tie counter) + norm ulps
    mismatches = 0
    slots = 0
    norm_max_ulps = 0
    decode_ok = True
    with _torch_twin():
        for u in range(M):
            c_twin = q.quant(x[u])
            for side in ("lo", "hi"):
                bits = int(c_twin.bits_lo if side == "lo"
                           else c_twin.bits_hi)
                n = int(c_twin.n_lo if side == "lo" else c_twin.n_hi)
                ku = unpack_bits_t((codes_lo if side == "lo"
                                    else codes_hi)[u],
                                   bits, n).to(torch.int64)
                tu = unpack_bits_t(getattr(c_twin, "idx_" + side), bits,
                                   n).to(torch.int64)
                mismatches += int((ku != tu).sum().item())
                slots += int(ku.numel())
            norm_max_ulps = max(norm_max_ulps, _ulp_distance_fp32(
                norms[u].reshape(()), c_twin.norm))
            # the isolated decode differential: the kernel's own codes,
            # decoded by the twin (per-unit TQCodes, the dispatcher's
            # own packing of the batched outputs)
            codes_u = _codes_from_batched_row(q, codes_lo, codes_hi,
                                              norms, u)
            r_twin = q.dequant(codes_u)
            decode_ok &= bool(torch.allclose(
                out[u], r_twin, rtol=RECON_ALLCLOSE_RTOL,
                atol=RECON_ALLCLOSE_ATOL))

    tie_fraction = (mismatches / slots) if slots else 0.0
    assert tie_fraction <= TIES_MAX_FRACTION, (
        f"M={M}: tie fraction {tie_fraction:.2e} > {TIES_MAX_FRACTION} "
        f"({mismatches}/{slots} slots)")
    assert norm_max_ulps <= NORM_MAX_ULPS, (
        f"M={M}: norm {norm_max_ulps} ulp > {NORM_MAX_ULPS}")
    assert decode_ok, (
        f"M={M}: the batched kernel decode drifts vs the twin decode of "
        "the SAME codes beyond §4's 1e-5")


# --- G20.4 level-1: run-to-run determinism (the dispatcher form) --------- #
@pytest.mark.parametrize("d,units,x_seed,frame_seed",
                         KERNEL_LADDER, ids=_LADDER_IDS)
def test_run_to_run_kernel_determinism(d, units, x_seed, frame_seed):
    """G20.4 level-1 (§4: "run-to-run torch.equal (codes + recon,
    kernel)"), the dispatcher's per-unit form: two kernel-path quant
    calls and two kernel-path dequant calls on the same seeded unit —
    torch.equal on the packed code streams, the norm AND the
    reconstruction (the hard contract, PROPOSAL.md §5 level 1: no
    atomics, fixed tile/warp/lane ownership, fixed-order folds). The
    batched-entry form of the same contract rides
    test_batched_grid_shapes_kernel_vs_twin."""
    if not RUN_TO_RUN_TORCH_EQUAL:
        pytest.skip("§4 G20.4 level-1 disabled by threshold mirror")
    q = _quantizer(d, frame_seed)
    x = _seeded_units(units, d, x_seed)
    for u in range(units):
        c1, c2 = q.quant(x[u]), q.quant(x[u])
        r1, r2 = q.dequant(c1), q.dequant(c2)
        assert torch.equal(c1.idx_lo, c2.idx_lo), (
            f"d={d} unit {u}: idx_lo run-to-run drift")
        assert torch.equal(c1.idx_hi, c2.idx_hi), (
            f"d={d} unit {u}: idx_hi run-to-run drift")
        assert torch.equal(c1.norm, c2.norm), (
            f"d={d} unit {u}: norm run-to-run drift")
        assert torch.equal(r1, r2), (
            f"d={d} unit {u}: recon run-to-run drift")


# --- the refusal paths: bad K --------------------------------------------- #
@pytest.mark.parametrize("d,because,pattern", (
    (8192, "d <= 16,384 (one 64 KiB tile)", r"16384"),
    (24576, "non-power-of-two (the conv full-window)", r"power of two"),
), ids=["d8192", "d24576"])
def test_refusals_bad_k(d, because, pattern):
    """§4/W20.8.3's refusal paths, the K family: ineligible geometry
    through the DIRECT bindings raises on BOTH orientations — d <=
    16,384 (the small-K paths' own tile) and the non-power-of-two conv
    full-window — with the binding's own message text (the dispatcher
    never routes here: the REV-3 table sends both to the torch twin;
    the binding's TORCH_CHECK is the belt). The zero/one fill is
    deterministic — the checks fire before any launch."""
    dev = torch.device("cuda")
    n_lo = n_hi = d // 2
    boundaries = torch.ones(2**3 - 1, dtype=torch.float32, device=dev)
    centroids = torch.ones(2**3, dtype=torch.float32, device=dev)
    units = torch.zeros(1, d, dtype=torch.float32, device=dev)
    signs = torch.ones(d, dtype=torch.float32, device=dev)
    codes = torch.zeros(1, _packed_len(n_lo, 3), dtype=torch.uint8,
                        device=dev)
    norms = torch.zeros(1, dtype=torch.float32, device=dev)

    with pytest.raises(RuntimeError, match=pattern):
        _C.tq_quant_batched(units, d, signs, boundaries, boundaries,
                            n_lo, 3, n_hi, 3)
    with pytest.raises(RuntimeError, match=pattern):
        _C.tq_dequant_batched(codes, codes, norms, signs,
                              centroids, centroids,
                              n_lo, 3, n_hi, 3, torch.float32)


# --- the refusal paths: bad dtype ----------------------------------------- #
def test_refusals_bad_dtype():
    """§4/W20.8.3's refusal paths, the dtype family: fp64 units into
    tq_quant_batched (the encode's one input width — fp32, the TQCodes
    rev-1 contract) and a float64 decode request into
    tq_dequant_batched (the decode's two output widths — fp32/fp16,
    the single-cast store) both raise with the binding's own message
    text."""
    dev = torch.device("cuda")
    d, n_lo, n_hi = 2**19, 2**18, 2**18
    boundaries = torch.ones(2**3 - 1, dtype=torch.float32, device=dev)
    centroids = torch.ones(2**3, dtype=torch.float32, device=dev)
    signs = torch.ones(d, dtype=torch.float32, device=dev)

    units64 = torch.zeros(1, d, dtype=torch.float64, device=dev)
    with pytest.raises(RuntimeError, match="float32"):
        _C.tq_quant_batched(units64, d, signs, boundaries, boundaries,
                            n_lo, 3, n_hi, 3)

    L = _packed_len(n_lo, 3)
    codes = torch.zeros(1, L, dtype=torch.uint8, device=dev)
    norms = torch.zeros(1, dtype=torch.float32, device=dev)
    with pytest.raises(RuntimeError, match="float32 or float16"):
        _C.tq_dequant_batched(codes, codes, norms, signs,
                              centroids, centroids,
                              n_lo, 3, n_hi, 3, torch.float64)


# --- the gate device ------------------------------------------------------- #
def test_device_report():
    """The guard's device_report() reports an Ampere device — capability
    major 8 (the A10G target is (8, 6) per PROPOSAL.md §4).
    device_report() itself raises outside the Ampere family, so this
    passing is the device-not-CPU proof for every test in this module."""
    rep = _gpu_guard.device_report()
    assert rep["capability"][0] == 8
