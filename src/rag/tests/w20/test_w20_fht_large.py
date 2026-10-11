"""test_w20_fht_large.py — W20.6.3: the G20.P2 + G20.6 pytest surface (K2).

The CUDA-gated suite of the fht_large family — W20.5's
fht_forward_large / fht_backward_large and W20.6.1's test-only scalar
twins tq_debug_fht_large_forward / tq_debug_fht_large_backward — over
TASKS.md W20.6.3's brief: the geometry ladder (2^19, 2^20, 2^22; the
ladder is about K, M stays 1-2 rows), the run-to-run determinism hard
contract, the K-eligibility refusals, and the W20.6.2 canary/trap
negative controls. Thresholds mirror TASKS.md §4 1:1 (the registry in
scripts/gpu/w20_gates.py THRESHOLDS is the source of truth; this file
defines NO thresholds beyond §4's):

    G20.P2 (§4): "kernel vs debug twin vs torch allclose <= 1e-5;
        run-to-run torch.equal; K%32!=0 and K<=16,384 refusals raise"
        -> ALLCLOSE_RTOL = ALLCLOSE_ATOL = 1e-5,
           RUN_TO_RUN_TORCH_EQUAL, REFUSAL_K_MOD32, REFUSAL_K_MAX
    G20.6 (§4): "canary build: zero poison survives; trap build: FAILS
        the determinism gate (the negative control must fail)"
        -> CANARY_ZERO_POISON, TRAP_FAILS_DETERMINISM_GATE

PROMPT.md §1: nothing here is RUN on a CPU box and nothing here is
CPU-runnable — importing w20._gpu_guard hard-skips this module loudly
on a CPU box (the guard's import-time require_cuda() raises
pytest.skip(..., allow_module_level=True)), and an extension built
without the large kernels skips the module loudly too (the same
allow_module_level=True module skip: a skip, never a collection
error). Everything is deterministic: every input row and sign vector
is rebuilt from a seed recorded in this file; no unrecorded
randomness anywhere.

THE REFERENCE (documented choice): the torch leg of G20.P2's parity is
fht.py's own fht_reference / fht_reference_adjoint — the repo's torch
butterfly, fp32-accumulated, valid on CUDA, with the pairing order the
kernel contract pins ("reference and kernel agree to fp32 rounding
order"). Chosen over recomputing the FWHT inline in this file: one
canonical ground truth instead of a test-local twin of a twin, and
zero convention drift (the segments/signs recipes live beside it in
fht.py, and the production wrapper itself dispatches through them).

THE VARIANT LOADS (canary/trap): per scripts/gpu/w20_build_variants.sh's
artifact contract, a variant .so is path-loaded as
ExtensionFileLoader("flute_extended._C",
<repo>/src/flute_extended/flute_extended/_C_{canary,trap}.so) in a
process that does NOT also import the production _C (the module name
would collide — this module has already imported it at collection
time). The canary/trap tests therefore run their probe in a fresh
subprocess; a variant .so that is absent (or not importable) skips
loudly with the build pointer.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

# Path-robust guard import. pytest's default prepend import mode resolves
# this module as w20.test_w20_fht_large with basedir src/rag/tests on
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

# The torch reference + the built extension — conftest.py's _paths
# anchors src/flute_extended on sys.path (house convention, the same
# import test_turboquant.py uses); the extension import rides fht.py's
# own lazy loader _load_kernel (the canonical path-robust dance,
# stale-namespace-package removal included). _C below is the
# flute_extended._C binding module.
import fht  # noqa: E402

_C = fht._load_kernel()
if (_C is None or not hasattr(_C, "fht_forward_large")
        or not hasattr(_C, "fht_backward_large")
        or not hasattr(_C, "tq_debug_fht_large_forward")
        or not hasattr(_C, "tq_debug_fht_large_backward")):
    pytest.skip(
        "flute_extended not built with the large kernels — build on "
        "the A10G box (scripts/gpu/w20_provision.sh)",
        allow_module_level=True,
    )

# --- thresholds: TASKS.md §4, mirrored 1:1 from w20_gates.py THRESHOLDS -- #
# §4 G20.P2: "kernel vs debug twin vs torch allclose <= 1e-5; run-to-run
# torch.equal; K%32!=0 and K<=16,384 refusals raise".
ALLCLOSE_RTOL = 1e-5              # THRESHOLDS["G20.P2"]["allclose_rtol"]
ALLCLOSE_ATOL = 1e-5              # THRESHOLDS["G20.P2"]["allclose_atol"]
RUN_TO_RUN_TORCH_EQUAL = True     # §4 G20.P2: the level-1 hard contract
REFUSAL_K_MOD32 = True            # §4 G20.P2: K % 32 != 0 must raise
REFUSAL_K_MAX = 16384             # §4 G20.P2: K <= 16_384 must raise
# §4 G20.6: "canary build: zero poison survives; trap build: FAILS the
# determinism gate (the negative control must fail)".
CANARY_ZERO_POISON = True         # THRESHOLDS["G20.6"]["canary_zero_poison"]
TRAP_FAILS_DETERMINISM_GATE = True  # THRESHOLDS["G20.6"]["trap_fails_determinism_gate"]

# --- the geometry ladder (§4 G20.P2; W20.6.3's K rungs) ------------------ #
# The ladder is about K — M stays 1-2 rows (2^22 rides M=1 so the twin's
# host-side staging stays bounded). Every seed is recorded right here.
GEOMETRY_LADDER = (
    # (K, M, x_seed, g_seed, sign_seed)
    (2**19, 2, 619011, 619021, 619031),
    (2**20, 2, 620012, 620022, 620032),
    (2**22, 1, 622014, 622024, 622034),
)
_LADDER_IDS = [f"K2p{K.bit_length() - 1}-M{M}"
               for K, M, *_ in GEOMETRY_LADDER]


def _seeded_rows(M: int, K: int, seed: int) -> torch.Tensor:
    """(M, K) fp32 CUDA rows from a recorded seed — drawn on the CPU
    generator (bit-exact across boxes) and moved to the device,
    contiguous as the binding's argument checks require."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(M, K, generator=gen,
                       dtype=torch.float32).to("cuda")


def _seeded_signs(K: int, seed: int) -> torch.Tensor:
    """(K,) fp32 ±1 CUDA signs — fht.rotation_signs, the production ±1
    recipe (generator seeded seed + 4242), moved to the device."""
    return fht.rotation_signs(K, seed).to("cuda")


# --- (a) the geometry ladder: kernel vs twin vs torch reference --------- #
@pytest.mark.parametrize("K,M,x_seed,g_seed,sign_seed",
                         GEOMETRY_LADDER, ids=_LADDER_IDS)
def test_geometry_ladder_parity(K, M, x_seed, g_seed, sign_seed):
    """One ladder rung through G20.P2's parity contract: seeded fp32 rows
    and ±1 signs through (i) the kernel, (ii) its test-only scalar twin,
    and (iii) fht.py's torch reference — all on CUDA, forward AND
    adjoint — with allclose at §4's 1e-5/1e-5 on both legs of each."""
    x = _seeded_rows(M, K, x_seed)
    g = _seeded_rows(M, K, g_seed)
    signs = _seeded_signs(K, sign_seed)

    # forward: kernel vs debug twin, then kernel vs torch reference
    out_kernel = _C.fht_forward_large(x, signs).cpu()
    out_twin = _C.tq_debug_fht_large_forward(x, signs).cpu()
    out_ref = fht.fht_reference(x, signs).cpu()
    assert torch.allclose(out_kernel, out_twin,
                          rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL), (
        f"K={K} forward: kernel vs debug twin drift")
    assert torch.allclose(out_kernel, out_ref,
                          rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL), (
        f"K={K} forward: kernel vs torch reference drift")

    # adjoint: the same two legs through fht_backward_large
    adj_kernel = _C.fht_backward_large(g, signs).cpu()
    adj_twin = _C.tq_debug_fht_large_backward(g, signs).cpu()
    adj_ref = fht.fht_reference_adjoint(g, signs).cpu()
    assert torch.allclose(adj_kernel, adj_twin,
                          rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL), (
        f"K={K} adjoint: kernel vs debug twin drift")
    assert torch.allclose(adj_kernel, adj_ref,
                          rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL), (
        f"K={K} adjoint: kernel vs torch reference drift")


# --- (b) run-to-run determinism (the level-1 hard contract) ------------- #
def test_run_to_run_determinism():
    """G20.P2's run-to-run line: forward+backward re-run twice on one
    geometry (the ladder's first rung, K = 2**19) — torch.equal on BOTH
    outputs: the same code stream from the same input, twice."""
    K, M, x_seed, g_seed, sign_seed = GEOMETRY_LADDER[0]
    x = _seeded_rows(M, K, x_seed)
    g = _seeded_rows(M, K, g_seed)
    signs = _seeded_signs(K, sign_seed)

    f1 = _C.fht_forward_large(x, signs)
    f2 = _C.fht_forward_large(x, signs)
    b1 = _C.fht_backward_large(g, signs)
    b2 = _C.fht_backward_large(g, signs)
    assert RUN_TO_RUN_TORCH_EQUAL and torch.equal(f1, f2), (
        f"K={K}: forward run-to-run drift")
    assert torch.equal(b1, b2), f"K={K}: backward run-to-run drift"


# --- (c) the K-eligibility refusals (§4 G20.P2: "refusals raise") ------- #
INELIGIBLE_GEOMETRIES = (
    # (K, why ineligible, regex on the TORCH_CHECK eligibility text)
    (2**19 + 33, "K % 32 != 0", r"divisible by 32"),
    (49152, "non-power-of-two", r"power of two"),
    (8192, "K <= 16,384", "<= " + str(REFUSAL_K_MAX)),
)


@pytest.mark.parametrize("K,because,match", INELIGIBLE_GEOMETRIES,
                         ids=["Kmod32", "nonpow2", "KtooSmall"])
def test_ineligible_K_refusals(K, because, match):
    """Ineligible K through the DIRECT binding (not fht.py's wrapper —
    its auto mode silently falls back to the reference; §4's refusals
    are the binding's own TORCH_CHECKs): K % 32 != 0, non-power-of-two,
    and K <= REFUSAL_K_MAX each raise RuntimeError carrying the
    eligibility text, on BOTH orientations (the one shared
    fht_large_check_args gate serves the whole family). The zero/one
    fill is deterministic — the checks fire before any launch."""
    assert ((REFUSAL_K_MOD32 and K % 32 != 0) or K <= REFUSAL_K_MAX
            or (K & (K - 1)) != 0), (
        f"K={K} ({because}) is eligible — bad parametrize entry")
    x = torch.zeros(1, K, dtype=torch.float32, device="cuda")
    signs = torch.ones(K, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match=match):
        _C.fht_forward_large(x, signs)
    with pytest.raises(RuntimeError, match=match):
        _C.fht_backward_large(x, signs)


# --- (d) the canary/trap negative controls (§4 G20.6) ------------------- #
# The variant .so contract (scripts/gpu/w20_build_variants.sh): the
# canary/trap builds land at src/flute_extended/flute_extended/
# _C_canary.so / _C_trap.so and are path-loaded as ExtensionFileLoader(
# "flute_extended._C", <so>) — in a process that does NOT also import
# the production _C. This module already imported the production _C at
# collection time, so the load CANNOT happen in-process here: the probe
# below runs in a fresh subprocess of the same interpreter.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_VARIANT_SO = {
    mode: _REPO_ROOT / "src" / "flute_extended" / "flute_extended"
    / f"_C_{mode}.so"
    for mode in ("canary", "trap")
}


def _variant_so_or_skip(mode: str) -> Path:
    """The variant .so path, or a loud skip with the build pointer."""
    so = _VARIANT_SO[mode]
    if not so.is_file():
        pytest.skip(
            f"_C_{mode}.so not built — build the variant on the A10G "
            f"box (scripts/gpu/w20_build_variants.sh {mode})")
    return so


@pytest.fixture()
def canary_so() -> Path:
    """The canary variant .so (skip-unless-built, loud with the build
    pointer). The ExtensionFileLoader try itself is the probe's first
    act — it cannot run in-process here (the production _C is already
    imported; W20.6.2's collision note), so 'importable' is decided in
    the clean subprocess below."""
    return _variant_so_or_skip("canary")


@pytest.fixture()
def trap_so() -> Path:
    """The trap variant .so (skip-unless-built, loud with the build
    pointer); the same load story as canary_so."""
    return _variant_so_or_skip("trap")


# The hermetic variant probe: a fresh interpreter that imports torch,
# path-loads the variant .so per the build script's contract, and runs
# one mode-specific check on one recorded geometry. Exit codes:
#   0 = the control's verdict is clean (canary: zero non-finite values;
#       trap: two runs differ), 1 = the control FAILED (poison survived
#       / the determinism gate passed with the trap armed), 3 = the .so
#       is present but not importable (the caller skips loudly), 2 = a
#       bad mode argument (a caller bug, never shipped).
_VARIANT_PROBE = r'''
import importlib.machinery
import importlib.util
import sys

import torch

mode, so_path = sys.argv[1], sys.argv[2]
K, M, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

# (1) the variant load per the build script's contract: the .so bakes
#     production's init symbol, so the module name must be
#     "flute_extended._C"; this process imports nothing else of the
#     package (no production _C in this process -- no name collision).
loader = importlib.machinery.ExtensionFileLoader(
    "flute_extended._C", so_path)
spec = importlib.util.spec_from_loader("flute_extended._C", loader)
variant = importlib.util.module_from_spec(spec)
try:
    loader.exec_module(variant)
except Exception as exc:
    print("VARIANT_IMPORT_FAILED:", exc, file=sys.stderr)
    raise SystemExit(3)
if not hasattr(variant, "fht_forward_large"):
    print("VARIANT_IMPORT_FAILED: no fht_forward_large binding",
          file=sys.stderr)
    raise SystemExit(3)

# (2) the deterministic geometry: a recorded-seed fp32 input plus a
#     seeded +-1 sign vector (fht.rotation_signs's randint recipe,
#     inline so the probe stays hermetic -- no repo imports).
gen = torch.Generator()
gen.manual_seed(seed)
x = torch.randn(M, K, generator=gen, dtype=torch.float32).to("cuda")
signs = (torch.randint(0, 2, (K,), generator=gen)
         .to("cuda").float().mul(2).sub(1))

f = variant.fht_forward_large
if mode == "canary":
    # zero poison: the NaN-pre-poisoned output must be FULLY overwritten
    # by the real kernel (the write-coverage proof).
    out = f(x, signs)
    torch.cuda.synchronize()
    nonfinite = int((~torch.isfinite(out)).sum().item())
    print(f"CANARY_NONFINITE {nonfinite}")
    raise SystemExit(0 if nonfinite == 0 else 1)
if mode == "trap":
    # the determinism gate MUST fail under the trap: same input, two
    # runs, torch.equal must be False (the negative control).
    o1 = f(x, signs)
    o2 = f(x, signs)
    torch.cuda.synchronize()
    identical = bool(torch.equal(o1, o2))
    print(f"TRAP_RUN_TO_RUN_IDENTICAL {identical}")
    raise SystemExit(1 if identical else 0)
print(f"VARIANT_MODE_UNKNOWN {mode}", file=sys.stderr)
raise SystemExit(2)
'''


def _run_variant_probe(mode: str, so: Path) -> subprocess.CompletedProcess:
    """Run the hermetic variant probe on one geometry — the ladder's
    first rung (2^19, M=2); its recorded seed rides GEOMETRY_LADDER."""
    K, M, x_seed, _, _ = GEOMETRY_LADDER[0]
    return subprocess.run(
        [sys.executable, "-c", _VARIANT_PROBE,
         mode, str(so), str(K), str(M), str(x_seed)],
        capture_output=True, text=True, timeout=600,
    )


def test_canary_zero_poison(canary_so):
    """G20.6's canary line: the canary build NaN-poisons every output
    byte BEFORE the kernels write — a surviving NaN is a write-coverage
    regression. Forward on one geometry through the variant module's
    fht_forward_large; assert ZERO non-finite values in the output (the
    poison fully overwritten)."""
    probe = _run_variant_probe("canary", canary_so)
    if probe.returncode == 3:  # present but not importable: loud skip
        pytest.skip(f"_C_canary.so not importable in a clean process: "
                    f"{probe.stderr.strip()}")
    assert CANARY_ZERO_POISON and probe.returncode == 0, (
        f"canary poison survived the forward (§4 G20.6 "
        f"canary_zero_poison): rc={probe.returncode} "
        f"stdout={probe.stdout.strip()} stderr={probe.stderr.strip()}")


def test_trap_must_fail(trap_so):
    """G20.6's trap line (the negative control): the trap build flips
    the pair arithmetic on a clock64 timing bit, so the run-to-run
    determinism gate MUST fail under it. Forward twice on one geometry
    through the variant module's fht_forward_large; assert NOT
    torch.equal — a determinism PASS with the trap armed is a broken
    assert (the gate is not actually testing anything)."""
    probe = _run_variant_probe("trap", trap_so)
    if probe.returncode == 3:
        pytest.skip(f"_C_trap.so not importable in a clean process: "
                    f"{probe.stderr.strip()}")
    assert TRAP_FAILS_DETERMINISM_GATE and probe.returncode == 0, (
        f"trap build did NOT break run-to-run determinism (§4 G20.6 "
        f"trap_fails_determinism_gate): rc={probe.returncode} "
        f"stdout={probe.stdout.strip()} stderr={probe.stderr.strip()}")
