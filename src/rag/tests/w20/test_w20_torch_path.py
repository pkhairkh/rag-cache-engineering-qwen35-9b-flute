"""test_w20_torch_path.py — W20.4.6.b: the G20.P1 pytest surface (K1).

The CUDA-gated, fixture-driven tests of the W20.3 torch path (the K1
single path: TurboQuant.quant/dequant device-parametric, TQCodes torch,
pack_bits_t/unpack_bits_t) replayed against the frozen-probe fixtures
(src/rag/tests/w20/fixtures/*.npz + fixtures_manifest.json, captured on
the A10G box per env/W20_GPU_PROTOCOL.md step 2). The comparison logic
and its thresholds mirror scripts/gpu/w20_gates.py gate G20.P1, which
mirrors TASKS.md §4 1:1 — the registry is the source of truth; this
file defines NO thresholds beyond §4's:

    ties_max_fraction    = 0.0001   (codes equal except ties <= 0.01%)
    norm_max_ulps        = 1        (norm <= 1 ulp, view-int32)
    recon_allclose_rtol = 1e-5, recon_allclose_atol = 1e-5

PROMPT.md §1: nothing here is RUN on a CPU box and nothing here is
CPU-runnable — importing w20._gpu_guard hard-skips this module loudly
on a CPU box (the guard's own import-time require_cuda() raises
pytest.skip(..., allow_module_level=True)), and the fixtures manifest
— absent until the A10G box captures it — skips the module loudly too
(the same allow_module_level=True module skip: a skip, never a
collection error). Everything is deterministic: the fixtures carry
the seeds and configs; no random seeds of our own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Path-robust guard import. pytest's default prepend import mode resolves
# this module as w20.test_w20_torch_path with basedir src/rag/tests on
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

# The torch path under test — conftest.py anchors src/rag on sys.path
# (house convention, the same import test_turboquant.py uses).
from turboquant import TurboQuant, unpack_bits_t  # noqa: E402

# TASKS.md §4 row G20.P1, mirrored 1:1 from scripts/gpu/w20_gates.py
# THRESHOLDS["G20.P1"] — the registry is the source of truth; no
# thresholds beyond §4's live in this file.
TIES_MAX_FRACTION = 0.0001    # codes equal except ties <= 0.01%
NORM_MAX_ULPS = 1             # norm <= 1 ulp (view-int32, -0.0-safe)
RECON_ALLCLOSE_RTOL = 1e-5
RECON_ALLCLOSE_ATOL = 1e-5

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_MANIFEST_PATH = _FIXTURES_DIR / "fixtures_manifest.json"

if not _MANIFEST_PATH.is_file():
    pytest.skip(
        "W20 fixtures not yet captured on the A10G box — run "
        "env/W20_GPU_PROTOCOL.md step 2",
        allow_module_level=True,
    )

# Collection-time manifest read — guarded by the module-level skip above:
# an absent manifest skips the module (never a collection error).
_MANIFEST = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
_PROBES = [dict(e) for e in _MANIFEST.get("probes", [])]
_ENTRY_BY_FILE = {str(e["file"]): e for e in _PROBES}


def _ulp_distance_fp32(a: torch.Tensor, b: torch.Tensor) -> int:
    """view(torch.int32) ulp distance, -0.0-safe — mirrors
    scripts/gpu/w20_gates.py::_ulp_distance_fp32 (the G20.P1 (b) check).
    Norms are non-negative, where the signed int32 bit view is monotonic:
    |ia - ib| IS the ulp distance; a -0.0 is normalized to +0.0 first so
    a degenerate zero norm never reads as 2^31 ulps."""
    a = a.detach().to("cpu", torch.float32).reshape(-1)
    b = b.detach().to("cpu", torch.float32).reshape(-1)
    a = torch.where(a == 0, torch.zeros_like(a), a)
    b = torch.where(b == 0, torch.zeros_like(b), b)
    if a.numel() == 0 or b.numel() == 0:
        return 0
    ia = a.contiguous().view(torch.int32).to(torch.int64)
    ib = b.contiguous().view(torch.int32).to(torch.int64)
    return int((ia - ib).abs().max().item())


def _rebuild_probe(entry: dict):
    """(x, q, fx): the probe's input rebuilt from the recorded seed (the
    capture's own torch.Generator recipe), the quantizer per the manifest
    config, and the loaded npz — mirrors w20_gates.py::_probe_parity's
    rebuild/construction half."""
    fx = np.load(str(_FIXTURES_DIR / str(entry["file"])),
                 allow_pickle=False)
    units, d = int(fx["units"]), int(fx["d"])
    gen = torch.Generator(device="cpu")       # the capture's own recipe
    gen.manual_seed(int(entry["seed"]))
    x = torch.randn(units, d, generator=gen, dtype=torch.float32)
    cfg = entry["config"]
    q = TurboQuant(kind=str(cfg["tq_kind"]), bits=float(cfg["b"]),
                   d=int(cfg["d"]), seed=int(cfg["frame_seed"]),
                   qjl=bool(cfg["qjl"]), group=int(cfg["chan"]))
    return x, q, fx


# --- the manifest itself ----------------------------------------------------- #
def test_fixture_manifest_loads():
    """The manifest parses, carries >= 1 probe, and every fixture file
    exists on the box (path check only — the values are exercised by the
    parametrized parity test below)."""
    assert isinstance(_MANIFEST, dict)
    assert len(_PROBES) >= 1
    for entry in _PROBES:
        path = _FIXTURES_DIR / str(entry["file"])
        assert path.is_file(), f"fixture absent: {path}"


# --- (a)/(b)/(c): the torch path vs the frozen fixtures ---------------------- #
@pytest.mark.parametrize("file_name", [str(e["file"]) for e in _PROBES])
def test_torch_path_matches_fixtures(file_name):
    """One frozen probe through G20.P1's checks (a)-(c): rebuild the input
    from the recorded seed/config, run the CURRENT torch-CUDA path
    (TurboQuant per the manifest config), unpack BOTH code streams, and
    compare against the frozen capture — (a) tie fraction <= 0.0001,
    (b) norm ulp distance <= 1 (view-int32, -0.0-safe), (c) recon
    allclose at §4's 1e-5/1e-5."""
    entry = _ENTRY_BY_FILE[file_name]
    x, q, fx = _rebuild_probe(entry)
    units = int(fx["units"])
    dev = torch.device("cuda")

    mismatches = 0
    slots = 0
    norm_max_ulps = 0
    for u in range(units):
        xu = x[u].to(dev)                      # the K1 torch-CUDA path
        codes = q.quant(xu)
        recon = q.dequant(codes)

        # (a) codes: unpack BOTH streams with the current torch decode
        # and count the tie mismatches vs the frozen unpacked twins
        for side, packed in (("lo", codes.idx_lo), ("hi", codes.idx_hi)):
            bits = int(codes.bits_lo if side == "lo" else codes.bits_hi)
            n = int(codes.n_lo if side == "lo" else codes.n_hi)
            cur = unpack_bits_t(packed, bits, n).to(torch.int64).cpu()
            fix = torch.from_numpy(np.ascontiguousarray(
                fx[f"idx_{side}_unpacked"][u])).to(torch.int64)
            mismatches += int((cur != fix).sum().item())
            slots += cur.numel()

        # (b) norm (and the outlier split's norm_hi): view-int32 ulps
        norm_max_ulps = max(norm_max_ulps, _ulp_distance_fp32(
            codes.norm, torch.from_numpy(fx["norm"][u:u + 1])))
        if codes.norm_hi is not None:
            norm_max_ulps = max(norm_max_ulps, _ulp_distance_fp32(
                codes.norm_hi, torch.from_numpy(fx["norm_hi"][u:u + 1])))

        # (c) reconstruction
        assert torch.allclose(
            recon.to("cpu", torch.float32),
            torch.from_numpy(np.ascontiguousarray(fx["recon"][u])),
            rtol=RECON_ALLCLOSE_RTOL, atol=RECON_ALLCLOSE_ATOL), (
            f"{file_name} unit {u}: recon drift vs the frozen fixture")

    tie_fraction = (mismatches / slots) if slots else 0.0
    assert tie_fraction <= TIES_MAX_FRACTION, (
        f"{file_name}: tie fraction {tie_fraction:.2e} > "
        f"{TIES_MAX_FRACTION} ({mismatches}/{slots} slots)")
    assert norm_max_ulps <= NORM_MAX_ULPS, (
        f"{file_name}: norm {norm_max_ulps} ulp > {NORM_MAX_ULPS}")


# --- (d): run-to-run determinism --------------------------------------------- #
def test_run_to_run_determinism():
    """G20.P1's (d) on one probe: re-quantize the probe's units twice and
    torch.equal the packed codes AND the reconstructions (the CUDA
    run-to-run check — the same code stream and the same recon from the
    same input, twice)."""
    entry = _PROBES[0]
    x, q, fx = _rebuild_probe(entry)
    dev = torch.device("cuda")
    for u in range(int(fx["units"])):
        xu = x[u].to(dev)
        c1, c2 = q.quant(xu), q.quant(xu)      # the run-to-run pair
        r1, r2 = q.dequant(c1), q.dequant(c2)
        assert torch.equal(c1.idx_lo, c2.idx_lo), f"unit {u}: idx_lo drift"
        assert torch.equal(c1.idx_hi, c2.idx_hi), f"unit {u}: idx_hi drift"
        assert torch.equal(c1.norm, c2.norm), f"unit {u}: norm drift"
        assert c1.mask is None or torch.equal(c1.mask, c2.mask)
        assert torch.equal(r1, r2), f"unit {u}: recon drift"


# --- the gate device --------------------------------------------------------- #
def test_device_report():
    """The guard's device_report() reports an Ampere device — capability
    major 8 (the A10G target is (8, 6) per PROPOSAL.md §4). device_report()
    itself raises outside the Ampere family, so this passing is the
    device-not-CPU proof for every test in this module."""
    rep = _gpu_guard.device_report()
    assert rep["capability"][0] == 8
