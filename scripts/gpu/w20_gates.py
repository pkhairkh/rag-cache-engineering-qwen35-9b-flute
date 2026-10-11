#!/usr/bin/env python3
"""w20_gates.py -- W20.4.6.a: the G20 gate harness (the A10G vehicle).

The execution vehicle for the W20 GPU-box gate registry (TASKS.md §4,
mirrored verbatim into env/W20_GPU_PROTOCOL.md): runs gates by id on the
CUDA device, prints one verdict line per gate ("G20.<id>: PASS/FAIL
(metrics summary)") + a final summary, and writes the per-gate JSON to
env/w20_gate_reports/<id>.json (gitignored BY DESIGN -- box-written,
never committed; the operator transcribes the statuses into TASKS.md §4).
CLI: --gates CSV|all (default all: AUTHORED gates run, stubs are listed,
not invoked); --report DIR (default env/w20_gate_reports); --list (the
registry, exit 0). Exit codes: 0 all executed gates PASS; 1 some FAIL;
2 a NOT-AUTHORED gate was invoked (a loud refusal, never a silent skip)
or a usage error; 3 no CUDA device.

THRESHOLDS are the source of truth, mirrored 1:1 from TASKS.md §4 —
drift is a wave-close failure. The registry table and this script are
authoritative over any transcription.

DISCIPLINE (PROMPT.md §1): the src/rag/tests/w20/_gpu_guard.py guard
discipline is replicated INLINE (importing the guard from scripts/gpu is
awkward -- its import-time require_cuda() pytest-skips). CUDA assert
FIRST: torch.cuda.is_available() or exit 3 -- BEFORE any repo import;
then the device report (warn off the A10G (8,6) target, note off the
Ampere family), then the repo imports via the house src/rag/_paths.py
convention. This script NEVER runs on the CPU dev box (authored there
under PROMPT.md §1: py_compile + rg only); it executes on the A10G box
per env/W20_GPU_PROTOCOL.md steps 4-5.

AUTHORED: G20.P1 (W20.4: the K1 gate -- the torch-CUDA single path
vs the frozen fixtures + the 237-suite regression); G20.P2 + G20.6
(W20.6: the K2 gate -- the fht_large differential twins + determinism
over the geometry ladder, and the canary/trap negative controls via
the variant probe's ExtensionFileLoader-in-a-fresh-subprocess
discipline, where an ABSENT variant .so FAILS the gate with the
build-script pointer -- a gate is never skippable); G20.P3 + G20.4
+ G20.1 + G20.2 (W20.8: the K3 gate -- the TQ kernel pair's parity
vs the torch twin over the geometry ladder and the batched-grid
shapes (120-192 units), the parity contract's levels 1-2, and the
PHASE-1 performance timers read from a finetune_m1m2.py short-run
history -- full confirmation after K4/K5 per TASKS.md §4). The
remaining 2 gates (G20.5, G20.3) are NOT-AUTHORED stubs -- their
waves (W20.10, W20.11) extend GATE_RUNNERS; invoking a stub prints
"G20.<id>: NOT-AUTHORED-YET (authored in <wave> per TASKS.md §4)"
and the harness exits 2.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch  # the ONLY import before the CUDA assert (no repo module)

# ---------------------------------------------------------------- paths ---
# repo root = scripts/gpu/w20_gates.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = REPO_ROOT / "src" / "rag"
FIXTURES_DIR = RAG_DIR / "tests" / "w20" / "fixtures"
MANIFEST_NAME = "fixtures_manifest.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "env" / "w20_gate_reports"
SUITE_TARGET = "src/rag/tests"           # the 237-suite regression target
SUITE_TIMEOUT_S = 3600.0
_A10G_CAPABILITY = (8, 6)                # PROPOSAL.md §4: the A10G target
_AMPERE_MAJOR = 8

EXIT_OK, EXIT_FAIL, EXIT_REFUSAL, EXIT_NO_CUDA = 0, 1, 2, 3

# ----------------------------------------------------------- THRESHOLDS ---
# Mirrored 1:1 from the TASKS.md §4 registry table (source of truth --
# drift is a wave-close failure). One entry per gate row; each comment
# cites the row's threshold text verbatim.
THRESHOLDS: Dict[str, Dict[str, Any]] = {
    # §4 G20.P1: "codes equal except ties ≤ 0.01%; norm ≤ 1 ulp; recon
    # allclose ≤ 1e-5 rel; run-to-run torch.equal (CUDA); 237-suite 0
    # failures"  (ties ≤ 0.01% == fraction ≤ 0.0001).
    "G20.P1": {
        "ties_max_fraction": 0.0001, "norm_max_ulps": 1,
        "recon_allclose_rtol": 1e-5, "recon_allclose_atol": 1e-5,
        "run_to_run_torch_equal": True, "suite_max_failures": 0,
    },
    # §4 G20.P2: "kernel vs debug twin vs torch allclose ≤ 1e-5;
    # run-to-run torch.equal; K%32≠0 and K≤16,384 refusals raise".
    "G20.P2": {
        "allclose_rtol": 1e-5, "allclose_atol": 1e-5,
        "run_to_run_torch_equal": True,
        "refusal_k_mod32": True,    # K % 32 != 0 must raise
        "refusal_k_max": 16384,     # K <= 16_384 must raise
    },
    # §4 G20.P3: "same as G20.P1 thresholds on kernel outputs;
    # batched-grid shapes 120–192 units".
    "G20.P3": {
        "ties_max_fraction": 0.0001, "norm_max_ulps": 1,
        "recon_allclose_rtol": 1e-5, "recon_allclose_atol": 1e-5,
        "run_to_run_torch_equal": True,
        "batched_grid_units_min": 120, "batched_grid_units_max": 192,
    },
    # §4 G20.1: "capture ≤ 10 s/cycle after K3; TQ ≤ 5 ms/prefill
    # (phase timers)".
    "G20.1": {
        "capture_cycle_max_s": 10.0, "tq_prefill_max_ms": 5.0,
    },
    # §4 G20.2: "step ≤ 2 s after K3; ≤ 1 s at refresh 8; 300 steps
    # ≤ 10 min".
    "G20.2": {
        "step_max_s": 2.0, "step_max_s_refresh8": 1.0,
        "run_300_steps_max_min": 10.0,
    },
    # §4 G20.3: "loss leaves ln(8) at step 0; gk/gv_absmax > 0; held-out
    # spread positive" (the W19 convergence lines).
    "G20.3": {
        "loss_leaves_ln8_step0": True, "gk_gv_absmax_positive": True,
        "heldout_spread_positive": True,
    },
    # §4 G20.4: "level 1: run-to-run torch.equal (codes + recon, kernel);
    # level 2: ties ≤ 0.01%, norm ≤ 1 ulp, allclose ≤ 1e-5".
    "G20.4": {
        "level1_run_to_run_torch_equal": True,
        "level2_ties_max_fraction": 0.0001, "level2_norm_max_ulps": 1,
        "level2_allclose_rtol": 1e-5, "level2_allclose_atol": 1e-5,
    },
    # §4 G20.5: "TRAIN → RE-INGEST → INDEX → EVAL → QUERY on the kernel
    # path; npz digests stable run-to-run; cross-backend decode allclose
    # ≤ 1e-5; W16 verbatim install bit-exact within backend".
    "G20.5": {
        "protocol_sequence": ["TRAIN", "RE-INGEST", "INDEX", "EVAL", "QUERY"],
        "npz_digests_stable_run_to_run": True,
        "cross_backend_decode_allclose_rtol": 1e-5,
        "cross_backend_decode_allclose_atol": 1e-5, "w16_install_bitexact_within_backend": True,
    },
    # §4 G20.6: "canary build: zero poison survives; trap build: FAILS the
    # determinism gate (the negative control must fail)".
    "G20.6": {
        "canary_zero_poison": True, "trap_fails_determinism_gate": True,
    },
}

# ------------------------------------------------------- the registry -----
# TASKS.md §4 rows (contract + authoring wave). Statuses here are the
# AUTHORING states only; GREEN/RED come solely from box reports under
# env/w20_gate_reports/ (transcribed into TASKS.md §4 by the operator).
GATE_REGISTRY: Dict[str, Dict[str, str]] = {
    "G20.P1": {"contract": "K1: the torch single path vs the frozen "
                           "fixtures + the 237-suite regression",
               "authored_in": "W20.4", "status": "AUTHORED"},
    "G20.P2": {"contract": "K2: fht_large differential twins + determinism",
               "authored_in": "W20.6", "status": "AUTHORED"},
    "G20.P3": {"contract": "K3+K4: TQ kernel parity (quant+dequant+qjl)",
               "authored_in": "W20.8 (qjl folds in W20.9)", "status": "AUTHORED"},
    "G20.1": {"contract": "capture cycle + TQ share",
              "authored_in": "W20.8 (phase-1)", "status": "AUTHORED"},
    "G20.2": {"contract": "step time + the 300-step run",
              "authored_in": "W20.8 (phase-1) / W20.11 (full)",
              "status": "AUTHORED"},
    "G20.3": {"contract": "W19 convergence (re-run on the kernel path)",
              "authored_in": "W20.11", "status": "NOT-AUTHORED"},
    "G20.4": {"contract": "the parity contract (levels 1-2)",
              "authored_in": "W20.8", "status": "AUTHORED"},
    "G20.5": {"contract": "protocol integrity",
              "authored_in": "W20.10", "status": "NOT-AUTHORED"},
    "G20.6": {"contract": "canary/trap",
              "authored_in": "W20.6", "status": "AUTHORED"},
}

# repo modules, populated by _import_repo_modules() AFTER the CUDA assert
_REPO: Dict[str, Any] = {}


# ------------------------------------------------------------ the guard ---
def assert_cuda() -> Dict[str, Any]:
    """The GPU-ONLY refusal (exit 3) + the device report, replicating
    _gpu_guard.py's discipline inline (require_cuda -> device_report);
    an off-family device is a NOTE, an off-target Ampere device a WARN --
    neither is a failure (guard semantics: advisory).
    """
    print("[w20-gates] step 1 -- CUDA assert (GPU box only, PROMPT.md §1)")
    if not torch.cuda.is_available():
        print()
        print("*" * 72)
        print("W20 GPU-ONLY: A10G required (PROMPT.md §1) -- REFUSING to "
              "run on a CPU box: exit 3, no repo import, no gate "
              "executed, no report written.")
        print("*" * 72)
        sys.exit(EXIT_NO_CUDA)
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    cap = (int(cap[0]), int(cap[1]))
    if cap[0] != _AMPERE_MAJOR:
        print(f"[w20-gates]   NOTE: capability {cap} is outside the Ampere "
              f"family (major != {_AMPERE_MAJOR}) -- results are advisory")
    elif cap != _A10G_CAPABILITY:
        print(f"[w20-gates]   WARN: Ampere but not the A10G target "
              f"(capability {cap} != {_A10G_CAPABILITY}) -- results are "
              "advisory for this device")
    print(f"[w20-gates]   device: {name} (capability {cap[0]}.{cap[1]}, "
          f"torch {torch.__version__}, cuda {torch.version.cuda})")
    return {"name": str(name), "capability": [cap[0], cap[1]],
            "torch": str(torch.__version__), "cuda": str(torch.version.cuda)}


def _import_repo_modules() -> None:
    """The repo imports -- ONLY after the CUDA assert (PROMPT.md §1).
    House convention (src/rag/_paths.py): sys.path gets src/rag, then
    `import _paths` anchors src/scripts + src/flute_extended; RAG modules
    import top-level (import turboquant; from tq_codes import TQCodes).
    """
    print("[w20-gates] step 2 -- repo imports (after the CUDA assert)")
    sys.path.insert(0, str(RAG_DIR))
    import _paths  # noqa: F401  (anchors idempotently, house style)
    import turboquant
    from tq_codes import TQCodes
    _REPO["turboquant"] = turboquant
    _REPO["TQCodes"] = TQCodes
    print(f"[w20-gates]   OK -- turboquant from {turboquant.__file__}")


# ------------------------------------------------------------ helpers -----
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ulp_distance_fp32(a: torch.Tensor, b: torch.Tensor) -> int:
    """ULP distance between two fp32 values: view(int32) bit patterns.
    Norms are non-negative, where the signed int32 view is monotonic --
    |ia - ib| IS the ulp distance. The degenerate zero-norm (-0.0 vs
    +0.0) is normalized to +0.0 first so it never reads as 2^31 ulps.
    """
    a = a.detach().to("cpu", torch.float32).reshape(-1)
    b = b.detach().to("cpu", torch.float32).reshape(-1)
    a = torch.where(a == 0, torch.zeros_like(a), a)
    b = torch.where(b == 0, torch.zeros_like(b), b)
    if a.numel() == 0 or b.numel() == 0:
        return 0
    ia = a.contiguous().view(torch.int32).to(torch.int64)
    ib = b.contiguous().view(torch.int32).to(torch.int64)
    return int((ia - ib).abs().max().item())


def _gate_result(gate: str, status: str, device: Dict[str, Any],
                 thresholds: Dict[str, Any], metrics: Dict[str, Any],
                 suite: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """The canonical per-gate report record (the JSON evidence artifact)."""
    return {
        "gate": gate, "status": status,
        "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": dict(device), "metrics": metrics,
        "thresholds": json.loads(json.dumps(thresholds)),  # drift record
        "suite": suite, "summary": summary,
    }


def _write_report(result: Dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{result['gate']}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n",
                    encoding="utf-8")
    return path


def _print_stub(gate_id: str, note: str = "") -> None:
    """The loud NOT-AUTHORED line (the call site exits 2 -- never a skip)."""
    print(f"{gate_id}: NOT-AUTHORED-YET (authored in "
          f"{GATE_REGISTRY[gate_id]['authored_in']} per TASKS.md §4){note}")


def _print_registry() -> None:
    print("W20 G20 gate registry -- TASKS.md §4 (authoring states; GREEN/RED "
          "come ONLY from env/w20_gate_reports/*.json box reports)")
    print(f"{'gate':<9}{'status':<14}{'authored in':<32}contract")
    for gate_id, row in GATE_REGISTRY.items():
        print(f"{gate_id:<9}{row['status']:<14}{row['authored_in']:<32}"
              f"{row['contract']}")
    print(f"[w20-gates] default report dir: {DEFAULT_REPORT_DIR}")


# ----------------------------------------------------------- G20.P1 ------
def _probe_parity(tq: Any, np: Any, entry: Dict[str, Any],
                  th: Dict[str, Any]) -> Dict[str, Any]:
    """One fixture probe through G20.P1's checks (a)-(d): rebuild the
    input from the recorded seed (torch.Generator, same shapes/dtype),
    run the CURRENT torch-CUDA path (TurboQuant per the manifest config:
    tq_kind/b/d/frame_seed/qjl/chan), compare vs the fixture -- (a) codes
    unpacked on BOTH sides with torch, tie fraction = mismatches/slots;
    (b) norm view-int32 ulp distance; (c) recon allclose; (d) re-quantize
    -- torch.equal on packed codes AND recon (PROPOSAL.md §5 levels 1+2).
    """
    file_name, cfg, seed = str(entry["file"]), entry["config"], int(entry["seed"])
    path = FIXTURES_DIR / file_name
    metrics: Dict[str, Any] = {"file": file_name, "seed": seed}
    if not path.is_file():
        metrics["error"], metrics["passed"] = f"fixture absent: {path}", False
        return metrics
    metrics["sha256_match"] = _sha256_file(path) == str(entry["sha256"])
    fx = np.load(str(path), allow_pickle=False)
    units, d = int(fx["units"]), int(fx["d"])
    gen = torch.Generator(device="cpu")       # the capture's own recipe
    gen.manual_seed(seed)
    x = torch.randn(units, d, generator=gen, dtype=torch.float32)
    metrics["input_rebuild_bitexact"] = bool(torch.equal(
        x, torch.from_numpy(np.ascontiguousarray(fx["input"]))))
    q = tq.TurboQuant(kind=str(cfg["tq_kind"]), bits=float(cfg["b"]),
                      d=int(cfg["d"]), seed=int(cfg["frame_seed"]),
                      qjl=bool(cfg["qjl"]), group=int(cfg["chan"]))
    dev = torch.device("cuda")
    bits_lo, bits_hi = int(fx["bits_lo"]), int(fx["bits_hi"])
    n_lo, n_hi = int(fx["n_lo"]), int(fx["n_hi"])
    partition = str(cfg["partition"])
    outlier = partition == "outlier"

    config_ok = units > 0
    if units > 0:  # self-describing codes must match the frozen config
        first = q.quant(x[0].to(dev))
        if not isinstance(first, _REPO["TQCodes"]):
            raise TypeError(f"quant() did not return TQCodes: {type(first)!r}")
        config_ok = (
            (int(first.bits_lo), int(first.bits_hi), int(first.n_lo),
             int(first.n_hi), str(first.partition))
            == (bits_lo, bits_hi, n_lo, n_hi, partition)
            == (int(cfg["bits_lo"]), int(cfg["bits_hi"]), int(cfg["n_lo"]),
                int(cfg["n_hi"]), str(cfg["partition"]))
            and (first.mask is not None) == outlier
            and (first.norm_hi is not None) == outlier)

    mismatches = 0
    slots = 0
    norm_max_ulps = 0
    layout_ok = recon_ok = True
    rtr_codes_ok = rtr_recon_ok = True
    for u in range(units):
        xu = x[u].to(dev)                      # the K1 torch-CUDA path
        c1, c2 = q.quant(xu), q.quant(xu)      # (d) the run-to-run pair
        r1, r2 = q.dequant(c1), q.dequant(c2)
        rtr_recon_ok &= bool(torch.equal(r1, r2))
        rtr_codes_ok &= bool(
            torch.equal(c1.idx_lo, c2.idx_lo)
            and torch.equal(c1.idx_hi, c2.idx_hi)
            and torch.equal(c1.norm, c2.norm)
            and (c1.mask is None or torch.equal(c1.mask, c2.mask)))
        # (a) codes: unpack BOTH sides with the current torch decode
        for side, packed in (("lo", c1.idx_lo), ("hi", c1.idx_hi)):
            bits = int(c1.bits_lo) if side == "lo" else int(c1.bits_hi)
            n = int(c1.n_lo) if side == "lo" else int(c1.n_hi)
            cur = tq.unpack_bits_t(packed, bits, n).to(torch.int64).cpu()
            fix = torch.from_numpy(np.ascontiguousarray(
                fx[f"idx_{side}_unpacked"][u])).to(torch.int64)
            mismatches += int((cur != fix).sum().item())
            slots += cur.numel()
            # layout pin (unpack_bits_t: "byte layout pinned by the frozen
            # fixtures"): torch-unpacking the FIXTURE's own packed stream
            # must reproduce its unpacked twin exactly
            layout_ok &= bool(torch.equal(
                tq.unpack_bits_t(torch.from_numpy(
                    np.ascontiguousarray(fx[f"idx_{side}"][u])), bits, n
                ).to(torch.int64), fix))
        # (b) norm (and norm_hi on the outlier split): view-int32 ulps
        norm_max_ulps = max(norm_max_ulps, _ulp_distance_fp32(
            c1.norm, torch.from_numpy(fx["norm"][u:u + 1])))
        if outlier and c1.norm_hi is not None:
            norm_max_ulps = max(norm_max_ulps, _ulp_distance_fp32(
                c1.norm_hi, torch.from_numpy(fx["norm_hi"][u:u + 1])))
        # (c) reconstruction
        recon_ok &= bool(torch.allclose(
            r1.to("cpu", torch.float32),
            torch.from_numpy(np.ascontiguousarray(fx["recon"][u])),
            rtol=float(th["recon_allclose_rtol"]),
            atol=float(th["recon_allclose_atol"])))

    tie_fraction = (mismatches / slots) if slots else 0.0
    rtr_equal = rtr_codes_ok and rtr_recon_ok
    checks = {
        "sha256_match": metrics["sha256_match"],
        "input_rebuild_bitexact": metrics["input_rebuild_bitexact"],
        "probe_nonempty": units > 0,
        "config_consistent": config_ok,
        "pack_layout_pinned": layout_ok,
        "ties_within_threshold": tie_fraction <= float(th["ties_max_fraction"]),
        "norm_within_ulp_budget": norm_max_ulps <= int(th["norm_max_ulps"]),
        "recon_allclose": recon_ok,
        "run_to_run_torch_equal": rtr_equal if bool(
            th["run_to_run_torch_equal"]) else True,
    }
    metrics.update({
        "config": {k: cfg[k] for k in ("group", "kind", "tq_kind", "d",
                                       "units", "b", "partition", "chan",
                                       "qjl", "frame_seed")},
        "mismatches": mismatches, "slots": slots,
        "tie_fraction": tie_fraction, "norm_max_ulps": norm_max_ulps,
        "recon_allclose": recon_ok, "run_to_run_codes_equal": rtr_codes_ok,
        "run_to_run_recon_equal": rtr_recon_ok,
        "checks": checks, "passed": all(checks.values()),
    })
    return metrics


def _run_suite() -> Dict[str, Any]:
    """(e) the 237-suite regression: pytest as a subprocess ON THE BOX,
    invoked exactly as the protocol's baseline step runs it (cwd = repo
    root); never in-process, never on the dev box (PROMPT.md §1)."""
    cmd = [sys.executable, "-m", "pytest", SUITE_TARGET, "-q", "--tb=no"]
    print(f"[G20.P1]   $ {' '.join(cmd)}  (cwd {REPO_ROOT}, "
          f"timeout {SUITE_TIMEOUT_S:.0f} s)")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                              text=True, timeout=SUITE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"invoked": True, "exit_code": None, "failures": None,
                "summary_line": f"TIMEOUT after {SUITE_TIMEOUT_S:.0f} s",
                "duration_s": round(time.monotonic() - t0, 1)}
    except OSError as exc:
        return {"invoked": True, "exit_code": None, "failures": None,
                "summary_line": f"launch failure: {exc!r}",
                "duration_s": round(time.monotonic() - t0, 1)}
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    summary_line = lines[-1] if lines else "(no pytest output)"
    m = re.search(r"(\d+)\s+failed", summary_line)
    return {"invoked": True, "exit_code": int(proc.returncode),
            "failures": int(m.group(1)) if m else 0,
            "summary_line": summary_line,
            "duration_s": round(time.monotonic() - t0, 1)}


def gate_g20_p1(device: Dict[str, Any]) -> Dict[str, Any]:
    """G20.P1 -- K1: the torch-CUDA single path vs the frozen fixtures +
    the 237-suite regression (TASKS.md §4 row 1; PROPOSAL.md §5 levels
    1-2). A10G box only."""
    import numpy as np  # the fixtures are npz -- box stack, post-assert
    tq = _REPO["turboquant"]
    th = THRESHOLDS["G20.P1"]

    print("[G20.P1] 1/4 -- the fixtures manifest")
    manifest_path = FIXTURES_DIR / MANIFEST_NAME
    if not manifest_path.is_file():
        print()
        print("*" * 72)
        print(f"G20.P1 FAIL: the fixtures manifest is absent:\n  {manifest_path}")
        print("  The frozen-probe fixtures must be captured FIRST on this box"
              " -- env/W20_GPU_PROTOCOL.md step 2 (the w20-baseline checkout"
              " + scripts/gpu/w20_capture_fixtures.py), then committed per"
              " step 3. Nothing to gate against without them.")
        print("*" * 72)
        return _gate_result(
            "G20.P1", "FAIL", device, th,
            metrics={"error": "fixtures_manifest.json absent -- capture "
                              "first (env/W20_GPU_PROTOCOL.md step 2)"},
            suite={"invoked": False, "exit_code": None, "failures": None,
                   "summary_line": None},
            summary="fixtures absent -- capture first "
                    "(env/W20_GPU_PROTOCOL.md step 2)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(manifest.get("probes", []))
    print(f"[G20.P1]   {manifest_path} -- {len(entries)} probe(s), "
          f"schema {manifest.get('schema_version')}")

    print(f"[G20.P1] 2/4 -- {len(entries)} probe(s): rebuild, run the "
          "torch-CUDA path, compare")
    per_probe: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        try:
            m = _probe_parity(tq, np, entry, th)
        except Exception as exc:  # loud, recorded -- never a silent skip
            m = {"file": str(entry.get("file", "?")), "error": repr(exc),
                 "passed": False}
        per_probe.append(m)
        if "checks" not in m:
            print(f"[G20.P1]   {i + 1:2d}/{len(entries)} "
                  f"{m['file']}: FAIL -- {m['error']}")
        else:
            rtr = m["run_to_run_codes_equal"] and m["run_to_run_recon_equal"]
            print(f"[G20.P1]   {i + 1:2d}/{len(entries)} {m['file']}: "
                  f"{'PASS' if m['passed'] else 'FAIL'}"
                  f"  ties {m['mismatches']}/{m['slots']}"
                  f" ({m['tie_fraction']:.2e})  norm {m['norm_max_ulps']} ulp"
                  f"  recon {'ok' if m['recon_allclose'] else 'DRIFT'}"
                  f"  rtr {'equal' if rtr else 'DRIFT'}")
    n_pass = sum(1 for m in per_probe if m.get("passed"))
    agg = {
        "probes_total": len(per_probe), "probes_passed": n_pass,
        "tie_mismatches": sum(int(m.get("mismatches", 0)) for m in per_probe),
        "tie_slots": sum(int(m.get("slots", 0)) for m in per_probe),
        "norm_max_ulps": max([int(m.get("norm_max_ulps", 0)) for m in per_probe] or [0]),
        "recon_allclose_all": bool(per_probe) and all(
            bool(m.get("recon_allclose")) for m in per_probe),
        "run_to_run_all": bool(per_probe) and all(
            bool(m.get("run_to_run_codes_equal"))
            and bool(m.get("run_to_run_recon_equal")) for m in per_probe),
    }
    agg["tie_fraction"] = ((agg["tie_mismatches"] / agg["tie_slots"])
                           if agg["tie_slots"] else 0.0)
    metrics = {"probes": per_probe, "aggregate": agg}

    print("[G20.P1] 3/4 -- the 237-suite regression (pytest subprocess)")
    suite = _run_suite()
    suite_ok = (bool(suite.get("invoked")) and suite.get("exit_code") == 0
                and int(suite.get("failures") or 0)
                <= int(th["suite_max_failures"]))
    suite["passed"] = suite_ok

    probes_ok = bool(per_probe) and n_pass == len(per_probe)
    status = "PASS" if (probes_ok and suite_ok) else "FAIL"
    summary = (f"{n_pass}/{len(per_probe)} probes; ties "
               f"{agg['tie_mismatches']}/{agg['tie_slots']} "
               f"({agg['tie_fraction']:.2e}); norm max "
               f"{agg['norm_max_ulps']} ulp; recon "
               f"{'allclose' if agg['recon_allclose_all'] else 'DRIFT'}; "
               f"run-to-run {'equal' if agg['run_to_run_all'] else 'DRIFT'}; "
               f"suite {suite.get('summary_line')!r} "
               f"(exit {suite.get('exit_code')})")
    print(f"[G20.P1] 4/4 -- {status}")
    return _gate_result("G20.P1", status, device, th, metrics, suite, summary)


# ------------------------------------------------------------ G20.P2 ------
# The geometry ladder (§4 G20.P2, via W20.6.3's brief): the ladder is
# about K; M stays 1-2 rows (2^22 rides M=1 so the twin's host-side
# staging stays bounded). The rungs and seeds are the pytest surface's
# own recorded ladder (src/rag/tests/w20/test_w20_fht_large.py) -- one
# deterministic ladder, two surfaces, identical inputs.
G20_P2_LADDER = (
    # (K, M, x_seed, g_seed, sign_seed)
    (2**19, 2, 619011, 619021, 619031),
    (2**20, 2, 620012, 620022, 620032),
    (2**22, 1, 622014, 622024, 622034),
)

# The K-eligibility refusals (§4 G20.P2: "K%32!=0 and K<=16,384
# refusals raise" -- the binding's own TORCH_CHECK family, which also
# refuses non-powers-of-two; the same three cases the pytest surface
# parametrizes). Patterns match fht_large_check_args's message text;
# the None pattern is built at run time from THRESHOLDS (never
# hardcoded here).
G20_P2_REFUSALS = (
    # (K, why ineligible, message pattern)
    (2**19 + 33, "K % 32 != 0", r"divisible by 32"),
    (49152, "non-power-of-two", r"power of two"),
    (8192, "K <= 16,384", None),
)

# the four fht_large bindings the gate needs (kernel pair + twin pair)
_FHT_LARGE_BINDINGS = (
    "fht_forward_large", "fht_backward_large",
    "tq_debug_fht_large_forward", "tq_debug_fht_large_backward",
)


def _probe_p2_rung(fht: Any, C: Any, K: int, M: int, x_seed: int,
                   g_seed: int, sign_seed: int,
                   th: Dict[str, Any]) -> Dict[str, Any]:
    """One ladder rung through G20.P2's contract: seeded (M, K) fp32 CUDA
    rows (a CPU generator -- bit-exact across boxes) and the +-1 signs of
    fht.rotation_signs's own recipe, through (i) the real kernel, (ii)
    its test-only scalar twin, (iii) fht.py's torch reference -- forward
    AND adjoint, allclose at §4's thresholds on both legs of each --
    plus the run-to-run pair (torch.equal, both orientations; the gate
    runs it on EVERY rung -- strictly stronger than the pytest surface's
    rung-0 spot check)."""
    rtol, atol = float(th["allclose_rtol"]), float(th["allclose_atol"])
    gen = torch.Generator()
    gen.manual_seed(x_seed)
    x = torch.randn(M, K, generator=gen, dtype=torch.float32).to("cuda")
    gen.manual_seed(g_seed)
    g = torch.randn(M, K, generator=gen, dtype=torch.float32).to("cuda")
    signs = fht.rotation_signs(K, sign_seed).to("cuda")

    f1 = C.fht_forward_large(x, signs)
    f2 = C.fht_forward_large(x, signs)      # the run-to-run pair
    b1 = C.fht_backward_large(g, signs)
    b2 = C.fht_backward_large(g, signs)
    twin_f = C.tq_debug_fht_large_forward(x, signs)
    twin_b = C.tq_debug_fht_large_backward(g, signs)
    ref_f = fht.fht_reference(x, signs)
    ref_b = fht.fht_reference_adjoint(g, signs)

    def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
        return bool(torch.allclose(a, b, rtol=rtol, atol=atol))

    def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max().item())

    ftr_eq, btr_eq = bool(torch.equal(f1, f2)), bool(torch.equal(b1, b2))
    checks = {
        "fwd_kernel_vs_twin": _close(f1, twin_f),
        "fwd_kernel_vs_ref": _close(f1, ref_f),
        "adj_kernel_vs_twin": _close(b1, twin_b),
        "adj_kernel_vs_ref": _close(b1, ref_b),
        "run_to_run_forward_equal": ftr_eq if bool(
            th["run_to_run_torch_equal"]) else True,
        "run_to_run_backward_equal": btr_eq if bool(
            th["run_to_run_torch_equal"]) else True,
    }
    return {
        "rung": f"K2p{K.bit_length() - 1}-M{M}", "K": K, "M": M,
        "seeds": {"x": x_seed, "g": g_seed, "signs": sign_seed},
        "fwd_max_abs_diff_vs_twin": _maxdiff(f1, twin_f),
        "fwd_max_abs_diff_vs_ref": _maxdiff(f1, ref_f),
        "adj_max_abs_diff_vs_twin": _maxdiff(b1, twin_b),
        "adj_max_abs_diff_vs_ref": _maxdiff(b1, ref_b),
        "run_to_run_forward_equal": ftr_eq,
        "run_to_run_backward_equal": btr_eq,
        "checks": checks, "passed": all(checks.values()),
    }


def _probe_p2_refusals(C: Any, th: Dict[str, Any]) -> List[Dict[str, Any]]:
    """§4 G20.P2's "refusals raise": ineligible K through the DIRECT
    binding (not fht.py's wrapper -- its auto mode silently falls back
    to the reference; the refusals are the binding's own TORCH_CHECKs),
    on BOTH orientations (the one shared fht_large_check_args gate
    serves the whole family). The zero/one fill is deterministic --
    the checks fire before any launch."""
    k_max = int(th["refusal_k_max"])
    want_mod32 = bool(th["refusal_k_mod32"])
    cases: List[Dict[str, Any]] = []
    for K, because, pattern in G20_P2_REFUSALS:
        pat = "<= " + str(k_max) if pattern is None else pattern
        # the parametrize-bug guard (the pytest surface's own assert):
        # each K must be ineligible under the CURRENT thresholds
        expected = ((want_mod32 and K % 32 != 0) or K <= k_max
                    or (K & (K - 1)) != 0)
        case: Dict[str, Any] = {"K": K, "because": because,
                                "pattern": pat,
                                "expected_to_raise": expected}
        x = torch.zeros(1, K, dtype=torch.float32, device="cuda")
        signs = torch.ones(K, dtype=torch.float32, device="cuda")
        for orient, fn in (("forward", C.fht_forward_large),
                           ("backward", C.fht_backward_large)):
            msg = None
            try:
                fn(x, signs)
            except RuntimeError as exc:
                msg = str(exc)
            case[orient + "_raised"] = msg is not None
            case[orient + "_message"] = (msg.splitlines()[0][:160]
                                         if msg else None)
            case[orient + "_match"] = bool(
                msg is not None and re.search(pat, msg))
        case["passed"] = bool(expected and all(
            case[o + "_" + k] for o in ("forward", "backward")
            for k in ("raised", "match")))
        cases.append(case)
    return cases


def gate_g20_p2(device: Dict[str, Any]) -> Dict[str, Any]:
    """G20.P2 -- K2: the fht_large differential twins + determinism
    (TASKS.md §4 row 2): the real kernel vs the debug twin vs the torch
    reference, allclose at §4's thresholds, on every rung of the
    geometry ladder and both orientations; run-to-run torch.equal; the
    K-eligibility refusals RAISE. test_w20_fht_large.py is this gate's
    pytest twin (the same ladder and checks; skip-vs-FAIL is the one
    semantic difference). A10G box only."""
    import fht  # the repo import (fht.py + its lazy _C) -- post-assert
    th = THRESHOLDS["G20.P2"]
    _no_suite = {"invoked": False, "exit_code": None, "failures": None,
                 "summary_line": None}

    print("[G20.P2] 1/3 -- the built extension (flute_extended._C via "
          "fht._load_kernel)")
    C = fht._load_kernel()
    missing = list(_FHT_LARGE_BINDINGS) if C is None else \
        [n for n in _FHT_LARGE_BINDINGS if not hasattr(C, n)]
    if missing:
        why = ("the extension is not built (flute_extended._C import "
               "failed)" if C is None else
               "the built extension lacks binding(s): "
               + ", ".join(missing))
        print()
        print("*" * 72)
        print(f"G20.P2 FAIL: {why}")
        print("  Build flute_extended on this A10G box first "
              "(env/W20_GPU_PROTOCOL.md step 1,")
        print("  scripts/gpu/w20_provision.sh). A gate is not skippable.")
        print("*" * 72)
        return _gate_result(
            "G20.P2", "FAIL", device, th, metrics={"error": why},
            suite=_no_suite,
            summary="extension incomplete -- build first "
                    "(env/W20_GPU_PROTOCOL.md step 1)")

    n_rungs = len(G20_P2_LADDER)
    print(f"[G20.P2] 2/3 -- the {n_rungs}-rung geometry ladder (2^19 / "
          "2^20 / 2^22): kernel vs debug twin vs torch reference, both "
          "orientations")
    per_rung: List[Dict[str, Any]] = []
    for i, (K, M, x_seed, g_seed, sign_seed) in enumerate(G20_P2_LADDER):
        rung_id = f"K2p{K.bit_length() - 1}-M{M}"
        try:
            m = _probe_p2_rung(fht, C, K, M, x_seed, g_seed, sign_seed, th)
        except Exception as exc:  # loud, recorded -- never a silent skip
            m = {"rung": rung_id, "K": K, "M": M, "error": repr(exc),
                 "passed": False}
        per_rung.append(m)
        if "checks" not in m:
            print(f"[G20.P2]   {i + 1}/{n_rungs} {rung_id}: FAIL -- "
                  f"{m['error']}")
        else:
            rtr = m["run_to_run_forward_equal"] and m[
                "run_to_run_backward_equal"]
            print(f"[G20.P2]   {i + 1}/{n_rungs} {rung_id}: "
                  f"{'PASS' if m['passed'] else 'FAIL'}"
                  f"  fwd |Δ| {m['fwd_max_abs_diff_vs_twin']:.2e}/"
                  f"{m['fwd_max_abs_diff_vs_ref']:.2e} (twin/ref)"
                  f"  adj |Δ| {m['adj_max_abs_diff_vs_twin']:.2e}/"
                  f"{m['adj_max_abs_diff_vs_ref']:.2e} (twin/ref)"
                  f"  rtr {'equal' if rtr else 'DRIFT'}")

    print("[G20.P2] 3/3 -- the K-eligibility refusals (must RAISE on both "
          "orientations)")
    try:
        refusal_cases = _probe_p2_refusals(C, th)
    except Exception as exc:  # loud, recorded -- never a silent skip
        refusal_cases = [{"K": None, "error": repr(exc), "passed": False}]
    for case in refusal_cases:
        if "error" in case:
            print(f"[G20.P2]   K={case.get('K')}: FAIL -- {case['error']}")
        else:
            both = case["forward_raised"] and case["backward_raised"]
            match = case["forward_match"] and case["backward_match"]
            print(f"[G20.P2]   K={case['K']} ({case['because']}): "
                  f"{'PASS' if case['passed'] else 'FAIL'}"
                  f"  raise {'both' if both else 'MISSING'}"
                  f"  msg match {'ok' if match else 'MISS'}")

    n_rung_pass = sum(1 for m in per_rung if m.get("passed"))
    n_refusal = sum(1 for c in refusal_cases if c.get("passed"))
    agg = {
        "rungs_total": len(per_rung), "rungs_passed": n_rung_pass,
        "fwd_max_abs_diff_vs_twin": max(
            [float(m.get("fwd_max_abs_diff_vs_twin", 0.0))
             for m in per_rung] or [0.0]),
        "fwd_max_abs_diff_vs_ref": max(
            [float(m.get("fwd_max_abs_diff_vs_ref", 0.0))
             for m in per_rung] or [0.0]),
        "adj_max_abs_diff_vs_twin": max(
            [float(m.get("adj_max_abs_diff_vs_twin", 0.0))
             for m in per_rung] or [0.0]),
        "adj_max_abs_diff_vs_ref": max(
            [float(m.get("adj_max_abs_diff_vs_ref", 0.0))
             for m in per_rung] or [0.0]),
        "run_to_run_all": bool(per_rung) and all(
            bool(m.get("run_to_run_forward_equal"))
            and bool(m.get("run_to_run_backward_equal"))
            for m in per_rung),
        "refusals_total": len(refusal_cases),
        "refusals_raise": n_refusal,
    }
    metrics = {"rungs": per_rung, "refusals": refusal_cases,
               "aggregate": agg}
    rungs_ok = bool(per_rung) and n_rung_pass == len(per_rung)
    refusals_ok = bool(refusal_cases) and n_refusal == len(refusal_cases)
    status = "PASS" if (rungs_ok and refusals_ok) else "FAIL"
    summary = (f"{n_rung_pass}/{len(per_rung)} rungs; fwd max |Δ| "
               f"{agg['fwd_max_abs_diff_vs_twin']:.2e} (twin) / "
               f"{agg['fwd_max_abs_diff_vs_ref']:.2e} (ref); adj max |Δ| "
               f"{agg['adj_max_abs_diff_vs_twin']:.2e} (twin) / "
               f"{agg['adj_max_abs_diff_vs_ref']:.2e} (ref); run-to-run "
               f"{'equal' if agg['run_to_run_all'] else 'DRIFT'}; refusals "
               f"{n_refusal}/{len(refusal_cases)} raise")
    print(f"[G20.P2] done -- {status}")
    return _gate_result("G20.P2", status, device, th, metrics, _no_suite,
                        summary)


# ------------------------------------------------------------ G20.P3 ------
# The TQ kernel differential surface (§4 G20.P3; TASKS W20.8.3): kernel
# quant+dequant vs the TORCH TWIN, both on CUDA. THE TWIN (documented
# choice): turboquant's own reference body, forced by monkeypatching the
# dispatcher's ONE probe _tq_kernel_available -> False (the W20.8.2 REV-3
# routing table's kernel-absent row). The SAME quantizer instance serves
# both legs (same D3 rotation signs, same codebook mirrors, same frame
# seed); only the routing flips, so the differential is exactly
# kernel-chain vs torch-body with ZERO gate-local math and zero
# convention drift (the twin's rotation itself rides fht.py's auto
# backend -- fht_large on CUDA at these d's -- the repo's reference
# path, not a re-implementation). The kernel leg rides the PUBLIC
# quant/dequant (the routing table's rows 1-3, asserted eligible first);
# the batched-grid rungs drive the DIRECT tq_quant_batched /
# tq_dequant_batched entries through the same _device_signs /
# _device_codebooks mirrors the dispatcher feeds them (the per-unit
# public API serves M=1 this wave; the batched surface is the
# prefill-scale contract, PROPOSAL.md §3.6). The pytest twin of this
# gate is src/rag/tests/w20/test_w20_tq_kernels.py (the same ladder,
# seeds and checks; skip-vs-FAIL is the one semantic difference).
G20_P3_LADDER = (
    # (d, units, x_seed, frame_seed) -- the kernel-eligible rungs; M
    # stays 1-2 rows (the twin's host-side staging stays bounded, the
    # G20_P2_LADDER convention)
    (2**19, 2, 819011, 819101),
    (2**22, 1, 822014, 822104),
)
# The torch-only rung (the REV-3 routing table's ineligible row):
# d=24,576 -- the conv full-window (two <=16,384 FHT segments,
# non-power-of-two) -- is served on the torch twin on CUDA; the gate
# records the ROUTING (ineligible by the dispatcher's own predicate,
# the twin serves it, the direct binding refuses loudly -- the belt),
# never kernel parity (the kernel pair refuses this geometry by design).
G20_P3_TORCH_ONLY = (24576, 2, 807601, 807701)
# The batched-grid rungs (§4 G20.P3: "batched-grid shapes 120-192
# units"; THRESHOLDS batched_grid_units_min/max hold the band's edges)
# at the S/M1/M2 geometry (d=2^19).
G20_P3_BATCHED_GRID = (
    # (M, d, x_seed, frame_seed)
    (120, 2**19, 811201, 811301),
    (192, 2**19, 819204, 819304),
)


@contextlib.contextmanager
def _torch_twin():
    """Force the torch twin ON CUDA for the G20.P3/G20.4 differentials:
    monkeypatch the dispatcher's ONE probe (turboquant.
    _tq_kernel_available) to return False for any argument, so
    quant/dequant take the torch reference body for CUDA inputs (the
    kernel-absent row -- the performance fallback that is NEVER a
    correctness fallback, W20.8.2). Restored on exit, exception-safe."""
    tq = _REPO["turboquant"]
    original = tq._tq_kernel_available
    tq._tq_kernel_available = lambda x=None: False
    try:
        yield
    finally:
        tq._tq_kernel_available = original


def _p3_units(units: int, d: int, seed: int) -> torch.Tensor:
    """(units, d) fp32 CUDA rows from a recorded seed -- drawn on the CPU
    generator (bit-exact across boxes) and moved to the device, the
    G20.P2 surface's own recipe."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(units, d, generator=gen, dtype=torch.float32).to(
        "cuda")


def _p3_quantizer(d: int, frame_seed: int) -> Any:
    """The ladder's quantizer: bits 3.5 (the b3/b4 split, the production
    recipe), flat partition, custom kind with the recorded frame seed
    (the constructor's tests-only channel -- the geometry is the
    ladder's variable, not the kind)."""
    return _REPO["turboquant"].TurboQuant(
        kind="custom", bits=3.5, d=d, seed=frame_seed)


def _probe_p3_unit(q: Any, x_unit: torch.Tensor,
                   th: Dict[str, Any]) -> Dict[str, Any]:
    """One unit through G20.P3's parity contract (the G20.P1 checks
    (a)-(d) re-pointed at the kernel-vs-twin differential): quantize
    BOTH ways on CUDA -- the kernel path (the dispatcher) and the torch
    twin (the probe patched off) -- then (a) codes unpacked on BOTH
    sides with torch, tie fraction = mismatches/slots; (b) norm
    view-int32 ulp distance; (c) recon allclose at §4's 1e-5/1e-5 on
    two legs: the end-to-end round trips and the isolated decode
    differential (the SAME kernel codes, both decoders); (d) the
    run-to-run pair on the kernel outputs (torch.equal, codes + recon
    -- §4 G20.P3: "same as G20.P1 thresholds on kernel outputs")."""
    tq = _REPO["turboquant"]
    rtol, atol = float(th["recon_allclose_rtol"]), float(
        th["recon_allclose_atol"])
    c_kernel = q.quant(x_unit)               # the kernel path
    r_kernel = q.dequant(c_kernel)           # the kernel decode
    with _torch_twin():
        c_twin = q.quant(x_unit)             # the torch twin encode
        r_round_twin = q.dequant(c_twin)     # the twin round trip
        r_decode_twin = q.dequant(c_kernel)  # the twin decode
    # (d) the run-to-run pair on the kernel outputs
    c2 = q.quant(x_unit)
    r2 = q.dequant(c2)
    rtr_codes = bool(torch.equal(c_kernel.idx_lo, c2.idx_lo)
                     and torch.equal(c_kernel.idx_hi, c2.idx_hi)
                     and torch.equal(c_kernel.norm, c2.norm))
    rtr_recon = bool(torch.equal(r_kernel, r2))
    # (a) the tie counter
    mismatches, slots = 0, 0
    for side in ("lo", "hi"):
        bits = int(c_kernel.bits_lo if side == "lo" else c_kernel.bits_hi)
        n = int(c_kernel.n_lo if side == "lo" else c_kernel.n_hi)
        ku = tq.unpack_bits_t(getattr(c_kernel, "idx_" + side),
                              bits, n).to(torch.int64)
        tu = tq.unpack_bits_t(getattr(c_twin, "idx_" + side),
                              bits, n).to(torch.int64)
        mismatches += int((ku != tu).sum().item())
        slots += int(ku.numel())
    # (b) the norm ulp budget
    norm_ulps = _ulp_distance_fp32(c_kernel.norm, c_twin.norm)
    # (c) the two recon legs
    recon_round = bool(torch.allclose(r_kernel, r_round_twin,
                                      rtol=rtol, atol=atol))
    recon_decode = bool(torch.allclose(r_kernel, r_decode_twin,
                                       rtol=rtol, atol=atol))
    return {
        "mismatches": mismatches, "slots": slots, "norm_max_ulps": norm_ulps,
        "recon_roundtrip_allclose": recon_round,
        "recon_decode_allclose": recon_decode,
        "run_to_run_codes_equal": rtr_codes,
        "run_to_run_recon_equal": rtr_recon,
    }


def _probe_p3_batched(q: Any, x: torch.Tensor,
                      th: Dict[str, Any]) -> Dict[str, Any]:
    """One batched-grid shape (§4's 120-192 units band) through G20.P3's
    contract at the DIRECT batched surface: ONE tq_quant_batched call
    over the seeded (M, d) prefill batch vs the torch twin quantized
    per unit (the probe patched off); ONE tq_dequant_batched call over
    the kernel's own batched codes vs the twin's per-unit decode of the
    SAME codes (the isolated decode differential at batched scale); the
    run-to-run pairs of both batched entries (torch.equal -- G20.4's
    level-1 hard contract at this surface)."""
    tq = _REPO["turboquant"]
    C = tq._load_tq_kernel()
    rtol, atol = float(th["recon_allclose_rtol"]), float(
        th["recon_allclose_atol"])
    M, d = int(x.shape[0]), int(q.d)
    dev = x.device
    # the dispatcher's own mirrors (the same tensors _quant_kernel feeds
    # the binding -- no gate-local re-derivation of signs or codebooks)
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)

    codes_lo, codes_hi, norms = C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    codes_lo2, codes_hi2, norms2 = C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    rtr_codes = bool(torch.equal(codes_lo, codes_lo2)
                     and torch.equal(codes_hi, codes_hi2)
                     and torch.equal(norms, norms2))
    out = C.tq_dequant_batched(
        codes_lo, codes_hi, norms, signs, cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
    out2 = C.tq_dequant_batched(
        codes_lo, codes_hi, norms, signs, cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
    rtr_recon = bool(torch.equal(out, out2))

    mismatches, slots, norm_max_ulps, decode_ok = 0, 0, 0, True
    with _torch_twin():
        for u in range(M):
            c_twin = q.quant(x[u])
            for side in ("lo", "hi"):
                bits = int(c_twin.bits_lo if side == "lo"
                           else c_twin.bits_hi)
                n = int(c_twin.n_lo if side == "lo" else c_twin.n_hi)
                ku = tq.unpack_bits_t((codes_lo if side == "lo"
                                       else codes_hi)[u], bits,
                                      n).to(torch.int64)
                tu = tq.unpack_bits_t(getattr(c_twin, "idx_" + side),
                                      bits, n).to(torch.int64)
                mismatches += int((ku != tu).sum().item())
                slots += int(ku.numel())
            norm_max_ulps = max(norm_max_ulps, _ulp_distance_fp32(
                norms[u].reshape(()), c_twin.norm))
            codes_u = _REPO["TQCodes"](
                q.kind, q.d, norms[u].reshape(()),
                q.bits_lo, q.bits_hi, q.n_lo, q.n_hi,
                codes_lo[u], codes_hi[u], q.seed)
            r_twin = q.dequant(codes_u)
            decode_ok &= bool(torch.allclose(out[u], r_twin,
                                             rtol=rtol, atol=atol))
    return {
        "M": M, "mismatches": mismatches, "slots": slots,
        "norm_max_ulps": norm_max_ulps,
        "recon_decode_allclose": decode_ok,
        "run_to_run_codes_equal": rtr_codes,
        "run_to_run_recon_equal": rtr_recon,
    }


def _probe_p3_torch_only(entry: tuple) -> Dict[str, Any]:
    """The torch-only rung (the REV-3 routing table's ineligible row),
    verified end to end: d=24,576 is NOT kernel-eligible by the
    dispatcher's own predicate, the dispatcher serves it on the torch
    twin on CUDA (quant + dequant, the W20.3 body verbatim), and the
    DIRECT binding refuses it loudly on both orientations (the geometry
    envelope's own TORCH_CHECK -- the belt behind the routing table)."""
    tq = _REPO["turboquant"]
    C = tq._load_tq_kernel()
    d, _, x_seed, frame_seed = entry
    q = _p3_quantizer(d, frame_seed)
    x = _p3_units(1, d, x_seed)
    rec: Dict[str, Any] = {
        "d": d,
        "note": "the conv full-window (two <=16,384 FHT segments; "
                "non-power-of-two): torch-only by the REV-3 routing "
                "table -- a routing row, never kernel parity",
        "geometry_eligible": bool(q._kernel_geometry_eligible()),
        "dispatcher_eligible": bool(q._quant_kernel_eligible(x[0])),
    }
    try:
        codes = q.quant(x[0])
        recon = q.dequant(codes)
        rec["twin_serves"] = bool(
            isinstance(codes, _REPO["TQCodes"]) and codes.d == d
            and tuple(recon.shape) == (d,) and recon.is_cuda)
    except Exception as exc:  # loud, recorded -- never a silent skip
        rec["twin_serves"], rec["twin_error"] = False, repr(exc)
    dev = x.device
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)
    codes_lo = torch.zeros(1, tq._packed_len(q.n_lo, q.bits_lo),
                           dtype=torch.uint8, device=dev)
    codes_hi = torch.zeros(1, tq._packed_len(q.n_hi, q.bits_hi),
                           dtype=torch.uint8, device=dev)
    norms = torch.zeros(1, dtype=torch.float32, device=dev)

    def _refuses(fn) -> bool:
        try:
            fn()
            return False
        except RuntimeError:
            return True

    rec["binding_quant_refuses"] = _refuses(
        lambda: C.tq_quant_batched(x[:1].contiguous(), d, signs,
                                   cb_lo.boundaries, cb_hi.boundaries,
                                   q.n_lo, q.bits_lo, q.n_hi, q.bits_hi))
    rec["binding_dequant_refuses"] = _refuses(
        lambda: C.tq_dequant_batched(
            codes_lo, codes_hi, norms, signs,
            cb_lo.centroids, cb_hi.centroids,
            q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32))
    rec["passed"] = bool(
        (not rec["geometry_eligible"]) and (not rec["dispatcher_eligible"])
        and rec["twin_serves"] and rec["binding_quant_refuses"]
        and rec["binding_dequant_refuses"])
    return rec


def gate_g20_p3(device: Dict[str, Any]) -> Dict[str, Any]:
    """G20.P3 -- K3+K4: the TQ kernel parity (TASKS.md §4 row 3): kernel
    quant+dequant vs the torch twin on CUDA, "same as G20.P1 thresholds
    on kernel outputs" (ties <= 0.01%, norm <= 1 ulp, recon allclose
    <= 1e-5, run-to-run torch.equal) over the geometry ladder (2^19,
    2^22) + the 24,576 torch-only routing row + the batched-grid shapes
    (120-192 units). The tie counter is recorded. A10G box only."""
    tq = _REPO["turboquant"]
    th = THRESHOLDS["G20.P3"]
    _no_suite = {"invoked": False, "exit_code": None, "failures": None,
                 "summary_line": None}

    print("[G20.P3] 1/5 -- the built extension (the TQ kernel pair via "
          "turboquant._load_tq_kernel)")
    C = tq._load_tq_kernel()
    if (C is None or not hasattr(C, "tq_quant_batched")
            or not hasattr(C, "tq_dequant_batched")):
        why = ("the extension is not built (flute_extended._C import "
               "failed)" if C is None else
               "the built extension lacks the TQ kernel pair "
               "(tq_quant_batched/tq_dequant_batched)")
        print()
        print("*" * 72)
        print(f"G20.P3 FAIL: {why}")
        print("  Build flute_extended on this A10G box first "
              "(env/W20_GPU_PROTOCOL.md step 1,")
        print("  scripts/gpu/w20_provision.sh). A gate is not skippable.")
        print("*" * 72)
        return _gate_result(
            "G20.P3", "FAIL", device, th, metrics={"error": why},
            suite=_no_suite,
            summary="extension incomplete -- build first "
                    "(env/W20_GPU_PROTOCOL.md step 1)")

    print("[G20.P3] 2/5 -- the geometry ladder (2^19 / 2^22): kernel vs "
          "the torch twin, quant + dequant")
    per_rung: List[Dict[str, Any]] = []
    for i, (d, units, x_seed, frame_seed) in enumerate(G20_P3_LADDER):
        rung_id = f"d2p{d.bit_length() - 1}"
        try:
            q = _p3_quantizer(d, frame_seed)
            x = _p3_units(units, d, x_seed)
            # the routing pin: rows 1-3 of the REV-3 table, in order --
            # the dispatcher MUST take the kernel path for this rung
            routed = bool(tq._tq_kernel_available(x[0])
                          and q._quant_kernel_eligible(x[0]))
            agg = {"mismatches": 0, "slots": 0, "norm_max_ulps": 0,
                   "recon_round": True, "recon_decode": True,
                   "rtr_codes": True, "rtr_recon": True}
            for u in range(units):
                m = _probe_p3_unit(q, x[u], th)
                agg["mismatches"] += m["mismatches"]
                agg["slots"] += m["slots"]
                agg["norm_max_ulps"] = max(agg["norm_max_ulps"],
                                           m["norm_max_ulps"])
                agg["recon_round"] &= m["recon_roundtrip_allclose"]
                agg["recon_decode"] &= m["recon_decode_allclose"]
                agg["rtr_codes"] &= m["run_to_run_codes_equal"]
                agg["rtr_recon"] &= m["run_to_run_recon_equal"]
            tie_fraction = ((agg["mismatches"] / agg["slots"])
                            if agg["slots"] else 0.0)
            checks = {
                "dispatcher_routes_kernel": routed,
                "ties_within_threshold": tie_fraction <= float(
                    th["ties_max_fraction"]),
                "norm_within_ulp_budget": agg["norm_max_ulps"] <= int(
                    th["norm_max_ulps"]),
                "recon_roundtrip_allclose": agg["recon_round"],
                "recon_decode_allclose": agg["recon_decode"],
                "run_to_run_torch_equal": (agg["rtr_codes"]
                                           and agg["rtr_recon"]) if bool(
                    th["run_to_run_torch_equal"]) else True,
            }
            per_rung.append({
                "rung": rung_id, "d": d, "units": units,
                "seeds": {"x": x_seed, "frame": frame_seed},
                "mismatches": agg["mismatches"], "slots": agg["slots"],
                "tie_fraction": tie_fraction,
                "norm_max_ulps": agg["norm_max_ulps"],
                "recon_roundtrip_allclose": agg["recon_round"],
                "recon_decode_allclose": agg["recon_decode"],
                "run_to_run_codes_equal": agg["rtr_codes"],
                "run_to_run_recon_equal": agg["rtr_recon"],
                "checks": checks, "passed": all(checks.values()),
            })
        except Exception as exc:  # loud, recorded -- never a silent skip
            per_rung.append({"rung": rung_id, "d": d, "error": repr(exc),
                             "passed": False})
        m = per_rung[-1]
        if "checks" not in m:
            print(f"[G20.P3]   {i + 1}/{len(G20_P3_LADDER)} {rung_id}: "
                  f"FAIL -- {m['error']}")
        else:
            rtr = m["run_to_run_codes_equal"] and m["run_to_run_recon_equal"]
            print(f"[G20.P3]   {i + 1}/{len(G20_P3_LADDER)} {rung_id}: "
                  f"{'PASS' if m['passed'] else 'FAIL'}"
                  f"  ties {m['mismatches']}/{m['slots']}"
                  f" ({m['tie_fraction']:.2e})  norm "
                  f"{m['norm_max_ulps']} ulp  recon "
                  f"{'ok' if m['recon_roundtrip_allclose'] else 'DRIFT'}"
                  f"  rtr {'equal' if rtr else 'DRIFT'}")

    print("[G20.P3] 3/5 -- the 24,576 torch-only routing row (the REV-3 "
          "table's ineligible geometry)")
    try:
        routing = _probe_p3_torch_only(G20_P3_TORCH_ONLY)
    except Exception as exc:  # loud, recorded -- never a silent skip
        routing = {"d": 24576, "error": repr(exc), "passed": False}
    if "error" in routing:
        print(f"[G20.P3]   d=24,576: FAIL -- {routing['error']}")
    else:
        print(f"[G20.P3]   d=24,576: "
              f"{'PASS' if routing['passed'] else 'FAIL'}"
              f"  ineligible {not routing['geometry_eligible']}"
              f"  twin serves {routing['twin_serves']}"
              f"  binding refuses "
              f"{routing['binding_quant_refuses'] and routing['binding_dequant_refuses']}")

    print("[G20.P3] 4/5 -- the batched-grid shapes (120-192 units, the "
          "DIRECT entries at d=2^19)")
    per_shape: List[Dict[str, Any]] = []
    for i, (M, d, x_seed, frame_seed) in enumerate(G20_P3_BATCHED_GRID):
        shape_id = f"M{M}"
        try:
            in_band = bool(int(th["batched_grid_units_min"]) <= M
                           <= int(th["batched_grid_units_max"]))
            q = _p3_quantizer(d, frame_seed)
            x = _p3_units(M, d, x_seed).contiguous()
            m = _probe_p3_batched(q, x, th)
            tie_fraction = ((m["mismatches"] / m["slots"])
                            if m["slots"] else 0.0)
            checks = {
                "units_within_band": in_band,
                "ties_within_threshold": tie_fraction <= float(
                    th["ties_max_fraction"]),
                "norm_within_ulp_budget": m["norm_max_ulps"] <= int(
                    th["norm_max_ulps"]),
                "recon_decode_allclose": m["recon_decode_allclose"],
                "run_to_run_torch_equal": (m["run_to_run_codes_equal"]
                                           and m["run_to_run_recon_equal"])
                if bool(th["run_to_run_torch_equal"]) else True,
            }
            m.update({"shape": shape_id, "d": d,
                      "seeds": {"x": x_seed, "frame": frame_seed},
                      "tie_fraction": tie_fraction,
                      "checks": checks, "passed": all(checks.values())})
        except Exception as exc:  # loud, recorded -- never a silent skip
            m = {"shape": shape_id, "d": d, "error": repr(exc),
                 "passed": False}
        per_shape.append(m)
        if "checks" not in m:
            print(f"[G20.P3]   {i + 1}/{len(G20_P3_BATCHED_GRID)} "
                  f"{shape_id}: FAIL -- {m['error']}")
        else:
            rtr = m["run_to_run_codes_equal"] and m["run_to_run_recon_equal"]
            print(f"[G20.P3]   {i + 1}/{len(G20_P3_BATCHED_GRID)} "
                  f"{shape_id}: {'PASS' if m['passed'] else 'FAIL'}"
                  f"  ties {m['mismatches']}/{m['slots']}"
                  f" ({m['tie_fraction']:.2e})  norm "
                  f"{m['norm_max_ulps']} ulp  decode "
                  f"{'ok' if m['recon_decode_allclose'] else 'DRIFT'}"
                  f"  rtr {'equal' if rtr else 'DRIFT'}")

    n_rung_pass = sum(1 for m in per_rung if m.get("passed"))
    n_shape_pass = sum(1 for m in per_shape if m.get("passed"))
    tie_mismatches = sum(int(m.get("mismatches", 0))
                         for m in per_rung + per_shape)
    tie_slots = sum(int(m.get("slots", 0)) for m in per_rung + per_shape)
    agg = {
        "rungs_total": len(per_rung), "rungs_passed": n_rung_pass,
        "shapes_total": len(per_shape), "shapes_passed": n_shape_pass,
        "torch_only_routing_passed": bool(routing.get("passed")),
        "tie_mismatches": tie_mismatches, "tie_slots": tie_slots,
        "tie_fraction": ((tie_mismatches / tie_slots) if tie_slots else 0.0),
        "norm_max_ulps": max(
            [int(m.get("norm_max_ulps", 0))
             for m in per_rung + per_shape] or [0]),
        "recon_allclose_all": bool(per_rung + per_shape) and all(
            bool(m.get("recon_roundtrip_allclose",
                       m.get("recon_decode_allclose", False)))
            for m in per_rung + per_shape),
        "run_to_run_all": bool(per_rung + per_shape) and all(
            bool(m.get("run_to_run_codes_equal"))
            and bool(m.get("run_to_run_recon_equal"))
            for m in per_rung + per_shape),
    }
    metrics = {
        "ladder": per_rung, "torch_only_routing": routing,
        "batched_grid": per_shape, "aggregate": agg,
        # the tie counter, recorded (§4 G20.P3; R20.2's acceptance line)
        "tie_counter": {"mismatches": tie_mismatches, "slots": tie_slots,
                        "fraction": agg["tie_fraction"]},
    }
    probes_ok = (bool(per_rung) and n_rung_pass == len(per_rung)
                 and bool(per_shape) and n_shape_pass == len(per_shape)
                 and bool(routing.get("passed")))
    status = "PASS" if probes_ok else "FAIL"
    summary = (f"{n_rung_pass}/{len(per_rung)} rungs + "
               f"{n_shape_pass}/{len(per_shape)} batched shapes + "
               f"routing {'ok' if routing.get('passed') else 'FAIL'}; ties "
               f"{tie_mismatches}/{tie_slots} "
               f"({agg['tie_fraction']:.2e}); norm max "
               f"{agg['norm_max_ulps']} ulp; recon "
               f"{'allclose' if agg['recon_allclose_all'] else 'DRIFT'}; "
               f"run-to-run {'equal' if agg['run_to_run_all'] else 'DRIFT'}")
    print(f"[G20.P3] 5/5 -- {status}")
    return _gate_result("G20.P3", status, device, th, metrics, _no_suite,
                        summary)


# --------------------------------------------------- G20.1 / G20.2 --------
# The PHASE-1 performance gates (§4 rows G20.1/G20.2; TASKS W20.8.3:
# "the capture-cycle + step-time phase timers read from a
# finetune_m1m2.py short-run history (thresholds per §4; full
# confirmation after K4/K5)"): a SHORT finetune run
# (scripts/gpu/finetune_m1m2.py --self-test --max-steps 8
# --refresh-every 4 --batch-pairs 8 = 2 capture cycles over 16
# prefills/cycle) in a SUBPROCESS on this box (cwd = repo root, the
# G20.P1 suite-subprocess discipline); the phase timers (capture_s /
# replay_s / backward_s, cumulative per history entry) read from the
# run's *_history.json.
#
# PHASE-1 SEMANTICS (marked in every report): G20.1 gates capture
# <= 10 s/cycle and TQ <= 5 ms/prefill; G20.2 gates step <= 2 s. The
# refresh-8 leg (step <= 1 s) and the 300-step run (<= 10 min) are
# W20.11's FULL G20.2 (K5's trainer levers) -- recorded
# not-evaluated-in-phase-1 in the metrics, never silently passed. The
# short run rides --self-test synthetic pairs (96-token members, the
# file's own mechanics-smoke mode -- corpus-independent, no pairs file
# needed at gate time); full confirmation after K4/K5 at production
# geometry.
#
# THE TQ-SHARE MEASUREMENT (documented limitation): the history's phase
# timers do NOT isolate TurboQuant from the model forward inside
# capture_s -- post-K3 the capture cycle is model-bound BY DESIGN
# (PROPOSAL.md §6: "TurboQuant drops out of the profile entirely").
# The TQ <= 5 ms/prefill leg is therefore measured by the gate itself,
# in-process: CUDA-event timing of the kernel pair (ONE
# tq_quant_batched + ONE tq_dequant_batched -- the per-unit round
# trip §6's ladder prices) over one seeded prefill-scale batch at the
# batched-grid band's top (192 units x d=2^19, G20_P3_BATCHED_GRID's
# own M=192 rung). Phase-1; full confirmation after K4/K5.
_FINETUNE_SCRIPT = REPO_ROOT / "scripts" / "gpu" / "finetune_m1m2.py"
_FINETUNE_SHORT_STEPS = 8          # 2 capture cycles at refresh 4
_FINETUNE_REFRESH_EVERY = 4        # §4 G20.1's own invocation row
_FINETUNE_BATCH_PAIRS = 8          # 16 prefills/cycle (§6's arithmetic)
_FINETUNE_TIMEOUT_S = 1800.0       # model load + 8 steps, generous
_FINETUNE_STDOUT_TAIL = 20         # the W19 phase lines, for the report
# one short run per harness process, shared transparently by the two
# phase-1 gates (they read the SAME protocol invocation, §4 G20.1's
# "--refresh-every 4" row); standalone invocations get a fresh run.
_FINETUNE_SHORT_CACHE: Dict[str, Any] = {}


def _run_finetune_short(gate_id: str) -> Dict[str, Any]:
    """The short finetune subprocess + the parsed history: runs
    scripts/gpu/finetune_m1m2.py (--self-test, 8 steps, refresh 4,
    batch 8) with the gates artifact under the DEFAULT report dir
    (gitignored by design), then reads the run's *_history.json (the
    phase timers' home). Returns the record (cmd/returncode/stdout
    tail/history/history_path/duration_s/reused) -- the caller turns a
    missing history into a loud FAIL, never a skip."""
    cached = _FINETUNE_SHORT_CACHE.get("record")
    if cached is not None:
        reused = dict(cached)
        reused["reused"] = True
        reused["first_run_by"] = _FINETUNE_SHORT_CACHE["gate_id"]
        print(f"[{gate_id}]   reusing the short run recorded by "
              f"{reused['first_run_by']} (same invocation: --self-test "
              f"{_FINETUNE_SHORT_STEPS} steps at refresh "
              f"{_FINETUNE_REFRESH_EVERY})")
        return reused
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = gate_id.lower().replace(".", "_")
    gates_out = DEFAULT_REPORT_DIR / f"finetune_short_{stem}.npz"
    hist_path = DEFAULT_REPORT_DIR / f"finetune_short_{stem}_history.json"
    cmd = [sys.executable, str(_FINETUNE_SCRIPT),
           "--self-test", "--device", "cuda",
           "--max-steps", str(_FINETUNE_SHORT_STEPS),
           "--refresh-every", str(_FINETUNE_REFRESH_EVERY),
           "--batch-pairs", str(_FINETUNE_BATCH_PAIRS),
           "--gates-out", str(gates_out)]
    print(f"[{gate_id}]   $ {' '.join(cmd)}  (cwd {REPO_ROOT}, timeout "
          f"{_FINETUNE_TIMEOUT_S:.0f} s)")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                              text=True, timeout=_FINETUNE_TIMEOUT_S)
        rc, out, err = int(proc.returncode), proc.stdout or "", \
            proc.stderr or ""
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "returncode": None, "stdout_tail": None,
                "stderr": f"TIMEOUT after {_FINETUNE_TIMEOUT_S:.0f} s",
                "history": None, "history_path": str(hist_path),
                "duration_s": round(time.monotonic() - t0, 1),
                "reused": False}
    except OSError as exc:
        return {"cmd": cmd, "returncode": None, "stdout_tail": None,
                "stderr": f"launch failure: {exc!r}", "history": None,
                "history_path": str(hist_path),
                "duration_s": round(time.monotonic() - t0, 1),
                "reused": False}
    tail = "\n".join(out.splitlines()[-_FINETUNE_STDOUT_TAIL:])
    history = None
    if hist_path.is_file():
        try:
            history = json.loads(hist_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            history = None
            err = f"history parse failure: {exc!r}" + ("\n" + err if err
                                                       else "")
    record = {"cmd": cmd, "returncode": rc, "stdout_tail": tail,
              "stderr": err.strip()[:2000],
              "history": history, "history_path": str(hist_path),
              "duration_s": round(time.monotonic() - t0, 1),
              "reused": False}
    if history is not None:
        _FINETUNE_SHORT_CACHE["gate_id"] = gate_id
        _FINETUNE_SHORT_CACHE["record"] = record
    return record


def _finetune_phase_metrics(history: List[Dict[str, Any]]
                            ) -> Dict[str, Any]:
    """The per-cycle / per-step phase-timer reads from the history: the
    entries' capture_s/replay_s/backward_s are CUMULATIVE (the trainer
    accumulates across steps), so the capture-cycle count is the number
    of positive capture deltas (+1 for the step-0 capture), the
    per-cycle capture time is the final capture_s over that count, and
    the per-step series is the per-entry delta sum (capture + replay +
    backward). The amortized step mean is the §4 "step" (the W19 log
    line's own convention -- every refresh_every-th step carries a
    capture); the max is recorded beside it."""
    steps = len(history)
    rec: Dict[str, Any] = {"steps": steps}
    if not steps:
        return rec
    cap = [float(e.get("capture_s", 0.0)) for e in history]
    rep = [float(e.get("replay_s", 0.0)) for e in history]
    bwd = [float(e.get("backward_s", 0.0)) for e in history]
    cycles = 1 + sum(1 for i in range(1, steps) if cap[i] > cap[i - 1])
    step_s = [cap[i] - cap[i - 1] + rep[i] - rep[i - 1]
              + bwd[i] - bwd[i - 1] for i in range(1, steps)]
    step_s.insert(0, cap[0] + rep[0] + bwd[0])
    rec.update({
        "capture_cycles": cycles,
        "capture_s_total": cap[-1],
        "capture_per_cycle_s": (cap[-1] / cycles) if cycles else None,
        "replay_s_total": rep[-1], "backward_s_total": bwd[-1],
        "step_mean_s": (sum(step_s) / steps),
        "step_max_s": max(step_s),
    })
    return rec


def _tq_prefill_ms(reps: int = 5) -> Dict[str, Any]:
    """The TQ share per prefill, phase-1 form: CUDA-event timing of the
    kernel pair (ONE tq_quant_batched + ONE tq_dequant_batched -- the
    per-unit round trip PROPOSAL §6's ladder prices at ~15-25 us/unit)
    over one seeded prefill-scale batch at the batched-grid band's top
    (G20_P3_BATCHED_GRID's M=192 rung: 192 units x d=2^19). Median over
    `reps` after one untimed warmup call (allocator + module load)."""
    tq = _REPO["turboquant"]
    C = tq._load_tq_kernel()
    M, d, x_seed, frame_seed = G20_P3_BATCHED_GRID[1]
    q = _p3_quantizer(d, frame_seed)
    x = _p3_units(M, d, x_seed).contiguous()
    dev = x.device
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)
    # warmup (untimed), then the timed round-trip pairs
    clo, chi, nrm = C.tq_quant_batched(
        x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
    C.tq_dequant_batched(clo, chi, nrm, signs, cb_lo.centroids,
                         cb_hi.centroids, q.n_lo, q.bits_lo, q.n_hi,
                         q.bits_hi, torch.float32)
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        clo, chi, nrm = C.tq_quant_batched(
            x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
            q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
        C.tq_dequant_batched(clo, chi, nrm, signs, cb_lo.centroids,
                             cb_hi.centroids, q.n_lo, q.bits_lo, q.n_hi,
                             q.bits_hi, torch.float32)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {"M": M, "d": d, "x_seed": x_seed, "reps": reps,
            "times_ms": [round(t, 3) for t in times],
            "median_ms": round(times[len(times) // 2], 3),
            "measured": "the kernel pair round trip (quant + dequant)"}


def gate_g20_1(device: Dict[str, Any]) -> Dict[str, Any]:
    """G20.1 -- the capture cycle + the TQ share, PHASE-1 (TASKS.md §4
    row 4: "capture <= 10 s/cycle after K3; TQ <= 5 ms/prefill (phase
    timers)"): the capture-cycle leg reads the finetune_m1m2.py
    short-run history's capture phase timer (per cycle, the delta
    count); the TQ-per-prefill leg is measured by this gate in-process
    (CUDA-event timing of the kernel pair over a prefill-scale batch --
    the phase timers do not isolate the TQ share from the model forward,
    post-K3 by design). Full confirmation after K4/K5. A10G box only."""
    th = THRESHOLDS["G20.1"]
    _no_suite = {"invoked": False, "exit_code": None, "failures": None,
                 "summary_line": None}

    print("[G20.1] 1/3 -- the short finetune (finetune_m1m2.py "
          "--self-test, 8 steps at refresh 4)")
    run = _run_finetune_short("G20.1")
    history = run.get("history")
    if not history:
        print()
        print("*" * 72)
        print("G20.1 FAIL: the short finetune produced no history "
              f"(returncode {run.get('returncode')})")
        print("  The run needs the model artifacts on this A10G box "
              "(env/W20_GPU_PROTOCOL.md).")
        print(f"  history expected at: {run.get('history_path')}")
        tail = run.get("stdout_tail") or run.get("stderr") or ""
        for line in tail.splitlines()[-8:]:
            print(f"    | {line}")
        print("*" * 72)
        return _gate_result(
            "G20.1", "FAIL", device, th,
            metrics={"phase": "PHASE-1", "finetune": run,
                     "error": "no history from the short finetune run"},
            suite=_no_suite,
            summary="short finetune failed -- model artifacts present? "
                    "(env/W20_GPU_PROTOCOL.md)")
    phase = _finetune_phase_metrics(history)
    print(f"[G20.1]   {phase['steps']} steps, "
          f"{phase['capture_cycles']} capture cycle(s): capture "
          f"{phase['capture_s_total']:.1f} s total = "
          f"{phase['capture_per_cycle_s']:.2f} s/cycle (threshold "
          f"{float(th['capture_cycle_max_s']):.0f} s)")

    print("[G20.1] 2/3 -- the TQ share per prefill (in-process "
          "CUDA-event timing; the phase timers cannot isolate it -- "
          "post-K3 capture is model-bound by design)")
    try:
        tq_share = _tq_prefill_ms()
        tq_ms = float(tq_share["median_ms"])
    except Exception as exc:  # loud, recorded -- never a silent skip
        tq_share, tq_ms = {"error": repr(exc)}, None
        print(f"[G20.1]   TQ share: FAIL -- {exc!r}")
    else:
        print(f"[G20.1]   TQ share: {tq_ms:.3f} ms/prefill "
              f"({tq_share['M']} units x d={tq_share['d']}, kernel pair "
              f"round trip, median of {tq_share['reps']}; threshold "
              f"{float(th['tq_prefill_max_ms']):.0f} ms)")

    print("[G20.1] 3/3 -- the verdict (PHASE-1)")
    capture_ok = (phase["capture_per_cycle_s"] is not None
                  and phase["capture_per_cycle_s"]
                  <= float(th["capture_cycle_max_s"]))
    tq_ok = tq_ms is not None and tq_ms <= float(th["tq_prefill_max_ms"])
    checks = {"short_run_completed": True,
              "capture_cycle_within_budget": bool(capture_ok),
              "tq_prefill_within_budget": bool(tq_ok)}
    metrics = {
        "phase": "PHASE-1 (full confirmation after K4/K5 per "
                 "TASKS.md §4; the short run rides --self-test "
                 "synthetic pairs, 96-token members)",
        "finetune": run, "phase_timers": phase,
        "tq_prefill": tq_share, "checks": checks,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    cap_s = (f"{phase['capture_per_cycle_s']:.2f}"
             if phase["capture_per_cycle_s"] is not None else "n/a")
    tq_txt = f"{tq_ms:.3f} ms" if tq_ms is not None else "n/a"
    summary = (f"PHASE-1: capture {cap_s} s/cycle (<= "
               f"{float(th['capture_cycle_max_s']):.0f}); TQ {tq_txt}/"
               f"prefill (<= {float(th['tq_prefill_max_ms']):.0f} ms)")
    print(f"[G20.1] done -- {status}")
    return _gate_result("G20.1", status, device, th, metrics, _no_suite,
                        summary)


def gate_g20_2(device: Dict[str, Any]) -> Dict[str, Any]:
    """G20.2 -- the step time + the 300-step run, PHASE-1 (TASKS.md §4
    row 5: "step <= 2 s after K3; <= 1 s at refresh 8; 300 steps
    <= 10 min"): the step-time leg reads the finetune_m1m2.py
    short-run history's phase timers (the amortized per-step capture +
    replay + backward, the W19 log line's own convention). The
    refresh-8 and 300-step legs are W20.11's FULL G20.2 (K5's trainer
    levers) -- recorded not-evaluated-in-phase-1, never silently
    passed. Full confirmation after K4/K5. A10G box only."""
    th = THRESHOLDS["G20.2"]
    _no_suite = {"invoked": False, "exit_code": None, "failures": None,
                 "summary_line": None}

    print("[G20.2] 1/3 -- the short finetune (finetune_m1m2.py "
          "--self-test, 8 steps at refresh 4)")
    run = _run_finetune_short("G20.2")
    history = run.get("history")
    if not history:
        print()
        print("*" * 72)
        print("G20.2 FAIL: the short finetune produced no history "
              f"(returncode {run.get('returncode')})")
        print("  The run needs the model artifacts on this A10G box "
              "(env/W20_GPU_PROTOCOL.md).")
        print(f"  history expected at: {run.get('history_path')}")
        tail = run.get("stdout_tail") or run.get("stderr") or ""
        for line in tail.splitlines()[-8:]:
            print(f"    | {line}")
        print("*" * 72)
        return _gate_result(
            "G20.2", "FAIL", device, th,
            metrics={"phase": "PHASE-1", "finetune": run,
                     "error": "no history from the short finetune run"},
            suite=_no_suite,
            summary="short finetune failed -- model artifacts present? "
                    "(env/W20_GPU_PROTOCOL.md)")
    phase = _finetune_phase_metrics(history)
    print(f"[G20.2]   {phase['steps']} steps: step mean "
          f"{phase['step_mean_s']:.2f} s, max "
          f"{phase['step_max_s']:.2f} s (threshold "
          f"{float(th['step_max_s']):.0f} s; refresh "
          f"{_FINETUNE_REFRESH_EVERY})")

    print("[G20.2] 2/3 -- the W20.11 legs (NOT evaluated in phase-1 -- "
          "recorded, never silently passed)")
    print(f"[G20.2]   refresh-8 step <= {float(th['step_max_s_refresh8'])}"
          " s: not evaluated (K5's trainer levers, W20.11's full G20.2)")
    print(f"[G20.2]   300 steps <= "
          f"{float(th['run_300_steps_max_min'])} min: not evaluated "
          "(W20.11's full G20.2; the phase-1 run is "
          f"{_FINETUNE_SHORT_STEPS} steps)")

    print("[G20.2] 3/3 -- the verdict (PHASE-1)")
    step_ok = (phase["step_mean_s"] is not None
               and phase["step_mean_s"] <= float(th["step_max_s"]))
    checks = {"short_run_completed": True,
              "step_time_within_budget": bool(step_ok)}
    metrics = {
        "phase": "PHASE-1 (full confirmation after K4/K5 per "
                 "TASKS.md §4; the short run rides --self-test "
                 "synthetic pairs, 96-token members)",
        "finetune": run, "phase_timers": phase, "checks": checks,
        "not_evaluated_phase1": {
            "step_max_s_refresh8": float(th["step_max_s_refresh8"]),
            "step_max_s_refresh8_note": "K5's trainer levers -- "
                                        "W20.11's full G20.2",
            "run_300_steps_max_min": float(th["run_300_steps_max_min"]),
            "run_300_steps_note": "W20.11's full G20.2 (the phase-1 "
                                  f"run is {_FINETUNE_SHORT_STEPS} steps)",
        },
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = (f"PHASE-1: step {phase['step_mean_s']:.2f} s mean "
               f"(max {phase['step_max_s']:.2f}) <= "
               f"{float(th['step_max_s']):.0f} s; refresh-8 and "
               "300-step legs not evaluated (W20.11 full)")
    print(f"[G20.2] done -- {status}")
    return _gate_result("G20.2", status, device, th, metrics, _no_suite,
                        summary)


# ------------------------------------------------------------ G20.4 -------
def gate_g20_4(device: Dict[str, Any]) -> Dict[str, Any]:
    """G20.4 -- the parity contract, levels 1-2 (TASKS.md §4 row 8):
    LEVEL-1 (the hard contract, PROPOSAL.md §5 level 1): run-to-run
    torch.equal on the KERNEL outputs (codes + recon, two calls) -- the
    dispatcher's per-unit form on every ladder rung AND the DIRECT
    batched entries' form at the M=192 batched-grid rung. LEVEL-2 (the
    cross-backend thresholds, §5 level 2): the same counters as G20.P3
    -- ties <= 0.01%, norm <= 1 ulp, allclose <= 1e-5 -- reported
    separately under level2_* keys on one ladder rung + one batched
    shape. A10G box only."""
    tq = _REPO["turboquant"]
    th = THRESHOLDS["G20.4"]
    _no_suite = {"invoked": False, "exit_code": None, "failures": None,
                 "summary_line": None}

    print("[G20.4] 1/4 -- the built extension (the TQ kernel pair)")
    C = tq._load_tq_kernel()
    if (C is None or not hasattr(C, "tq_quant_batched")
            or not hasattr(C, "tq_dequant_batched")):
        why = ("the extension is not built (flute_extended._C import "
               "failed)" if C is None else
               "the built extension lacks the TQ kernel pair "
               "(tq_quant_batched/tq_dequant_batched)")
        print()
        print("*" * 72)
        print(f"G20.4 FAIL: {why}")
        print("  Build flute_extended on this A10G box first "
              "(env/W20_GPU_PROTOCOL.md step 1,")
        print("  scripts/gpu/w20_provision.sh). A gate is not skippable.")
        print("*" * 72)
        return _gate_result(
            "G20.4", "FAIL", device, th, metrics={"error": why},
            suite=_no_suite,
            summary="extension incomplete -- build first "
                    "(env/W20_GPU_PROTOCOL.md step 1)")

    print("[G20.4] 2/4 -- LEVEL-1: run-to-run torch.equal on the kernel "
          "outputs (codes + recon, two calls)")
    level1_records: List[Dict[str, Any]] = []
    # (a) the dispatcher's per-unit form, every ladder rung
    for d, units, x_seed, frame_seed in G20_P3_LADDER:
        rung_id = f"d2p{d.bit_length() - 1}"
        try:
            q = _p3_quantizer(d, frame_seed)
            x = _p3_units(units, d, x_seed)
            codes_ok = recon_ok = True
            for u in range(units):
                c1, c2 = q.quant(x[u]), q.quant(x[u])
                r1, r2 = q.dequant(c1), q.dequant(c2)
                codes_ok &= bool(
                    torch.equal(c1.idx_lo, c2.idx_lo)
                    and torch.equal(c1.idx_hi, c2.idx_hi)
                    and torch.equal(c1.norm, c2.norm))
                recon_ok &= bool(torch.equal(r1, r2))
            rec = {"form": "dispatcher-per-unit", "rung": rung_id,
                   "d": d, "units": units,
                   "codes_equal": codes_ok, "recon_equal": recon_ok}
        except Exception as exc:  # loud, recorded -- never a silent skip
            rec = {"form": "dispatcher-per-unit", "rung": rung_id, "d": d,
                   "error": repr(exc), "codes_equal": False,
                   "recon_equal": False}
        level1_records.append(rec)
        if "error" in rec:
            print(f"[G20.4]   {rung_id}: FAIL -- {rec['error']}")
        else:
            print(f"[G20.4]   {rung_id}: "
                  f"{'PASS' if rec['codes_equal'] and rec['recon_equal'] else 'FAIL'}"
                  f"  codes {'equal' if rec['codes_equal'] else 'DRIFT'}"
                  f"  recon {'equal' if rec['recon_equal'] else 'DRIFT'}")
    # (b) the DIRECT batched entries' form, the M=192 batched-grid rung
    M, d, x_seed, frame_seed = G20_P3_BATCHED_GRID[1]
    try:
        q = _p3_quantizer(d, frame_seed)
        x = _p3_units(M, d, x_seed).contiguous()
        dev = x.device
        signs = q._device_signs(dev)
        cb_lo, cb_hi = q._device_codebooks(dev)
        clo1, chi1, nrm1 = C.tq_quant_batched(
            x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
            q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
        clo2, chi2, nrm2 = C.tq_quant_batched(
            x, d, signs, cb_lo.boundaries, cb_hi.boundaries,
            q.n_lo, q.bits_lo, q.n_hi, q.bits_hi)
        out1 = C.tq_dequant_batched(
            clo1, chi1, nrm1, signs, cb_lo.centroids, cb_hi.centroids,
            q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
        out2 = C.tq_dequant_batched(
            clo2, chi2, nrm2, signs, cb_lo.centroids, cb_hi.centroids,
            q.n_lo, q.bits_lo, q.n_hi, q.bits_hi, torch.float32)
        rec = {"form": "batched-direct", "shape": f"M{M}", "M": M, "d": d,
               "codes_equal": bool(
                   torch.equal(clo1, clo2) and torch.equal(chi1, chi2)
                   and torch.equal(nrm1, nrm2)),
               "recon_equal": bool(torch.equal(out1, out2))}
    except Exception as exc:  # loud, recorded -- never a silent skip
        rec = {"form": "batched-direct", "shape": f"M{M}", "M": M, "d": d,
               "error": repr(exc), "codes_equal": False,
               "recon_equal": False}
    level1_records.append(rec)
    if "error" in rec:
        print(f"[G20.4]   M{M} (batched direct): FAIL -- {rec['error']}")
    else:
        print(f"[G20.4]   M{M} (batched direct): "
              f"{'PASS' if rec['codes_equal'] and rec['recon_equal'] else 'FAIL'}"
              f"  codes {'equal' if rec['codes_equal'] else 'DRIFT'}"
              f"  recon {'equal' if rec['recon_equal'] else 'DRIFT'}")

    print("[G20.4] 3/4 -- LEVEL-2: the cross-backend thresholds (the "
          "same counters as G20.P3, reported separately)")
    level2: Dict[str, Any] = {}
    try:
        p3_th = dict(THRESHOLDS["G20.P3"])
        p3_th["ties_max_fraction"] = th["level2_ties_max_fraction"]
        p3_th["norm_max_ulps"] = th["level2_norm_max_ulps"]
        p3_th["recon_allclose_rtol"] = th["level2_allclose_rtol"]
        p3_th["recon_allclose_atol"] = th["level2_allclose_atol"]
        # one ladder rung (the 2^19 rung) + one batched shape (M=192):
        # the G20.P3 probes re-pointed at G20.4's level-2 thresholds
        d, units, x_seed, frame_seed = G20_P3_LADDER[0]
        q = _p3_quantizer(d, frame_seed)
        x = _p3_units(units, d, x_seed)
        mismatches = slots = norm_ulps = 0
        recon_ok = True
        for u in range(units):
            m = _probe_p3_unit(q, x[u], p3_th)
            mismatches += m["mismatches"]
            slots += m["slots"]
            norm_ulps = max(norm_ulps, m["norm_max_ulps"])
            recon_ok &= (m["recon_roundtrip_allclose"]
                         and m["recon_decode_allclose"])
        M, d, x_seed, frame_seed = G20_P3_BATCHED_GRID[1]
        q = _p3_quantizer(d, frame_seed)
        xb = _p3_units(M, d, x_seed).contiguous()
        mb = _probe_p3_batched(q, xb, p3_th)
        mismatches += mb["mismatches"]
        slots += mb["slots"]
        norm_ulps = max(norm_ulps, mb["norm_max_ulps"])
        recon_ok &= mb["recon_decode_allclose"]
        tie_fraction = (mismatches / slots) if slots else 0.0
        level2 = {
            "probes": "G20_P3_LADDER[0] (2^19) + G20_P3_BATCHED_GRID[1] "
                      "(M=192), the G20.P3 probe functions re-pointed at "
                      "G20.4's level-2 thresholds",
            "level2_tie_mismatches": mismatches, "level2_tie_slots": slots,
            "level2_tie_fraction": tie_fraction,
            "level2_norm_max_ulps": norm_ulps,
            "level2_recon_allclose": bool(recon_ok),
        }
        print(f"[G20.4]   ties {mismatches}/{slots} "
              f"({tie_fraction:.2e}; threshold "
              f"{float(th['level2_ties_max_fraction'])})  norm "
              f"{norm_ulps} ulp (threshold "
              f"{int(th['level2_norm_max_ulps'])})  recon "
              f"{'allclose' if recon_ok else 'DRIFT'} (rtol/atol "
              f"{float(th['level2_allclose_rtol'])}/"
              f"{float(th['level2_allclose_atol'])})")
    except Exception as exc:  # loud, recorded -- never a silent skip
        level2 = {"error": repr(exc), "level2_recon_allclose": False}
        print(f"[G20.4]   level-2: FAIL -- {exc!r}")

    print("[G20.4] 4/4 -- the verdict")
    level1_ok = bool(level1_records) and all(
        bool(r.get("codes_equal")) and bool(r.get("recon_equal"))
        for r in level1_records)
    level2_ok = ("error" not in level2
                 and level2["level2_tie_fraction"]
                 <= float(th["level2_ties_max_fraction"])
                 and level2["level2_norm_max_ulps"]
                 <= int(th["level2_norm_max_ulps"])
                 and bool(level2["level2_recon_allclose"]))
    checks = {
        "level1_run_to_run_torch_equal": level1_ok if bool(
            th["level1_run_to_run_torch_equal"]) else True,
        "level2_ties_within_threshold": bool(
            "error" not in level2
            and level2["level2_tie_fraction"]
            <= float(th["level2_ties_max_fraction"])),
        "level2_norm_within_ulp_budget": bool(
            "error" not in level2
            and level2["level2_norm_max_ulps"]
            <= int(th["level2_norm_max_ulps"])),
        "level2_recon_allclose": bool(
            "error" not in level2 and level2["level2_recon_allclose"]),
    }
    metrics = {"level1": level1_records, "level2": level2,
               "checks": checks}
    status = "PASS" if all(checks.values()) else "FAIL"
    if "error" in level2:
        l2_txt = f"level-2 error: {level2['error']}"
    else:
        l2_txt = (f"level-2 ties {level2['level2_tie_fraction']:.2e}, "
                  f"norm {level2['level2_norm_max_ulps']} ulp, recon "
                  f"{'allclose' if level2['level2_recon_allclose'] else 'DRIFT'}")
    summary = (f"level-1 run-to-run "
               f"{'equal' if level1_ok else 'DRIFT'} "
               f"({len(level1_records)} forms); {l2_txt}")
    print(f"[G20.4] done -- {status}")
    return _gate_result("G20.4", status, device, th, metrics, _no_suite,
                        summary)


# ------------------------------------------------------------ G20.6 -------
# The canary/trap negative controls (§4 G20.6). The variant artifacts
# per scripts/gpu/w20_build_variants.sh's contract:
# src/flute_extended/flute_extended/_C_{canary,trap}.so, path-loaded as
# ExtensionFileLoader("flute_extended._C", <so>) in a process that does
# NOT also import the production _C (the module name would collide --
# and this harness may well have imported it already, e.g. G20.P2
# under --gates all). The probe is therefore ALWAYS a fresh subprocess
# (the pytest surface's discipline). The one deliberate divergence
# from that surface: an ABSENT variant .so is a FAIL here (a gate is
# never skippable -- the operator must build the variants) where the
# pytest surface skips.
_VARIANT_SO_DIR = REPO_ROOT / "src" / "flute_extended" / "flute_extended"
_VARIANT_SO = {mode: _VARIANT_SO_DIR / f"_C_{mode}.so"
               for mode in ("canary", "trap")}
_VARIANT_BUILD_SCRIPT = "scripts/gpu/w20_build_variants.sh"
_VARIANT_PROBE_TIMEOUT_S = 600.0

# The hermetic variant probe (the pytest surface's twin probe): a fresh
# interpreter that imports torch, path-loads the variant .so, and runs
# one mode-specific control on one recorded geometry (G20_P2_LADDER's
# first rung). Exit codes: 0 = the control holds (canary: zero
# non-finite values; trap: two runs differ); 1 = the control FAILED
# (poison survived / the determinism gate passed with the trap armed);
# 3 = the .so is present but not importable; 2 = a bad mode (a caller
# bug, never shipped).
_G20_VARIANT_PROBE = r'''
import importlib.machinery
import importlib.util
import sys

import torch

mode, so_path = sys.argv[1], sys.argv[2]
K, M, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

# (1) the variant load per the build script's contract: the .so bakes
#     production's init symbol, so the module name must be
#     "flute_extended._C"; this process imports nothing else of the
#     package (no production _C -- no name collision).
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

# (2) the deterministic geometry (G20_P2_LADDER's first rung + its
#     recorded seed): seeded fp32 input and +-1 signs (fht's randint
#     recipe inline -- hermetic, no repo imports).
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


def _run_g20_variant_probe(mode: str, so: Path) -> Dict[str, Any]:
    """One variant control in a fresh subprocess on the recorded rung
    (G20_P2_LADDER's first rung + its recorded x seed -- the same rung
    the pytest surface's probe rides). Returns the probe record
    (mode/so/returncode/stdout/stderr/duration_s) plus the verdict:
    'clean' (rc 0, the control holds), 'control-failed' (rc 1),
    'not-importable' (rc 3), 'bad-mode' (rc 2), 'timeout',
    'launch-failure', or 'unexpected-rc-<n>'."""
    K, M, x_seed = G20_P2_LADDER[0][:3]
    print(f"[G20.6]   $ {sys.executable} -c <variant probe> {mode} {so} "
          f"(K {K}, M {M}, seed {x_seed}, timeout "
          f"{_VARIANT_PROBE_TIMEOUT_S:.0f} s)")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _G20_VARIANT_PROBE,
             mode, str(so), str(K), str(M), str(x_seed)],
            capture_output=True, text=True,
            timeout=_VARIANT_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"mode": mode, "so": str(so), "returncode": None,
                "stdout": None,
                "stderr": f"TIMEOUT after "
                          f"{_VARIANT_PROBE_TIMEOUT_S:.0f} s",
                "verdict": "timeout",
                "duration_s": round(time.monotonic() - t0, 1)}
    except OSError as exc:
        return {"mode": mode, "so": str(so), "returncode": None,
                "stdout": None, "stderr": f"launch failure: {exc!r}",
                "verdict": "launch-failure",
                "duration_s": round(time.monotonic() - t0, 1)}
    rc = int(proc.returncode)
    verdict = {0: "clean", 1: "control-failed", 2: "bad-mode",
               3: "not-importable"}.get(rc, f"unexpected-rc-{rc}")
    return {"mode": mode, "so": str(so), "returncode": rc,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(), "verdict": verdict,
            "duration_s": round(time.monotonic() - t0, 1)}


def _probe_stdout_value(record: Dict[str, Any], tag: str) -> Any:
    """The value token of the LAST '<tag> <value>' line of a probe's
    stdout (None if absent) -- the probe's one-line control report."""
    for line in reversed((record.get("stdout") or "").splitlines()):
        parts = line.split()
        if len(parts) == 2 and parts[0] == tag:
            return parts[1]
    return None


def gate_g20_6(device: Dict[str, Any]) -> Dict[str, Any]:
    """G20.6 -- the canary/trap negative controls (TASKS.md §4 row 9):
    the canary build NaN-poisons every output byte BEFORE the kernels
    write (zero surviving poison = the write-coverage proof); the trap
    build flips the pair arithmetic on a clock64 timing bit (the
    determinism gate MUST fail under it -- a determinism PASS with the
    trap armed is a broken assert). Variant .so ABSENT on this box =
    FAIL with the build-script pointer (a gate is never skippable;
    the pytest surface's skip is the one deliberate divergence).
    A10G box only."""
    th = THRESHOLDS["G20.6"]
    _no_suite = {"invoked": False, "exit_code": None, "failures": None,
                 "summary_line": None}

    print("[G20.6] 1/3 -- the variant artifacts (the "
          "w20_build_variants.sh contract)")
    absent = [mode for mode, so in _VARIANT_SO.items() if not so.is_file()]
    if absent:
        names = ", ".join("_C_" + m + ".so" for m in absent)
        print()
        print("*" * 72)
        print(f"G20.6 FAIL: variant .so absent: {names}")
        print(f"  expected at: {_VARIANT_SO_DIR}")
        print("  Build them on this A10G box first "
              "(env/W20_GPU_PROTOCOL.md):")
        print(f"    bash {_VARIANT_BUILD_SCRIPT} canary")
        print(f"    bash {_VARIANT_BUILD_SCRIPT} trap")
        print("  A gate is not skippable -- the negative controls are part")
        print("  of the K2 close. The pytest surface may skip an absent")
        print("  variant; this gate FAILS. That is the honest state.")
        print("*" * 72)
        return _gate_result(
            "G20.6", "FAIL", device, th,
            metrics={"error": "variant .so absent: " + names,
                     "absent": absent,
                     "build_script": _VARIANT_BUILD_SCRIPT,
                     "expected_dir": str(_VARIANT_SO_DIR)},
            suite=_no_suite,
            summary="variant .so absent -- build first "
                    "(" + _VARIANT_BUILD_SCRIPT + " canary|trap)")

    print("[G20.6] 2/3 -- the canary control (zero poison survives)")
    canary = _run_g20_variant_probe("canary", _VARIANT_SO["canary"])
    nf_tok = _probe_stdout_value(canary, "CANARY_NONFINITE")
    canary_nonfinite = int(nf_tok) if nf_tok is not None else None
    canary_ok = (canary["verdict"] == "clean") if bool(
        th["canary_zero_poison"]) else True
    print(f"[G20.6]   canary: {'PASS' if canary_ok else 'FAIL'}"
          f"  verdict {canary['verdict']}  nonfinite {canary_nonfinite}")

    print("[G20.6] 3/3 -- the trap control (determinism MUST fail)")
    trap = _run_g20_variant_probe("trap", _VARIANT_SO["trap"])
    id_tok = _probe_stdout_value(trap, "TRAP_RUN_TO_RUN_IDENTICAL")
    trap_identical = ({"True": True, "False": False}.get(id_tok)
                      if id_tok is not None else None)
    trap_ok = (trap["verdict"] == "clean") if bool(
        th["trap_fails_determinism_gate"]) else True
    print(f"[G20.6]   trap: {'PASS' if trap_ok else 'FAIL'}"
          f"  verdict {trap['verdict']}"
          f"  run-to-run identical {trap_identical}")

    metrics = {
        "variants": {
            "canary": {"so": str(_VARIANT_SO["canary"]),
                       "sha256": _sha256_file(_VARIANT_SO["canary"]),
                       "probe": canary, "nonfinite": canary_nonfinite,
                       "control_passed": canary_ok},
            "trap": {"so": str(_VARIANT_SO["trap"]),
                     "sha256": _sha256_file(_VARIANT_SO["trap"]),
                     "probe": trap,
                     "run_to_run_identical": trap_identical,
                     "control_passed": trap_ok},
        },
        "build_script": _VARIANT_BUILD_SCRIPT,
    }
    status = "PASS" if (canary_ok and trap_ok) else "FAIL"
    canary_txt = ("zero poison" if canary_ok else
                  "verdict " + canary["verdict"]
                  + (f" (nonfinite {canary_nonfinite})"
                     if canary_nonfinite is not None else ""))
    trap_txt = ("broke determinism (the control holds)" if trap_ok else
                "verdict " + trap["verdict"]
                + (f" (identical {trap_identical})"
                   if trap_identical is not None else ""))
    summary = f"canary: {canary_txt}; trap: {trap_txt}"
    print(f"[G20.6] done -- {status}")
    return _gate_result("G20.6", status, device, th, metrics, _no_suite,
                        summary)


# The runner table: G20.P1 (W20.4) + W20.6's pair (G20.P2 + G20.6) +
# W20.8's four (G20.P3 parity, G20.1/G20.2 phase-1 timers, G20.4 the
# parity contract); each authoring wave appends its gates per the §4
# registry (W20.9 folds qjl into G20.P3; W20.10: G20.5; W20.11:
# G20.2 full + G20.3). A gate in GATE_REGISTRY but not here is
# a NOT-AUTHORED stub.
GATE_RUNNERS: Dict[str, Any] = {
    "G20.P1": gate_g20_p1,
    "G20.P2": gate_g20_p2,
    "G20.P3": gate_g20_p3,
    "G20.1": gate_g20_1,
    "G20.2": gate_g20_2,
    "G20.4": gate_g20_4,
    "G20.6": gate_g20_6,
}


# ---------------------------------------------------------------- main ----
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="The W20 G20 gate harness (TASKS.md §4): runs the A10G "
                    "box gates, writes env/w20_gate_reports/<id>.json; "
                    "CUDA device required (exit 3 on a CPU box).")
    p.add_argument("--gates", default="all", metavar="CSV|all",
                   help="comma-separated gate ids (e.g. G20.P1,G20.6) or "
                        "'all' (every AUTHORED gate runs; NOT-AUTHORED "
                        "stubs are listed, not invoked). Default: all")
    p.add_argument("--report", default=None, metavar="DIR",
                   help=f"report directory (default: {DEFAULT_REPORT_DIR})")
    p.add_argument("--list", action="store_true",
                   help="print the gate registry + statuses and exit 0")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = assert_cuda()          # FIRST: exit 3 on a CPU box, always
    if args.list:
        _print_registry()
        return EXIT_OK
    report_dir = Path(args.report) if args.report else DEFAULT_REPORT_DIR
    _import_repo_modules()          # repo imports only after the assert
    print(f"[w20-gates] report dir: {report_dir}")

    raw = args.gates.strip()
    if raw.lower() == "all":
        to_run = [g for g in GATE_REGISTRY if g in GATE_RUNNERS]
        listed_stubs = [g for g in GATE_REGISTRY if g not in GATE_RUNNERS]
    else:
        requested = [g.strip() for g in raw.split(",") if g.strip()]
        unknown = [g for g in requested if g not in GATE_REGISTRY]
        if unknown:
            print(f"[w20-gates] ERROR: unknown gate id(s): {', '.join(unknown)}; "
                  f"valid ids: {', '.join(GATE_REGISTRY)}")
            return EXIT_REFUSAL
        invoked_stubs = [g for g in requested if g not in GATE_RUNNERS]
        if invoked_stubs:
            for g in invoked_stubs:
                _print_stub(g)
            print("[w20-gates] REFUSAL: NOT-AUTHORED gate(s) invoked -- "
                  "nothing executed, exit 2 (a refusal, never a skip)")
            return EXIT_REFUSAL
        to_run = requested
        listed_stubs = []
    if not to_run:
        print("[w20-gates] nothing to run (no AUTHORED gate requested)")
        return EXIT_OK
    print(f"[w20-gates] executing: {', '.join(to_run)}")

    statuses: List[str] = []
    for gate_id in to_run:
        try:
            result = GATE_RUNNERS[gate_id](device)
        except Exception as exc:  # loud, recorded -- never a silent skip
            result = _gate_result(
                gate_id, "FAIL", device, THRESHOLDS[gate_id],
                metrics={"error": repr(exc)},
                suite={"invoked": False, "exit_code": None, "failures": None,
                       "summary_line": None},
                summary=f"harness error: {exc!r}")
        statuses.append(result["status"])
        print(f"{result['gate']}: {result['status']} ({result['summary']})")
        print(f"[w20-gates]   report: {_write_report(result, report_dir)}")
    for g in listed_stubs:
        _print_stub(g, note=" (listed: --gates all)")

    n_pass, n_fail = statuses.count("PASS"), statuses.count("FAIL")
    print(f"[w20-gates] summary: {len(statuses)} executed, {n_pass} PASS, "
          f"{n_fail} FAIL; {len(listed_stubs)} NOT-AUTHORED (listed)")
    return EXIT_OK if n_fail == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
