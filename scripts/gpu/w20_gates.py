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
build-script pointer -- a gate is never skippable). The remaining 6
gates are NOT-AUTHORED stubs -- their waves (W20.8..W20.11) extend
GATE_RUNNERS; invoking a stub prints "G20.<id>: NOT-AUTHORED-YET
(authored in <wave> per TASKS.md §4)" and the harness exits 2.
"""
from __future__ import annotations

import argparse
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
               "authored_in": "W20.8 (qjl folds in W20.9)", "status": "NOT-AUTHORED"},
    "G20.1": {"contract": "capture cycle + TQ share",
              "authored_in": "W20.8 (phase-1)", "status": "NOT-AUTHORED"},
    "G20.2": {"contract": "step time + the 300-step run",
              "authored_in": "W20.8 (phase-1) / W20.11 (full)",
              "status": "NOT-AUTHORED"},
    "G20.3": {"contract": "W19 convergence (re-run on the kernel path)",
              "authored_in": "W20.11", "status": "NOT-AUTHORED"},
    "G20.4": {"contract": "the parity contract (levels 1-2)",
              "authored_in": "W20.8", "status": "NOT-AUTHORED"},
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


# The runner table: G20.P1 (W20.4) + W20.6's pair (G20.P2 + G20.6);
# each authoring wave appends its gates per the §4 registry (W20.8:
# G20.P3/G20.1/G20.2/G20.4; W20.9 folds qjl into G20.P3; W20.10: G20.5;
# W20.11: G20.2 full + G20.3). A gate in GATE_REGISTRY but not here is
# a NOT-AUTHORED stub.
GATE_RUNNERS: Dict[str, Any] = {
    "G20.P1": gate_g20_p1,
    "G20.P2": gate_g20_p2,
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
