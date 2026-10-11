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

G20.P1 is IMPLEMENTED (the K1 gate: the torch-CUDA single path vs the
frozen fixtures + the 237-suite regression); the other 8 gates are
NOT-AUTHORED stubs -- their waves (W20.6..W20.11) extend GATE_RUNNERS;
invoking a stub prints "G20.<id>: NOT-AUTHORED-YET (authored in <wave>
per TASKS.md §4)" and the harness exits 2.
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
               "authored_in": "W20.6", "status": "NOT-AUTHORED"},
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
              "authored_in": "W20.6", "status": "NOT-AUTHORED"},
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


# The runner table: G20.P1 now; each authoring wave appends its gates per
# the §4 registry (W20.6: G20.P2 + G20.6; W20.8: G20.P3/G20.1/G20.2/G20.4;
# W20.9 folds qjl into G20.P3; W20.10: G20.5; W20.11: G20.2 full + G20.3).
# A gate in GATE_REGISTRY but not here is a NOT-AUTHORED stub.
GATE_RUNNERS: Dict[str, Any] = {
    "G20.P1": gate_g20_p1,
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
