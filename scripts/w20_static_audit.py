#!/usr/bin/env python3
"""this tool NEVER imports or executes any target module (PROMPT.md §1.2); it parses and greps only.

The W20 static auditor — the Tier-D DoD vehicle (PROMPT.md §5, TASKS.md §1).
THE LAW above is binding on this tool at every level (PROMPT.md §1): no test
is ever run, no CPU test is authored, and NOTHING under src/ is imported or
executed by this auditor — every check is one of: a path fact (pathlib), a
regex over file text (re, MULTILINE), an AST parse (ast.parse — parse only,
never exec/import), a byte-compile (`sys.executable -m py_compile` via
subprocess — py_compile compiles to bytecode, it does NOT import the target),
or a READ-ONLY git query (`git tag -l <tag>` / `git log --oneline`; no git
command that mutates state is ever issued). Imports are stdlib-only: ast,
argparse, re, subprocess, sys, pathlib.

CLI:
    python3 scripts/w20_static_audit.py --list          # waves known to CHECKS, exit 0
    python3 scripts/w20_static_audit.py --wave W20.<n>  # run one wave's checks
    python3 scripts/w20_static_audit.py --all           # run every wave's checks

Check kinds (one small function each; see CHECK_KINDS):
    exists     — path exists (optionally "executable": True → any x mode bit)
    grep       — pattern (Python re, MULTILINE) has >= 1 hit in the file
    absent     — pattern has exactly 0 hits in the file
    ast_def    — ast.parse (parse ONLY): top-level def/ClassDef "symbol" whose
                 signature contains all of "args" (names only, order-insensitive
                 subset; a ClassDef is checked via its __init__)
    py_compile — subprocess `sys.executable -m py_compile <file>`; PASS iff rc 0
    git_tag    — subprocess `git tag -l <tag>` (read-only) output non-empty
    git_log    — subprocess `git log --oneline` (read-only) contains the subject
    ledger     — TASKS.md wave closure: between `## WAVE <id>` and the next
                 `## WAVE` heading (or EOF) there are zero `- [ ]` and zero
                 `- [~]` boxes

Output: per wave one header `== W20.<n>: <k> checks ==`, exactly ONE line per
check (`W20.<n> <kind> <target> … PASS` / `… FAIL (<short reason>)`), then a
summary `W20.<n>: <pass>/<total> PASS|FAIL`; `--all` adds a final
`ALL WAVES: <waves-green>/<waves-total> PASS|FAIL`. Exit 0 iff every executed
check passed (1 on any failure; 2 on usage error / unknown wave). A missing
file, a bad regex, or any other error is reported as a FAIL line — never a
traceback, never a crash.

CHECKS is plain data (list of dicts per wave). Wave entries mirror the
TASKS.md DoD of that wave 1:1 — drift is a wave-close-out failure (PROMPT.md
§3 step 5a). Later waves append their check lists here at their close-out
tasks; W20.0 was seeded at W20.0.3.
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repo root


# ------------------------------------------------------------------ helpers --

def _read_text(rel_path):
    """Read a repo-relative file. Returns (text, None) or (None, error-detail)."""
    path = REPO_ROOT / rel_path
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except FileNotFoundError:
        return None, "file missing"
    except OSError as exc:
        return None, f"unreadable: {exc}"


def _run(cmd, timeout=300):
    """Run a read-only subprocess in the repo root.

    Returns (returncode, stdout, stderr) or (None, "", error-detail) — never raises.
    """
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                              text=True, errors="replace", timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return None, "", f"timeout after {timeout}s: {' '.join(cmd[:4])}"
    except OSError as exc:
        return None, "", f"cannot run {cmd[0]!r}: {exc}"


def _pattern_and_text(spec):
    """Shared by grep/absent: compile the spec pattern, read the target file.

    Returns (compiled_regex, text, None) or (None, None, error-detail).
    """
    text, err = _read_text(spec["target"])
    if err:
        return None, None, err
    try:
        return re.compile(spec["pattern"], re.MULTILINE), text, None
    except re.error as exc:
        return None, None, f"regex error: {exc}"


def _arg_names(fn_node):
    """All argument names of an ast FunctionDef (posonly/args/kwonly/vararg/kwarg)."""
    a = fn_node.args
    names = [x.arg for x in getattr(a, "posonlyargs", [])]
    names += [x.arg for x in a.args]
    names += [x.arg for x in a.kwonlyargs]
    if a.vararg is not None:
        names.append(a.vararg.arg)
    if a.kwarg is not None:
        names.append(a.kwarg.arg)
    return names


# ------------------------------------------------------------- check kinds --
# Each function takes a spec dict and returns (ok: bool, detail: str); it never
# raises and never prints (run_check owns the ONE result line per check).

def check_exists(spec):
    path = REPO_ROOT / spec["target"]
    if not path.exists():
        return False, "path missing"
    if spec.get("executable"):
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            return False, f"stat failed: {exc}"
        if not mode & 0o111:  # any execute bit (user/group/other)
            return False, "exists but not executable"
    return True, ""


def check_grep(spec):
    rx, text, err = _pattern_and_text(spec)
    if err:
        return False, err
    return rx.search(text) is not None, ""


def check_absent(spec):
    rx, text, err = _pattern_and_text(spec)
    if err:
        return False, err
    hits = len(rx.findall(text))
    if hits:
        return False, f"{hits} forbidden hit(s)"
    return True, ""


def check_ast_def(spec):
    text, err = _read_text(spec["target"])
    if err:
        return False, err
    try:
        tree = ast.parse(text)  # parse ONLY — never exec, never import
    except SyntaxError as exc:
        return False, f"syntax error line {exc.lineno}: {exc.msg}"
    want = spec["symbol"]
    node = next((n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and n.name == want), None)
    if node is None:
        return False, f"top-level symbol {want!r} not found"
    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body
                     if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:
            return False, f"class {want!r} has no __init__ signature to check"
        node = init
    have = set(_arg_names(node))
    missing = sorted(set(spec.get("args", ())) - have)
    if missing:
        return False, f"signature missing arg(s): {', '.join(missing)}"
    return True, ""


def check_py_compile(spec):
    path = REPO_ROOT / spec["target"]
    if not path.is_file():
        return False, "file missing"
    rc, _out, err = _run([sys.executable, "-m", "py_compile", str(path)])
    if rc is None:
        return False, err
    if rc != 0:
        lines = (err or "").strip().splitlines()
        return False, f"py_compile rc={rc}: {lines[-1] if lines else 'no diagnostics'}"
    return True, ""


def check_git_tag(spec):
    tag = spec.get("tag", spec["target"])
    rc, out, _err = _run(["git", "tag", "-l", tag])  # read-only
    if rc is None:
        return False, "git not runnable"
    if rc != 0:
        return False, f"git tag -l rc={rc}"
    if not out.strip():
        return False, "tag not found"
    return True, ""


def check_git_log(spec):
    subject = spec.get("subject", spec["target"])
    rc, out, _err = _run(["git", "log", "--oneline"])  # read-only
    if rc is None:
        return False, "git not runnable"
    if rc != 0:
        return False, f"git log rc={rc}"
    for line in out.splitlines():
        if subject in line:
            return True, ""
    return False, f"no commit subject contains {subject!r}"


def check_ledger(spec):
    wave = spec.get("wave", spec["target"])
    text, err = _read_text("TASKS.md")
    if err:
        return False, err
    start = re.search(r"^## WAVE %s\b" % re.escape(wave), text, re.MULTILINE)
    if start is None:
        return False, f"no '## WAVE {wave}' heading in TASKS.md"
    after = text[start.end():]
    nxt = re.search(r"^## WAVE ", after, re.MULTILINE)
    section = after[:nxt.start()] if nxt else after
    open_boxes = section.count("- [ ]")
    blocked_boxes = section.count("- [~]")
    if open_boxes or blocked_boxes:
        return False, (f"{open_boxes} open '- [ ]', "
                       f"{blocked_boxes} blocked '- [~]' in wave section")
    return True, ""


def check_git_diff_empty(spec):
    # spec: {"kind": "git_diff_empty", "target": "<path>",
    #        "from": "<rev>", "to": "<rev-or-HEAD>"}
    # PASS iff `git diff <from>..<to> -- <path>` output is empty (the
    # byte-identical guarantee, e.g. kernel_fht.cu vs w20-wave-4).
    path = spec["target"]
    rev_from = spec["from"]
    rev_to = spec.get("to", "HEAD")
    rc, out, _err = _run(["git", "diff", f"{rev_from}..{rev_to}", "--", path])
    if rc is None:
        return False, "git not runnable"
    if rc != 0:
        return False, f"git diff rc={rc}"
    if out.strip():
        return False, f"diff not empty ({len(out.splitlines())} lines)"
    return True, ""


CHECK_KINDS = {
    "exists": check_exists,
    "grep": check_grep,
    "absent": check_absent,
    "ast_def": check_ast_def,
    "py_compile": check_py_compile,
    "git_tag": check_git_tag,
    "git_log": check_git_log,
    "ledger": check_ledger,
    "git_diff_empty": check_git_diff_empty,
}


# --------------------------------------------------------------- the checks --
# Plain data: wave id -> list of check specs ({"kind", "target", ...extras}).
# Each wave's list mirrors that wave's TASKS.md DoD 1:1 (drift = wave-close
# failure). Append new waves at their close-out tasks (PROMPT.md §3 step 5a).

CHECKS = {
    # W20.0 — seeded at W20.0.3 (env + audit infrastructure, TASKS.md W20.0)
    "W20.0": [
        # W20.0.1 — CPU-box toolchain audit + latest-stable provisioning
        {"kind": "exists", "target": "env/W20_ENV.md"},
        {"kind": "grep", "target": "env/W20_ENV.md",
         "pattern": "pip index versions torch"},
        {"kind": "grep", "target": "env/W20_ENV.md", "pattern": "ninja"},
        {"kind": "grep", "target": "env/W20_ENV.md", "pattern": r"^nvcc:"},
        # W20.0.2 — the A10G provisioning script (idempotent, latest-stable)
        {"kind": "exists", "target": "scripts/gpu/w20_provision.sh",
         "executable": True},
        {"kind": "grep", "target": "scripts/gpu/w20_provision.sh",
         "pattern": "nvidia-smi"},
        {"kind": "grep", "target": "scripts/gpu/w20_provision.sh",
         "pattern": "FLUTE_CUDA_ARCHES=86"},
        {"kind": "grep", "target": "scripts/gpu/w20_provision.sh",
         "pattern": "pytest src/rag/tests"},
        {"kind": "grep", "target": "scripts/gpu/w20_provision.sh",
         "pattern": "w20_gpu_env.lock"},
        {"kind": "absent", "target": "scripts/gpu/w20_provision.sh",
         "pattern": r"(torch|nvcc|numpy|transformers|faiss-cpu)=="},
        # W20.0.3 — this auditor (parses, never imports)
        {"kind": "exists", "target": "scripts/w20_static_audit.py"},
        {"kind": "py_compile", "target": "scripts/w20_static_audit.py"},
        {"kind": "grep", "target": "scripts/w20_static_audit.py",
         "pattern": "NEVER imports"},
        # W20.0.4 — the GPU test guard module (CUDA-only test authoring surface)
        {"kind": "exists", "target": "src/rag/tests/w20/__init__.py"},
        {"kind": "exists", "target": "src/rag/tests/w20/_gpu_guard.py"},
        {"kind": "grep", "target": "src/rag/tests/w20/_gpu_guard.py",
         "pattern": "torch.cuda.is_available"},
        {"kind": "grep", "target": "src/rag/tests/w20/_gpu_guard.py",
         "pattern": "allow_module_level"},
        {"kind": "grep", "target": "src/rag/tests/w20/_gpu_guard.py",
         "pattern": "8, 6"},
        {"kind": "absent", "target": "src/rag/tests/w20/_gpu_guard.py",
         "pattern": r"def test_"},
        # W20.0.5 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.0"},
        {"kind": "git_log", "target": "W20.0.1 —"},
        {"kind": "git_log", "target": "W20.0.2 —"},
        {"kind": "git_log", "target": "W20.0.3 —"},
        {"kind": "git_log", "target": "W20.0.4 —"},
        {"kind": "git_log", "target": "W20.0.5 —"},
        {"kind": "git_tag", "target": "w20-wave-0"},
    ],
    # W20.1 — baseline tag + frozen probe fixtures (appended at W20.1.4)
    "W20.1": [
        # W20.1.1 — tag the pre-rewrite tree
        {"kind": "git_tag", "target": "w20-baseline"},
        {"kind": "grep", "target": "TASKS.md", "pattern": "e9ea357"},
        # W20.1.2 — the fixture capture script (GPU-box, pinned checkout)
        {"kind": "exists", "target": "scripts/gpu/w20_capture_fixtures.py"},
        {"kind": "py_compile", "target": "scripts/gpu/w20_capture_fixtures.py"},
        {"kind": "grep", "target": "scripts/gpu/w20_capture_fixtures.py",
         "pattern": "w20-baseline"},
        {"kind": "grep", "target": "scripts/gpu/w20_capture_fixtures.py",
         "pattern": "sha256"},
        {"kind": "grep", "target": "scripts/gpu/w20_capture_fixtures.py",
         "pattern": "torch.cuda.is_available"},
        {"kind": "grep", "target": "scripts/gpu/w20_capture_fixtures.py",
         "pattern": "fixtures_manifest.json"},
        {"kind": "grep", "target": "scripts/gpu/w20_capture_fixtures.py",
         "pattern": "e9ea357fee"},
        {"kind": "absent", "target": "scripts/gpu/w20_capture_fixtures.py",
         "pattern": r"def test_"},
        # W20.1.3 — the A10G protocol doc (five steps in order)
        {"kind": "exists", "target": "env/W20_GPU_PROTOCOL.md"},
        {"kind": "grep", "target": "env/W20_GPU_PROTOCOL.md",
         "pattern": "w20-baseline"},
        {"kind": "grep", "target": "env/W20_GPU_PROTOCOL.md",
         "pattern": "w20_gates.py"},
        {"kind": "grep", "target": "env/W20_GPU_PROTOCOL.md",
         "pattern": "Step 1"},
        {"kind": "grep", "target": "env/W20_GPU_PROTOCOL.md",
         "pattern": "Step 2"},
        {"kind": "grep", "target": "env/W20_GPU_PROTOCOL.md",
         "pattern": "Step 3"},
        {"kind": "grep", "target": "env/W20_GPU_PROTOCOL.md",
         "pattern": "Step 4"},
        {"kind": "grep", "target": "env/W20_GPU_PROTOCOL.md",
         "pattern": "Step 5"},
        {"kind": "grep", "target": ".gitignore",
         "pattern": "env/w20_gate_reports/"},
        # W20.1.4 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.1"},
        {"kind": "git_log", "target": "W20.1.1 —"},
        {"kind": "git_log", "target": "W20.1.2 —"},
        {"kind": "git_log", "target": "W20.1.3 —"},
        {"kind": "git_log", "target": "W20.1.4 —"},
        {"kind": "git_tag", "target": "w20-wave-1"},
    ],
    # W20.2 — TQCodes device contract + codebook mirror (appended at W20.2.3)
    "W20.2": [
        # W20.2.1 — TQCodes torch-native
        {"kind": "exists", "target": "src/rag/tq_codes.py"},
        {"kind": "py_compile", "target": "src/rag/tq_codes.py"},
        {"kind": "grep", "target": "src/rag/tq_codes.py",
         "pattern": "class TQCodes"},
        {"kind": "grep", "target": "src/rag/tq_codes.py",
         "pattern": "norm:"},
        {"kind": "grep", "target": "src/rag/tq_codes.py",
         "pattern": "idx_lo:"},
        {"kind": "grep", "target": "src/rag/tq_codes.py",
         "pattern": "qjl_signs"},
        {"kind": "grep", "target": "src/rag/tq_codes.py",
         "pattern": "def to_arrays"},
        {"kind": "grep", "target": "src/rag/tq_codes.py",
         "pattern": "def from_arrays"},
        {"kind": "grep", "target": "src/rag/tq_codes.py",
         "pattern": "torch"},
        {"kind": "absent", "target": "src/rag/tq_codes.py",
         "pattern": r"\.item\(\)"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "from tq_codes import TQCodes"},
        {"kind": "absent", "target": "src/rag/turboquant.py",
         "pattern": r"^class TQCodes"},
        {"kind": "py_compile", "target": "src/rag/turboquant.py"},
        # W20.2.2 — codebook device mirror
        {"kind": "grep", "target": "src/rag/codebooks.py",
         "pattern": "OFFLINE-NUMPY-ISLAND"},
        {"kind": "grep", "target": "src/rag/codebooks.py",
         "pattern": "DeviceCodebook"},
        {"kind": "grep", "target": "src/rag/codebooks.py",
         "pattern": "_DEVICE_TENSOR_CACHE"},
        {"kind": "grep", "target": "src/rag/codebooks.py",
         "pattern": "def device_tensors"},
        {"kind": "grep", "target": "src/rag/codebooks.py",
         "pattern": "device=None"},
        {"kind": "py_compile", "target": "src/rag/codebooks.py"},
        # W20.2.3 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.2"},
        {"kind": "git_log", "target": "W20.2.1 —"},
        {"kind": "git_log", "target": "W20.2.2 —"},
        {"kind": "git_log", "target": "W20.2.3 —"},
        {"kind": "git_tag", "target": "w20-wave-2"},
    ],
    # W20.3 — turboquant.py torch rewrite (appended at W20.3.6)
    "W20.3": [
        # W20.3.1 — quant encode torch-native
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "boundary-scaling"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "torch.searchsorted"},
        # W20.3.2 — torch pack/unpack
        {"kind": "py_compile", "target": "src/rag/turboquant.py"},
        {"kind": "absent", "target": "src/rag/turboquant.py",
         "pattern": r"np\.packbits|np\.unpackbits"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "def pack_bits_t"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "def unpack_bits_t"},
        {"kind": "py_compile", "target": "src/rag/tests/test_turboquant.py"},
        {"kind": "grep", "target": "src/rag/tests/test_turboquant.py",
         "pattern": "pack_bits_t"},
        # W20.3.3 — dequant torch-native
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "fht_adjoint"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "def dequant"},
        # W20.3.4 — qjl torch + THE module-wide numpy-zero audit line
        {"kind": "absent", "target": "src/rag/turboquant.py",
         "pattern": r"np\.|numpy"},
        {"kind": "absent", "target": "src/rag/turboquant.py",
         "pattern": r"\.item\(\)|\.cpu\(\)|\.numpy\(\)|from_numpy"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "scale invariance"},
        # W20.3.5 — the module contract
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "differential twin"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "class TurboQuant"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "from tq_codes import TQCodes"},
        # W20.3.6 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.3"},
        {"kind": "git_log", "target": "W20.3.1 —"},
        {"kind": "git_log", "target": "W20.3.2 —"},
        {"kind": "git_log", "target": "W20.3.3 —"},
        {"kind": "git_log", "target": "W20.3.4 —"},
        {"kind": "git_log", "target": "W20.3.5 —"},
        {"kind": "git_log", "target": "W20.3.6 —"},
        {"kind": "git_tag", "target": "w20-wave-3"},
    ],
    # W20.4 — wiring + serving sweep + the K1 gates (appended at W20.4.7)
    "W20.4": [
        # W20.4.1 — tq_cache torch-native
        {"kind": "absent", "target": "src/rag/tq_cache.py",
         "pattern": r"np\.|numpy|\.item\(\)"},
        {"kind": "py_compile", "target": "src/rag/tq_cache.py"},
        # W20.4.2 — install requant
        {"kind": "absent", "target": "src/rag/install.py",
         "pattern": r"np\.|numpy|\.item\(\)"},
        {"kind": "grep", "target": "src/rag/install.py",
         "pattern": "def sum_turboquant_codes"},
        {"kind": "py_compile", "target": "src/rag/install.py"},
        # W20.4.3 — the serving sweep (faiss edge allowlisted)
        {"kind": "grep", "target": "src/rag/index.py",
         "pattern": "FAISS-EDGE"},
        {"kind": "absent", "target": "src/rag/query.py",
         "pattern": r"np\.|numpy"},
        {"kind": "absent", "target": "src/rag/ingest.py",
         "pattern": r"np\.|numpy"},
        {"kind": "py_compile", "target": "src/rag/index.py"},
        {"kind": "py_compile", "target": "src/rag/query.py"},
        {"kind": "py_compile", "target": "src/rag/ingest.py"},
        {"kind": "py_compile", "target": "src/rag/snapshot.py"},
        # W20.4.4 — evals/m1m2
        {"kind": "absent", "target": "src/rag/m1m2.py",
         "pattern": r"\.item\(\)"},
        {"kind": "grep", "target": "src/rag/evals.py",
         "pattern": "REPORT-EDGE"},
        {"kind": "py_compile", "target": "src/rag/evals.py"},
        {"kind": "py_compile", "target": "src/rag/m1m2.py"},
        # W20.4.5 — the CPU-elimination audit report
        {"kind": "exists", "target": "env/w20_cpu_elimination_audit.md"},
        {"kind": "grep", "target": "env/w20_cpu_elimination_audit.md",
         "pattern": "103"},
        {"kind": "grep", "target": "env/w20_cpu_elimination_audit.md",
         "pattern": "G20.P1"},
        # W20.4.6.a — the gate harness
        {"kind": "exists", "target": "scripts/gpu/w20_gates.py"},
        {"kind": "py_compile", "target": "scripts/gpu/w20_gates.py"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "G20.P1"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "THRESHOLDS"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "torch.cuda.is_available"},
        {"kind": "absent", "target": "scripts/gpu/w20_gates.py",
         "pattern": r"def test_"},
        # W20.4.6.b — the CUDA-gated pytest surface
        {"kind": "exists", "target":
         "src/rag/tests/w20/test_w20_torch_path.py"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_torch_path.py",
         "pattern": "def test_"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_torch_path.py",
         "pattern": "_gpu_guard"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_torch_path.py",
         "pattern": "allow_module_level"},
        {"kind": "py_compile", "target":
         "src/rag/tests/w20/test_w20_torch_path.py"},
        {"kind": "py_compile", "target": "src/rag/tests/test_turboquant.py"},
        {"kind": "py_compile", "target": "src/rag/tests/test_tq_cache.py"},
        # W20.4.7 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.4"},
        {"kind": "git_log", "target": "W20.4.1 —"},
        {"kind": "git_log", "target": "W20.4.2 —"},
        {"kind": "git_log", "target": "W20.4.3 —"},
        {"kind": "git_log", "target": "W20.4.4 —"},
        {"kind": "git_log", "target": "W20.4.5 —"},
        {"kind": "git_log", "target": "W20.4.6 —"},
        {"kind": "git_log", "target": "W20.4.6.a —"},
        {"kind": "git_log", "target": "W20.4.6.b —"},
        {"kind": "git_log", "target": "W20.4.7 —"},
        {"kind": "git_tag", "target": "w20-wave-4"},
    ],
    # W20.5 — fht_forward_large / fht_backward_large (appended at W20.5.6)
    "W20.5": [
        # W20.5.1 — the fht_large math contract header
        {"kind": "exists", "target":
         "src/flute_extended/include/flute/fht_large.cuh"},
        {"kind": "grep", "target":
         "src/flute_extended/include/flute/fht_large.cuh",
         "pattern": "16384"},
        {"kind": "grep", "target":
         "src/flute_extended/include/flute/fht_large.cuh",
         "pattern": "no atomics"},
        {"kind": "grep", "target":
         "src/flute_extended/include/flute/fht_large.cuh",
         "pattern": "cross"},
        {"kind": "grep", "target":
         "src/flute_extended/include/flute/fht_large.cuh",
         "pattern": "fht_large_stages"},
        # W20.5.2 — kernel_tq.cu born: fht_forward_large
        {"kind": "exists", "target": "src/flute_extended/src/kernel_tq.cu"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "cp\\.async"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "__launch_bounds__"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "cudaFuncSetAttribute"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "fht_forward_large"},
        # W20.5.3 — fht_backward_large (adjoint)
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "fht_backward_large"},
        # The determinism contract, module-wide (5.2+5.3)
        {"kind": "absent", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "atomic"},
        # W20.5.4 — bindings + build registration
        {"kind": "grep", "target": "src/flute_extended/src/bindings.cpp",
         "pattern": "fht_forward_large|fht_backward_large"},
        {"kind": "grep", "target": "src/flute_extended/setup.py",
         "pattern": "kernel_tq"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "FLUTE_TQ_CANARY|FLUTE_TQ_TRAP"},
        # W20.5.5 — fht.py eligibility
        {"kind": "grep", "target": "src/flute_extended/fht.py",
         "pattern": "fht_forward_large|fht_backward_large"},
        {"kind": "grep", "target": "src/flute_extended/fht.py",
         "pattern": "16384"},
        {"kind": "py_compile", "target": "src/flute_extended/fht.py"},
        # The byte-identical guarantee: the <=16,384 kernel untouched
        {"kind": "git_diff_empty", "target":
         "src/flute_extended/src/kernel_fht.cu",
         "from": "w20-wave-4", "to": "HEAD"},
        # W20.5.6 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.5"},
        {"kind": "git_log", "target": "W20.5.1 —"},
        {"kind": "git_log", "target": "W20.5.2 —"},
        {"kind": "git_log", "target": "W20.5.3 —"},
        {"kind": "git_log", "target": "W20.5.4 —"},
        {"kind": "git_log", "target": "W20.5.5 —"},
        {"kind": "git_log", "target": "W20.5.6 —"},
        {"kind": "git_tag", "target": "w20-wave-5"},
    ],
    # W20.6 — debug twins + canary/trap builds (appended at W20.6.4)
    "W20.6": [
        # W20.6.1 — the scalar debug twins
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "tq_debug_fht_large_forward"},
        {"kind": "grep", "target": "src/flute_extended/src/bindings.cpp",
         "pattern": "tq_debug"},
        # W20.6.2 — canary/trap variants
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "FLUTE_TQ_CANARY"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "FLUTE_TQ_TRAP"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "#ifdef"},
        {"kind": "exists", "target": "scripts/gpu/w20_build_variants.sh",
         "executable": True},
        # W20.6.3 — the CUDA-gated fht_large suite
        {"kind": "exists", "target":
         "src/rag/tests/w20/test_w20_fht_large.py"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_fht_large.py",
         "pattern": "def test_"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_fht_large.py",
         "pattern": "_gpu_guard"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_fht_large.py",
         "pattern": "allow_module_level"},
        {"kind": "py_compile", "target":
         "src/rag/tests/w20/test_w20_fht_large.py"},
        # W20.6.4 — the harness gates G20.P2 + G20.6
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "gate_g20_p2|gate_g20_6"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "NOT-AUTHORED-YET"},
        {"kind": "py_compile", "target": "scripts/gpu/w20_gates.py"},
        # W20.6.6 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.6"},
        {"kind": "git_log", "target": "W20.6.1 —"},
        {"kind": "git_log", "target": "W20.6.2 —"},
        {"kind": "git_log", "target": "W20.6.3 —"},
        {"kind": "git_log", "target": "W20.6.4 —"},
        {"kind": "git_tag", "target": "w20-wave-6"},
    ],
    # W20.7 — tq_quant_batched (appended at W20.7.5)
    "W20.7": [
        # W20.7.1 — the deterministic norm pass
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "shfl"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "finalizer|last-ticket|fold"},
        # W20.7.2 — boundary scaling + the compare ladder
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "boundary-scaling"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "ladder"},
        # W20.7.3 — butterfly + bfi packing
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "n_lo"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "pack_bits"},
        # W20.7.4 — the batched entry + bindings
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "tq_quant_batched"},
        {"kind": "grep", "target": "src/flute_extended/src/bindings.cpp",
         "pattern": "tq_quant_batched"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "grid-stride|grid_stride|overshoot"},
        # The determinism contract, module-wide (all of W20.7)
        {"kind": "absent", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "atomic"},
        # W20.7.5 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.7"},
        {"kind": "git_log", "target": "W20.7.1 —"},
        {"kind": "git_log", "target": "W20.7.2 —"},
        {"kind": "git_log", "target": "W20.7.3 —"},
        {"kind": "git_log", "target": "W20.7.4 —"},
        {"kind": "git_log", "target": "W20.7.5 —"},
        {"kind": "git_tag", "target": "w20-wave-7"},
    ],
    # W20.8 — tq_dequant_batched + the dispatcher (appended at W20.8.4)
    "W20.8": [
        # W20.8.1 — tq_dequant_batched
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "tq_dequant_batched"},
        {"kind": "grep", "target": "src/flute_extended/src/bindings.cpp",
         "pattern": "tq_dequant_batched"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "lut|LUT"},
        # W20.8.2 — the kernel-first dispatcher
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "kernel_available|dispatch"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "tq_quant_batched|tq_dequant_batched"},
        {"kind": "py_compile", "target": "src/rag/turboquant.py"},
        {"kind": "absent", "target": "src/rag/turboquant.py",
         "pattern": r"np\.|numpy"},
        # W20.8.3 — the suite + the gates
        {"kind": "exists", "target":
         "src/rag/tests/w20/test_w20_tq_kernels.py"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_tq_kernels.py",
         "pattern": "def test_"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_tq_kernels.py",
         "pattern": "_gpu_guard"},
        {"kind": "py_compile", "target":
         "src/rag/tests/w20/test_w20_tq_kernels.py"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "gate_g20_p3|gate_g20_4"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "NOT-AUTHORED-YET"},
        {"kind": "py_compile", "target": "scripts/gpu/w20_gates.py"},
        # The determinism contract, module-wide
        {"kind": "absent", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "atomic"},
        # W20.8.4 — wave close: ledger + git facts
        {"kind": "ledger", "target": "W20.8"},
        {"kind": "git_log", "target": "W20.8.1 —"},
        {"kind": "git_log", "target": "W20.8.2 —"},
        {"kind": "git_log", "target": "W20.8.3 —"},
        {"kind": "git_log", "target": "W20.8.4 —"},
        {"kind": "git_tag", "target": "w20-wave-8"},
    ],
    # W20.9 — the QJL chain + tq_l2norm_rows (appended at W20.9.4)
    "W20.9": [
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "tq_qjl"},
        {"kind": "grep", "target": "src/flute_extended/src/bindings.cpp",
         "pattern": "tq_qjl"},
        {"kind": "grep", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "tq_l2norm_rows"},
        {"kind": "grep", "target": "src/flute_extended/src/bindings.cpp",
         "pattern": "tq_l2norm_rows"},
        {"kind": "exists", "target":
         "src/rag/tests/w20/test_w20_qjl_l2norm.py"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_qjl_l2norm.py",
         "pattern": "def test_"},
        {"kind": "grep", "target":
         "src/rag/tests/w20/test_w20_qjl_l2norm.py",
         "pattern": "_gpu_guard"},
        {"kind": "py_compile", "target":
         "src/rag/tests/w20/test_w20_qjl_l2norm.py"},
        {"kind": "absent", "target": "src/flute_extended/src/kernel_tq.cu",
         "pattern": "atomic"},
        {"kind": "ledger", "target": "W20.9"},
        {"kind": "git_log", "target": "W20.9.1 —"},
        {"kind": "git_log", "target": "W20.9.2 —"},
        {"kind": "git_log", "target": "W20.9.3 —"},
        {"kind": "git_log", "target": "W20.9.4 —"},
        {"kind": "git_tag", "target": "w20-wave-9"},
    ],
    # W20.10 — the serving-side batched passes (appended at W20.10.4)
    "W20.10": [
        {"kind": "grep", "target": "src/rag/install.py",
         "pattern": "tq_quant_batched|tq_dequant_batched|_tq_kernel_available"},
        {"kind": "absent", "target": "src/rag/install.py",
         "pattern": r"np\.|numpy|\.item\(\)"},
        {"kind": "py_compile", "target": "src/rag/install.py"},
        {"kind": "grep", "target": "src/rag/index.py",
         "pattern": "tq_l2norm_rows|tq_dequant_batched|_tq_kernel_available"},
        {"kind": "grep", "target": "src/rag/index.py",
         "pattern": "FAISS-EDGE"},
        {"kind": "py_compile", "target": "src/rag/index.py"},
        {"kind": "grep", "target": "src/rag/evals.py",
         "pattern": "tq_quant_batched|tq_dequant_batched|_tq_kernel_available"},
        {"kind": "py_compile", "target": "src/rag/evals.py"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "gate_g20_5"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "NOT-AUTHORED-YET"},
        {"kind": "py_compile", "target": "scripts/gpu/w20_gates.py"},
        {"kind": "ledger", "target": "W20.10"},
        {"kind": "git_log", "target": "W20.10.1 —"},
        {"kind": "git_log", "target": "W20.10.2 —"},
        {"kind": "git_log", "target": "W20.10.3 —"},
        {"kind": "git_log", "target": "W20.10.4 —"},
        {"kind": "git_tag", "target": "w20-wave-10"},
    ],
    # W20.11 — the trainer levers (appended at W20.11.3)
    "W20.11": [
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": "refresh"},
        {"kind": "py_compile", "target": "scripts/gpu/finetune_m1m2.py"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "pin_memory|non_blocking"},
        {"kind": "py_compile", "target": "src/rag/m1m2_finetune.py"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "gate_g20_3"},
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "G20.2"},
        {"kind": "py_compile", "target": "scripts/gpu/w20_gates.py"},
        {"kind": "ledger", "target": "W20.11"},
        {"kind": "git_log", "target": "W20.11.1 —"},
        {"kind": "git_log", "target": "W20.11.2 —"},
        {"kind": "git_log", "target": "W20.11.3 —"},
        {"kind": "git_tag", "target": "w20-wave-11"},
    ],
    # W20.12 — closure (appended at W20.12.2)
    "W20.12": [
        # W20.12.1 — the handover + spec addendum
        # W20.14.5: the box operator's post-finetune push (8e73674)
        # legitimately REPLACED the W20 summary handover with the
        # post-finetune handover (the "spread 0" report W20.14 answers)
        # — the check now pins the LIVE handover's W20 lineage; the
        # retired heading stays a git fact on the w20-complete tag.
        {"kind": "grep", "target": "HANDOVER.md",
         "pattern": "Fails to Learn Discrimination"},
        {"kind": "grep", "target": "SPECIFICATION.md",
         "pattern": "W20 implementation status"},
        # W20.12.2 — the final audit facts
        {"kind": "grep", "target": "TASKS.md",
         "pattern": "13/13 waves closed"},
        {"kind": "ledger", "target": "W20.12"},
        # W20.12.3 — the final tag
        {"kind": "git_log", "target": "W20.12.1 —"},
        {"kind": "git_log", "target": "W20.12.2 —"},
        {"kind": "git_log", "target": "W20.12.3 —"},
        {"kind": "git_tag", "target": "w20-complete"},
    ],
    # W20.13 — the box-feedback stabilization wave (appended at W20.13.5)
    "W20.13": [
        # W20.13.1 — the pack domain restored (the legacy 1..8)
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "def _check_pack_width"},
        {"kind": "grep", "target": "src/rag/turboquant.py",
         "pattern": "integer width in 1..8"},
        {"kind": "absent", "target": "src/rag/turboquant.py",
         "pattern": r"bits not in \(3, 4\)"},
        {"kind": "grep", "target": "src/rag/evals.py",
         "pattern": r"RESOLVED \(W20.13.1\)"},
        {"kind": "py_compile", "target": "src/rag/turboquant.py"},
        # W20.13.2 — the rerank device contract
        {"kind": "grep", "target": "src/rag/index.py",
         "pattern": "def device"},
        {"kind": "grep", "target": "src/rag/index.py",
         "pattern": "W20.13.2"},
        {"kind": "py_compile", "target": "src/rag/index.py"},
        # W20.13.3 — the mechanical torch adaptation (six files)
        {"kind": "py_compile", "target": "src/rag/tests/test_evals.py"},
        {"kind": "py_compile", "target": "src/rag/tests/test_ingest.py"},
        {"kind": "py_compile", "target": "src/rag/tests/test_query.py"},
        {"kind": "py_compile", "target": "src/rag/tests/test_index.py"},
        {"kind": "py_compile",
         "target": "src/rag/tests/test_turboquant_split.py"},
        {"kind": "py_compile", "target": "src/rag/tests/test_w16_retrieval.py"},
        {"kind": "absent", "target": "src/rag/tests/test_ingest.py",
         "pattern": r"assert np\.array_equal"},
        {"kind": "absent", "target": "src/rag/tests/test_evals.py",
         "pattern": r"assert np\.array_equal"},
        # W20.13.4 — the parity re-derivation
        {"kind": "grep", "target": "scripts/gpu/w20_gates.py",
         "pattern": "norm_max_ulps.: 4"},
        {"kind": "grep", "target": "src/rag/tests/w20/test_w20_tq_kernels.py",
         "pattern": "NORM_MAX_ULPS = 4"},
        {"kind": "grep", "target": "TASKS.md",
         "pattern": "the W20.13.4 threshold re-derivation"},
        {"kind": "ledger", "target": "W20.13"},
        # W20.13.5 — the wave close
        {"kind": "git_log", "target": "W20.13.1 —"},
        {"kind": "git_log", "target": "W20.13.2 —"},
        {"kind": "git_log", "target": "W20.13.3 —"},
        {"kind": "git_log", "target": "W20.13.4 —"},
        {"kind": "git_log", "target": "W20.13.5 —"},
        {"kind": "git_tag", "target": "w20-wave-13"},
    ],
    # W20.14 — the box-feedback M1/M2 discrimination-metric wave (the
    # post-finetune handover: cos(pos) == cos(neg) EXACTLY at every step)
    "W20.14": [
        # W20.14.1 — the roll bug fixed (the metric that could not see)
        {"kind": "absent", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"shifted = sim\.roll"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "THE ROLL BUG"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def _pair_sim_stats"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"off-diagonal"},
        {"kind": "py_compile", "target": "src/rag/m1m2_finetune.py"},
        # W20.14.2 — the instrumentation
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "\"off_max\": float\\(off\\.max\\(\\)\\)"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "\"pool\": steps_on_batch"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "aliases the pool cycle"},
        # W20.14.3 — the layer mask (the H6 knob)
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "train_layers: Optional\\[Sequence\\[int\\]\\]"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "p\\.grad\\[~keep_mask\\] = 0"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"--train-layers"},
        {"kind": "py_compile", "target": "scripts/gpu/finetune_m1m2.py"},
        # W20.14.4 — the box-run regression gates (AUTHORED, PENDING)
        {"kind": "grep", "target": "src/rag/tests/test_w19_finetune.py",
         "pattern": "test_w19f_7_cosine_spread_measures_the_in_batch"},
        {"kind": "grep", "target": "src/rag/tests/test_w19_finetune.py",
         "pattern": "test_w19f_8_history_carries_the_w2014_fields"},
        {"kind": "py_compile", "target": "src/rag/tests/test_w19_finetune.py"},
        # W20.14.5 — the wave close
        {"kind": "ledger", "target": "W20.14"},
        {"kind": "git_log", "target": "W20.14.1 —"},
        {"kind": "git_log", "target": "W20.14.2 —"},
        {"kind": "git_log", "target": "W20.14.3 —"},
        {"kind": "git_log", "target": "W20.14.4 —"},
        {"kind": "git_log", "target": "W20.14.5 —"},
        {"kind": "git_tag", "target": "w20-wave-14"},
    ],
    # W20.15 — the fine-tune regime: the sawtooth decomposed (between-
    # block loss variance 7,595x within; 4.1% pair coverage; the
    # position-locked write's ceiling made measurable)
    "W20.15": [
        # W20.15.1 — the trainer observability (where the gates ARE)
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def _gate_value_stats"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r'"batch": batch_idx'},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "gk_absmean"},
        {"kind": "py_compile", "target": "src/rag/m1m2_finetune.py"},
        # W20.15.2 — the regime: per-epoch re-batching
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def epoch_batches"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "seed \\* 100003 \\+ epoch"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"--fixed-batches"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"--shuffle-seed"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": "coverage: \\{consumed"},
        {"kind": "py_compile", "target": "scripts/gpu/finetune_m1m2.py"},
        # W20.15.3 — the instruments: overlap buckets + the label-0 control
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def lexical_overlap"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def bucketed_spread"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "POSITION-ALIGNED"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": "curated negatives"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"--val-pairs"},
        # W20.15.4 — the box-run regression gates (AUTHORED, PENDING)
        {"kind": "grep", "target": "src/rag/tests/test_w20_15_regime.py",
         "pattern": "test_w20_15a_epoch_batches_rebatch_per_epoch"},
        {"kind": "grep", "target": "src/rag/tests/test_w20_15_regime.py",
         "pattern": "test_w20_15b_epoch_batches_contracts"},
        {"kind": "grep", "target": "src/rag/tests/test_w20_15_regime.py",
         "pattern": "test_w20_15c_lexical_overlap_math"},
        {"kind": "grep", "target": "src/rag/tests/test_w20_15_regime.py",
         "pattern": "test_w20_15d_bucketed_spread_splits_the_rigged_rig"},
        {"kind": "grep", "target": "src/rag/tests/test_w20_15_regime.py",
         "pattern": "test_w20_15e_history_carries_batch_ordinal_and_gate"},
        {"kind": "py_compile", "target": "src/rag/tests/test_w20_15_regime.py"},
        # W20.15.5 — the wave close
        {"kind": "ledger", "target": "W20.15"},
        {"kind": "git_log", "target": "W20.15.1 —"},
        {"kind": "git_log", "target": "W20.15.2 —"},
        {"kind": "git_log", "target": "W20.15.3 —"},
        {"kind": "git_log", "target": "W20.15.4 —"},
        {"kind": "git_log", "target": "W20.15.5 —"},
        {"kind": "git_tag", "target": "w20-wave-15"},
    ],

    # W21 — the content-addressed global cache (the position-lock and
    # the anisotropy pedestal, answered at the gates + the write)
    "W21": [
        # W21.1 — m1m2.py: the content address + centering + per-head gates
        {"kind": "grep", "target": "src/rag/m1m2.py",
         "pattern": r"def _content_slots"},
        {"kind": "grep", "target": "src/rag/m1m2.py",
         "pattern": r"def _sylvester_hadamard"},
        {"kind": "grep", "target": "src/rag/m1m2.py",
         "pattern": r'write_addr: str = "content"'},
        {"kind": "grep", "target": "src/rag/m1m2.py",
         "pattern": r"center_write: bool = True"},
        {"kind": "grep", "target": "src/rag/m1m2.py",
         "pattern": r"torch.zeros\(num_linear_layers, num_heads\)"},
        {"kind": "grep", "target": "src/rag/m1m2.py",
         "pattern": r"def _scatter_delta_content"},
        {"kind": "grep", "target": "src/rag/m1m2.py",
         "pattern": "W21 — THE ADDRESS"},
        {"kind": "py_compile", "target": "src/rag/m1m2.py"},
        # W21.2 — the trainer side: per-head grads + the one sim site
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def pair_sim_matrix"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def sim_matrix_stats"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"def bucketed_spread_from_sim"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": r"gk\[layer_idx\] \+= \(grad_m1 \* d1\)\.sum\(dim=\(1, 2\)\)"},
        {"kind": "grep", "target": "src/rag/m1m2_finetune.py",
         "pattern": "addressing regimes disagree"},
        {"kind": "py_compile", "target": "src/rag/m1m2_finetune.py"},
        # W21.3 — the driver: flags, artifact-first, the streaming ladder
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"--write-addr"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"--no-center-write"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"--addr-blocks"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": "_ADDR_LADDER"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"def _sim_from_units"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": r"\[5\] Gates artifact"},
        {"kind": "grep", "target": "scripts/gpu/finetune_m1m2.py",
         "pattern": "def _overlap_profile"},
        {"kind": "py_compile", "target": "scripts/gpu/finetune_m1m2.py"},
        # W21.4 — the box-run regression gates (AUTHORED, PENDING)
        {"kind": "grep", "target": "src/rag/tests/test_w21_content_addr.py",
         "pattern": "test_w21a_order_invariance"},
        {"kind": "grep", "target": "src/rag/tests/test_w21_content_addr.py",
         "pattern": "test_w21b_kernel_same_content_high_disjoint_low"},
        {"kind": "grep", "target": "src/rag/tests/test_w21_content_addr.py",
         "pattern": "test_w21c_centering_removes_the_common_direction"},
        {"kind": "grep", "target": "src/rag/tests/test_w21_content_addr.py",
         "pattern": "test_w21d_simhash_collision_monotone"},
        {"kind": "grep", "target": "src/rag/tests/test_w21_content_addr.py",
         "pattern": "test_w21g_write_deltas_match_and_grads_exact"},
        {"kind": "grep", "target": "src/rag/tests/test_w21_content_addr.py",
         "pattern": "test_w21i_loud_guards"},
        {"kind": "py_compile", "target": "src/rag/tests/test_w21_content_addr.py"},
        # W21.5 — the wave close
        {"kind": "ledger", "target": "W21"},
        {"kind": "git_log", "target": "W21.1 —"},
        {"kind": "git_log", "target": "W21.2 —"},
        {"kind": "git_log", "target": "W21.3 —"},
        {"kind": "git_log", "target": "W21.4 —"},
        {"kind": "git_log", "target": "W21.5 —"},
        {"kind": "git_tag", "target": "w21-content-addr"},
    ],
}


# ------------------------------------------------------------------ runner --

def run_check(wave, spec, failures):
    """Execute one check: print exactly ONE result line, append to failures."""
    kind = spec["kind"]
    fn = CHECK_KINDS.get(kind)
    if fn is None:
        ok, detail = False, f"unknown check kind {kind!r}"
    else:
        try:
            ok, detail = fn(spec)
        except Exception as exc:  # a failing check is NEVER a traceback
            ok, detail = False, f"{type(exc).__name__}: {exc}"
    target = str(spec.get("target", ""))
    line = "%s %s %s … %s" % (wave, kind, target, "PASS" if ok else "FAIL")
    if not ok and detail:
        line += " (%s)" % detail
    print(line)
    if not ok:
        failures.append(line)
    return ok


def run_wave(wave):
    """Run one wave's checks; print header + one line per check + summary.

    Returns True iff every check of the wave passed.
    """
    checks = CHECKS.get(wave, [])
    print("== %s: %d checks ==" % (wave, len(checks)))
    failures = []
    for spec in checks:
        run_check(wave, spec, failures)
    passed = len(checks) - len(failures)
    print("%s: %d/%d %s" % (wave, passed, len(checks),
                            "FAIL" if failures else "PASS"))
    return not failures


def _wave_sort_key(wave):
    """Numeric wave order: W20.2 before W20.10 (lexicographic would not be)."""
    try:
        return int(wave.split(".")[1])
    except (IndexError, ValueError):
        return 10 ** 9  # unknown shapes sort last, deterministically


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="w20_static_audit.py",
        description=("W20 Tier-D static auditor (PROMPT.md §5): parses and "
                     "greps only — NEVER imports or executes a target module."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wave", metavar="W20.<n>",
                      help="run one wave's checks")
    mode.add_argument("--all", action="store_true",
                      help="run every wave's checks")
    mode.add_argument("--list", action="store_true",
                      help="print the waves known to CHECKS and exit 0")
    args = parser.parse_args(argv)

    if args.list:
        for wave in sorted(CHECKS, key=_wave_sort_key):
            print(wave)
        return 0

    if args.wave is not None:
        if args.wave not in CHECKS:
            print("error: unknown wave %r; known waves: %s"
                  % (args.wave, ", ".join(sorted(CHECKS, key=_wave_sort_key))),
                  file=sys.stderr)
            return 2
        return 0 if run_wave(args.wave) else 1

    # --all: every wave, numeric order, plus the final roll-up line.
    results = [run_wave(wave) for wave in sorted(CHECKS, key=_wave_sort_key)]
    green = sum(1 for ok in results if ok)
    print("ALL WAVES: %d/%d %s" % (green, len(results),
                                   "FAIL" if len(results) != green else "PASS"))
    return 0 if len(results) == green else 1


if __name__ == "__main__":
    sys.exit(main())
