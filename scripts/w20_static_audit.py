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


CHECK_KINDS = {
    "exists": check_exists,
    "grep": check_grep,
    "absent": check_absent,
    "ast_def": check_ast_def,
    "py_compile": check_py_compile,
    "git_tag": check_git_tag,
    "git_log": check_git_log,
    "ledger": check_ledger,
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
    # W20.1 … W20.12 — appended by each wave's close-out task.
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
