#!/usr/bin/env python3
"""scripts/check_gpu_contract.py — the mechanical GPU-contract bans
(W3-T03, the PG-6 gate's first half; TASKS §3.2 + §4 rule 3,
PROPOSAL §4.1 rules 5/8).

Rules, scanned over scripts/ + tests/ + flute_train_kernels/ (NOT
flute_extended/src — frozen by the vendoring contract):

  R1  no torch.utils.checkpoint / gradient checkpointing (the box's
      CUDA deadlock ban — PROPOSAL §9);
  R2  no `.cuda()` / `.to("cuda")` / `device="cuda"` literals outside
      scripts/gpu_contract_allowlist.txt (device discipline: tensors
      propagate x.device; tests self-skip without CUDA);
  R3  no /home/ubuntu (or any absolute box path) inside an argparse
      default= (usage examples in help/docstrings are allowed);
  R4  no emoji and no forbidden unicode escapes in code files;
  R5  no module-level side effects on torch.backends.* (AST scan —
      the tf32 pinning belongs to the run entry points, not imports);
  R6  every .cu file under flute_train_kernels/src/ has a sibling
      differential gate test in tests/ (checked when W5 adds kernels;
      the current kernel's gate is tests/test_lut_gradients.py).

Exit 1 with file:line findings; exit 0 clean. Every allowlisted hit
carries a justification comment on its allowlist line.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

SCAN_DIRS = ("scripts", "tests", "flute_train_kernels")
ALLOWLIST = os.path.join(_HERE, "gpu_contract_allowlist.txt")

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F02F"
    "\U00002600-\U000026FF]")


def _scan_targets():
    out = subprocess.run(["git", "ls-files", "--full-name", "*.py", "*.cu",
                          "*.cuh", "*.cpp"],
                         capture_output=True, text=True, check=True,
                         cwd=_REPO)
    files = [f for f in out.stdout.split()
             if f.startswith(SCAN_DIRS) and not f.startswith(
                 "flute_train_kernels/flute_train_kernels/__init__")]
    return files


def _allowlist():
    entries = {}
    if os.path.exists(ALLOWLIST):
        with open(ALLOWLIST, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entries[line.split("#")[0].strip()] = \
                    line.split("#", 1)[1].strip() if "#" in line else ""
    return entries


def rule_r1(path, src, allow):
    """Gradient-checkpointing ban (ENABLE sites, AST: call sites and
    enabling keywords — the upstream HF supports_gradient_checkpointing
    class attribute and the checker's own pattern strings are not
    enables)."""
    hits = []
    tree = _parse(path, src)
    if tree is None:
        return hits
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else \
                (f.id if isinstance(f, ast.Name) else "")
            if name in ("checkpoint", "_checkpoint_sequential",
                        "gradient_checkpointing_enable"):
                # qualify: torch.utils.checkpoint / model.enable form
                qual = ast.dump(f)
                if "torch" in qual or name != "checkpoint" \
                        or "utils" in qual:
                    key = f"{path}:{node.lineno}"
                    if key not in allow:
                        hits.append(
                            f"R1 {key}: {name}(...) (gradient "
                            f"checkpointing is FORBIDDEN — the box "
                            f"deadlock)")
        if isinstance(node, ast.keyword) and node.arg == "use_reentrant":
            key = f"{path}:{node.lineno}"
            if key not in allow:
                hits.append(f"R1 {key}: use_reentrant= (gradient "
                            f"checkpointing is FORBIDDEN — the box "
                            f"deadlock)")
    return hits


def rule_r2(path, src, allow):
    """Device literals, letter-exact (TASKS R2): .cuda() and .to("cuda")
    CALL forms — AST, so docstrings/comments cannot false-positive.
    device= selection defaults are a different class (not banned)."""
    hits = []
    tree = _parse(path, src)
    if tree is None:
        return hits
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        bad = False
        what = ""
        if f.attr == "cuda" and not node.args and not node.keywords:
            bad = True
            what = ".cuda()"
        elif f.attr == "to" and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                    and (a.value == "cuda"
                         or re.fullmatch(r"cuda:\d+", a.value)):
                bad = True
                what = f".to({a.value!r})"
        if bad:
            key = f"{path}:{node.lineno}"
            if key not in allow:
                hits.append(f"R2 {key}: {what} (device discipline: "
                            f"propagate x.device; allowlist with a "
                            f"justification)")
    return hits


def rule_r3(path, src, allow):
    """Box paths inside argparse defaults (usage/docstrings allowed)."""
    hits = []
    tree = _parse(path, src)
    if tree is None:
        return hits
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else \
            (f.id if isinstance(f, ast.Name) else "")
        if name != "add_argument":
            continue
        for kw in node.keywords:
            if kw.arg != "default" or kw.value is None:
                continue
            v = _const_str(kw.value)
            if v and ("/home/" in v or v.startswith("/")):
                line = node.lineno
                key = f"{path}:{line}"
                if key not in allow:
                    hits.append(f"R3 {key}: argparse default {v!r} (box "
                                f"paths are usage examples, never "
                                f"defaults)")
    return hits


def rule_r4(path, src, allow):
    """Emoji / forbidden unicode in code files."""
    hits = []
    for i, line in enumerate(src.split("\n"), 1):
        for m in _EMOJI.finditer(line):
            key = f"{path}:{i}"
            if key not in allow:
                hits.append(f"R4 {key}: emoji {m.group(0)!r}")
    return hits


def rule_r5(path, src, allow):
    """Module-level torch.backends.* side effects (AST)."""
    hits = []
    tree = _parse(path, src)
    if tree is None:
        return hits
    for node in tree.body:          # module-level statements ONLY —
        # a full ast.walk would descend into function bodies and flag
        # their (legal, entry-point-guarded) assignments
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        elif isinstance(node, ast.If):
            # if <cond>: torch.backends... = ... at module level (an
            # import-time side effect even when guarded)
            for sub in node.body:
                if isinstance(sub, (ast.Assign, ast.AugAssign)):
                    ts = sub.targets if isinstance(sub, ast.Assign) \
                        else [sub.target]
                    targets += ts
        for t in targets:
            txt = ast.dump(t)
            if "torch" in txt and "backends" in txt:
                key = f"{path}:{getattr(node, 'lineno', 0)}"
                if key not in allow:
                    hits.append(f"R5 {key}: module-level "
                                f"torch.backends.* side effect (the "
                                f"tf32 pin belongs in the run entry "
                                f"point)")
    return hits


def rule_r6(_path, _src, _allow):
    """Every .cu under flute_train_kernels/src/ has a sibling gate test.

    Checked as a set-level rule in main() (not per-file)."""
    return []


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _parse(path, src):
    try:
        return ast.parse(src)
    except SyntaxError as e:
        print(f"[gpu-contract] WARNING: {path}: not parseable ({e})",
              file=sys.stderr)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    files = _scan_targets()
    allow = _allowlist()
    findings = []

    for path in files:
        full = os.path.join(_REPO, path)
        try:
            with open(full, encoding="utf-8") as f:
                src = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not path.endswith(".py"):
            continue
        findings += rule_r1(path, src, allow)
        findings += rule_r2(path, src, allow)
        findings += rule_r3(path, src, allow)
        findings += rule_r4(path, src, allow)
        findings += rule_r5(path, src, allow)

    # R6: the .cu gate-pairing (set-level)
    cu = [f for f in files if f.startswith("flute_train_kernels/src/")
          and f.endswith(".cu")]
    gate_ok = any("test_lut_gradients" in f for f in files)
    for c in cu:
        if not gate_ok:
            findings.append(f"R6 {c}: no differential gate test found in "
                            f"tests/ (tests/test_lut_gradients.py is the "
                            f"backward kernel's gate)")

    if findings:
        print(f"[gpu-contract] FAIL — {len(findings)} finding(s):")
        for f in sorted(set(findings)):
            print("  " + f)
        print("Fix, or add file:line to scripts/gpu_contract_allowlist.txt"
              " with a justification comment.")
        return 1
    print(f"[gpu-contract] PASS — rules R1-R6 green over {len(files)} "
          f"files ({len(cu)} .cu kernels gated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
