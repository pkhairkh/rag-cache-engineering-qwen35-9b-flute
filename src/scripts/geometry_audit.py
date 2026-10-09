#!/usr/bin/env python3
"""Geometry audit gate (wave ritual step V5) for docs/MODEL_GEOMETRY.md.

Part A pins the byte-level sha256 of docs/qwen3_5_9b_config.json (the pin
in MODEL_GEOMETRY §4) and parses it. Part B recomputes every geometry row
of MODEL_GEOMETRY.md §1/§2/§3 from that config -- config constants
verbatim, per-layer module shapes and element counts, backbone totals,
and the head arithmetic -- asserting each against the doc-recorded value
(rows without a config attribute are asserted as doc rows, § cited).
Part C scans the git-tracked tree for the stale constants of §5 (the
retired Qwen3/Qwen2.5 vocab and its derived tile/GB numbers).

Exit contract: 0 = green; 1 = red, with geometry failures (computed vs
expected) and stale hits printed as file:line: text. --inject-stale is
the scanner self-test: 1 = detection proven (probe file:line named,
cleanup verified), 2 = self-test broken; that arm never exits 0.

Stale-scan exclusions, and why: reports/ is the evidence shelf
(historical runs keep their era's numbers); TASKS.md is the ledger that
names the scan constants (self-reference); docs/MODEL_GEOMETRY.md hosts
the §5 correction table (the task-mandated narrative site); this script
must itself contain the pattern literals and the probe fixture
(self-reference). Stdlib only.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "docs/qwen3_5_9b_config.json"
EXPECTED_SHA256 = "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
# The retired pre-correction constants (MODEL_GEOMETRY §5) the scan hunts.
STALE_PATTERNS = ("151936", "1187 tiles", "1.246 GB", "0.633 GB")
SCAN_EXCLUDE_PREFIXES = ("reports/",)   # evidence shelf
SCAN_EXCLUDE_FILES = ("TASKS.md", "docs/MODEL_GEOMETRY.md")
SELF_PATH = "scripts/geometry_audit.py"  # the scanner's own literals
PROBE_REL = "docs/_geometry_inject_probe.md"
PROBE_TEXT = "lm_head: [151936, 4096]  # injected stale probe\n"
GB = 1_000_000_000  # the doc states GB as 1e9 bytes (2.034 GB = 2,034,237,440 B)


def check(failures, name, got, want, tol=None):
    """One geometry row: PASS line on match, collected failure otherwise.

    tol serves the §3 GB rows, which the doc rounds to three decimals.
    """
    ok = abs(got - want) <= tol if tol is not None else got == want
    if ok:
        print(f"PASS {name}")
    else:
        failures.append(f"{name}: computed {got!r}, expected {want!r}")


def check_modules(failures, mods):
    """Assert one layer's §2 module table; returns the element total."""
    for row, n, k, want_n, want_k in mods:
        check(failures, f"§2 {row} (N,K,elts)",
              (n, k, n * k), (want_n, want_k, want_n * want_k))
    return sum(n * k for _, n, k, _, _ in mods)


def audit_geometry(cfg, tie, failures):
    """Part B: recompute every §1/§2/§3 row of MODEL_GEOMETRY.md."""
    H, I, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    types = cfg["layer_types"]
    interval = cfg["full_attention_interval"]
    n_lin, n_full = types.count("linear_attention"), types.count("full_attention")

    # §1 -- config constants, verbatim (MODEL_GEOMETRY §1 table).
    sec1 = [("hidden_size", 4096), ("intermediate_size", 12288),
            ("num_hidden_layers", 32), ("vocab_size", 248320),
            ("head_dim", 256), ("num_attention_heads", 16),
            ("num_key_value_heads", 4), ("attn_output_gate", True),
            ("linear_num_key_heads", 16), ("linear_key_head_dim", 128),
            ("linear_num_value_heads", 32), ("linear_value_head_dim", 128),
            ("linear_conv_kernel_dim", 4), ("mtp_num_hidden_layers", 1),
            ("mtp_use_dedicated_embeddings", False)]
    for key, want in sec1:
        check(failures, f"§1 {key}", cfg[key], want)
    check(failures, "§1 tie_word_embeddings (top level)", tie, False)
    check(failures, "§1 layer_types linear_attention count", n_lin, 24)
    check(failures, "§1 layer_types full_attention count", n_full, 8)
    check(failures, "§1 full_attention_interval", interval, 4)
    follow_interval = all((t == "full_attention") == ((i + 1) % interval == 0)
                          for i, t in enumerate(types))
    check(failures, "§1 layer_types follow the interval", follow_interval, True)

    # §2 -- linear-attention layer: 8 palettized GEMM modules. N dims are
    # recomputed from §1 head arithmetic; in_proj_z / out_proj widths are
    # §2 doc rows (z follows the V width; no config attribute).
    kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]      # 2048
    vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]  # 4096
    lin_mods = [
        ("lin in_proj_qkv -> Q", kd, H, 2048, 4096),   # 16*128
        ("lin in_proj_qkv -> K", kd, H, 2048, 4096),   # 16*128
        ("lin in_proj_qkv -> V", vd, H, 4096, 4096),   # 32*128
        ("lin in_proj_z", vd, H, 4096, 4096),          # §2 doc row
        ("lin out_proj", vd, H, 4096, 4096),           # §2 doc row
        ("lin gate_proj", I, H, 12288, 4096),
        ("lin up_proj", I, H, 12288, 4096),
        ("lin down_proj", H, I, 4096, 12288),
    ]
    lin_total = check_modules(failures, lin_mods)
    # §2 non-palettized companions (doc rows, excluded from the layer
    # total): the in_proj low-rank pair (r=32) and the depthwise conv.
    check(failures, "§2 lin in_proj_b/in_proj_a (N,K,elts)",
          (32, H, 32 * H), (32, 4096, 131072))
    check(failures, "§2 lin conv1d (N,K,k,elts)",
          (kd, H, cfg["linear_conv_kernel_dim"],
           kd * H * cfg["linear_conv_kernel_dim"]),
          (2048, 4096, 4, 33554432))
    check(failures, "§2 lin layer palettized modules", len(lin_mods), 8)
    check(failures, "§2 lin layer palettized elements", lin_total, 218103808)

    # §2 -- full-attention layer: 7 palettized modules; q_proj packs
    # Q+gate (out = heads*head_dim*2) because attn_output_gate is true.
    q_gate = cfg["num_attention_heads"] * cfg["head_dim"] * 2
    kv = cfg["num_key_value_heads"] * cfg["head_dim"]
    full_mods = [
        ("full q_proj (Q+gate packed)", q_gate, H, 8192, 4096),
        ("full k_proj", kv, H, 1024, 4096),           # 4*256 GQA
        ("full v_proj", kv, H, 1024, 4096),
        ("full o_proj", H, H, 4096, 4096),            # §2 doc row
        ("full gate_proj", I, H, 12288, 4096),
        ("full up_proj", I, H, 12288, 4096),
        ("full down_proj", H, I, 4096, 12288),
    ]
    full_total = check_modules(failures, full_mods)
    check(failures, "§2 full layer palettized modules", len(full_mods), 7)
    check(failures, "§2 full layer palettized elements", full_total, 209715200)

    # §2 -- whole text backbone (24 linear + 8 full layers).
    check(failures, "§2 backbone palettized modules",
          len(lin_mods) * n_lin + len(full_mods) * n_full, 248)
    check(failures, "§2 backbone palettized elements",
          lin_total * n_lin + full_total * n_full, 6912212992)
    check(failures, "§2 lm_head / embed_tokens (N,K,elts)",
          (V, H, V * H), (248320, 4096, 1017118720))
    dense = 2 * V * H  # untied embed + lm_head, fp16 today
    check(failures, "§2 dense remainder elements (embed + lm_head)",
          dense, 2034237440)
    check(failures, "§2 dense remainder fp16 GB (rounded to 2 dp)",
          round(dense * 2 / GB, 2), 4.07)

    # §3 -- the head arithmetic (per lm_head / embed_tokens matrix).
    check(failures, "§3 vocab % 128 (idx4-eligible)", V % 128, 0)
    check(failures, "§3 x-tiles", V // 128, 1940)
    check(failures, "§3 vocab % 64", V % 64, 0)
    check(failures, "§3 groups at gs=64", V // 64, 3880)
    check(failures, "§3 groups at gs=32", V // 32, 7760)
    fp16 = V * H * 2
    check(failures, "§3 fp16 bytes per head", fp16, 2034237440)
    check(failures, "§3 fp16 per head (GB)", fp16 / GB, 2.034, tol=1e-3)
    idx4 = 2 * (V * H * 4 // 8)       # two 4-bit index blobs = 1 B/elt
    luts = 2 * (V // 64) * 16 * 2     # 2 blobs x 3880 groups x 16 entries x fp16
    r32 = (V * 32 + 32 * H) * 2       # rank-32 residual, fp16
    check(failures, "§3 R2 idx4 pair bytes per head", idx4, 1017118720)
    check(failures, "§3 R2 LUT bytes per head (gs=64)", luts, 248320)
    check(failures, "§3 R2 rank-32 residual bytes per head", r32, 16154624)
    hybrid = idx4 + luts + r32
    check(failures, "§3 R2 hybrid bytes per head", hybrid, 1033521664)
    check(failures, "§3 R2 hybrid per head (GB)", hybrid / GB, 1.034, tol=1e-3)
    check(failures, "§3 R2 bits/elt", round(8 + 512 * (1 / V + 1 / H), 3), 8.127)
    check(failures, "§3 full W-read fp16 (ms at 600 GB/s)",
          round(fp16 / 600e9 * 1000, 2), 3.39)
    check(failures, "§3 full W-read hybrid (ms at 600 GB/s)",
          round(hybrid / 600e9 * 1000, 2), 1.72)


def scan_stale_constants():
    """Part C: literal scan of the tracked tree for §5 stale constants.

    File set is `git ls-files`; exclusions and rationale are in the module
    docstring. Returns [(path, lineno, text)] for every hit.
    """
    listing = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
    hits = []
    for rel in listing.stdout.splitlines():
        if (rel.startswith(SCAN_EXCLUDE_PREFIXES) or rel in SCAN_EXCLUDE_FILES
                or rel == SELF_PATH):
            continue
        # errors="replace": binaries and odd encodings still get a
        # literal line-by-line check without crashing the gate.
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(pat in line for pat in STALE_PATTERNS):
                hits.append((rel, lineno, line.strip()))
    return hits


def run_audit():
    """Parts A + B + C in one pass; returns (ok, failures, hits)."""
    failures = []
    # Part A -- byte-pinned config integrity (MODEL_GEOMETRY §4).
    raw = (REPO_ROOT / CONFIG_PATH).read_bytes()
    check(failures, "A config sha256 (MODEL_GEOMETRY §4)",
          hashlib.sha256(raw).hexdigest(), EXPECTED_SHA256)
    doc = json.loads(raw)
    check(failures, "A top-level architecture",
          doc["architectures"], ["Qwen3_5ForConditionalGeneration"])
    audit_geometry(doc["text_config"], doc["tie_word_embeddings"], failures)
    # Part C -- stale-constant scan over the tracked tree.
    hits = scan_stale_constants()
    for rel, lineno, text in hits:
        print(f"{rel}:{lineno}: {text}")
    print(f"FAIL stale-scan: {len(hits)} hit(s)" if hits
          else "PASS stale-scan: 0 hits")
    for failure in failures:
        print(f"FAIL {failure}")
    total = len(failures) + len(hits)
    print(f"geometry-audit: RED ({total} failures)" if total
          else "geometry-audit: GREEN")
    return (not failures and not hits), failures, hits


def inject_and_verify():
    """--inject-stale arm: plant a stale constant, prove detection."""
    probe = REPO_ROOT / PROBE_REL
    detected, error = None, None
    try:
        probe.write_text(PROBE_TEXT)
        # Intent-to-add: the probe becomes visible to `git ls-files` (the
        # scan's file set) without staging any content.
        subprocess.run(["git", "add", "-N", PROBE_REL], cwd=REPO_ROOT,
                       check=True, capture_output=True)
        _, _, hits = run_audit()
        detected = next((f"{rel}:{ln}" for rel, ln, _ in hits
                         if rel == PROBE_REL), None)
    except Exception as exc:  # a broken self-test, never a green outcome
        error = exc
    finally:
        # Cleanup always runs: unstage the intent-to-add, delete the file,
        # then verify the probe is gone from the tracked listing.
        subprocess.run(["git", "reset", "-q", "--", PROBE_REL], cwd=REPO_ROOT,
                       capture_output=True)
        if probe.exists():
            probe.unlink()
        listed = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.split()
        cleanup_ok = PROBE_REL not in listed
    if error is None and detected and cleanup_ok:
        print(f"inject-stale: DETECTED at {detected}; cleanup verified")
        return 1
    print(f"inject-stale: BROKEN (detected={detected!r}, cleanup_ok="
          f"{cleanup_ok}, error={error!r}) -- the scanner failed its own "
          "self-test")
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Geometry audit gate for docs/MODEL_GEOMETRY.md.")
    parser.add_argument("--inject-stale", action="store_true",
                        help="plant a stale constant and prove the scan "
                             "detects it (never exits 0)")
    args = parser.parse_args(argv)
    if args.inject_stale:
        return inject_and_verify()
    try:
        ok, _, _ = run_audit()
    except Exception as exc:  # unreadable config or broken git plumbing
        print(f"geometry-audit: RED (audit error: {exc!r})")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
