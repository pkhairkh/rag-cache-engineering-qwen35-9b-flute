#!/usr/bin/env python3
"""scripts/vram_ledger.py — the executable §2.7 VRAM budget (W3-T02,
TASKS §3.2 rule 1 / PROPOSAL §2.7).

Every residency-bearing component of the joint training job as name
+ formula + bytes + running peak, at the box geometry. The default
parameters are CALIBRATED to reproduce the PROPOSAL §2.7 table (the
design of record — the ledger is its executable form; the watermark
telemetry on the box verifies it, never the other way).

Scope (W6-T02, the DDD record of the behavior change): the per-layer
rows are the W3-T02 §2.7 calibration; the ledger now ALSO carries the
model-scale (NOT per-layer) head artifacts — lm_head and embed_tokens
R2 two-stream artifacts at the MODEL_GEOMETRY.md §3 head arithmetic
(N=248320, K=4096, gs=64, r32: 2.034 GB fp16 → 1.034 GB hybrid per
head; PROPOSAL.md §2's VRAM table).

Usage:
  python scripts/vram_ledger.py            # print the ledger
  python scripts/vram_ledger.py --check    # exit 1 if peak > 21 GiB
  python scripts/vram_ledger.py --json     # machine-readable (RUNBOOK)
"""
from __future__ import annotations

import argparse
import json
import sys

GIB = 2 ** 30

# The box geometry (TASKS §3.2): the arithmetic anchor. The
# palettized-element and teacher-extra counts are the PROPOSAL §2.7
# calibration (a full-attention Qwen3.5-9B layer's 4-bit module set and
# its dense complement at the real artifact geometry). The head
# constants are docs/MODEL_GEOMETRY.md rows (the geometry of record,
# never restated elsewhere): §1 vocab_size 248320, §2 "lm_head /
# embed_tokens (248320, 4096) each", §3 the R2 hybrid bytes per head.
BOX = dict(
    rows_batch=16,          # packed rows per training step
    seq=2048,               # tokens per row
    hidden=4096,
    holdout_rows=77,        # the 691/77 capture split (PROPOSAL §7.2)
    palettized_elements_per_layer=216_000_000,   # sum(N*K) over the layer's modules
    teacher_extra_params_per_layer=20_000_000,   # norms, rotary, non-palettized dense
    modules_per_layer=7,
    group_size=64,          # idx4 rows per LUT group
    lut_entries=16,
    avg_lora_rank=64,       # rank-map average across the layer's modules
    avg_module_fan=12_000,  # avg (N + K) per module (the rank geometry)
    avg_module_k=6_000,     # avg module fan-in K (the row/LUT arithmetic)
    head_vocab_n=248_320,   # MODEL_GEOMETRY §1/§2: N of lm_head / embed_tokens
    residual_rank=32,       # the R2 recipe's r32 residual rank (PROPOSAL §1)
)

STEADY_TARGET_GIB = 21.0    # TASKS §3.2
HARD_CEILING_GIB = 22.35    # the g5.xlarge CUDA-visible figure


def head_artifact_bytes(p) -> int:
    """One model-scale head's R2 artifact (MODEL_GEOMETRY.md §3 +
    PROPOSAL.md §2's VRAM table): two idx4 blobs (stream-1 0.5 B/elt +
    composite stream-2 0.5 B/elt = 1 B/elt total), two LUT streams
    (2 x (N/gs) groups x 16 entries x 2 B), and the r32 resA/resB fp16
    pair (rank x (N + K) x 2 B). At N=248320, K=4096, gs=64, r32:
    1,017,118,720 + 248,320 + 16,154,624 = 1,033,521,664 B per head."""
    n, k = p["head_vocab_n"], p["hidden"]
    return (2 * (n * k // 2)
            + 2 * (n // p["group_size"]) * p["lut_entries"] * 2
            + p["residual_rank"] * (n + k) * 2)


def ledger(params=None) -> list:
    """The §2.7 component rows: (name, formula, bytes)."""
    p = dict(BOX)
    if params:
        p.update(params)
    M = p["rows_batch"] * p["seq"]
    H = p["hidden"]
    elems = p["palettized_elements_per_layer"]
    rows = [
        ("teacher layer L, fp16",
         "(palettized + extra params) x 2 B",
         (elems + p["teacher_extra_params_per_layer"]) * 2),
        # The W6-T02 two-stream upgrade of the frozen student branch:
        # the R2 recipe's per-module artifact is two idx4 blobs + two
        # LUT streams + the r32 fp16 resA/resB pair (PROPOSAL §1.2/§1.3).
        ("student frozen branch, two-stream R2 (avg layer)",
         "elements x 1 B (two idx4 blobs: 0.5 + 0.5 B/elt)"
         " + 2 x (elements/avg_K/GS) x 16 x 2 B (two LUT streams)"
         " + modules x r32 x avg_fan x 2 B (resA/resB fp16)",
         2 * (elems // 2)
         + 2 * (elems // p["avg_module_k"] // p["group_size"])
         * p["lut_entries"] * 2
         + p["modules_per_layer"] * p["residual_rank"]
         * p["avg_module_fan"] * 2),
        # The model-scale head artifacts (W6-T02): lm_head and
        # embed_tokens share the [248320, 4096] geometry and the exact
        # R2 artifact family (MODEL_GEOMETRY §3's arithmetic, mirrored
        # from PROPOSAL §2's VRAM table; embed is PROPOSAL §3, identical
        # format). Model-scale singletons — NOT per-layer rows.
        ("lm_head R2 artifact (two-stream idx4 + LUTs + r32)",
         "N x K x 1 B (two idx4 blobs) + 2 x (N/GS) x 16 x 2 B"
         " (two LUT streams) + r32 x (N + K) x 2 B (resA/resB fp16)",
         head_artifact_bytes(p)),
        ("embed_tokens R2 artifact (two-stream idx4 + LUTs + r32)",
         "N x K x 1 B (two idx4 blobs) + 2 x (N/GS) x 16 x 2 B"
         " (two LUT streams) + r32 x (N + K) x 2 B (resA/resB fp16)",
         head_artifact_bytes(p)),
        ("LoRA fp32 + AdamW moments (per layer)",
         "modules x rank x (N+K) x 4 B x 3 (param + 2 moments)",
         p["modules_per_layer"] * p["avg_lora_rank"]
         * p["avg_module_fan"] * 4 * 3),
        ("LUT fp32 masters + codebook moments (per layer)",
         "(elements / avg_K / group_size) groups x 16 x 4 B x 3",
         (elems // p["avg_module_k"] // p["group_size"])
         * p["lut_entries"] * 4 * 3),
        ("logical-index cache (N,K) int64 (per layer)",
         "palettized_elements x 8 B",
         elems * 8),
        ("transient W (N,K) fp32 per forward",
         "palettized_elements x 4 B (full-layer materialization)",
         elems * 4),
        ("activations/graph + workspace (16 rows, no ckpt)",
         "modules x (M x avg_K x 2 B saved-x + M x rank x 4 B mid)"
         " + M x hidden x 2 x 2 workspace",
         p["modules_per_layer"] * (M * p["avg_module_k"] * 2
                                   + M * p["avg_lora_rank"] * 4)
         + M * H * 2 * 2),
        ("resident eval pair (x, t) fp16",
         "2 x holdout_rows x seq x hidden x 2 B",
         2 * p["holdout_rows"] * p["seq"] * H * 2),
        ("CUDA context + allocator overhead (the §2.7 allowance)",
         "context ~0.8 GiB + fragmentation/fragment-pool slop",
         2_690_000_000),
    ]
    return rows


def peak_bytes(rows) -> int:
    """The running peak: every component is co-resident during a
    training step (the transient W overlaps the cache by design — the
    reference path materializes W FROM the cache; the model-scale head
    artifacts are singletons co-resident with every layer row)."""
    return sum(b for _, _, b in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="The executable PROPOSAL §2.7 VRAM ledger")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the predicted peak exceeds the "
                         f"{STEADY_TARGET_GIB} GiB steady-state target")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the ledger as JSON (RUNBOOK telemetry)")
    ap.add_argument("--rows-batch", type=int, default=None)
    ap.add_argument("--avg-lora-rank", type=int, default=None)
    args = ap.parse_args(argv)

    over = {}
    for k in ("rows_batch", "avg_lora_rank"):
        v = getattr(args, k)
        if v is not None:
            over[k] = v
    rows = ledger(over or None)
    peak = peak_bytes(rows)

    if args.as_json:
        doc = {
            "schema": "vram_ledger_v1",
            "geometry": {**BOX, **over},
            "components": [
                {"name": n, "formula": f, "bytes": b, "gib": round(b / GIB, 3)}
                for n, f, b in rows
            ],
            "peak_bytes": peak,
            "peak_gib": round(peak / GIB, 3),
            "steady_target_gib": STEADY_TARGET_GIB,
            "hard_ceiling_gib": HARD_CEILING_GIB,
        }
        print(json.dumps(doc, indent=2))
        return 0

    print("VRAM ledger (per-layer joint job + model-scale heads,"
          " PROPOSAL §2.7)")
    print(f"geometry: {BOX if not over else {**BOX, **over}}")
    print("-" * 72)
    for name, formula, b in rows:
        print(f"  {name:<56s} {b / GIB:7.3f} GiB")
        print(f"      = {formula}")
    print("-" * 72)
    print(f"  predicted peak{'':<38s} {peak / GIB:7.3f} GiB")
    print(f"  steady-state target{'':34s} {STEADY_TARGET_GIB:7.2f} GiB")
    print(f"  hard ceiling (g5.xlarge CUDA-visible){'':13s}"
          f"{HARD_CEILING_GIB:7.2f} GiB")
    if args.check:
        if peak > STEADY_TARGET_GIB * GIB:
            print(f"  CHECK FAILED: peak {peak / GIB:.3f} GiB > "
                  f"{STEADY_TARGET_GIB} GiB — a design bug, not a tuning "
                  f"problem (TASKS §3.2)")
            return 1
        print(f"  CHECK PASSED: peak within the steady-state target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
