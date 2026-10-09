# Performance

The measured decode state of the deployed model on the A10G, the time
budget it implies, and the ranked list of what is still on the table.
Every number below comes from the 2026-10-09 box run (probe + greedy
eval on the current tree); the commands that reproduce them are in
[RUNBOOK.md](RUNBOOK.md).

## 1. The headline numbers

| Metric | Dense FP16 | Palettized | Delta |
|---|---|---|---|
| Greedy decode | 24.017 tok/s | 24.557 tok/s | **1.022x** |
| Decode budget | 41.64 ms/token | 40.72 ms/token | |
| WikiText-2 PPL | 9.2495 | 9.5749 | +3.52% |
| Weight streams/token | ~16.9 GiB | 5.38 GiB | 0.32x bytes |
| Graph GEMM time/token | — | 34.74 ms | 162.4 GB/s effective |
| Non-GEMM residual | — | 5.98 ms | attention state, norms, host |
| Load time | 8.71 s | 14.76 s | |
| VRAM peak (eval) | 18.23 GiB | 20.21 GiB | allocator pressure |

The quant model wins by a hair while moving a third of the bytes: the
GEMM kernel family is running at ~1/3 of the bandwidth this GPU
demonstrates in the same run. The rest of this file is that sentence
with numbers attached.

## 2. The time budget (per token)

```
40.72 ms  total decode
 34.74    module GEMMs (CUDA-graph replay: 5.25 GiB of streams => 162.4 GB/s)
  5.98     everything else (attention state, norms, rotary, sampling)
```

The GEMM side decomposes by route (graph-timed):

| route | modules | time | bytes | effective |
|---|---|---|---|---|
| `mlp_merge` (gate+up, 32 groups) | 32 | 18.70 ms | ~2.1 GiB | 97.6-135 GB/s |
| single-module split-K (z, out, down, self-attn q/k/v/o) | 113 | 13.12 ms | ~2.66 GiB | ~110-205 GB/s |
| `qkv_merge` (24 groups) | 24 | 2.92 ms | ~0.7 GiB | 186-286 GB/s |
| wide streamer (`lm_head`) | 1 | 1.68 ms | 0.77 GiB | **492.0 GB/s** |

The `lm_head` row is the existence proof: same GPU, same run, same
artifact format — 492 GB/s. The split-K family averages 189 GB/s over
the same module census. That gap is the project.

## 3. The acceptance gate and where it stands

The kernel gate is **aggregate ≥ 300 GB/s at graph timing** (stretch
450+). Current: 207.7 GB/s over the 249-module census; **162.4 GB/s
effective in the real graph** once the merged launches (which are
slower per byte than the modules they replace — §4) are counted.
Status: **failing**, both ways of counting.

## 4. The merged-launch ledger

| merge | modules | merged | as separate | net |
|---|---|---|---|---|
| `qkv_merge` | 24 linear-attn QKV groups | 2.92 ms | 4.73 ms | **−1.81 ms (win)** |
| `mlp_merge` | 32 MLP gate+up | 18.70 ms | 9.97 ms | **+8.73 ms (regression)** |

The merged MLP kernel carries both LUT/code streams of gate+up in one
launch and measures 97-135 GB/s where the two separate split-K
launches measure 170-285 GB/s. The QKV merge — the same design with
per-segment heterogeneity — is a clean win. Attribution is one
environment variable: `FLUTE_NO_MERGE=1` ([ROUTING.md](ROUTING.md)).

## 5. The headroom ladder

Weight streams are fixed at 5.25-5.38 GiB/token; the A10G wall is
600 GB/s (lm_head demonstrates 492). With the current 5.98 ms
non-GEMM residual:

| scenario | GEMM GB/s | GEMM ms | total ms | tok/s | vs dense |
|---|---|---|---|---|---|
| today | 162.4 | 34.74 | 40.72 | 24.6 | 1.02x |
| revert/fix `mlp_merge` only | — | 26.01 | 31.99 | 31.3 | 1.30x |
| acceptance gate | 300 | 17.6 | 23.6 | 42.4 | 1.77x |
| **2x target** | ~366 | 14.4 | 20.4 | 48.0 | **2.00x** |
| lm_head's demonstrated rate | 492 | 10.7 | 16.7 | 59.9 | 2.49x |
| the 600 wall | 600 | 8.8 | 14.8 | 67.6 | 2.81x |

2x needs the split-K family at ~366 GB/s effective at the current
residual — or less if the residual shrinks (attention-state machinery,
MTP). The physics ceiling (wall + zero residual) is ~3x; with MTP on a
composed 2.3-2.5x base, 3-4x is plausible.

## 6. Prefill / batched regime (M ≥ 16)

The streaming tensor-core kernel is a different regime with its own
bands and gates (FP32-acc ceiling 62.5 TFLOPS sustained):

| Shape class | Band (A10G, L2 flushed) |
|---|---|
| gs=32 gate/up, M ≥ 2048, idx4 | 48-62 TFLOPS (measured top: 60.38) |
| gs=64 down/qkv/attn, M ≥ 2048 | 42-56 TFLOPS |
| M = 512-1024 | 35-50 TFLOPS |
| M ≤ 128 | decode regime — this file, not TFLOPS |

Correctness rows with cosine < 0.999 are invalid and must not be
reported. The ncu success gates (tensor-pipe ≥ 70%, bank conflicts
≤ 1%, ≤ 255 regs + zero spills, occupancy ≥ 50%) and the ptxas audit
are hard prerequisites: [BUILD.md](BUILD.md) §4,
`flute_extended/tools/ncu_profile.sh`.

## 7. Known cost items (open, ranked)

1. **The merged-MLP kernel regression** (+8.73 ms/token) — fix the
   kernel's stream locality or route gate+up separately; the A/B
   switch exists.
2. **Split-K family bandwidth** (189 GB/s aggregate vs 492
   demonstrated) — the deep-pipeline work: the diagnosis is
   latency-boundedness (DRAM duty cycle ~30-40%), not DRAM
   saturation; the 492 datum proves the fix is possible.
3. **Narrow modules**: the 1024-row k/v projections are the worst
   per-byte modules (88-121 GB/s, 128 CTAs); the full-attention
   layers (8 × 4 modules ≈ 2.1 ms) also have no merged route —
   the heterogeneous QKV merge covers only the linear-attention
   groups so far.
4. **Load time** 14.76 s vs 8.71 s — artifact reconstruction +
   heads merge; not a decode cost, but the box experience.
5. **VRAM pressure**: quant-phase peak 20.21/22.06 GiB; the PPL
   ladder auto-recovers via batch halving (allocator
   expandable-segments warnings appear under pressure). Anything
   that adds workspace must be checked against this ceiling.
6. **exact-match 0.0 / first-divergence median 9 tokens** — the
   expected noise signature of 2-4-bit palettization at the
   0.9995-cosine gate; not actionable.
7. **Stale `group_size` metadata** (132 components) — cosmetic
   loader warnings; writer-side fix pending
   ([QUANTIZATION.md](QUANTIZATION.md) §3).

## 8. Measurement discipline

- Lock clocks (`flute_extended/tools/lock_clocks.sh`) for A/B; a
  fresh cool GPU briefly shows boost numbers ~10% high.
- Default probe/bench mode flushes L2 between timed iterations
  (serving-realistic); `--no-flush-l2` is the optimistic bound.
- Graph timing vs eager: the graph number excludes ~42 µs/module of
  host dispatch — always state which one a number is.
- Never report a row the harness marked invalid (cosine gate), and
  never compare numbers across different flush/clock/graph modes.
