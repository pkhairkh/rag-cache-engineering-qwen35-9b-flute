# Palettization pipeline

Purpose: from a dense checkpoint to a loadable artifact directory — calibration capture, per-tensor recipe selection, codebook optimization, residual extraction, and the artifact writer.
Authority: authoritative for the artifact formats and recipe vocabulary (the contract `src/scripts/palettized_modules.py` consumes); subordinate to `SPECIFICATION.md` for RAG semantics.
Status: synced from the main project @ ab78893, scoped to this repo Wv2-8 — the producer (engine + calibration capture) is main project, not part of this repo; this repo receives the artifact directories pre-built and loads them via `src/scripts/loader.py::load_quant_model`. The full producer workflow lives in the main project repo.

The engine is `scripts/palettize_qwen3_5_9b.py` (main project — not
part of this repo); its inputs come from
`scripts/calibrate_real_text.py` (driven by `scripts/capture.py`;
main project).

## 1. Calibration

Real-text calibration over WikiText-2 train, FineWeb-Edu, or a local
file. One pass over the hooked tensors accumulates, per tensor:

- the running per-input-channel Hessian diagonal `h_k = Σ_t x_tk²`
  (always; a few MB — the weighted-Lloyd quantizer's weights);
- a retained activation sample (first `retain_rows` rows, CPU) for the
  per-tensor cosine gate;
- full Gram matrices `X^T X` when needed (GPTQ/GPTVQ assignment and the
  whitened-SVD residual both consume them), accumulated on GPU across
  capture windows with one bulk CPU checkpoint per key per window;
- the head Gram for top-level modules (`lm_head`), hooked input-side,
  accumulated across capture forwards at zero extra cost.

Memory planning is measured, not hardcoded: probe-forward batch fit and
greedy Gram windows from live free VRAM, with per-layer GC back to
baseline and OOM retry (`--oom-retries`, default hard-fail after 5 — no
silent skips, no CPU fallback).

## 2. Assignment engines (`--assign`)

| engine | palette | method |
|---|---|---|
| `lloyd` | any (2..16) | weighted Lloyd k-means on the codebook, activation-weighted; the only engine for sub-4-bit bases |
| `gptq` / `gptvq` | 16 | the GPTQ family (block-wise Hessian greedy, `--gptvq-blocksize`); refused loudly for sub-4-bit bases |

The quantization side has always been width-parameterized; the stored
width equals the palette the optimization actually uses (2/4/8/16
entries → idx1/2/3/4).

## 3. Recipes (`--recipe`)

Fixed recipes are the tuples of [QUANTIZATION.md §5](QUANTIZATION.md)
(`mixed:4,2`, `hybrid422`, `r1`, …). `--recipe auto` resolves **per
weight tensor**, in compression order:

1. Visit candidates by ascending total storage bits (single tensors:
   `(1,) (2,) (1,1) (3,) (2,1) (4,) (3,1) (2,2) (4,1) (3,2) (4,2) (4,3) (4,2,2)`;
   QKV fused: `(4,) (4,1) (4,2) (4,1,1) (4,2,2)`).
2. The **first** candidate whose calibration cosine meets `--auto-cos`
   (default 0.9995) wins.
3. None passing → the best-cos candidate wins (ties to the cheaper rung).
4. The climb into two-stream territory (palette > 16) is gated by
   **gain-per-bit**: a refinement must buy ≥ 2e-4 cosine per extra bit,
   else the resolver settles on the best composite rung.

Budgets (all hard caps, never silently exceeded):

| knob | effect |
|---|---|
| `--auto-target-bits 3.5` | per-model bits/elt ceiling; a miss is recorded (`budget_miss: true`) and warned loudly |
| `--auto-layer-budget file.json` | per-layer / per-suffix ceilings (exact name first, then longest suffix); the tighter of the two caps wins |
| `--auto-speed-priority 0..1` | demotes two-stream rungs by `2*priority` effective bits in visit order; `1.0` = trainable-only (palette ≤ 16, single-GEMM, QLoRA-ready) |
| `--auto-nmse-max` | an nMSE ceiling in addition to the cosine gate |
| `--bits-plan file.json` | pin individual tensors to fixed recipes |

Every auto run records the full decision under `"auto_decision"` in the
tensor metadata: the winning spec, bits, cosine, nMSE, the gate values,
and every evaluated candidate with its numbers — the audit trail for why
each tensor landed where it did.

Typical outcome at defaults: easy spectra at 2-3 bits, hard ones at 4-6,
model average ~3.2-3.5 bits/elt.

## 4. The residual

The whitened-SVD low-rank branch (default `--residual-auto-rank`, cap
`--max-rank-cap 128`, energy threshold `--energy-thresh`): the top-r
components of the quantization residual, folded as `resA/resB` FP16
factors. The split-K decode GEMV fuses it in the epilogue up to rank
256; higher ranks ride the two-launch fallback at decode.

## 5. Group-size policy (`--gs-policy`)

`auto` sweeps `--gs-candidates` (e.g. 2048…16) per tensor against the
quality gate, with the kernel's supported set `{16,32,64,128,256,512}`
enforced. The GS probe at startup hard-fails on a stale kernel (before
the model loads) unless `--gs-lenient`; `--gs-allow-unsupported` is the
explicit escape hatch. The deployed composition's LUT geometry is what
the loaders trust at runtime (see the stale-metadata note in
[QUANTIZATION.md §3](QUANTIZATION.md)).

## 6. The two passes

| pass | flag | covers |
|---|---|---|
| body | (default) | the 248 layer modules, interleaved capture → palettize → GC per layer |
| heads | `--palettize-heads` / `--only-heads` | `lm_head` + `embed_tokens`, the 1940-tile geometry, decoupled memory plan |

A head-only run reproduces the full run's head artifacts
byte-identically, so the passes compose without a combined rerun. The
loader merges a separate heads directory (`--heads-dir` on every eval
CLI).

Additional run modes: `--target-weights` (a named subset),
`--max-layers`, `--resume` (crash-safe incremental metadata + orphan-set
recovery), `--persist-grams` (reuse a Gram store across runs),
`--rotate` (the AWQ/rotation fold), `--seed`.

## 7. Full CLI reference (`scripts/palettize_qwen3_5_9b.py`, main project — not part of this repo)

| flag | default | meaning |
|---|---|---|
| `--output` | required | artifact directory (must be fresh without metadata) |
| `--recipe` | `r1` | fixed tuple, `hybrid422`, `mixed:…`, or `auto` |
| `--assign` | `gptvq` | `lloyd` / `gptq` / `gptvq` |
| `--auto-cos` | 0.9995 | per-tensor cosine gate |
| `--auto-target-bits` / `--auto-layer-budget` / `--auto-speed-priority` / `--auto-nmse-max` | off | the budget knobs of §3 |
| `--bits-plan` | off | per-tensor recipe overrides (JSON) |
| `--residual-rank` / `--residual-auto-rank` / `--max-rank-cap` / `--energy-thresh` | auto / 128 | residual branch |
| `--gs-policy` / `--gs-candidates` / `--gs-lenient` / `--gs-allow-unsupported` / `--gs-auto-thresh` | auto / 2048…16 | group-size selection |
| `--calib-source` / `--calib-seqs` / `--calib-seq-len` / `--calib-batch-size` / `--calib-gram-window` | fineweb / 64 / 2048 | calibration corpus shape |
| `--retain-rows` / `--awq-scale` / `--awq-alpha` | | activation sample size; AWQ fold |
| `--rotate` | on | rotation fold composition |
| `--mem-temp-mb` / `--oom-retries` / `--no-alloc-tune` | 128 / 5 | memory plan |
| `--palettize-heads` / `--only-heads` / `--target-weights` / `--max-layers` / `--resume` / `--persist-grams` | off | run modes |
| `--bcd-iters` / `--auto-workers` / `--verbose` / `--seed` | | optimizer/UX |

## 8. Quality gates in-repo

- CPU differential: the toy ladder and the auto-resolver suites (main
  project: `tests/test_toy_*.py`, `tests/test_auto_compression.py`) pin
  the resolver's decisions on fixed spectra;
- The per-tensor cosine/nMSE numbers land in `metadata.json`
  (`auto_decision` ledger) and are re-read by the eval stack;
- The greedy-equivalence + PPL end-to-end verdicts:
  [EVALUATION.md](EVALUATION.md) (currently exact-match 0 / median
  first-divergence 9 tokens / PPL +3.52% — the expected cost of
  2-4-bit palettization at this cosine gate).
