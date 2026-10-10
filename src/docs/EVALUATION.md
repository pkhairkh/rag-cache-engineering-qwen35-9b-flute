# Evaluation

Purpose: the measurement tools — the routing/bandwidth probe, the greedy equivalence eval, the PPL protocol, and this repo's RAG phase-gate harness — what each one reports and how to read it.
Authority: authoritative for measurement protocol; subordinate to `SPECIFICATION.md` for RAG semantics.
Status: synced from the main project @ ab78893, scoped to this repo Wv2-8 — the probe/verdict/PPL plane is main project (§0); the carried measurement surfaces here are `src/flute_extended/benchmark_kernel.py` and `src/rag/evals.py` (the RAG phase gates).

## 0. Scoping: what is measured where

| surface | in this repo? | what it is |
|---|---|---|
| `src/rag/evals.py` | YES | the RAG phase-gate harness (`roundtrip`/`streaming`/`margin`/`recall`/`e2e`/`ledger`; CPU boxes run `<cmd> --self-test`, the four real-data commands are GPU-box only) |
| `src/flute_extended/benchmark_kernel.py` | YES | kernel-level prefill sweeps (TFLOPS, cosine-gated) + `--decode-sweep` (M = 1..128) + `--energy` + `--graphs` |
| routing/bandwidth census | main project: `scripts/probe_decode_routing.py` — not part of this repo | §1 |
| end-to-end verdict | main project: `scripts/eval_greedy_match.py` — not part of this repo | §2 |
| standalone PPL | main project: `scripts/eval_ppl.py` — not part of this repo | §3 |
| supporting kit | main project: `scripts/measure_decode.sh`, `scripts/measure_energy.py`, `scripts/doctor.py`, `scripts/verify_gemv.py`, `scripts/check_gpu_contract.py` — not part of this repo | §4 |

The main project's eval stack produced the reference verdict numbers
below; the commands that reproduce them live there.

## 1. `scripts/probe_decode_routing.py` — the routing + bandwidth census (main project — not part of this repo)

Loads the palettized model exactly as the eval does (same loader
flags), then per `PalettizedLinear` module:

1. **ROUTE** — replicates the forward's M == 1 gate chain (merged →
   wide preference → split-K FHT/plain → dual → two-launch) without
   executing it, printing the chosen route and the reason every
   alternative was refused.
2. **TIME** — times the module's real M == 1 forward. `--timing graph`
   (default on CUDA) captures the reps into one CUDA graph and times
   replays — zero host dispatch, the number the decode graph actually
   pays. `--timing eager` includes the ~42 µs/module host floor.
3. **COMPOSITE** — times the merged launches (QKV/MLP groups) the
   decode graph actually replays.

Flags: `--artifacts-dir`, `--heads-dir`, `--residual`,
`--model Qwen/Qwen3.5-9B`, `--only <module-substring>` (a single
module, for ncu work), `--reps`, `--timing`, `--device`, `--json
<out>`.

Reading the table: [ROUTING.md](ROUTING.md) §3. The acceptance number
is the aggregate effective GB/s (gate: ≥ 300; current state:
[PERFORMANCE.md](PERFORMANCE.md) §3).

## 2. `scripts/eval_greedy_match.py` — the end-to-end verdict (main project — not part of this repo)

Greedy-decodes 32 built-in deterministic prompts (96 new tokens each)
with the dense FP16 model and the palettized model — both with KV-cache
greedy decoding (never a fixed 1-token forward), both decode-backend
`graphs` — and reports:

- **exact-match fraction** — full generated sequences identical;
- **first-divergence position** — index of the first differing token
  (mean / median / min);
- **timing** — per arm: load seconds, decode seconds, tokens/second,
  and the quant-vs-fp16 speedup ratio;
- **PPL** — WikiText-2 test perplexity for BOTH arms in the same run
  (145 windows × 2048 tokens), with an OOM ladder that halves the
  batch until a window fits;
- **VRAM** — per-phase peak (alloc / reserved).

Extra flags: `--n-prompts`, `--max-new-tokens`, `--ppl-batch`,
`--ppl-max-windows`, `--no-ppl`, `--no-awq-compensation`,
`--qlora-adapters` (trained adapter overlay), `--forward <mode>`,
`--decode-backend`, `--seed`, `--output`. The JSON report lands in
`reports/` (git-ignored).

Current expected output on the A10G
([PERFORMANCE.md](PERFORMANCE.md) §1): speedup 1.022x, PPL +3.52%,
exact-match 0 / first-divergence median 9 — the noise signature of
2-4-bit palettization at the 0.9995 cosine gate, stable across runs.

Known log noise, all benign: unauthenticated HF Hub requests (set
`HF_TOKEN` for faster config fetches), the corpus-length note (the
PPL corpus is 297k tokens vs the model's 262k context — the windows
handle it), and allocator `expandable_segments` warnings when the VRAM
headroom runs out mid-ladder (the ladder recovers by halving the
batch).

## 3. `scripts/eval_ppl.py` — the standalone PPL protocol (main project — not part of this repo)

WikiText-2 perplexity, 145 windows × 2048 tokens, batch-1 unless told
otherwise, FP32 log-softmax accumulation. Used standalone when only
quality (not speed) is needed.

## 4. The supporting tools

| Script | What it measures |
|---|---|
| `src/flute_extended/benchmark_kernel.py` | kernel-level prefill sweeps (TFLOPS, cosine-gated) + `--decode-sweep` (M = 1..128) + `--energy` (pynvml J/token) + `--graphs` |
| `scripts/measure_decode.sh` | the box measurement kit: dmon clock telemetry, clock locking, nsys capture, the FHT A/B, the narrow/wide/deep-K ncu follow-ups (main project — not part of this repo) |
| `scripts/measure_energy.py` | dense-vs-palettized energy/performance harness (main project — not part of this repo) |
| `scripts/doctor.py` | the session-settling pass: import states, path table, numerics triple-check, attention parity — run it FIRST on a new box session (main project — not part of this repo) |
| `scripts/verify_gemv.py` | box-side numerics gate: the deployed split-K GEMV vs the fp32 reference, every deployed residual rank × split regime (main project — not part of this repo) |
| `scripts/check_gpu_contract.py` | the mechanical ban list (which modules may touch CUDA where) (main project — not part of this repo) |

## 5. Interpreting results

- **Speedup vs dense** is the headline metric; it is only meaningful
  at identical prompts, token counts, and decode backends (both
  `graphs`), which the eval enforces.
- **PPL delta** is the quality currency: +3.5% at the current recipe.
  A recipe change that costs more than a few tenths of a percent here
  is usually not worth its speed.
- **exact-match 0** is expected and not a defect; first-divergence
  position is the useful statistic for tracking numerics changes (a
  load-time fold fix moved it; a kernel numerics change should not).
- **VRAM peaks** matter as a ceiling: the quant phase runs at
  20.2/22.06 GiB — new workspace has to fit under that.
- A probe bandwidth number and an eval tok/s number must agree
  arithmetic-wise (graph GEMM + non-GEMM residual = 1/tok/s); if they
  do not, the measurement is broken somewhere — check clocks and
  graph mode first.

The same reading discipline applies to this repo's `src/rag/evals.py`
gates: the CPU `--self-test` verifies the MEASUREMENT (harness plumbing,
shape checks, ledger keys), never the model quality — the real-data
modes are the GPU-box gates.
