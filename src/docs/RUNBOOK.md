# Runbook — the box workflow

The exact command sequence for a GPU box session, from a fresh clone
to a verdict. Workstation: one AWS g5.xlarge (A10G, 24 GB), Ubuntu,
CUDA 12/13, a venv.

## 0. First session on a box

```bash
# environment (once)
bash scripts/provision_env.sh                 # torch, deps, triton, ninja
cd flute_extended && python setup.py build_ext --inplace && cd ..
cd flute_train_kernels && python setup.py build_ext --inplace && cd ..

# session settling (first command of EVERY session)
python scripts/doctor.py
```

`doctor.py` settles import states, prints the dequant path table, and
runs the numerics triple-check + attention parity probe; a CPU-only
run reports the import states and says so explicitly — never a silent
no-op. Exit nonzero means fix it before anything else.

Build verification ladder (new hardware or after a compiler change):
[BUILD.md](BUILD.md) §4-5.

## 1. Palettize (once per recipe; hours)

```bash
# body pass — the 248 layer modules
python scripts/palettize_qwen3_5_9b.py \
    --output /home/ubuntu/qwen3_5_9B_palettized \
    --recipe auto --auto-cos 0.9995 \
    --gs-policy auto --gs-candidates 2048,1024,512,256,128,64,32,16 \
    --calib-source fineweb --calib-seqs 64 --calib-seq-len 2048 \
    --mem-temp-mb 128 --oom-retries 5

# heads pass — lm_head + embed_tokens (decoupled memory plan)
python scripts/palettize_qwen3_5_9b.py \
    --output /home/ubuntu/qwen3_5_9B_palettized_heads \
    --only-heads --recipe auto ...     # same calibration flags
```

The recipe/knob reference is [PALETTIZATION.md](PALETTIZATION.md). The
engine is crash-safe: `--resume` continues an interrupted run from
`metadata.json`, and an orphan-set recovery runs automatically. Watch
stdout `[TAG]` lines for the per-tensor decisions (the
`auto_decision` ledger lands in the metadata).

## 2. Probe routing and bandwidth (minutes)

```bash
python scripts/probe_decode_routing.py \
    --artifacts-dir /home/ubuntu/qwen3_5_9B_palettized \
    --heads-dir /home/ubuntu/qwen3_5_9B_palettized_heads \
    --residual --model Qwen/Qwen3.5-9B \
    --json /home/ubuntu/probe.json
```

Read: the aggregate GB/s line (gate ≥ 300), the per-module GB/s
column, and the composite section's merged bandwidths
([ROUTING.md](ROUTING.md) §3, [PERFORMANCE.md](PERFORMANCE.md) §2-4).

A/B switches for attribution — set per run, no rebuild:

```bash
FLUTE_NO_MERGE=1     python scripts/probe_decode_routing.py ...   # unmerged routes
FLUTE_NO_FHT_FUSE=1  ...
FLUTE_NO_SPLITK=1    ...
FLUTE_NO_WIDE_PREF=1 ...   # lm_head forced onto split-K
```

## 3. The end-to-end verdict (one run, all numbers)

```bash
python scripts/eval_greedy_match.py \
    --artifacts-dir /home/ubuntu/qwen3_5_9B_palettized \
    --heads-dir /home/ubuntu/qwen3_5_9B_palettized_heads \
    --residual
```

Prints the speedup, PPL delta, exact-match / first-divergence, load
times, and VRAM peaks; the JSON lands in `reports/`. Expected on the
current tree: [PERFORMANCE.md](PERFORMANCE.md) §1.

## 4. Deep measurement (when a number needs explaining)

```bash
bash scripts/measure_decode.sh          # dmon clocks, nsys, the FHT A/B

# one narrow, one wide, one deep-K module under Nsight Compute:
ncu -k "regex:flute_kernel_gemv" --set full \
    python scripts/probe_decode_routing.py <same flags> \
        --only out_proj --reps 3 --timing eager

sudo bash flute_extended/tools/lock_clocks.sh 1530 <maxmem>   # reproducible A/B
```

The ptxas audit and the ncu success gates: [BUILD.md](BUILD.md).

## 5. Training (optional path)

```bash
bash scripts/train_two_phase.sh    # the two-phase layerwise distillation driver
```

The stack: `scripts/trainer.py` (the layerwise distillation engine),
`scripts/data.py`, `scripts/loss.py`, `scripts/muon_optimizer.py`,
`scripts/qlora.py` + `scripts/qlora_gemm.py` (the fused QLoRA autograd
paths), `scripts/qlora_merge.py` (fold a trained LoRA back into the
frozen LUT artifacts), `scripts/qlora_fallback.py` (reference
materialization). Post-training: re-run §3 with `--qlora-adapters`.

## 6. Full script inventory

| Script | Role |
|---|---|
| `provision_env.sh` / `provision_env_cpu.sh` | environment bring-up (GPU / CPU boxes) |
| `doctor.py` | session-settling pass (imports, paths, numerics, attention parity) |
| `capture.py` | boundary-state capture store + CLI (the layerwise-pipeline input) |
| `calibrate_real_text.py` | real-text calibration capture (Hessians, Grams, samples) |
| `palettize_qwen3_5_9b.py` | the palettization engine (see [PALETTIZATION.md](PALETTIZATION.md)) |
| `palettized_modules.py` | the runtime routing/wiring layer (see [ROUTING.md](ROUTING.md)) |
| `modeling.py` | the Qwen3.5-9B hybrid forward on palettized modules |
| `attn_sm86.py` | SM_86 Triton flash attention (fwd + bwd), the full-attention path |
| `probe_decode_routing.py` / `eval_greedy_match.py` / `eval_ppl.py` / `eval_common.py` | the measurement plane (see [EVALUATION.md](EVALUATION.md)) |
| `verify_gemv.py` | box-side numerics gate for the split-K GEMV |
| `check_gpu_contract.py` + `gpu_contract_allowlist.txt` | the mechanical GPU-contract bans |
| `measure_decode.sh` / `measure_energy.py` | measurement kit / energy harness |
| `trainer.py` / `data.py` / `loss.py` / `muon_optimizer.py` / `train_two_phase.sh` | the training stack |
| `qlora.py` / `qlora_gemm.py` / `qlora_merge.py` / `qlora_fallback.py` | QLoRA on palettized weights |
| `generate.py` | interactive dense-vs-palettized generation comparison |
| `spectrum.py` / `sensitivity_rank.py` / `distill_rank_alloc.py` / `distill_eval.py` / `report.py` / `o1_baseline_check.py` | analysis: spectra, rank allocation, verification orchestration |
| `toy_common.py` / `toy_e5_ladder.py` / `toy_e7_recipes.py` / `toy_e8_lut2bit.py` / `toy_real_geometry.py` / `lutgrad_sim.py` | the toy/differential verification drivers behind the CPU gates |
| `vram_ledger.py` | the executable VRAM budget (git-tracked plan vs live peaks) |
| `diagnose_greedy_bug.py` | the rotation × AWQ composition proof (CPU) |

## 7. Operational warnings (what is noise, what is not)

| Message | Meaning | Action |
|---|---|---|
| `metadata group_size ... inconsistent with the LUT geometry` + `[GS] ... 132 tensor component(s)` | the stale metadata field; the loader trusts the LUT geometry | none (writer-side fix pending) |
| `expandable_segments: memory mapping failed with OOM` | the allocator ran out of headroom mid-ladder; the PPL ladder halves the batch and recovers | none unless it aborts; then lower `--ppl-batch` |
| unauthenticated HF Hub requests | no `HF_TOKEN` | `export HF_TOKEN=...` (config fetches only) |
| `Token indices sequence length ... > 262144` | the PPL corpus vs the model context | none (windowed) |
| CUDA 13.x vs torch 13.0 minor-mismatch warning | toolkit/torch skew | none in practice |
| `[flute_extended] CUTLASS: NOT FOUND` | no CUTLASS checkout | only needed for the dense baseline |
| Palettizer abort: `stale kernel` GS verdict | the loaded `_C` is not the rebuilt one | [BUILD.md](BUILD.md) §3 |
