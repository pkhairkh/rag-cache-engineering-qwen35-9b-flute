# docs/GPU_SPEC.md — the A10G hardware and performance contract

W3-T01 (TASKS.md §3, PROPOSAL §2.3/§2.7). The deployment target is
one AWS g5.xlarge (1 × NVIDIA A10G, sm_86). Every number in this file
carries a `Source:` line naming a whitepaper-shelf document (TASKS §5,
S1–S10) or a committed in-repo artifact; a number with no source does
not appear here. The arithmetic-intensity table is COMPUTED (formulas
shown, spot-checks exact).

## 1. Hardware facts (verified rows)

| Item | Value | Source |
|---|---|---|
| GPU / compute capability | NVIDIA A10G, sm_86 (Ampere GA102-class) | S2 GA102 whitepaper; S3 AWS g5 page (g5.xlarge = 1 × A10G) |
| SMs / CUDA cores | **80 SMs / 10,240 (128/SM)** — corrected 2026-10-09 (W29 audit, docs/A10G_DECODE_INVESTIGATION.md §2): the earlier "72 SMs / 9,216" was the **A10 non-G** die. The A10G carries 80 RT cores + 320 3rd-gen tensor cores (AWS News blog, g5 launch: "Each A10G GPU has 24 GB of memory, 80 RT cores and 320 third-generation Tensor Cores"); 4 tensor cores/SM on GA10x ⇒ 320/4 = **80 SMs**; 80 × 128 = 10,240. Cross-checked against flute_extended/docs/PERFORMANCE.md §1 ("80 SMs", whose 69.8 TFLOPS FP32-acc figure is exactly the 80-SM rate). Box verification: `torch.cuda.get_device_properties(0).multi_processor_count` | AWS News blog (g5/A10G launch post); PERFORMANCE.md §1 |
| Tensor cores | 3rd-gen, 4/SM → **320** | AWS News blog (A10G: 320 third-generation Tensor Cores); 80 SMs × 4 |
| L2 cache | 6 MB | TechPowerUp A10G database entry (W29 audit; GA102-class 6 MB L2) |
| FP32 (CUDA cores) | 31.2 TFLOPS | S1 (nvidia.com A10 spec: "FP32, 31,2 teraFLOPS") |
| TF32 Tensor Core (dense) | 62.5 TFLOPS | S1 ("TF32 Tensor Core, 62,5 teraFLOPS") |
| FP16/BF16 Tensor Core (dense) | 125 TFLOPS (250 sparse) | S1 ("BFLOAT16 Tensor Core, 125 teraFLOPS \| 250") |
| Memory | 24 GB GDDR6, 384-bit, ~600 GB/s, ECC | S1 (A10 datasheet mirrors: "Memory Interface Width, 384-bit. Peak Memory Bandwidth, 600 GB/s") |
| CUDA-visible memory | ≈ 22.35 GiB (24 GB decimal) | S3 (AWS g5 instance pages list "GPU Memory: 22.35 GiB") |
| Registers | 65,536 × 32-bit per SM | S4 CUDA C++ Programming Guide, compute-capability 8.6 table |
| Max threads / SM | 1,536 | S4 (CC 8.6 row; 2,048 on sm_80 — the GA10x cut) |
| Max threads / block | 1,024 | S4 |
| Shared memory | 128 KB unified L1 per SM; ≤ 100 KB per SM / 99 KB per block opt-in; 48 KB static default | S4 (CC 8.6 feature and technical-spec tables) |
| Max registers / thread | 255 (practical; 65,536/256 threads) | S4 |
| Interconnect | PCIe Gen4 ~64 GB/s | S1 (A10 spec page) |

Dropped as unverifiable from the shelf: none this pass — every §3.1 row
above traced. (The g5.xlarge host RAM/vCPU figures are irrelevant to
the contract and omitted.)

## 2. Box geometry (the arithmetic anchor; TASKS §3.2)

B = 16 packed rows × S = 2,048 tokens → M = 32,768 activation tokens
per training step; hidden H = 4,096; group size GS = 64 (idx4 rows per
LUT group); 16-entry fp16 LUTs; 32 layers; 24 palettized modules per
layer. The binding module-side (N, K) source is the box artifacts
`metadata.json` (`pmod.load_metadata(artifacts)["tensors"]`); the
per-module arithmetic below uses the task's spot-check anchor
(N = 8,192, K = 4,096) — a representative mid-geometry module.

## 3. Arithmetic-intensity table (computed)

Formulas: GEMM FLOPs = 2·M·K·N. Bytes = X + W-read + Y (fp16 = 2 B/elem,
fp32 = 4, idx4 = 0.5 B/elem + LUT 16·2 B per group). Roofline ridge =
FLOPS-rate ÷ bandwidth; a path is memory-bound when its AI (FLOP/byte)
is below the ridge. Ridges: tensor-core fp16 125e12/600e9 = 208
FLOP/B; CUDA-core fp32 31.2e12/600e9 = 52 FLOP/B.

| Path | Operands (box) | FLOPs | Bytes | AI (FLOP/B) | Bound (vs 208 / 52) |
|---|---|---|---|---|---|
| (a) `flute_extended` forward LUT-GEMM | X(32,768×4,096) fp16, W idx4 (8,192×4,096), Y fp16 | 2·M·K·N = 2.20e12 | X 2.68e8 + W 1.68e7+LUT 1.3e3 + Y 5.37e8 = 8.22e8 | 2,676 | COMPUTE-bound at the 125 TFLOPS tensor rate (17.6 ms floor); the idx4 W read is 2 % of traffic |
| (b) `flute_train_kernels` backward GEMM (grad_X = grad_Y @ W) | dY(32,768×8,192) fp16, W fp16-materialized (8,192×4,096), dX fp16 | 2.20e12 | dY 5.37e8 + W 6.71e7 + dX 2.68e8 = 8.72e8 | 2,524 | COMPUTE-bound (17.6 ms floor) |
| (c) reference gather path (fp32 W + (N,K) int64 cache + fp32 GEMM) | same shapes, W fp32, cache int64 | 2.20e12 (fp32 CUDA cores) | X 2.68e8 + W 1.34e8 + cache 2.68e8 + Y 1.07e9 = 1.74e9 | 1,264 | COMPUTE-bound on FLOPs (70.5 ms at 31.2 TFLOPS) AND the cache+W materialization moves 4.0e8 B ≈ 0.67 ms of pure I/O per module forward — the §4 rule-6 wall binds this path's I/O, not the tensor GEMM's |
| (d) `attn_sm86` full attention, head_dim 256 | Q/K/V (16×2,048 × 16 heads × 256) fp16 | scores+PV ≈ 2·2·B·h·S²·D = 4.30e12 | QKV 3×1.34e8 + scores fp32 16·16·2,048²·4 = 4.30e9 (if materialized) | 1,000 (fused) / 5 (materialized scores) | COMPUTE-bound fused (34 ms floor); MEMORY-bound if the (B,H,S,S) fp32 score tensor materializes — 4.3 GB at 600 GB/s = 7.2 s — the P7/T5 eval-cap signature (measured: 0.268 GB/row, in-repo `reports/o1_baseline.log` era evidence; T5 caps eval rows) |
| (e) chunked linear attention (GatedDeltaNet chunks) | per-chunk states, K/V dims 1,024 | chunk recurrence ≈ 2·B·S·(2·k_dim + v_dim)·H ≈ 1.6e12 | states + QKV ≈ 6.7e8 | ~2,400 | COMPUTE-bound; memory-bound in its state UPDATE if per-token (16-token chunks × 32K tokens: 4,096 × state I/O 6.7e8 B ≈ 1.1 ms/chunk-pass at 600 GB/s — the F16.2 reader discipline applies) |

Spot-checks (exact): the (c) fp32 `W` transient at (N = 8,192,
K = 4,096) = 8,192·4,096·4 B = 134,217,728 B = 128 MiB ✓ (the task's
spot-check value); the (a) idx4 W read = 8,192·4,096/2 = 16,777,216 B
= 16 MiB; ridge 125e12/600e9 = 208.3 FLOP/B.

**The wall, stated precisely (a recorded refinement of TASKS §4 rule
6):** at the box geometry the GEMM-shaped ops are compute-bound
(AI ≈ 2.5–2.7 kFLOP/B ≫ the 208 ridge); the 600 GB/s wall binds
(i) the reference path's per-module W + index-cache I/O (0.67 ms × 24
modules ≈ 16 ms/layer of pure read traffic the fused kernel avoids),
(ii) any materialized (B,H,S,S) fp32 attention score tensor (7.2 s —
forbidden by the T5 eval cap), and (iii) the activation store/eval
pair residency (the §2.7 2.4 GB-pair I/O discipline). The rule's
shorthand ("a GEMM-shaped op ... is memory-bound at these sizes")
holds only for the reference (c) path and the unfused (d) — the doc
records the computed distinction rather than the shorthand.

## 4. Step-time budget (PROPOSAL §2.7, cross-checked)

Adapter-only measured: 1.41–1.45 s/step (in-repo
`reports/stage1_train.log`). FLOP floor at the box geometry: one
layer's forward+backward across 24 modules ≈ 3×2·M·K·N_mean ≈ 21.5
TFLOP/step (PROPOSAL §2.7) → 0.17 s at 125 TFLOPS; the measured 1.41 s
is ~8× the tensor-core floor (the reference dequant path + host
syncs), consistent with (c)'s computed penalty. Joint estimate 2.5–4
s/step; sweep 18–30 h. Verified on the box by WAVE B.

## 5. The dtype ladder (TASKS §3.3, binding) and budget rules (§3.2)

fp32 masters (LUT codebooks, LoRA A/B, norm gains, optimizer state) →
fp16 operands at every kernel/tensor-core boundary → fp32 accumulation
into masters. Deviations REQUIRE a row in the table below.

| Deviation | Path | Justification | Source |
|---|---|---|---|
| (none recorded) | | | |

Budget: steady-state ≤ 21 GiB; +1.5 GiB transient headroom; hard
ceiling 22.35 GiB (S3). `PYTORCH_CUDA_ALLOC_CONF=expandable_
segments:True` is a RUNBOOK runtime env, never a code default.
Gradient checkpointing FORBIDDEN (the box deadlock; PROPOSAL §9;
enforced by `scripts/check_gpu_contract.py` rule R1).
