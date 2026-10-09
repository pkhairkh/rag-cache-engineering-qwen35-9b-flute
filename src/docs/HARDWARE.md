# HARDWARE.md — the AWS A10G, verified

Every performance number in this repo is calibrated against the actual
deployment target: the **NVIDIA A10G as deployed in AWS EC2 G5 instances**
(24 GB, single-slot, 8-pin EPS). This is NOT the 150 W A10 PCIe card.
Facts below were verified against AWS's own G5 announcement, real
`nvidia-smi` output from G5 instances, and the TechPowerUp GPU database.

## Specification

| Attribute | Value | Source |
|---|---|---|
| GPU | GA102 (Ampere, SM_86) | TechPowerUp / Modal |
| SMs | **80** (= 320 Tensor Cores = 80 RT cores / 4) | AWS News Blog (320 TC, 80 RT) |
| Board power limit | **300 W** (not the 150 W of A10 PCIe) | nvidia-smi on G5: "Power Limit : 300.00 W" |
| Memory | 24 GB GDDR6, ECC | AWS / Modal |
| Memory bandwidth | 600 GB/s | AWS / Baseten |
| Memory clock | 12.5 Gbps effective | TechPowerUp |
| Boost clock | 1710 MHz | TechPowerUp |
| Sustained TC clock | ~1.53 GHz (derived from AWS's "250 TOPS INT8") | AWS blog + arithmetic |
| L2 cache | 6 MB | GA102 family |
| Shared memory | 100 KB/SM, 99 KB max/block (SM86 class) | CUDA Ampere tuning guide |

## Throughput ceilings (80 SM — derived from GA102 per-SM rates)

Per SM per clock: FP16-acc TC 1024 ops, FP32-acc TC 512 ops, INT8 2048 ops,
FP32 CUDA 256 ops.

| Metric | @ boost 1710 MHz | @ sustained ~1.53 GHz |
|---|---|---|
| FP16 TC dense, **FP16 accumulate** | 139.6 TFLOPS | ~125 TFLOPS |
| FP16 TC dense, **FP32 accumulate** | 69.8 TFLOPS | ~62.5 TFLOPS |
| INT8 TC dense | 279 TOPS | 250 TOPS (AWS's number) |
| FP32 (CUDA cores) | 34.9 TFLOPS | ~31.3 TFLOPS |

Three consequences that the earlier benchmarking missed:

1. **The 142 TFLOPS figure in the original kernel notes is explained** — it
   is the A10G FP16-ACCUMULATE dense rate at full boost. It is NOT
   attainable by this kernel, by cuBLAS fp16 with default settings, or by
   `torch.matmul`: they all accumulate in FP32, whose honest ceiling is
   **62.5 TFLOPS sustained / ~70 at boost**. `benchmark_kernel.py` defaults
   `--peak 62.5` accordingly (use `--peak 125` only for FP16-acc kernels).
2. **The measured 60.38 TFLOPS (MLP-32) was ~97 % of the FP32-acc
   sustained peak** — the v5 streaming kernel was already near that
   roofline; the remaining prefill gap vs dense was dequant overhead and
   clock behavior, not HMMA scheduling.
3. **The 239–272 W power readings are physical**: 80–91 % of the 300 W
   limit. The earlier "impossible on a 150 W A10, must be an RTX 3090"
   hypothesis was wrong — it compared against the wrong A10 variant.

## Verify on your box

```bash
nvidia-smi -q -i 0 | grep -A4 "Power"        # Power Limit : 300.00 W
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv
deviceQuery | grep -E "multiprocessors|Shared mem"   # 80 SMs, 100 KB smem/SM
sudo tools/lock_clocks.sh status
```

For reproducible A/B numbers, lock clocks for the WHOLE comparison:
`sudo tools/lock_clocks.sh 1530 <max-mem-clock>` (suggested sustained
point — see the script header), and re-measure with
`benchmark_kernel.py --energy` for the power/energy columns.

## Decode-regime physics (unchanged, now grounded)

600 GB/s of bandwidth and 2.0 (dense) vs 0.5 (4-bit) bytes per weight put
the compute-vs-weight-stream crossover at M ≈ 62.5e12 / 600e9 ≈ 104
FP32-acc MAC-equivalents per weight byte. Below that (decode, M = 1..32),
throughput is weight-stream-bound and the 4-bit model's ceiling advantage
is the byte ratio, ~4x. That regime is what
`benchmark_kernel.py --decode-sweep` measures.
