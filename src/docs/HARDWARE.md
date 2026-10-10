# Hardware — the A10G deployment target

Purpose: the facts every performance claim in this repo is calibrated against.
Authority: authoritative for hardware facts; subordinate to `SPECIFICATION.md` for RAG semantics.
Status: synced from the main project @ ab78893, scoped to this repo Wv2-8 (all values verbatim — the A10G is the deployment target of the carried kernels).
Target: one AWS g5.xlarge — the **NVIDIA A10G as deployed in EC2 G5**
(24 GB, single-slot, 300 W), NOT the 150 W A10 PCIe card.

## 1. Specification

| Item | Value |
|---|---|
| GPU | NVIDIA A10G (AWS g5), GA102, SM_86 |
| SMs | **80** (320 third-gen tensor cores ÷ 4; cross-checked against CUDA core count 10,240 = 80 × 128) |
| CUDA cores | 10,240 |
| L2 cache | 6 MB |
| DRAM | 24 GB GDDR6, 384-bit, **~600 GB/s** |
| Shared memory | 100 KB/SM (dynamic smem opt-in; the streaming kernel's 42-50 KB tiles fit 2 CTAs/SM) |
| Board power | 300 W (expect ~80-90% of the power limit under decode load) |
| Visible memory | ≈22.06 GiB to the process |
| FP16/BF16 tensor, FP32-acc | 69.8 TFLOPS @ 1710 MHz boost; **~62.5 TFLOPS sustained** (~1.53 GHz) |
| FP16 tensor, FP16-acc | 139.6 TFLOPS boost (the 2x-rate — not this repo's regime; every kernel here and cuBLAS default accumulate in FP32) |

Peak-efficiency percentages in the docs use **62.5 TFLOPS** (FP32-acc
sustained). Sustained clock droop under memory load is real; lock
clocks for A/B comparisons (`src/flute_extended/tools/lock_clocks.sh`).

## 2. The memory-wall arithmetic

The decode regime (M = 1) is bandwidth-bound, and the wall is simple:

- Per-SM bandwidth share = 600/80 = **7.5 GB/s**. A kernel that leaves
  an SM without a resident CTA donates that 7.5 GB/s to nobody — which
  is why the split-K grid policy targets ≥ 160 CTAs (2 waves).
- DRAM latency through L2 ≈ 600-900 ns. To saturate 7.5 GB/s an SM
  needs ~(7.5 GB/s × 700 ns) ≈ **5.3 KB of loads in flight,
  sustained** — with 16 B/thread loads that is ~330 in-flight lanes,
  i.e. ≥ 10 warps issuing back-to-back or explicit prefetch. This is
  the number behind the double-buffered K loop design.
- L2 = 6 MB: every LUT (2-24 KB per module) and every resB (≤ 131 KB)
  is L2-resident after the first touch; the streams themselves (0.5-25
  MB per module) are not — they are the traffic.
- The crossover: below ~104 FP32-acc MAC-equivalents per streamed byte,
  every GEMM is bandwidth-bound and the ceiling is bytes/time. Dense
  FP16 moves 2.0 B/param; the deployed palettized model ~0.5-0.6 B/param
  + LUT — the format's ceiling advantage is ~3.5-4x, and the 2x target
  lives comfortably inside it.

Floors at the wall (5.38 GiB of streams per token):

| stream | at 600 GB/s | at 492 GB/s (demonstrated) |
|---|---|---|
| 5.38 GiB | 9.6 ms/token | 11.7 ms/token |
| dense ~16.9 GiB | 28.5 ms/token | — |

Dense decode at 41.6 ms/token runs at ~1.46x ITS floor; the palettized
model at 40.7 ms/token runs at ~4.2x its floor. The difference is
kernel efficiency, not physics — see [PERFORMANCE.md](PERFORMANCE.md).

## 3. sm_86 kernel engineering facts

The constraints the kernels are built against:

| Resource | sm_86 limit |
|---|---|
| Registers/thread | 255 (the streaming kernel targets 130-210; zero spills is a hard gate) |
| Shared memory/SM | 100 KB (opt-in via `cudaFuncSetAttribute`, requested once per process) |
| Max threads/SM | 1536 (2 CTAs of 128 threads × 4 warps design) |
| `mma.m16n8k16` f16 | requires SM_80+ (SM_75 is excluded from the gencode set) |
| `ld.global.nc` | legal; `evict_first` L2 hint is ILLEGAL on Ampere (SM_80-89) — not emitted |

The full worked occupancy arithmetic and the ptxas audit commands:
[BUILD.md](BUILD.md) §4; the design patterns applied from the CUTLASS
docs: `src/flute_extended/docs/CUTLASS_PATTERNS.txt`.

## 4. Other GPUs

The same kernels build for any SM_80+ part (A100 / L40 / H100) via
`FLUTE_CUDA_ARCHES` ([BUILD.md](BUILD.md)) — with adjusted clocks and
peak expectations. The routing policy's CTA targets (≥ 160 CTAs) are
A10G-calibrated (80 SMs × 2 waves); re-derive for a different SM count
before trusting a bandwidth number off-target.
