# Performance

What to expect from the production kernel on the target hardware, how to
read the numbers, and what to do when they miss.

## 1. Reference hardware — the AWS A10G (see docs/HARDWARE.md)

GA102, SM_86, **80 SMs**, 6 MB L2, 600 GB/s GDDR6, 100 KB shared memory
per SM, **300 W board power**. The FP16 Tensor Core rates:

| Accumulate | @ 1710 MHz boost | @ ~1.53 GHz sustained |
|---|---|---|
| FP16-acc (2x rate) | 139.6 TFLOPS | ~125 TFLOPS |
| **FP32-acc (this kernel, cuBLAS default, torch.matmul)** | **69.8 TFLOPS** | **~62.5 TFLOPS** |

All efficiency percentages below use **62.5** (the FP32-acc sustained
ceiling). The historical "125 TFLOPS" figure in earlier revisions was the
FP16-ACCUMULATE rate and was never attainable by an FP32-accumulating
kernel; the previously measured 60.38 TFLOPS on MLP-32 was therefore
~97% of peak, not 48%.

## 2. Correctness prerequisites

No performance number is meaningful unless the kernel is right for that
shape. The benchmark prints an inline cosine check per data point; rows
with cosine < 0.999 are invalid and must not be reported.

| Check | Criterion | Where |
|-------|-----------|-------|
| Reference agreement | cosine > 0.999, rel max diff < 1e-3, no NaN/Inf | `test_flute.py` |
| Nibble order | bit-exact vs raw nibble values | nibble gate |
| Staging equivalence | bit-exact `cutlass_streaming == debug_simple` (`torch.equal`), incl. multi-N-block shapes N=512/1024 | differential gate |
| Layout equivalence | bit-exact `idx4 == kernel-internal legacy == debug_simple` on identical data | differential gate 2b + `flute_extended.idx4.self_test()` |
| Input guards | degenerate shapes and misaligned views handled correctly | guards gate |

Precision rationale: the kernel accumulates in FP32, same as the torch
reference, so differences are dominated by the final FP16 rounding
(<= ~1 ulp of max|C|). For unit-variance data the relative max diff stays
below 1e-3 and cosine is ~1.000000.

## 3. Expected throughput bands (A10G, L2 flushed, clocks settled)

Bands are stated against the FP32-acc sustained ceiling (62.5 TFLOPS);
with unlocked clocks a fresh cool GPU may briefly show boost-clock numbers
~10% higher. Lock clocks (tools/lock_clocks.sh) for A/B comparisons.

| Regime | Band | Notes |
|--------|------|-------|
| gs=32 gate/up_proj, M >= 2048, idx4 (production) | 48-62 TFLOPS | register-direct dequant removes the sW STS/ldmatrix round trip and deepens the A pipeline; expect parity-to-better vs cuBLAS |
| gs=32 gate/up_proj, kernel-internal legacy byte order | 45-58 TFLOPS | 72-93% of FP32-acc peak; the measured 60.38 (MLP-32, idx4-class geometry) sat at ~97% and is the realistic top |
| gs=64 down/qkv/attn, M >= 2048 | 42-56 TFLOPS | BM=64 pipeline, 64 accumulators per thread |
| M = 512-1024 | 35-50 TFLOPS | partial-wave + weight re-streaming; E4 (m-major rasterization) is the fix to measure first |
| M <= 128 (decode) | see section 8 | weight-stream bound: tok/s, not TFLOPS, is the metric |
| `debug_simple` | 15-40 TFLOPS | intentionally untuned; never a production backend |
| `naive` / `optimized` | 0.8 / 5-10 TFLOPS | references |

Target-setting guidance: **50 TFLOPS on MLP shapes (M >= 2048) is a
defensible acceptance bar** against the 62.5 ceiling; the kernel
already measured 60.38 on a good day. The idx4 path targets
parity-or-better versus the dense cuBLAS FP16 GEMM at equal shapes —
the dequant tax it removes was the entire 0.80-0.89x gap measured in the
user's prefill benchmarks.

## 4. Profiling gates (`tools/ncu_profile.sh`)

| Gate | Metric | Threshold | If it fails |
|------|--------|-----------|-------------|
| 1 | `smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | >= 70% (M >= 4096, gs=32) | high long-scoreboard stalls means DRAM-bound: inspect `lts__t_sector_hit_rate`, raise M |
| 2 | shared bank conflicts / wavefronts | <= 1% | a swizzle assumption broke - inspect the sA/sW layout math against `mma.cuh` |
| 3 | `launch__registers_per_thread` + local-memory traffic | <= 255 AND 0 bytes | spills - see DEPLOY.md remedies; do not benchmark further |
| 4 | `sm__warps_active` | >= ~50% (2 CTAs/SM design) | occupancy regression - check the smem attribute opt-in succeeded |

## 5. Register expectations (first nvcc build, `-Xptxas -v`)

| Config | Accumulators | Expected regs/thread | Risk |
|--------|--------------|----------------------|------|
| `<128,128,32,32>` idx4 (gs=32 default) | 128 f32 + 16 Q words | ~185-215 | q_cur/q_nxt double buffer replaces the sW staging; watch for spills |
| `<128,128,32,32>` kernel-internal legacy | 128 f32 | ~190-210 | the tight one; cp.async staging is what removes the 16-register A prefetch buffers |
| `<64,128,64,32>` deep tiles | 64 f32 | ~130-150 (legacy) / ~140-160 (idx4) | low |
| `<64,128,64,64>` gs=64 | 64 f32 | ~130-150 (legacy) / ~140-160 (idx4) | low |

Zero spills is a hard gate: spills crater throughput 2-3x and invalidate
every band above.

## 6. Reading a benchmark run

- Rows with cosine < 0.999 are invalid - the kernel is wrong for that
  shape; do not report the TFLOPS.
- Default mode flushes L2 between timed iterations (serving-realistic).
  `--no-flush-l2` gives the optimistic upper bound; expect +5-15%.
- Compare like with like: same M sweep, same flush mode, settled clocks
  (`nvidia-smi -q -d CLOCK` between runs if throttling is suspected).
- Against cuBLAS (`--compare-cublas`): the dense FP16 GEMM does no
  dequantization and no 4-bit weight streaming; expect it ~10-25% above
  this kernel at equal shapes. That gap is the dequant tax.

## 7. Known limitations

1. The idx4 production layout requires N % 128 == 0 and K % 64 == 0
   (all Qwen3.5-9B projection layers qualify); the kernel's internal
   legacy byte order covers everything else and is reachable only through
   the `_C` extension for differential testing.
2. The CUTLASS dense backend requires a CUTLASS checkout at build time
   and is exercised only on the target machine; the streaming production
   kernel has no such dependency.
3. PTX instruction selection and register allocation are decided by ptxas
   on the first real build; the ptxas audit and the ncu gates are the
   only arbiters there.

## 8. The decode regime (M = 1..128) — where W4 wins on all three axes

At M below the compute-vs-bandwidth crossover (~104 FP32-acc
MAC-equivalents per byte), every GEMM streams the whole weight matrix per
step and the ceiling is bandwidth: dense FP16 moves 2.0 B/param, the
palettized model 0.5 B/param + a negligible LUT, so the W4 ceiling
advantage is ~4x. Measured on v5 (user's table): dense 2860-3027 tok/s vs
palettized 2289-2662 at 512-token PREFILL — the wrong regime entirely.
In decode the same physics reverses the ordering.

What to expect from `benchmark_kernel.py --decode-sweep` (L2 flushed):

| M | W4(idx4)/dense tok/s ratio | Notes |
|---|---|---|
| 1 | 2.5-3.8x | both sides launch-bound without CUDA Graphs; use `--graphs` |
| 4-16 | 2.5-3.8x | weight-stream bound; register-direct idx4 dequant at full effect |
| 32-128 | 1.5-3x | dense starts amortizing; crossover near M~104 |

Energy: J/token should follow the same ratios (fewer bytes moved, fewer
LSU ops per byte); the `--energy` pynvml columns make it measurable on
the 300 W A10G (expect ~80-90% of the power limit under load).
Acceptance bar for the palettized model in decode: **>= 2x dense tok/s
and <= 0.7x dense J/token at M <= 16**; if the ratio is below ~2x, run
`tools/ncu_profile.sh` and check dram__bytes per invocation first (the
weights must actually be read once, not cached), then launch overhead
(CUDA Graphs), then the L2 hit rate on A.
