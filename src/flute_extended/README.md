# FLUTE-Extended

Quantized GEMM for inference on NVIDIA Ampere+ GPUs: FP16 activations
against LUT-palettized weights, dequantized on the fly inside the kernel.

Target workload: the Qwen3.5-9B projection layer family on one AWS A10G
(SM_86) — weights stored as per-group b-bit codebook indices
(b ∈ {1,2,3,4}) plus a `2^b`-entry FP16 lookup table, never materialized
as FP16. The repository-level docs live one level up in `docs/`
(format: [QUANTIZATION.md](../docs/QUANTIZATION.md); kernels:
[KERNELS.md](../docs/KERNELS.md); build: [BUILD.md](../docs/BUILD.md);
performance: [PERFORMANCE.md](../docs/PERFORMANCE.md)).

## Python API

```python
import flute_extended

C = flute_extended.qgemm_per_group_lut(
    A,                 # [M, K] fp16 CUDA
    indices_blob,      # flat uint8 idxN artifact (N*K*b/8 bytes)
    lut,               # [ceil(N/group_size), 2^b] fp16
    bitwidth=4,        # 1/2/3/4; must agree with indices_layout
    group_size=32,     # {16, 32, 64, 128, 256, 512}
    backend="cutlass_streaming",
    indices_layout="idx4",   # f"idx{b}"; the kernel-consumable permutation
)   # -> [M, N] fp16,  Y = X @ W^T
```

Layouts other than `idx{b}` are refused (the kernel's internal
q_layout=0 byte order exists only as a differential-test reference,
reachable through `_C` by the test suite). Eligibility: `N % 128 == 0`
and `K % 64 == 0`; misaligned storage-offset views are staged via
`.clone()`. `example.py` runs a complete call with a built-in
correctness check.

Decode (M = 1) does not go through `qgemm_per_group_lut`: the routing
layer in `scripts/palettized_modules.py` dispatches the GEMV family
(`qgemm_gemv_*`, see [KERNELS.md](../docs/KERNELS.md)).

## Backends

| Backend | Tensor Cores | Notes |
|---------|--------------|-------|
| `naive` | no | one thread per output, FP32 accumulation; correctness golden reference |
| `optimized` | no | shared-memory LUT cache, float4 A loads, register blocking |
| `debug_simple` | yes | scalar-load differential twin of the streaming kernel; bring-up and debugging only |
| `cutlass_streaming` | yes | production kernel (FP32-acc band ~55-62 TFLOPS on A10G MLP shapes) |
| `cutlass_dense` | yes | CUTLASS dense FP16 baseline on a pre-dequantized W; benchmarking aid; needs a CUTLASS checkout (Stream-K config via `FLUTE_DENSE_STREAMK=1`) |

`backend="auto"` picks `cutlass_streaming` for palettized inputs and
`cutlass_dense` for a dense W.

## The streaming kernel

`src/kernel_streaming.cu`: a one-barrier, double-buffered pipeline for
all tile configurations — the A tile is staged by `cp.async` (16 B
copies) with tile i+1's copy issued before tile i's MMA phase; each
thread's 64-k share of packed indices is contiguous, one byte is exactly
one B-fragment register, and dequant is a single paired-LUT lookup per
byte straight into the MMA register file (FLUTE, arXiv 2407.10960
§3.1-3.2; Marlin's dequant-into-registers pattern, arXiv 2408.11743).
No shared-memory W tile, no STS/ldmatrix round trip. Fragments are fed
by non-transposed `ldmatrix.x4` on a padded (BK=32) or XOR-8 swizzled
(BK=64) layout; FP32 accumulation; vectorized FP16 epilogue; m-major
grid rasterization so consecutive blocks share W strips through L2.

Tile configurations (BM, BN, BK, GS), all 4 warps / 128 threads:

| Config | When | Shared memory (legacy / idx4) |
|--------|------|-------------------------------|
| `<128, 128, 32, 32>` | gs=32 default | 42,240 / 35,840 B |
| `< 64, 128, 64, 32>` | gs=32 deep tiles (`FLUTE_GS32_BK=64`) | 50,432 / 29,696 B |
| `< 64, 128, 64, 64>` | gs=64 | 49,920 / 27,648 B |

The LUT depends only on the N group index, staged once per block for the
whole K loop; BK is decoupled from group_size. Host wrappers validate
shapes/dtypes, handle degenerate inputs, and report launch errors with
grid/smem context.

## Source layout

```
include/flute/      mma.cuh, dequant.cuh, gemv.cuh, fht.cuh, entrypoints.h, gemv_host.h
src/                kernel TUs (one ninja job per family):
                    kernel_streaming.cu    prefill tensor-core family + dual-stream
                    kernel_gemv.cu         M=1 streamer (plain + FHT-fused)
                    kernel_gemv_splitk.cu  split-K + double-buffer decode GEMV
                    kernel_gemv_multi.cu   grouped multi-blob launch (QKV merge)
                    kernel_gemv_mlp.cu     merged gate+up + SiLU-mul epilogue
                    kernel_fht.cu          Fast Hadamard Transform
                    kernel_debug_simple.cu differential twin
                    kernel_cutlass_dense.cu CUTLASS baseline
                    bindings.cpp, gemv_host.cpp
flute_extended/     Python package: API wrapper, idxN.py, idx4.py (main project)
tools/              ncu_profile.sh, lock_clocks.sh
benchmark_kernel.py box benchmark (prefill + decode sweeps)
test_flute.py       GPU multi-backend + differential gates
test_qwen_weights.py GPU real-weights oracle
docs/               CUTLASS_PATTERNS.txt (design-pattern reference)
```

## Testing and benchmarking

```bash
python test_flute.py                              # multi-backend + 5 differential gates
python test_qwen_weights.py                       # real-weights oracle
python benchmark_kernel.py                        # L2-flushed real shapes
python benchmark_kernel.py --decode-sweep --energy --graphs
bash tools/ncu_profile.sh 4096 gate_proj          # Nsight Compute gates
sudo tools/lock_clocks.sh 1530 <maxmem>           # reproducible clocks
```

Build instructions, the ptxas register audit, and the first-silicon
bisection ladder: [docs/BUILD.md](../docs/BUILD.md). Expected bands and
profiling gates: [docs/PERFORMANCE.md](../docs/PERFORMANCE.md).
