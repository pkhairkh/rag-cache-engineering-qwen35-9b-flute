# docs/PTX_NOTES.md — sm_86 kernel engineering facts (W3-T04)

The WAVE-5 kernel contract: every fact below cites the TASKS §5 shelf
(S4 CUDA C++ Programming Guide, S5 CUDA Best Practices Guide, S6 PTX
ISA, S7 CUTLASS efficiency docs) or an in-repo artifact (the live
kernel sources). The worked example computes the CURRENT kernel's
occupancy from its own constants — W5 kernels follow it as the
template.

## 1. The sm_86 budget table (per SM; S4 CC-8.6 tables)

| Resource | sm_86 limit | Source |
|---|---|---|
| Registers | 65,536 × 32-bit | S4 |
| Max threads / SM | 1,536 (1,024/block) | S4 |
| Shared memory / SM | 100 KiB unified with L1 (128 KiB L1 total) | S4 |
| Shared memory / block | 48 KiB static; ≤ 99 KiB opt-in (`cudaFuncSetAttribute`) | S4 |
| Practical regs / thread | 255 | S4 |
| Warp size | 32 threads (4 warps = 128 threads, the repo's block shape) | S4 |
| Global coalescing | 128 B per warp transaction (4 B × 32 lanes, or 16 B×8 via `uint4`) | S5 |

Occupancy arithmetic (S5): blocks/SM = min(regs: 65,536 ÷ (regs/thread
× threads/block), threads: 1,536 ÷ threads/block, smem: 100 KiB ÷
smem/block). Every W5 kernel shows this division.

## 2. The PTX instruction facts (S6)

| Fact | Detail | Source |
|---|---|---|
| `mma.sync.aligned.m16n8k16` (f16/bf16 operands, f32 acc) | The repo's GEMM atom: A 16×16 (row), B 8×16 (col→k), D 16×8; fragment layout via `ldmatrix.x4` for A (.trans for the backward's A=dY) and the bT address math | S6; in-repo `flute_train_kernels/include/flute/mma_bwd.cuh` (the `mma_m16n8k16_f32acc` wrapper) + `flute/mma.cuh` |
| `ldmatrix.sync.aligned.m8n8.x4` | 4 8×8 matrices per call, 32 addresses (one per lane), row-major sourcing; `.trans` for the transposed load | S6; in-repo `ldmatrix_x4_trans` |
| `cp.async.ca.shared.global [dst], [src], 16` | 16 B async copy, the double-buffer pipeline; the zero-fill variant (`cp_async_16_zfill`) predicates the source size for tail tiles | S6; in-repo `cp_async_16_zfill` |
| `PRMT` byte-permute / `LOP3` three-input logic | The idx4 nibble unpack: a 32-bit word carries 8 4-bit indices; `PRMT` selects bytes, `LOP3(AND, AND, SHR)` isolates nibbles — 2 ops per 8 indices, no table lookup | S6; in-repo `dequant_w_tile` (the nibble extract pattern) |
| fp32 `atom.add` | Global contention hazard: per-block shared-memory reduction, then ONE atomic per block | S6 + S5 |

## 3. Worked example: `kernel_backward_gemm.cu` occupancy (the W5 template)

Constants (in-repo `flute_train_kernels/src/kernel_backward_gemm.cu`:
BM=128, BN=64, BK=64, THREADS=128 (4 warps, 2×2 grid),
`__launch_bounds__(THREADS, 2)`, dynamic smem
DYN_SMEM = 2·BM·BN·2 + 2·BN·BK·2 = 32,768 + 16,384 = 48 KiB
(double-buffered sA[2][128][64] + sW[2][64][64], 2 B elements), opted
in via `cudaFuncSetAttribute`.

Occupancy at 2 CTAs/SM (the `__launch_bounds__` request):
* smem: 2 × 48 KiB = **96 KiB ≤ 100 KiB/SM** ✓ (4 KiB spare — the
  binding resource: this kernel is SMEM-limited, not thread-limited)
* threads: 2 × 128 = **256 ≤ 1,536** ✓ (17 % thread occupancy — fine
  for a smem-pipeline kernel: latency hides in cp.async, not warp
  count)
* registers: 65,536 ÷ 256 = **256 regs/thread ceiling ≥ 255** ✓ (the
  compiler cap fits; `__launch_bounds__(_, 2)` enforces it)

Verdict: 2 blocks/SM is the designed occupancy; a third CTA is
impossible (3 × 48 = 144 KiB > 100). W5 kernels replicate this
arithmetic before writing PTX.

## 4. The W5 kernel review checklist (12 yes/no items)

1. Global loads coalesced (128 B/warp; vectorized `uint4` where the
   layout allows)? [S5]
2. Shared-memory accesses conflict-free (swizzle on tile banks — the
   XOR-8 pattern is the in-repo precedent)? [S5]
3. Register budget computed (regs/thread × threads/block ≤ 65,536/SM
   × blocks)? [S4]
4. Shared-memory budget computed (blocks × smem/block ≤ 100 KiB, with
   the opt-in if > 48 KiB)? [S4]
5. Occupancy ≥ 2 blocks/SM, or a written justification (smem-pipeline
   kernels may trade warps for buffers — the worked example does)? [S5]
6. fp32 accumulation in the mma epilogue (never fp16/bf16 acc)? [S6]
7. Deterministic reduction order, or documented atomics (per-block
   reduction, one atomic per block)? [S5/S6]
8. `__launch_bounds__(THREADS, MIN_BLOCKS)` declared? [S4]
9. Vectorized loads where legal (16 B `cp.async` /
   `uint4`)? [S6]
10. Index bounds checked on every tail tile (the `zfill` predicate
    pattern)? [in-repo]
11. NaN propagation preserved (no early `isfinite` clamps inside the
    math — the tripwire's EMA canary depends on honest NaNs)? [in-repo
    contract]
12. Compile-arch guards for sm_86 (`-gencode arch=compute_86,
    code=sm_86` or the equivalent CMake guard)? [S4]
