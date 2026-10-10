/**
 * include/flute/fht.cuh
 *
 * Fast Walsh-Hadamard Transform (FHT) device primitives.
 *
 * FLUTE Extension (EXTENSION_REQUIREMENT.md, main project — not part of
 * this repo): replace the explicit K x K
 * rotation-matrix multiply of the Hadamard boundary fold
 *
 *     x_rot = x @ T,      T = blockdiag_b( H_b diag(s_b) / sqrt(b) )
 *
 * with an O(K log K) butterfly. Because T's Hadamard blocks are
 * COLUMN-scaled (T[i,j] = H[i,j] * s[j] / sqrt(b) — the production
 * _rot_matrix convention `hadamard(k) * s.view(1, k)`), the forward is
 *
 *     x @ T   =  ((x @ H) * s) / sqrt(b)      per power-of-two segment
 *
 * i.e. the sign vector multiplies the OUTPUT columns (NOT the input —
 * the requirement document's "(x*s) @ H" sketch is the transposed
 * convention; this implementation pins the repo's). The adjoint /
 * backward (H is symmetric and H @ H = b * I):
 *
 *     y @ T^T =  ((y * s) @ H) / sqrt(b)      per segment
 *
 * so gradient support is the SAME butterfly with the sign applied on
 * the input side and no sign on the output.
 *
 * The butterfly below is the classic iterative FWHT with stage length
 * doubling (len = 1, 2, ..., b/2):
 *
 *     for len in powers of 2 < b:
 *         for each group g of 2*len and offset o < len:
 *             (v[2*g*len + o], v[2*g*len + o + len])
 *                 -> (a + b, a - b)
 *
 * which reproduces the Sylvester-constructed H_b EXACTLY (same row
 * order — verified bit-level against hadamard(b) matmul in
 * tests/test_fht.py (main project)).
 *
 * Storage layout: one row of x per CUDA block; the active segment is
 * staged in dynamic shared memory as fp32 (requirement: "FP16
 * input/output, FP32 accumulation"). Block-diagonal K (production
 * down_proj K=12288 = 8192 + 4096) is handled by iterating segments
 * inside the kernel; each segment's butterfly is independent (the
 * off-diagonal blocks of T are zero).
 *
 * sm_86 budget (docs/PTX_NOTES.md section 1, main project — not part
 * of this repo): dynamic smem is
 * 4 * b_max bytes (16 KiB at K=4096, 32 KiB at the K=12288 8192-segment
 * — under the 48 KiB static-equivalent default; a > 48 KiB segment
 * opts in via cudaFuncSetAttribute at launch time). THREADS in
 * {128, 256, 512, 1024}, one row per blockIdx.x.
 */

#pragma once

#include <cstdint>

namespace flute {

// ---------------------------------------------------------------------------
// One butterfly stage over a power-of-two segment staged in shared memory.
//
// All THREADS threads of the block cooperate; each thread takes pair
// indices p = tid, tid + THREADS, ... (pair p = group g = p >> sh, offset
// o = p & (len-1), elements (2*g*len + o, 2*g*len + o + len)). The stage
// is indexed by STAGE SHIFT sh (len = 1 << sh) rather than len: the // re-read of the split-K GEMV critical path (main project:
// docs/A10G_DECODE_INVESTIGATION.md
// §12.3) found the runtime `p / len` and `p - g*len` compile to a full
// integer division + remainder (~20 SASS instructions each pair-visit on
// sm_86) because the compiler cannot prove len is a power of two — a
// per-element tax paid across every FHT butterfly in the file (the 
// fused-GEMV prologues, ~12 stages x b/2 pairs each). sh-indexing makes
// both single shift/mask instructions. __syncthreads() between stages
// remains the CALLER's duty — or use fht_block below, which owns the
// whole stage loop.
//
// Bank behaviour: stage len=1 pairs read (2t, 2t+1) — stride-2 words,
// 2-way conflict on the even/odd halves; every other stage is
// conflict-free or 2-way. At 12 stages per K=4096 row this is noise
// (the kernel is launch-latency bound at decode batch sizes; see the
// FHT benchmarks in src/docs/KERNELS.md section 5).
// ---------------------------------------------------------------------------
template <int THREADS>
__device__ __forceinline__ void fht_stage_sh(float* __restrict__ s, int b,
                                             int sh) {
    const int pairs = b >> 1;                 // pairs at this stage
    const int len = 1 << sh;                  // the power-of-two stride
    const int lmask = len - 1;
    for (int p = threadIdx.x; p < pairs; p += THREADS) {
        const int i = ((p >> sh) << (sh + 1)) + (p & lmask);
        const float a = s[i];
        const float bv = s[i + len];
        s[i] = a + bv;
        s[i + len] = a - bv;
    }
}

// ---------------------------------------------------------------------------
// The whole in-block butterfly: fht_block<THREADS>(s, b) replaces the
// open-coded `for (len...) { fht_stage; __syncthreads(); }` loop at every
// call site. Two changes vs that loop (the FHT-prologue tax fix):
//
//   1. shift/mask pair math (fht_stage_sh above — no runtime division).
//
//   2. REGISTER-LOCAL EARLY STAGES: when the segment is wide enough that
//      each thread owns a contiguous run of ELEMS = b/THREADS elements,
//      every stage with len <= ELEMS/2 keeps its pairs INSIDE one
//      thread's run (pair (i, i+len): i mod ELEMS < len <= ELEMS/2, so
//      i and i+len share the run — the classic multi-stage FWHT). Those
//      stages run on a register array with NO __syncthreads between
//      them: for K=4096 at 256 threads (ELEMS=16) the first four stages
//      (len 1/2/4/8) collapse from 4 barriers to 1, and 12 barriers
//      total drop to 9; for the K=12288 8192-segment (ELEMS=32) it is
//      5 barriers saved of 14. The barrier latency — not the butterfly
//      ALU — is the FHT prologue's dominant term at the split-K GEMV's 2-CTA/SM
//      occupancy (§12.3 phase 1), so this is the direct lever.
//
//      The register array is fixed at MAX_ELEMS (32) and every index is
//      a compile-time constant (the sh and j loops are both #pragma
//      unroll with constant trip bounds, predicated on the runtime
//      ELEMS) — no dynamic register indexing, no spill. Runs with
//      ELEMS > MAX_ELEMS or ELEMS == 0 (b < THREADS, non-divisible)
//      keep the all-global path: same semantics, still shift/mask.
//
// Same numerics as the stage loop it replaces, stage for stage, pair for
// pair (ascending len; (a+b, a-b) per pair) — bit-identical output.
// ---------------------------------------------------------------------------
template <int THREADS, int MAX_ELEMS = 32>
__device__ __forceinline__ void fht_block(float* __restrict__ s, int b) {
    static_assert(MAX_ELEMS <= 32,
                  "the local-stage unroll below is bounded at len 16 "
                  "(MAX_ELEMS 32); raise the sh bound with it");
    static_assert(MAX_ELEMS >= 2, "a single-element run has no local stage");
    const int elems = ((b % THREADS) == 0 && b >= THREADS)
        ? (b / THREADS) : 0;
    const bool use_local = (elems >= 2) && (elems <= MAX_ELEMS);
    // local stages cover sh in [0, local_sh) — len = 1..elems/2
    const int local_sh = use_local ? (__ffs(elems) - 1) : 0;

    if constexpr (MAX_ELEMS > 1) {
        if (use_local) {
            const int c = threadIdx.x * elems;   // this thread's run
            float v[MAX_ELEMS];
            #pragma unroll
            for (int j = 0; j < MAX_ELEMS; ++j)
                v[j] = (j < elems) ? s[c + j] : 0.0f;
            #pragma unroll
            for (int sh = 0; sh < 5; ++sh) {   // len 1,2,4,8,16 (<= 32/2)
                if (sh < local_sh) {
                    const int len = 1 << sh;
                    const int lmask = len - 1;
                    #pragma unroll
                    for (int j = 0; j < MAX_ELEMS / 2; ++j) {
                        if (j < (elems >> 1)) {
                            const int i = ((j >> sh) << (sh + 1))
                                        + (j & lmask);
                            const float a = v[i];
                            const float bv = v[i + len];
                            v[i] = a + bv;
                            v[i + len] = a - bv;
                        }
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < MAX_ELEMS; ++j)
                if (j < elems) s[c + j] = v[j];
            __syncthreads();   // the local stages are published
        }
    }
    // the global stages (everything with len >= elems; all stages when
    // the local path did not engage)
    for (int sh = local_sh; (1 << sh) < b; ++sh) {
        fht_stage_sh<THREADS>(s, b, sh);
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Segment table (by value — fits kernel params; K % 32 == 0 keeps every
// trailing block >= 32 so THREADS >= 32 always has work).
//
// The decomposition is the production _rot_matrix convention: DESCENDING
// powers of two off the front (12288 -> 8192 + 4096, 6144 -> 4096 +
// 2048, ...). Max 8 segments covers K up to 2^16 - 32; larger K is
// refused by the host launcher (loud TORCH_CHECK).
// ---------------------------------------------------------------------------
struct FhtSegs {
    int n;
    int off[8];
    int len[8];
};

__host__ __forceinline__ FhtSegs fht_segments(int K) {
    FhtSegs seg;
    seg.n = 0;
    int off = 0, rem = K;
    while (rem > 0 && seg.n < 8) {
        int b = 1;
        while ((b << 1) <= rem) b <<= 1;
        seg.off[seg.n] = off;
        seg.len[seg.n] = b;
        seg.n++;
        off += b;
        rem -= b;
    }
    return seg;
}

}  // namespace flute
