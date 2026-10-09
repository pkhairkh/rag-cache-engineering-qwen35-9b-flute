/**
 * include/flute/fht.cuh
 *
 * Fast Walsh-Hadamard Transform (FHT) device primitives.
 *
 * FLUTE Extension (EXTENSION_REQUIREMENT.md): replace the explicit K x K
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
 * tests/test_fht.py).
 *
 * Storage layout: one row of x per CUDA block; the active segment is
 * staged in dynamic shared memory as fp32 (requirement: "FP16
 * input/output, FP32 accumulation"). Block-diagonal K (production
 * down_proj K=12288 = 8192 + 4096) is handled by iterating segments
 * inside the kernel; each segment's butterfly is independent (the
 * off-diagonal blocks of T are zero).
 *
 * sm_86 budget (docs/PTX_NOTES.md section 1): dynamic smem is
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
// indices p = tid, tid + THREADS, ... (pair p = group g = p / len, offset
// o = p % len, elements (2*g*len + o, 2*g*len + o + len)). __syncthreads()
// between stages is the CALLER's duty (the stage itself is
// synchronization-free).
//
// Bank behaviour: stage len=1 pairs read (2t, 2t+1) — stride-2 words,
// 2-way conflict on the even/odd halves; every other stage is
// conflict-free or 2-way. At 12 stages per K=4096 row this is noise
// (the kernel is launch-latency bound at decode batch sizes; see
// docs/FHT.md benchmarks).
// ---------------------------------------------------------------------------
template <int THREADS>
__device__ __forceinline__ void fht_stage(float* __restrict__ s, int b,
                                          int len) {
    const int pairs = b >> 1;                 // pairs at this stage
    for (int p = threadIdx.x; p < pairs; p += THREADS) {
        const int g = p / len;                // group of 2*len elements
        const int o = p - g * len;            // offset inside the group
        const int i = 2 * g * len + o;        // first element of the pair
        const float a = s[i];
        const float bv = s[i + len];
        s[i] = a + bv;
        s[i + len] = a - bv;
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
