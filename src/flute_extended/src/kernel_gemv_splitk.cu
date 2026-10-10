/**
 * src/kernel_gemv_splitk.cu
 *
 * The split-K GEMV — flute_kernel_gemv_splitk (split-K grid +
 * double-buffered K loop + the wide 20-pair table, residual rank
 * <= 256) — extracted from the former monolith. The kloop
 * template lives in flute/gemv.cuh (shared with the multi/MLP
 * kernels); the split-K workspace lives in src/gemv_host.cpp.
 *
 * Entry points (flute/entrypoints.h):
 *   qgemm_cutlass_gemv_splitk_stream     — the plain split-K GEMV (A pre-rotated)
 *   qgemm_cutlass_gemv_splitk_fht_stream — the FHT-fused split-K GEMV (signs/s operands)
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <algorithm>

#include "flute/mma.cuh"
#include "flute/dequant.cuh"
#include "flute/fht.cuh"
#include "flute/gemv.cuh"
#include "flute/gemv_host.h"
#include "flute/entrypoints.h"

namespace {

// split-K GEMV — flute_kernel_gemv_splitk (entries qgemm_cutlass_gemv_splitk_stream
// and qgemm_cutlass_gemv_splitk_fht_stream). The box probe
// (main project: scripts/probe_decode_routing.py) measured the plain-GEMV GEMV at ~90-110
// GB/s effective — 5-6.5x below the A10G's 600 GB/s wall (aggregate
// module-GEMM 52.57 ms/token for 5.38 GiB => 109.9 GB/s; the route split:
// dual_stream 31.17 ms / two_launch 17.31 ms / w28 4.09 ms). Three structural
// causes (main project: docs/A10G_DECODE_INVESTIGATION.md section 4):
//   (1) grid = N/128 CTAs with NO K-split — k/v_proj run 8 CTAs on 80 SMs;
//   (2) no prefetch — one g-tile of code loads in flight per warp, the
//       ~600-900 ns DRAM stall is exposed 8-10 warps/SM deep;
//   (3) the routing holes — lm_head is the (4,4) pair + rank-32 residual
//       (in NEITHER pair table, over the R<=16 cap), 6 modules at GS
//       16/32, the QKV composite pairs (4,3)/(3,4)/(4,4)/(2,2) — all fall
//       to the older chain.
//
// split-K GEMV answers all of them in ONE kernel family:
//
//   * SPLIT-K GRID: grid = (N/128, SPLIT), SPLIT picked at launch so
//     (N/128)*SPLIT >= 160 (2 waves of the 80 SMs; powers of two 1..16,
//     G % (4*SPLIT) == 0 so every j4 warp keeps >= 1 g-tile). Narrow
//     modules stop donating their SMs to idle: v/k_proj 8 tiles -> 128
//     CTAs, Q/K 16 -> 256, V/Z/out 32 -> 256, down 32 (K=12288) -> 256,
//     q_proj 64 -> 256, gate/up 96 -> 192, lm_head 1940 -> SPLIT 1.
//
//   * DETERMINISTIC SPLIT REDUCTION (the decode-determinism contract,
//     generalized): every split-CTA writes its 128-row fp32 partials to
//     P[sp][n] (a cached per-device workspace, NOT atomicAdd — the float
//     atomics would make the sum order run-dependent). Each CTA then
//     __threadfence()s and takes an atomicAdd TICKET on tickets[t]; the
//     CTA that draws ticket SPLIT-1 is the FINALIZER (the CUDA
//     threadFenceReduction pattern) and folds P in FIXED split order
//     0..SPLIT-1, adds the folded residual + bias, and rounds ONCE to
//     fp16. Ticket ORDER is irrelevant — only the last arrival folds, and
//     it sums by INDEX — so every replay produces identical bits. The
//     finalizer then RESETS tickets[t] = 0 (plain store, stream-ordered
//     against the next launch), which keeps the workspace zeroed across
//     CUDA-graph replays WITHOUT a per-call memset node: the one-time
//     torch::zeros at (re)allocation is the only fill, the finalizer's
//     self-reset is the invariant. SPLIT == 1 skips the workspace
//     entirely (the direct epilogue, streaming-style).
//
//   * DOUBLE-BUFFERED K LOOP: qv_cur / qv_nxt register buffers — the
//     g+4 code chunks load BEFORE the g compute; with SPLIT-K the per-warp
//     chain is K/(256*SPLIT) g-tiles, so the pipeline plus the 2-4x deeper
//     CTA occupancy covers the DRAM latency that plain-GEMV stalled on.
//
//   * SMEM TRIM: the staged x row shrinks to the split's k-range
//     (2*Kc bytes, Kc = K/SPLIT) — down_proj's FHT-fused variant returns
//     to 2 CTAs/SM (its 57.3 KB -> ~41 KB). The FHT prologue itself
//     stays WHOLE per split-CTA (the boundary-fold butterfly is global
//     over K; each split computes the full transform and consumes only
//     its k-slice of the rotated row — O(K log K) duplicated across
//     splits, still one launch, still cheaper than the FHT+GEMV pair it
//     replaces).
//
//   * THE WIDE TABLE: every (B1, B2) pair with B1 in 1..4, B2 in 0..4
//     (20 instantiations per FHT variant — GS is RUNTIME, the design),
//     group_size 16..2048, residual rank 1..256 (the cap of 32
//     left the box census' 75 modules at R=64/128/256 on the /
//     two-launch routes — 2.19 GiB of streams at 29-170 GB/s). GS 16/32
//     switch the per-warp palette from the shfl.idx register serving
//     (GS >= 64: the warp's 64 rows lie in ONE LUT group) to a per-CTA
//     SHARED palette ([128/GS][2^B] fp16, <= 512 B — the warp's rows span
//     2-4 groups); the group index of every accumulator row is a
//     compile-time function of (v, d) in the unrolled pair loop
//     ((v*16+d*8)>>log2(GS) — d*8+7 < 16 so the 8 lanes-per-accumulator
//     never straddle a group). The residual rides 4 ranks per warp
//     per 32-rank block (warp w owns {w*4 + 32*j + rr}) and a runtime-R
//     epilogue loop — R = 64/128/256 fuses like the r32 did.
//
// Numerics contract (decode-only, like plain-GEMV): per output row the
// k-pairs accumulate in ascending-k per lane within each SPLIT, the 4
// i-lane partials butterfly (shfl_xor 1, 2), the 4 K-quarter warp-pairs
// add in fixed order red[row][0..3], and the SPLIT partials fold in
// fixed split order — deterministic on every replay, but NOT bit-identical
// to the plain-GEMV reduction tree (different fp32 association). Routed only
// at M == 1, so prefill/PPL numerics stay byte-identical to the
// two-launch route; decode greedy tokens may flip on near-ties
// (first-divergence is the aggregate to read — the same class of contract
// every decode kernel in this file carries). The FHT prologue is the // transcription verbatim (same fp32 staging, same butterflies, same
// epilogue rounding — the rotated fp16 row the K loop consumes is
// bit-identical to the standalone fht_forward output).
//
// Workspace contract: the split-K staging P ([SPLIT, N+R] fp32, // the xBf tail grew with the rank cap) and the ticket array ([N/128]
// uint) live in a per-device STATIC cache (grown geometrically,
// torch::zeros'd at (re)allocation). Single-stream sequential use (the
// decode graph replay order — documented); CUDA-graph capture reuses the
// cached addresses, and the finalizer's self-reset keeps the tickets
// zeroed across replays. P needs no zeroing (every split-CTA fully
// overwrites its 128-row slice and its R xBf slots before the ticket).
// ---------------------------------------------------------------------------

// The double-buffered, palette-parameterized K loop. kRegPal selects the
// palette serving: true = the register palette + shfl.sync.idx (GS >=
// 64, one LUT group per warp); false = the per-CTA shared palette (GS
// 16/32 — the warp's 64 rows span 2-4 groups). The g stride stays 4 (the
// j4 K-quarter warp partition, verbatim); g is LOCAL to the split.

template <int B1, int B2, bool kFht, bool kAwq>
__global__ void __launch_bounds__(256, 2)
flute_kernel_gemv_splitk(const __half*    __restrict__ A,      // [1, K] (rotated when !kFht)
    const float*     __restrict__ signs,  // (K,) fold signs (kFht only)
    const float*     __restrict__ s,      // (K,) AWQ scales (kFht+kAwq)
    const uint8_t*   __restrict__ Q1,     // stream-1 idxN blob
    const __half*    __restrict__ LUT1,   // [ceil(N/GS), 2^B1]
    const uint8_t*   __restrict__ Q2,     // stream-2 idxN blob (Q1 when B2 == 0)
    const __half*    __restrict__ LUT2,   // [ceil(N/GS), 2^B2] (LUT1 when B2 == 0)
    const __half*    __restrict__ resB,   // (R, K) fp16 or nullptr (R <= 256)
    const __half*    __restrict__ resA,   // (N, R) fp16 or nullptr
    const __half*    __restrict__ bias,   // (N,) fp16 or nullptr
    __half*          __restrict__ C,      // [1, N]
    float*           __restrict__ P,      // [SPLIT, N+R] fp32 or nullptr
    unsigned int*    __restrict__ tickets,// [N/128] or nullptr
    int N, int K, int R, int GS, int SPLIT, int pstride,
    flute::FhtSegs segs
) {
    const int t   = blockIdx.x;              // one 128-row tile per CTA
    const int sp  = blockIdx.y;              // the K-split index
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wx = warp & 1;                 // rows [wx*64, wx*64 + 64)
    const int j4 = warp >> 1;                // K-quarters: g = j4, j4+4, ...

    const int Kc = K / SPLIT;                // the split's k-range (>= 64)
    const int k0 = sp * Kc;
    const int Gc = Kc >> 6;                  // g-tiles in this split
    const int G  = K >> 6;                   // global g-tiles (blob stride)

    // ---- shared memory: [ x fp16 slice: 2*Kc | tail ] ------------------
    // tail (post-prologue region) = red[128][4] + xBf[256] + the GS<64
    // palettes + the ticket slot, 16-B aligned; kFht overlays it on the
    // DEAD END of the FHT fp32 staging (max(4*b_max, tail_need) — the // race-free arrangement: staging is dead before red/xBf/pal are
    // written, and x lives disjointly at [0, 2*Kc)). xBf grew from
    // 32 to 256 floats (the residual rank cap 32 -> 256 — the box census
    // had 75 modules at R=64/128/256 stuck on the /two-launch routes).
    const int pal_bytes = (GS < 64)
        ? (((128 / GS) << B1) * 2 + ((128 / GS) << B2) * 2) : 0;
    const int tail_need = (2048 + 1024 + pal_bytes + 4 + 15) & ~15;
    int b_max = 0;
    if constexpr (kFht) {
        for (int i = 0; i < segs.n; ++i)
            b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    }
    const int tail = kFht
        ? ((4 * b_max >= tail_need) ? 4 * b_max : tail_need) : tail_need;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* xrow = reinterpret_cast<__half*>(smem_raw);          // [Kc]
    uint4*  x2v  = reinterpret_cast<uint4*>(smem_raw);           // loader view
    unsigned char* tail_base = smem_raw + 2 * Kc + tail - tail_need;
    float*  red  = reinterpret_cast<float*>(tail_base);          // 512 floats
    float*  xBf  = red + 512;                                    // 256 floats
    __half* pal1_s = reinterpret_cast<__half*>(xBf + 256);
    __half* pal2_s = pal1_s + ((GS < 64) ? ((128 / GS) << B1) : 0);
    unsigned int* ticket_s =
        reinterpret_cast<unsigned int*>(tail_base + 2048 + 1024 + pal_bytes);
    float* fstage = reinterpret_cast<float*>(smem_raw + 2 * Kc); // kFht only

    // ---- the x row: the split's k-slice --------------------------------
    if constexpr (kFht) {
        // the prologue verbatim (fp32 staging, flute::fht_stage
        // butterflies, sign*rsqrt(b) epilogue) — except the epilogue
        // writes ONLY the [k0, k0+Kc) overlap of each segment (the other
        // splits write the rest; this CTA consumes only its slice).
        for (int sgi = 0; sgi < segs.n; ++sgi) {
            const int off = segs.off[sgi];
            const int b = segs.len[sgi];
            const float inv_sqrt_b = rsqrtf(static_cast<float>(b));

            for (int j = tid; j < b; j += 256) {
                float v = __half2float(A[off + j]);
                if constexpr (kAwq) {
                    v = v * s[off + j];
                }
                fstage[j] = v;
            }
            __syncthreads();

            flute::fht_block<256>(fstage, b);   // shift/mask + local stages

            const int lo = (off > k0) ? off : k0;
            const int hi = (off + b < k0 + Kc) ? (off + b) : (k0 + Kc);
            for (int j = lo - off + tid; j < hi - off; j += 256) {
                const float v = fstage[j] * signs[off + j] * inv_sqrt_b;
                if constexpr (kAwq) {
                    xrow[off + j - k0] = __float2half(v / s[off + j]);
                } else {
                    xrow[off + j - k0] = __float2half(v);
                }
            }
            __syncthreads();   // the slice published; staging reusable
        }
    } else {
        for (int i = tid; i < (Kc >> 3); i += 256) {
            flute::ldg_nc_evict_first_v4(x2v[i], A + (size_t)k0 + ((size_t)i << 3));
        }
        __syncthreads();
    }

    // ---- the palette: register per warp (GS >= 64) or shared (GS < 64) -
    float pal1_r = 0.0f, pal2_r = 0.0f;
    if (GS >= 64) {
        const int grow = (t * 128 + wx * 64) / GS;
        pal1_r = __half2float(
            LUT1[(size_t)grow << B1 | (lane & ((1 << B1) - 1))]);
        if constexpr (B2 > 0) {
            pal2_r = __half2float(
                LUT2[(size_t)grow << B2 | (lane & ((1 << B2) - 1))]);
        }
    } else {
        // the CTA's 128 rows cover LUT groups [t*128/GS, +128/GS): 4 rows
        // of the palette at GS=32, 8 at GS=16 (<= 512 B).
        const int ngrp = 128 / GS;
        const int g0 = (t * 128) / GS;
        for (int i = tid; i < (ngrp << B1); i += 256)
            pal1_s[i] = LUT1[((size_t)g0 << B1) + (size_t)i];
        if constexpr (B2 > 0) {
            for (int i = tid; i < (ngrp << B2); i += 256)
                pal2_s[i] = LUT2[((size_t)g0 << B2) + (size_t)i];
        }
    }

    // ---- the residual partials xBf[r] = <x_slice, resB[r][k0, k0+Kc)> --
    // R <= 256. Warp w owns the ranks {w*4 + 32*j + rr} (j = 0..): at
    // R <= 32 (j == 0 only) this is the 8-warps-x-4-ranks contract
    // verbatim; the 32-rank blocks repeat across the 8 warps for
    // R = 64/128/256 with the same per-warp K-slice stride (each rank's
    // partial is one full <x_slice, resB_row> dot — the split publishes
    // it to P for the finalizer's fixed-order fold).
    if (resB != nullptr) {
        for (int r0 = warp * 4; r0 < R; r0 += 32) {
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = r0 + rr;
                if (r < R) {
                    float p = 0.0f;
                    const __half* brow = resB + (size_t)r * K + k0;
                    for (int kk = 0; kk < Kc; kk += 32) {
                        const int k = kk + lane;
                        p = fmaf(__half2float(xrow[k]),
                                 __half2float(brow[k]), p);
                    }
                    #pragma unroll
                    for (int soff = 16; soff > 0; soff >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, soff);
                    if (lane == 0) {
                        xBf[r] = p;
                        if (SPLIT > 1) {
                            P[(size_t)sp * pstride + N + r] = p;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();       // publish the palettes AND xBf

    // ---- the K loop (double-buffered; palette-serving by GS) ----------
    float acc[8];
    #pragma unroll
    for (int a = 0; a < 8; ++a) acc[a] = 0.0f;

    const uint8_t* cb1 = Q1
        + (((size_t)t * G + (size_t)sp * Gc) * (1024 * B1))
        + (size_t)(wx * 512 + lane * 16) * B1;
    const uint8_t* cb2 = (B2 > 0)
        ? (Q2 + (((size_t)t * G + (size_t)sp * Gc) * (1024 * B2))
                + (size_t)(wx * 512 + lane * 16) * B2)
        : Q1;
    const uint32_t* x2 = reinterpret_cast<const uint32_t*>(smem_raw);
    const int gs_shift = (GS == 16) ? 4 : ((GS == 32) ? 5 : 0);

    if (GS >= 64) {
        gemv_kloop<B1, B2, true>(cb1, cb2, x2, Gc, j4, lane, wx, gs_shift,
                                  pal1_r, pal2_r, pal1_s, pal2_s, acc);
    } else {
        gemv_kloop<B1, B2, false>(cb1, cb2, x2, Gc, j4, lane, wx, gs_shift,
                                   pal1_r, pal2_r, pal1_s, pal2_s, acc);
    }

    // ---- reduction: the i-lane butterfly, then red[row][j4] ----------
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        acc[a] += __shfl_xor_sync(0xffffffffu, acc[a], 1);
        acc[a] += __shfl_xor_sync(0xffffffffu, acc[a], 2);
    }
    if ((lane & 3) == 0) {
        #pragma unroll
        for (int v = 0; v < 4; ++v)
            #pragma unroll
            for (int d = 0; d < 2; ++d)
                red[(wx * 64 + v * 16 + d * 8 + (lane >> 2)) * 4 + (warp >> 1)]
                    = acc[v * 2 + d];
    }
    __syncthreads();

    // ---- the epilogue: direct (SPLIT == 1) or fold via P --------------
    if (SPLIT == 1) {
        if (tid < 128) {
            const int n = t * 128 + tid;
            float y = red[tid * 4 + 0] + red[tid * 4 + 1] +
                      red[tid * 4 + 2] + red[tid * 4 + 3];
            if (resA != nullptr) {
                // the runtime-R epilogue (R <= 256; the unrolled
                // r < 32 loop kept every fused residual at rank 32).
                const __half* ra = resA + (size_t)n * R;
                #pragma unroll 8
                for (int r = 0; r < R; ++r) {
                    y = fmaf(xBf[r], __half2float(ra[r]), y);
                }
            }
            if (bias != nullptr) y += __half2float(bias[n]);
            C[n] = __float2half(y);
        }
    } else {
        // publish this split's row partials (fp32; every element of the
        // 128-row slice is written — P needs no zeroing)
        if (tid < 128) {
            P[(size_t)sp * pstride + (size_t)t * 128 + tid] =
                red[tid * 4 + 0] + red[tid * 4 + 1] +
                red[tid * 4 + 2] + red[tid * 4 + 3];
        }
        __syncthreads();   // all of this CTA's P writes are complete

        // the deterministic arrival ticket (CUDA threadFenceReduction):
        // fence -> atomicAdd -> the LAST arrival finalizes. Ticket ORDER
        // is irrelevant; the fold below sums by INDEX.
        if (tid == 0) {
            __threadfence();
            ticket_s[0] = (unsigned int)atomicAdd(&tickets[t], 1u);
        }
        __syncthreads();

        if (ticket_s[0] == (unsigned int)(SPLIT - 1)) {
            __threadfence();   // acquire side (belt and braces)
            // fold the xBf partials in fixed split order -> xBf smem
            if (tid < R) {
                float xb = 0.0f;
                for (int s2i = 0; s2i < SPLIT; ++s2i)
                    xb += P[(size_t)s2i * pstride + N + tid];
                xBf[tid] = xb;
            }
            __syncthreads();
            // fold the row partials in fixed split order -> C (ONE fp16
            // round, after the residual + bias — the epilogue order)
            if (tid < 128) {
                float y = 0.0f;
                for (int s2i = 0; s2i < SPLIT; ++s2i)
                    y += P[(size_t)s2i * pstride + (size_t)t * 128 + tid];
                if (resA != nullptr) {
                    // runtime R (<= 256) — same epilogue order.
                    const int n = t * 128 + tid;
                    const __half* ra = resA + (size_t)n * R;
                    #pragma unroll 8
                    for (int r = 0; r < R; ++r) {
                        y = fmaf(xBf[r], __half2float(ra[r]), y);
                    }
                }
                if (bias != nullptr) {
                    y += __half2float(bias[(size_t)t * 128 + tid]);
                }
                C[(size_t)t * 128 + tid] = __float2half(y);
            }
            // the self-reset: the ticket returns to zero for the NEXT
            // call/replay (stream-ordered — no CTA of tile t can touch
            // tickets[t] again inside THIS launch: all SPLIT have already
            // added). This is what keeps the workspace replay-invariant
            // without a per-call memset.
            if (tid == 0) tickets[t] = 0u;
        }
    }
}


//  split-K GEMV launch helper. smem = 2*Kc + max(4*b_max, tail_need) for the
// FHT variant, 2*Kc + tail_need otherwise (Kc = K/SPLIT — the staged x
// row is the split's k-slice, the -c trim).
template <int B1, int B2, bool kFht, bool kAwq>
void launch_gemv_splitk(const __half* A, const float* signs, const float* s,
    const uint8_t* Q1, const __half* LUT1,
    const uint8_t* Q2, const __half* LUT2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* C, int N, int K, int R, int GS, const flute::FhtSegs& segs
) {
    const int tiles = N / 128;
    const int G = K / 64;
    const int SPLIT = flute::gemv_pick_split(tiles, G);
    const int Kc = K / SPLIT;

    const int pal_bytes = (GS < 64)
        ? (((128 / GS) << B1) * 2 + ((128 / GS) << B2) * 2) : 0;
    const int tail_need = (2048 + 1024 + pal_bytes + 4 + 15) & ~15;
    int b_max = 0;
    if constexpr (kFht) {
        for (int i = 0; i < segs.n; ++i)
            b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    }
    const int tail = kFht
        ? ((4 * b_max >= tail_need) ? 4 * b_max : tail_need) : tail_need;
    const int smem_bytes = 2 * Kc + tail;

    TORCH_CHECK(smem_bytes <= 99 * 1024,
                "flute_kernel_gemv_splitk: smem ", smem_bytes, " B exceeds "
                "the SM_86 99 KB block limit (K=", K, ", Kc=", Kc,
                ", GS=", GS, ") — the caller's gate should have refused");

    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_gemv_splitk<B1, B2, kFht, kAwq>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=101376) "
                "failed for the split-K GEMV kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    float* P = nullptr;
    unsigned int* tickets = nullptr;
    int pstride = 0;
    if (SPLIT > 1) {
        // the xBf tail grows with the residual rank — pstride = N+R
        pstride = N + R;
        flute::GemvWorkspace& ws = flute::gemv_workspace_for(flute::gemv_current_device(), (int64_t)SPLIT * pstride, tiles);
        P = ws.P.data_ptr<float>();
        tickets =
            reinterpret_cast<unsigned int*>(ws.tickets.data_ptr<int32_t>());
    }

    dim3 block(256);
    dim3 grid(tiles, SPLIT);
    flute_kernel_gemv_splitk<B1, B2, kFht, kAwq>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, signs, s, Q1, LUT1, Q2, LUT2, resB, resA, bias, C,
            P, tickets, N, K, R, GS, SPLIT, pstride, segs);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_gemv_splitk launch failed (grid ", grid.x, "x",
                grid.y, ", 256 threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

template <int B1, int B2>
void dispatch_gemv_splitk(const __half* a, const float* signs, const float* s,
    const uint8_t* q1, const __half* l1,
    const uint8_t* q2, const __half* l2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* c, int N, int K, int R, int GS, const flute::FhtSegs& segs,
    bool want_fht, bool awq
) {
    if (want_fht) {
        if (awq) {
            launch_gemv_splitk<B1, B2, true, true>(a, signs, s, q1, l1, q2, l2, resB, resA, bias,
                c, N, K, R, GS, segs);
        } else {
            launch_gemv_splitk<B1, B2, true, false>(a, signs, s, q1, l1, q2, l2, resB, resA, bias,
                c, N, K, R, GS, segs);
        }
    } else {
        launch_gemv_splitk<B1, B2, false, false>(a, signs, s, q1, l1, q2, l2, resB, resA, bias,
            c, N, K, R, GS, segs);
    }
}

#define FLUTE_GEMV_SPLITK_PAIR(b1v, b2v)                                            \
    do {                                                                      \
        if (B1 == b1v && B2 == b2v) {                                          \
            dispatch_gemv_splitk<b1v, b2v>(                                           \
                a_p, signs_p, s_p, q1_p, l1_p, q2_p, l2_p, rb_p, ra_p, bi_p,  \
                c_p, N, K, R, (int)group_size, segs, want_fht, awq);           \
            return C;                                                          \
        }                                                                      \
    } while (0)

// The split-K GEMV impl — ONE body for both entries: signs.numel() == 0
// selects the plain variant (A is the ROTATED row, the contract);
// signs populated selects the FHT-fused variant (A is the UNROTATED row,
// the contract — s adds the AWQ compensation). All 20 width pairs,
// GS 16..2048, residual rank 1..256 (the box census' R=64/128/256
// modules join the fused epilogue).
torch::Tensor qgemm_cutlass_gemv_splitk_stream_impl(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size,
    torch::Tensor signs,
    torch::Tensor s
) {
    const int B1 = (int)bitwidth;
    const int B2 = (int)bitwidth2;
    const bool has_s2 = (B2 > 0);
    const bool want_fht = signs.numel() > 0;

    TORCH_CHECK(A.is_cuda(), "qgemm_gemv_splitk_stream: A must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda() && lut.is_cuda(),
                "qgemm_gemv_splitk_stream: stream-1 indices/lut must be CUDA");
    if (has_s2) {
        TORCH_CHECK(indices2.is_cuda() && lut2.is_cuda(),
                    "qgemm_gemv_splitk_stream: stream-2 indices/lut must be CUDA");
    }
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(indices.dtype() == torch::kUInt8,
                "stream-1 indices must be uint8");
    TORCH_CHECK(lut.dtype() == torch::kFloat16,
                "stream-1 lut must be float16");
    TORCH_CHECK(B1 >= 1 && B1 <= 4,
                "bitwidth must be 1, 2, 3 or 4");
    TORCH_CHECK(B2 >= 0 && B2 <= 4,
                "bitwidth2 must be 0 (single stream) or 1..4");
    TORCH_CHECK(!flute::fd_disabled_by_env(),
                "qgemm_gemv_splitk_stream: FLUTE_NO_FD=1 is set (the GEMV is an "
                "idxN-layout consumer)");

    TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
    const int M = A.size(0);
    const int K = A.size(1);
    TORCH_CHECK(M == 1,
                "qgemm_gemv_splitk_stream: the decode split-K GEMV serves M == 1 (got "
                "M=", M, ") — route M 2..16 through qgemm_dual_stream and "
                "larger shapes through the two-launch qgemm_per_group_lut "
                "path");
    TORCH_CHECK(K % 64 == 0,
                "qgemm_gemv_splitk_stream: K must be a multiple of 64 (the idxN "
                "blob's 64-k tiles; got K=", K, ")");
    TORCH_CHECK(group_size == 16 || group_size == 32 ||
                group_size == 64 || group_size == 128 ||
                group_size == 256 || group_size == 512 ||
                group_size == 1024 || group_size == 2048,
                "qgemm_gemv_splitk_stream: group_size must be one of "
                "16/32/64/128/256/512/1024/2048 (got ", int(group_size),
                ") — GS 16/32 use the per-CTA shared palette");

    // N from the stream-1 blob byte count (identical derivation to the
    // plain-GEMV impls; both streams share N).
    const int64_t row_bytes1 = (int64_t)(K * B1) >> 3;
    const int N = (int)(indices.numel() / row_bytes1);
    TORCH_CHECK(N > 0 && (int64_t)N * row_bytes1 == indices.numel(),
                "qgemm_gemv_splitk_stream: stream-1 byte count ", indices.numel(),
                " is not N*(K*", B1, "/8) for any N (K = ", K, ")");
    TORCH_CHECK(N % 128 == 0,
                "qgemm_gemv_splitk_stream: idxN layout requires N % 128 == 0 "
                "(got N = ", N, ")");
    if (has_s2) {
        const int64_t row_bytes2 = (int64_t)(K * B2) >> 3;
        TORCH_CHECK(indices2.numel() == (int64_t)N * row_bytes2,
                    "qgemv_gemv_splitk_stream: stream-2 byte count ",
                    indices2.numel(), " != N*K*", B2, "/8 = ",
                    (int64_t)N * row_bytes2);
        TORCH_CHECK(indices2.dtype() == torch::kUInt8,
                    "stream-2 indices must be uint8");
        TORCH_CHECK(lut2.dtype() == torch::kFloat16,
                    "stream-2 lut must be float16");
    }

    TORCH_CHECK(lut.dim() == 2 &&
                lut.size(0) == (N + group_size - 1) / group_size &&
                lut.size(1) == (1 << B1),
                "stream-1 lut must be [ceil(N/group_size), 2^bitwidth] = [",
                (N + group_size - 1) / group_size, ", ", (1 << B1), "]");
    if (has_s2) {
        TORCH_CHECK(lut2.dim() == 2 &&
                    lut2.size(0) == (N + group_size - 1) / group_size &&
                    lut2.size(1) == (1 << B2),
                    "stream-2 lut must be [ceil(N/group_size), "
                    "2^bitwidth2] = [", (N + group_size - 1) / group_size,
                    ", ", (1 << B2), "]");
    }
    TORCH_CHECK(A.is_contiguous() && indices.is_contiguous() &&
                lut.is_contiguous(),
                "A, stream-1 indices and lut must be contiguous");
    if (has_s2) {
        TORCH_CHECK(indices2.is_contiguous() && lut2.is_contiguous(),
                    "stream-2 indices/lut must be contiguous");
    }

    // The FHT contracts (mirror the entry; only when signs is given).
    const float* signs_p = nullptr;
    const float* s_p = nullptr;
    bool awq = false;
    const flute::FhtSegs segs = flute::fht_segments(K);
    if (want_fht) {
        TORCH_CHECK(signs.is_cuda() && signs.dim() == 1 &&
                    signs.scalar_type() == at::kFloat &&
                    signs.numel() == K && signs.is_contiguous(),
                    "qgemm_gemv_splitk_stream: signs must be a contiguous (K,) "
                    "float32 CUDA tensor matching A's last dim (got numel=",
                    signs.numel(), ", K=", K, ")");
        signs_p = signs.data_ptr<float>();
        awq = s.numel() > 0;
        if (awq) {
            TORCH_CHECK(s.is_cuda() && s.dim() == 1 &&
                        s.scalar_type() == at::kFloat &&
                        s.numel() == K && s.is_contiguous(),
                        "qgemv_gemv_splitk_stream: the AWQ scale s must be a "
                        "contiguous (K,) float32 CUDA tensor (got numel=",
                        s.numel(), ", K=", K, ") — pass an empty tensor "
                        "for the plain (uncompensated) rotation");
            s_p = s.data_ptr<float>();
        }
        TORCH_CHECK(segs.n > 0 &&
                    (segs.off[segs.n - 1] + segs.len[segs.n - 1]) == K,
                    "qgemv_gemv_splitk_stream: the FHT segment table does not "
                    "tile K=", K, " (n=", segs.n, ") - internal error");
    }

    // Residual + bias contracts (the plain-GEMV impls, with the R cap raised
    // to 32 — the heads' rank-32 residual).
    const bool has_res = resB.numel() > 0;
    TORCH_CHECK(has_res == (resA.numel() > 0),
                "qgemv_gemv_splitk_stream: resB and resA must be supplied "
                "together (both empty or both populated)");
    int R = 0;
    if (has_res) {
        TORCH_CHECK(resB.is_cuda() && resA.is_cuda(),
                    "residual tensors must be CUDA");
        TORCH_CHECK(resB.dtype() == torch::kFloat16 &&
                    resA.dtype() == torch::kFloat16,
                    "resB/resA must be float16 (the module's residual "
                    "contract)");
        TORCH_CHECK(resB.dim() == 2 && resA.dim() == 2,
                    "resB must be (R, K) and resA (N, R)");
        R = (int)resB.size(0);
        TORCH_CHECK(R >= 1 && R <= 256,
                    "residual rank must be 1..256 (got R=", R, ") — the "
                    " rank-256 epilogue covers every deployed residual "
                    "(the box census tops out at r256)");
        TORCH_CHECK(resB.size(1) == K && resA.size(0) == N &&
                    resA.size(1) == R,
                    "residual shape mismatch: resB (", resB.size(0), ", ",
                    resB.size(1), "), resA (", resA.size(0), ", ",
                    resA.size(1), ") for N=", N, ", K=", K);
        TORCH_CHECK(resB.is_contiguous() && resA.is_contiguous(),
                    "resB/resA must be contiguous");
    }
    if (bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat16 &&
                    bias.dim() == 1 && bias.size(0) == N &&
                    bias.is_contiguous(),
                    "bias must be a contiguous (N,) float16 CUDA tensor "
                    "(N=", N, ")");
    }

    // Degenerate shapes (mirrors the plain-GEMV impls).
    auto C = torch::empty({1, N}, A.options());
    if (N == 0 || K == 0) {
        return (K == 0 && N > 0)
            ? torch::zeros({1, N}, A.options()) : C;
    }

    // The split-K smem bound (the kernel's own formula, host-checked so
    // the launcher's TORCH_CHECK never fires in practice).
    {
        const int tiles = N / 128;
        const int G = K / 64;
        const int SPLIT = flute::gemv_pick_split(tiles, G);
        const int Kc = K / SPLIT;
        const int pal_bytes = (group_size < 64)
            ? (((128 / (int)group_size) << B1) * 2 +
               ((128 / (int)group_size) << B2) * 2) : 0;
        const int tail_need = (2048 + 1024 + pal_bytes + 4 + 15) & ~15;
        int b_max = 0;
        if (want_fht) {
            for (int i = 0; i < segs.n; ++i)
                b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
        }
        const int tail = want_fht
            ? ((4 * b_max >= tail_need) ? 4 * b_max : tail_need) : tail_need;
        TORCH_CHECK(2 * Kc + tail <= 99 * 1024,
                    "qgemm_gemv_splitk_stream: K=", K, " (SPLIT=", SPLIT,
                    ", GS=", int(group_size), ") needs ",
                    2 * Kc + tail, " B of shared memory (over the SM_86 "
                    "99 KB block limit)");
    }

    // 16 B alignment for the vectorized loads (fresh torch allocations
    // are 256 B aligned; views/slices may not be).
    if (reinterpret_cast<uintptr_t>(A.data_ptr<at::Half>()) % 16 != 0) {
        A = A.clone();
    }
    if (reinterpret_cast<uintptr_t>(indices.data_ptr<uint8_t>()) % 16 != 0) {
        indices = indices.clone();
    }
    if (has_s2 &&
        reinterpret_cast<uintptr_t>(indices2.data_ptr<uint8_t>()) % 16 != 0) {
        indices2 = indices2.clone();
    }

    const __half*  a_p  = reinterpret_cast<const __half*>(
        A.data_ptr<at::Half>());
    const uint8_t* q1_p = indices.data_ptr<uint8_t>();
    const __half*  l1_p = reinterpret_cast<const __half*>(lut.data_ptr<at::Half>());
    const uint8_t* q2_p = has_s2 ? indices2.data_ptr<uint8_t>() : q1_p;
    const __half*  l2_p = has_s2
        ? reinterpret_cast<const __half*>(lut2.data_ptr<at::Half>())
        : l1_p;
    const __half* rb_p = has_res
        ? reinterpret_cast<const __half*>(resB.data_ptr<at::Half>())
        : nullptr;
    const __half* ra_p = has_res
        ? reinterpret_cast<const __half*>(resA.data_ptr<at::Half>())
        : nullptr;
    const __half* bi_p = (bias.numel() > 0)
        ? reinterpret_cast<const __half*>(bias.data_ptr<at::Half>())
        : nullptr;
    __half* c_p = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    // The width-pair table: ALL 20 (B1, B2) combinations — the wide
    // table (the deployed artifacts' QKV composite pairs (4,3)/(3,4)/
    // (4,4)/(2,2) and the heads' (4,4) ride the same kernel).
    FLUTE_GEMV_SPLITK_PAIR(1, 0);
    FLUTE_GEMV_SPLITK_PAIR(1, 1);
    FLUTE_GEMV_SPLITK_PAIR(1, 2);
    FLUTE_GEMV_SPLITK_PAIR(1, 3);
    FLUTE_GEMV_SPLITK_PAIR(1, 4);
    FLUTE_GEMV_SPLITK_PAIR(2, 0);
    FLUTE_GEMV_SPLITK_PAIR(2, 1);
    FLUTE_GEMV_SPLITK_PAIR(2, 2);
    FLUTE_GEMV_SPLITK_PAIR(2, 3);
    FLUTE_GEMV_SPLITK_PAIR(2, 4);
    FLUTE_GEMV_SPLITK_PAIR(3, 0);
    FLUTE_GEMV_SPLITK_PAIR(3, 1);
    FLUTE_GEMV_SPLITK_PAIR(3, 2);
    FLUTE_GEMV_SPLITK_PAIR(3, 3);
    FLUTE_GEMV_SPLITK_PAIR(3, 4);
    FLUTE_GEMV_SPLITK_PAIR(4, 0);
    FLUTE_GEMV_SPLITK_PAIR(4, 1);
    FLUTE_GEMV_SPLITK_PAIR(4, 2);
    FLUTE_GEMV_SPLITK_PAIR(4, 3);
    FLUTE_GEMV_SPLITK_PAIR(4, 4);

    TORCH_CHECK(false,
                "qgemv_gemv_splitk_stream: unsupported (bitwidth, bitwidth2) = (",
                B1, ", ", B2, ") — the compiled  table is every pair "
                "with B1 in 1..4 and B2 in 0..4; this is an internal "
                "routing error, never a deployment shape");
    return C;   // unreachable (TORCH_CHECK(false) above)
}
}  // namespace

// Public entrypoint : the split-K GEMV decode kernel — split-K grid +
// double-buffered K loop + the wide table (every (B1,B2) pair, GS
// 16..2048, residual rank <= 32). A is the ROTATED [1, K] fp16 row (the
//  contract; the explicit-rotation route above). Everything else
// mirrors qgemm_cutlass_gemv_stream; the Python wrapper
// (flute_extended/flute_extended/__init__.py: qgemm_gemv_splitk_stream)
// mirrors the gate and routes everything else to the fallback ladder.
torch::Tensor qgemm_cutlass_gemv_splitk_stream(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size
) {
    return qgemm_cutlass_gemv_splitk_stream_impl(
        A, indices, lut, bitwidth, indices2, lut2, bitwidth2,
        resB, resA, bias, group_size,
        torch::Tensor(), torch::Tensor());
}

// Public entrypoint : the FHT-fused split-K GEMV — the boundary-fold
// rotation runs as the split-CTA prologue (A is the UNROTATED raw [1, K]
// fp16 row; signs is the (K,) fp32 fold sign vector; s the (K,) fp32 AWQ
// scale vector, empty = the plain rotation). Same wide table and split-K
// design as qgemm_cutlass_gemv_splitk_stream — the contract on the // kernel. The Python wrapper (flute_extended/flute_extended/__init__.py:
// qgemm_gemv_splitk_stream's rot_signs/awq_scale parameters) mirrors the gate.
torch::Tensor qgemm_cutlass_gemv_splitk_fht_stream(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2,
    torch::Tensor lut2,
    int64_t bitwidth2,
    torch::Tensor resB,
    torch::Tensor resA,
    torch::Tensor bias,
    int64_t group_size,
    torch::Tensor signs,
    torch::Tensor s
) {
    return qgemm_cutlass_gemv_splitk_stream_impl(
        A, indices, lut, bitwidth, indices2, lut2, bitwidth2,
        resB, resA, bias, group_size, signs, s);
}
