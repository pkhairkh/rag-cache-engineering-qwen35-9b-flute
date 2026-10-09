/**
 * src/kernel_gemv_mlp.cu
 *
 * The merged gate+up split-K GEMV with the SiLU*mul epilogue (entry
 * qgemm_cutlass_gemv_mlp, flute/entrypoints.h).
 *
 * ONE split-K launch computes silu(gate(x)) * up(x): each CTA computes
 * its 128-row tile of BOTH blobs (the gate K-loop pass, then the up
 * pass — same x slice, same FHT, same palette structure) and the
 * epilogue folds BOTH rows in fixed split order and writes the product
 * with ONE fp16 round. The gate/up launches, the SiLU + mul elementwise
 * kernels and the [1, N] transients are gone.
 *
 * THE HETEROGENEOUS ROUND: the build refused 32/32 MLP groups
 * on the box ("gate/up spec mismatch", "rotation seeds differ") because
 * gate_proj and up_proj carry DIFFERENT (bitwidth, bitwidth2,
 * group_size) and different AWQ sign vectors in the deployed
 * mixed-radix artifact (e.g. layer 0: gate (4,1) vs up (1,4)). This
 * kernel takes all of those PER BLOB at RUNTIME (CTA-uniform switches
 * into the SAME gemv_kloop instantiations the split-K GEMV runs; per-blob
 * signs/s pointers in the FHT prologue). The seg-table contract is the
 * multi kernel's [2, 15] (row 0 = gate, row 1 = up; N identical).
 *
 * P layout: pstride = 2N + Rg + Ru — gate rows [0, N), up rows [N, 2N),
 * xBf_g [2N, 2N+Rg), xBf_u [2N+Rg, ...).
 *
 * Numerics (decode-only, the family contract): both row folds are the
 * split-K GEMV fixed-order chains; silu is computed in fp32 on the folded values
 * (y * sigmoid(y), expf) with ONE fp16 round at the product. Same
 * near-tie class as every decode kernel here; M > 1 never routes here
 * (PPL/prefill byte-identical).
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

#include "flute/mma.cuh"
#include "flute/fht.cuh"
#include "flute/gemv.cuh"
#include "flute/gemv_host.h"
#include "flute/entrypoints.h"

namespace {

__host__ __device__ inline int gemv_gs_shift(int gs) {
    return (gs == 16) ? 4 : ((gs == 32) ? 5 : 0);
}

// The merged smem bill (the two blobs' worst-case tail).
static inline int gemv_mlp_smem(int tiles, int K,
    const int* b1s, const int* b2s, const int* gss,  // [2] gate, up
    bool fht
) {
    const int G = K / 64;
    const int SPLIT = flute::gemv_pick_split(tiles, G);
    const int Kc = K / SPLIT;
    int pal_max = 0;
    for (int i = 0; i < 2; ++i) {
        if (gss[i] < 64) {
            const int ngrp = 128 / gss[i];
            const int pal = 2 * ((ngrp << b1s[i]) * 2
                                 + (ngrp << b2s[i]) * 2);
            pal_max = (pal > pal_max) ? pal : pal_max;
        }
    }
    const int tail_need =
        (2048 + 2048 + 1024 + 1024 + pal_max + 4 + 15) & ~15;
    int b_max = 0;
    if (fht) {
        const flute::FhtSegs segs = flute::fht_segments(K);
        for (int i = 0; i < segs.n; ++i)
            b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    }
    const int tail = fht
        ? ((4 * b_max >= tail_need) ? 4 * b_max : tail_need) : tail_need;
    return 2 * Kc + tail;
}

// ---------------------------------------------------------------------------
// The heterogeneous MLP kernel. grid = (N/128, SPLIT); 256 threads. The
// per-blob specs (b1, b2, gs, signs, s) are kernel params resolved by
// the impl from the seg table; the two K-loop passes switch into their
// own pair's kloop instantiation.
// ---------------------------------------------------------------------------
template <bool kFht>
__global__ void __launch_bounds__(256, 2)
flute_kernel_gemv_mlp(const __half*    __restrict__ A,      // [1, K] (rotated when !kFht)
    float*           __restrict__ P,      // [SPLIT, 2N+Rg+Ru] fp32 or null
    unsigned int*    __restrict__ tickets,// [N/128] or nullptr
    flute::FhtSegs segs,
    int N, int K, int Rg, int Ru, int SPLIT, int pstride,
    // ---- gate blob ----
    const uint8_t*   __restrict__ qg1,
    const __half*    __restrict__ lutg1,
    const uint8_t*   __restrict__ qg2,    // qg1 when b2g == 0
    const __half*    __restrict__ lutg2,
    const __half*    __restrict__ resBg,  // (Rg, K) fp16 or nullptr
    const __half*    __restrict__ resAg,  // (N, Rg) fp16 or nullptr
    const __half*    __restrict__ biasg,  // (N,) fp16 or nullptr
    int b1g, int b2g, int gsg,
    const float*     __restrict__ signs_g,  // (K,) or nullptr (kFht)
    const float*     __restrict__ s_g,      // (K,) or nullptr (kFht)
    // ---- up blob ----
    const uint8_t*   __restrict__ qu1,
    const __half*    __restrict__ lutu1,
    const uint8_t*   __restrict__ qu2,    // qu1 when b2u == 0
    const __half*    __restrict__ lutu2,
    const __half*    __restrict__ resBu,  // (Ru, K) fp16 or nullptr
    const __half*    __restrict__ resAu,  // (N, Ru) fp16 or nullptr
    const __half*    __restrict__ biasu,  // (N,) fp16 or nullptr
    int b1u, int b2u, int gsu,
    const float*     __restrict__ signs_u,  // (K,) or nullptr (kFht)
    const float*     __restrict__ s_u,      // (K,) or nullptr (kFht)
    // ---- output ----
    __half*          __restrict__ C        // [1, N] = silu(g) * u
) {
    const int t   = blockIdx.x;              // one 128-row tile (BOTH blobs)
    const int sp  = blockIdx.y;              // the K-split index
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wx = warp & 1;                 // rows [wx*64, wx*64 + 64)
    const int j4 = warp >> 1;                // K-quarters: g = j4, j4+4, ...

    const int Kc = K / SPLIT;
    const int k0 = sp * Kc;
    const int Gc = Kc >> 6;
    const int G  = K >> 6;

    // ---- shared memory: [ x fp16 slice: 2*Kc | tail ] ------------------
    // tail = red_g[512] + red_u[512] + xBf_g[256] + xBf_u[256] + the
    // GS<64 palettes (gate+up, sized by the blobs' OWN specs) + the
    // ticket (the fstage overlays the dead end, the split-K GEMV arrangement).
    const int palg_bytes = (gsg < 64)
        ? ((128 / gsg) << b1g) * 2 + ((128 / gsg) << b2g) * 2 : 0;
    const int palu_bytes = (gsu < 64)
        ? ((128 / gsu) << b1u) * 2 + ((128 / gsu) << b2u) * 2 : 0;
    const int pal1_bytes = (gsg < 64) ? ((128 / gsg) << b1g) * 2 : 0;
    const int pal2_bytes = (gsg < 64 && b2g > 0)
        ? ((128 / gsg) << b2g) * 2 : 0;
    const int pal3_bytes = (gsu < 64) ? ((128 / gsu) << b1u) * 2 : 0;
    const int tail_need =
        (2048 + 2048 + 1024 + 1024 + palg_bytes + palu_bytes
         + 4 + 15) & ~15;
    int b_max = 0;
    if (kFht) {
        for (int i = 0; i < segs.n; ++i)
            b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    }
    const int tail = kFht
        ? ((4 * b_max >= tail_need) ? 4 * b_max : tail_need) : tail_need;

    // when the up fold DIFFERS (different signs / AWQ scale — the
    // per-tensor seeds of the deployed artifact), the launcher doubles
    // the x region: xrow (the gate fold) + xrow_u (the up fold). The
    // up prologue reuses the fstage overlay (dead until the tail is
    // written — the same discipline as the gate pass).
    const bool up_fold_differs = kFht && (signs_u != signs_g || s_u != s_g);
    const int x_bytes = (2 * Kc) * (up_fold_differs ? 2 : 1);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* xrow = reinterpret_cast<__half*>(smem_raw);          // [Kc]
    unsigned char* tail_base = smem_raw + x_bytes + tail - tail_need;
    float*  red_g = reinterpret_cast<float*>(tail_base);         // 512 floats
    float*  red_u = red_g + 512;                                 // 512 floats
    float*  xBf_g = red_u + 512;                                 // 256 floats
    float*  xBf_u = xBf_g + 256;                                 // 256 floats
    __half* palg1_s = reinterpret_cast<__half*>(xBf_u + 256);
    __half* palg2_s = palg1_s + (pal1_bytes >> 1);
    __half* palu1_s = palg2_s + (pal2_bytes >> 1);
    __half* palu2_s = palu1_s + (pal3_bytes >> 1);
    unsigned int* ticket_s = reinterpret_cast<unsigned int*>(tail_base + 2048 + 2048 + 1024 + 1024 + palg_bytes + palu_bytes);
    float* fstage = reinterpret_cast<float*>(smem_raw + x_bytes); // kFht only

    // ---- the x rows: the split's k-slice, per blob ---------------------
    // The gate fold publishes xrow; when the up fold's operands differ,
    // a second prologue publishes xrow_u (the doubled region). Both
    // prologues reuse the SAME fstage overlay — dead until the tail is
    // written (the split-K GEMV arrangement, twice).
    gemv_fht_prologue<kFht>(A, signs_g, s_g, segs, K, Kc, k0,
                             xrow, fstage);
    __half* xrow_u = xrow;
    if (up_fold_differs) {
        xrow_u = xrow + Kc;
        gemv_fht_prologue<kFht>(A, signs_u, s_u, segs, K, Kc, k0,
                                 xrow_u, fstage);
    }

    // ---- the palettes: gate's and up's (register or shared) -----------
    float palg1_r = 0.0f, palg2_r = 0.0f;
    float palu1_r = 0.0f, palu2_r = 0.0f;
    if (gsg >= 64) {
        const int grow = (t * 128 + wx * 64) / gsg;
        palg1_r = __half2float(lutg1[(size_t)grow << b1g | (lane & ((1 << b1g) - 1))]);
        if (b2g > 0) {
            palg2_r = __half2float(lutg2[(size_t)grow << b2g | (lane & ((1 << b2g) - 1))]);
        }
    } else {
        const int ngrp = 128 / gsg;
        const int g0 = (t * 128) / gsg;
        for (int i = tid; i < (ngrp << b1g); i += 256)
            palg1_s[i] = lutg1[((size_t)g0 << b1g) + (size_t)i];
        if (b2g > 0) {
            for (int i = tid; i < (ngrp << b2g); i += 256)
                palg2_s[i] = lutg2[((size_t)g0 << b2g) + (size_t)i];
        }
    }
    if (gsu >= 64) {
        const int grow = (t * 128 + wx * 64) / gsu;
        palu1_r = __half2float(lutu1[(size_t)grow << b1u | (lane & ((1 << b1u) - 1))]);
        if (b2u > 0) {
            palu2_r = __half2float(lutu2[(size_t)grow << b2u | (lane & ((1 << b2u) - 1))]);
        }
    } else {
        const int ngrp = 128 / gsu;
        const int g0 = (t * 128) / gsu;
        for (int i = tid; i < (ngrp << b1u); i += 256)
            palu1_s[i] = lutu1[((size_t)g0 << b1u) + (size_t)i];
        if (b2u > 0) {
            for (int i = tid; i < (ngrp << b2u); i += 256)
                palu2_s[i] = lutu2[((size_t)g0 << b2u) + (size_t)i];
        }
    }

    // ---- the residual partials: gate's, then up's ----------------------
    // (warp w owns ranks {w*4 + 32*j + rr} of EACH blob — the split-K GEMV layout
    // twice; the xBf partials land in P's tail regions for the fold).
    // NOTE: the up residual consumes xrow_u (the up fold's rotated row).
    if (resBg != nullptr) {
        for (int r0 = warp * 4; r0 < Rg; r0 += 32) {
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = r0 + rr;
                if (r < Rg) {
                    float p = 0.0f;
                    const __half* brow = resBg + (size_t)r * K + k0;
                    for (int kk = 0; kk < Kc; kk += 32) {
                        const int k = kk + lane;
                        p = fmaf(__half2float(xrow[k]),
                                 __half2float(brow[k]), p);
                    }
                    #pragma unroll
                    for (int soff = 16; soff > 0; soff >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, soff);
                    if (lane == 0) {
                        xBf_g[r] = p;
                        if (SPLIT > 1) {
                            P[(size_t)sp * pstride + 2 * N + r] = p;
                        }
                    }
                }
            }
        }
    }
    if (resBu != nullptr) {
        for (int r0 = warp * 4; r0 < Ru; r0 += 32) {
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = r0 + rr;
                if (r < Ru) {
                    float p = 0.0f;
                    const __half* brow = resBu + (size_t)r * K + k0;
                    for (int kk = 0; kk < Kc; kk += 32) {
                        const int k = kk + lane;
                        p = fmaf(__half2float(xrow_u[k]),
                                 __half2float(brow[k]), p);
                    }
                    #pragma unroll
                    for (int soff = 16; soff > 0; soff >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, soff);
                    if (lane == 0) {
                        xBf_u[r] = p;
                        if (SPLIT > 1) {
                            P[(size_t)sp * pstride + 2 * N + Rg + r] = p;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();       // publish the palettes AND both xBf

    // ---- the K loops: the gate pass, then the up pass ------------------
    // (the up pass reads xrow_u when the folds differ — the second slice)
    float acc_g[8];
    float acc_u[8];
    #pragma unroll
    for (int a = 0; a < 8; ++a) { acc_g[a] = 0.0f; acc_u[a] = 0.0f; }

    const uint32_t* x2   = reinterpret_cast<const uint32_t*>(smem_raw);
    const uint32_t* x2_u = reinterpret_cast<const uint32_t*>(xrow_u);
    const int gs_shift_g = gemv_gs_shift(gsg);
    const int gs_shift_u = gemv_gs_shift(gsu);

    const uint8_t* cbg1 = qg1
        + (((size_t)t * G + (size_t)sp * Gc) * (1024 * (size_t)b1g))
        + (size_t)(wx * 512 + lane * 16) * (size_t)b1g;
    const uint8_t* cbg2 = (b2g > 0)
        ? (qg2 + (((size_t)t * G + (size_t)sp * Gc) * (1024 * (size_t)b2g))
                  + (size_t)(wx * 512 + lane * 16) * (size_t)b2g)
        : qg1;
    const uint8_t* cbu1 = qu1
        + (((size_t)t * G + (size_t)sp * Gc) * (1024 * (size_t)b1u))
        + (size_t)(wx * 512 + lane * 16) * (size_t)b1u;
    const uint8_t* cbu2 = (b2u > 0)
        ? (qu2 + (((size_t)t * G + (size_t)sp * Gc) * (1024 * (size_t)b2u))
                  + (size_t)(wx * 512 + lane * 16) * (size_t)b2u)
        : qu1;

    if (gsg >= 64) {
        FLUTE_HET_KLOOP_SWITCH(true, b1g * 8 + b2g,
                               cbg1, cbg2, x2, Gc, j4, lane, wx,
                               gs_shift_g, palg1_r, palg2_r,
                               palg1_s, palg2_s, acc_g)
    } else {
        FLUTE_HET_KLOOP_SWITCH(false, b1g * 8 + b2g,
                               cbg1, cbg2, x2, Gc, j4, lane, wx,
                               gs_shift_g, palg1_r, palg2_r,
                               palg1_s, palg2_s, acc_g)
    }
    if (gsu >= 64) {
        FLUTE_HET_KLOOP_SWITCH(true, b1u * 8 + b2u,
                               cbu1, cbu2, x2_u, Gc, j4, lane, wx,
                               gs_shift_u, palu1_r, palu2_r,
                               palu1_s, palu2_s, acc_u)
    } else {
        FLUTE_HET_KLOOP_SWITCH(false, b1u * 8 + b2u,
                               cbu1, cbu2, x2_u, Gc, j4, lane, wx,
                               gs_shift_u, palu1_r, palu2_r,
                               palu1_s, palu2_s, acc_u)
    }

    // ---- reduction: the i-lane butterfly, then red[row][j4] ----------
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        acc_g[a] += __shfl_xor_sync(0xffffffffu, acc_g[a], 1);
        acc_g[a] += __shfl_xor_sync(0xffffffffu, acc_g[a], 2);
        acc_u[a] += __shfl_xor_sync(0xffffffffu, acc_u[a], 1);
        acc_u[a] += __shfl_xor_sync(0xffffffffu, acc_u[a], 2);
    }
    if ((lane & 3) == 0) {
        #pragma unroll
        for (int v = 0; v < 4; ++v)
            #pragma unroll
            for (int d = 0; d < 2; ++d) {
                const int row = wx * 64 + v * 16 + d * 8 + (lane >> 2);
                red_g[row * 4 + (warp >> 1)] = acc_g[v * 2 + d];
                red_u[row * 4 + (warp >> 1)] = acc_u[v * 2 + d];
            }
    }
    __syncthreads();

    // ---- the epilogue: direct (SPLIT == 1) or fold via P --------------
    if (SPLIT == 1) {
        if (tid < 128) {
            const int n = t * 128 + tid;
            float yg = red_g[tid * 4 + 0] + red_g[tid * 4 + 1] +
                       red_g[tid * 4 + 2] + red_g[tid * 4 + 3];
            float yu = red_u[tid * 4 + 0] + red_u[tid * 4 + 1] +
                       red_u[tid * 4 + 2] + red_u[tid * 4 + 3];
            if (resAg != nullptr) {
                const __half* ra = resAg + (size_t)n * Rg;
                #pragma unroll 8
                for (int r = 0; r < Rg; ++r)
                    yg = fmaf(xBf_g[r], __half2float(ra[r]), yg);
            }
            if (biasg != nullptr) yg += __half2float(biasg[n]);
            if (resAu != nullptr) {
                const __half* ra = resAu + (size_t)n * Ru;
                #pragma unroll 8
                for (int r = 0; r < Ru; ++r)
                    yu = fmaf(xBf_u[r], __half2float(ra[r]), yu);
            }
            if (biasu != nullptr) yu += __half2float(biasu[n]);
            const float sg = yg / (1.0f + expf(-yg));   // silu, fp32
            C[n] = __float2half(sg * yu);               // ONE fp16 round
        }
    } else {
        // publish this split's row partials (fp32; every element written)
        if (tid < 128) {
            P[(size_t)sp * pstride + (size_t)t * 128 + tid] =
                red_g[tid * 4 + 0] + red_g[tid * 4 + 1] +
                red_g[tid * 4 + 2] + red_g[tid * 4 + 3];
            P[(size_t)sp * pstride + N + (size_t)t * 128 + tid] =
                red_u[tid * 4 + 0] + red_u[tid * 4 + 1] +
                red_u[tid * 4 + 2] + red_u[tid * 4 + 3];
        }
        __syncthreads();   // all of this CTA's P writes are complete

        // the deterministic arrival ticket (the split-K GEMV contract)
        if (tid == 0) {
            __threadfence();
            ticket_s[0] = (unsigned int)atomicAdd(&tickets[t], 1u);
        }
        __syncthreads();

        if (ticket_s[0] == (unsigned int)(SPLIT - 1)) {
            __threadfence();   // acquire side (belt and braces)
            // fold both xBf partials in fixed split order -> smem
            if (tid < Rg) {
                float xb = 0.0f;
                for (int s2i = 0; s2i < SPLIT; ++s2i)
                    xb += P[(size_t)s2i * pstride + 2 * N + tid];
                xBf_g[tid] = xb;
            }
            if (tid < Ru) {
                float xb = 0.0f;
                for (int s2i = 0; s2i < SPLIT; ++s2i)
                    xb += P[(size_t)s2i * pstride + 2 * N + Rg + tid];
                xBf_u[tid] = xb;
            }
            __syncthreads();
            // fold both row partials in fixed split order -> silu*mul -> C
            if (tid < 128) {
                float yg = 0.0f;
                float yu = 0.0f;
                for (int s2i = 0; s2i < SPLIT; ++s2i) {
                    yg += P[(size_t)s2i * pstride + (size_t)t * 128 + tid];
                    yu += P[(size_t)s2i * pstride + N + (size_t)t * 128
                            + tid];
                }
                const int n = t * 128 + tid;
                if (resAg != nullptr) {
                    const __half* ra = resAg + (size_t)n * Rg;
                    #pragma unroll 8
                    for (int r = 0; r < Rg; ++r)
                        yg = fmaf(xBf_g[r], __half2float(ra[r]), yg);
                }
                if (biasg != nullptr) yg += __half2float(biasg[n]);
                if (resAu != nullptr) {
                    const __half* ra = resAu + (size_t)n * Ru;
                    #pragma unroll 8
                    for (int r = 0; r < Ru; ++r)
                        yu = fmaf(xBf_u[r], __half2float(ra[r]), yu);
                }
                if (biasu != nullptr) yu += __half2float(biasu[n]);
                const float sg = yg / (1.0f + expf(-yg));
                C[n] = __float2half(sg * yu);
            }
            if (tid == 0) tickets[t] = 0u;   // the split-K GEMV self-reset
        }
    }
}

// The MLP launch helper (the split-K GEMV machinery: workspace, tickets, smem).
// When the up fold differs (per-blob signs / AWQ scale) the x region
// DOUBLES (xrow + xrow_u). kAwq is GONE (per-blob s pointers) — 2
// instantiations instead of the its 60.
template <bool kFht>
void launch_gemv_mlp(const __half* A,
    int N, int K, int Rg, int Ru,
    const uint8_t* qg1, const __half* lutg1,
    const uint8_t* qg2, const __half* lutg2,
    const __half* resBg, const __half* resAg, const __half* biasg,
    int b1g, int b2g, int gsg,
    const float* signs_g, const float* s_g,
    const uint8_t* qu1, const __half* lutu1,
    const uint8_t* qu2, const __half* lutu2,
    const __half* resBu, const __half* resAu, const __half* biasu,
    int b1u, int b2u, int gsu,
    const float* signs_u, const float* s_u,
    __half* C
) {
    const int tiles = N / 128;
    const int G = K / 64;
    const int SPLIT = flute::gemv_pick_split(tiles, G);
    const int Kc = K / SPLIT;

    const int b1s[2] = {b1g, b1u};
    const int b2s[2] = {b2g, b2u};
    const int gss[2] = {gsg, gsu};
    const bool fold_differs = kFht && (
        (signs_g != signs_u) || (s_g != s_u));
    int smem_bytes = gemv_mlp_smem(tiles, K, b1s, b2s, gss, kFht);
    if (fold_differs) {
        smem_bytes += 2 * Kc;   // the doubled x region
    }

    TORCH_CHECK(smem_bytes <= 99 * 1024,
                "flute_kernel_gemv_mlp: smem ", smem_bytes, " B exceeds "
                "the SM_86 99 KB block limit (K=", K, ", Kc=", Kc,
                ") — the caller's gate should have refused");

    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_gemv_mlp<kFht>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=101376) "
                "failed for the split-K GEMV MLP kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    float* P = nullptr;
    unsigned int* tickets = nullptr;
    int pstride = 0;
    if (SPLIT > 1) {
        pstride = 2 * N + Rg + Ru;
        flute::GemvWorkspace& ws = flute::gemv_workspace_for(flute::gemv_current_device(), (int64_t)SPLIT * pstride, tiles);
        P = ws.P.data_ptr<float>();
        tickets =
            reinterpret_cast<unsigned int*>(ws.tickets.data_ptr<int32_t>());
    }

    const flute::FhtSegs segs = flute::fht_segments(K);
    dim3 block(256);
    dim3 grid(tiles, SPLIT);
    flute_kernel_gemv_mlp<kFht>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, P, tickets, segs, N, K, Rg, Ru, SPLIT, pstride,
            qg1, lutg1, qg2, lutg2, resBg, resAg, biasg,
            b1g, b2g, gsg, signs_g, s_g,
            qu1, lutu1, qu2, lutu2, resBu, resAu, biasu,
            b1u, b2u, gsu, signs_u, s_u,
            C);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_gemv_mlp launch failed (grid ", grid.x, "x",
                grid.y, ", 256 threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

// The MLP impl. The segment table is the multi kernel's [2, 15]
// (row 0 = gate, row 1 = up; N must be identical — the shapes are by
// construction, the impl re-checks). Returns C [1, N] = silu(g)*u — a
// FRESH tensor (the persistent output is the caller's choice: the
// wrapper passes its persistent buffer as the gate row's C when it owns
// one; the impl writes through that pointer).
torch::Tensor qgemm_cutlass_gemv_mlp_impl(torch::Tensor A,
    torch::Tensor seg_table,
    torch::Tensor C
) {
    TORCH_CHECK(A.is_cuda(), "qgemm_gemv_mlp: A must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16,
                "qgemm_gemv_mlp: A must be float16");
    TORCH_CHECK(A.is_contiguous(), "qgemm_gemv_mlp: A must be contiguous");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1,
                "qgemm_gemv_mlp: the merged decode GEMV serves M == 1 "
                "(A is [M, K])");
    const int K = (int)A.size(1);
    TORCH_CHECK(K % 64 == 0,
                "qgemm_gemv_mlp: K must be a multiple of 64 (the idxN "
                "blob's 64-k tiles; got K=", K, ")");
    TORCH_CHECK(!flute::fd_disabled_by_env(),
                "qgemm_gemv_mlp: FLUTE_NO_FD=1 is set (the GEMV is an "
                "idxN-layout consumer)");
    TORCH_CHECK(seg_table.device().is_cpu(),
                "qgemm_gemv_mlp: seg_table must be a CPU tensor");
    TORCH_CHECK(seg_table.dtype() == torch::kInt64,
                "qgemm_gemv_mlp: seg_table must be int64");
    TORCH_CHECK(seg_table.dim() == 2 && seg_table.size(0) == 2
                && seg_table.size(1) == 15,
                "qgemm_gemv_mlp: seg_table must be [2, 15] (gate, up; "
                "the multi kernel's column contract)");
    TORCH_CHECK(C.is_cuda() && C.dtype() == torch::kFloat16
                && C.is_contiguous(),
                "qgemm_gemv_mlp: C must be a contiguous CUDA fp16 tensor");

    const int64_t* rows = seg_table.data_ptr<int64_t>();

    int b1s[2], b2s[2], gss[2], Ns[2], Rs[2];
    const uint8_t* q1s[2];
    const __half* l1s[2];
    const uint8_t* q2s[2];
    const __half* l2s[2];
    const __half* rBs[2];
    const __half* rAs[2];
    const __half* bis[2];
    const float* signss[2];
    const float* ss[2];
    int rotated = 0;
    for (int i = 0; i < 2; ++i) {
        const int64_t* r = rows + (size_t)i * 15;
        q1s[i]   = reinterpret_cast<const uint8_t*>((uintptr_t)r[0]);
        l1s[i]   = reinterpret_cast<const __half*>((uintptr_t)r[1]);
        q2s[i]   = reinterpret_cast<const uint8_t*>((uintptr_t)r[2]);
        l2s[i]   = reinterpret_cast<const __half*>((uintptr_t)r[3]);
        rBs[i]   = reinterpret_cast<const __half*>((uintptr_t)r[4]);
        rAs[i]   = reinterpret_cast<const __half*>((uintptr_t)r[5]);
        bis[i]   = reinterpret_cast<const __half*>((uintptr_t)r[6]);
        Ns[i]    = (int)r[8];
        Rs[i]    = (int)r[9];
        b1s[i]   = (int)r[10];
        b2s[i]   = (int)r[11];
        gss[i]   = (int)r[12];
        signss[i] = reinterpret_cast<const float*>((uintptr_t)r[13]);
        ss[i]    = reinterpret_cast<const float*>((uintptr_t)r[14]);

        TORCH_CHECK(b1s[i] >= 1 && b1s[i] <= 4,
                    "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                    " bitwidth ", b1s[i], " is outside 1..4");
        TORCH_CHECK(b2s[i] >= 0 && b2s[i] <= 4,
                    "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                    " bitwidth2 ", b2s[i], " is outside 0..4");
        TORCH_CHECK(gss[i] == 16 || gss[i] == 32 || gss[i] == 64 ||
                    gss[i] == 128 || gss[i] == 256 || gss[i] == 512 ||
                    gss[i] == 1024 || gss[i] == 2048,
                    "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                    " group_size ", gss[i], " is not one of "
                    "16/32/64/128/256/512/1024/2048");
        TORCH_CHECK(q1s[i] != nullptr && l1s[i] != nullptr,
                    "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                    " is missing its stream-1 indices/lut");
        if (b2s[i] > 0) {
            TORCH_CHECK(q2s[i] != nullptr && l2s[i] != nullptr,
                        "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                        " declares bitwidth2=", b2s[i], " but is missing "
                        "its stream-2 indices/lut");
        }
        TORCH_CHECK(Ns[i] > 0 && Ns[i] % 128 == 0,
                    "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                    " N=", Ns[i], " is not a positive multiple of 128");
        TORCH_CHECK(Rs[i] >= 0 && Rs[i] <= 256,
                    "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                    " residual rank ", Rs[i], " is outside 0..256");
        if (Rs[i] > 0) {
            TORCH_CHECK(rBs[i] != nullptr && rAs[i] != nullptr,
                        "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                        " needs resB and resA at rank ", Rs[i]);
        }
        if (signss[i] != nullptr) {
            ++rotated;
            TORCH_CHECK(reinterpret_cast<uintptr_t>(signss[i]) % 16 == 0,
                        "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                        " signs pointer is not 16-B aligned");
        }
        if (ss[i] != nullptr) {
            TORCH_CHECK(signss[i] != nullptr,
                        "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                        " carries an AWQ scale without rotation");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(ss[i]) % 16 == 0,
                        "qgemm_gemv_mlp: ", (i == 0 ? "gate" : "up"),
                        " s pointer is not 16-B aligned");
        }
    }
    TORCH_CHECK(Ns[0] == Ns[1],
                "qgemm_gemv_mlp: gate N=", Ns[0], " != up N=", Ns[1],
                " (the MLP blobs share N by construction)");
    TORCH_CHECK((int)C.numel() == Ns[0],
                "qgemm_gemv_mlp: C has ", (int)C.numel(), " elements, "
                "expected N=", Ns[0]);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(C.data_ptr<at::Half>()) % 16 == 0,
                "qgemm_gemv_mlp: C is not 16-B aligned");
    TORCH_CHECK(rotated == 0 || rotated == 2,
                "qgemm_gemv_mlp: mixed rotation presence (", rotated,
                " of 2 blobs rotated) — all or none");

    const int N = Ns[0];
    const int smem_check = gemv_mlp_smem(
        N / 128, K, b1s, b2s, gss, rotated == 2);
    TORCH_CHECK(smem_check <= 99 * 1024,
                "qgemm_gemv_mlp: K=", K, " (N=", N, ", Rg=", Rs[0],
                ", Ru=", Rs[1], ") needs ", smem_check,
                " B of shared memory (over the SM_86 99 KB block limit)");

    const flute::FhtSegs segs = flute::fht_segments(K);
    if (rotated == 2) {
        TORCH_CHECK(segs.n > 0 && (segs.off[segs.n - 1]
                                   + segs.len[segs.n - 1]) == K,
                    "qgemm_gemv_mlp: the FHT segment table does not "
                    "tile K=", K);
    }

    const __half* a_p = reinterpret_cast<const __half*>(
        A.data_ptr<at::Half>());
    __half* c_p = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    if (rotated == 2) {
        launch_gemv_mlp<true>(a_p, N, K, Rs[0], Rs[1],
            q1s[0], l1s[0], q2s[0], l2s[0], rBs[0], rAs[0], bis[0],
            b1s[0], b2s[0], gss[0], signss[0], ss[0],
            q1s[1], l1s[1], q2s[1], l2s[1], rBs[1], rAs[1], bis[1],
            b1s[1], b2s[1], gss[1], signss[1], ss[1],
            c_p);
    } else {
        launch_gemv_mlp<false>(a_p, N, K, Rs[0], Rs[1],
            q1s[0], l1s[0], q2s[0], l2s[0], rBs[0], rAs[0], bis[0],
            b1s[0], b2s[0], gss[0], signss[0], ss[0],
            q1s[1], l1s[1], q2s[1], l2s[1], rBs[1], rAs[1], bis[1],
            b1s[1], b2s[1], gss[1], signss[1], ss[1],
            c_p);
    }
    return C;
}

}  // namespace

// Public entrypoint (): the merged gate+up split-K GEMV with the
// SiLU*mul epilogue, heterogeneous (per-blob bitwidths /
// group sizes / rotation signs / AWQ scales). seg_table is the multi
// kernel's [2, 15] (row 0 = gate, row 1 = up); the gate row's C column
// is IGNORED (the impl writes the product into the returned/fresh C —
// pass the persistent buffer via the wrapper's C argument).
torch::Tensor qgemm_cutlass_gemv_mlp(torch::Tensor A,
    torch::Tensor seg_table,
    torch::Tensor C
) {
    return qgemm_cutlass_gemv_mlp_impl(A, seg_table, C);
}
