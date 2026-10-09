/**
 * src/kernel_gemv_multi.cu
 *
 * The grouped multi-blob split-K GEMV — the QKV merge (entry
 * qgemm_cutlass_gemv_multi, flute/entrypoints.h).
 *
 * ONE split-K launch computes 2-4 modules' outputs for the SAME input
 * row x: the concatenated N-tile space is the grid, each CTA resolves
 * its segment from the cumulative tile table and serves exactly that
 * module's blobs (SplitQKV's torch.cat + three separate launches gone).
 *
 * HETEROGENEOUS SEGMENTS: the build
 * refused 24/30 QKV groups and 32/32 MLP groups on the box — "spec
 * mismatch across components" / "rotation seeds differ" — because the
 * deployed mixed-radix palette gives every tensor its own
 * (bitwidth, bitwidth2, group_size) and every AWQ-folded tensor its own
 * sign vector. This kernel takes all of those PER SEGMENT at RUNTIME:
 *
 *   * (b1, b2): a CTA-uniform switch (FLUTE_HET_KLOOP_SWITCH,
 *     flute/gemv.cuh) into the SAME compile-time gemv_kloop the
 *     single-module split-K GEMV runs — one uniform branch, zero divergence.
 *   * gs: runtime per segment (GS 16/32 switch to the shared-palette
 *     path, GS >= 64 the register/shfl path — the design).
 *   * signs / s (the boundary-fold and AWQ operands): per-segment
 *     pointers; a segment with s == 0 skips the AWQ compensation
 *     (CTA-uniform branch). Components with DIFFERENT rotation seeds
 *     now merge in one launch — each CTA's FHT prologue applies its own
 *     segment's signs.
 *
 * Everything else is the design verbatim: the deterministic
 * split-K ticket finalization (fixed-order folds, one fp16 round, the
 * self-resetting workspace), the double-buffered K loop, the rank<=256
 * residual epilogue, the persistent CPU segment table (pointers stable
 * across CUDA-graph replays).
 *
 * Numerics contract: per segment identical to the single-module split-K GEMV of
 * the SAME (b1, b2, gs) — the kloop arms are the same template
 * instantiations. Decode-only (M == 1); prefill/PPL never routes here.
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

// The one-liner GS index helper shared by the kernel + launcher: GS
// 16/32 shift (the shared-palette path), else 0 (register palette —
// gs_shift is dead on that path).
__host__ __device__ inline int gemv_gs_shift(int gs) {
    return (gs == 16) ? 4 : ((gs == 32) ? 5 : 0);
}

// The merged smem bill: the x slice (2*Kc at the MERGED SPLIT) + the
// tail sized for the WORST segment (the launch allocates one size for
// every CTA; each CTA lays out its own tail inside it).
static inline int gemv_multi_smem(int tiles, int K, const int* b1s, const int* b2s, const int* gss,
    int count, bool fht
) {
    const int G = K / 64;
    const int SPLIT = flute::gemv_pick_split(tiles, G);
    const int Kc = K / SPLIT;
    int pal_max = 0;
    for (int i = 0; i < count; ++i) {
        if (gss[i] < 64) {
            const int ngrp = 128 / gss[i];
            const int pal = (ngrp << b1s[i]) * 2 + (ngrp << b2s[i]) * 2;
            pal_max = (pal > pal_max) ? pal : pal_max;
        }
    }
    const int tail_need = (2048 + 1024 + pal_max + 4 + 15) & ~15;
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
// The heterogeneous multi kernel. grid = (tiles_total, SPLIT); 256
// threads. smem = [ x fp16 slice: 2*Kc | tail ] with the tail laid out
// per-CTA: red[512] | xBf[256] | pal1/pal2 (GS<64) | ticket; the FHT
// fp32 staging overlays the tail's dead end (the split-K GEMV arrangement).
// ---------------------------------------------------------------------------
template <bool kFht>
__global__ void __launch_bounds__(256, 2)
flute_kernel_gemv_multi(const __half*    __restrict__ A,      // [1, K] (rotated when !kFht)
    float*           __restrict__ P,      // [SPLIT, pstride] fp32 or nullptr
    unsigned int*    __restrict__ tickets,// [tiles_total] or nullptr
    flute::FhtSegs segs,
    int K, int SPLIT, int pstride,
    flute::GemvSegTab tab
) {
    const int t   = blockIdx.x;              // GLOBAL tile (across segs)
    const int sp  = blockIdx.y;              // the K-split index
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wx = warp & 1;                 // rows [wx*64, wx*64 + 64)
    const int j4 = warp >> 1;                // K-quarters: g = j4, j4+4, ...

    // ---- resolve this CTA's segment (uniform across the CTA) ---------
    int seg = 0;
    #pragma unroll 1
    for (int i = 0; i < 4; ++i) {
        if (i < tab.count && t >= tab.tile_base[i + 1]) seg = i + 1;
    }
    const int t_loc = t - tab.tile_base[seg];       // tile within the seg
    const int n0    = t_loc * 128;                  // row base within the seg
    const int R     = tab.r_seg[seg];
    const int tiles_rows = tab.tiles_total * 128;   // P's row region size

    // ---- THIS segment's runtime spec -----------------------------
    const int b1 = tab.b1[seg];
    const int b2 = tab.b2[seg];
    const int gs = tab.gs[seg];
    const float* signs_seg = kFht ? tab.signs[seg] : nullptr;
    const float* s_seg     = (kFht ? tab.s[seg] : nullptr);

    const int Kc = K / SPLIT;                // the split's k-range (>= 64)
    const int k0 = sp * Kc;
    const int Gc = Kc >> 6;                  // g-tiles in this split
    const int G  = K >> 6;                   // global g-tiles (blob stride)

    // ---- shared memory: [ x fp16 slice: 2*Kc | tail ] ------------------
    // (the launch sized the tail for the WORST segment; this CTA lays
    // out its own — the GS<64 palettes are sized by THIS seg's b1/b2/gs)
    const int pal_bytes = (gs < 64)
        ? (((128 / gs) << b1) * 2 + ((128 / gs) << b2) * 2) : 0;
    const int tail_need = (2048 + 1024 + pal_bytes + 4 + 15) & ~15;
    int b_max = 0;
    if (kFht) {
        for (int i = 0; i < segs.n; ++i)
            b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    }
    const int tail = kFht
        ? ((4 * b_max >= tail_need) ? 4 * b_max : tail_need) : tail_need;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* xrow = reinterpret_cast<__half*>(smem_raw);          // [Kc]
    unsigned char* tail_base = smem_raw + 2 * Kc + tail - tail_need;
    float*  red  = reinterpret_cast<float*>(tail_base);          // 512 floats
    float*  xBf  = red + 512;                                    // 256 floats
    __half* pal1_s = reinterpret_cast<__half*>(xBf + 256);
    __half* pal2_s = pal1_s + ((gs < 64) ? ((128 / gs) << b1) : 0);
    unsigned int* ticket_s =
        reinterpret_cast<unsigned int*>(tail_base + 2048 + 1024 + pal_bytes);
    float* fstage = reinterpret_cast<float*>(smem_raw + 2 * Kc); // kFht only

    // ---- the x row: the split's k-slice (per-seg signs/AWQ) -----------
    gemv_fht_prologue<kFht>(A, signs_seg, s_seg, segs, K, Kc, k0,
                             xrow, fstage);

    // ---- the palette: THIS segment's LUT (register or shared) --------
    float pal1_r = 0.0f, pal2_r = 0.0f;
    if (gs >= 64) {
        const int grow = (n0 + wx * 64) / gs;
        pal1_r = __half2float(tab.lut1[seg][(size_t)grow << b1 | (lane & ((1 << b1) - 1))]);
        if (b2 > 0) {
            pal2_r = __half2float(tab.lut2[seg][(size_t)grow << b2
                              | (lane & ((1 << b2) - 1))]);
        }
    } else {
        const int ngrp = 128 / gs;
        const int g0 = n0 / gs;
        for (int i = tid; i < (ngrp << b1); i += 256)
            pal1_s[i] = tab.lut1[seg][((size_t)g0 << b1) + (size_t)i];
        if (b2 > 0) {
            for (int i = tid; i < (ngrp << b2); i += 256)
                pal2_s[i] = tab.lut2[seg][((size_t)g0 << b2) + (size_t)i];
        }
    }

    // ---- the residual partials: THIS segment's resB -------------------
    if (tab.resB[seg] != nullptr) {
        for (int r0 = warp * 4; r0 < R; r0 += 32) {
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int r = r0 + rr;
                if (r < R) {
                    float p = 0.0f;
                    const __half* brow = tab.resB[seg]
                        + (size_t)r * K + k0;
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
                            P[(size_t)sp * pstride + (size_t)tiles_rows
                              + tab.r_off[seg] + r] = p;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();       // publish the palettes AND xBf

    // ---- the K loop (the split-K GEMV double-buffered loop, THIS seg's blobs) --
    float acc[8];
    #pragma unroll
    for (int a = 0; a < 8; ++a) acc[a] = 0.0f;

    const uint8_t* cb1 = tab.q1[seg]
        + (((size_t)t_loc * G + (size_t)sp * Gc) * (1024 * (size_t)b1))
        + (size_t)(wx * 512 + lane * 16) * (size_t)b1;
    const uint8_t* cb2 = (b2 > 0)
        ? (tab.q2[seg]
            + (((size_t)t_loc * G + (size_t)sp * Gc) * (1024 * (size_t)b2))
            + (size_t)(wx * 512 + lane * 16) * (size_t)b2)
        : cb1;
    const uint32_t* x2 = reinterpret_cast<const uint32_t*>(smem_raw);
    const int gs_shift = gemv_gs_shift(gs);

    if (gs >= 64) {
        FLUTE_HET_KLOOP_SWITCH(true, b1 * 8 + b2,
                               cb1, cb2, x2, Gc, j4, lane, wx, gs_shift,
                               pal1_r, pal2_r, pal1_s, pal2_s, acc)
    } else {
        FLUTE_HET_KLOOP_SWITCH(false, b1 * 8 + b2,
                               cb1, cb2, x2, Gc, j4, lane, wx, gs_shift,
                               pal1_r, pal2_r, pal1_s, pal2_s, acc)
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
            const int n = n0 + tid;
            float y = red[tid * 4 + 0] + red[tid * 4 + 1] +
                      red[tid * 4 + 2] + red[tid * 4 + 3];
            if (tab.resA[seg] != nullptr) {
                const __half* ra = tab.resA[seg] + (size_t)n * R;
                #pragma unroll 8
                for (int r = 0; r < R; ++r) {
                    y = fmaf(xBf[r], __half2float(ra[r]), y);
                }
            }
            if (tab.bias[seg] != nullptr) {
                y += __half2float(tab.bias[seg][n]);
            }
            tab.c[seg][n] = __float2half(y);
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

        // the deterministic arrival ticket (the split-K GEMV contract, keyed by the
        // GLOBAL tile): fence -> atomicAdd -> the LAST arrival finalizes.
        if (tid == 0) {
            __threadfence();
            ticket_s[0] = (unsigned int)atomicAdd(&tickets[t], 1u);
        }
        __syncthreads();

        if (ticket_s[0] == (unsigned int)(SPLIT - 1)) {
            __threadfence();   // acquire side (belt and braces)
            // fold THIS seg's xBf partials in fixed split order -> xBf smem
            if (tid < R) {
                float xb = 0.0f;
                for (int s2i = 0; s2i < SPLIT; ++s2i)
                    xb += P[(size_t)s2i * pstride + (size_t)tiles_rows
                            + tab.r_off[seg] + tid];
                xBf[tid] = xb;
            }
            __syncthreads();
            // fold the row partials in fixed split order -> C (ONE fp16
            // round, after the residual + bias — the epilogue order)
            if (tid < 128) {
                float y = 0.0f;
                for (int s2i = 0; s2i < SPLIT; ++s2i)
                    y += P[(size_t)s2i * pstride + (size_t)t * 128 + tid];
                const int n = n0 + tid;
                if (tab.resA[seg] != nullptr) {
                    const __half* ra = tab.resA[seg] + (size_t)n * R;
                    #pragma unroll 8
                    for (int r = 0; r < R; ++r) {
                        y = fmaf(xBf[r], __half2float(ra[r]), y);
                    }
                }
                if (tab.bias[seg] != nullptr) {
                    y += __half2float(tab.bias[seg][n]);
                }
                tab.c[seg][n] = __float2half(y);
            }
            // the self-reset: the ticket returns to zero for the NEXT
            // call/replay (the split-K GEMV workspace invariant, verbatim).
            if (tid == 0) tickets[t] = 0u;
        }
    }
}

// The multi launch helper: same smem/workspace/ticket machinery as the
// single-module split-K GEMV, keyed by the MERGED tile space (tiles_total). The
// kAwq template is GONE (per-segment s pointers; a null s skips the
// compensation) — 2 kernel instantiations instead of the its 60.
template <bool kFht>
void launch_gemv_multi(const __half* A, const flute::GemvSegTab& tab, int K,
    const flute::FhtSegs& segs
) {
    const int tiles = tab.tiles_total;
    const int G = K / 64;
    const int SPLIT = flute::gemv_pick_split(tiles, G);
    const int Kc = K / SPLIT;

    const int smem_bytes = gemv_multi_smem(tiles, K, tab.b1, tab.b2, tab.gs, tab.count, kFht);

    TORCH_CHECK(smem_bytes <= 99 * 1024,
                "flute_kernel_gemv_multi: smem ", smem_bytes, " B exceeds "
                "the SM_86 99 KB block limit (K=", K, ", Kc=", Kc,
                ") — the caller's gate should have refused");

    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_gemv_multi<kFht>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=101376) "
                "failed for the split-K GEMV multi kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    float* P = nullptr;
    unsigned int* tickets = nullptr;
    int pstride = 0;
    if (SPLIT > 1) {
        pstride = tiles * 128 + tab.r_total;
        flute::GemvWorkspace& ws = flute::gemv_workspace_for(flute::gemv_current_device(), (int64_t)SPLIT * pstride, tiles);
        P = ws.P.data_ptr<float>();
        tickets =
            reinterpret_cast<unsigned int*>(ws.tickets.data_ptr<int32_t>());
    }

    dim3 block(256);
    dim3 grid(tiles, SPLIT);
    flute_kernel_gemv_multi<kFht>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, P, tickets, segs, K, SPLIT, pstride, tab);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_gemv_multi launch failed (grid ", grid.x, "x",
                grid.y, ", 256 threads, smem ", smem_bytes, " B, ",
                tab.count, " segments): ",
                cudaGetErrorString(launch_err));
}

// The multi impl. The segment table arrives as a CPU int64 tensor
// [count, 15] — per segment
//   [q1_addr, lut1_addr, q2_addr, lut2_addr, resb_addr, resa_addr,
//    bias_addr, c_addr, n_seg, r_seg, b1, b2, gs, signs_addr, s_addr]
// (null tensors = address 0). The PYTHON wrapper
// (flute_extended.qgemm_gemv_multi) owns the dtype/contiguity/shape
// mirrors and builds the table from LIVE tensors ONCE (all pointers are
// stable module weights + persistent output buffers; the per-seg
// signs/s pointers must equally stay stable). This impl re-checks
// everything the raw addresses can still see (N/R bounds, the per-seg
// spec ranges, GS/K divisibility, smem, 16 B alignment).
void qgemm_cutlass_gemv_multi_impl(torch::Tensor A,
    torch::Tensor seg_table
) {
    TORCH_CHECK(A.is_cuda(), "qgemm_gemv_multi: A must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16,
                "qgemm_gemv_multi: A must be float16");
    TORCH_CHECK(A.is_contiguous(), "qgemm_gemv_multi: A must be contiguous");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1,
                "qgemm_gemv_multi: the merged decode GEMV serves M == 1 "
                "(A is [M, K])");
    const int K = (int)A.size(1);
    TORCH_CHECK(K % 64 == 0,
                "qgemm_gemv_multi: K must be a multiple of 64 (the idxN "
                "blob's 64-k tiles; got K=", K, ")");
    TORCH_CHECK(!flute::fd_disabled_by_env(),
                "qgemm_gemv_multi: FLUTE_NO_FD=1 is set (the GEMV is an "
                "idxN-layout consumer)");
    TORCH_CHECK(seg_table.device().is_cpu(),
                "qgemm_gemv_multi: seg_table must be a CPU tensor");
    TORCH_CHECK(seg_table.dtype() == torch::kInt64,
                "qgemm_gemv_multi: seg_table must be int64");
    TORCH_CHECK(seg_table.dim() == 2 && seg_table.size(1) == 15,
                "qgemm_gemv_multi: seg_table must be [count, 15] "
                "(q1, lut1, q2, lut2, resB, resA, bias, C, N, R, "
                "b1, b2, gs, signs, s)");
    const int count = (int)seg_table.size(0);
    TORCH_CHECK(count >= 2 && count <= 4,
                "qgemm_gemv_multi: 2-4 segments (got ", count,
                ") — a single module routes through qgemm_gemv_splitk_stream");

    const int64_t* rows = seg_table.data_ptr<int64_t>();
    flute::GemvSegTab tab;
    tab.count = count;
    tab.tiles_total = 0;
    tab.r_total = 0;
    int rotated = 0;      // all-or-none rotation presence (the fold gate)
    for (int i = 0; i < count; ++i) {
        const int64_t* r = rows + (size_t)i * 15;
        tab.q1[i]   = reinterpret_cast<const uint8_t*>((uintptr_t)r[0]);
        tab.lut1[i] = reinterpret_cast<const __half*>((uintptr_t)r[1]);
        tab.q2[i]   = reinterpret_cast<const uint8_t*>((uintptr_t)r[2]);
        tab.lut2[i] = reinterpret_cast<const __half*>((uintptr_t)r[3]);
        tab.resB[i] = reinterpret_cast<const __half*>((uintptr_t)r[4]);
        tab.resA[i] = reinterpret_cast<const __half*>((uintptr_t)r[5]);
        tab.bias[i] = reinterpret_cast<const __half*>((uintptr_t)r[6]);
        tab.c[i]    = reinterpret_cast<__half*>((uintptr_t)r[7]);
        tab.n_seg[i]  = (int)r[8];
        tab.r_seg[i]  = (int)r[9];
        tab.b1[i]     = (int)r[10];
        tab.b2[i]     = (int)r[11];
        tab.gs[i]     = (int)r[12];
        tab.signs[i]  = reinterpret_cast<const float*>((uintptr_t)r[13]);
        tab.s[i]      = reinterpret_cast<const float*>((uintptr_t)r[14]);

        // ---- the per-segment spec validation (its whole point) -----
        TORCH_CHECK(tab.b1[i] >= 1 && tab.b1[i] <= 4,
                    "qgemm_gemv_multi: segment ", i, " bitwidth ",
                    tab.b1[i], " is outside 1..4");
        TORCH_CHECK(tab.b2[i] >= 0 && tab.b2[i] <= 4,
                    "qgemm_gemv_multi: segment ", i, " bitwidth2 ",
                    tab.b2[i], " is outside 0..4");
        TORCH_CHECK(tab.gs[i] == 16 || tab.gs[i] == 32 ||
                    tab.gs[i] == 64 || tab.gs[i] == 128 ||
                    tab.gs[i] == 256 || tab.gs[i] == 512 ||
                    tab.gs[i] == 1024 || tab.gs[i] == 2048,
                    "qgemm_gemv_multi: segment ", i, " group_size ",
                    tab.gs[i], " is not one of 16/32/64/128/256/512/"
                    "1024/2048");
        TORCH_CHECK(tab.q1[i] != nullptr && tab.lut1[i] != nullptr,
                    "qgemm_gemv_multi: segment ", i, " is missing its "
                    "stream-1 indices/lut");
        if (tab.b2[i] > 0) {
            TORCH_CHECK(tab.q2[i] != nullptr && tab.lut2[i] != nullptr,
                        "qgemm_gemv_multi: segment ", i, " declares "
                        "bitwidth2=", tab.b2[i], " but is missing its "
                        "stream-2 indices/lut");
        }
        TORCH_CHECK(tab.n_seg[i] > 0 && tab.n_seg[i] % 128 == 0,
                    "qgemm_gemv_multi: segment ", i, " N=", tab.n_seg[i],
                    " is not a positive multiple of 128 (the idxN layout)");
        TORCH_CHECK(tab.r_seg[i] >= 0 && tab.r_seg[i] <= 256,
                    "qgemm_gemv_multi: segment ", i, " residual rank ",
                    tab.r_seg[i], " is outside 0..256");
        if (tab.r_seg[i] > 0) {
            TORCH_CHECK(tab.resB[i] != nullptr && tab.resA[i] != nullptr,
                        "qgemm_gemv_multi: segment ", i, " needs resB and "
                        "resA at rank ", tab.r_seg[i]);
        }
        TORCH_CHECK(tab.c[i] != nullptr,
                    "qgemm_gemv_multi: segment ", i, " is missing its "
                    "output buffer");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tab.c[i]) % 16 == 0,
                    "qgemm_gemv_multi: segment ", i, " has a non-16-B-"
                    "aligned output row (rebuild the persistent buffer)");
        if (tab.signs[i] != nullptr) {
            ++rotated;
            TORCH_CHECK(reinterpret_cast<uintptr_t>(tab.signs[i]) % 16 == 0,
                        "qgemm_gemv_multi: segment ", i, " signs pointer "
                        "is not 16-B aligned");
        }
        if (tab.s[i] != nullptr) {
            TORCH_CHECK(tab.signs[i] != nullptr,
                        "qgemm_gemv_multi: segment ", i, " carries an AWQ "
                        "scale without rotation (the compensation only "
                        "exists as a correction to the rotated fold)");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(tab.s[i]) % 16 == 0,
                        "qgemm_gemv_multi: segment ", i, " s pointer is "
                        "not 16-B aligned");
        }
        tab.tile_base[i] = tab.tiles_total;
        tab.tiles_total += tab.n_seg[i] / 128;
        tab.r_off[i] = tab.r_total;
        tab.r_total += tab.r_seg[i];
    }
    tab.tile_base[count] = tab.tiles_total;
    TORCH_CHECK(tab.tiles_total >= 2,
                "qgemm_gemv_multi: the merged group needs >= 2 tiles");
    TORCH_CHECK(rotated == 0 || rotated == count,
                "qgemm_gemv_multi: mixed rotation presence (", rotated,
                " of ", count, " segments rotated) — the plain contract "
                "needs A pre-rotated for EVERY segment or none");

    const flute::FhtSegs segs = flute::fht_segments(K);
    if (rotated == count) {
        TORCH_CHECK(segs.n > 0 && (segs.off[segs.n - 1]
                                   + segs.len[segs.n - 1]) == K,
                    "qgemm_gemv_multi: the FHT segment table does not "
                    "tile K=", K);
    }

    const int smem_check = gemv_multi_smem(tab.tiles_total, K, tab.b1, tab.b2, tab.gs, tab.count,
        rotated == count);
    TORCH_CHECK(smem_check <= 99 * 1024,
                "qgemm_gemv_multi: K=", K, " (tiles=", tab.tiles_total,
                ", r_total=", tab.r_total, ") needs ", smem_check,
                " B of shared memory (over the SM_86 99 KB block limit)");

    const __half* a_p = reinterpret_cast<const __half*>(
        A.data_ptr<at::Half>());

    if (rotated == count) {
        launch_gemv_multi<true>(a_p, tab, K, segs);
    } else {
        launch_gemv_multi<false>(a_p, tab, K, segs);
    }
}

}  // namespace

// Public entrypoint (): the grouped multi-blob split-K GEMV — the QKV
// merge, heterogeneous (per-segment bitwidths / group sizes /
// rotation signs / AWQ scales in the 15-column segment table).
void qgemm_cutlass_gemv_multi(torch::Tensor A,
    torch::Tensor seg_table
) {
    qgemm_cutlass_gemv_multi_impl(A, seg_table);
}
