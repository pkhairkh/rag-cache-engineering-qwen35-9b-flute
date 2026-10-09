/**
 * include/flute/gemv.cuh
 *
 * Shared GEMV device primitives (the  decode-GEMV families).
 *
 * this header is the shared-translation-unit seam of the kernel
 * monolith split (kernel_streaming.cu 6290 lines ->
 * kernel_streaming.cu / kernel_gemv.cu / kernel_gemv_splitk.cu /
 * kernel_gemv_multi.cu / kernel_gemv_mlp.cu). Everything the GEMV
 * families share lives here ONCE:
 *
 *   - gemv_chunk_word / gemv_pair_codes: the idxN blob pair decode
 *     (the idxN.py NORMATIVE blob map transcription, verbatim).
 *   - gemv_kloop: the double-buffered K loop (B1/B2 templates,
 *     runtime-GS palette selection).
 *   - flute::GemvSegTab: the grouped-segment table (the
 *     heterogeneous extension - per-segment bitwidths, group sizes,
 *     rotation signs and AWQ scales).
 *   - FLUTE_HET_KLOOP_SWITCH: the runtime-pair dispatch for the *     heterogeneous multi/MLP kernels (a uniform per-CTA switch over
 *     the segment's (b1, b2); every arm is a compile-time kloop
 *     instantiation of the SAME loop the single-module split-K GEMV runs).
 *   - gemv_fht_prologue: the shared FHT boundary-fold prologue (the
 *     fp32 staging + butterfly + per-segment signs/AWQ fold into the
 *     smem x row) - the block that was open-coded identically in the
 *     split-K/multi/mlp kernels.
 *
 * The pair decode below is a transcription of idxN.py's blob map:
 *   tile (t = n/128, g = k/64) at blob offset ((t*K/64) + g)*1024*b bytes;
 *   thread (wx, lane) owns the 16*b-byte chunk at tile +
 *   (wx*512 + lane*16)*b; its 64 k-PAIRS p = kt*16 + v*4 + d*2 + s2 cover
 *   n = t*128 + wx*64 + v*16 + d*8 + (lane >> 2) and
 *   k = g*64 + kt*16 + 2*(lane & 3) + 8*s2 (+1).
 */

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "flute/mma.cuh"
#include "flute/fht.cuh"

namespace flute {

// ---------------------------------------------------------------------------
// The chunk's u32 word w (w in [0, 4*B)) - static-indexed: the pair loop
// is fully unrolled, so w is a compile-time constant and the uint4 select
// folds into a register read (no dynamic register indexing, no spill).
// ---------------------------------------------------------------------------
template <int B>
__device__ __forceinline__ uint32_t gemv_chunk_word(const uint4 (&qv)[B], int w
) {
    const uint4 v = qv[w >> 2];
    const int s = 8 * (w & 3);
    return (s == 0) ? v.x : ((s == 8) ? v.y : ((s == 16) ? v.z : v.w));
}

// ---------------------------------------------------------------------------
// The two codes of k-PAIR p (p in [0,64)) from one stream's chunk
// registers, LSB-first per idxN.py::_bit_pack_chunk:
//   b=4: byte p                       -> the two nibbles
//   b=2: byte p>>1, nibble p&1        -> bits 4*(p&1) and 4*(p&1)+2
//   b=1: byte p>>2, bits 2*(p&3)      -> the pair's two 1-bit fields
//   b=3: 24-bit LE group 3*(p>>2), 6-bit field 6*(p&3) -> two 3-bit fields
// ---------------------------------------------------------------------------
template <int B>
__device__ __forceinline__ void gemv_pair_codes(const uint4 (&qv)[B], int p, uint32_t& c0, uint32_t& c1
) {
    if constexpr (B == 4) {
        const uint32_t byte =
            (gemv_chunk_word(qv, p >> 2) >> (8 * (p & 3))) & 0xFFu;
        c0 = byte & 0xFu;
        c1 = byte >> 4;
    } else if constexpr (B == 2) {
        const uint32_t byte =
            (gemv_chunk_word(qv, p >> 3) >> (8 * ((p >> 1) & 3))) & 0xFFu;
        const int sh = 4 * (p & 1);
        c0 = (byte >> sh) & 0x3u;
        c1 = (byte >> (sh + 2)) & 0x3u;
    } else if constexpr (B == 1) {
        const uint32_t byte =
            (gemv_chunk_word(qv, p >> 4) >> (8 * ((p >> 2) & 3))) & 0xFFu;
        const int sh = 2 * (p & 3);
        c0 = (byte >> sh) & 0x1u;
        c1 = (byte >> (sh + 1)) & 0x1u;
    } else {                                   // B == 3
        const int byte_off = 3 * (p >> 2);     // the 24-bit LE group
        const int w0 = byte_off >> 2;
        const int sh = (byte_off & 3) * 8;
        uint32_t val = gemv_chunk_word(qv, w0) >> sh;
        if (sh > 8) val |= gemv_chunk_word(qv, w0 + 1) << (32 - sh);
        val &= 0xFFFFFFu;
        c0 = (val >> (6 * (p & 3))) & 0x7u;
        c1 = (val >> (6 * (p & 3) + 3)) & 0x7u;
    }
}

// ---------------------------------------------------------------------------
// The heterogeneous grouped-segment table . One kernel launch
// serves 1-4 segments that may DIFFER in (b1, b2, group_size, rotation
// signs, AWQ scale) - the deployed mixed-radix palette gives every tensor
// its own bit allocation (INSPECTION 2026-10-09: 0/32 MLP and 24/30 QKV
// groups were refused by the uniform-spec gate, "spec mismatch across
// components" / "gate/up spec mismatch" / "rotation seeds differ").
// Everything per segment:
//   q1/lut1/q2/lut2, resB/resA, bias, c      - the module's tensors
//   signs, s                                  - the fold operands (0 = none)
//   n_seg, r_seg, r_off, b1, b2, gs          - the segment's spec
// Cumulative tile_base[0..count] = tiles_total. Passed BY VALUE (kernel
// param, ~460 B - well under the 4 KB param budget).
// ---------------------------------------------------------------------------
struct GemvSegTab {
    const uint8_t* q1[4];
    const __half*  lut1[4];
    const uint8_t* q2[4];
    const __half*  lut2[4];
    const __half*  resB[4];
    const __half*  resA[4];
    const __half*  bias[4];
    __half*        c[4];
    const float*   signs[4];   // (K,) fold signs, kFht only; 0 = unrotated
    const float*   s[4];       // (K,) AWQ scale, kFht only; 0 = none
    int n_seg[4];
    int r_seg[4];
    int r_off[4];              // cumulative residual-rank offset
    int b1[4];                 // stream-1 bitwidth (1..4)
    int b2[4];                 // stream-2 bitwidth (0..4; 0 = single)
    int gs[4];                 // LUT group size (16..2048)
    int tile_base[5];          // cumulative tiles; [count] = tiles_total
    int count;
    int tiles_total;
    int r_total;
};

// ---------------------------------------------------------------------------
// The runtime-(b1, b2) K-loop dispatch for the heterogeneous kernels. The
// segment's pair is CTA-UNIFORM (every thread of the CTA resolved the same
// segment), so the switch is a single uniform branch - no divergence; each
// arm is a compile-time instantiation of the SAME gemv_kloop the
// single-module split-K GEMV runs. The full 20-pair table (b1 1..4 x b2 0..4)
// matches GEMV_SPLITK_PAIRS exactly; an out-of-table pair cannot reach
// the kernel (the host impl validates and TORCH_CHECKs loudly).
// ---------------------------------------------------------------------------
#define FLUTE_HET_KLOOP_CASE(b1v, b2v, kRegPal, ...)                        \
    case ((b1v) * 8 + (b2v)):                                               \
        gemv_kloop<b1v, b2v, kRegPal>(__VA_ARGS__);                        \
        break;

#define FLUTE_HET_KLOOP_SWITCH(kRegPal, pair_code, ...)                     \
    switch (pair_code) {                                                    \
        FLUTE_HET_KLOOP_CASE(1, 0, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(1, 1, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(1, 2, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(1, 3, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(1, 4, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(2, 0, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(2, 1, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(2, 2, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(2, 3, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(2, 4, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(3, 0, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(3, 1, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(3, 2, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(3, 3, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(3, 4, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(4, 0, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(4, 1, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(4, 2, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(4, 3, kRegPal, __VA_ARGS__)                    \
        FLUTE_HET_KLOOP_CASE(4, 4, kRegPal, __VA_ARGS__)                    \
        default:                                                            \
            __trap();   /* unreachable: the host validated the pair */      \
            break;                                                          \
    }

}  // namespace flute

// ---------------------------------------------------------------------------
// The double-buffered, palette-parameterized K loop (verbatim; lives
// at global template scope so every TU instantiates identical arms).
// kRegPal selects the palette serving: true = the register palette +
// shfl.sync.idx (GS >= 64, one LUT group per warp); false = the per-CTA
// shared palette (GS 16/32 - the warp's 64 rows span 2-4 groups). The g
// stride stays 4 (the j4 K-quarter warp partition, verbatim); g is
// LOCAL to the split.
// ---------------------------------------------------------------------------
template <int B1, int B2, bool kRegPal>
__device__ __forceinline__ void gemv_kloop(const uint8_t* __restrict__ cb1,
    const uint8_t* __restrict__ cb2,
    const uint32_t* __restrict__ x2,
    int Gc, int j4, int lane, int wx, int gs_shift,
    float pal1_r, float pal2_r,
    const __half* __restrict__ pal1_s,
    const __half* __restrict__ pal2_s,
    float (&acc)[8]
) {
    uint4 qv1c[B1];
    uint4 qv2c[B2 > 0 ? B2 : 1];
    uint4 qv1n[B1];
    uint4 qv2n[B2 > 0 ? B2 : 1];

    // the prologue load: g = j4 (this warp's first g-tile in the split)
    if (j4 < Gc) {
        #pragma unroll
        for (int u = 0; u < B1; ++u)
            flute::ldg_nc_evict_first_v4(qv1c[u], cb1 + (size_t)j4 * (1024 * B1) + u * 16);
        if constexpr (B2 > 0) {
            #pragma unroll
            for (int u = 0; u < B2; ++u)
                flute::ldg_nc_evict_first_v4(qv2c[u], cb2 + (size_t)j4 * (1024 * B2) + u * 16);
        }
    }

    for (int g = j4; g < Gc; g += 4) {
        // ---- prefetch the g+4 chunk pair BEFORE the g compute ----------
        const int gn = g + 4;
        const bool have_nxt = (gn < Gc);
        if (have_nxt) {
            #pragma unroll
            for (int u = 0; u < B1; ++u)
                flute::ldg_nc_evict_first_v4(qv1n[u], cb1 + (size_t)gn * (1024 * B1) + u * 16);
            if constexpr (B2 > 0) {
                #pragma unroll
                for (int u = 0; u < B2; ++u)
                    flute::ldg_nc_evict_first_v4(qv2n[u], cb2 + (size_t)gn * (1024 * B2) + u * 16);
            }
        }

        // ---- the 64-pair compute on the CURRENT buffers (body) ----
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                #pragma unroll
                for (int d = 0; d < 2; ++d) {
                    // the accumulator row's LUT group (compile-time in v,
                    // d; runtime GS): (v*16 + d*8) never straddles a group
                    // for GS >= 16 because d*8 + 7 < 16.
                    const int grp_vd = (wx * 64 + v * 16 + d * 8) >> gs_shift;
                    #pragma unroll
                    for (int s2 = 0; s2 < 2; ++s2) {
                        const int p = kt * 16 + v * 4 + d * 2 + s2;
                        const int xi =
                            g * 32 + kt * 8 + (lane & 3) + (s2 << 2);
                        const __half2 xh2 =
                            *reinterpret_cast<const __half2*>(&x2[xi]);
                        const float2 xf = __half22float2(xh2);
                        const int ai = v * 2 + d;
                        {
                            uint32_t c0, c1;
                            flute::gemv_pair_codes<B1>(qv1c, p, c0, c1);
                            const float w0 = kRegPal
                                ? __shfl_sync(0xffffffffu, pal1_r, (int)c0)
                                : __half2float(pal1_s[(size_t)(grp_vd << B1) | c0]);
                            const float w1 = kRegPal
                                ? __shfl_sync(0xffffffffu, pal1_r, (int)c1)
                                : __half2float(pal1_s[(size_t)(grp_vd << B1) | c1]);
                            acc[ai] = fmaf(xf.x, w0, acc[ai]);
                            acc[ai] = fmaf(xf.y, w1, acc[ai]);
                        }
                        if constexpr (B2 > 0) {
                            uint32_t c0, c1;
                            flute::gemv_pair_codes<B2>(qv2c, p, c0, c1);
                            const float w0 = kRegPal
                                ? __shfl_sync(0xffffffffu, pal2_r, (int)c0)
                                : __half2float(pal2_s[(size_t)(grp_vd << B2) | c0]);
                            const float w1 = kRegPal
                                ? __shfl_sync(0xffffffffu, pal2_r, (int)c1)
                                : __half2float(pal2_s[(size_t)(grp_vd << B2) | c1]);
                            acc[ai] = fmaf(xf.x, w0, acc[ai]);
                            acc[ai] = fmaf(xf.y, w1, acc[ai]);
                        }
                    }
                }
            }
        }

        // ---- advance: nxt becomes cur (register move only) ------------
        if (have_nxt) {
            #pragma unroll
            for (int u = 0; u < B1; ++u) qv1c[u] = qv1n[u];
            if constexpr (B2 > 0) {
                #pragma unroll
                for (int u = 0; u < B2; ++u) qv2c[u] = qv2n[u];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The shared FHT boundary-fold prologue (transcription; the block the
// split-K/multi/mlp kernels each open-coded). Stages the WHOLE-K FHT of the
// input row into fp32 shared memory segment by segment, then writes the
// split's k-slice of the rotated row (with the per-segment signs / AWQ
// compensation) into xrow as fp16.
//
// signs_seg / s_seg are PER-SEGMENT pointers (nullptr = absent) -
// the heterogeneous rotation contract (components with different seeds
// merge in one launch; each CTA serves exactly one segment). s_seg null
// handling is a CTA-uniform branch (predicated, no divergence).
//
// fstage must be float* shared with >= 4*b_max bytes free; xrow is the
// [Kc] __half slice at smem base. THREADS = 256.
// ---------------------------------------------------------------------------
template <bool kFht>
__device__ __forceinline__ void gemv_fht_prologue(const __half* __restrict__ A,
    const float* __restrict__ signs_seg,
    const float* __restrict__ s_seg,
    flute::FhtSegs segs,
    int K, int Kc, int k0,
    __half* __restrict__ xrow,
    float* __restrict__ fstage
) {
    if constexpr (kFht) {
        for (int sgi = 0; sgi < segs.n; ++sgi) {
            const int off = segs.off[sgi];
            const int b = segs.len[sgi];
            const float inv_sqrt_b = rsqrtf(static_cast<float>(b));

            for (int j = threadIdx.x; j < b; j += 256) {
                float v = __half2float(A[off + j]);
                if (s_seg != nullptr) {
                    v = v * s_seg[off + j];
                }
                fstage[j] = v;
            }
            __syncthreads();

            flute::fht_block<256>(fstage, b);   // shift/mask + local

            const int lo = (off > k0) ? off : k0;
            const int hi = (off + b < k0 + Kc) ? (off + b) : (k0 + Kc);
            for (int j = lo - off + threadIdx.x; j < hi - off;
                 j += 256) {
                const float v = fstage[j] * signs_seg[off + j] * inv_sqrt_b;
                if (s_seg != nullptr) {
                    xrow[off + j - k0] = __float2half(v / s_seg[off + j]);
                } else {
                    xrow[off + j - k0] = __float2half(v);
                }
            }
            __syncthreads();   // the slice published; staging reusable
        }
    } else {
        uint4* x2v = reinterpret_cast<uint4*>(xrow);
        for (int i = threadIdx.x; i < (Kc >> 3); i += 256) {
            flute::ldg_nc_evict_first_v4(x2v[i], A + (size_t)k0 + ((size_t)i << 3));
        }
        __syncthreads();
    }
}
