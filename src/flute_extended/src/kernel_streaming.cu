/**
 * src/kernel_streaming.cu
 *
 * The streaming/prefill GEMM family (Tensor Cores, SM_80+) and the * dual-stream fused decode kernel — extracted from the former kernel_streaming.cu monolith (6290 lines) into its own
 * translation unit. See docs/A10G_DECODE_INVESTIGATION.md section 14 for
 * the split map and the compile-time rationale (per-TU ninja jobs +
 * FLUTE_CUDA_ARCHES).
 *
 *   C[M, N] = A[M, K] @ W[N, K]^T  (W never exists in FP16 — the LUT is
 *   dequantized on the fly into shared memory / fragments; see the
 *   original file header, preserved in docs).
 *
 * Entry points (flute/entrypoints.h):
 *   qgemm_cutlass_streaming  — the streaming GEMM (prefill/PPL)
 *   qgemm_cutlass_dual_stream — the dual-stream fused decode (M <= 16)
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

// ---------------------------------------------------------------------------
// Tile configuration
// ---------------------------------------------------------------------------
// BK (K-tile depth) and GS (LUT group size) are independent: the LUT is
// indexed by N group only, never by k, so BK may be 32 or 64 for either GS.
// The dispatch below guarantees K % BK == 0 for whichever config runs.
//  GS extension: the LUT group size is now one of 16/32/64/128/
// 256/512 (the operator's full-GS-range campaign). The in-kernel group
// index is computed from the ABSOLUTE row n (never n_local/GS), so a
// 128-row N-tile may straddle a group boundary when GS > BN — the
// staged LUT rows and the per-thread offsets both key off
// (n0 + n_local)/GS - n0/GS, which degenerates to the older
// n_local/GS arithmetic verbatim whenever BN % GS == 0.
template <int GS>
struct gs_supported : std::bool_constant<
    GS == 16 || GS == 32 || GS == 64 || GS == 128 || GS == 256 || GS == 512 ||
    GS == 1024 || GS == 2048> {};

template <int BM_, int BN_, int BK_, int GS_>
struct TileConfig {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;                  // K-tile depth (32 or 64)
    static constexpr int GS = GS_;                  // LUT group size (16..2048; 1024/2048)

    static constexpr int WM = BM / 2;               // warp rows: 64 (BK=32) / 32 (BK=64)
    static constexpr int WN = 64;                   // warp cols
    static constexpr int WARPS_M = BM / WM;         // 2
    static constexpr int WARPS_N = BN / WN;         // 2
    static constexpr int WARPS  = WARPS_M * WARPS_N;// 4
    static constexpr int THREADS = WARPS * 32;      // 128

    static constexpr int M_TILES = WM / 16;         // 4 (BK=32) / 2 (BK=64)
    static constexpr int N_TILES = WN / 8;          // 8
    static constexpr int K_TILES = BK / 16;         // 2 (BK=32) / 4 (BK=64)

    // Upper bound on LUT groups touched by one block (one extra row for
    // the N-boundary / a straddled group). Depends on GS only. GS <= BN
    // (BN % GS == 0): exactly BN/GS + 1, the older bound; GS > BN: the
    // 128-row tile straddles at most one group boundary -> 2.
    static constexpr int LUT_GRPS = ((BN + GS - 1) / GS) + 1;  // 9/5/3/2/2/2/2/2

    // ---- pipeline / layout policy ------------------------------------------
    // All configs double-buffer sA AND sW (one barrier per K-tile).
    //   BK=32: padded rows (BK+8 halves = 80 B pitch) — every ldmatrix row
    //          hits all 32 banks exactly once; no swizzle needed.
    //   BK=64: 16-byte-block XOR swizzle (block ^ (row & 7)), zero padding —
    //          conflict-free ldmatrix rows and a 16 B row pitch that keeps
    //          cp.async destinations aligned.
    static constexpr int  SA_BUFFERS = 2;
    static constexpr int  SW_BUFFERS = 2;
    static constexpr int  SA_LD = (BK_ == 32) ? (BK_ + 8) : BK_;   // halves/row
    static constexpr int  SW_LD = (BK_ == 32) ? (BK_ + 8) : BK_;   // halves/row
    static constexpr int  SA_XOR = (BK_ == 32) ? 1 : (BK_ / 8);    // 16B-block XOR mask (1 = off)
    static constexpr int  SW_XOR = (BK_ == 32) ? 1 : (BK_ / 8);

    static constexpr int QW  = BK_ / 8;             // u32 words of packed Q per W row (4 or 8)
    static constexpr int AF4 = (BM_ * BK_) / 8 / THREADS;  // A-tile 16 B chunks per thread (4)

    static constexpr int SA_ELEMS  = SA_BUFFERS * BM * SA_LD;   // halves (total)
    static constexpr int SW_ELEMS  = SW_BUFFERS * BN * SW_LD;   // halves (total)
    static constexpr int SA_BUF_ELEMS = BM * SA_LD;             // per sA buffer
    static constexpr int SW_BUF_ELEMS = BN * SW_LD;             // per sW buffer
    static constexpr int LUT_WORDS = LUT_GRPS * 64;             // u32 (4 lane-copies interleaved)
    static constexpr int SMEM_BYTES = (SA_ELEMS + SW_ELEMS) * 2 + LUT_WORDS * 4;

    // ---- fragment-direct path constants (E1+E2+E3) ------------------------
    // Layout "idx4" (tools/repack_idx4.py): no sW, no smem Q, no
    // B-operand ldmatrix. Shared memory = FD_STAGES A buffers (E3) + the
    // 256-entry byte-indexed paired LUT (E1, PLUT_WORDS u32). FD_WORDS =
    // u32 Q words per thread per K-tile (one 64-B chunk per 64-k group).
    static constexpr int FD_STAGES  = 3;                 // sA pipeline depth
    static constexpr int FD_WORDS   = BK_ / 4;            // 8 (BK=32) / 16 (BK=64)
    static constexpr int PLUT_WORDS = LUT_GRPS * 256;    // paired LUT, u32
    static constexpr int FD_SMEM_BYTES =
        FD_STAGES * SA_BUF_ELEMS * 2 + PLUT_WORDS * 4;
    static constexpr int FD_CHUNK_BYTES = 64;            // per thread per 64-k group
    static constexpr int FD_TILE_BYTES  = 4096;          // per (128-row, 64-k) tile
};

// ---------------------------------------------------------------------------
// Global -> shared: asynchronous A-tile copy (one 16 B chunk per step).
// cp.async has no predicated fill, so out-of-range M rows are zero-filled
// with an ordinary 16 B shared store (the barrier at the end of the tile
// orders both).
// ---------------------------------------------------------------------------
template <typename Cfg>
__device__ __forceinline__ void a_copy_async(const __half* __restrict__ A, int m0, int k0, int M, int K,
    __half* sA_buf
) {
    #pragma unroll
    for (int f = 0; f < Cfg::AF4; ++f) {
        const int i        = f * Cfg::THREADS + threadIdx.x;  // 16 B chunk index
        const int m_local  = (i * 8) / Cfg::BK;
        const int k_local  = (i * 8) % Cfg::BK;               // multiple of 8
        __half* dst = sA_buf + (size_t)m_local * Cfg::SA_LD +
                      flute::swz_col16(m_local, k_local, Cfg::SA_XOR);
        if (m0 + m_local < M) {
            // 16 B aligned: A base is host-checked (auto-cloned if needed);
            // K % BK == 0 makes every row pitch a multiple of 64 B; k_local
            // is a multiple of 8 halves.
            flute::cp_async_16(dst, A + (size_t)(m0 + m_local) * K + k0 + k_local);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// ---------------------------------------------------------------------------
// Q prefetch: BK/2 bytes (one full packed row) as 1-2 uint4 vectors
// ---------------------------------------------------------------------------
template <typename Cfg>
__device__ __forceinline__ void q_prefetch(const uint8_t* q_tile, uint32_t (&q)[Cfg::QW]
) {
    if constexpr (Cfg::QW == 4) {
        *reinterpret_cast<uint4*>(q) = *reinterpret_cast<const uint4*>(q_tile);
    } else {
        reinterpret_cast<uint4*>(q)[0] = reinterpret_cast<const uint4*>(q_tile)[0];
        reinterpret_cast<uint4*>(q)[1] = reinterpret_cast<const uint4*>(q_tile)[1];
    }
}

// ---------------------------------------------------------------------------
// Sub-4-bit extension (idxN family: B in {1, 2, 3}).
//
// Same approach, same style as the 4-bit paths above, at every width:
//   * legacy path (q_layout=0, LOGICAL packed rows [N, K*B/8]):
//     thread t still owns W row t and dequantizes a full K-tile into sW;
//     only the field extraction from the packed words is width-specific.
//   * fragment-direct path (q_layout=1, idxN blob): the same 64-thread
//     chunk tiling of one (128-row, 64-k) tile; each thread's chunk is
//     16*B bytes holding its 64 k-pairs (128 elements), pair p at bit
//     2*B*p, exactly the layout flute_extended/idxN.py produces (b=4
//     reduces to idx4 byte-for-byte, which is why the 4-bit
//     kernels above stay untouched and keep their binaries).
//
// Paired-LUT tables (why the fd path stays at one LDS per fragment u32):
//   * B=1: four 256-entry tables — table t holds the pair at byte bits
//     (2t, 2t+1): U_t[b] = LUT[(b >> 2t) & 1] | LUT[(b >> (2t+1)) & 1] << 16
//   * B=2: two 256-entry tables — table t holds the pair at nibble t:
//     T_t[b] = LUT[(b >> 4t) & 3] | LUT[(b >> (4t+2)) & 3] << 16
//   * B=3: one 64-entry table indexed by the 6-bit pair field itself:
//     V[x] = LUT[x & 7] | LUT[(x >> 3) & 7] << 16
// Per mma step (kt, v) the thread needs its 4 k-pairs (d, s2 grid):
// pair p = kt*16 + v*4 + d*2 + s2 (half-local within a BK=32 K-tile).
// ---------------------------------------------------------------------------
template <typename Cfg, int B>
struct SubCfg {
    static_assert(B == 1 || B == 2 || B == 3,
                  "SubCfg is the sub-4-bit extension; 4-bit runs the "
                  "legacy kernels above");
    static constexpr int PAL = 1 << B;               // LUT entries per group
    static constexpr int QW_B = Cfg::BK * B / 32;    // legacy Q words / K-tile / row
    static constexpr int FDW = Cfg::BK * B / 16;     // fd Q words / K-tile / thread
    static constexpr int CHUNK_BYTES = 16 * B;       // per thread per 64-k group
    static constexpr int TILE_BYTES = 1024 * B;      // per (128-row, 64-k) tile
    static constexpr int PLUT_TABS = (B == 3) ? 1 : (4 / B);
    static constexpr int PLUT_ENT = (B == 3) ? 64 : 256;
    static constexpr int PLUT_GRP = PLUT_TABS * PLUT_ENT;   // u32 per LUT group
    static constexpr int PLUT_WORDS = Cfg::LUT_GRPS * PLUT_GRP;
    static constexpr int FD_SMEM =
        Cfg::FD_STAGES * Cfg::SA_BUF_ELEMS * 2 + PLUT_WORDS * 4;
    // uint4 prefetch is legal only when the per-K-tile Q share is a whole
    // number of 16 B vectors AND its offset inside the chunk is 16 B
    // aligned: BK=64 always reads the whole (offset-0) chunk; BK=32 reads
    // a BK*B/4-byte half, aligned only for B=2 (16 B). B=1 (8 B) and
    // B=3 (24 B) fall back to 4 B-aligned scalar u32 loads.
    static constexpr bool FD_VEC4 =
        (FDW % 4 == 0) &&
        (Cfg::BK == 64 || ((Cfg::BK * B / 4) % 16) == 0);
};

// Legacy Q prefetch at B bits: QW_B u32 words (the tile start inside the
// row is 4 B aligned — k0*B is a multiple of 8 for every k0 the K loop
// produces, and the row stride K*B/8 is a multiple of 4).
template <typename Cfg, int B>
__device__ __forceinline__ void q_prefetch_sub4(const uint8_t* q_tile, uint32_t (&q)[SubCfg<Cfg, B>::QW_B]
) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(q_tile);
    #pragma unroll
    for (int w = 0; w < SubCfg<Cfg, B>::QW_B; ++w) q[w] = src[w];
}

template <int QW>
__device__ __forceinline__ void q_zero(uint32_t (&q)[QW]) {
    #pragma unroll
    for (int w = 0; w < QW; ++w) q[w] = 0u;
}

// ---------------------------------------------------------------------------
// Dequantize one sub-4-bit packed-Q register set into sW (legacy path).
// Pair pp covers k = k0 + 2*pp; its 2*B bits sit at bit 2*B*pp of the
// tile's word stream (the tile start is word-aligned, so the offset is
// exact). The pair dequantizes into ONE sW u32 {W[k], W[k+1]} exactly
// like the 4-bit path — sW, the swizzle, the mma phase and the epilogue
// are width-independent. For B == 3 the 6-bit field may span two words;
// a spanning field ends strictly inside the tile (the last pair ends
// exactly at the tile boundary), so q[w0 + 1] is always in range.
// ---------------------------------------------------------------------------
template <typename Cfg, int B>
__device__ __forceinline__ void dequant_tile_sub4(const uint32_t (&q)[SubCfg<Cfg, B>::QW_B],
    const uint32_t* sLUT32,
    int lut_off,          // g * 64 + (tid & 3)   [precomputed]
    int row,              // this thread's row within the tile (n_local)
    uint32_t* sW_buf      // current sW buffer, u32 view
) {
    const int ldw = Cfg::SW_LD / 2;                // row stride in u32 words
    uint32_t* dst = sW_buf + (size_t)row * ldw;
    #pragma unroll
    for (int pp = 0; pp < Cfg::BK / 2; ++pp) {
        uint32_t v0, v1;
        if constexpr (B == 1) {
            const int bit = 2 * pp;
            const uint32_t w = q[bit >> 5];
            v0 = (w >> (bit & 31)) & 1u;
            v1 = (w >> ((bit & 31) + 1)) & 1u;
        } else if constexpr (B == 2) {
            const int bit = 4 * pp;
            const uint32_t w = q[bit >> 5];
            v0 = (w >> (bit & 31)) & 3u;
            v1 = (w >> ((bit & 31) + 2)) & 3u;
        } else {                                   // B == 3
            const int bit = 6 * pp;
            const int w0 = bit >> 5;
            const int s = bit & 31;
            uint32_t f = q[w0] >> s;
            if (s + 6 > 32) f |= q[w0 + 1] << (32 - s);
            v0 = f & 7u;
            v1 = (f >> 3) & 7u;
        }
        const uint32_t lo = sLUT32[lut_off + (v0 << 2)];
        const uint32_t hi = sLUT32[lut_off + (v1 << 2)];
        dst[flute::swz_word(row, pp, Cfg::SW_XOR)] =
            flute::prmt_merge_lo_hi(lo, hi);
    }
}

// ---------------------------------------------------------------------------
// Dequantize one packed-Q register set into sW (one full W row per thread).
//   W[n, k   ] = LUT[group(n), byte(k>>1) & 0x0F]   (low nibble  -> even k)
//   W[n, k+1 ] = LUT[group(n), byte(k>>1) >> 4  ]   (high nibble -> odd  k)
// group(n) = n / GS — the group index depends only on the row, never on k,
// so this routine is identical for every BK.
// The two halves are merged into one u32 with a single prmt.b32 (0x5410):
//   out = { hi[15:0], lo[15:0] }  ==  lo | (hi << 16)
// sLUT32 layout: [group][idx*4 + copy], copy = tid & 3 (bank spreading).
// ---------------------------------------------------------------------------
template <typename Cfg>
__device__ __forceinline__ void dequant_tile(const uint32_t (&q)[Cfg::QW],
    const uint32_t* sLUT32,
    int lut_off,          // g * 64 + (tid & 3)   [precomputed]
    int row,              // this thread's row within the tile (n_local)
    uint32_t* sW_buf      // current sW buffer, u32 view
) {
    const int ldw = Cfg::SW_LD / 2;                // row stride in u32 words
    uint32_t* dst = sW_buf + (size_t)row * ldw;
    #pragma unroll
    for (int wq = 0; wq < Cfg::QW; ++wq) {
        const uint32_t qw = q[wq];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (qw >> (8 * b)) & 0xFFu;
            const uint32_t lo = sLUT32[lut_off + ((byte & 0x0Fu) << 2)];
            const uint32_t hi = sLUT32[lut_off + (((byte >> 4) & 0x0Fu) << 2)];
            const int w = wq * 4 + b;
            dst[flute::swz_word(row, w, Cfg::SW_XOR)] =
                flute::prmt_merge_lo_hi(lo, hi);
        }
    }
}

// ---------------------------------------------------------------------------
// Fragment-direct path (E1+E2+E3) — requires the offline-repacked
// "idx4" Q layout (tools/repack_idx4.py is the normative spec).
// ---------------------------------------------------------------------------
// One packed byte = one B-fragment u32 of mma.m16n8k16 ({W[k], W[k+1]} of
// one output row), and the repack permutes the .idx4 bytes so that every
// thread's share of a 64-k tile is one contiguous 64-byte chunk:
//
//   chunk(t, g, wx, lane) at ((t*K/64) + g)*4096 + wx*2048 + lane*64
//   chunk byte (kt, v, d, s2) at kt*16 + v*4 + d*2 + s2  ==  original
//       Q[n, kp],  n  = t*128 + wx*64 + v*16 + d*8 + (lane >> 2)
//                  kp = g*32 + kt*8 + (lane & 3) + 4*s2
//
// so u32 word q[kt*4 + v] of the chunk holds the four B-fragment registers
// of mma step (kt, v): dequant becomes ONE paired-LUT LDS per byte, writing
// straight into the register mma.sync consumes (FLUTE, arXiv 2407.10960,
// sections 3.1-3.2; Marlin's dequant-into-registers pattern). sW, its
// STS/ldmatrix traffic and the dequant->MMA barrier coupling all vanish.
// BK=32 configs read the two 32-byte halves of a chunk per K-tile; BK=64
// configs read the whole chunk — both stay 16 B aligned.
// ---------------------------------------------------------------------------

// Stage the paired LUT (E1): sPLUT[g][b] = LUT[grp_first+g][b & 15] in the
// low half and LUT[grp_first+g][b >> 4] in the high half — the exact u32
// that prmt_merge_lo_hi used to assemble. 256 entries per group row; the
// raw LUT row (32 B) stays in L1 while the table is built.
template <typename Cfg>
__device__ __forceinline__ void plut_stage(const __half* __restrict__ LUT, uint32_t* __restrict__ sPLUT,
    int grp_first, int n_grps, int tid
) {
    for (int i = tid; i < n_grps * 256; i += Cfg::THREADS) {
        const int gl = i >> 8;
        const int b  = i & 255;
        const uint16_t* row = reinterpret_cast<const uint16_t*>(LUT) +
                              (size_t)(grp_first + gl) * 16;
        const uint32_t lo = row[b & 15];
        const uint32_t hi = row[b >> 4];
        sPLUT[i] = lo | (hi << 16);
    }
}

// Prefetch this thread's Q chunk (FD_WORDS u32 = FD_WORDS/4 uint4 vectors)
// with the L2 evict_first hint (E5) — Q is stream-once data.
template <typename Cfg>
__device__ __forceinline__ void q_fd_prefetch(const uint8_t* chunk, uint32_t (&q)[Cfg::FD_WORDS]
) {
    uint4* qq = reinterpret_cast<uint4*>(q);
    const uint4* src = reinterpret_cast<const uint4*>(chunk);
    #pragma unroll
    for (int v = 0; v < Cfg::FD_WORDS / 4; ++v) {
        flute::ldg_nc_evict_first_v4(qq[v], src + v);
    }
}

// This thread's chunk address for K-tile `ktile` (K-tiles are BK deep).
// BK=32: k-tile j = half (j&1) of k-group (j>>1);  BK=64: k-tile j = group j.
template <typename Cfg>
__device__ __forceinline__ const uint8_t* q_fd_chunk(const uint8_t* chunk_base, int ktile
) {
    // For idx4 (idx4) layout, each (wx, lane) thread owns a 64-byte chunk
    // per (128-row, 64-k) tile. Within one 64-k group:
    // - Bytes [0, FD_WORDS*4) cover the first BK K values
    // - Bytes [FD_WORDS*4, 64) cover the next BK K values (for BK=32)
    // The tile offset selects which 64-k group tile we're accessing.
    const int kp0 = ktile * (Cfg::BK / 2);          // first k-pair of the tile
    const size_t kgroup_off = (size_t)(kp0 >> 5) * Cfg::FD_TILE_BYTES;
    const size_t half_off = (size_t)((kp0 & 31) >> 4) * (Cfg::FD_WORDS * 4);
    return chunk_base + kgroup_off + half_off;
}

// Tensor-core phase, fragment-direct: consume one sA buffer + one Q register
// set. Per (kt, v): one u32 of Q -> four bytes -> four paired-LUT LDS -> the
// B fragments of mma n-tiles 2v and 2v+1. The mma ORDER is identical to the
// legacy mma_phase (kt outer, nt inner, mt innermost), so the FP32
// accumulation sequence — and therefore the result — is bit-identical.
template <typename Cfg>
__device__ __forceinline__ void fd_mma_phase(const __half* sA_buf, const uint32_t* __restrict__ sPLUT,
    const uint32_t (&q)[Cfg::FD_WORDS],
    const int (&plut_off)[4][2],          // [v][d] = LUT group row * 256
    int wy, int lane,
    float (&acc)[Cfg::M_TILES][Cfg::N_TILES][4]
) {
    #pragma unroll
    for (int kt = 0; kt < Cfg::K_TILES; ++kt) {
        uint32_t afrag[Cfg::M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
            flute::ldmatrix_x4(afrag[mt][0], afrag[mt][1], afrag[mt][2], afrag[mt][3],
                flute::ldmatrix_a_addr(sA_buf, Cfg::SA_LD,
                                       wy * Cfg::WM + mt * 16, kt * 16,
                                       lane, Cfg::SA_XOR));
        }
        #pragma unroll
        for (int v = 0; v < Cfg::N_TILES / 2; ++v) {
            const uint32_t w = q[kt * 4 + v];
            const uint32_t b0 = sPLUT[plut_off[v][0] + ((w) & 0xFFu)];
            const uint32_t b1 = sPLUT[plut_off[v][0] + ((w >>  8) & 0xFFu)];
            const uint32_t b2 = sPLUT[plut_off[v][1] + ((w >> 16) & 0xFFu)];
            const uint32_t b3 = sPLUT[plut_off[v][1] + (w >> 24)];
            #pragma unroll
            for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
                const uint32_t bL[2] = {b0, b1};
                const uint32_t bR[2] = {b2, b3};
                flute::mma_m16n8k16_f32acc(afrag[mt], bL, acc[mt][2 * v    ]);
                flute::mma_m16n8k16_f32acc(afrag[mt], bR, acc[mt][2 * v + 1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fragment-direct sub-4-bit helpers (idxN blob; flute_extended/idxN.py is
// the normative producer). One 16*B-byte chunk per (wx, lane) thread per
// (128-row, 64-k) tile; chunk byte layout and the (kt, v, d, s2) pair
// grid are documented in SubCfg above.
// ---------------------------------------------------------------------------

// Stage the paired LUTs (E1 generalized). sPLUT layout per group:
//   B=1: [table 0 (256) | table 1 | table 2 | table 3]
//   B=2: [table 0 (256) | table 1]
//   B=3: [pair table (64)]
// The raw LUT row (2^B fp16 entries, <= 16 B) stays in L1 while built.
template <typename Cfg, int B>
__device__ __forceinline__ void plut_stage_sub4(const __half* __restrict__ LUT, uint32_t* __restrict__ sPLUT,
    int grp_first, int n_grps, int tid
) {
    using SC = SubCfg<Cfg, B>;
    for (int i = tid; i < n_grps * SC::PLUT_GRP; i += Cfg::THREADS) {
        const int gl = i / SC::PLUT_GRP;
        const int j  = i - gl * SC::PLUT_GRP;
        const uint16_t* row = reinterpret_cast<const uint16_t*>(LUT) +
                              (size_t)(grp_first + gl) * SC::PAL;
        if constexpr (B == 3) {
            // pair table: x = (v1 << 3) | v0 (the 6-bit pair field)
            const uint32_t lo = row[j & 7];
            const uint32_t hi = row[(j >> 3) & 7];
            sPLUT[i] = lo | (hi << 16);
        } else {
            const int t = j >> 8;                    // table id
            const int b = j & 255;                   // byte index
            if constexpr (B == 1) {
                const uint32_t lo = row[(b >> (2 * t)) & 1];
                const uint32_t hi = row[(b >> (2 * t + 1)) & 1];
                sPLUT[i] = lo | (hi << 16);
            } else {                                 // B == 2
                const uint32_t lo = row[(b >> (4 * t)) & 3];
                const uint32_t hi = row[(b >> (4 * t + 2)) & 3];
                sPLUT[i] = lo | (hi << 16);
            }
        }
    }
}

// Prefetch this thread's Q chunk share for one K-tile (FDW u32 words),
// with the same .nc (L2 evict-hint-free) stream-once load discipline as
// the 4-bit path (E5). uint4 vectors only where the half share is 16 B
// aligned (SubCfg::FD_VEC4); otherwise 4 B-aligned scalar u32 loads.
template <typename Cfg, int B>
__device__ __forceinline__ void q_fd_prefetch_sub4(const uint8_t* chunk, uint32_t (&q)[SubCfg<Cfg, B>::FDW]
) {
    using SC = SubCfg<Cfg, B>;
    if constexpr (SC::FD_VEC4) {
        uint4* qq = reinterpret_cast<uint4*>(q);
        const uint4* src = reinterpret_cast<const uint4*>(chunk);
        #pragma unroll
        for (int v = 0; v < SC::FDW / 4; ++v) {
            flute::ldg_nc_evict_first_v4(qq[v], src + v);
        }
    } else {
        #pragma unroll
        for (int w = 0; w < SC::FDW; ++w) {
            flute::ldg_nc_evict_first_u32(q[w], chunk + 4 * w);
        }
    }
}

// This thread's chunk address for K-tile `ktile` (K-tiles are BK deep):
// same half-selection arithmetic as the 4-bit path — the pair sequence is
// kt-major, so a BK=32 K-tile is exactly the first/second 16-pair half of
// the 64-k group, i.e. the BK*B/4-byte half of the 16*B-byte chunk.
template <typename Cfg, int B>
__device__ __forceinline__ const uint8_t* q_fd_chunk_sub4(const uint8_t* chunk_base, int ktile
) {
    using SC = SubCfg<Cfg, B>;
    const int kp0 = ktile * (Cfg::BK / 2);          // first k-pair of the tile
    const size_t kgroup_off = (size_t)(kp0 >> 5) * SC::TILE_BYTES;
    const size_t half_off = (size_t)((kp0 & 31) >> 4) * (SC::FDW * 4);
    return chunk_base + kgroup_off + half_off;
}

// Tensor-core phase, fragment-direct, sub-4-bit: consume one sA buffer +
// one Q register set. Per (kt, v): extract the 4 pair fields of the mma
// step, look each up in the paired tables, and feed the SAME mma call
// order as fd_mma_phase (bL = pairs (d=0, s2={0,1}) with group
// plut_off[v][0], bR = (d=1) with plut_off[v][1]) — the accumulation
// sequence, and therefore the result, follows the 4-bit contract.
//   B=1: byte (kt*4 + v) of the half's word stream (8 pairs per word).
//   B=2: 16-bit field at bit 16*(kt*2 + v)  (2 pairs per byte).
//   B=3: 24-bit field at bit 24*(kt*4 + v)  (4 pairs per 3 bytes; the
//        field may span two q words — a spanning field ends strictly
//        inside the chunk, so q[w0 + 1] is always in range).
template <typename Cfg, int B>
__device__ __forceinline__ void fd_mma_phase_sub4(const __half* sA_buf, const uint32_t* __restrict__ sPLUT,
    const uint32_t (&q)[SubCfg<Cfg, B>::FDW],
    const int (&plut_off)[4][2],          // [v][d] = LUT group row * PLUT_GRP
    int wy, int lane,
    float (&acc)[Cfg::M_TILES][Cfg::N_TILES][4]
) {
    #pragma unroll
    for (int kt = 0; kt < Cfg::K_TILES; ++kt) {
        uint32_t afrag[Cfg::M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
            flute::ldmatrix_x4(afrag[mt][0], afrag[mt][1], afrag[mt][2], afrag[mt][3],
                flute::ldmatrix_a_addr(sA_buf, Cfg::SA_LD,
                                       wy * Cfg::WM + mt * 16, kt * 16,
                                       lane, Cfg::SA_XOR));
        }
        #pragma unroll
        for (int v = 0; v < Cfg::N_TILES / 2; ++v) {
            uint32_t b0, b1, b2, b3;
            if constexpr (B == 1) {
                const uint32_t byte = (q[kt] >> (8 * v)) & 0xFFu;
                b0 = sPLUT[plut_off[v][0] +         byte];
                b1 = sPLUT[plut_off[v][0] + 256   + byte];
                b2 = sPLUT[plut_off[v][1] + 512   + byte];
                b3 = sPLUT[plut_off[v][1] + 768   + byte];
            } else if constexpr (B == 2) {
                const uint32_t f =
                    (q[2 * kt + (v >> 1)] >> (16 * (v & 1))) & 0xFFFFu;
                const uint32_t blo = f & 0xFFu;
                const uint32_t bhi = (f >> 8) & 0xFFu;
                b0 = sPLUT[plut_off[v][0] +         blo];
                b1 = sPLUT[plut_off[v][0] + 256   + blo];
                b2 = sPLUT[plut_off[v][1] +         bhi];
                b3 = sPLUT[plut_off[v][1] + 256   + bhi];
            } else {                                 // B == 3
                const int byte_off = 12 * kt + 3 * v;
                const int w0 = byte_off >> 2;
                const int s  = (byte_off & 3) * 8;
                uint32_t val = q[w0] >> s;
                if (s > 8) val |= q[w0 + 1] << (32 - s);
                val &= 0xFFFFFFu;
                b0 = sPLUT[plut_off[v][0] +   (val       & 0x3Fu)];
                b1 = sPLUT[plut_off[v][0] +  ((val >>  6) & 0x3Fu)];
                b2 = sPLUT[plut_off[v][1] +  ((val >> 12) & 0x3Fu)];
                b3 = sPLUT[plut_off[v][1] +  ((val >> 18) & 0x3Fu)];
            }
            #pragma unroll
            for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
                const uint32_t bL[2] = {b0, b1};
                const uint32_t bR[2] = {b2, b3};
                flute::mma_m16n8k16_f32acc(afrag[mt], bL, acc[mt][2 * v    ]);
                flute::mma_m16n8k16_f32acc(afrag[mt], bR, acc[mt][2 * v + 1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the dual-stream fused decode kernel (items 2-3).
//
// One launch computes
//     C[M, N] = A @ W1^T + (B2 > 0 ? A @ W2^T : 0)
//             + (resB ? (A @ resB^T) @ resA^T : 0) + (bias ? bias : 0)
// for the two-stream (Route-A pair-composite) deployment shape, with the
// rank-16 residual and the bias folded into the epilogue. Stream 2 is a
// template constexpr (B2 = 0 -> single-stream modules ride the same fused
// epilogue); the two streams share the sA pipeline and the mma A fragments,
// so the A-tile global traffic halves and the (M, N) fp16 add of the
// two-launch route disappears entirely.
//
// Numerics contract (documented, NOT bit-identical to the two-launch
// route by construction): the stream partials, the residual and the bias
// all accumulate into the SAME fp32 accumulator set before the single
// fp16 store, instead of the fp16 pairwise chain
//   fp16(fp16(y1) + fp16(y2)) + fp16((x@resB)@resA) + bias
// — strictly tighter rounding, and the same association the module's
// fp32 reference path already computes. The deployment gate keeps this
// route on the M <= 16 decode shape only (PERFORMANCE.md §8's regime),
// so prefill/PPL numerics are byte-identical to the older path.
//
// The paired-LUT dequant below is the per-width extraction of
// fd_mma_phase / fd_mma_phase_sub4 lifted verbatim into one helper so
// both streams of a mixed pair (e.g. idx1 base + idx4 refinement, the
// artifacts' most common composite) dequantize with their own table
// geometry while feeding the SAME accumulators.
// ---------------------------------------------------------------------------

template <typename Cfg, int B>
constexpr int dual_fdw() { return Cfg::BK * B / 16; }
// u32 words of paired LUT per group row (SubCfg::PLUT_GRP at B < 4;
// PLUT_WORDS/1 == 256 at B == 4).
template <int B>
constexpr int dual_plut_grp() {
    // B == 0 is the DualCfg "no second operand" sentinel: the value is
    // never consumed (the second sub-config does not exist), but the
    // (4 / B) * 256 arm would still instantiate a division by zero at
    // compile time. Guard it — same values for every real B.
    if constexpr (B == 0)
        return 0;
    else if constexpr (B == 3)
        return 64;
    else if constexpr (B == 4)
        return 256;
    else
        return (4 / B) * 256;
}

// The four B-fragment u32 of mma step (kt, v) from one stream's Q words.
// Width-specific extraction copied VERBATIM from fd_mma_phase (B == 4)
// and fd_mma_phase_sub4 (B in {1,2,3}); po[v][d] is that stream's
// LUT-group row base (group row * dual_plut_grp<B>()).
template <int B, int FDW>
__device__ __forceinline__ void fd_frag_dual(const uint32_t (&q)[FDW],
    const uint32_t* __restrict__ sPLUT,
    const int (&po)[4][2],
    int kt, int v,
    uint32_t& b0, uint32_t& b1, uint32_t& b2, uint32_t& b3
) {
    if constexpr (B == 4) {
        const uint32_t w = q[kt * 4 + v];
        b0 = sPLUT[po[v][0] + ((w) & 0xFFu)];
        b1 = sPLUT[po[v][0] + ((w >>  8) & 0xFFu)];
        b2 = sPLUT[po[v][1] + ((w >> 16) & 0xFFu)];
        b3 = sPLUT[po[v][1] + (w >> 24)];
    } else if constexpr (B == 1) {
        const uint32_t byte = (q[kt] >> (8 * v)) & 0xFFu;
        b0 = sPLUT[po[v][0] +         byte];
        b1 = sPLUT[po[v][0] + 256   + byte];
        b2 = sPLUT[po[v][1] + 512   + byte];
        b3 = sPLUT[po[v][1] + 768   + byte];
    } else if constexpr (B == 2) {
        const uint32_t f =
            (q[2 * kt + (v >> 1)] >> (16 * (v & 1))) & 0xFFFFu;
        const uint32_t blo = f & 0xFFu;
        const uint32_t bhi = (f >> 8) & 0xFFu;
        b0 = sPLUT[po[v][0] +         blo];
        b1 = sPLUT[po[v][0] + 256   + blo];
        b2 = sPLUT[po[v][1] +         bhi];
        b3 = sPLUT[po[v][1] + 256   + bhi];
    } else {                                 // B == 3
        const int byte_off = 12 * kt + 3 * v;
        const int w0 = byte_off >> 2;
        const int s  = (byte_off & 3) * 8;
        uint32_t val = q[w0] >> s;
        if (s > 8) val |= q[w0 + 1] << (32 - s);
        val &= 0xFFFFFFu;
        b0 = sPLUT[po[v][0] +   (val       & 0x3Fu)];
        b1 = sPLUT[po[v][0] +  ((val >>  6) & 0x3Fu)];
        b2 = sPLUT[po[v][1] +  ((val >> 12) & 0x3Fu)];
        b3 = sPLUT[po[v][1] +  ((val >> 18) & 0x3Fu)];
    }
}

// Q chunk prefetch, width-generic (BK = 64 makes every width's per-tile
// share a whole number of uint4 vectors: FDW = 4*B, 16 B-aligned).
template <int FDW>
__device__ __forceinline__ void q_fd_prefetch_w(const uint8_t* chunk, uint32_t (&q)[FDW]
) {
    static_assert(FDW % 4 == 0,
                  "the dual kernel's uint4 Q prefetch needs FDW % 4 == 0 "
                  "(BK = 64 guarantees it at every bitwidth)");
    uint4* qq = reinterpret_cast<uint4*>(q);
    const uint4* src = reinterpret_cast<const uint4*>(chunk);
    #pragma unroll
    for (int v = 0; v < FDW / 4; ++v) {
        flute::ldg_nc_evict_first_v4(qq[v], src + v);
    }
}

// This thread's chunk address for K-tile `ktile` (BK = 64: one K-tile ==
// one 64-k group; the half-offset of the generic helper is always 0).
// TILE_BYTES per stream = 64 chunks * 16 B * B = FDW * 256 bytes.
template <int FDW>
__device__ __forceinline__ const uint8_t* q_fd_chunk_dual(const uint8_t* chunk_base, int ktile
) {
    return chunk_base + (size_t)ktile * (FDW * 256);
}

// Tensor-core phase, dual: one afrag load per (kt, mt), then BOTH
// streams' B fragments mma into the same accumulators (stream 1 first —
// the stream order of the module's ordered add, preserved here as the
// fp32 accumulation order).
template <typename Cfg, int B1, int B2, int FDW1, int FDW2>
__device__ __forceinline__ void fd_mma_phase_dual(const __half* sA_buf,
    const uint32_t* __restrict__ sPLUT1,
    const uint32_t* __restrict__ sPLUT2,          // unused when B2 == 0
    const uint32_t (&q1)[FDW1],
    const uint32_t (&q2)[FDW2],                  // unused when B2 == 0
    const int (&po1)[4][2], const int (&po2)[4][2],
    int wy, int lane,
    float (&acc)[Cfg::M_TILES][Cfg::N_TILES][4]
) {
    #pragma unroll
    for (int kt = 0; kt < Cfg::K_TILES; ++kt) {
        uint32_t afrag[Cfg::M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
            flute::ldmatrix_x4(afrag[mt][0], afrag[mt][1], afrag[mt][2], afrag[mt][3],
                flute::ldmatrix_a_addr(sA_buf, Cfg::SA_LD,
                                       wy * Cfg::WM + mt * 16, kt * 16,
                                       lane, Cfg::SA_XOR));
        }
        #pragma unroll
        for (int v = 0; v < Cfg::N_TILES / 2; ++v) {
            {
                uint32_t b0, b1, b2, b3;
                fd_frag_dual<B1, FDW1>(q1, sPLUT1, po1, kt, v,
                                       b0, b1, b2, b3);
                #pragma unroll
                for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
                    const uint32_t bL[2] = {b0, b1};
                    const uint32_t bR[2] = {b2, b3};
                    flute::mma_m16n8k16_f32acc(afrag[mt], bL, acc[mt][2 * v    ]);
                    flute::mma_m16n8k16_f32acc(afrag[mt], bR, acc[mt][2 * v + 1]);
                }
            }
            if constexpr (B2 > 0) {
                uint32_t b0, b1, b2, b3;
                fd_frag_dual<B2, FDW2>(q2, sPLUT2, po2, kt, v,
                                       b0, b1, b2, b3);
                #pragma unroll
                for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
                    const uint32_t bL[2] = {b0, b1};
                    const uint32_t bR[2] = {b2, b3};
                    flute::mma_m16n8k16_f32acc(afrag[mt], bL, acc[mt][2 * v    ]);
                    flute::mma_m16n8k16_f32acc(afrag[mt], bR, acc[mt][2 * v + 1]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor-core phase: consume one (sA, sW) buffer pair into the accumulators
// ---------------------------------------------------------------------------
template <typename Cfg>
__device__ __forceinline__ void mma_phase(const __half* sA_buf, const __half* sW_buf,
    int wy, int wx, int lane,
    float (&acc)[Cfg::M_TILES][Cfg::N_TILES][4]
) {
    #pragma unroll
    for (int kt = 0; kt < Cfg::K_TILES; ++kt) {
        // A fragments: ldmatrix.x4 per 16x16 (m, k) tile.
        // r0..r3 = (T00, T10, T01, T11) = mma A-fragment registers a0..a3.
        uint32_t afrag[Cfg::M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
            flute::ldmatrix_x4(afrag[mt][0], afrag[mt][1], afrag[mt][2], afrag[mt][3],
                flute::ldmatrix_a_addr(sA_buf, Cfg::SA_LD,
                                       wy * Cfg::WM + mt * 16, kt * 16,
                                       lane, Cfg::SA_XOR));
        }
        // B fragments: NON-trans ldmatrix.x4 loads TWO adjacent 8-wide
        // n-tiles at once (r0,r1 = tile n_base; r2,r3 = tile n_base+8).
        // sW is [N,K] row-major and B fragments pack K-pairs, so the plain
        // ldmatrix distribution is exactly the PTX B-fragment layout.
        uint32_t bfrag[Cfg::N_TILES][2];
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; nt += 2) {
            flute::ldmatrix_x4(bfrag[nt][0], bfrag[nt][1],
                bfrag[nt + 1][0], bfrag[nt + 1][1],
                flute::ldmatrix_b_addr(sW_buf, Cfg::SW_LD,
                                       wx * Cfg::WN + nt * 8, kt * 16,
                                       lane, Cfg::SW_XOR));
        }
        #pragma unroll
        for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
                flute::mma_m16n8k16_f32acc(afrag[mt], bfrag[nt], acc[mt][nt]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Epilogue: FP32 accumulators -> FP16 output, vectorized where safe.
// C-fragment (PTX ISA 9.4): c0,c1 = C[g][2t'], C[g][2t'+1];
//                          c2,c3 = C[g+8][2t'], C[g+8][2t'+1]
// with g = lane/4, t' = lane%4. (c0,c1) and (c2,c3) are ADJACENT columns,
// so each pair is one 32-bit store when N is even (column 2t' is even, the
// address is 4 B aligned) and not past the N edge.
// ---------------------------------------------------------------------------
template <typename Cfg>
__device__ __forceinline__ void epilogue(const float (&acc)[Cfg::M_TILES][Cfg::N_TILES][4],
    __half* __restrict__ C, int M, int N,
    int m0, int n0, int wy, int wx, int lane
) {
    const bool n_even = (N % 2) == 0;
    #pragma unroll
    for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
            const int m_base = m0 + wy * Cfg::WM + mt * 16;
            const int n_base = n0 + wx * Cfg::WN + nt * 8;
            const int r = lane / 4;
            const int c = (lane % 4) * 2;

            const int m_r0 = m_base + r;
            const int m_r1 = m_base + r + 8;
            const int n_c0 = n_base + c;
            const int n_c1 = n_c0 + 1;

            if (m_r0 < M) {
                if (n_even && n_c1 < N) {
                    __half2 h2 = __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
                    *reinterpret_cast<__half2*>(&C[(size_t)m_r0 * N + n_c0]) = h2;
                } else if (n_c0 < N) {
                    C[(size_t)m_r0 * N + n_c0] = __float2half(acc[mt][nt][0]);
                    if (n_c1 < N) C[(size_t)m_r0 * N + n_c1] = __float2half(acc[mt][nt][1]);
                }
            }
            if (m_r1 < M) {
                if (n_even && n_c1 < N) {
                    __half2 h2 = __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
                    *reinterpret_cast<__half2*>(&C[(size_t)m_r1 * N + n_c0]) = h2;
                } else if (n_c0 < N) {
                    C[(size_t)m_r1 * N + n_c0] = __float2half(acc[mt][nt][2]);
                    if (n_c1 < N) C[(size_t)m_r1 * N + n_c1] = __float2half(acc[mt][nt][3]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <typename Cfg>
__global__ void __launch_bounds__(Cfg::THREADS, 2)
flute_kernel_streaming(const __half*    __restrict__ A,    // [M, K] row-major
    const uint8_t*   __restrict__ Q,    // [N, (K+1)/2] packed 4-bit
    const __half*    __restrict__ LUT,  // [ceil(N/GS), 16]
    __half*          __restrict__ C,    // [M, N] row-major
    int M, int N, int K
) {
    static_assert(Cfg::BK == 32 || Cfg::BK == 64, "BK (K-tile depth) must be 32 or 64");
    static_assert(gs_supported<Cfg::GS>::value,
                  "GS (LUT group size) must be 16..512 or 1024/2048 ");
    static_assert(Cfg::GS > Cfg::BN || Cfg::BN % Cfg::GS == 0,
                  "GS <= BN requires BN % GS == 0 (n0 stays GS-aligned);");
    static_assert(Cfg::WARPS == 4, "expected 4 warps (2x2)");
    static_assert(Cfg::SA_LD % 8 == 0 && Cfg::SW_LD % 8 == 0,
                  "sA/sW row strides must be multiples of 8 halves (16 B)");
    static_assert((Cfg::SA_ELEMS * 2) % 16 == 0 && ((Cfg::SA_ELEMS + Cfg::SW_ELEMS) * 2) % 16 == 0,
                  "shared regions must stay 16 B aligned");
    static_assert(Cfg::M_TILES * Cfg::N_TILES * Cfg::K_TILES == 64,
                  "per-warp HMMA count must stay at 64 per K-tile");

    // ---- dynamic shared memory: [ sA | sW | sLUT32 ] -----------------------
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half*    sA     = reinterpret_cast<__half*>(smem_raw);
    __half*    sW     = sA + Cfg::SA_ELEMS;
    uint32_t*  sLUT32 = reinterpret_cast<uint32_t*>(sW + Cfg::SW_ELEMS);

    // E4: m-major rasterization — blockIdx.x walks M-tiles (fastest),
    // blockIdx.y the N-tiles. A scheduler wave then covers few W strips
    // with many M-tiles, so W tiles are re-read from L2 instead of HBM
    // (the legacy order — N fastest — made every wave touch 80+ distinct
    // W tiles, re-streaming weights once per M-block-row).
    const int m0 = blockIdx.x * Cfg::BM;
    const int n0 = blockIdx.y * Cfg::BN;
    const int row_stride = (K + 1) >> 1;          // packed bytes per W row

    const int tid     = threadIdx.x;              // 0..127
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int wy      = warp_id / Cfg::WARPS_N;   // 0..1
    const int wx      = warp_id % Cfg::WARPS_N;   // 0..1

    // ---- Step 0: cache LUT rows (u32 entries, 4 lane-copies) ---------------
    // The LUT is indexed by N group only, so it is staged ONCE for the whole
    // K loop. n0 is BN-aligned and BN % GS == 0, so n0 is GS-aligned; the
    // group range is clamped to the groups that exist — the last block's
    // N-range may extend past N.
    const int grp_first  = n0 / Cfg::GS;
    const int total_grps = (N + Cfg::GS - 1) / Cfg::GS;
    // NOTE: explicit ternary instead of min() to avoid any std::min/builtin
    // ambiguity when torch headers are in scope under nvcc.
    const int grp_last_r = (n0 + Cfg::BN - 1) / Cfg::GS;
    const int grp_last   = (grp_last_r < total_grps - 1) ? grp_last_r : (total_grps - 1);
    const int n_grps     = grp_last - grp_first + 1;

    for (int i = tid; i < Cfg::LUT_WORDS; i += Cfg::THREADS) sLUT32[i] = 0u;
    for (int i = tid; i < n_grps * 16; i += Cfg::THREADS) {
        const int g   = i >> 4;
        const int idx = i & 15;
        const uint32_t bits =
            reinterpret_cast<const uint16_t*>(LUT)[(size_t)(grp_first + g) * 16 + idx];
        #pragma unroll
        for (int cp = 0; cp < 4; ++cp) sLUT32[g * 64 + idx * 4 + cp] = bits;
    }

    // Per-thread dequant constants: thread t owns W row n_local = t.
    const int drow     = tid;                    // 0..BN-1
    const int drow_abs = n0 + drow;
    const bool drow_ok = drow_abs < N;
    const int g_lut    = (n0 + drow) / Cfg::GS - grp_first;  // absolute-row group (== drow/GS when n0 is GS-aligned; handles the GS > BN straddle)
    const int lut_off  = g_lut * 64 + (tid & 3); // + lane copy
    const uint8_t* q_row = drow_ok ? Q + (size_t)drow_abs * row_stride : Q;

    // ---- Accumulators -------------------------------------------------------
    float acc[Cfg::M_TILES][Cfg::N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < Cfg::M_TILES; ++mt)
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
            acc[mt][nt][0] = 0.0f;
            acc[mt][nt][1] = 0.0f;
            acc[mt][nt][2] = 0.0f;
            acc[mt][nt][3] = 0.0f;
        }

    // ---- Prologue: stage tile 0. --------------------------------------------
    // cp.async A(0) is issued before the LUT barrier (it targets sA[0],
    // which nothing reads before the second barrier). Q(0) is prefetched
    // into registers; the one-off LDG latency is absorbed here.
    // ------------------------------------------------------------------------
    a_copy_async<Cfg>(A, m0, 0, M, K, sA);              // -> sA[0] (async)
    flute::cp_async_commit();

    alignas(16) uint32_t q_cur[Cfg::QW];
    if (drow_ok) q_prefetch<Cfg>(q_row, q_cur);
    else         q_zero(q_cur);

    __syncthreads();                                    // LUT staged
    dequant_tile<Cfg>(q_cur, sLUT32, lut_off, drow,
                      reinterpret_cast<uint32_t*>(sW)); // -> sW[0]
    flute::cp_async_wait_all();                         // A(0) landed
    __syncthreads();                                    // sA[0] + sW[0] visible

    // ---- Main loop — one code path for all three configs. ------------------
    // Per K-tile (single barrier at the end):
    //   1. issue cp.async A(i+1) -> sA[buf^1]  (no registers; overlaps MMA)
    //      and LDG Q(i+1) -> registers         (latency covered by MMA)
    //   2. tensor cores consume (sA[buf], sW[buf])
    //   3. dequant Q(i+1) -> sW[buf^1]         (stores overlap HMMA drain)
    //   4. wait for the async copy, then one __syncthreads()
    // ------------------------------------------------------------------------
    for (int k0 = 0, buf = 0; k0 < K; k0 += Cfg::BK, buf ^= 1) {
        const int  next     = k0 + Cfg::BK;
        const bool has_next = next < K;

        alignas(16) uint32_t q_nxt[Cfg::QW];
        if (has_next) {
            a_copy_async<Cfg>(A, m0, next, M, K,
                              sA + (size_t)(buf ^ 1) * Cfg::SA_BUF_ELEMS);
            flute::cp_async_commit();
            if (drow_ok) q_prefetch<Cfg>(q_row + (next >> 1), q_nxt);
            else         q_zero(q_nxt);
        }

        mma_phase<Cfg>(sA + (size_t)buf * Cfg::SA_BUF_ELEMS,
                       sW + (size_t)buf * Cfg::SW_BUF_ELEMS,
                       wy, wx, lane, acc);

        if (has_next) {
            dequant_tile<Cfg>(q_nxt, sLUT32, lut_off, drow,
                              reinterpret_cast<uint32_t*>(sW + (size_t)(buf ^ 1) * Cfg::SW_BUF_ELEMS));
        }

        flute::cp_async_wait_all();   // A(i+1) landed (copy had all of MMA)
        __syncthreads();              // buffers visible; next iteration safe
    }

    // ---- Epilogue ----
    epilogue<Cfg>(acc, C, M, N, m0, n0, wy, wx, lane);
}

// ---------------------------------------------------------------------------
// Kernel — fragment-direct path (layout "idx4" only)
// ---------------------------------------------------------------------------
template <typename Cfg>
__global__ void __launch_bounds__(Cfg::THREADS, 2)
flute_kernel_streaming_fd(const __half*    __restrict__ A,    // [M, K] row-major
    const uint8_t*   __restrict__ Q,    // idx4 layout, same byte count
    const __half*    __restrict__ LUT,  // [ceil(N/GS), 16]
    __half*          __restrict__ C,    // [M, N] row-major
    int M, int N, int K
) {
    static_assert(Cfg::FD_SMEM_BYTES <= 99 * 1024,
                  "fd shared memory must fit the SM_86 99 KB per-block limit");
    static_assert(gs_supported<Cfg::GS>::value,
                  "GS (LUT group size) must be 16..512 or 1024/2048 ");
    static_assert(Cfg::BN == 128 && Cfg::WN == 64 && Cfg::WARPS_N == 2,
                  "idx4 assumes the 128-row B tiling of this kernel");
    static_assert(Cfg::N_TILES == 8,
                  "fd Q word packing assumes WN/8 == 8 n-tiles per warp");

    // ---- dynamic shared memory: [ sA(3 stages) | sPLUT ] -------------------
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half*   sA    = reinterpret_cast<__half*>(smem_raw);
    uint32_t* sPLUT = reinterpret_cast<uint32_t*>(sA + Cfg::FD_STAGES * Cfg::SA_BUF_ELEMS);

    // E4: m-major rasterization (see the legacy kernel above).
    const int m0 = blockIdx.x * Cfg::BM;
    const int n0 = blockIdx.y * Cfg::BN;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int wy      = warp_id / Cfg::WARPS_N;
    const int wx      = warp_id % Cfg::WARPS_N;

    // ---- Step 0: paired LUT (E1) -------------------------------------------
    // idx4 guarantees N % 128 == 0, so this block's N range covers
    // exactly BN/GS full groups; the clamp below is kept for uniformity
    // with the legacy path and costs nothing.
    const int grp_first  = n0 / Cfg::GS;
    const int total_grps = (N + Cfg::GS - 1) / Cfg::GS;
    const int grp_last_r = (n0 + Cfg::BN - 1) / Cfg::GS;
    const int grp_last   = (grp_last_r < total_grps - 1) ? grp_last_r : (total_grps - 1);
    const int n_grps     = grp_last - grp_first + 1;
    plut_stage<Cfg>(LUT, sPLUT, grp_first, n_grps, tid);

    // ---- Per-thread Q addressing + paired-LUT row offsets ------------------
    const int K64 = K / 64;                       // idx4 k-groups
    const uint8_t* chunk_base =
        Q + ((size_t)(n0 / 128) * K64) * Cfg::FD_TILE_BYTES
          + (size_t)wx * 2048 + lane * Cfg::FD_CHUNK_BYTES;

    const int g_lane = lane >> 2;                 // B-fragment column group
    int plut_off[4][2];                           // [v][d] = group row * 256
    // group row from the ABSOLUTE row (n0 + row-in-tile) — identical
    // to row-in-tile/GS when n0 is GS-aligned, correct across the GS > BN
    // straddle. n0 and grp_first are block-uniform.
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            plut_off[v][d] =
                ((n0 + wx * 64 + v * 16 + d * 8 + g_lane) / Cfg::GS
                  - grp_first) * 256;
        }
    }

    // ---- Accumulators -------------------------------------------------------
    float acc[Cfg::M_TILES][Cfg::N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < Cfg::M_TILES; ++mt)
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
            acc[mt][nt][0] = 0.0f;
            acc[mt][nt][1] = 0.0f;
            acc[mt][nt][2] = 0.0f;
            acc[mt][nt][3] = 0.0f;
        }

    // ---- Prologue: A stages 0..S-2 in flight; Q(0) in registers -------------
    const int T = K / Cfg::BK;                    // K-tile count
    a_copy_async<Cfg>(A, m0, 0, M, K, sA);
    flute::cp_async_commit();
    if (T > 1) {
        a_copy_async<Cfg>(A, m0, Cfg::BK, M, K,
                          sA + (size_t)Cfg::SA_BUF_ELEMS);
        flute::cp_async_commit();
    }

    alignas(16) uint32_t q_cur[Cfg::FD_WORDS];
    q_fd_prefetch<Cfg>(q_fd_chunk<Cfg>(chunk_base, 0), q_cur);

    // ---- Main loop (E3: 3-stage A pipeline, one barrier per K-tile) ---------
    // Steady state: two A stages in flight (current + next), the third
    // buffer being refilled. The tail iterations (< 3 tiles left) fall back
    // to wait-all so the last copies can never be consumed half-landed.
    alignas(16) uint32_t q_nxt[Cfg::FD_WORDS];
    for (int i = 0, buf = 0; i < T; ++i, buf = (buf + 1) % Cfg::FD_STAGES) {
        if (i <= T - 3) flute::cp_async_wait_1();   // A(i) landed, A(i+1) in flight
        else            flute::cp_async_wait_all();
        __syncthreads();                            // publish sA[i] + sPLUT

        const bool has_next = (i + 1 < T);
        if (has_next) {
            q_fd_prefetch<Cfg>(q_fd_chunk<Cfg>(chunk_base, i + 1), q_nxt);
        }
        if (i + 2 < T) {
            a_copy_async<Cfg>(A, m0, (i + 2) * Cfg::BK, M, K,
                              sA + (size_t)((i + 2) % Cfg::FD_STAGES) *
                                   Cfg::SA_BUF_ELEMS);
            flute::cp_async_commit();
        }

        fd_mma_phase<Cfg>(sA + (size_t)buf * Cfg::SA_BUF_ELEMS, sPLUT,
                          q_cur, plut_off, wy, lane, acc);

        if (has_next) {
            #pragma unroll
            for (int w = 0; w < Cfg::FD_WORDS; ++w) q_cur[w] = q_nxt[w];
        }
    }

    // ---- Epilogue ----
    epilogue<Cfg>(acc, C, M, N, m0, n0, wy, wx, lane);
}

// ---------------------------------------------------------------------------
// Kernel — sub-4-bit, legacy layout (q_layout = 0, logical packed rows)
// ---------------------------------------------------------------------------
// Identical pipeline to flute_kernel_streaming (same sA/sW/sLUT32 layout,
// same barriers, same mma/epilogue) — only the Q field extraction
// (dequant_tile_sub4), the row stride (K*B/8) and the LUT entry count
// (2^B) are width-specific. Consumed for differential testing against
// the fragment-direct path and for shapes the idxN blob layout does not
// cover.
template <typename Cfg, int B>
__global__ void __launch_bounds__(Cfg::THREADS, 2)
flute_kernel_streaming_sub4(const __half*    __restrict__ A,    // [M, K] row-major
    const uint8_t*   __restrict__ Q,    // [N, K*B/8] packed b-bit, LSB-first
    const __half*    __restrict__ LUT,  // [ceil(N/GS), 2^B]
    __half*          __restrict__ C,    // [M, N] row-major
    int M, int N, int K
) {
    using SC = SubCfg<Cfg, B>;
    static_assert(Cfg::BK == 32 || Cfg::BK == 64, "BK (K-tile depth) must be 32 or 64");
    static_assert(gs_supported<Cfg::GS>::value,
                  "GS (LUT group size) must be 16..512 or 1024/2048 ");
    static_assert(Cfg::GS > Cfg::BN || Cfg::BN % Cfg::GS == 0,
                  "GS <= BN requires BN % GS == 0 (n0 stays GS-aligned)");
    static_assert(Cfg::WARPS == 4, "expected 4 warps (2x2)");
    static_assert(SC::QW_B >= 1, "K-tile must hold at least one Q word");

    // ---- dynamic shared memory: [ sA | sW | sLUT32 ] -----------------------
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half*    sA     = reinterpret_cast<__half*>(smem_raw);
    __half*    sW     = sA + Cfg::SA_ELEMS;
    uint32_t*  sLUT32 = reinterpret_cast<uint32_t*>(sW + Cfg::SW_ELEMS);

    const int m0 = blockIdx.x * Cfg::BM;
    const int n0 = blockIdx.y * Cfg::BN;
    const int row_stride = (K * B) >> 3;          // packed bytes per W row

    const int tid     = threadIdx.x;              // 0..127
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int wy      = warp_id / Cfg::WARPS_N;   // 0..1
    const int wx      = warp_id % Cfg::WARPS_N;   // 0..1

    // ---- Step 0: cache LUT rows (u32 entries, 4 lane-copies) ---------------
    // Same 64-word-per-group sLUT32 stride as the 4-bit path; entries
    // beyond 2^B stay zero (harmless — the dequant only gathers v < 2^B).
    const int grp_first  = n0 / Cfg::GS;
    const int total_grps = (N + Cfg::GS - 1) / Cfg::GS;
    const int grp_last_r = (n0 + Cfg::BN - 1) / Cfg::GS;
    const int grp_last   = (grp_last_r < total_grps - 1) ? grp_last_r : (total_grps - 1);
    const int n_grps     = grp_last - grp_first + 1;

    for (int i = tid; i < Cfg::LUT_WORDS; i += Cfg::THREADS) sLUT32[i] = 0u;
    for (int i = tid; i < n_grps * SC::PAL; i += Cfg::THREADS) {
        const int g   = i / SC::PAL;
        const int idx = i - g * SC::PAL;
        const uint32_t bits =
            reinterpret_cast<const uint16_t*>(LUT)[(size_t)(grp_first + g) * SC::PAL + idx];
        #pragma unroll
        for (int cp = 0; cp < 4; ++cp) sLUT32[g * 64 + idx * 4 + cp] = bits;
    }

    // Per-thread dequant constants: thread t owns W row n_local = t.
    const int drow     = tid;                    // 0..BN-1
    const int drow_abs = n0 + drow;
    const bool drow_ok = drow_abs < N;
    const int g_lut    = (n0 + drow) / Cfg::GS - grp_first;  // absolute-row group (== drow/GS when n0 is GS-aligned; handles the GS > BN straddle)
    const int lut_off  = g_lut * 64 + (tid & 3); // + lane copy
    const uint8_t* q_row = drow_ok ? Q + (size_t)drow_abs * row_stride : Q;

    // ---- Accumulators -------------------------------------------------------
    float acc[Cfg::M_TILES][Cfg::N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < Cfg::M_TILES; ++mt)
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
            acc[mt][nt][0] = 0.0f;
            acc[mt][nt][1] = 0.0f;
            acc[mt][nt][2] = 0.0f;
            acc[mt][nt][3] = 0.0f;
        }

    // ---- Prologue: stage tile 0. --------------------------------------------
    a_copy_async<Cfg>(A, m0, 0, M, K, sA);              // -> sA[0] (async)
    flute::cp_async_commit();

    alignas(16) uint32_t q_cur[SC::QW_B];
    if (drow_ok) q_prefetch_sub4<Cfg, B>(q_row, q_cur);
    else         q_zero(q_cur);

    __syncthreads();                                    // LUT staged
    dequant_tile_sub4<Cfg, B>(q_cur, sLUT32, lut_off, drow,
                              reinterpret_cast<uint32_t*>(sW)); // -> sW[0]
    flute::cp_async_wait_all();                         // A(0) landed
    __syncthreads();                                    // sA[0] + sW[0] visible

    // ---- Main loop — one code path for all configs. ------------------------
    for (int k0 = 0, buf = 0; k0 < K; k0 += Cfg::BK, buf ^= 1) {
        const int  next     = k0 + Cfg::BK;
        const bool has_next = next < K;

        alignas(16) uint32_t q_nxt[SC::QW_B];
        if (has_next) {
            a_copy_async<Cfg>(A, m0, next, M, K,
                              sA + (size_t)(buf ^ 1) * Cfg::SA_BUF_ELEMS);
            flute::cp_async_commit();
            if (drow_ok) q_prefetch_sub4<Cfg, B>(q_row + ((next * B) >> 3), q_nxt);
            else         q_zero(q_nxt);
        }

        mma_phase<Cfg>(sA + (size_t)buf * Cfg::SA_BUF_ELEMS,
                       sW + (size_t)buf * Cfg::SW_BUF_ELEMS,
                       wy, wx, lane, acc);

        if (has_next) {
            dequant_tile_sub4<Cfg, B>(q_nxt, sLUT32, lut_off, drow,
                                      reinterpret_cast<uint32_t*>(sW + (size_t)(buf ^ 1) * Cfg::SW_BUF_ELEMS));
        }

        flute::cp_async_wait_all();   // A(i+1) landed (copy had all of MMA)
        __syncthreads();              // buffers visible; next iteration safe
    }

    // ---- Epilogue ----
    epilogue<Cfg>(acc, C, M, N, m0, n0, wy, wx, lane);
}

// ---------------------------------------------------------------------------
// Kernel — sub-4-bit, fragment-direct (idxN blob, q_layout = 1)
// ---------------------------------------------------------------------------
// Identical pipeline to flute_kernel_streaming_fd (3-stage sA, Q chunks
// in registers, paired-LUT dequant into the mma B fragments, m-major
// rasterization) at width B: chunk = 16*B bytes per thread per 64-k
// group, sPLUT = the width-specific paired tables. blob size is
// N*K*B/8 bytes; N % 128 == 0 and K % 64 == 0 are hard requirements.
template <typename Cfg, int B>
__global__ void __launch_bounds__(Cfg::THREADS, 2)
flute_kernel_streaming_fd_sub4(const __half*    __restrict__ A,    // [M, K] row-major
    const uint8_t*   __restrict__ Q,    // idxN blob (flute_extended/idxN.py)
    const __half*    __restrict__ LUT,  // [ceil(N/GS), 2^B]
    __half*          __restrict__ C,    // [M, N] row-major
    int M, int N, int K
) {
    using SC = SubCfg<Cfg, B>;
    static_assert(SC::FD_SMEM <= 99 * 1024,
                  "fd shared memory must fit the SM_86 99 KB per-block limit");
    static_assert(gs_supported<Cfg::GS>::value,
                  "GS (LUT group size) must be 16..512 or 1024/2048 ");
    static_assert(Cfg::BN == 128 && Cfg::WN == 64 && Cfg::WARPS_N == 2,
                  "idxN assumes the 128-row B tiling of this kernel");
    static_assert(Cfg::N_TILES == 8,
                  "fd Q word packing assumes WN/8 == 8 n-tiles per warp");

    // ---- dynamic shared memory: [ sA(3 stages) | sPLUT ] -------------------
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half*   sA    = reinterpret_cast<__half*>(smem_raw);
    uint32_t* sPLUT = reinterpret_cast<uint32_t*>(sA + Cfg::FD_STAGES * Cfg::SA_BUF_ELEMS);

    // E4: m-major rasterization (see the legacy kernel above).
    const int m0 = blockIdx.x * Cfg::BM;
    const int n0 = blockIdx.y * Cfg::BN;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int wy      = warp_id / Cfg::WARPS_N;
    const int wx      = warp_id % Cfg::WARPS_N;

    // ---- Step 0: paired LUT tables (E1 generalized) ------------------------
    const int grp_first  = n0 / Cfg::GS;
    const int total_grps = (N + Cfg::GS - 1) / Cfg::GS;
    const int grp_last_r = (n0 + Cfg::BN - 1) / Cfg::GS;
    const int grp_last   = (grp_last_r < total_grps - 1) ? grp_last_r : (total_grps - 1);
    const int n_grps     = grp_last - grp_first + 1;
    plut_stage_sub4<Cfg, B>(LUT, sPLUT, grp_first, n_grps, tid);

    // ---- Per-thread Q addressing + paired-LUT row offsets ------------------
    const int K64 = K / 64;                       // idxN k-groups
    const uint8_t* chunk_base =
        Q + ((size_t)(n0 / 128) * K64) * SC::TILE_BYTES
          + (size_t)wx * (512 * B) + lane * SC::CHUNK_BYTES;

    const int g_lane = lane >> 2;                 // B-fragment column group
    int plut_off[4][2];                           // [v][d] = group row * PLUT_GRP
    // absolute-row group (see the 4-bit fd kernel above).
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            plut_off[v][d] =
                ((n0 + wx * 64 + v * 16 + d * 8 + g_lane) / Cfg::GS
                  - grp_first) * SC::PLUT_GRP;
        }
    }

    // ---- Accumulators -------------------------------------------------------
    float acc[Cfg::M_TILES][Cfg::N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < Cfg::M_TILES; ++mt)
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
            acc[mt][nt][0] = 0.0f;
            acc[mt][nt][1] = 0.0f;
            acc[mt][nt][2] = 0.0f;
            acc[mt][nt][3] = 0.0f;
        }

    // ---- Prologue: A stages 0..S-2 in flight; Q(0) in registers -------------
    const int T = K / Cfg::BK;                    // K-tile count
    a_copy_async<Cfg>(A, m0, 0, M, K, sA);
    flute::cp_async_commit();
    if (T > 1) {
        a_copy_async<Cfg>(A, m0, Cfg::BK, M, K,
                          sA + (size_t)Cfg::SA_BUF_ELEMS);
        flute::cp_async_commit();
    }

    alignas(16) uint32_t q_cur[SC::FDW];
    q_fd_prefetch_sub4<Cfg, B>(q_fd_chunk_sub4<Cfg, B>(chunk_base, 0), q_cur);

    // ---- Main loop (E3: 3-stage A pipeline, one barrier per K-tile) ---------
    alignas(16) uint32_t q_nxt[SC::FDW];
    for (int i = 0, buf = 0; i < T; ++i, buf = (buf + 1) % Cfg::FD_STAGES) {
        if (i <= T - 3) flute::cp_async_wait_1();   // A(i) landed, A(i+1) in flight
        else            flute::cp_async_wait_all();
        __syncthreads();                            // publish sA[i] + sPLUT

        const bool has_next = (i + 1 < T);
        if (has_next) {
            q_fd_prefetch_sub4<Cfg, B>(q_fd_chunk_sub4<Cfg, B>(chunk_base, i + 1), q_nxt);
        }
        if (i + 2 < T) {
            a_copy_async<Cfg>(A, m0, (i + 2) * Cfg::BK, M, K,
                              sA + (size_t)((i + 2) % Cfg::FD_STAGES) *
                                   Cfg::SA_BUF_ELEMS);
            flute::cp_async_commit();
        }

        fd_mma_phase_sub4<Cfg, B>(sA + (size_t)buf * Cfg::SA_BUF_ELEMS,
                                  sPLUT, q_cur, plut_off, wy, lane, acc);

        if (has_next) {
            #pragma unroll
            for (int w = 0; w < SC::FDW; ++w) q_cur[w] = q_nxt[w];
        }
    }

    // ---- Epilogue ----
    epilogue<Cfg>(acc, C, M, N, m0, n0, wy, wx, lane);
}

// ---------------------------------------------------------------------------
// Kernel dual-stream fused (both streams + residual + bias, ONE
// launch; decode-slim BM = 32 tiles). Layout contract identical to the
// fragment-direct path above (idxN blobs; N % 128 == 0, K % 64 == 0).
// ---------------------------------------------------------------------------
// Why BM = 32: at decode M <= 16 the BM = 64/128 tiles burn the tensor
// pipe on padded rows (an m16n8k16 atom is 16 rows tall — 2-8 m-tiles of
// HMMA work for ONE real row). WM = 16 (one m-tile per warp) halves the
// per-strip tensor work vs BM = 64, and grid.x = ceil(M/32) keeps one
// m-block at the decode shape (the m-major E4 rasterization is
// unchanged). Per-warp HMMA per K-tile is 32 (one stream) / 64 (two
// streams) — the invariant pinned by the other kernels (64) is
// deliberately relaxed for this decode config.
//
// Shared memory (GS = 512, the (1,4) worst case): sA 3 stages 12,288 B +
// sPLUT1 2 x 1024 u32 8,192 B + sPLUT2 2 x 256 u32 2,048 B + sResB
// 2 x 16 x 72 halves 4,608 B + sxB 32 x 16 f32 2,048 B = 29,184 B —
// comfortably under the 48 KB default; the > 48 KB opt-in call below is
// kept for uniformity with the other launch helpers.
//
// Residual dataflow (rank R <= 16, fp16 resB (R, K) / resA (N, R)):
//   * xB[m, r] = sum_k A[m, k] * resB[r, k] accumulated per K-tile from
//     the ALREADY-STAGED sA tile (the A operand is the rotated input —
//     exactly what the module's residual branch consumes), FMA'd against
//     a 2-buffer sResB tile staged one K-tile ahead (staging happens
//     AFTER the tile's barrier, reading is after the NEXT one — the
//     same ordering discipline the 3-stage sA pipeline uses).
//   * Predication: only rows m0 + m_local < M accumulate — at M = 1 the
//     whole xB phase is 4 active threads per block.
//   * Epilogue: C[m, n] += sum_r xB[m, r] * resA[n, r] — the exchange of
//     xB through sxB adds ONE extra __syncthreads after the K loop.
//   * The launch is only taken at M <= 16 (host gate): the FFMA cost of
//     the xB phase scales with the real row count, and at prefill M the
//     cuBLAS residual route amortizes better (: "the
//     kernel amortizes when M grows").
template <typename Cfg, int B1, int B2>
struct DualCfg {
    static constexpr int FDW1 = Cfg::BK * B1 / 16;
    static constexpr int FDW2 = (B2 > 0) ? Cfg::BK * B2 / 16 : 1;
    static constexpr int PG1  = dual_plut_grp<B1>();
    static constexpr int PG2  = (B2 > 0) ? dual_plut_grp<B2>() : 0;
    static constexpr int RES_STAGES = 2;          // sResB double buffer
    static constexpr int RES_ROWS   = 16;         // max residual rank
    static constexpr int RES_LD     = Cfg::BK + 8;// halves; 16 B rows + bank spread
    static constexpr int XB_COLS    = 16;
    static constexpr int SA_BYTES     = Cfg::FD_STAGES * Cfg::SA_BUF_ELEMS * 2;
    static constexpr int PLUT1_BYTES  = Cfg::LUT_GRPS * PG1 * 4;
    static constexpr int PLUT2_BYTES  = Cfg::LUT_GRPS * PG2 * 4;
    static constexpr int RESB_BYTES   = RES_STAGES * RES_ROWS * RES_LD * 2;
    static constexpr int XB_BYTES     = Cfg::BM * XB_COLS * 4;
    static constexpr int SMEM_BYTES   =
        SA_BYTES + PLUT1_BYTES + PLUT2_BYTES + RESB_BYTES + XB_BYTES;
};

// Stage one resB K-tile (rows 0..R-1, cols [k0, k0+BK)) into a plain
// (unswizzled) sResB buffer. resB row stride K is a multiple of 64 (the
// host contract), so every uint4 source is 16 B aligned; the destination
// row stride BK+8 halves = (BK+8)*2 B stays a multiple of 16 B.
template <typename Cfg>
__device__ __forceinline__ void stage_resb_tile(const __half* __restrict__ resB, int R, int k0, int K,
    __half* sResB_buf, int tid
) {
    constexpr int LD = Cfg::BK + 8;               // halves per sResB row
    constexpr int CH = Cfg::BK / 8;               // 16 B chunks per row
    for (int i = tid; i < R * CH; i += Cfg::THREADS) {
        const int r  = i / CH;
        const int k8 = (i - r * CH) * 8;          // 8-half chunk offset
        const uint4 v = *reinterpret_cast<const uint4*>(resB + (size_t)r * K + k0 + k8);
        *reinterpret_cast<uint4*>(sResB_buf + (size_t)r * LD + k8) = v;
    }
}

// The xB FMA phase of K-tile `i` (runs AFTER fd_mma_phase_dual(i), so the
// FFMA issue slots overlap the tensor-pipe drain): thread tid owns rows
// m_local = tid >> 2 and rank slab r = (tid & 3) * 4 .. +4. Reads sA[buf]
// through the same 16 B-block swizzle a_copy_async wrote (pairs (k, k+1)
// never straddle a swizzle block), sResB unswizzled.
template <typename Cfg>
__device__ __forceinline__ void xb_accum_fma(const __half* sA_buf, const __half* sResB_buf,
    int R, int M, int m0, int tid,
    float (&xbr)[4]
) {
    constexpr int LD = Cfg::BK + 8;
    const int m_local = tid >> 2;
    const int r_base  = (tid & 3) << 2;
    if (m0 + m_local >= M) return;                // padded row: nothing to do
    const __half* arow = sA_buf + (size_t)m_local * Cfg::SA_LD;
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int r = r_base + rr;
        if (r >= R) continue;
        const __half* brow = sResB_buf + (size_t)r * LD;
        float acc_r = xbr[rr];
        for (int k = 0; k < Cfg::BK; k += 2) {
            const __half2 a2 = *reinterpret_cast<const __half2*>(arow + flute::swz_col16(m_local, k, Cfg::SA_XOR) + (k & 7));
            const __half2 b2 = *reinterpret_cast<const __half2*>(brow + k);
            const float2 af = __half22float2(a2);
            const float2 bf = __half22float2(b2);
            acc_r = fmaf(af.x, bf.x, acc_r);
            acc_r = fmaf(af.y, bf.y, acc_r);
        }
        xbr[rr] = acc_r;
    }
}

// Epilogue, dual: the fragment-direct epilogue with the residual
// (C += xB @ resA^T, fp32) and bias folded in BEFORE the single fp16
// store. xB rows come from sxB (block-local row index m_r0 - m0).
template <typename Cfg>
__device__ __forceinline__ void epilogue_dual(const float (&acc)[Cfg::M_TILES][Cfg::N_TILES][4],
    __half* __restrict__ C, int M, int N,
    int m0, int n0, int wy, int wx, int lane,
    const float* sxB, const __half* __restrict__ resA, int R,
    const __half* __restrict__ bias
) {
    const bool n_even = (N % 2) == 0;
    #pragma unroll
    for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
            const int m_base = m0 + wy * Cfg::WM + mt * 16;
            const int n_base = n0 + wx * Cfg::WN + nt * 8;
            const int r = lane / 4;
            const int c = (lane % 4) * 2;

            const int m_r0 = m_base + r;
            const int m_r1 = m_base + r + 8;
            const int n_c0 = n_base + c;
            const int n_c1 = n_c0 + 1;

            if (m_r0 < M) {
                float s00 = acc[mt][nt][0];
                float s01 = acc[mt][nt][1];
                if (resA != nullptr) {
                    const float* xb = sxB + (size_t)(m_r0 - m0) * 16;
                    const __half* ra0 = resA + (size_t)n_c0 * R;
                    const __half* ra1 = resA + (size_t)n_c1 * R;
                    float t0 = 0.0f, t1 = 0.0f;
                    for (int rr = 0; rr < R; ++rr) {
                        t0 = fmaf(xb[rr], __half2float(ra0[rr]), t0);
                        t1 = fmaf(xb[rr], __half2float(ra1[rr]), t1);
                    }
                    s00 += t0;
                    s01 += t1;
                }
                if (bias != nullptr) {
                    s00 += __half2float(bias[n_c0]);
                    if (n_c1 < N) s01 += __half2float(bias[n_c1]);
                }
                if (n_even && n_c1 < N) {
                    __half2 h2 = __floats2half2_rn(s00, s01);
                    *reinterpret_cast<__half2*>(&C[(size_t)m_r0 * N + n_c0]) = h2;
                } else if (n_c0 < N) {
                    C[(size_t)m_r0 * N + n_c0] = __float2half(s00);
                    if (n_c1 < N) C[(size_t)m_r0 * N + n_c1] = __float2half(s01);
                }
            }
            if (m_r1 < M) {
                float s10 = acc[mt][nt][2];
                float s11 = acc[mt][nt][3];
                if (resA != nullptr) {
                    const float* xb = sxB + (size_t)(m_r1 - m0) * 16;
                    const __half* ra0 = resA + (size_t)n_c0 * R;
                    const __half* ra1 = resA + (size_t)n_c1 * R;
                    float t0 = 0.0f, t1 = 0.0f;
                    for (int rr = 0; rr < R; ++rr) {
                        t0 = fmaf(xb[rr], __half2float(ra0[rr]), t0);
                        t1 = fmaf(xb[rr], __half2float(ra1[rr]), t1);
                    }
                    s10 += t0;
                    s11 += t1;
                }
                if (bias != nullptr) {
                    s10 += __half2float(bias[n_c0]);
                    if (n_c1 < N) s11 += __half2float(bias[n_c1]);
                }
                if (n_even && n_c1 < N) {
                    __half2 h2 = __floats2half2_rn(s10, s11);
                    *reinterpret_cast<__half2*>(&C[(size_t)m_r1 * N + n_c0]) = h2;
                } else if (n_c0 < N) {
                    C[(size_t)m_r1 * N + n_c0] = __float2half(s10);
                    if (n_c1 < N) C[(size_t)m_r1 * N + n_c1] = __float2half(s11);
                }
            }
        }
    }
}

template <typename Cfg, int B1, int B2>
__global__ void __launch_bounds__(Cfg::THREADS, 2)
flute_kernel_streaming_fd_dual(const __half*    __restrict__ A,     // [M, K] row-major
    const uint8_t*   __restrict__ Q1,    // stream-1 idxN blob
    const __half*    __restrict__ LUT1,  // [ceil(N/GS), 2^B1]
    const uint8_t*   __restrict__ Q2,    // stream-2 idxN blob (Q1 when B2 == 0)
    const __half*    __restrict__ LUT2,  // [ceil(N/GS), 2^B2] (LUT1 when B2 == 0)
    const __half*    __restrict__ resB,  // (R, K) fp16, or nullptr
    const __half*    __restrict__ resA,  // (N, R) fp16, or nullptr
    const __half*    __restrict__ bias,  // (N,) fp16, or nullptr
    __half*          __restrict__ C,     // [M, N] row-major
    int M, int N, int K, int R
) {
    using DC = DualCfg<Cfg, B1, B2>;
    static_assert(Cfg::BK == 64,
                  "the dual kernel is the BK = 64 decode shape");
    static_assert(Cfg::BM == 32 && Cfg::WN == 64 && Cfg::WARPS_N == 2 &&
                      Cfg::WARPS == 4 && Cfg::M_TILES == 1,
                  "dual assumes the BM=32 decode-slim geometry "
                  "(WM=16, one m-tile per warp, 4 warps in 2x2)");
    static_assert(Cfg::GS >= 64,
                  "dual dispatch covers GS 64..512; GS <= 32 keeps the "
                  "two-launch qgemm route");
    static_assert(DC::SMEM_BYTES <= 99 * 1024,
                  "dual shared memory must fit the SM_86 99 KB per-block "
                  "limit");
    static_assert(Cfg::N_TILES == 8,
                  "dual Q word packing assumes WN/8 == 8 n-tiles per warp");

    // ---- dynamic shared memory:
    //      [ sA(3 stages) | sPLUT1 | sPLUT2 | sResB(2 stages) | sxB ] ------
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half*   sA     = reinterpret_cast<__half*>(smem_raw);
    uint32_t* sPLUT1 = reinterpret_cast<uint32_t*>(sA + Cfg::FD_STAGES * Cfg::SA_BUF_ELEMS);
    uint32_t* sPLUT2 = sPLUT1 + Cfg::LUT_GRPS * DC::PG1;
    __half*   sResB  = reinterpret_cast<__half*>(sPLUT2 + Cfg::LUT_GRPS * DC::PG2);
    float*    sxB    = reinterpret_cast<float*>(sResB + DC::RES_STAGES * DC::RES_ROWS * DC::RES_LD);

    // E4: m-major rasterization (see the legacy kernel above).
    const int m0 = blockIdx.x * Cfg::BM;
    const int n0 = blockIdx.y * Cfg::BN;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int wy      = warp_id / Cfg::WARPS_N;
    const int wx      = warp_id % Cfg::WARPS_N;

    // ---- Step 0: the paired LUTs of both streams --------------------------
    const int grp_first  = n0 / Cfg::GS;
    const int total_grps = (N + Cfg::GS - 1) / Cfg::GS;
    const int grp_last_r = (n0 + Cfg::BN - 1) / Cfg::GS;
    const int grp_last   = (grp_last_r < total_grps - 1) ? grp_last_r
                                                         : (total_grps - 1);
    const int n_grps     = grp_last - grp_first + 1;
    if constexpr (B1 == 4) {
        plut_stage<Cfg>(LUT1, sPLUT1, grp_first, n_grps, tid);
    } else {
        plut_stage_sub4<Cfg, B1>(LUT1, sPLUT1, grp_first, n_grps, tid);
    }
    if constexpr (B2 > 0) {
        if constexpr (B2 == 4) {
            plut_stage<Cfg>(LUT2, sPLUT2, grp_first, n_grps, tid);
        } else {
            plut_stage_sub4<Cfg, B2>(LUT2, sPLUT2, grp_first, n_grps, tid);
        }
    }

    // ---- Per-stream Q addressing + paired-LUT row offsets -----------------
    // Both streams share the group rows (same N, same GS); only the row
    // stride (per-width PLUT geometry) differs.
    const int K64 = K / 64;
    const uint8_t* cb1 =
        Q1 + ((size_t)(n0 / 128) * K64) * (1024 * B1)
              + (size_t)wx * (512 * B1) + lane * (16 * B1);
    const uint8_t* cb2 = (B2 > 0)
        ? (Q2 + ((size_t)(n0 / 128) * K64) * (1024 * B2)
                + (size_t)wx * (512 * B2) + lane * (16 * B2))
        : Q1;

    const int g_lane = lane >> 2;                 // B-fragment column group
    int po1[4][2], po2[4][2];
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            const int grow =
                (n0 + wx * 64 + v * 16 + d * 8 + g_lane) / Cfg::GS - grp_first;
            po1[v][d] = grow * DC::PG1;
            po2[v][d] = grow * DC::PG2;
        }
    }

    // ---- Accumulators ------------------------------------------------------
    float acc[Cfg::M_TILES][Cfg::N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < Cfg::M_TILES; ++mt)
        #pragma unroll
        for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
            acc[mt][nt][0] = 0.0f;
            acc[mt][nt][1] = 0.0f;
            acc[mt][nt][2] = 0.0f;
            acc[mt][nt][3] = 0.0f;
        }

    // The residual xB registers (this thread's 4 ranks of its m row).
    float xbr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool has_res = (resB != nullptr);

    // ---- Prologue: A stages 0..1 in flight; Q(0) both streams; resB(0) ---
    const int T = K / Cfg::BK;                    // K-tile count
    a_copy_async<Cfg>(A, m0, 0, M, K, sA);
    flute::cp_async_commit();
    if (T > 1) {
        a_copy_async<Cfg>(A, m0, Cfg::BK, M, K,
                          sA + (size_t)Cfg::SA_BUF_ELEMS);
        flute::cp_async_commit();
    }
    if (has_res) {
        // tile 0 only — the in-loop staging (i+1) covers tile 1 onward
        stage_resb_tile<Cfg>(resB, R, 0, K, sResB, tid);
    }

    alignas(16) uint32_t q1_cur[DC::FDW1];
    q_fd_prefetch_w<DC::FDW1>(q_fd_chunk_dual<DC::FDW1>(cb1, 0), q1_cur);
    alignas(16) uint32_t q2_cur[DC::FDW2];
    if constexpr (B2 > 0) {
        q_fd_prefetch_w<DC::FDW2>(q_fd_chunk_dual<DC::FDW2>(cb2, 0), q2_cur);
    }
    alignas(16) uint32_t q1_nxt[DC::FDW1];
    alignas(16) uint32_t q2_nxt[DC::FDW2];

    // ---- Main loop (E3 pipeline, one barrier per K-tile) -------------------
    // Ordering notes (the barrier discipline the sA pipeline already uses):
    //   * resB(i+1) is staged AFTER the tile's barrier into buffer
    //     (i+1) % 2 — the previous reader of that buffer, xb_accum_fma(i-1),
    //     finished before barrier(i), so no stage/read race exists;
    //   * xb_accum_fma(i) runs after the mma phase of tile i — the FFMA
    //     issue slots overlap the tensor-pipe drain, off the critical
    //     path — and reads sA[buf(i)], which the next writer (the
    //     a_copy_async of tile i+3, issued after barrier(i+1)) cannot
    //     touch until every thread (including this one) has arrived at
    //     barrier(i+1).
    for (int i = 0, buf = 0, rbuf = 0; i < T; ++i,
             buf = (buf + 1) % Cfg::FD_STAGES, rbuf = (rbuf + 1) % 2) {
        if (i <= T - 3) flute::cp_async_wait_1();  // A(i) landed, A(i+1) in flight
        else            flute::cp_async_wait_all();
        __syncthreads();                           // publish sA[i] + sPLUT + resB(i)

        const bool has_next = (i + 1 < T);
        if (has_next) {
            q_fd_prefetch_w<DC::FDW1>(q_fd_chunk_dual<DC::FDW1>(cb1, i + 1), q1_nxt);
            if constexpr (B2 > 0) {
                q_fd_prefetch_w<DC::FDW2>(q_fd_chunk_dual<DC::FDW2>(cb2, i + 1), q2_nxt);
            }
            if (has_res) {
                stage_resb_tile<Cfg>(resB, R, (i + 1) * Cfg::BK, K,
                    sResB + (size_t)((i + 1) % 2) * DC::RES_ROWS * DC::RES_LD,
                    tid);
            }
        }
        if (i + 2 < T) {
            a_copy_async<Cfg>(A, m0, (i + 2) * Cfg::BK, M, K,
                              sA + (size_t)((i + 2) % Cfg::FD_STAGES) *
                                   Cfg::SA_BUF_ELEMS);
            flute::cp_async_commit();
        }

        fd_mma_phase_dual<Cfg, B1, B2, DC::FDW1, DC::FDW2>(sA + (size_t)buf * Cfg::SA_BUF_ELEMS, sPLUT1, sPLUT2,
            q1_cur, q2_cur, po1, po2, wy, lane, acc);

        if (has_res) {
            xb_accum_fma<Cfg>(sA + (size_t)buf * Cfg::SA_BUF_ELEMS,
                              sResB + (size_t)rbuf * DC::RES_ROWS * DC::RES_LD,
                              R, M, m0, tid, xbr);
        }

        if (has_next) {
            #pragma unroll
            for (int w = 0; w < DC::FDW1; ++w) q1_cur[w] = q1_nxt[w];
            if constexpr (B2 > 0) {
                #pragma unroll
                for (int w = 0; w < DC::FDW2; ++w) q2_cur[w] = q2_nxt[w];
            }
        }
    }

    // ---- Epilogue (residual xB exchange through sxB + fused bias) ---------
    if (has_res) {
        const int m_local = tid >> 2;
        if (m0 + m_local < M) {
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                sxB[m_local * 16 + ((tid & 3) << 2) + rr] = xbr[rr];
            }
        }
        __syncthreads();
    }
    epilogue_dual<Cfg>(acc, C, M, N, m0, n0, wy, wx, lane,
                       sxB, resA, R, bias);
}

// ---------------------------------------------------------------------------
// Host-side launch helper (per-instantiation shared-memory attribute opt-in)
// ---------------------------------------------------------------------------
template <typename Cfg>
void launch_streaming(const __half* A, const uint8_t* Q, const __half* LUT, __half* C,
    int M, int N, int K
) {
    constexpr int smem_bytes = Cfg::SMEM_BYTES;

    // > 48 KB dynamic shared memory requires an explicit opt-in (once per
    // process per kernel instantiation). BK=64 needs 49,920-50,432 B; BK=32
    // uses 42,240 B and sets the attribute uniformly for simplicity.
    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_streaming<Cfg>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", smem_bytes,
                ") failed: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    dim3 block(Cfg::THREADS);
    // E4: M-tiles on the fast axis so a wave shares W strips through L2.
    dim3 grid((M + Cfg::BM - 1) / Cfg::BM, (N + Cfg::BN - 1) / Cfg::BN);
    flute_kernel_streaming<Cfg>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, Q, LUT, C, M, N, K);

    // Surface launch-config errors immediately, with enough context to
    // diagnose a bad grid or smem size without a debugger.
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_streaming launch failed (grid ",
                grid.x, "x", grid.y, ", ", Cfg::THREADS,
                " threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

// Fragment-direct launch helper (layout "idx4"). All fd configs fit
// in 48 KB, so the attribute call is a formality — kept for uniformity and
// for any future deeper-stage experiment.
template <typename Cfg>
void launch_streaming_fd(const __half* A, const uint8_t* Q, const __half* LUT, __half* C,
    int M, int N, int K
) {
    constexpr int smem_bytes = Cfg::FD_SMEM_BYTES;

    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_streaming_fd<Cfg>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", smem_bytes,
                ") failed for the fragment-direct kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    dim3 block(Cfg::THREADS);
    dim3 grid((M + Cfg::BM - 1) / Cfg::BM, (N + Cfg::BN - 1) / Cfg::BN);
    flute_kernel_streaming_fd<Cfg>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, Q, LUT, C, M, N, K);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_streaming_fd launch failed (grid ",
                grid.x, "x", grid.y, ", ", Cfg::THREADS,
                " threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

// Sub-4-bit launch helpers (idxN family). Same per-instantiation
// cudaFuncSetAttribute opt-in discipline; the B=1 gs=32 BK=32 fd config
// needs 51.2 KB (4 paired tables), which is exactly why the attribute
// call exists — every other sub-4 config fits in 48 KB.
template <typename Cfg, int B>
void launch_streaming_sub4(const __half* A, const uint8_t* Q, const __half* LUT, __half* C,
    int M, int N, int K
) {
    constexpr int smem_bytes = Cfg::SMEM_BYTES;
    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_streaming_sub4<Cfg, B>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", smem_bytes,
                ") failed: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    dim3 block(Cfg::THREADS);
    dim3 grid((M + Cfg::BM - 1) / Cfg::BM, (N + Cfg::BN - 1) / Cfg::BN);
    flute_kernel_streaming_sub4<Cfg, B>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, Q, LUT, C, M, N, K);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_streaming_sub4 launch failed (grid ",
                grid.x, "x", grid.y, ", ", Cfg::THREADS,
                " threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

template <typename Cfg, int B>
void launch_streaming_fd_sub4(const __half* A, const uint8_t* Q, const __half* LUT, __half* C,
    int M, int N, int K
) {
    constexpr int smem_bytes = SubCfg<Cfg, B>::FD_SMEM;
    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_streaming_fd_sub4<Cfg, B>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", smem_bytes,
                ") failed for the sub-4-bit fragment-direct kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    dim3 block(Cfg::THREADS);
    dim3 grid((M + Cfg::BM - 1) / Cfg::BM, (N + Cfg::BN - 1) / Cfg::BN);
    flute_kernel_streaming_fd_sub4<Cfg, B>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, Q, LUT, C, M, N, K);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_streaming_fd_sub4 launch failed (grid ",
                grid.x, "x", grid.y, ", ", Cfg::THREADS,
                " threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

//  dual-stream launch helper. Every dual instantiation fits in 48 KB
// (GS 64..512, all width pairs), so the attribute call is a formality —
// kept for uniformity with the other launch helpers and for any future
// deeper-stage experiment.
template <typename Cfg, int B1, int B2>
void launch_streaming_fd_dual(const __half* A, const uint8_t* Q1, const __half* LUT1,
    const uint8_t* Q2, const __half* LUT2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* C, int M, int N, int K, int R
) {
    constexpr int smem_bytes = DualCfg<Cfg, B1, B2>::SMEM_BYTES;

    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_streaming_fd_dual<Cfg, B1, B2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=",
                smem_bytes, ") failed for the dual-stream kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    dim3 block(Cfg::THREADS);
    dim3 grid((M + Cfg::BM - 1) / Cfg::BM, (N + Cfg::BN - 1) / Cfg::BN);
    flute_kernel_streaming_fd_dual<Cfg, B1, B2>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, Q1, LUT1, Q2, LUT2, resB, resA, bias, C, M, N, K, R);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_streaming_fd_dual launch failed (grid ",
                grid.x, "x", grid.y, ", ", Cfg::THREADS,
                " threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

// The chunk's u32 word w (w in [0, 4*B)) — static-indexed: the pair loop
// is fully unrolled, so w is a compile-time constant and the uint4 select
// folds into a register read (no dynamic register indexing, no spill).

// ---------------------------------------------------------------------------
// Dispatch: BK (K-tile depth) is decoupled from group_size. Every launched
// config satisfies K % BK == 0, which keeps the vectorized Q/A loads
// inside the row.
// ---------------------------------------------------------------------------
bool gs32_use_deep_tiles() {
    // FLUTE_GS32_BK=64 opts gs=32 layers into BK=64 K-tiles (half the K-loop
    // iterations; A/B with benchmark_kernel.py --gs32-bk 64). Read once per
    // process — set the env var BEFORE the first call.
    static const bool deep = [] {
        const char* env = std::getenv("FLUTE_GS32_BK");
        return env != nullptr && env[0] == '6' && env[1] == '4' && env[2] == '\0';
    }();
    return deep;
}

bool fd_disabled_by_env() {
    // FLUTE_NO_FD=1 is a compile-time-style escape hatch for A/B runs: it
    // makes the dispatcher REJECT idxN indices loudly instead of running
    // them (the legacy kernel cannot read that layout, so there is no
    // silent fallback). Read once per process.
    static const bool off = [] {
        const char* env = std::getenv("FLUTE_NO_FD");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    return off;
}

// ---------------------------------------------------------------------------
//  GS-generic dispatch: BK (K-tile depth) selection is DECOUPLED from
// GS (the LUT group size) — the LUT is indexed by N group only, never by
// k. Selection rule (preserves every older behavior verbatim):
//   * GS in {16, 32}: BK = 64 only via the FLUTE_GS32_BK=64 opt-in AND
//     K % 64 == 0 (the A/B experiment knob); default BK = 32.
//   * GS in {64, 128, 256, 512}: BK = 64 whenever K % 64 == 0 (the
//     older gs=64 rule), else BK = 32 (K % 32 == 0 is host-checked).
// B = 4 never routes through the sub-4 dispatchers — it keeps the
// original kernel entry points and their exact binaries.
// ---------------------------------------------------------------------------
template <int B, int GS>
void dispatch_streaming_sub4(const __half* a, const uint8_t* q, const __half* l, __half* c,
    int M, int N, int K
) {
    if constexpr (GS <= 32) {
        if (gs32_use_deep_tiles() && K % 64 == 0) {
            launch_streaming_sub4<TileConfig<64, 128, 64, GS>, B>(a, q, l, c, M, N, K);
        } else {
            launch_streaming_sub4<TileConfig<128, 128, 32, GS>, B>(a, q, l, c, M, N, K);
        }
    } else {
        if (K % 64 == 0) {
            launch_streaming_sub4<TileConfig<64, 128, 64, GS>, B>(a, q, l, c, M, N, K);
        } else {
            launch_streaming_sub4<TileConfig<128, 128, 32, GS>, B>(a, q, l, c, M, N, K);
        }
    }
}

template <int B, int GS>
void dispatch_fd_sub4(const __half* a, const uint8_t* q, const __half* l, __half* c,
    int M, int N, int K
) {
    if constexpr (GS <= 32) {
        if (gs32_use_deep_tiles() && K % 64 == 0) {
            launch_streaming_fd_sub4<TileConfig<64, 128, 64, GS>, B>(a, q, l, c, M, N, K);
        } else {
            launch_streaming_fd_sub4<TileConfig<128, 128, 32, GS>, B>(a, q, l, c, M, N, K);
        }
    } else {
        if (K % 64 == 0) {
            launch_streaming_fd_sub4<TileConfig<64, 128, 64, GS>, B>(a, q, l, c, M, N, K);
        } else {
            launch_streaming_fd_sub4<TileConfig<128, 128, 32, GS>, B>(a, q, l, c, M, N, K);
        }
    }
}

// the runtime-GS -> template-GS switch shared by every path below.
// Every branch is a hard TORCH_CHECK failure by construction (the default
// arm), so an unsupported GS can never fall through to a wrong config.
#define FLUTE_DISPATCH_GS(gs_val, ...)                                       \
    do {                                                                     \
        switch (int(gs_val)) {                                               \
            case 16: { constexpr int GS = 16;  __VA_ARGS__; break; }          \
            case 32: { constexpr int GS = 32;  __VA_ARGS__; break; }          \
            case 64: { constexpr int GS = 64;  __VA_ARGS__; break; }          \
            case 128: { constexpr int GS = 128; __VA_ARGS__; break; }         \
            case 256: { constexpr int GS = 256; __VA_ARGS__; break; }         \
            case 512: { constexpr int GS = 512; __VA_ARGS__; break; }         \
            case 1024: { constexpr int GS = 1024; __VA_ARGS__; break; }       \
            case 2048: { constexpr int GS = 2048; __VA_ARGS__; break; }       \
            default:                                                          \
                TORCH_CHECK(false, "group_size must be 16, 32, 64, 128, "    \
                                  "256, 512, 1024 or 2048 (got ",           \
                                  int(gs_val), ")");                        \
        }                                                                   \
    } while (0)

// The 4-bit legacy-path GS dispatch (the older inline gs==32/gs==64
// blocks, generalized; TileConfig keeps its BM/BN/BK selection rules).
template <int GS>
void dispatch_streaming_b4(const __half* a, const uint8_t* q, const __half* l, __half* c,
    int M, int N, int K
) {
    if constexpr (GS <= 32) {
        if (gs32_use_deep_tiles() && K % 64 == 0) {
            // Deep tiles: half the K-loop iterations, BM=64, 64 accumulator
            // registers per thread, smem 50,432 B.
            launch_streaming<TileConfig<64, 128, 64, GS>>(a, q, l, c, M, N, K);
        } else {
            // Default config; also the fallback when K % 64 != 0 (K % 32 == 0
            // is host-checked).
            launch_streaming<TileConfig<128, 128, 32, GS>>(a, q, l, c, M, N, K);
        }
    } else {
        if (K % 64 == 0) {
            launch_streaming<TileConfig<64, 128, 64, GS>>(a, q, l, c, M, N, K);
        } else {
            launch_streaming<TileConfig<128, 128, 32, GS>>(a, q, l, c, M, N, K);
        }
    }
}

// The 4-bit fd-path GS dispatch.
template <int GS>
void dispatch_fd_b4(const __half* a, const uint8_t* q, const __half* l, __half* c,
    int M, int N, int K
) {
    if constexpr (GS <= 32) {
        if (gs32_use_deep_tiles() && K % 64 == 0) {
            launch_streaming_fd<TileConfig<64, 128, 64, GS>>(a, q, l, c, M, N, K);
        } else {
            launch_streaming_fd<TileConfig<128, 128, 32, GS>>(a, q, l, c, M, N, K);
        }
    } else {
        if (K % 64 == 0) {
            launch_streaming_fd<TileConfig<64, 128, 64, GS>>(a, q, l, c, M, N, K);
        } else {
            launch_streaming_fd<TileConfig<128, 128, 32, GS>>(a, q, l, c, M, N, K);
        }
    }
}

// ---------------------------------------------------------------------------
// the dual-stream fused dispatch. Instantiation policy (kept SMALL on
// purpose — nvcc time is a real cost on the box):
//   * width pairs: the five two-stream composites the deployed artifacts
//     actually contain (parsed from the layer metadata: (1,4) x45, (4,1)
//     x24, (3,3) x61, (2,3) x3, (2,4) x6) plus the single-stream riders
//     (3,0)/(4,0) so lone modules get the same fused epilogue;
//   * GS: 64..512 (the deployed range covers 240/248 modules; GS 16/32
//     modules and any other pair keep the older two-launch route).
// 7 pairs x 4 GS = 28 instantiations. Unlisted combinations are REFUSED
// loudly here — the Python wrapper routes them to the two-launch path
// BEFORE reaching this entry (a perf fallback, never a correctness one).
// ---------------------------------------------------------------------------
#define FLUTE_DISPATCH_DUAL_GS(gs_val, ...)                                   \
    do {                                                                     \
        switch (int(gs_val)) {                                               \
            case 64:  { constexpr int GS = 64;  __VA_ARGS__; break; }        \
            case 128: { constexpr int GS = 128; __VA_ARGS__; break; }        \
            case 256: { constexpr int GS = 256; __VA_ARGS__; break; }        \
            case 512: { constexpr int GS = 512; __VA_ARGS__; break; }        \
            default:                                                          \
                TORCH_CHECK(false, "qgemm_dual_stream: group_size must be "  \
                                  "64, 128, 256 or 512 (got ", int(gs_val), \
                                  ") — the two-launch qgemm_per_group_lut "\
                                  "route serves other group sizes");        \
        }                                                                    \
    } while (0)

template <int B1, int B2, int GS>
void dispatch_fd_dual(const __half* a, const uint8_t* q1, const __half* l1,
    const uint8_t* q2, const __half* l2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* c, int M, int N, int K, int R
) {
    // BK = 64 unconditionally (the dual is the decode-slim shape; the
    // K % 64 == 0 precondition is host-checked).
    launch_streaming_fd_dual<TileConfig<32, 128, 64, GS>, B1, B2>(a, q1, l1, q2, l2, resB, resA, bias, c, M, N, K, R);
}

#define FLUTE_DUAL_PAIR(b1v, b2v)                                            \
    do {                                                                     \
        if (B1 == b1v && B2 == b2v) {                                         \
            FLUTE_DISPATCH_DUAL_GS(group_size,                                \
                dispatch_fd_dual<b1v, b2v, GS>(a_p, q1_p, l1_p, q2_p, l2_p,  \
                                               rb_p, ra_p, bi_p, c_p,        \
                                               M, N, K, R));                 \
            return C;                                                         \
        }                                                                     \
    } while (0)


torch::Tensor qgemm_cutlass_dual_stream_impl(torch::Tensor A,
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
    const int B1 = (int)bitwidth;
    const int B2 = (int)bitwidth2;
    const bool has_s2 = (B2 > 0);

    TORCH_CHECK(A.is_cuda(), "qgemm_dual_stream: A must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda() && lut.is_cuda(),
                "qgemm_dual_stream: stream-1 indices/lut must be CUDA");
    if (has_s2) {
        TORCH_CHECK(indices2.is_cuda() && lut2.is_cuda(),
                    "qgemm_dual_stream: stream-2 indices/lut must be CUDA");
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
    TORCH_CHECK(!fd_disabled_by_env(),
                "qgemm_dual_stream: FLUTE_NO_FD=1 is set (the dual kernel "
                "is a fragment-direct-path consumer)");

    TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
    const int M = A.size(0);
    const int K = A.size(1);
    TORCH_CHECK(K % 64 == 0,
                "qgemm_dual_stream: K must be a multiple of 64 (BK=64 "
                "K-tiles; got K=", K, ") — route the call through the "
                "two-launch qgemm_per_group_lut path instead");
    TORCH_CHECK(group_size == 64 || group_size == 128 ||
                group_size == 256 || group_size == 512,
                "qgemm_dual_stream: group_size must be 64/128/256/512 "
                "(got ", int(group_size), ") — route the call through the "
                "two-launch qgemm_per_group_lut path instead");

    // N from the stream-1 blob byte count (identical derivation to
    // qgemm_cutlass_streaming_impl; the module guarantees the module-level
    // blob contract, and both streams share N).
    const int64_t row_bytes1 = (int64_t)(K * B1) >> 3;
    const int N = (int)(indices.numel() / row_bytes1);
    TORCH_CHECK(N > 0 && (int64_t)N * row_bytes1 == indices.numel(),
                "qgemm_dual_stream: stream-1 byte count ", indices.numel(),
                " is not N*(K*", B1, "/8) for any N (K = ", K, ")");
    TORCH_CHECK(N % 128 == 0,
                "qgemm_dual_stream: idxN layout requires N % 128 == 0 "
                "(got N = ", N, ")");
    if (has_s2) {
        const int64_t row_bytes2 = (int64_t)(K * B2) >> 3;
        TORCH_CHECK(indices2.numel() == (int64_t)N * row_bytes2,
                    "qgemm_dual_stream: stream-2 byte count ",
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
                    "stream-2 lut must be [ceil(N/group_size), 2^bitwidth2] "
                    "= [", (N + group_size - 1) / group_size, ", ",
                    (1 << B2), "]");
    }
    TORCH_CHECK(A.is_contiguous() && indices.is_contiguous() &&
                lut.is_contiguous(),
                "A, stream-1 indices and lut must be contiguous");
    if (has_s2) {
        TORCH_CHECK(indices2.is_contiguous() && lut2.is_contiguous(),
                    "stream-2 indices/lut must be contiguous");
    }

    // Residual + bias contracts (all optional; empty tensor = absent).
    const bool has_res = resB.numel() > 0;
    TORCH_CHECK(has_res == (resA.numel() > 0),
                "qgemm_dual_stream: resB and resA must be supplied "
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
        TORCH_CHECK(R >= 1 && R <= 16,
                    "residual rank must be 1..16 (got R=", R, ")");
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

    // Degenerate shapes (mirrors qgemm_cutlass_streaming_impl).
    auto C = torch::empty({M, N}, A.options());
    if (M == 0 || N == 0 || K == 0) {
        return (K == 0 && M > 0 && N > 0)
            ? torch::zeros({M, N}, A.options()) : C;
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

    // The width-pair table (see the dispatch comment above).
    FLUTE_DUAL_PAIR(1, 4);
    FLUTE_DUAL_PAIR(4, 1);
    FLUTE_DUAL_PAIR(2, 3);
    FLUTE_DUAL_PAIR(2, 4);
    FLUTE_DUAL_PAIR(3, 3);
    FLUTE_DUAL_PAIR(3, 0);
    FLUTE_DUAL_PAIR(4, 0);

    TORCH_CHECK(false,
                "qgemm_dual_stream: unsupported (bitwidth, bitwidth2) = (",
                B1, ", ", B2, ") — the compiled pairs are (1,4), (4,1), "
                "(2,3), (2,4), (3,3), (3,0) and (4,0); route other pairs "
                "through the two-launch qgemm_per_group_lut path");
    return C;   // unreachable (TORCH_CHECK(false) above)
}
}  // namespace

torch::Tensor qgemm_cutlass_streaming_impl(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    int64_t group_size,
    int64_t q_layout
) {
    TORCH_CHECK(A.is_cuda() && indices.is_cuda() && lut.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(indices.dtype() == torch::kUInt8, "indices must be uint8");
    TORCH_CHECK(lut.dtype() == torch::kFloat16, "lut must be float16");
    TORCH_CHECK(bitwidth >= 1 && bitwidth <= 4,
                "bitwidth must be 1, 2, 3 or 4 (the idxN family; 4 is the "
                "classic 16-entry LUT path)");
    TORCH_CHECK(q_layout == 0 || q_layout == 1,
                "q_layout must be 0 (legacy packed) or 1 (idxN fragment "
                "blob; produced by flute_extended/idxN.py)");
    TORCH_CHECK(group_size == 16 || group_size == 32 || group_size == 64 ||
                group_size == 128 || group_size == 256 || group_size == 512 ||
                group_size == 1024 || group_size == 2048,
                "group_size must be one of 16/32/64/128/256/512/1024/2048 "
                "(full-GS-range +  GS>512 kernels; GS>512 requires a "
                "flute_extended rebuild: python setup.py build_ext --inplace)");
    TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
    TORCH_CHECK(lut.dim() == 2, "lut must be 2-D [num_groups, 2^bitwidth]");

    const int M = A.size(0);
    const int K = A.size(1);
    const int B = (int)bitwidth;
    // N comes from the indices tensor: the legacy layout is [N, K*B/8];
    // the idxN blob is tile-organized with the same byte count, so N is
    // derived from the total size (K % 64 == 0 — enforced below for the
    // blob path and implied by K % group_size for the legacy path —
    // makes K*B/8 exact for every width, including b=3).
    const int64_t row_bytes = (int64_t)(K * B) >> 3;
    const int N = (q_layout == 1)
        ? (int)(indices.numel() / row_bytes)
        : (int)indices.size(0);
    if (q_layout == 1) {
        TORCH_CHECK(indices.dim() == 1 || indices.dim() == 2,
                    "idxN indices must be a flat or 2-D uint8 tensor");
        TORCH_CHECK((int64_t)N * row_bytes == indices.numel() && N > 0,
                    "idxN byte count ", indices.numel(),
                    " is not N*(K*", B, "/8) for any N (K = ", K, ")");
    } else {
        TORCH_CHECK(indices.dim() == 2, "indices must be 2-D [N, K*bitwidth/8]");
    }
    TORCH_CHECK(K % 32 == 0,
                "K must be a multiple of 32 (the BK K-tile constraint; "
                "GS no longer constrains K — the decoupling)");
    if (q_layout == 0) {
        TORCH_CHECK(indices.size(1) == (int)row_bytes,
                    "indices must have shape [N, K*bitwidth/8]");
    }
    TORCH_CHECK(lut.size(1) == (1 << B),
                "lut must have 2^bitwidth = ", (1 << B),
                " entries per group, got ", lut.size(1));
    // The kernel indexes LUT row n/group_size for n in [0, N).
    TORCH_CHECK(lut.size(0) == (N + group_size - 1) / group_size,
                "lut must have shape [ceil(N/group_size), 2^bitwidth]: "
                "expected ",
                (N + group_size - 1) / group_size, " rows, got ", lut.size(0));
    TORCH_CHECK(A.is_contiguous() && indices.is_contiguous() && lut.is_contiguous(),
                "A, indices, lut must be contiguous");

    // Degenerate shapes: zero-sized grids are invalid launches; K == 0 must
    // yield zeros, not uninitialized memory.
    auto C = torch::empty({M, N}, A.options());
    if (M == 0 || N == 0 || K == 0) {
        return (K == 0 && M > 0 && N > 0) ? torch::zeros({M, N}, A.options()) : C;
    }

    // The vectorized loads (cp.async 16 B for A, uint4 for Q) require 16 B
    // alignment. Fresh torch allocations are 256 B aligned; views/slices
    // may not be — transparently clone into aligned storage.
    if (reinterpret_cast<uintptr_t>(A.data_ptr<at::Half>()) % 16 != 0) {
        A = A.clone();          // torch allocations are >= 256 B aligned
    }
    if (reinterpret_cast<uintptr_t>(indices.data_ptr<uint8_t>()) % 16 != 0) {
        indices = indices.clone();
    }

    const __half*  a_p = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
    const uint8_t* q_p = indices.data_ptr<uint8_t>();
    const __half*  l_p = reinterpret_cast<const __half*>(lut.data_ptr<at::Half>());
    __half*        c_p = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    // Fragment-direct path (E1+E2+E3): the repacked Q layout and the
    // divisibility it bakes in are HARD requirements — the legacy kernel
    // cannot read an idxN blob, so a fallback would silently compute
    // garbage. Fail loudly instead. (A/B against the legacy path = pass the
    // ORIGINAL layout-0 indices to the same entrypoint.)
    if (q_layout == 1) {
        TORCH_CHECK(!flute::fd_disabled_by_env(),
                    "idxN indices supplied but FLUTE_NO_FD=1 is set");
        TORCH_CHECK(N % 128 == 0 && K % 64 == 0,
                    "idxN indices require N % 128 == 0 and K % 64 == 0 "
                    "(got N = ", N, ", K = ", K, "); flute_extended/idxN.py "
                    "refuses to pack such layers — use the legacy layout");
        if (B < 4) {
            // sub-4-bit idxN family (bitwidth 1/2/3)
            switch (B) {
                case 1:
                    FLUTE_DISPATCH_GS(group_size,
                        dispatch_fd_sub4<1, GS>(a_p, q_p, l_p, c_p, M, N, K));
                    break;
                case 2:
                    FLUTE_DISPATCH_GS(group_size,
                        dispatch_fd_sub4<2, GS>(a_p, q_p, l_p, c_p, M, N, K));
                    break;
                case 3:
                    FLUTE_DISPATCH_GS(group_size,
                        dispatch_fd_sub4<3, GS>(a_p, q_p, l_p, c_p, M, N, K));
                    break;
            }
            return C;
        }
        FLUTE_DISPATCH_GS(group_size,
            dispatch_fd_b4<GS>(a_p, q_p, l_p, c_p, M, N, K));
        return C;
    }

    if (B < 4) {
        // sub-4-bit legacy path (logical packed rows [N, K*B/8])
        switch (B) {
            case 1:
                FLUTE_DISPATCH_GS(group_size,
                    dispatch_streaming_sub4<1, GS>(a_p, q_p, l_p, c_p, M, N, K));
                break;
            case 2:
                FLUTE_DISPATCH_GS(group_size,
                    dispatch_streaming_sub4<2, GS>(a_p, q_p, l_p, c_p, M, N, K));
                break;
            case 3:
                FLUTE_DISPATCH_GS(group_size,
                    dispatch_streaming_sub4<3, GS>(a_p, q_p, l_p, c_p, M, N, K));
                break;
        }
        return C;
    }

    FLUTE_DISPATCH_GS(group_size,
        dispatch_streaming_b4<GS>(a_p, q_p, l_p, c_p, M, N, K));

    return C;
}

// Public entrypoint
// q_layout: 0 = legacy packed layout [N, K*bitwidth/8] (LSB-first);
//           1 = idxN fragment blob (flute_extended/idxN.py; b=4 is the
//               classic idx4/idx4 byte layout) — register-direct
//               dequant path; shapes it does not cover are rejected loudly.
// bitwidth: 1, 2, 3 or 4 (lut width 2^bitwidth).
torch::Tensor qgemm_cutlass_streaming(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    int64_t group_size,
    int64_t q_layout
) {
    return qgemm_cutlass_streaming_impl(A, indices, lut, bitwidth, group_size,
                                        q_layout);
}


// Public entrypoint : the dual-stream fused decode kernel.
//   C = A @ W1^T (+ A @ W2^T) + (A @ resB^T) @ resA^T + bias
// ONE launch for the whole per-module op chain (both streams, the
// rank-<=16 residual, the bias; the (M, N) stream add never exists as a
// tensor). bitwidth2 = 0 selects the single-stream rider. Deployment
// contract: idxN blobs (the same layout family qgemm_cutlass_streaming
// consumes at q_layout=1), N % 128 == 0, K % 64 == 0, group_size in
// {64, 128, 256, 512}, (bitwidth, bitwidth2) one of the compiled pairs —
// the Python wrapper (flute_extended/flute_extended/__init__.py) mirrors
// the gate and routes everything else to the two-launch path.
torch::Tensor qgemm_cutlass_dual_stream(torch::Tensor A,
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
    return qgemm_cutlass_dual_stream_impl(A, indices, lut, bitwidth,
                                          indices2, lut2, bitwidth2,
                                          resB, resA, bias, group_size);
}

