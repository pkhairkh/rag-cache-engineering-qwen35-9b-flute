/**
 * src/kernel_lut_grad.cu — dL/dLUT scatter kernel (W5-T02,
 * docs/KERNEL_SPEC_DLDLUT.md; PROPOSAL §3.2).
 *
 *   dL/dW [n,k]   = sum_m grad_Y[m,n] * X[m,k]        (Tensor Cores)
 *   dL/dLUT[g,c]  = sum_{n in g} sum_{k: idx[n,k]=c} dL/dW[n,k]
 *
 * idxN extension (the backward-kernel family, DEQUANT_SPEC section 8):
 *   the same deterministic two-pass at bit widths 1/2/3 over the
 *   unified idxN blob — the scatter keys become 2^B codes, the blob
 *   walk is the width-parameterized pair walk (identical (n_local, k)
 *   positions at every width), and the workspace/reduce strides scale
 *   with 2^B. The 4-bit kernels and their binaries are UNCHANGED.
 *
 * The (N,K) dW tile never leaves shared memory: the mma contraction
 * accumulates it in registers, the epilogue writes it to sDW (fp32,
 * 16B-block XOR-8 swizzled), and the scatter pass — the byte-exact
 * inverse of the backward kernel's dequant_w_tile walk — keys each
 * (n_local, k) value into a per-block (BM/GS, 16) fp32 partial.
 *
 * Determinism (spec §2, binding): deterministic two-pass. Pass 1 (this
 * kernel): private register partials per thread -> fixed warp-shuffle
 * tree -> fixed warp-order smem sum -> ONE workspace slot owned
 * exclusively by the block. No atomics anywhere. Pass 2 (the reduce
 * kernel at the bottom): one thread per (group, code), fixed ascending
 * block order. G-B5b's torch.equal is assertable because of this.
 *
 * Geometry (spec §3): the backward GEMM's transpose problem — the
 * output tile is (BM=64 n-rows x BK=64 k-cols), the contraction runs
 * over m in BN=64 steps, 128 threads (4 warps, 2x2), warp tile 32x32.
 *   sA: grad_Y tile [2][BN=64 m][BM=64 n] fp16   16 KiB
 *   sB: X tile      [2][BN=64 m][BK=64 k] fp16   16 KiB
 *   sDW: dW tile    [BM=64 n][BK=64 k]    fp32   16 KiB
 *   warp partial stage: 4 x (BM/GS<=2) x 16 fp32      <= 512 B
 *   DYN_SMEM = 48 KiB + 512 B  ->  2 CTAs/SM (97.25 <= 100 KiB),
 *   smem-limited (the PTX_NOTES §1 arithmetic), __launch_bounds__(128,2).
 *
 * Fragment sourcing (spec §3; both operands are contraction-major in
 * their shared tiles, so both load .trans):
 *   A' [n, m] = grad_Y[m, n]  — ldmatrix.x4.trans over the b_addr-style
 *                              source tiles of sA (the derivation is in
 *                              the spec; the distribution lands exactly
 *                              on the mma A-fragment slots).
 *   B' [m, k] = X[m, k]       — the backward kernel's own B pattern:
 *                              ldmatrix_x4_trans + ldmatrix_bT_addr on
 *                              sB.
 *
 * Debug surface: FLUTE_LUTGRAD_CANARY poisons sDW with the fp32 NaN
 * pattern before the epilogue (spec §4) — the scatter walk must consume
 * every sDW cell, so a surviving NaN is a coverage regression.
 *
 * Target: NVIDIA A10G (sm_86). No changes to flute_extended (frozen).
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "flute/mma.cuh"
#include "flute/mma_bwd.cuh"

// ---------------------------------------------------------------------------
// Local dtype helper (the same reinterpret as the backward kernel's
// as_half; separate translation unit, no linkage sharing).
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ const __half* lut_as_half(const T* p) {
    return reinterpret_cast<const __half*>(p);
}

// ---------------------------------------------------------------------------
// Tile configuration (spec §3)
// ---------------------------------------------------------------------------
constexpr int LBM      = 64;   // dW tile n-rows per block
constexpr int LBN      = 64;   // m-contraction steps per iteration
constexpr int LBK      = 64;   // dW tile k-cols per block
constexpr int LTHREADS = 128;

constexpr int LM_TILES = LBN / 16;          // 4 contraction m-slices
constexpr int LWARP_M  = LBM / 2;           // 32 n-rows per warp
constexpr int LWARP_N  = LBK / 2;           // 32 k-cols per warp
constexpr int LWMT     = LWARP_M / 16;      // 2 mma m-tiles per warp
constexpr int LWNT     = LWARP_N / 8;       // 4 mma n-tiles per warp

constexpr int LSA_XOR  = 8;                 // 16B-block swizzle, both tiles
constexpr int LSB_XOR  = 8;

constexpr int LSA_BYTES = 2 * LBN * LBM * 2;   // 16 KiB
constexpr int LSB_BYTES = 2 * LBN * LBK * 2;   // 16 KiB
constexpr int LDW_BYTES = LBM * LBK * 4;       // 16 KiB
constexpr int LWP_PART = 4 * 4 * 16 * 4;        // 4 warps x max 4 groups
                                                // (GS=16 worst case) x 16
                                                // fp32 codes = 1024 B
constexpr int LDYN_SMEM = LSA_BYTES + LSB_BYTES + LDW_BYTES + LWP_PART;

// ---------------------------------------------------------------------------
// idxN sub-4-bit format constants (DEQUANT_SPEC section 8; the unified
// idxN layout family, flute_extended/flute_extended/idxN.py — the same
// numbers as kernel_backward_gemm.cu's SubB; separate translation unit,
// no linkage sharing). The pair positions of the 4-bit walk are
// width-independent; only the field packing (2*B bits per pair, 4*B
// bytes per thread segment) and the code count (2^B) differ.
// ---------------------------------------------------------------------------
template <int B>
struct SubB {
    static_assert(B == 1 || B == 2 || B == 3,
                  "SubB is the sub-4-bit backward extension; B=4 runs the "
                  "existing kernels above, unchanged");
    static constexpr int PAL        = 1 << B;   // LUT entries (scatter codes)
    static constexpr int SEG_WORDS  = B;        // u32 words per segment
    static constexpr int SEG_BYTES  = 4 * B;    // 16 pairs * 2B bits / 8
    static constexpr int CHUNK_BY   = 16 * B;   // per (wx, lane) chunk
    static constexpr int HALF_BY    = 512 * B;  // per 64-row half-tile
    static constexpr int TILE_BY    = 1024 * B; // per (128-row, 64-k) tile
};

// This thread's 4*B-byte segment for the block's (n0, k0) tile — the
// 4-bit arithmetic (the 4096/2048/64/16 constants) scaled by B/4.
template <int B>
__device__ __forceinline__ size_t blob_segment_offset_sub4(
    int n0, int k0, int K, int tid
) {
    using SB = SubB<B>;
    const int t = n0 / 128;
    const int gtile = k0 / 64;
    const int blob_half = (n0 % 128) / 64;      // 0 or 1 (the tile split)
    const size_t tile_base =
        ((size_t)t * (K / 64) + (size_t)gtile) * SB::TILE_BY;
    return tile_base + (size_t)blob_half * SB::HALF_BY
         + (size_t)(tid >> 2) * SB::CHUNK_BY
         + (size_t)(tid & 3) * SB::SEG_BYTES;
}

// Prefetch the segment as B u32 words (.nc scalar loads — the forward
// kernels' q_fd_prefetch_sub4 discipline; the segments are 4 B- but
// never 16 B-aligned below B=4).
template <int B>
__device__ __forceinline__ void q_seg_load_sub4(
    const uint8_t* seg, uint32_t (&q)[SubB<B>::SEG_WORDS]
) {
    #pragma unroll
    for (int w = 0; w < SubB<B>::SEG_WORDS; ++w)
        flute::ldg_nc_evict_first_u32(q[w], seg + 4 * w);
}

// Pair j's 2*B-bit field at bit offset 2*B*j of the segment, LSB-first:
// v0 = idx[n, k], v1 = idx[n, k+1]. b=1/2 never span a word; b=3 spans
// only when shift+6 > 32, and a spanning field ends strictly inside the
// segment, so q[w0+1] is always in range (the forward's
// dequant_tile_sub4 arithmetic, DEQUANT_SPEC section 8).
template <int B>
__device__ __forceinline__ void decode_pair_sub4(
    const uint32_t (&q)[SubB<B>::SEG_WORDS], int j, uint32_t& v0, uint32_t& v1
) {
    if constexpr (B == 1) {
        const uint32_t w = q[0];
        v0 = (w >> (2 * j)) & 1u;
        v1 = (w >> (2 * j + 1)) & 1u;
    } else if constexpr (B == 2) {
        const uint32_t f = (q[j >> 3] >> (4 * (j & 7))) & 0xFu;
        v0 = f & 3u;
        v1 = f >> 2;
    } else {                                           // B == 3
        const int bit = 6 * j;
        const int w0 = bit >> 5;
        const int s = bit & 31;
        uint32_t f = q[w0] >> s;
        if (s + 6 > 32) f |= q[w0 + 1] << (32 - s);
        f &= 0x3Fu;
        v0 = f & 7u;
        v1 = (f >> 3) & 7u;
    }
}

// ---------------------------------------------------------------------------
// sA staging: grad_Y [M, N] -> sA[m_local][n_local], 16 B chunks, the
// m-guard zero-fills ragged rows (the backward kernel's stage_sA with
// the m/n roles of the tile swapped).
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void lut_stage_sA(
    T* sA, const T* __restrict__ grad_y,
    int m_step, int n0, int M, int N, int tid
) {
    constexpr int CHUNKS = LBN * (LBM / 8);     // 512 x 16 B
    #pragma unroll 4
    for (int i = tid; i < CHUNKS; i += LTHREADS) {
        const int m_local = i >> 3;            // 8 chunks per row
        const int col8 = (i & 7) << 3;         // 8-half column block
        const bool in_bounds = (m_step + m_local) < M;
        const T* src = grad_y + (size_t)(m_step + m_local) * N + n0 + col8;
        T* dst = sA + (size_t)m_local * LBM
               + flute::swz_col16(m_local, col8, LSA_XOR);
        flute::cp_async_16_zfill(dst, src, in_bounds);
    }
}

// ---------------------------------------------------------------------------
// sB staging: X [M, K] -> sB[m_local][k_local], 16 B chunks, the same
// m-guard (K % 64 == 0 makes the k guard vacuous).
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void lut_stage_sB(
    T* sB, const T* __restrict__ x,
    int m_step, int k0, int M, int K, int tid
) {
    constexpr int CHUNKS = LBN * (LBK / 8);    // 512 x 16 B
    #pragma unroll 4
    for (int i = tid; i < CHUNKS; i += LTHREADS) {
        const int m_local = i >> 3;
        const int col8 = (i & 7) << 3;
        const bool in_bounds = (m_step + m_local) < M;
        const T* src = x + (size_t)(m_step + m_local) * K + k0 + col8;
        T* dst = sB + (size_t)m_local * LBK
               + flute::swz_col16(m_local, col8, LSB_XOR);
        flute::cp_async_16_zfill(dst, src, in_bounds);
    }
}

// ---------------------------------------------------------------------------
// MMA phase: acc += A'(sA[buf]) @ B'(sB[buf]) for one m-step — the
// contraction over the 64 staged m rows, 16-wide slices.
// ---------------------------------------------------------------------------
template <typename T, bool kTwin>
__device__ __forceinline__ void lut_mma_phase(
    const T* sA, const T* sB,
    int lane, int wy, int wx,
    float (&acc)[LWMT][LWNT][4]
) {
    #pragma unroll
    for (int mt = 0; mt < LM_TILES; ++mt) {
        // A-fragments of A'[n, m] over the 16x16 tile at
        // (n = wy*LWARP_M + wmt*16, m = mt*16): ldmatrix.x4.trans over
        // the sA source tiles — the b_addr-style formulas (spec §3's
        // derivation: .trans hands thread t {sA[2(l%4)][l/4], ...} =
        // {A'[l/4][2(l%4)], ...}, exactly the mma A-fragment).
        uint32_t afrag[LWMT][4];
        #pragma unroll
        for (int wmt = 0; wmt < LWMT; ++wmt) {
            const int n_base = wy * LWARP_M + wmt * 16;
            const int m_base = mt * 16;
            if constexpr (kTwin) {
                // scalar twin: read the A'-fragment positions directly
                // from sA (the transposed positions of the fragment
                // table); packs the same u32 pairs.
                const int g = lane >> 2;
                const int c = 2 * (lane & 3);
                auto load2 = [&](int n, int m) -> uint32_t {
                    const T h0 = sA[(size_t)m * LBM
                                    + flute::swz_half(m, n, LSA_XOR)];
                    const T h1 = sA[(size_t)(m + 1) * LBM
                                    + flute::swz_half(m + 1, n, LSA_XOR)];
                    return (uint32_t)*reinterpret_cast<const uint16_t*>(&h0)
                         | ((uint32_t)*reinterpret_cast<const uint16_t*>(&h1)
                            << 16);
                };
                afrag[wmt][0] = load2(n_base + g,     m_base + c);
                afrag[wmt][1] = load2(n_base + g + 8, m_base + c);
                afrag[wmt][2] = load2(n_base + g,     m_base + c + 8);
                afrag[wmt][3] = load2(n_base + g + 8, m_base + c + 8);
            } else {
                // source 8x8 tiles: matrix0 = sA[m_base+0..8][n_base],
                // matrix1 = sA[m_base+0..8][n_base+8], matrix2 =
                // sA[m_base+8..][n_base], matrix3 = sA[m_base+8..][n_base+8]
                // — the b_addr row/col selection on the [m][n] tile.
                const __half* addr = flute::ldmatrix_b_addr(
                    lut_as_half(sA), LBM, m_base, n_base, lane, LSA_XOR);
                flute::ldmatrix_x4_trans(
                    afrag[wmt][0], afrag[wmt][1],
                    afrag[wmt][2], afrag[wmt][3], addr);
            }
        }

        // B-fragments of B'[m, k] = sB[m][k] over the 16x8 output tiles
        // — the backward kernel's own B pattern (ldmatrix_x4_trans +
        // bT_addr), applied to sB.
        uint32_t bfrag[LWNT][2];
        if constexpr (kTwin) {
            #pragma unroll
            for (int nt = 0; nt < LWNT; ++nt) {
                const int g = lane >> 2;
                const int c = 2 * (lane & 3);
                const int m_base = mt * 16;  // contraction m for B-fragments
                auto load2 = [&](int m) -> uint32_t {
                    const T h0 = sB[(size_t)m * LBK
                                    + flute::swz_half(m, wx * LWARP_N
                                                      + nt * 8 + g,
                                                      LSB_XOR)];
                    const T h1 = sB[(size_t)(m + 1) * LBK
                                    + flute::swz_half(m + 1, wx * LWARP_N
                                                      + nt * 8 + g,
                                                      LSB_XOR)];
                    return (uint32_t)*reinterpret_cast<const uint16_t*>(&h0)
                         | ((uint32_t)*reinterpret_cast<const uint16_t*>(&h1)
                            << 16);
                };
                bfrag[nt][0] = load2(m_base + c);
                bfrag[nt][1] = load2(m_base + c + 8);
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < LWNT; nt += 2) {
                uint32_t r0, r1, r2, r3;
                const int m_base_kt = mt * 16;              // contraction m
                const int k_base = wx * LWARP_N + nt * 8;   // output k
                const __half* addr = flute::ldmatrix_bT_addr(
                    lut_as_half(sB), LBK, m_base_kt, k_base, lane,
                    LSB_XOR);
                flute::ldmatrix_x4_trans(r0, r1, r2, r3, addr);
                bfrag[nt][0] = r0;         bfrag[nt][1] = r1;
                bfrag[nt + 1][0] = r2;     bfrag[nt + 1][1] = r3;
            }
        }

        #pragma unroll
        for (int wmt = 0; wmt < LWMT; ++wmt) {
            #pragma unroll
            for (int nt = 0; nt < LWNT; ++nt) {
                if constexpr (std::is_same<T, __half>::value) {
                    flute::mma_m16n8k16_f32acc(afrag[wmt], bfrag[nt],
                                               acc[wmt][nt]);
                } else {
                    flute::mma_m16n8k16_f32acc_bf16(afrag[wmt], bfrag[nt],
                                                    acc[wmt][nt]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scatter walk — dequant_w_tile's inverse (the byte mapping is the
// backward kernel's, verbatim): for each of the 16 bytes of this
// thread's uint4 blob segment, the (n_local, k) and (n_local, k+1)
// values of sDW scatter into the private partial keyed by the nibbles.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// W12: groups per LBM tile. The dW tile is LBM=64 n-rows per block (n0 is
// 64-aligned). Every supported GS is either a divisor of 64 (16/32/64:
// exactly LBM/GS groups, n0 GS-aligned) or a multiple of 64 (128/256/512:
// the whole 64-row tile sits inside ONE group — a 64-row interval never
// contains an interior multiple of 128+, so no straddle). partial[] and
// the workspace slots are sized by this constant at every GS.
// ---------------------------------------------------------------------------
template <int GS>
constexpr int lbm_groups() {
    return (LBM % GS == 0) ? (LBM / GS) : 1;   // GS > LBM: 1
}

template <int GS>
struct lutgrad_gs_supported : std::bool_constant<
    GS == 16 || GS == 32 || GS == 64 || GS == 128 || GS == 256 || GS == 512> {};

template <int GS>
__device__ __forceinline__ void lut_scatter_segment(
    const uint4& w16, int chunk, int seg,
    const float* __restrict__ sDW,
    float (&partial)[lbm_groups<GS>()][16]
) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w16);
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t b = bytes[j];
        const int v  = j >> 2;
        const int d  = (j >> 1) & 1;
        const int s2 = j & 1;
        const int n_local = v * 16 + d * 8 + (chunk >> 2);
        const int k = 2 * (seg * 8 + (chunk & 3) + 4 * s2);
        const int row_w = n_local * LBK;
        const float w0 = sDW[row_w + flute::swz_word(n_local, k, LSA_XOR)];
        const float w1 =
            sDW[row_w + flute::swz_word(n_local, k + 1, LSA_XOR)];
        partial[n_local / GS][b & 0x0F] += w0;
        partial[n_local / GS][b >> 4]   += w1;
    }
}

// ---------------------------------------------------------------------------
// idxN sub-4-bit scatter walk — the 4-bit walk's inverse role at 2^B
// codes: pair j covers the SAME (n_local, k) cells; the keys are the
// pair-field values instead of the nibbles.
// ---------------------------------------------------------------------------
template <int B, int GS>
__device__ __forceinline__ void lut_scatter_segment_sub4(
    const uint32_t (&q)[SubB<B>::SEG_WORDS],
    int chunk, int seg,
    const float* __restrict__ sDW,
    float (&partial)[lbm_groups<GS>()][SubB<B>::PAL]
) {
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        uint32_t v0, v1;
        decode_pair_sub4<B>(q, j, v0, v1);
        const int v  = j >> 2;
        const int d  = (j >> 1) & 1;
        const int s2 = j & 1;
        const int n_local = v * 16 + d * 8 + (chunk >> 2);
        const int k = 2 * (seg * 8 + (chunk & 3) + 4 * s2);
        const int row_w = n_local * LBK;
        const float w0 = sDW[row_w + flute::swz_word(n_local, k, LSA_XOR)];
        const float w1 =
            sDW[row_w + flute::swz_word(n_local, k + 1, LSA_XOR)];
        partial[n_local / GS][v0] += w0;
        partial[n_local / GS][v1] += w1;
    }
}

// ---------------------------------------------------------------------------
// Pass 1: the fused GEMM + scatter kernel.
// ---------------------------------------------------------------------------
template <typename T, int GS, bool kTwin>
__global__ void __launch_bounds__(LTHREADS, 2)
lut_grad_scatter_kernel(
    const T* __restrict__ grad_y,        // [M, N]
    const T* __restrict__ x,             // [M, K]
    const uint8_t* __restrict__ blob,    // flat idx4 blob (N*K/2 bytes)
    float* __restrict__ workspace,       // [n_blocks][LBM/GS][16]
    int M, int N, int K
) {
    static_assert(lutgrad_gs_supported<GS>::value,
                  "group_size must be 16, 32, 64, 128, 256 or 512");
    static_assert((LBM % GS == 0) || (GS % LBM == 0),
                  "the n-tile must not straddle groups (GS divides LBM or "
                  "LBM divides GS)");

    // Grid: x = k-blocks (fastest), y = n-blocks (spec §3's L2 wave).
    const int n0 = blockIdx.y * LBM;
    const int k0 = blockIdx.x * LBK;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wy = warp_id >> 1;    // the n half of the warp tile
    const int wx = warp_id & 1;     // the k half

    extern __shared__ char smem_raw[];
    T* sA_all = reinterpret_cast<T*>(smem_raw);                    // [2][LBN][LBM]
    T* sB_all = reinterpret_cast<T*>(smem_raw + LSA_BYTES);        // [2][LBN][LBK]
    float* sDW = reinterpret_cast<float*>(smem_raw + LSA_BYTES + LSB_BYTES);
    float* warp_part = reinterpret_cast<float*>(
        smem_raw + LSA_BYTES + LSB_BYTES + LDW_BYTES);             // [4][16]
    T* sA[2] = {sA_all, sA_all + (size_t)LBN * LBM};
    T* sB[2] = {sB_all, sB_all + (size_t)LBN * LBK};

    float acc[LWMT][LWNT][4];
    #pragma unroll
    for (int wmt = 0; wmt < LWMT; ++wmt)
        #pragma unroll
        for (int nt = 0; nt < LWNT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[wmt][nt][i] = 0.0f;

    const int m_steps = (M + LBN - 1) / LBN;
    const int blob_half = (n0 % 128) / 64;   // 0 or 1 (the tile split)

    // Prologue: stage sA[0] and sB[0]; one barrier publishes both.
    lut_stage_sA<T>(sA[0], grad_y, 0, n0, M, N, tid);
    lut_stage_sB<T>(sB[0], x, 0, k0, M, K, tid);
    flute::cp_async_commit();
    flute::cp_async_wait_all();
    __syncthreads();

    for (int step = 0; step < m_steps; ++step) {
        const int buf = step & 1;
        lut_mma_phase<T, kTwin>(sA[buf], sB[buf], lane, wy, wx, acc);
        if (step + 1 < m_steps) {
            lut_stage_sA<T>(sA[buf ^ 1], grad_y, (step + 1) * LBN, n0, M, N,
                            tid);
            lut_stage_sB<T>(sB[buf ^ 1], x, (step + 1) * LBN, k0, M, K, tid);
            flute::cp_async_commit();
            flute::cp_async_wait_all();
        }
        __syncthreads();
    }

#ifdef FLUTE_LUTGRAD_CANARY
    // Poison sDW (debug builds): the scatter walk must consume every
    // cell — a surviving NaN is a coverage regression (spec §4).
    {
        uint32_t* words = reinterpret_cast<uint32_t*>(sDW);
        for (int i = tid; i < LBM * LBK; i += LTHREADS)
            words[i] = 0x7FC00000u;
        __syncthreads();
    }
#endif

    // Epilogue: the warp accumulator fragments -> sDW (fp32 words at
    // the 16B-block XOR-8-swizzled positions; the m/n roles of the
    // backward's epilogue, transposed).
    #pragma unroll
    for (int wmt = 0; wmt < LWMT; ++wmt) {
        #pragma unroll
        for (int nt = 0; nt < LWNT; ++nt) {
            const int row_base = wy * LWARP_M + wmt * 16;
            const int col_base = wx * LWARP_N + nt * 8;
            const int g = lane >> 2;
            const int c = 2 * (lane & 3);
            const int nr = row_base + g;
            const int kc = col_base + c;
            sDW[nr * LBK + flute::swz_word(nr, kc, LSA_XOR)] =
                acc[wmt][nt][0];
            sDW[nr * LBK + flute::swz_word(nr, kc + 1, LSA_XOR)] =
                acc[wmt][nt][1];
            sDW[(nr + 8) * LBK + flute::swz_word(nr + 8, kc, LSA_XOR)] =
                acc[wmt][nt][2];
            sDW[(nr + 8) * LBK + flute::swz_word(nr + 8, kc + 1, LSA_XOR)] =
                acc[wmt][nt][3];
        }
    }
    __syncthreads();

    // Scatter pass: this thread's blob segment (the backward's
    // blob_segment_offset, verbatim) keys the sDW values into the
    // private register partial.
    {
        const int t = n0 / 128;
        const int gtile = k0 / 64;
        const size_t tile_base =
            ((size_t)t * (K / 64) + (size_t)gtile) * 4096;
        const size_t seg_off = tile_base + (size_t)blob_half * 2048
                             + (size_t)(tid >> 2) * 64
                             + (size_t)(tid & 3) * 16;
        uint4 w16;
        flute::ldg_nc_evict_first_v4(w16, blob + seg_off);
        float partial[lbm_groups<GS>()][16];
        #pragma unroll
        for (int gi = 0; gi < lbm_groups<GS>(); ++gi)
            #pragma unroll
            for (int c = 0; c < 16; ++c)
                partial[gi][c] = 0.0f;
        lut_scatter_segment<GS>(w16, tid >> 2, tid & 3, sDW, partial);

        // Deterministic in-block reduction (spec §2): a fixed
        // warp-shuffle tree (offsets 16, 8, 4, 2, 1), then a fixed
        // warp-order smem sum — no atomics, no ordering hazards.
        // warp_part layout: [4 warps][lbm_groups groups][16 codes].
        #pragma unroll
        for (int gi = 0; gi < lbm_groups<GS>(); ++gi) {
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float v = partial[gi][c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_down_sync(0xFFFFFFFFu, v, off);
                if (lane == 0)
                    warp_part[(warp_id * lbm_groups<GS>() + gi) * 16 + c] = v;
            }
        }
        __syncthreads();
        // One thread per (gi, c) sums the four warp partials in the
        // FIXED warp order 0, 1, 2, 3 and writes the block's workspace
        // slot (owned exclusively by this block — no atomics).
        const int slot = (int)(blockIdx.y * gridDim.x + blockIdx.x);
        float* wslot = workspace + (size_t)slot * lbm_groups<GS>() * 16;
        for (int i = tid; i < lbm_groups<GS>() * 16; i += LTHREADS) {
            const int gi = i >> 4;
            const int c = i & 15;
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; ++w)
                v += warp_part[(w * lbm_groups<GS>() + gi) * 16 + c];
            wslot[i] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1, idxN sub-4-bit family (B in {1,2,3}): the GEMM, epilogue and
// determinism contract are the 4-bit kernel's verbatim; the scatter keys
// become 2^B codes and the segment is the 4*B-byte pair walk. The
// workspace slot and warp_part strides scale with PAL = 2^B.
// ---------------------------------------------------------------------------
template <typename T, int B, int GS, bool kTwin>
__global__ void __launch_bounds__(LTHREADS, 2)
lut_grad_scatter_sub4_kernel(
    const T* __restrict__ grad_y,        // [M, N]
    const T* __restrict__ x,             // [M, K]
    const uint8_t* __restrict__ blob,    // flat idxN blob (N*K*B/8 bytes)
    float* __restrict__ workspace,       // [n_blocks][LBM/GS][2^B]
    int M, int N, int K
) {
    using SB = SubB<B>;
    static_assert(lutgrad_gs_supported<GS>::value,
                  "group_size must be 16, 32, 64, 128, 256 or 512");
    static_assert((LBM % GS == 0) || (GS % LBM == 0),
                  "the n-tile must not straddle groups (GS divides LBM or "
                  "LBM divides GS)");

    // Grid: x = k-blocks (fastest), y = n-blocks (spec §3's L2 wave).
    const int n0 = blockIdx.y * LBM;
    const int k0 = blockIdx.x * LBK;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wy = warp_id >> 1;    // the n half of the warp tile
    const int wx = warp_id & 1;     // the k half

    extern __shared__ char smem_raw[];
    T* sA_all = reinterpret_cast<T*>(smem_raw);                    // [2][LBN][LBM]
    T* sB_all = reinterpret_cast<T*>(smem_raw + LSA_BYTES);        // [2][LBN][LBK]
    float* sDW = reinterpret_cast<float*>(smem_raw + LSA_BYTES + LSB_BYTES);
    float* warp_part = reinterpret_cast<float*>(
        smem_raw + LSA_BYTES + LSB_BYTES + LDW_BYTES);             // [4][16]
    T* sA[2] = {sA_all, sA_all + (size_t)LBN * LBM};
    T* sB[2] = {sB_all, sB_all + (size_t)LBN * LBK};

    float acc[LWMT][LWNT][4];
    #pragma unroll
    for (int wmt = 0; wmt < LWMT; ++wmt)
        #pragma unroll
        for (int nt = 0; nt < LWNT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[wmt][nt][i] = 0.0f;

    const int m_steps = (M + LBN - 1) / LBN;

    // Prologue: stage sA[0] and sB[0]; one barrier publishes both.
    lut_stage_sA<T>(sA[0], grad_y, 0, n0, M, N, tid);
    lut_stage_sB<T>(sB[0], x, 0, k0, M, K, tid);
    flute::cp_async_commit();
    flute::cp_async_wait_all();
    __syncthreads();

    for (int step = 0; step < m_steps; ++step) {
        const int buf = step & 1;
        lut_mma_phase<T, kTwin>(sA[buf], sB[buf], lane, wy, wx, acc);
        if (step + 1 < m_steps) {
            lut_stage_sA<T>(sA[buf ^ 1], grad_y, (step + 1) * LBN, n0, M, N,
                            tid);
            lut_stage_sB<T>(sB[buf ^ 1], x, (step + 1) * LBN, k0, M, K, tid);
            flute::cp_async_commit();
            flute::cp_async_wait_all();
        }
        __syncthreads();
    }

#ifdef FLUTE_LUTGRAD_CANARY
    // Poison sDW (debug builds): the scatter walk must consume every
    // cell — a surviving NaN is a coverage regression (spec §4).
    {
        uint32_t* words = reinterpret_cast<uint32_t*>(sDW);
        for (int i = tid; i < LBM * LBK; i += LTHREADS)
            words[i] = 0x7FC00000u;
        __syncthreads();
    }
#endif

    // Epilogue: the warp accumulator fragments -> sDW (fp32 words at
    // the 16B-block XOR-8-swizzled positions; the m/n roles of the
    // backward's epilogue, transposed).
    #pragma unroll
    for (int wmt = 0; wmt < LWMT; ++wmt) {
        #pragma unroll
        for (int nt = 0; nt < LWNT; ++nt) {
            const int row_base = wy * LWARP_M + wmt * 16;
            const int col_base = wx * LWARP_N + nt * 8;
            const int g = lane >> 2;
            const int c = 2 * (lane & 3);
            const int nr = row_base + g;
            const int kc = col_base + c;
            sDW[nr * LBK + flute::swz_word(nr, kc, LSA_XOR)] =
                acc[wmt][nt][0];
            sDW[nr * LBK + flute::swz_word(nr, kc + 1, LSA_XOR)] =
                acc[wmt][nt][1];
            sDW[(nr + 8) * LBK + flute::swz_word(nr + 8, kc, LSA_XOR)] =
                acc[wmt][nt][2];
            sDW[(nr + 8) * LBK + flute::swz_word(nr + 8, kc + 1, LSA_XOR)] =
                acc[wmt][nt][3];
        }
    }
    __syncthreads();

    // Scatter pass: this thread's 4*B-byte pair segment keys the sDW
    // values into the private register partial (2^B codes).
    {
        uint32_t q[SB::SEG_WORDS];
        q_seg_load_sub4<B>(
            blob + blob_segment_offset_sub4<B>(n0, k0, K, tid), q);
        float partial[lbm_groups<GS>()][SB::PAL];
        #pragma unroll
        for (int gi = 0; gi < lbm_groups<GS>(); ++gi)
            #pragma unroll
            for (int c = 0; c < SB::PAL; ++c)
                partial[gi][c] = 0.0f;
        lut_scatter_segment_sub4<B, GS>(q, tid >> 2, tid & 3, sDW, partial);

        // Deterministic in-block reduction (spec §2, verbatim): a fixed
        // warp-shuffle tree (offsets 16, 8, 4, 2, 1), then a fixed
        // warp-order smem sum — no atomics, no ordering hazards.
        // warp_part layout: [4 warps][lbm_groups groups][2^B codes].
        #pragma unroll
        for (int gi = 0; gi < lbm_groups<GS>(); ++gi) {
            #pragma unroll
            for (int c = 0; c < SB::PAL; ++c) {
                float v = partial[gi][c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_down_sync(0xFFFFFFFFu, v, off);
                if (lane == 0)
                    warp_part[(warp_id * lbm_groups<GS>() + gi) * SB::PAL + c] = v;
            }
        }
        __syncthreads();
        // One thread per (gi, c) sums the four warp partials in the
        // FIXED warp order 0, 1, 2, 3 and writes the block's workspace
        // slot (owned exclusively by this block — no atomics).
        const int slot = (int)(blockIdx.y * gridDim.x + blockIdx.x);
        float* wslot = workspace + (size_t)slot * lbm_groups<GS>() * SB::PAL;
        for (int i = tid; i < lbm_groups<GS>() * SB::PAL; i += LTHREADS) {
            const int gi = i / SB::PAL;
            const int c = i % SB::PAL;
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; ++w)
                v += warp_part[(w * lbm_groups<GS>() + gi) * SB::PAL + c];
            wslot[i] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2, idxN sub-4-bit: the deterministic reduce over [n_blocks]
// [LBM/GS][2^B] workspace into [n_groups, 2^B] — the 4-bit reduce's
// fixed ascending k-block order, the 16-thread-per-group granularity
// kept (threads with c >= 2^B retire; the block count is unchanged).
// ---------------------------------------------------------------------------
template <int GS, int B>
__global__ void lut_grad_reduce_sub4_kernel(
    const float* __restrict__ workspace,  // [n_blocks][lbm_groups][2^B]
    float* __restrict__ grad_lut,         // [n_groups, 2^B]
    int n_groups, int n_kblocks
) {
    constexpr int PAL = 1 << B;
    const int g = blockIdx.x * 8 + (threadIdx.x >> 4);
    const int c = threadIdx.x & 15;
    if (g >= n_groups || c >= PAL) return;
    // W12: the n-blocks owning group g (LBM = 64 rows per block). GS <=
    // LBM (GS | 64): exactly ONE block, b1 == b0 — the pre-W12 rule,
    // bit-identical order. GS > LBM: blocks [b0, b1] each hold their
    // single slot for g — summed in the fixed ascending (b, kb) order.
    const int stride = lbm_groups<GS>() * PAL;
    const int b0 = (g * GS) / LBM;
    const int b1 = (g * GS + GS - 1) / LBM;
    const int gi = g - (b0 * LBM) / GS;
    float v = 0.0f;
    for (int b = b0; b <= b1; ++b) {
        const float* base = workspace + (size_t)b * n_kblocks * stride
                          + (size_t)gi * PAL + c;
        for (int kb = 0; kb < n_kblocks; ++kb)
            v += base[(size_t)kb * stride];
    }
    grad_lut[(size_t)g * PAL + c] = v;
}

// ---------------------------------------------------------------------------
// Pass 2: the deterministic reduce — one thread per (group, code),
// fixed ascending k-block order (spec §2).
// ---------------------------------------------------------------------------
template <int GS>
__global__ void lut_grad_reduce_kernel(
    const float* __restrict__ workspace,  // [n_blocks][lbm_groups][16]
    float* __restrict__ grad_lut,         // [n_groups, 16]
    int n_groups, int n_kblocks
) {
    // Block b owns groups [b*8, min((b+1)*8, n_groups)); thread
    // (gi_local*16 + c) reduces one (group, code) cell.
    const int g = blockIdx.x * 8 + (threadIdx.x >> 4);
    const int c = threadIdx.x & 15;
    if (g >= n_groups) return;
    // W12: the owning n-block range (see the sub-4 twin above) — b1 == b0
    // for GS <= LBM (the pre-W12 single-block rule, bit-identical order).
    const int stride = lbm_groups<GS>() * 16;
    const int b0 = (g * GS) / LBM;
    const int b1 = (g * GS + GS - 1) / LBM;
    const int gi = g - (b0 * LBM) / GS;
    float v = 0.0f;
    for (int b = b0; b <= b1; ++b) {
        const float* base = workspace + (size_t)b * n_kblocks * stride
                          + (size_t)gi * 16 + c;
        for (int kb = 0; kb < n_kblocks; ++kb)
            v += base[(size_t)kb * stride];
    }
    grad_lut[(size_t)g * 16 + c] = v;
}

// ---------------------------------------------------------------------------
// Host launcher + wrapper (the backward's validation style; spec §4)
// ---------------------------------------------------------------------------
namespace {

template <typename T, int GS, bool kTwin>
void launch_lut_grad(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor indices,
    torch::Tensor workspace, torch::Tensor grad_lut,
    int M, int N, int K, int n_kblocks
) {
    static const bool smem_attr_set = [] {
        const cudaError_t err = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(
                &lut_grad_scatter_kernel<T, GS, kTwin>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, LDYN_SMEM);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaFuncSetAttribute(smem ", LDYN_SMEM,
                    ") failed: ", cudaGetErrorString(err));
        return true;
    }();
    (void)smem_attr_set;

    dim3 grid((unsigned)n_kblocks, (unsigned)(N / LBM));
    dim3 block(LTHREADS);
    auto stream = at::cuda::getCurrentCUDAStream();

    lut_grad_scatter_kernel<T, GS, kTwin>
        <<<grid, block, LDYN_SMEM, stream>>>(
            reinterpret_cast<const T*>(grad_y.data_ptr()),
            reinterpret_cast<const T*>(x.data_ptr()),
            indices.data_ptr<uint8_t>(),
            workspace.data_ptr<float>(),
            M, N, K);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "lut_grad_scatter launch failed: ",
                cudaGetErrorString(err), " (grid ", grid.x, "x", grid.y,
                ", smem ", LDYN_SMEM, " B)");

    const int n_groups = (N + GS - 1) / GS;
    lut_grad_reduce_kernel<GS>
        <<<dim3((unsigned)((n_groups + 7) / 8)), dim3(128), 0, stream>>>(
            workspace.data_ptr<float>(),
            grad_lut.data_ptr<float>(), n_groups, n_kblocks);
    const cudaError_t err2 = cudaGetLastError();
    TORCH_CHECK(err2 == cudaSuccess,
                "lut_grad_reduce launch failed: ",
                cudaGetErrorString(err2));
}

// idxN sub-4-bit launcher (the same pattern on the sub4 symbols; the
// reduce kernel is the width-parameterized twin).
template <typename T, int B, int GS, bool kTwin>
void launch_lut_grad_sub4(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor indices,
    torch::Tensor workspace, torch::Tensor grad_lut,
    int M, int N, int K, int n_kblocks
) {
    static const bool smem_attr_set = [] {
        const cudaError_t err = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(
                &lut_grad_scatter_sub4_kernel<T, B, GS, kTwin>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, LDYN_SMEM);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaFuncSetAttribute(smem ", LDYN_SMEM,
                    ") failed: ", cudaGetErrorString(err));
        return true;
    }();
    (void)smem_attr_set;

    dim3 grid((unsigned)n_kblocks, (unsigned)(N / LBM));
    dim3 block(LTHREADS);
    auto stream = at::cuda::getCurrentCUDAStream();

    lut_grad_scatter_sub4_kernel<T, B, GS, kTwin>
        <<<grid, block, LDYN_SMEM, stream>>>(
            reinterpret_cast<const T*>(grad_y.data_ptr()),
            reinterpret_cast<const T*>(x.data_ptr()),
            indices.data_ptr<uint8_t>(),
            workspace.data_ptr<float>(),
            M, N, K);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "lut_grad_scatter_sub4(b=", B, ") launch failed: ",
                cudaGetErrorString(err), " (grid ", grid.x, "x", grid.y,
                ", smem ", LDYN_SMEM, " B)");

    const int n_groups = (N + GS - 1) / GS;
    lut_grad_reduce_sub4_kernel<GS, B>
        <<<dim3((unsigned)((n_groups + 7) / 8)), dim3(128), 0, stream>>>(
            workspace.data_ptr<float>(),
            grad_lut.data_ptr<float>(), n_groups, n_kblocks);
    const cudaError_t err2 = cudaGetLastError();
    TORCH_CHECK(err2 == cudaSuccess,
                "lut_grad_reduce_sub4 launch failed: ",
                cudaGetErrorString(err2));
}

// W12 GS dispatch helper, sub-4 tree: the runtime group_size ->
// template GS switch (all six values valid — see lbm_groups()).
template <typename T, int B, bool kTwin>
void lut_grad_sub4_gs_dispatch(
    torch::Tensor grad_y_a, torch::Tensor x_a, torch::Tensor indices_a,
    torch::Tensor workspace, torch::Tensor grad_lut, int M, int N, int K,
    int n_kblocks, int64_t group_size
) {
    switch ((int)group_size) {
        case 16:
            launch_lut_grad_sub4<T, B, 16, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 32:
            launch_lut_grad_sub4<T, B, 32, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 64:
            launch_lut_grad_sub4<T, B, 64, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 128:
            launch_lut_grad_sub4<T, B, 128, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 256:
            launch_lut_grad_sub4<T, B, 256, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 512:
            launch_lut_grad_sub4<T, B, 512, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        default:
            TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512, "
                              "got ", group_size);
    }
}

// W12 GS dispatch helper, 4-bit tree.
template <typename T, bool kTwin>
void lut_grad_b4_gs_dispatch(
    torch::Tensor grad_y_a, torch::Tensor x_a, torch::Tensor indices_a,
    torch::Tensor workspace, torch::Tensor grad_lut, int M, int N, int K,
    int n_kblocks, int64_t group_size
) {
    switch ((int)group_size) {
        case 16:
            launch_lut_grad<T, 16, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 32:
            launch_lut_grad<T, 32, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 64:
            launch_lut_grad<T, 64, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 128:
            launch_lut_grad<T, 128, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 256:
            launch_lut_grad<T, 256, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        case 512:
            launch_lut_grad<T, 512, kTwin>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks);
            break;
        default:
            TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512, "
                              "got ", group_size);
    }
}

// idxN sub-4-bit dtype/GS dispatch (B bound by the caller's switch).
template <int B>
void lut_grad_sub4_dispatch(
    torch::Tensor grad_y_a, torch::Tensor x_a, torch::Tensor indices_a,
    torch::Tensor workspace, torch::Tensor grad_lut, bool fp16, bool twin,
    int M, int N, int K, int n_kblocks, int64_t group_size
) {
    if (twin) {
        if (fp16)
            lut_grad_sub4_gs_dispatch<__half, B, true>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks, group_size);
        else
            lut_grad_sub4_gs_dispatch<__nv_bfloat16, B, true>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks, group_size);
    } else {
        if (fp16)
            lut_grad_sub4_gs_dispatch<__half, B, false>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks, group_size);
        else
            lut_grad_sub4_gs_dispatch<__nv_bfloat16, B, false>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, M, N, K,
                n_kblocks, group_size);
    }
}

torch::Tensor lut_grad_scatter_impl(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor indices,
    int64_t N_, int64_t K_, int64_t group_size, bool twin,
    int64_t bitwidth, const std::string& indices_layout
) {
    TORCH_CHECK(bitwidth >= 1 && bitwidth <= 4,
                "bitwidth must be 1, 2, 3 or 4 (the idxN family), got ",
                bitwidth);
    if (!indices_layout.empty()) {
        TORCH_CHECK(indices_layout == "idx" + std::to_string(bitwidth),
                    "indices_layout \"", indices_layout, "\" must match "
                    "bitwidth ", bitwidth, " (expected \"idx",
                    bitwidth, "\")");
    }
    TORCH_CHECK(grad_y.is_cuda() && x.is_cuda() && indices.is_cuda(),
                "all tensors must be CUDA");
    const bool fp16 = grad_y.scalar_type() == torch::kFloat16;
    const bool bf16 = grad_y.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(fp16 || bf16, "grad_y must be float16 or bfloat16");
    TORCH_CHECK(x.scalar_type() == grad_y.scalar_type(),
                "x and grad_y must share a dtype (dispatch contract)");
    TORCH_CHECK(indices.scalar_type() == torch::kUInt8,
                "indices must be uint8");
    TORCH_CHECK(grad_y.dim() == 2, "grad_y must be 2-D [M, N]");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [M, K]");
    TORCH_CHECK(group_size == 16 || group_size == 32 || group_size == 64 ||
                group_size == 128 || group_size == 256 || group_size == 512,
                "group_size must be one of 16/32/64/128/256/512 "
                "(W12 full-GS-range kernels)");

    const int64_t M = grad_y.size(0);
    const int64_t N = grad_y.size(1);
    const int64_t K = x.size(1);
    TORCH_CHECK(N == N_, "grad_y N mismatch");
    TORCH_CHECK(K == K_, "x K mismatch");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64");
    TORCH_CHECK(M > 0, "M must be positive");
    const int64_t expected_bytes = N * K * bitwidth / 8;
    TORCH_CHECK(indices.numel() == expected_bytes,
                "idx", bitwidth, " indices must hold N*K*", bitwidth,
                "/8 = ", expected_bytes, " bytes, got ", indices.numel());
    TORCH_CHECK(grad_y.is_contiguous() && x.is_contiguous()
                && indices.is_contiguous(),
                "grad_y, x and indices must be contiguous");

    // 16 B alignment for cp.async / ld.global.nc; clone to repair
    // (the backward kernel's pattern).
    auto grad_y_a = (reinterpret_cast<uintptr_t>(grad_y.data_ptr()) & 15)
                    == 0 ? grad_y : grad_y.clone();
    auto x_a = (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0
               ? x : x.clone();
    auto indices_a =
        (reinterpret_cast<uintptr_t>(indices.data_ptr()) & 15) == 0
        ? indices : indices.clone();

    const int n_kblocks = (int)(K / LBK);
    const int n_nblocks = (int)(N / LBM);
    // W12: groups per n-block — LBM/group_size when GS <= LBM, else 1
    // (the whole 64-row block sits inside one group; the reduce kernel
    // sums the owning block range).
    const int64_t groups_per_block =
        ((int)group_size <= LBM) ? (LBM / (int)group_size) : 1;
    const int64_t codes = 1 << bitwidth;
    auto workspace = torch::zeros(
        {n_nblocks * n_kblocks, groups_per_block * codes},
        grad_y.options().dtype(torch::kFloat32));
    const int64_t n_groups = (N + group_size - 1) / group_size;
    auto grad_lut = torch::zeros({n_groups, codes},
                                 grad_y.options().dtype(torch::kFloat32));

    if (bitwidth < 4) {
        // idxN sub-4-bit dispatch (the forward kernels' switch style)
        switch (bitwidth) {
            case 1: lut_grad_sub4_dispatch<1>(
                        grad_y_a, x_a, indices_a, workspace, grad_lut, fp16,
                        twin, (int)M, (int)N, (int)K, n_kblocks, group_size);
                    break;
            case 2: lut_grad_sub4_dispatch<2>(
                        grad_y_a, x_a, indices_a, workspace, grad_lut, fp16,
                        twin, (int)M, (int)N, (int)K, n_kblocks, group_size);
                    break;
            case 3: lut_grad_sub4_dispatch<3>(
                        grad_y_a, x_a, indices_a, workspace, grad_lut, fp16,
                        twin, (int)M, (int)N, (int)K, n_kblocks, group_size);
                    break;
            default:
                TORCH_CHECK(false, "unreachable bitwidth ", bitwidth);
        }
        return grad_lut;
    }
    if (twin) {
        if (fp16)
            lut_grad_b4_gs_dispatch<__half, true>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, (int)M,
                (int)N, (int)K, n_kblocks, group_size);
        else
            lut_grad_b4_gs_dispatch<__nv_bfloat16, true>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, (int)M,
                (int)N, (int)K, n_kblocks, group_size);
    } else {
        if (fp16)
            lut_grad_b4_gs_dispatch<__half, false>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, (int)M,
                (int)N, (int)K, n_kblocks, group_size);
        else
            lut_grad_b4_gs_dispatch<__nv_bfloat16, false>(
                grad_y_a, x_a, indices_a, workspace, grad_lut, (int)M,
                (int)N, (int)K, n_kblocks, group_size);
    }
    return grad_lut;
}

}  // namespace

torch::Tensor lut_grad_scatter(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor indices,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout
) {
    return lut_grad_scatter_impl(grad_y, x, indices, N, K, group_size,
                                 /*twin=*/false, bitwidth, indices_layout);
}

torch::Tensor lut_grad_scatter_twin(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor indices,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout
) {
    // Scalar-fragment differential twin (the G-B2 lineage): the same
    // staging, mma instruction order and scatter walk, fragments
    // assembled by scalar reads instead of ldmatrix.
    return lut_grad_scatter_impl(grad_y, x, indices, N, K, group_size,
                                 /*twin=*/true, bitwidth, indices_layout);
}
