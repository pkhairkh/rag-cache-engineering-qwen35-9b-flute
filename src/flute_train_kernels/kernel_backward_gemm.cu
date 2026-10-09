/**
 * src/kernel_backward_gemm.cu
 *
 * Fused backward GEMM for QLoRA training (kernel v2, TASKS.md waves 2-4):
 *   grad_X[M, K] = grad_Y[M, N] @ W[N, K]
 *   where W[n, k] = LUT[n / group_size, idx4_index(n, k)]
 *
 * idxN extension (the backward-kernel family, DEQUANT_SPEC section 8):
 *   the same GEMM at bit widths 1/2/3 — W[n, k] = LUT[n/GS, idx_b(n, k)]
 *   over the unified idxN blob (flute_extended/idxN.py). The 4-bit
 *   kernels, their geometry, pipeline and binaries are UNCHANGED (the
 *   production regression contract); the sub-4-bit family runs the
 *   SAME tile geometry with the width-parameterized blob walk: each
 *   thread's share of one N-step is always 16 k-PAIRS, packed into
 *   4*B bytes at the b=4-identical (n_local, k) positions — only the
 *   pair-field extraction and the LUT row width (2^B) differ.
 *
 * Design:
 *   - Canonical idx4 decoder (CS-1.1): each thread reads one 16 B segment
 *     (uint4) of the 2 048-byte half-tile per N-step; 128 threads x 16 B
 *     cover it bijectively, so every sW cell is written exactly once.
 *   - B-fragments (CS-1.2, Option A): sW staged row-major [n][k] with a
 *     16 B-block XOR-8 swizzle; fragments loaded with ldmatrix.x4.trans
 *     through A-style address math (ldmatrix_bT_addr). The naive
 *     ".trans + b_addr" combination is the K3 trap (executed cos 0.507)
 *     and is pinned by the G-B3b gate (FLUTE_BWD_TRAP build).
 *   - Geometry (unchanged): BM=128, BN=64, BK=64, 4 warps in a 2x2 grid,
 *     warp tile 64x32 (4 mma m-tiles x 4 n-tiles).
 *   - Pipeline (CS-1.3/1.4/1.5/1.11): dynamic shared memory with double
 *     buffering (sA 32 KB + sW 16 KB), cp.async sA staging with a
 *     zero-fill M-guard, blob prefetch through ld.global.nc.v4, one
 *     __syncthreads per N-step, vectorized 2-wide epilogue.
 *   - Grid (CS-1.6): x = k-blocks (fastest), y = m-blocks, so a
 *     scheduling wave covers all k-blocks of one m-strip and the grad_Y
 *     strip stays L2-resident.
 *   - Dtypes (CS-1.8): templated on the activation type (__half or
 *     __nv_bfloat16); the LUT stays fp16 (artifact contract) and is
 *     converted once per dequant gather. Dispatch on grad_y.dtype.
 *
 * Debug surfaces (test-only builds; production is unaffected):
 *   FLUTE_BWD_CANARY  pre-poison sW with the fp16 NaN pattern before each
 *                     dequant (G-B3): the decoder's coverage is an
 *     arithmetic identity, so surviving poison = decoder regression.
 *   FLUTE_BWD_TRAP    load B-fragments with the K3-trap combination
 *                     (.trans + b_addr). This build MUST fail G-B1 (G-B3b).
 *   backward_simple_twin host entry: scalar fragment assembly instead of
 *                     ldmatrix (G-B2) — bit-exact versus the production
 *                     path when ldmatrix delivers the documented layout.
 *
 * Target: NVIDIA A10G (SM_86, 80 SM, 600 GB/s, 62.5 TFLOPS FP32-acc).
 * Kernel-level target after repair: >= 40 TF at M >= 2048 (spec section 7).
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
// Format constants (DEQUANT_SPEC section 7)
// ---------------------------------------------------------------------------
constexpr int ROWS_PER_TILE = 128;
constexpr int K_PER_TILE    = 64;
constexpr int TILE_BYTES    = 4096;

// ---------------------------------------------------------------------------
// idxN sub-4-bit format constants (DEQUANT_SPEC section 8; the unified
// idxN layout family, flute_extended/flute_extended/idxN.py). The
// (128-row, 64-k) tile is 1024*B bytes; the 64-row half-tile an N-step
// consumes is 512*B; the (wx, lane) chunk 16*B; the 128 threads split
// each chunk into 4 segments of 4*B bytes = 16 k-PAIRS — the pair count
// per thread is width-independent, so the (n_local, k) walk of the
// canonical 4-bit decoder is preserved verbatim. At B=4 the numbers
// degenerate to the constants above (4096 / 2048 / 64 / 16).
// ---------------------------------------------------------------------------
template <int B>
struct SubB {
    static_assert(B == 1 || B == 2 || B == 3,
                  "SubB is the sub-4-bit backward extension; B=4 runs the "
                  "existing kernels above, unchanged");
    static constexpr int PAL        = 1 << B;   // LUT entries per group
    static constexpr int SEG_WORDS  = B;        // u32 words per segment
    static constexpr int SEG_BYTES  = 4 * B;    // 16 pairs * 2B bits / 8
    static constexpr int CHUNK_BY   = 16 * B;   // per (wx, lane) chunk
    static constexpr int HALF_BY    = 512 * B;  // per 64-row half-tile
    static constexpr int TILE_BY    = 1024 * B; // per (128-row, 64-k) tile
};

// ---------------------------------------------------------------------------
// Tile configuration: BM=128, BN=64, BK=64, 128 threads (4 warps, 2x2)
// ---------------------------------------------------------------------------
constexpr int BM       = 128;  // output M rows per block
constexpr int BN       = 64;   // contraction step (half an idx4 row-tile)
constexpr int BK       = 64;   // output K depth (one idx4 K-tile)
constexpr int THREADS  = 128;

constexpr int K_TILES  = BN / 16;             // 4 contraction mma k-slices
constexpr int WARP_M   = BM / 2;              // 64 (4 m-tiles per warp)
constexpr int WARP_N   = BK / 2;              // 32 (4 n-tiles per warp)
constexpr int WARP_M_TILES = WARP_M / 16;     // 4
constexpr int WARP_N_TILES = WARP_N / 8;      // 4

// 16 B-block XOR-8 swizzle on both shared tiles (CS-1.4).
constexpr int SA_XOR = 8;
constexpr int SW_XOR = 8;

// Dynamic shared memory: double-buffered sA[2][BM][BN] + sW[2][BN][BK]
// (2 B elements in both dtype instantiations): 32 KB + 16 KB = 48 KB,
// opt-in above the 48 KB default via cudaFuncSetAttribute (2 CTAs/SM).
constexpr int SA_BYTES   = 2 * BM * BN * 2;
constexpr int SW_BYTES   = 2 * BN * BK * 2;
constexpr int DYN_SMEM   = SA_BYTES + SW_BYTES;

// ---------------------------------------------------------------------------
// Small dtype helpers
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T lut_to(const __half h) {
    if constexpr (std::is_same<T, __half>::value) {
        return h;
    } else {
        return __float2bfloat16(__half2float(h));
    }
}

template <typename T>
__device__ __forceinline__ uint32_t pack_pair(T lo, T hi) {
    if constexpr (std::is_same<T, __half>::value) {
        __half2 h2;
        h2.x = lo; h2.y = hi;
        return *reinterpret_cast<const uint32_t*>(&h2);
    } else {
        __nv_bfloat162 b2;
        b2.x = lo; b2.y = hi;
        return *reinterpret_cast<const uint32_t*>(&b2);
    }
}

template <typename T>
__device__ __forceinline__ void store_pair(T* dst, float v0, float v1) {
    if constexpr (std::is_same<T, __half>::value) {
        __half2 h2;
        h2.x = __float2half(v0); h2.y = __float2half(v1);
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(&h2);
    } else {
        __nv_bfloat162 b2;
        b2.x = __float2bfloat16(v0); b2.y = __float2bfloat16(v1);
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(&b2);
    }
}

template <typename T>
__device__ __forceinline__ const __half* as_half(const T* p) {
    return reinterpret_cast<const __half*>(p);
}

// ---------------------------------------------------------------------------
// Blob addressing: this thread's 16 B segment for the N-step at n_base.
// ---------------------------------------------------------------------------
__device__ __forceinline__ size_t blob_segment_offset(
    int n_base, int k0, int K, int tid
) {
    const int t  = n_base / ROWS_PER_TILE;
    const int wx = (n_base % ROWS_PER_TILE) / 64;      // 0 or 1
    const int g  = k0 / K_PER_TILE;
    const size_t tile_base = ((size_t)t * (K / K_PER_TILE) + (size_t)g) * TILE_BYTES;
    return tile_base + (size_t)wx * 2048
         + (size_t)(tid >> 2) * 64 + (size_t)(tid & 3) * 16;
}

// ---------------------------------------------------------------------------
// idxN sub-4-bit blob addressing: this thread's 4*B-byte segment (16
// pairs) for the N-step at n_base — the b=4 arithmetic above with every
// byte count scaled by B/4. At B=4 it reduces to blob_segment_offset
// exactly (the pair positions are the width-independent contract).
// ---------------------------------------------------------------------------
template <int B>
__device__ __forceinline__ size_t blob_segment_offset_sub4(
    int n_base, int k0, int K, int tid
) {
    using SB = SubB<B>;
    const int t  = n_base / ROWS_PER_TILE;
    const int wx = (n_base % ROWS_PER_TILE) / 64;      // 0 or 1
    const int g  = k0 / K_PER_TILE;
    const size_t tile_base =
        ((size_t)t * (K / K_PER_TILE) + (size_t)g) * SB::TILE_BY;
    return tile_base + (size_t)wx * SB::HALF_BY
         + (size_t)(tid >> 2) * SB::CHUNK_BY
         + (size_t)(tid & 3) * SB::SEG_BYTES;
}

// Prefetch the segment as B u32 words (.nc scalar loads — the forward
// kernels' q_fd_prefetch_sub4 discipline: the 4/8/12-byte segments are
// 4 B- but never 16 B-aligned below B=4).
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
// segment (the last pair ends exactly at bit 32*B), so q[w0+1] is always
// in range — the same arithmetic as the forward's dequant_tile_sub4
// (flute_extended kernel_cutlass_streaming.cu, DEQUANT_SPEC section 8).
// ---------------------------------------------------------------------------
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
// sA staging: 16 B chunks, coalesced, M-guarded with zero-fill (CS-1.5).
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void stage_sA(
    T* sA, const T* __restrict__ grad_y,
    int m0, int n_base, int M, int N, int tid
) {
    constexpr int CHUNKS = BM * (BN / 8);   // 1024 x 16 B
    #pragma unroll 8
    for (int i = tid; i < CHUNKS; i += THREADS) {
        const int m_local = i >> 3;         // BN/8 = 8 chunks per row
        const int col8 = (i & 7) << 3;      // 8-half column block
        const bool in_bounds = (m0 + m_local) < M;
        const T* src = grad_y + (size_t)(m0 + m_local) * N + n_base + col8;
        T* dst = sA + (size_t)m_local * BN + flute::swz_col16(m_local, col8, SA_XOR);
        flute::cp_async_16_zfill(dst, src, in_bounds);
    }
}

// ---------------------------------------------------------------------------
// Canonical idx4 decoder (CS-1.1): one uint4 per thread covers the
// half-tile bijectively. Byte j of the segment maps to
//   n_local = v*16 + d*8 + (chunk >> 2),  k = 2*(seg*8 + (chunk & 3) + 4*s2)
// with v = j>>2, d = (j>>1)&1, s2 = j&1 — the inverse of idx4's
// _tile_permutation, executed bit-exactly (spec Appendix A, scenario D).
// Writes one u32 pair per byte at the swizzled word position.
// ---------------------------------------------------------------------------
template <typename T, int GS>
__device__ __forceinline__ void dequant_w_tile(
    const uint4& w16,
    int chunk, int seg,
    const __half* __restrict__ lut,
    int n_base,
    T* sW
) {
    uint32_t* sw_words = reinterpret_cast<uint32_t*>(sW);   // [BN][BK/2]
    const int lut_row0 = n_base / GS;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&w16);
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t b = bytes[j];
        const int v  = j >> 2;
        const int d  = (j >> 1) & 1;
        const int s2 = j & 1;
        const int n_local = v * 16 + d * 8 + (chunk >> 2);
        const int k = 2 * (seg * 8 + (chunk & 3) + 4 * s2);
        const int grp = n_local / GS;
        const __half lo = __ldg(&lut[(size_t)(lut_row0 + grp) * 16 + (b & 0x0F)]);
        const __half hi = __ldg(&lut[(size_t)(lut_row0 + grp) * 16 + (b >> 4)]);
        const int w = flute::swz_word(n_local, k >> 1, SW_XOR);
        sw_words[(size_t)n_local * (BK / 2) + w] =
            pack_pair<T>(lut_to<T>(lo), lut_to<T>(hi));
    }
}

// G-B3 canary: fill one sW buffer with the fp16 NaN bit pattern.
__device__ __forceinline__ void poison_sW(void* sW_buf, int tid) {
    uint32_t* words = reinterpret_cast<uint32_t*>(sW_buf);
    for (int i = tid; i < BN * BK / 2; i += THREADS) words[i] = 0x7FFF7FFFu;
}

// ---------------------------------------------------------------------------
// Canonical idxN sub-4-bit decoder: the segment's 16 pairs cover the
// same (n_local, k) walk as dequant_w_tile — pair j maps to
//   n_local = v*16 + d*8 + (chunk >> 2),  k = 2*(seg*8 + (chunk & 3) + 4*s2)
// (v = j>>2, d = (j>>1)&1, s2 = j&1) — the inverse of idxN's pair
// permutation at EVERY width (the pair positions are width-independent;
// only the field packing changes, DEQUANT_SPEC section 8). The LUT row
// is 2^B entries; sW, the swizzle, the mma phase and the epilogue are
// width-independent (the forward legacy-sub4 contract).
// ---------------------------------------------------------------------------
template <typename T, int B, int GS>
__device__ __forceinline__ void dequant_w_tile_sub4(
    const uint32_t (&q)[SubB<B>::SEG_WORDS],
    int chunk, int seg,
    const __half* __restrict__ lut,
    int n_base,
    T* sW
) {
    using SB = SubB<B>;
    uint32_t* sw_words = reinterpret_cast<uint32_t*>(sW);   // [BN][BK/2]
    const int lut_row0 = n_base / GS;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        uint32_t v0, v1;
        decode_pair_sub4<B>(q, j, v0, v1);
        const int v  = j >> 2;
        const int d  = (j >> 1) & 1;
        const int s2 = j & 1;
        const int n_local = v * 16 + d * 8 + (chunk >> 2);
        const int k = 2 * (seg * 8 + (chunk & 3) + 4 * s2);
        const int grp = n_local / GS;
        const __half lo =
            __ldg(&lut[(size_t)(lut_row0 + grp) * SB::PAL + v0]);
        const __half hi =
            __ldg(&lut[(size_t)(lut_row0 + grp) * SB::PAL + v1]);
        const int w = flute::swz_word(n_local, k >> 1, SW_XOR);
        sw_words[(size_t)n_local * (BK / 2) + w] =
            pack_pair<T>(lut_to<T>(lo), lut_to<T>(hi));
    }
}

// ---------------------------------------------------------------------------
// MMA phase: acc += sA[buf] @ sW[buf] for one N-step.
// ---------------------------------------------------------------------------
template <typename T, bool kTwin>
__device__ __forceinline__ void mma_phase(
    const T* sA, const T* sW,
    int lane, int wy, int wx,
    float (&acc)[WARP_M_TILES][WARP_N_TILES][4]
) {
    #pragma unroll
    for (int kt = 0; kt < K_TILES; ++kt) {
        // A-fragments: grad_Y rows (m) x contraction cols (n), non-trans.
        uint32_t afrag[WARP_M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < WARP_M_TILES; ++mt) {
            const int row_base = wy * WARP_M + mt * 16;
            const int col_base = kt * 16;
            if constexpr (kTwin) {
                flute::a_frag_scalar<T>(afrag[mt], sA, BN,
                                        row_base, col_base, lane, SA_XOR);
            } else {
                flute::ldmatrix_x4(
                    afrag[mt][0], afrag[mt][1], afrag[mt][2], afrag[mt][3],
                    flute::ldmatrix_a_addr(as_half(sA), BN, row_base,
                                           col_base, lane, SA_XOR));
            }
        }

        // B-fragments: contraction rows (n) x output cols (k) of sW —
        // .trans with A-style addresses (the K3 trap's negative pole).
        uint32_t bfrag[WARP_N_TILES][2];
        if constexpr (kTwin) {
            #pragma unroll
            for (int nt = 0; nt < WARP_N_TILES; ++nt) {
                flute::b_frag_scalar<T>(bfrag[nt], sW, BK,
                                        kt * 16, wx * WARP_N + nt * 8,
                                        lane, SW_XOR);
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < WARP_N_TILES; nt += 2) {
                uint32_t r0, r1, r2, r3;
                const int n_base_kt = kt * 16;                 // contraction n
                const int k_base = wx * WARP_N + nt * 8;       // output k
                const __half* addr = flute::ldmatrix_bT_addr(
                    as_half(sW), BK, n_base_kt, k_base, lane, SW_XOR);
#if defined(FLUTE_BWD_TRAP)
                // G-B3b (test build only): the K3 trap — .trans with the
                // b_addr formulas. This configuration must FAIL G-B1.
                addr = flute::ldmatrix_b_addr(
                    as_half(sW), BK, n_base_kt, k_base, lane, SW_XOR);
#endif
                flute::ldmatrix_x4_trans(r0, r1, r2, r3, addr);
                bfrag[nt][0] = r0;         bfrag[nt][1] = r1;
                bfrag[nt + 1][0] = r2;     bfrag[nt + 1][1] = r3;
            }
        }

        #pragma unroll
        for (int mt = 0; mt < WARP_M_TILES; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < WARP_N_TILES; ++nt) {
                if constexpr (std::is_same<T, __half>::value) {
                    flute::mma_m16n8k16_f32acc(afrag[mt], bfrag[nt], acc[mt][nt]);
                } else {
                    flute::mma_m16n8k16_f32acc_bf16(afrag[mt], bfrag[nt], acc[mt][nt]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The fused backward GEMM kernel
// ---------------------------------------------------------------------------
template <typename T, int GS, bool kTwin>
__global__ void __launch_bounds__(THREADS, 2)
fused_backward_gemm_kernel(
    const T* __restrict__ grad_y,      // [M, N]
    const uint8_t* __restrict__ blob,  // flat idx4 blob (N*K/2 bytes)
    const __half* __restrict__ lut,    // [N/GS, 16]
    T* __restrict__ grad_x,            // [M, K]
    int M, int N, int K
) {
    static_assert(GS == 16 || GS == 32 || GS == 64 || GS == 128 || GS == 256
                  || GS == 512,
                  "group_size must be 16, 32, 64, 128, 256 or 512");
    // W12 no-straddle note: the dequant walks N in BN=64 N-steps
    // (n_base is 64-aligned). Every supported GS is either a divisor of
    // 64 (16/32/64 — n_base is GS-aligned) or a multiple of 64
    // (128/256/512 — a 64-row interval never contains an interior
    // multiple of 128+), so lut_row0 + n_local/GS == (n_base+n_local)/GS
    // at every GS and the dequant walk is bit-identical to the pre-W12
    // arithmetic.

    // Grid: x = k-blocks (fastest), y = m-blocks (CS-1.6).
    const int m0 = blockIdx.y * BM;
    const int k0 = blockIdx.x * BK;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wy = warp_id >> 1;   // M half
    const int wx = warp_id & 1;    // K half

    extern __shared__ char smem_raw[];
    T* sA_all = reinterpret_cast<T*>(smem_raw);             // [2][BM][BN]
    T* sW_all = reinterpret_cast<T*>(smem_raw + SA_BYTES);  // [2][BN][BK]
    T* sA[2] = {sA_all, sA_all + (size_t)BM * BN};
    T* sW[2] = {sW_all, sW_all + (size_t)BN * BK};

    float acc[WARP_M_TILES][WARP_N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < WARP_M_TILES; ++mt)
        #pragma unroll
        for (int nt = 0; nt < WARP_N_TILES; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[mt][nt][i] = 0.0f;

    const int n_steps = N / BN;
    uint4 blob_regs;

    // Prologue: stage sA[0] (cp.async) and the step-0 blob segment, then
    // dequant sW[0]; one barrier publishes both.
    stage_sA<T>(sA[0], grad_y, m0, 0, M, N, tid);
    flute::cp_async_commit();
    flute::ldg_nc_evict_first_v4(
        blob_regs, blob + blob_segment_offset(0, k0, K, tid));
    flute::cp_async_wait_all();
#ifdef FLUTE_BWD_CANARY
    poison_sW(sW[0], tid);
    __syncthreads();   // debug build: separate poison pass before the dequant
#endif
    dequant_w_tile<T, GS>(blob_regs, tid >> 2, tid & 3, lut, 0, sW[0]);
    __syncthreads();

    // Steady state: one __syncthreads per N-step. Per iteration (buf):
    //   mma(sA[buf], sW[buf]) — both published by the previous barrier;
    //   then stage sA[buf^1] (cp.async, overlaps the mma), prefetch the
    //   next blob segment into registers, and dequant sW[buf^1] — the
    //   buffer's previous reader (mma of step-1) retired before this
    //   iteration's top barrier, and its next reader (mma of step+1) only
    //   runs after the trailing barrier below.
    for (int step = 0; step < n_steps; ++step) {
        const int buf = step & 1;
        const int n_base = step * BN;

        mma_phase<T, kTwin>(sA[buf], sW[buf], lane, wy, wx, acc);

        if (step + 1 < n_steps) {
            const int nb = n_base + BN;
            stage_sA<T>(sA[buf ^ 1], grad_y, m0, nb, M, N, tid);
            flute::cp_async_commit();
            flute::ldg_nc_evict_first_v4(
                blob_regs, blob + blob_segment_offset(nb, k0, K, tid));
#ifdef FLUTE_BWD_CANARY
            poison_sW(sW[buf ^ 1], tid);
            __syncthreads();   // debug build only
#endif
            dequant_w_tile<T, GS>(blob_regs, tid >> 2, tid & 3, lut, nb, sW[buf ^ 1]);
            flute::cp_async_wait_all();
        }
        __syncthreads();
    }

    // Epilogue: 2-wide stores; K % 64 == 0 makes the k guard always true,
    // so only the (ragged) M rows are guarded.
    #pragma unroll
    for (int mt = 0; mt < WARP_M_TILES; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < WARP_N_TILES; ++nt) {
            const int row_base = wy * WARP_M + mt * 16;
            const int col_base = wx * WARP_N + nt * 8;
            const int g = lane >> 2;
            const int c = 2 * (lane & 3);
            const int m_row = m0 + row_base + g;
            const int k_col = k0 + col_base + c;
            if (m_row < M)
                store_pair<T>(grad_x + (size_t)m_row * K + k_col,
                              acc[mt][nt][0], acc[mt][nt][1]);
            if (m_row + 8 < M)
                store_pair<T>(grad_x + (size_t)(m_row + 8) * K + k_col,
                              acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

// ---------------------------------------------------------------------------
// The fused backward GEMM kernel, idxN sub-4-bit family (B in {1,2,3}).
// The geometry, pipeline, mma phase, epilogue and debug surfaces are the
// 4-bit kernel's verbatim; only the segment prefetch (4*B bytes as B u32
// words) and the dequant (dequant_w_tile_sub4 over the 2^B-entry LUT)
// differ. FLUTE_BWD_CANARY / FLUTE_BWD_TRAP / the scalar twin apply
// unchanged (the sW layout and the fragment path are width-independent).
// ---------------------------------------------------------------------------
template <typename T, int B, int GS, bool kTwin>
__global__ void __launch_bounds__(THREADS, 2)
fused_backward_gemm_sub4_kernel(
    const T* __restrict__ grad_y,      // [M, N]
    const uint8_t* __restrict__ blob,  // flat idxN blob (N*K*B/8 bytes)
    const __half* __restrict__ lut,    // [N/GS, 2^B]
    T* __restrict__ grad_x,            // [M, K]
    int M, int N, int K
) {
    using SB = SubB<B>;
    static_assert(GS == 16 || GS == 32 || GS == 64 || GS == 128 || GS == 256
                  || GS == 512,
                  "group_size must be 16, 32, 64, 128, 256 or 512");
    // W12 no-straddle note: the dequant walks N in BN=64 N-steps
    // (n_base is 64-aligned). Every supported GS is either a divisor of
    // 64 (16/32/64 — n_base is GS-aligned) or a multiple of 64
    // (128/256/512 — a 64-row interval never contains an interior
    // multiple of 128+), so lut_row0 + n_local/GS == (n_base+n_local)/GS
    // at every GS and the dequant walk is bit-identical to the pre-W12
    // arithmetic.

    // Grid: x = k-blocks (fastest), y = m-blocks (CS-1.6).
    const int m0 = blockIdx.y * BM;
    const int k0 = blockIdx.x * BK;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wy = warp_id >> 1;   // M half
    const int wx = warp_id & 1;    // K half

    extern __shared__ char smem_raw[];
    T* sA_all = reinterpret_cast<T*>(smem_raw);             // [2][BM][BN]
    T* sW_all = reinterpret_cast<T*>(smem_raw + SA_BYTES);  // [2][BN][BK]
    T* sA[2] = {sA_all, sA_all + (size_t)BM * BN};
    T* sW[2] = {sW_all, sW_all + (size_t)BN * BK};

    float acc[WARP_M_TILES][WARP_N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < WARP_M_TILES; ++mt)
        #pragma unroll
        for (int nt = 0; nt < WARP_N_TILES; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[mt][nt][i] = 0.0f;

    const int n_steps = N / BN;
    uint32_t q[SB::SEG_WORDS];

    // Prologue: stage sA[0] (cp.async) and the step-0 blob segment, then
    // dequant sW[0]; one barrier publishes both.
    stage_sA<T>(sA[0], grad_y, m0, 0, M, N, tid);
    flute::cp_async_commit();
    q_seg_load_sub4<B>(
        blob + blob_segment_offset_sub4<B>(0, k0, K, tid), q);
    flute::cp_async_wait_all();
#ifdef FLUTE_BWD_CANARY
    poison_sW(sW[0], tid);
    __syncthreads();   // debug build: separate poison pass before the dequant
#endif
    dequant_w_tile_sub4<T, B, GS>(q, tid >> 2, tid & 3, lut, 0, sW[0]);
    __syncthreads();

    // Steady state: one __syncthreads per N-step (the 4-bit pipeline,
    // verbatim — see the kernel above for the buffer-ownership argument).
    for (int step = 0; step < n_steps; ++step) {
        const int buf = step & 1;
        const int n_base = step * BN;

        mma_phase<T, kTwin>(sA[buf], sW[buf], lane, wy, wx, acc);

        if (step + 1 < n_steps) {
            const int nb = n_base + BN;
            stage_sA<T>(sA[buf ^ 1], grad_y, m0, nb, M, N, tid);
            flute::cp_async_commit();
            q_seg_load_sub4<B>(
                blob + blob_segment_offset_sub4<B>(nb, k0, K, tid), q);
#ifdef FLUTE_BWD_CANARY
            poison_sW(sW[buf ^ 1], tid);
            __syncthreads();   // debug build only
#endif
            dequant_w_tile_sub4<T, B, GS>(q, tid >> 2, tid & 3, lut, nb,
                                          sW[buf ^ 1]);
            flute::cp_async_wait_all();
        }
        __syncthreads();
    }

    // Epilogue: 2-wide stores; K % 64 == 0 makes the k guard always true,
    // so only the (ragged) M rows are guarded.
    #pragma unroll
    for (int mt = 0; mt < WARP_M_TILES; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < WARP_N_TILES; ++nt) {
            const int row_base = wy * WARP_M + mt * 16;
            const int col_base = wx * WARP_N + nt * 8;
            const int g = lane >> 2;
            const int c = 2 * (lane & 3);
            const int m_row = m0 + row_base + g;
            const int k_col = k0 + col_base + c;
            if (m_row < M)
                store_pair<T>(grad_x + (size_t)m_row * K + k_col,
                              acc[mt][nt][0], acc[mt][nt][1]);
            if (m_row + 8 < M)
                store_pair<T>(grad_x + (size_t)(m_row + 8) * K + k_col,
                              acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher (CS-1.7): full argument validation, transparent alignment
// clones, dynamic-smem opt-in (once per instantiation), launch check.
// ---------------------------------------------------------------------------
namespace {

template <typename T, int GS, bool kTwin>
void launch_backward(
    torch::Tensor grad_y, torch::Tensor indices, torch::Tensor lut,
    torch::Tensor grad_x, int M, int N, int K
) {
    static const bool smem_attr_set = [] {
        const cudaError_t err = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(&fused_backward_gemm_kernel<T, GS, kTwin>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaFuncSetAttribute(smem ", DYN_SMEM, ") failed: ",
                    cudaGetErrorString(err));
        return true;
    }();
    (void)smem_attr_set;

    dim3 grid((unsigned)(K / BK), (unsigned)((M + BM - 1) / BM));
    dim3 block(THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();

    fused_backward_gemm_kernel<T, GS, kTwin>
        <<<grid, block, DYN_SMEM, stream>>>(
            reinterpret_cast<const T*>(grad_y.data_ptr()),
            indices.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(lut.data_ptr()),
            reinterpret_cast<T*>(grad_x.data_ptr()),
            M, N, K);

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_backward_gemm launch failed: ", cudaGetErrorString(err),
                " (grid ", grid.x, "x", grid.y, ", smem ", DYN_SMEM, " B)");
}

// idxN sub-4-bit launcher (CS-1.7's pattern on the sub4 kernel symbol).
template <typename T, int B, int GS, bool kTwin>
void launch_backward_sub4(
    torch::Tensor grad_y, torch::Tensor indices, torch::Tensor lut,
    torch::Tensor grad_x, int M, int N, int K
) {
    static const bool smem_attr_set = [] {
        const cudaError_t err = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(
                &fused_backward_gemm_sub4_kernel<T, B, GS, kTwin>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaFuncSetAttribute(smem ", DYN_SMEM, ") failed: ",
                    cudaGetErrorString(err));
        return true;
    }();
    (void)smem_attr_set;

    dim3 grid((unsigned)(K / BK), (unsigned)((M + BM - 1) / BM));
    dim3 block(THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();

    fused_backward_gemm_sub4_kernel<T, B, GS, kTwin>
        <<<grid, block, DYN_SMEM, stream>>>(
            reinterpret_cast<const T*>(grad_y.data_ptr()),
            indices.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(lut.data_ptr()),
            reinterpret_cast<T*>(grad_x.data_ptr()),
            M, N, K);

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_backward_gemm_sub4(b=", B, ") launch failed: ",
                cudaGetErrorString(err), " (grid ", grid.x, "x", grid.y,
                ", smem ", DYN_SMEM, " B)");
}

// W12 GS dispatch helper: the runtime group_size -> template GS switch
// (all six values are valid for the BN=64 N-step geometry — see the
// kernels' no-straddle note). The default arm is a hard TORCH_CHECK, so
// an unsupported GS can never fall through to a wrong instantiation.
template <typename T, int B, bool kTwin>
void backward_sub4_gs_dispatch(
    torch::Tensor grad_y_a, torch::Tensor indices_a, torch::Tensor lut_a,
    torch::Tensor grad_x, int M, int N, int K, int64_t group_size
) {
    switch ((int)group_size) {
        case 16:
            launch_backward_sub4<T, B, 16, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 32:
            launch_backward_sub4<T, B, 32, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 64:
            launch_backward_sub4<T, B, 64, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 128:
            launch_backward_sub4<T, B, 128, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 256:
            launch_backward_sub4<T, B, 256, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 512:
            launch_backward_sub4<T, B, 512, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        default:
            TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512, "
                              "got ", group_size);
    }
}

// W12 GS dispatch helper, 4-bit tree.
template <typename T, bool kTwin>
void backward_b4_gs_dispatch(
    torch::Tensor grad_y_a, torch::Tensor indices_a, torch::Tensor lut_a,
    torch::Tensor grad_x, int M, int N, int K, int64_t group_size
) {
    switch ((int)group_size) {
        case 16:
            launch_backward<T, 16, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 32:
            launch_backward<T, 32, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 64:
            launch_backward<T, 64, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 128:
            launch_backward<T, 128, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 256:
            launch_backward<T, 256, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        case 512:
            launch_backward<T, 512, kTwin>(
                grad_y_a, indices_a, lut_a, grad_x, M, N, K);
            break;
        default:
            TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512, "
                              "got ", group_size);
    }
}

// idxN sub-4-bit dtype/GS dispatch (the 4-bit tree's shape, B bound by
// the caller's switch — the forward kernels' dispatch style).
template <int B>
torch::Tensor backward_gemm_sub4_impl(
    torch::Tensor grad_y_a, torch::Tensor indices_a, torch::Tensor lut_a,
    torch::Tensor grad_x, bool fp16, bool twin, int64_t M, int64_t N,
    int64_t K, int64_t group_size
) {
    if (twin) {
        if (fp16)
            backward_sub4_gs_dispatch<__half, B, true>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K,
                group_size);
        else
            backward_sub4_gs_dispatch<__nv_bfloat16, B, true>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K,
                group_size);
    } else {
        if (fp16)
            backward_sub4_gs_dispatch<__half, B, false>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K,
                group_size);
        else
            backward_sub4_gs_dispatch<__nv_bfloat16, B, false>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K,
                group_size);
    }
    return grad_x;
}

torch::Tensor backward_gemm_impl(
    torch::Tensor grad_y, torch::Tensor indices, torch::Tensor lut,
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
    TORCH_CHECK(grad_y.is_cuda() && indices.is_cuda() && lut.is_cuda(),
                "all tensors must be CUDA");
    const bool fp16 = grad_y.scalar_type() == torch::kFloat16;
    const bool bf16 = grad_y.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(fp16 || bf16, "grad_y must be float16 or bfloat16");
    TORCH_CHECK(indices.scalar_type() == torch::kUInt8, "indices must be uint8");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat16,
                "lut must be float16 (artifact contract)");
    TORCH_CHECK(grad_y.dim() == 2, "grad_y must be 2-D [M, N]");
    TORCH_CHECK(group_size == 16 || group_size == 32 || group_size == 64 ||
                group_size == 128 || group_size == 256 || group_size == 512,
                "group_size must be one of 16/32/64/128/256/512 "
                "(W12 full-GS-range kernels)");

    const int64_t M = grad_y.size(0);
    const int64_t N = grad_y.size(1);
    TORCH_CHECK(N == N_, "grad_y N mismatch");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    TORCH_CHECK(K_ % 64 == 0, "K must be a multiple of 64");
    TORCH_CHECK(M > 0, "M must be positive");
    const int64_t expected_bytes = N * K_ * bitwidth / 8;
    TORCH_CHECK(indices.numel() == expected_bytes,
                "idx", bitwidth, " indices must hold N*K*", bitwidth,
                "/8 = ", expected_bytes, " bytes, got ", indices.numel());
    TORCH_CHECK(lut.dim() == 2 && lut.size(1) == (1 << bitwidth)
                && lut.size(0) == (N + group_size - 1) / group_size,
                "lut must be [ceil(N/group_size), 2^bitwidth=",
                (1 << bitwidth), "]");
    TORCH_CHECK(grad_y.is_contiguous() && indices.is_contiguous()
                && lut.is_contiguous(),
                "grad_y, indices and lut must be contiguous");

    // 16 B alignment for cp.async / ld.global.nc; clone to repair (the
    // sub-4 segments load u32 words — 4 B suffices, 16 B is a superset).
    auto grad_y_a = (reinterpret_cast<uintptr_t>(grad_y.data_ptr()) & 15) == 0
                    ? grad_y : grad_y.clone();
    auto indices_a = (reinterpret_cast<uintptr_t>(indices.data_ptr()) & 15) == 0
                     ? indices : indices.clone();
    auto lut_a = (reinterpret_cast<uintptr_t>(lut.data_ptr()) & 15) == 0
                 ? lut : lut.clone();

    auto grad_x = torch::empty({M, K_}, grad_y.options());
    if (bitwidth < 4) {
        // idxN sub-4-bit dispatch (the forward kernels' switch style)
        switch (bitwidth) {
            case 1: return backward_gemm_sub4_impl<1>(
                        grad_y_a, indices_a, lut_a, grad_x, fp16, twin, M, N,
                        K_, group_size);
            case 2: return backward_gemm_sub4_impl<2>(
                        grad_y_a, indices_a, lut_a, grad_x, fp16, twin, M, N,
                        K_, group_size);
            case 3: return backward_gemm_sub4_impl<3>(
                        grad_y_a, indices_a, lut_a, grad_x, fp16, twin, M, N,
                        K_, group_size);
        }
        TORCH_CHECK(false, "unreachable bitwidth ", bitwidth);
    }
    if (twin) {
        if (fp16)
            backward_b4_gs_dispatch<__half, true>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K_,
                group_size);
        else
            backward_b4_gs_dispatch<__nv_bfloat16, true>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K_,
                group_size);
    } else {
        if (fp16)
            backward_b4_gs_dispatch<__half, false>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K_,
                group_size);
        else
            backward_b4_gs_dispatch<__nv_bfloat16, false>(
                grad_y_a, indices_a, lut_a, grad_x, (int)M, (int)N, (int)K_,
                group_size);
    }
    return grad_x;
}

}  // namespace

torch::Tensor fused_backward_gemm(
    torch::Tensor grad_y, torch::Tensor indices, torch::Tensor lut,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout
) {
    return backward_gemm_impl(grad_y, indices, lut, N, K, group_size,
                              /*twin=*/false, bitwidth, indices_layout);
}

torch::Tensor backward_simple_twin(
    torch::Tensor grad_y, torch::Tensor indices, torch::Tensor lut,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout
) {
    return backward_gemm_impl(grad_y, indices, lut, N, K, group_size,
                              /*twin=*/true, bitwidth, indices_layout);
}
