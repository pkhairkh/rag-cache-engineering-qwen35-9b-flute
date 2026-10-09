/**
 * src/kernel_debug_simple.cu
 *
 * Differential-testing twin of kernel_streaming.cu: the same GEMM,
 * the same mma.sync.m16n8k16 (f32 accumulate) instruction, the same
 * fragment mapping and the same epilogue — with deliberately simple
 * staging:
 *
 *   - single-buffered sA / sW, no software pipeline, no prefetch
 *   - padded rows (BK+8 halves), no XOR swizzle, no ldmatrix: fragments
 *     are assembled with 32-bit shared-memory reads
 *   - static __shared__ (<= 36,960 B; no cudaFuncSetAttribute needed)
 *   - no 16-byte input alignment requirement (scalar global loads)
 *
 * The two tensor-core kernels share only the fragment mapping and the MMA
 * wrapper, so they form a strict differential pair:
 *
 *   naive vs debug_simple     isolates the fragment mapping + mma semantics
 *   debug_simple vs streaming isolates ldmatrix / swizzle / cp.async staging
 *
 * Both kernels issue the same mma instructions over the same FP16 values in
 * the same per-element accumulation order, so their outputs must agree
 * bit-exactly (torch.equal). This also holds across the tile-geometry
 * difference: for BK=64 the production kernel repartitions to BM=64 while
 * this twin stays at BM=128 (and for the gs=32 deep-tile A/B it runs BK=32
 * against the production BK=64) — tile boundaries never reorder a row's
 * K-accumulation, so the differential gate re-proves the repartition too.
 *
 * Not performance-tuned (~15-40 TFLOPS). Do not use in production.
 *
 * Shared memory per block (static):
 *   BK=32: 2 * 128 * 40 * 2 B (sA+sW) + 5 * 16 * 2 B (sLUT) = 20,640 B
 *   BK=64: 2 * 128 * 72 * 2 B (sA+sW) + 3 * 16 * 2 B (sLUT) = 36,960 B
 *
 * Bank conflicts (padded stride SK = BK+8 halves):
 *   BK=32 -> 40 halves = 20 words; row r starts at word 20r -> banks
 *     {0,20,8,28,16,4,24,12} + {0..3} -> all 32 banks once (conflict-free)
 *   BK=64 -> 72 halves = 36 words; row r starts at word 36r -> 4r mod 32
 *     -> rows map to consecutive 4-word (16 B) bank groups (conflict-free)
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "flute/mma.cuh"
#include "flute/dequant.cuh"

namespace {

// ---------------------------------------------------------------------------
// Tile configuration (mirrors the production kernel's geometry)
// ---------------------------------------------------------------------------
template <int BK_, int GS_>
struct SimpleConfig {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = BK_;                  // K-tile depth (32/64)
    static constexpr int GS = GS_;                  // LUT group size (16..512, decoupled from BK)

    static constexpr int WM = 64;
    static constexpr int WN = 64;
    static constexpr int WARPS_M = BM / WM;         // 2
    static constexpr int WARPS_N = BN / WN;         // 2
    static constexpr int WARPS  = WARPS_M * WARPS_N;// 4
    static constexpr int THREADS = WARPS * 32;      // 128

    static constexpr int M_TILES = WM / 16;         // 4
    static constexpr int N_TILES = WN / 8;          // 8
    static constexpr int K_TILES = BK / 16;         // 2 (BK=32) or 4 (BK=64)

    // LUT groups from GS (not BK) — GS <= BN: BN/GS + 1; GS > BN: the
    // 128-row N-tile straddles at most one group boundary -> 2.
    static constexpr int LUT_GRPS = ((BN + GS - 1) / GS) + 1;

    // Padded row stride: +8 halves kills bank conflicts, keeps rows 16 B
    // aligned (SK even => every row start and every even-column __half2
    // access is 4-byte aligned).
    static constexpr int SK = BK + 8;
};

template <int GS>
struct debug_gs_supported : std::bool_constant<
    GS == 16 || GS == 32 || GS == 64 || GS == 128 || GS == 256 || GS == 512> {};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <typename Cfg>
__global__ void flute_kernel_debug_simple(const __half*    __restrict__ A,    // [M, K] row-major
    const uint8_t*   __restrict__ Q,    // [N, (K+1)/2] packed 4-bit
    const __half*    __restrict__ LUT,  // [ceil(N/GS), 16]
    __half*          __restrict__ C,    // [M, N] row-major
    int M, int N, int K
) {
    static_assert(Cfg::BK == 32 || Cfg::BK == 64, "BK must be 32 or 64");
    static_assert(debug_gs_supported<Cfg::GS>::value,
                  "GS must be 16, 32, 64, 128, 256 or 512");
    static_assert(Cfg::GS > Cfg::BN || Cfg::BN % Cfg::GS == 0,
                  "GS <= BN requires BN % GS == 0");
    static_assert(Cfg::WARPS == 4, "expected 4 warps (2x2)");
    static_assert(Cfg::SK % 2 == 0, "padded stride must be even (4 B alignment)");

    // ---- static shared memory: sA | sW | sLUT --------------------------------
    __shared__ __align__(16) __half sA[Cfg::BM][Cfg::SK];
    __shared__ __align__(16) __half sW[Cfg::BN][Cfg::SK];
    __shared__ __half sLUT[Cfg::LUT_GRPS][16];

    const int n0 = blockIdx.x * Cfg::BN;
    const int m0 = blockIdx.y * Cfg::BM;
    const int row_stride = (K + 1) >> 1;          // packed bytes per W row

    const int tid     = threadIdx.x;              // 0..127
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int wy      = warp_id / Cfg::WARPS_N;   // 0..1
    const int wx      = warp_id % Cfg::WARPS_N;   // 0..1

    // ---- Step 0: cache the (clamped) LUT rows -------------------------------
    // Staged once: the LUT is indexed by N group only. The group range is
    // clamped to ceil(N/GS)-1 (the last block's N-range may cross N).
    const int grp_first  = n0 / Cfg::GS;
    const int total_grps = (N + Cfg::GS - 1) / Cfg::GS;
    const int grp_last_r = (n0 + Cfg::BN - 1) / Cfg::GS;
    const int grp_last   = (grp_last_r < total_grps - 1) ? grp_last_r : (total_grps - 1);
    const int n_grps     = grp_last - grp_first + 1;

    for (int i = tid; i < Cfg::LUT_GRPS * 16; i += Cfg::THREADS) {
        sLUT[i / 16][i % 16] = __float2half(0.0f);   // zero-fill (safety)
    }
    for (int i = tid; i < n_grps * 16; i += Cfg::THREADS) {
        const int g   = i >> 4;
        const int idx = i & 15;
        sLUT[g][idx] = LUT[(size_t)(grp_first + g) * 16 + idx];
    }

    // ---- Per-thread dequant assignment: thread t owns W row n_local = t -----
    const int drow     = tid;                    // 0..BN-1
    const int drow_abs = n0 + drow;
    const bool drow_ok = drow_abs < N;
    // group row from the ABSOLUTE row (== drow/GS when n0 is
    // GS-aligned; handles the GS > BN straddle).
    const int g_lut    = (n0 + drow) / Cfg::GS - grp_first;
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

    __syncthreads();                 // LUT staged

    // ---- Main K loop: stage -> barrier -> mma -> barrier --------------------
    for (int k0 = 0; k0 < K; k0 += Cfg::BK) {
        // -- stage the A tile (scalar, zero-padded past M) --------------------
        for (int e = tid; e < Cfg::BM * Cfg::BK; e += Cfg::THREADS) {
            const int m_local = e / Cfg::BK;
            const int k_local = e % Cfg::BK;
            const int m = m0 + m_local;
            sA[m_local][k_local] =
                (m < M) ? A[(size_t)m * K + k0 + k_local] : __float2half(0.0f);
        }

        // -- dequantize this K-tile of W into sW (one row per thread) ----------
        // W[n, k] = LUT[n/BK, idx], idx packed LSB-first:
        //   even k -> low nibble, odd k -> high nibble of byte (k/2).
        if (drow_ok) {
            const uint8_t* src = q_row + (k0 >> 1);
            #pragma unroll 4
            for (int i = 0; i < Cfg::BK / 2; ++i) {
                const uint8_t byte = src[i];
                const __half lo = sLUT[g_lut][byte & 0x0F];
                const __half hi = sLUT[g_lut][(byte >> 4) & 0x0F];
                *reinterpret_cast<__half2*>(&sW[drow][2 * i]) =
                    __halves2half2(lo, hi);
            }
        } else {
            #pragma unroll 4
            for (int i = 0; i < Cfg::BK / 2; ++i) {
                *reinterpret_cast<__half2*>(&sW[drow][2 * i]) =
                    __halves2half2(__float2half(0.0f), __float2half(0.0f));
            }
        }

        __syncthreads();             // sA + sW ready

        // -- tensor-core phase: scalar fragment loads + mma --------------------
        // PTX ISA 9.4 fragment layout, lane t (g = t/4, c = 2*(t%4)):
        //   A: a0={A[g][c],A[g+1]}... (see mma.cuh header for the full map)
        //   B: b0={sW[n+g][k+c], sW[n+g][k+c+1]}, b1={... k+c+8, k+c+9}
        //   (K-pairs at fixed n = g — the same (row=g, col=c) pattern as A).
        #pragma unroll
        for (int kt = 0; kt < Cfg::K_TILES; ++kt) {
            const int k_base = kt * 16;
            #pragma unroll
            for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
                const int a_row = wy * Cfg::WM + mt * 16;
                #pragma unroll
                for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
                    const int b_row = wx * Cfg::WN + nt * 8;
                    __half a_frag[8];
                    __half b_frag[4];
                    // 32-bit fragment loads: each __half2 read fetches
                    // one packed register's worth of elements.
                    const __half2 a01 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4][k_base + (lane % 4) * 2]);
                    const __half2 a23 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4 + 8][k_base + (lane % 4) * 2]);
                    const __half2 a45 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4][k_base + (lane % 4) * 2 + 8]);
                    const __half2 a67 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4 + 8][k_base + (lane % 4) * 2 + 8]);
                    const __half2 b01 = *reinterpret_cast<const __half2*>(
                        &sW[b_row + lane / 4][k_base + (lane % 4) * 2]);
                    const __half2 b23 = *reinterpret_cast<const __half2*>(
                        &sW[b_row + lane / 4][k_base + (lane % 4) * 2 + 8]);
                    a_frag[0] = a01.x; a_frag[1] = a01.y;
                    a_frag[2] = a23.x; a_frag[3] = a23.y;
                    a_frag[4] = a45.x; a_frag[5] = a45.y;
                    a_frag[6] = a67.x; a_frag[7] = a67.y;
                    b_frag[0] = b01.x; b_frag[1] = b01.y;
                    b_frag[2] = b23.x; b_frag[3] = b23.y;
                    flute::mma_m16n8k16_f32acc_half(a_frag, b_frag,
                                                    acc[mt][nt]);
                }
            }
        }

        __syncthreads();             // done reading sA/sW; next tile may stage
    }

    // ---- Epilogue: same fragment -> global mapping as the production path
    // C-fragment (PTX ISA 9.4): c0,c1 = C[g][2t'], C[g][2t'+1];
    //                          c2,c3 = C[g+8][2t'], C[g+8][2t'+1]
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
            const int n_c1 = n_base + c + 1;

            if (m_r0 < M && n_c0 < N) C[(size_t)m_r0 * N + n_c0] = __float2half(acc[mt][nt][0]);
            if (m_r0 < M && n_c1 < N) C[(size_t)m_r0 * N + n_c1] = __float2half(acc[mt][nt][1]);
            if (m_r1 < M && n_c0 < N) C[(size_t)m_r1 * N + n_c0] = __float2half(acc[mt][nt][2]);
            if (m_r1 < M && n_c1 < N) C[(size_t)m_r1 * N + n_c1] = __float2half(acc[mt][nt][3]);
        }
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Sub-4-bit differential twin (idxN family, B in {1,2,3}): the same
// deliberately-simple staging and the same fragment/mma mapping as the
// 4-bit kernel above, with the LSB-first sub-byte pair decode of
// flute/dequant.cuh (decode_pair_b1/b2/b3). The 4-bit kernel above stays
// untouched; this twin keeps the
//   naive vs debug_simple vs streaming
// differential chain alive at every width.
// ---------------------------------------------------------------------------
namespace {

template <typename Cfg, int B>
__global__ void flute_kernel_debug_simple_sub4(const __half*    __restrict__ A,    // [M, K] row-major
    const uint8_t*   __restrict__ Q,    // [N, K*B/8] packed b-bit, LSB-first
    const __half*    __restrict__ LUT,  // [ceil(N/GS), 2^B]
    __half*          __restrict__ C,    // [M, N] row-major
    int M, int N, int K
) {
    static_assert(B == 1 || B == 2 || B == 3, "sub-4-bit twin");
    static_assert(Cfg::BK == 32 || Cfg::BK == 64, "BK must be 32 or 64");
    static_assert(debug_gs_supported<Cfg::GS>::value,
                  "GS must be 16, 32, 64, 128, 256 or 512");
    static_assert(Cfg::GS > Cfg::BN || Cfg::BN % Cfg::GS == 0,
                  "GS <= BN requires BN % GS == 0");
    static_assert(Cfg::WARPS == 4, "expected 4 warps (2x2)");
    static_assert(Cfg::SK % 2 == 0, "padded stride must be even (4 B alignment)");

    constexpr int PAL = 1 << B;                    // LUT entries per group

    __shared__ __align__(16) __half sA[Cfg::BM][Cfg::SK];
    __shared__ __align__(16) __half sW[Cfg::BN][Cfg::SK];
    __shared__ __align__(16) __half sLUT[Cfg::LUT_GRPS][PAL];

    const int n0 = blockIdx.x * Cfg::BN;
    const int m0 = blockIdx.y * Cfg::BM;
    const int row_stride = (K * B) >> 3;          // packed bytes per W row

    const int tid     = threadIdx.x;              // 0..127
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int wy      = warp_id / Cfg::WARPS_N;   // 0..1
    const int wx      = warp_id % Cfg::WARPS_N;   // 0..1

    // ---- Step 0: cache the (clamped) LUT rows -------------------------------
    const int grp_first  = n0 / Cfg::GS;
    const int total_grps = (N + Cfg::GS - 1) / Cfg::GS;
    const int grp_last_r = (n0 + Cfg::BN - 1) / Cfg::GS;
    const int grp_last   = (grp_last_r < total_grps - 1) ? grp_last_r : (total_grps - 1);
    const int n_grps     = grp_last - grp_first + 1;

    for (int i = tid; i < Cfg::LUT_GRPS * PAL; i += Cfg::THREADS) {
        sLUT[i / PAL][i % PAL] = __float2half(0.0f);   // zero-fill (safety)
    }
    for (int i = tid; i < n_grps * PAL; i += Cfg::THREADS) {
        const int g   = i / PAL;
        const int idx = i % PAL;
        sLUT[g][idx] = LUT[(size_t)(grp_first + g) * PAL + idx];
    }

    // ---- Per-thread dequant assignment: thread t owns W row n_local = t -----
    const int drow     = tid;                    // 0..BN-1
    const int drow_abs = n0 + drow;
    const bool drow_ok = drow_abs < N;
    // absolute-row group (see the 4-bit debug kernel above).
    const int g_lut    = (n0 + drow) / Cfg::GS - grp_first;
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

    __syncthreads();                 // LUT staged

    // ---- Main K loop: stage -> barrier -> mma -> barrier --------------------
    for (int k0 = 0; k0 < K; k0 += Cfg::BK) {
        // -- stage the A tile (scalar, zero-padded past M) --------------------
        for (int e = tid; e < Cfg::BM * Cfg::BK; e += Cfg::THREADS) {
            const int m_local = e / Cfg::BK;
            const int k_local = e % Cfg::BK;
            const int m = m0 + m_local;
            sA[m_local][k_local] =
                (m < M) ? A[(size_t)m * K + k0 + k_local] : __float2half(0.0f);
        }

        // -- dequantize this K-tile of W into sW (one row per thread) ----------
        // W[n, k] = LUT[n/BK, idx]; pair i covers k = k0 + 2*i and its 2*B
        // bits sit at bit offset 2*B*i of the (byte-aligned) tile start.
        if (drow_ok) {
            const uint8_t* src = q_row + ((k0 * B) >> 3);
            #pragma unroll 4
            for (int i = 0; i < Cfg::BK / 2; ++i) {
                uint8_t v0, v1;
                if constexpr (B == 1) flute::decode_pair_b1(src, 2 * i, v0, v1);
                else if constexpr (B == 2) flute::decode_pair_b2(src, 2 * i, v0, v1);
                else flute::decode_pair_b3(src, 2 * i, v0, v1);
                const __half lo = sLUT[g_lut][v0];
                const __half hi = sLUT[g_lut][v1];
                *reinterpret_cast<__half2*>(&sW[drow][2 * i]) =
                    __halves2half2(lo, hi);
            }
        } else {
            #pragma unroll 4
            for (int i = 0; i < Cfg::BK / 2; ++i) {
                *reinterpret_cast<__half2*>(&sW[drow][2 * i]) =
                    __halves2half2(__float2half(0.0f), __float2half(0.0f));
            }
        }

        __syncthreads();             // sA + sW ready

        // -- tensor-core phase: scalar fragment loads + mma --------------------
        #pragma unroll
        for (int kt = 0; kt < Cfg::K_TILES; ++kt) {
            const int k_base = kt * 16;
            #pragma unroll
            for (int mt = 0; mt < Cfg::M_TILES; ++mt) {
                const int a_row = wy * Cfg::WM + mt * 16;
                #pragma unroll
                for (int nt = 0; nt < Cfg::N_TILES; ++nt) {
                    const int b_row = wx * Cfg::WN + nt * 8;
                    __half a_frag[8];
                    __half b_frag[4];
                    const __half2 a01 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4][k_base + (lane % 4) * 2]);
                    const __half2 a23 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4 + 8][k_base + (lane % 4) * 2]);
                    const __half2 a45 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4][k_base + (lane % 4) * 2 + 8]);
                    const __half2 a67 = *reinterpret_cast<const __half2*>(
                        &sA[a_row + lane / 4 + 8][k_base + (lane % 4) * 2 + 8]);
                    const __half2 b01 = *reinterpret_cast<const __half2*>(
                        &sW[b_row + lane / 4][k_base + (lane % 4) * 2]);
                    const __half2 b23 = *reinterpret_cast<const __half2*>(
                        &sW[b_row + lane / 4][k_base + (lane % 4) * 2 + 8]);
                    a_frag[0] = a01.x; a_frag[1] = a01.y;
                    a_frag[2] = a23.x; a_frag[3] = a23.y;
                    a_frag[4] = a45.x; a_frag[5] = a45.y;
                    a_frag[6] = a67.x; a_frag[7] = a67.y;
                    b_frag[0] = b01.x; b_frag[1] = b01.y;
                    b_frag[2] = b23.x; b_frag[3] = b23.y;
                    flute::mma_m16n8k16_f32acc_half(a_frag, b_frag,
                                                    acc[mt][nt]);
                }
            }
        }

        __syncthreads();             // done reading sA/sW; next tile may stage
    }

    // ---- Epilogue: same fragment -> global mapping as the production path
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
            const int n_c1 = n_base + c + 1;

            if (m_r0 < M && n_c0 < N) C[(size_t)m_r0 * N + n_c0] = __float2half(acc[mt][nt][0]);
            if (m_r0 < M && n_c1 < N) C[(size_t)m_r0 * N + n_c1] = __float2half(acc[mt][nt][1]);
            if (m_r1 < M && n_c0 < N) C[(size_t)m_r1 * N + n_c0] = __float2half(acc[mt][nt][2]);
            if (m_r1 < M && n_c1 < N) C[(size_t)m_r1 * N + n_c1] = __float2half(acc[mt][nt][3]);
        }
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Host-side dispatch (public entrypoint; mirrors the streaming dispatcher's
// validation but without the 16-byte alignment requirement — scalar loads)
// ---------------------------------------------------------------------------
torch::Tensor qgemm_debug_simple(torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    int64_t bitwidth,
    int64_t group_size
) {
    TORCH_CHECK(A.is_cuda() && indices.is_cuda() && lut.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(indices.dtype() == torch::kUInt8, "indices must be uint8");
    TORCH_CHECK(lut.dtype() == torch::kFloat16, "lut must be float16");
    TORCH_CHECK(bitwidth >= 1 && bitwidth <= 4,
                "bitwidth must be 1, 2, 3 or 4 (the idxN family)");
    TORCH_CHECK(group_size == 16 || group_size == 32 || group_size == 64 ||
                group_size == 128 || group_size == 256 || group_size == 512,
                "group_size must be one of 16/32/64/128/256/512 "
                "(full-GS-range kernels)");
    TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2-D [N, K*bitwidth/8]");
    const int B = (int)bitwidth;
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = indices.size(0);
    const int64_t row_bytes = ((int64_t)K * B + 7) >> 3;
    TORCH_CHECK(lut.dim() == 2 && lut.size(1) == (1 << B),
                "lut must be 2-D [num_groups, 2^bitwidth]");
    TORCH_CHECK(A.is_contiguous() && indices.is_contiguous() && lut.is_contiguous(),
                "A, indices, lut must be contiguous");
    TORCH_CHECK(K % 32 == 0,
                "K must be a multiple of 32 (the debug twin's BK constraint; "
                "GS no longer constrains K — the decoupling)");
    TORCH_CHECK(indices.size(1) == row_bytes,
                "indices must have shape [N, K*bitwidth/8]");
    TORCH_CHECK(lut.size(0) == (N + group_size - 1) / group_size,
                "lut must have shape [ceil(N/group_size), 2^bitwidth]");

    auto C = torch::empty({M, N}, A.options());
    // Degenerate shapes: zero-sized grids are invalid launches; K == 0
    // must yield zeros, not uninitialized memory.
    if (M == 0 || N == 0 || K == 0) {
        return (K == 0 && M > 0 && N > 0) ? torch::zeros({M, N}, A.options()) : C;
    }

    dim3 block(128);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    // BK decoupled from GS (the streaming kernel's rule): GS <= 32 ->
    // BK=32; GS >= 64 -> BK=64 whenever K % 64 == 0, else BK=32. The GS
    // arm is a hard TORCH_CHECK by construction (the default case).
#define FLUTE_LAUNCH_DEBUG_SIMPLE(KER)                                        \
    do {                                                                       \
        KER<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(            \
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),          \
            indices.data_ptr<uint8_t>(),                                       \
            reinterpret_cast<const __half*>(lut.data_ptr<at::Half>()),        \
            reinterpret_cast<__half*>(C.data_ptr<at::Half>()),                 \
            M, N, K);                                                          \
    } while (0)

#define FLUTE_DEBUG_GS_BODY_B4()                                            \
    do {                                                                       \
        const bool bk64 = (GS >= 64) && (K % 64 == 0);                       \
        if (bk64) FLUTE_LAUNCH_DEBUG_SIMPLE(                                  \
            (flute_kernel_debug_simple<SimpleConfig<64, GS>>));               \
        else      FLUTE_LAUNCH_DEBUG_SIMPLE(                                  \
            (flute_kernel_debug_simple<SimpleConfig<32, GS>>));               \
    } while (0)

#define FLUTE_DEBUG_GS_BODY(KER_TMPL, B)                                      \
    do {                                                                       \
        const bool bk64 = (GS >= 64) && (K % 64 == 0);                       \
        if (bk64) FLUTE_LAUNCH_DEBUG_SIMPLE((KER_TMPL<SimpleConfig<64, GS>,   \
                                               B>));                           \
        else      FLUTE_LAUNCH_DEBUG_SIMPLE((KER_TMPL<SimpleConfig<32, GS>,   \
                                               B>));                           \
    } while (0)

    // (B, kernel-template) pairs; the GS switch provides constexpr GS. The
    // 4-bit kernel is single-template (no B argument).
    if (B == 4) {
        switch ((int)group_size) {
            case 16:  { constexpr int GS = 16;  FLUTE_DEBUG_GS_BODY_B4(); break; }
            case 32:  { constexpr int GS = 32;  FLUTE_DEBUG_GS_BODY_B4(); break; }
            case 64:  { constexpr int GS = 64;  FLUTE_DEBUG_GS_BODY_B4(); break; }
            case 128: { constexpr int GS = 128; FLUTE_DEBUG_GS_BODY_B4(); break; }
            case 256: { constexpr int GS = 256; FLUTE_DEBUG_GS_BODY_B4(); break; }
            case 512: { constexpr int GS = 512; FLUTE_DEBUG_GS_BODY_B4(); break; }
            default: TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512");
        }
    } else if (B == 3) {
        switch ((int)group_size) {
            case 16:  { constexpr int GS = 16;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 3); break; }
            case 32:  { constexpr int GS = 32;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 3); break; }
            case 64:  { constexpr int GS = 64;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 3); break; }
            case 128: { constexpr int GS = 128; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 3); break; }
            case 256: { constexpr int GS = 256; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 3); break; }
            case 512: { constexpr int GS = 512; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 3); break; }
            default: TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512");
        }
    } else if (B == 2) {
        switch ((int)group_size) {
            case 16:  { constexpr int GS = 16;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 2); break; }
            case 32:  { constexpr int GS = 32;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 2); break; }
            case 64:  { constexpr int GS = 64;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 2); break; }
            case 128: { constexpr int GS = 128; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 2); break; }
            case 256: { constexpr int GS = 256; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 2); break; }
            case 512: { constexpr int GS = 512; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 2); break; }
            default: TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512");
        }
    } else {
        switch ((int)group_size) {
            case 16:  { constexpr int GS = 16;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 1); break; }
            case 32:  { constexpr int GS = 32;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 1); break; }
            case 64:  { constexpr int GS = 64;  FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 1); break; }
            case 128: { constexpr int GS = 128; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 1); break; }
            case 256: { constexpr int GS = 256; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 1); break; }
            case 512: { constexpr int GS = 512; FLUTE_DEBUG_GS_BODY(flute_kernel_debug_simple_sub4, 1); break; }
            default: TORCH_CHECK(false, "group_size must be 16/32/64/128/256/512");
        }
    }
#undef FLUTE_DEBUG_GS_BODY_B4
#undef FLUTE_DEBUG_GS_BODY
#undef FLUTE_LAUNCH_DEBUG_SIMPLE
    // Surface launch-config errors immediately.
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_debug_simple launch failed (grid ",
                grid.x, "x", grid.y, ", 128 threads): ",
                cudaGetErrorString(launch_err));
    return C;
}
