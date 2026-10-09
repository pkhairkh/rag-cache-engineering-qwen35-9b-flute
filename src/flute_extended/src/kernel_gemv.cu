/**
 * src/kernel_gemv.cu
 *
 * The M=1 decode GEMV (flute_kernel_gemv_dual — the memory-bound
 * one-launch-per-module streamer) and the FHT-fused variant —
 * extracted from the former monolith. Shared pair-decode
 * primitives live in flute/gemv.cuh; the split-K workspace in
 * src/gemv_host.cpp.
 *
 * Entry points (flute/entrypoints.h):
 *   qgemm_cutlass_gemv_stream     — the plain GEMV (A pre-rotated)
 *   qgemm_cutlass_gemv_fht_stream — the boundary-fold-fused GEMV
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
// the M=1 decode GEMV — flute_kernel_gemv_dual (entry
// qgemm_cutlass_gemv_stream). ONE memory-bound launch computes
//     C[1, N] = A @ W1^T  (+ A @ W2^T)  +  (A @ resB^T) @ resA^T  +  bias
// for the DECODE shape (M == 1) — the shape the whole-forward CUDA-graph
// replays of scripts/eval_greedy_match.py dominate with (the box
// report: both arms decode_backend "graphs", 13.514 tok/s = 74.0 ms/token
// of GPU-side kernel time; the dual mma kernel above is ~12x above the
// code-stream memory floor at that shape).
//
// Why no tensor cores here: at M == 1 an m16n8k16 atom does 16-32x the
// necessary row work (the BM=32 "decode-slim" tiles still burn 31 of 32
// rows). A GEMV-shaped kernel is bounded by the code stream instead:
//   * the idxN chunk reads stay exactly the coalesced warp pattern the
//     layout was designed for (CUDA_C_Best_Practices_Guide section 10.2.1
//     "Coalesced Access to Global Memory"; the dot-product example in its
//     shared-memory chapter is the same uncoalesced-B-operand trap this
//     layout avoids by construction);
//   * the A operand is staged to shared memory ONCE and read as broadcast
//     u32 k-pairs (the same guide's shared-memory-tile technique);
//   * the LUT never touches shared memory: the palette of the warp's group
//     (one 2^b-entry table per stream — the warp's 64 rows [t*128+wx*64,
//     +64) lie in ONE LUT group because GS is a multiple of 64) is held in
//     registers, one entry per lane, and served by shfl.sync.idx at one
//     shuffle per code (PTX ISA 9.4 section 9.7.10.6 shfl.sync, "register
//     data shuffle within threads of a warp" — the non-.sync section
//     9.7.10.5 variant is deprecated since PTX ISA 6.0) — zero bank
//     conflicts, zero smem;
//   * 256 threads / ~60 registers / <= 35 KB smem keeps 4+ blocks per SM
//     (Ampere_Tuning_Guide section 4.1.1: 48 warps/SM, 16 blocks/SM at
//     CC 8.6).
//
// Numerics contract (documented, decode-only): per output row the k-pairs
// accumulate in ascending-k order per lane, the 4 i-lanes' partials
// butterfly-add (shfl_xor 1, 2), the 4 K-quarter warp-pairs add in fixed
// order red[row][0..3] — deterministic, but NOT bit-identical to the dual
// mma path (different fp32 association). Same class as the dual-stream
// contract: routed only at M == 1, so prefill/PPL numerics stay
// byte-identical to the two-launch route; decode greedy tokens may flip
// on near-ties (first-divergence is the aggregate to read). All the eval
// gates compare like-with-like (warmup, capture and generate() all route
// M == 1 through this kernel), so the graph verifications stay exact.
//
// Layout contract: identical to the fragment-direct family (idxN blobs,
// N % 128 == 0, K % 64 == 0, GS in {64,128,256,512,1024,2048}); the pair
// decode below is a transcription of idxN.py's NORMATIVE blob map:
//   tile (t = n/128, g = k/64) at blob offset ((t*K/64) + g)*1024*b bytes;
//   thread (wx, lane) owns the 16*b-byte chunk at tile +
//   (wx*512 + lane*16)*b; its 64 k-PAIRS p = kt*16 + v*4 + d*2 + s2 cover
//   n = t*128 + wx*64 + v*16 + d*8 + (lane >> 2) and
//   k = g*64 + kt*16 + 2*(lane & 3) + 8*s2 (+1).
// ---------------------------------------------------------------------------


template <int B1, int B2>
__global__ void __launch_bounds__(256, 4)
flute_kernel_gemv_dual(const __half*    __restrict__ A,      // [1, K] row-major
    const uint8_t*   __restrict__ Q1,     // stream-1 idxN blob
    const __half*    __restrict__ LUT1,   // [ceil(N/GS), 2^B1]
    const uint8_t*   __restrict__ Q2,     // stream-2 idxN blob (Q1 when B2 == 0)
    const __half*    __restrict__ LUT2,   // [ceil(N/GS), 2^B2] (LUT1 when B2 == 0)
    const __half*    __restrict__ resB,   // (R, K) fp16 or nullptr
    const __half*    __restrict__ resA,   // (N, R) fp16 or nullptr
    const __half*    __restrict__ bias,   // (N,) fp16 or nullptr
    __half*          __restrict__ C,      // [1, N]
    int N, int K, int R, int GS
) {
    const int t   = blockIdx.x;              // one 128-row tile per CTA
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wx = warp & 1;                 // rows [wx*64, wx*64 + 64)
    const int j4 = warp >> 1;                // K-quarters: g = j4, j4+4, ...

    extern __shared__ __align__(16) unsigned char smem_raw[];
    // [ x staged as u32 k-pairs | red[128][4] row partials | xBf[16] ]
    uint4*  x2v = reinterpret_cast<uint4*>(smem_raw);         // K/8 vectors
    float*  red = reinterpret_cast<float*>(x2v + (K >> 3));   // 512 floats
    float*  xBf = red + 512;                                  // 16 floats

    // ---- stage x = A[0, :] as u32 k-pairs (16 B vectors) ----------------
    for (int i = tid; i < (K >> 3); i += 256) {
        flute::ldg_nc_evict_first_v4(x2v[i], A + ((size_t)i << 3));
    }

    // ---- the residual partials xBf[r] = <x, resB[r]> ---------------------
    // Warp w owns ranks 2w and 2w+1 (8 warps x 2 = the 16-rank contract);
    // lane-per-k keeps the resB reads coalesced, the A reads broadcast.
    if (resB != nullptr) {
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int r = warp * 2 + rr;
            if (r < R) {
                float p = 0.0f;
                const __half* brow = resB + (size_t)r * K;
                for (int k0 = 0; k0 < K; k0 += 32) {
                    const int k = k0 + lane;
                    p = fmaf(__half2float(A[k]), __half2float(brow[k]), p);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    p += __shfl_down_sync(0xffffffffu, p, off);
                if (lane == 0) xBf[r] = p;
            }
        }
    }
    __syncthreads();       // publish x2v AND xBf

    // ---- the per-warp palette: ONE group per warp (GS a multiple of 64,
    //      host-checked), one register per lane, shfl.idx-served ----------
    const int grow = (t * 128 + wx * 64) / GS;
    const float pal1 = __half2float(
        LUT1[(size_t)grow << B1 | (lane & ((1 << B1) - 1))]);
    float pal2 = 0.0f;
    if constexpr (B2 > 0) {
        pal2 = __half2float(
            LUT2[(size_t)grow << B2 | (lane & ((1 << B2) - 1))]);
    }

    // ---- the K loop: 8 fp32 accumulators (rows r = v*16 + d*8 + g_lane,
    //      static index v*2+d — the pair loop is fully unrolled) ----------
    float acc[8];
    #pragma unroll
    for (int a = 0; a < 8; ++a) acc[a] = 0.0f;

    const int G = K >> 6;                    // 64-k tiles
    const uint8_t* cb1 = Q1 + ((size_t)t * G) * (1024 * B1)
                              + (size_t)(wx * 512 + lane * 16) * B1;
    const uint8_t* cb2 = (B2 > 0)
        ? (Q2 + ((size_t)t * G) * (1024 * B2)
                + (size_t)(wx * 512 + lane * 16) * B2)
        : Q1;
    const uint32_t* x2 = reinterpret_cast<const uint32_t*>(smem_raw);

    for (int g = j4; g < G; g += 4) {
        uint4 qv1[B1];
        #pragma unroll
        for (int u = 0; u < B1; ++u)
            flute::ldg_nc_evict_first_v4(qv1[u], cb1 + (size_t)g * (1024 * B1) + u * 16);
        uint4 qv2[B2 > 0 ? B2 : 1];
        if constexpr (B2 > 0) {
            #pragma unroll
            for (int u = 0; u < B2; ++u)
                flute::ldg_nc_evict_first_v4(qv2[u], cb2 + (size_t)g * (1024 * B2) + u * 16);
        }

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                #pragma unroll
                for (int d = 0; d < 2; ++d) {
                    #pragma unroll
                    for (int s2 = 0; s2 < 2; ++s2) {
                        const int p = kt * 16 + v * 4 + d * 2 + s2;
                        const int xi = g * 32 + kt * 8 + (lane & 3) + (s2 << 2);
                        const __half2 xh2 =
                            *reinterpret_cast<const __half2*>(&x2[xi]);
                        const float2 xf = __half22float2(xh2);
                        const int ai = v * 2 + d;
                        {
                            uint32_t c0, c1;
                            flute::gemv_pair_codes<B1>(qv1, p, c0, c1);
                            acc[ai] = fmaf(xf.x, __shfl_sync(
                                0xffffffffu, pal1, (int)c0), acc[ai]);
                            acc[ai] = fmaf(xf.y, __shfl_sync(
                                0xffffffffu, pal1, (int)c1), acc[ai]);
                        }
                        if constexpr (B2 > 0) {
                            uint32_t c0, c1;
                            flute::gemv_pair_codes<B2>(qv2, p, c0, c1);
                            acc[ai] = fmaf(xf.x, __shfl_sync(
                                0xffffffffu, pal2, (int)c0), acc[ai]);
                            acc[ai] = fmaf(xf.y, __shfl_sync(
                                0xffffffffu, pal2, (int)c1), acc[ai]);
                        }
                    }
                }
            }
        }
    }

    // ---- reduction 1: the 4 i-lanes (lane & 3) hold partials of the SAME
    //      8 rows (their k's partition k mod 8) — butterfly them ----------
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        acc[a] += __shfl_xor_sync(0xffffffffu, acc[a], 1);
        acc[a] += __shfl_xor_sync(0xffffffffu, acc[a], 2);
    }
    // ---- reduction 2: the 4 K-quarter warp-pairs -> red[row][j4] -------
    if ((lane & 3) == 0) {
        #pragma unroll
        for (int v = 0; v < 4; ++v)
            #pragma unroll
            for (int d = 0; d < 2; ++d)
                red[(wx * 64 + v * 16 + d * 8 + (lane >> 2)) * 4 + (warp >> 1)]
                    = acc[v * 2 + d];
    }
    __syncthreads();

    // ---- the CTA epilogue: one thread per row, everything in fp32 ------
    if (tid < 128) {
        const int n = t * 128 + tid;
        float y = red[tid * 4 + 0] + red[tid * 4 + 1] +
                  red[tid * 4 + 2] + red[tid * 4 + 3];
        if (resA != nullptr) {
            const __half* ra = resA + (size_t)n * R;
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r < R) y = fmaf(xBf[r], __half2float(ra[r]), y);
            }
        }
        if (bias != nullptr) y += __half2float(bias[n]);
        C[n] = __float2half(y);
    }
}

//  GEMV launch helper. smem = 2*K + 2112 bytes (x pairs + red + xBf):
// 10.3 KB at K=4096, 26.8 KB at K=12288 — every deployed shape fits the
// 48 KB default; the attribute opt-in is kept for uniformity with the
// other launch helpers (and for any future deep-K experiment).
template <int B1, int B2>
void launch_gemv_dual(const __half* A, const uint8_t* Q1, const __half* LUT1,
    const uint8_t* Q2, const __half* LUT2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* C, int N, int K, int R, int GS
) {
    const int smem_bytes = K * 2 + 2112;
    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_gemv_dual<B1, B2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=100224) "
                "failed for the decode-GEMV kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    dim3 block(256);
    dim3 grid(N / 128);
    flute_kernel_gemv_dual<B1, B2>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, Q1, LUT1, Q2, LUT2, resB, resA, bias, C, N, K, R, GS);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_gemv_dual launch failed (grid ", grid.x,
                ", 256 threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}



// ---------------------------------------------------------------------------
// the decode-GEMV dispatch. ONE instantiation per width pair (GS is a
// RUNTIME argument — the GEMV has no GS-shaped smem or tile geometry, the
// group only indexes the LUT rows), so the whole table is 7 kernels — the
// compile-time cost the dual-stream round paid 28 instantiations for is
// deliberately NOT repeated here. GS is still constrained to the
// 64-multiples {64,128,256,512,1024,2048}: the kernel's per-warp palette
// argument requires the warp's 64 rows [t*128+wx*64, +64) to lie in ONE
// LUT group. Unlisted pairs are REFUSED loudly — the Python wrapper routes
// them to the dual / two-launch paths BEFORE reaching this entry.
// ---------------------------------------------------------------------------

template <int B1, int B2>
void dispatch_gemv(const __half* a, const uint8_t* q1, const __half* l1,
    const uint8_t* q2, const __half* l2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* c, int N, int K, int R, int GS
) {
    launch_gemv_dual<B1, B2>(a, q1, l1, q2, l2, resB, resA, bias,
                             c, N, K, R, GS);
}

#define FLUTE_GEMV_PAIR(b1v, b2v)                                             \
    do {                                                                      \
        if (B1 == b1v && B2 == b2v) {                                          \
            dispatch_gemv<b1v, b2v>(\
                a_p, q1_p, l1_p, q2_p, l2_p, rb_p, ra_p, bi_p, c_p,           \
                N, K, R, (int)group_size);                                     \
            return C;                                                          \
        }                                                                      \
    } while (0)

torch::Tensor qgemm_cutlass_gemv_stream_impl(torch::Tensor A,
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

    TORCH_CHECK(A.is_cuda(), "qgemm_gemv_stream: A must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda() && lut.is_cuda(),
                "qgemm_gemv_stream: stream-1 indices/lut must be CUDA");
    if (has_s2) {
        TORCH_CHECK(indices2.is_cuda() && lut2.is_cuda(),
                    "qgemm_gemv_stream: stream-2 indices/lut must be CUDA");
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
                "qgemm_gemv_stream: FLUTE_NO_FD=1 is set (the GEMV is an "
                "idxN-layout consumer)");

    TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
    const int M = A.size(0);
    const int K = A.size(1);
    TORCH_CHECK(M == 1,
                "qgemm_gemv_stream: the decode GEMV serves M == 1 (got M=",
                M, ") — route M 2..16 through qgemm_dual_stream and larger "
                "shapes through the two-launch qgemm_per_group_lut path");
    TORCH_CHECK(K % 64 == 0,
                "qgemm_gemv_stream: K must be a multiple of 64 (the idxN "
                "blob's 64-k tiles; got K=", K, ")");
    TORCH_CHECK(group_size == 64 || group_size == 128 ||
                group_size == 256 || group_size == 512 ||
                group_size == 1024 || group_size == 2048,
                "qgemm_gemv_stream: group_size must be one of "
                "64/128/256/512/1024/2048 (got ", int(group_size), ") — the "
                "per-warp palette needs the 64-row span inside ONE group; "
                "GS 16/32 keeps the two-launch route");
    TORCH_CHECK(2 * K + 2112 <= 99 * 1024,
                "qgemm_gemv_stream: K=", K, " needs ", 2 * K + 2112,
                " B of shared memory (over the SM_86 99 KB block limit)");

    // N from the stream-1 blob byte count (identical derivation to the
    // dual impl; both streams share N).
    const int64_t row_bytes1 = (int64_t)(K * B1) >> 3;
    const int N = (int)(indices.numel() / row_bytes1);
    TORCH_CHECK(N > 0 && (int64_t)N * row_bytes1 == indices.numel(),
                "qgemm_gemv_stream: stream-1 byte count ", indices.numel(),
                " is not N*(K*", B1, "/8) for any N (K = ", K, ")");
    TORCH_CHECK(N % 128 == 0,
                "qgemm_gemv_stream: idxN layout requires N % 128 == 0 "
                "(got N = ", N, ")");
    if (has_s2) {
        const int64_t row_bytes2 = (int64_t)(K * B2) >> 3;
        TORCH_CHECK(indices2.numel() == (int64_t)N * row_bytes2,
                    "qgemm_gemv_stream: stream-2 byte count ",
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

    // Residual + bias contracts (identical to the dual impl).
    const bool has_res = resB.numel() > 0;
    TORCH_CHECK(has_res == (resA.numel() > 0),
                "qgemm_gemv_stream: resB and resA must be supplied "
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

    // Degenerate shapes (mirrors the dual impl).
    auto C = torch::empty({1, N}, A.options());
    if (N == 0 || K == 0) {
        return (K == 0 && N > 0)
            ? torch::zeros({1, N}, A.options()) : C;
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

    // The width-pair table (mirrors FLUTE_DUAL_PAIR).
    FLUTE_GEMV_PAIR(1, 4);
    FLUTE_GEMV_PAIR(4, 1);
    FLUTE_GEMV_PAIR(2, 3);
    FLUTE_GEMV_PAIR(2, 4);
    FLUTE_GEMV_PAIR(3, 3);
    FLUTE_GEMV_PAIR(3, 0);
    FLUTE_GEMV_PAIR(4, 0);

    TORCH_CHECK(false,
                "qgemm_gemv_stream: unsupported (bitwidth, bitwidth2) = (",
                B1, ", ", B2, ") — the compiled pairs are (1,4), (4,1), "
                "(2,3), (2,4), (3,3), (3,0) and (4,0); route other pairs "
                "through the dual / two-launch paths");
    return C;   // unreachable (TORCH_CHECK(false) above)
}

// ---------------------------------------------------------------------------
// the FHT-fused decode GEMV — flute_kernel_gemv_fht_dual (entry
// qgemm_cutlass_gemv_fht_stream). The box report (2026-10-08) left the
// quant arm at 22.361 tok/s = 0.945x dense with PPL +3.52% unchanged: the
// GEMV itself rides the code stream, and the residual 2.47 ms/token gap is
// the 281 per-module FHT kernel launches bracketing every GEMV (562 quant
// launches vs the dense arm's 282 — the box's own accounting, the box read-out
// §3.5's fuse-the-rotation item).  folds the boundary-fold rotation
// INTO the GEMV as a prologue:
//
//     C[1, N] = fht(A) @ W1^T  (+ fht(A) @ W2^T)
//               +  (fht(A) @ resB^T) @ resA^T  +  bias
//     fht(A)   =  ((A [ * s]) @ T)  [ / s]        (T per flute/fht.cuh)
//
// ONE launch per module — 281 total, the dense arm's own launch count
// (the whole-forward CUDA-graph replays then carry half the kernels per
// token; CUDA_C_Programming_Guide's graph-replay section: a graph node
// costs ~1 launch regardless, so the node count is the decode currency).
//
// Numerics contract: the prologue is a line-for-line transcription of
// fht_forward_kernel / fht_forward_awq_kernel (src/kernel_fht.cu,) —
// the same fp32 staging cast (float(x) [* s]), the same
// flute::fht_stage butterflies in fp32 shared memory with
// __syncthreads() between stages, and the same epilogue
// multiply-multiply-(divide) sequence with rsqrtf(b) and the same final
// __float2half rounding. The smem x row the GEMV consumes is therefore
// BIT-IDENTICAL to the standalone kernel's global-memory output, and the
// body below is the body verbatim (the residual dots read the same
// fp16 bits from smem that  read from global A). Decode outputs are
// bit-identical to the FHT-then-GEMV chain — PPL (M > 1, never routed
// here) and the greedy texts both stay exactly where the run left
// them; only the launch count changes. The butterflies run on 256 GEMV
// threads instead of the 512/1024 the FHT launcher picks — fht_stage
// values are thread-count-independent (each pair (i, i+len) -> (a+b, a-b)
// is computed by exactly one thread; see include/flute/fht.cuh).
//
// Shared-memory layout (the race-free arrangement):
//     [ x fp16: 2*K bytes ][ FHT staging + red/xBf: max(4*b_max, 2112) ]
// The per-segment fp32 staging lives at [2*K, 2*K + 4*b_seg) — disjoint
// from the x row it writes, so the fp32->fp16 epilogue can never alias
// its own inputs. (The compacted alternative — x fp16 growing INSIDE the
// staging window — is racy: write(j) at bytes [2j, 2j+2) aliases the
// still-live read(floor(j/2)) at bytes [4*floor(j/2), +4), and those two
// accesses belong to different threads; no phase split fixes every
// alias, so the disjoint layout it is.) red[128][4] and xBf[16] overlay
// the staging TAIL — staging is dead after the prologue and red/xBf are
// written after it, so the union is safe (the tail is >= 2112 B because
// b_max >= 528 for every K the host check admits).
//   K=4096  (one 4096 segment):      24576 B — 4 CTAs/SM by smem, and the
//                                     grid (N/128 <= 96) is the real cap;
//   K=12288 (down_proj, 8192+4096):  57344 B — 1 CTA/SM by smem, but the
//                                     down_proj grid is N/128 = 32 < 72
//                                     SMs, so the block count, not smem,
//                                     is the occupancy limiter there too
// (Ampere_Tuning_Guide section 4.1.1's occupancy arithmetic; the
// Best-Practices shared-memory chapter's "stage once, broadcast many"
// pattern is exactly what the x row does for both the residual dots and
// the K loop).
//
// Layout contract: identical to the GEMV (idxN blobs, N % 128 == 0,
// K % 64 == 0, GS in {64,128,256,512,1024,2048}, the same 7-pair width
// table) plus the FHT contracts of src/kernel_fht.cu (the (K,) fp32
// sign vector; the (K,) fp32 AWQ scale vector when compensated) and the
// fused smem bound 2*K + max(4*b_max, 2112) <= 99 KB (b_max = the first
// segment = the largest power of two <= K).
// ---------------------------------------------------------------------------

template <int B1, int B2, bool kAwq>
__global__ void __launch_bounds__(256, 4)
flute_kernel_gemv_fht_dual(const __half*    __restrict__ A,      // [1, K] row-major, UNROTATED
    const float*     __restrict__ signs,  // (K,) +-1 (T's column signs)
    const float*     __restrict__ s,      // (K,) AWQ scales (kAwq only)
    const uint8_t*   __restrict__ Q1,     // stream-1 idxN blob
    const __half*    __restrict__ LUT1,   // [ceil(N/GS), 2^B1]
    const uint8_t*   __restrict__ Q2,     // stream-2 idxN blob (Q1 when B2 == 0)
    const __half*    __restrict__ LUT2,   // [ceil(N/GS), 2^B2] (LUT1 when B2 == 0)
    const __half*    __restrict__ resB,   // (R, K) fp16 or nullptr
    const __half*    __restrict__ resA,   // (N, R) fp16 or nullptr
    const __half*    __restrict__ bias,   // (N,) fp16 or nullptr
    __half*          __restrict__ C,      // [1, N]
    int N, int K, int R, int GS, flute::FhtSegs segs
) {
    const int t   = blockIdx.x;              // one 128-row tile per CTA
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wx = warp & 1;                 // rows [wx*64, wx*64 + 64)
    const int j4 = warp >> 1;                // K-quarters: g = j4, j4+4, ...

    extern __shared__ __align__(16) unsigned char smem_raw[];
    // [ x fp16: 2*K | FHT fp32 staging: max(4*b_max, 2112) ] — red/xBf
    // overlay the staging tail (dead after the prologue; see above).
    int b_max = 0;
    for (int i = 0; i < segs.n; ++i)
        b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    const int tail = (4 * b_max >= 2112) ? 4 * b_max : 2112;
    float* fstage = reinterpret_cast<float*>(smem_raw + 2 * K);
    float* red    = reinterpret_cast<float*>(smem_raw + 2 * K + tail - 2112);
    float* xBf    = red + 512;                                  // 16 floats

    // ----  prologue: the boundary-fold FHT, transcribed line for line
    //      from fht_forward[_awq]_kernel (src/kernel_fht.cu) -------------
    for (int sgi = 0; sgi < segs.n; ++sgi) {
        const int off = segs.off[sgi];
        const int b = segs.len[sgi];
        const float inv_sqrt_b = rsqrtf(static_cast<float>(b));

        // stage the segment in fp32 (the *s prologue when compensated)
        for (int j = tid; j < b; j += 256) {
            float v = __half2float(A[off + j]);
            if constexpr (kAwq) {
                v = v * s[off + j];
            }
            fstage[j] = v;
        }
        __syncthreads();

        flute::fht_block<256>(fstage, b);   // shift/mask + local stages

        // epilogue: the rotated fp16 row lands at x[off .. off+b)
        __half* xdst = reinterpret_cast<__half*>(smem_raw) + off;
        for (int j = tid; j < b; j += 256) {
            const float v = fstage[j] * signs[off + j] * inv_sqrt_b;
            if constexpr (kAwq) {
                xdst[j] = __float2half(v / s[off + j]);
            } else {
                xdst[j] = __float2half(v);
            }
        }
        __syncthreads();   // x[off, off+b) published; staging reusable
    }

    // ---- the residual partials xBf[r] = <x_rot, resB[r]> — the rotated
    //      row from SMEM (read the identical bits from global A) ----
    // Warp w owns ranks 2w and 2w+1 (8 warps x 2 = the 16-rank contract);
    // lane-per-k keeps the resB reads coalesced, the x reads broadcast.
    if (resB != nullptr) {
        const __half* xrow = reinterpret_cast<const __half*>(smem_raw);
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int r = warp * 2 + rr;
            if (r < R) {
                float p = 0.0f;
                const __half* brow = resB + (size_t)r * K;
                for (int k0 = 0; k0 < K; k0 += 32) {
                    const int k = k0 + lane;
                    p = fmaf(__half2float(xrow[k]), __half2float(brow[k]), p);
                }
                #pragma unroll
                for (int soff = 16; soff > 0; soff >>= 1)
                    p += __shfl_down_sync(0xffffffffu, p, soff);
                if (lane == 0) xBf[r] = p;
            }
        }
    }
    __syncthreads();       // publish xBf

    // ---- the per-warp palette: ONE group per warp (GS a multiple of 64,
    //      host-checked), one register per lane, shfl.idx-served ----------
    const int grow = (t * 128 + wx * 64) / GS;
    const float pal1 = __half2float(
        LUT1[(size_t)grow << B1 | (lane & ((1 << B1) - 1))]);
    float pal2 = 0.0f;
    if constexpr (B2 > 0) {
        pal2 = __half2float(
            LUT2[(size_t)grow << B2 | (lane & ((1 << B2) - 1))]);
    }

    // ---- the K loop: 8 fp32 accumulators (rows r = v*16 + d*8 + g_lane,
    //      static index v*2+d — the pair loop is fully unrolled) ----------
    float acc[8];
    #pragma unroll
    for (int a = 0; a < 8; ++a) acc[a] = 0.0f;

    const int G = K >> 6;                    // 64-k tiles
    const uint8_t* cb1 = Q1 + ((size_t)t * G) * (1024 * B1)
                              + (size_t)(wx * 512 + lane * 16) * B1;
    const uint8_t* cb2 = (B2 > 0)
        ? (Q2 + ((size_t)t * G) * (1024 * B2)
                + (size_t)(wx * 512 + lane * 16) * B2)
        : Q1;
    const uint32_t* x2 = reinterpret_cast<const uint32_t*>(smem_raw);

    for (int g = j4; g < G; g += 4) {
        uint4 qv1[B1];
        #pragma unroll
        for (int u = 0; u < B1; ++u)
            flute::ldg_nc_evict_first_v4(qv1[u], cb1 + (size_t)g * (1024 * B1) + u * 16);
        uint4 qv2[B2 > 0 ? B2 : 1];
        if constexpr (B2 > 0) {
            #pragma unroll
            for (int u = 0; u < B2; ++u)
                flute::ldg_nc_evict_first_v4(qv2[u], cb2 + (size_t)g * (1024 * B2) + u * 16);
        }

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                #pragma unroll
                for (int d = 0; d < 2; ++d) {
                    #pragma unroll
                    for (int s2 = 0; s2 < 2; ++s2) {
                        const int p = kt * 16 + v * 4 + d * 2 + s2;
                        const int xi = g * 32 + kt * 8 + (lane & 3) + (s2 << 2);
                        const __half2 xh2 =
                            *reinterpret_cast<const __half2*>(&x2[xi]);
                        const float2 xf = __half22float2(xh2);
                        const int ai = v * 2 + d;
                        {
                            uint32_t c0, c1;
                            flute::gemv_pair_codes<B1>(qv1, p, c0, c1);
                            acc[ai] = fmaf(xf.x, __shfl_sync(
                                0xffffffffu, pal1, (int)c0), acc[ai]);
                            acc[ai] = fmaf(xf.y, __shfl_sync(
                                0xffffffffu, pal1, (int)c1), acc[ai]);
                        }
                        if constexpr (B2 > 0) {
                            uint32_t c0, c1;
                            flute::gemv_pair_codes<B2>(qv2, p, c0, c1);
                            acc[ai] = fmaf(xf.x, __shfl_sync(
                                0xffffffffu, pal2, (int)c0), acc[ai]);
                            acc[ai] = fmaf(xf.y, __shfl_sync(
                                0xffffffffu, pal2, (int)c1), acc[ai]);
                        }
                    }
                }
            }
        }
    }

    // ---- reduction 1: the 4 i-lanes (lane & 3) hold partials of the SAME
    //      8 rows (their k's partition k mod 8) — butterfly them ----------
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        acc[a] += __shfl_xor_sync(0xffffffffu, acc[a], 1);
        acc[a] += __shfl_xor_sync(0xffffffffu, acc[a], 2);
    }
    // ---- reduction 2: the 4 K-quarter warp-pairs -> red[row][j4] -------
    if ((lane & 3) == 0) {
        #pragma unroll
        for (int v = 0; v < 4; ++v)
            #pragma unroll
            for (int d = 0; d < 2; ++d)
                red[(wx * 64 + v * 16 + d * 8 + (lane >> 2)) * 4 + (warp >> 1)]
                    = acc[v * 2 + d];
    }
    __syncthreads();

    // ---- the CTA epilogue: one thread per row, everything in fp32 ------
    if (tid < 128) {
        const int n = t * 128 + tid;
        float y = red[tid * 4 + 0] + red[tid * 4 + 1] +
                  red[tid * 4 + 2] + red[tid * 4 + 3];
        if (resA != nullptr) {
            const __half* ra = resA + (size_t)n * R;
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r < R) y = fmaf(xBf[r], __half2float(ra[r]), y);
            }
        }
        if (bias != nullptr) y += __half2float(bias[n]);
        C[n] = __float2half(y);
    }
}

//  FHT-fused GEMV launch helper. smem = 2*K + max(4*b_max, 2112) bytes:
// 24576 B at K=4096, 57344 B at K=12288 (8192+4096) — every deployed shape
// fits the 99 KB opt-in; the attribute call is unconditional for
// uniformity with the other launch helpers (K=4096's 24576 B is under the
// 48 KB default, but one code path keeps the audit surface small).
template <int B1, int B2, bool kAwq>
void launch_gemv_fht_dual(const __half* A, const float* signs, const float* s,
    const uint8_t* Q1, const __half* LUT1,
    const uint8_t* Q2, const __half* LUT2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* C, int N, int K, int R, int GS, const flute::FhtSegs& segs
) {
    int b_max = 0;
    for (int i = 0; i < segs.n; ++i)
        b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    const int tail = (4 * b_max >= 2112) ? 4 * b_max : 2112;
    const int smem_bytes = 2 * K + tail;
    static const int attr_status = [] {
        return static_cast<int>(cudaFuncSetAttribute(flute_kernel_gemv_fht_dual<B1, B2, kAwq>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    }();
    TORCH_CHECK(attr_status == static_cast<int>(cudaSuccess),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=101376) "
                "failed for the FHT-fused decode-GEMV kernel: ",
                cudaGetErrorString(static_cast<cudaError_t>(attr_status)));

    dim3 block(256);
    dim3 grid(N / 128);
    flute_kernel_gemv_fht_dual<B1, B2, kAwq>
        <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            A, signs, s, Q1, LUT1, Q2, LUT2, resB, resA, bias, C,
            N, K, R, GS, segs);

    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
                "flute_kernel_gemv_fht_dual launch failed (grid ", grid.x,
                ", 256 threads, smem ", smem_bytes, " B): ",
                cudaGetErrorString(launch_err));
}

// the FHT-fused GEMV dispatch. ONE instantiation per (width pair,
// awq) — 14 kernels total (the 7-pair table x the two FHT variants of
// src/kernel_fht.cu). kAwq is a TEMPLATE bool, not a runtime branch: the
// AWQ epilogue's division and the s contract fold at compile time, and
// the plain variant never dereferences s (it is passed nullptr).
template <int B1, int B2>
void dispatch_gemv_fht(const __half* a, const float* signs, const float* s,
    const uint8_t* q1, const __half* l1,
    const uint8_t* q2, const __half* l2,
    const __half* resB, const __half* resA, const __half* bias,
    __half* c, int N, int K, int R, int GS, const flute::FhtSegs& segs,
    bool awq
) {
    if (awq) {
        launch_gemv_fht_dual<B1, B2, true>(a, signs, s, q1, l1, q2, l2, resB, resA, bias,
            c, N, K, R, GS, segs);
    } else {
        launch_gemv_fht_dual<B1, B2, false>(a, signs, nullptr, q1, l1, q2, l2, resB, resA, bias,
            c, N, K, R, GS, segs);
    }
}

#define FLUTE_GEMV_FHT_PAIR(b1v, b2v)                                          \
    do {                                                                       \
        if (B1 == b1v && B2 == b2v) {                                          \
            dispatch_gemv_fht<b1v, b2v>(\
                a_p, signs_p, s_p, q1_p, l1_p, q2_p, l2_p, rb_p, ra_p, bi_p,   \
                c_p, N, K, R, (int)group_size, segs, awq);                     \
            return C;                                                          \
        }                                                                      \
    } while (0)

torch::Tensor qgemm_cutlass_gemv_fht_stream_impl(torch::Tensor A,
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

    TORCH_CHECK(A.is_cuda(), "qgemm_gemv_fht_stream: A must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda() && lut.is_cuda(),
                "qgemm_gemv_fht_stream: stream-1 indices/lut must be CUDA");
    if (has_s2) {
        TORCH_CHECK(indices2.is_cuda() && lut2.is_cuda(),
                    "qgemm_gemv_fht_stream: stream-2 indices/lut must be "
                    "CUDA");
    }
    TORCH_CHECK(A.dtype() == torch::kFloat16,
                "qgemm_gemv_fht_stream: A must be float16 (the UNROTATED "
                "raw input — the kernel applies the fold itself)");
    TORCH_CHECK(indices.dtype() == torch::kUInt8,
                "stream-1 indices must be uint8");
    TORCH_CHECK(lut.dtype() == torch::kFloat16,
                "stream-1 lut must be float16");
    TORCH_CHECK(B1 >= 1 && B1 <= 4,
                "bitwidth must be 1, 2, 3 or 4");
    TORCH_CHECK(B2 >= 0 && B2 <= 4,
                "bitwidth2 must be 0 (single stream) or 1..4");
    TORCH_CHECK(!flute::fd_disabled_by_env(),
                "qgemm_gemv_fht_stream: FLUTE_NO_FD=1 is set (the fused "
                "GEMV is an idxN-layout consumer)");

    TORCH_CHECK(A.dim() == 2, "A must be 2-D [M, K]");
    const int M = A.size(0);
    const int K = A.size(1);
    TORCH_CHECK(M == 1,
                "qgemm_gemv_fht_stream: the fused decode GEMV serves M == 1 "
                "(got M=", M, ") — route M 2..16 through qgemm_dual_stream "
                "and larger shapes through the two-launch "
                "qgemm_per_group_lut path (after the explicit rotation)");
    TORCH_CHECK(K % 64 == 0,
                "qgemm_gemv_fht_stream: K must be a multiple of 64 (the "
                "idxN blob's 64-k tiles AND the FHT segment contract; got "
                "K=", K, ")");
    TORCH_CHECK(group_size == 64 || group_size == 128 ||
                group_size == 256 || group_size == 512 ||
                group_size == 1024 || group_size == 2048,
                "qgemm_gemv_fht_stream: group_size must be one of "
                "64/128/256/512/1024/2048 (got ", int(group_size), ")");
    TORCH_CHECK(2 * K + 2112 <= 99 * 1024,
                "qgemm_gemv_fht_stream: K=", K, " needs ", 2 * K + 2112,
                " B of x-row shared memory (over the SM_86 99 KB block "
                "limit)");

    // ---- the FHT contracts (mirrors fht_check_args / fht_dispatch) -----
    TORCH_CHECK(signs.is_cuda() && signs.dim() == 1 &&
                signs.scalar_type() == at::kFloat &&
                signs.numel() == K && signs.is_contiguous(),
                "qgemm_gemv_fht_stream: signs must be a contiguous (K,) "
                "float32 CUDA tensor matching A's last dim (got numel=",
                signs.numel(), ", K=", K, ")");
    const bool awq = s.numel() > 0;
    if (awq) {
        TORCH_CHECK(s.is_cuda() && s.dim() == 1 &&
                    s.scalar_type() == at::kFloat &&
                    s.numel() == K && s.is_contiguous(),
                    "qgemm_gemv_fht_stream: the AWQ scale s must be a "
                    "contiguous (K,) float32 CUDA tensor (got numel=",
                    s.numel(), ", K=", K, ") — pass an empty tensor for "
                    "the plain (uncompensated) rotation");
    }
    const flute::FhtSegs segs = flute::fht_segments(K);
    TORCH_CHECK(segs.n > 0 && (segs.off[segs.n - 1] + segs.len[segs.n - 1]) == K,
                "qgemm_gemv_fht_stream: the FHT segment table does not "
                "tile K=", K, " (n=", segs.n, ") - internal error");
    int b_max = 0;
    for (int i = 0; i < segs.n; ++i)
        b_max = (segs.len[i] > b_max) ? segs.len[i] : b_max;
    const int fht_tail = (4 * b_max >= 2112) ? 4 * b_max : 2112;
    TORCH_CHECK(2 * K + fht_tail <= 99 * 1024,
                "qgemm_gemv_fht_stream: K=", K, " (largest FHT segment ",
                b_max, ") needs ", 2 * K + fht_tail, " B of shared memory "
                "(over the SM_86 99 KB block limit) — keep the "
                "explicit-FHT route for this shape");

    // N from the stream-1 blob byte count (identical derivation to the
    //  impl; both streams share N).
    const int64_t row_bytes1 = (int64_t)(K * B1) >> 3;
    const int N = (int)(indices.numel() / row_bytes1);
    TORCH_CHECK(N > 0 && (int64_t)N * row_bytes1 == indices.numel(),
                "qgemm_gemv_fht_stream: stream-1 byte count ",
                indices.numel(), " is not N*(K*", B1, "/8) for any N (K = ",
                K, ")");
    TORCH_CHECK(N % 128 == 0,
                "qgemm_gemv_fht_stream: idxN layout requires N % 128 == 0 "
                "(got N = ", N, ")");
    if (has_s2) {
        const int64_t row_bytes2 = (int64_t)(K * B2) >> 3;
        TORCH_CHECK(indices2.numel() == (int64_t)N * row_bytes2,
                    "qgemm_gemv_fht_stream: stream-2 byte count ",
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

    // Residual + bias contracts (identical to the impl).
    const bool has_res = resB.numel() > 0;
    TORCH_CHECK(has_res == (resA.numel() > 0),
                "qgemm_gemv_fht_stream: resB and resA must be supplied "
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

    // Degenerate shapes (mirrors the impl).
    auto C = torch::empty({1, N}, A.options());
    if (N == 0 || K == 0) {
        return (K == 0 && N > 0)
            ? torch::zeros({1, N}, A.options()) : C;
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
    const float*   signs_p = signs.data_ptr<float>();
    const float*   s_p  = awq ? s.data_ptr<float>() : nullptr;
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

    // The width-pair table (mirrors FLUTE_GEMV_PAIR).
    FLUTE_GEMV_FHT_PAIR(1, 4);
    FLUTE_GEMV_FHT_PAIR(4, 1);
    FLUTE_GEMV_FHT_PAIR(2, 3);
    FLUTE_GEMV_FHT_PAIR(2, 4);
    FLUTE_GEMV_FHT_PAIR(3, 3);
    FLUTE_GEMV_FHT_PAIR(3, 0);
    FLUTE_GEMV_FHT_PAIR(4, 0);

    TORCH_CHECK(false,
                "qgemm_gemv_fht_stream: unsupported (bitwidth, bitwidth2) "
                "= (", B1, ", ", B2, ") — the compiled pairs are (1,4), "
                "(4,1), (2,3), (2,4), (3,3), (3,0) and (4,0); route other "
                "pairs through the dual / two-launch paths");
    return C;   // unreachable (TORCH_CHECK(false) above)
}

// ---------------------------------------------------------------------------
}  // namespace

// Public entrypoint : the M=1 decode GEMV.
//   C = A @ W1^T (+ A @ W2^T) + (A @ resB^T) @ resA^T + bias
// ONE memory-bound launch for the decode shape — the whole per-module op
// chain at M == 1 (the shape the whole-forward CUDA-graph replays
// dominate with). GS is RUNTIME here (no GS template dimension — one
// instantiation per width pair). Deployment contract: idxN blobs, M == 1,
// N % 128 == 0, K % 64 == 0, group_size in {64,128,256,512,1024,2048},
// (bitwidth, bitwidth2) one of the compiled pairs — the Python wrapper
// (flute_extended/flute_extended/__init__.py) mirrors the gate and routes
// everything else to the dual / two-launch paths.
torch::Tensor qgemm_cutlass_gemv_stream(torch::Tensor A,
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
    return qgemm_cutlass_gemv_stream_impl(A, indices, lut, bitwidth,
                                          indices2, lut2, bitwidth2,
                                          resB, resA, bias, group_size);
}

// Public entrypoint : the FHT-fused M=1 decode GEMV.
//   C = fht(A) @ W1^T (+ fht(A) @ W2^T) + (fht(A) @ resB^T) @ resA^T + bias
// ONE launch for the whole per-module chain INCLUDING the boundary-fold
// rotation — the FHT (plain, or the AWQ-compensated ((A*s) @ T)/s when s
// is populated) runs as the kernel's prologue, bit-identical to the
// standalone fht_forward / fht_forward_awq kernels it replaces (see the
//  block comment above). A is the UNROTATED raw [1, K] fp16 row;
// signs is the (K,) fp32 fold sign vector; s is the (K,) fp32 AWQ scale
// vector (empty tensor = the plain rotation). Everything else matches
// the GEMV contract (idxN blobs, M == 1, N % 128 == 0, K % 64 == 0,
// GS in {64,128,256,512,1024,2048}, the compiled width-pair table) plus
// the FHT-fused smem bound 2*K + max(4*b_max, 2112) <= 99 KB (b_max =
// the largest power of two <= K) — the Python wrapper
// (flute_extended/flute_extended/__init__.py: qgemm_gemv_stream's
// rot_signs / awq_scale parameters, gemv_fht_stream_supported) mirrors
// the gate and routes everything else to the explicit-FHT route.
torch::Tensor qgemm_cutlass_gemv_fht_stream(torch::Tensor A,
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
    return qgemm_cutlass_gemv_fht_stream_impl(
        A, indices, lut, bitwidth, indices2, lut2, bitwidth2,
        resB, resA, bias, group_size, signs, s);
}

