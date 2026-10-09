/**
 * src/kernel_fht.cu
 *
 * FLUTE Extension: the Fast Hadamard Transform kernel (see
 * include/flute/fht.cuh for the math contract and EXTENSION_REQUIREMENT.md
 * for the requirement).
 *
 * Entry points (bound in src/bindings.cpp):
 *   fht_forward(x, signs)     -> out = x @ T          (out-of-place)
 *   fht_backward(g, signs)    -> gx = g @ T^T         (autograd adjoint)
 *   fht_inplace_(x, signs)    -> x  <- x @ T          (in-place variant)
 *   fht_forward_awq(x, signs, s) -> out = ((x*s) @ T) / s
 *                              (the legacy rotate-then-AWQ fold's
 *                              compensated transform, one kernel — see
 *                              fht_forward_awq_kernel)
 *
 * All three take a 2-D contiguous (M, K) tensor on CUDA (the Python
 * wrapper flattens (batch, seq, K) and dispatches dtype), fp16 / bf16 /
 * fp32 in and out (identical dtype), and a (K,) fp32 sign vector with
 * entries +-1. K must satisfy K % 32 == 0 and K <= 2^16 - 32 (the
 * by-value segment table holds 8 descending power-of-two blocks; the
 * production shapes 4096 (one block) and 12288 = 8192 + 4096
 * (block-diagonal, production down_proj) both qualify).
 *
 * Numerics: the butterfly accumulates in fp32 shared memory regardless
 * of the input dtype; the sign multiply and the 1/sqrt(b) scaling are
 * folded into the load/store epilogues. NaN propagation is honest (no
 * isfinite clamps — the repo kernel contract), and the in-place variant
 * is safe because a segment is fully staged into shared memory before
 * any of its columns are written back.
 *
 * Resource accounting (sm_86, docs/PTX_NOTES.md section 1 — the * checklist):
 *   * one CUDA block per row (grid.x = M), THREADS in {128..1024}
 *     chosen from the largest segment (>= 4 elements per thread);
 *   * dynamic smem = 4 * b_max bytes: 16 KiB (K=4096), 32 KiB
 *     (K=12288) — below the 48 KiB default; a b_max of 16384 (64 KiB)
 *     opts in via cudaFuncSetAttribute (checked, loud on failure);
 *   * __launch_bounds__(THREADS) on every instantiation; registers are
 *     a handful (the working set is smem, not registers);
 *   * occupancy: at THREADS=1024 / 16 KiB smem the kernel is
 *     thread-limited at 1 block/SM for one-row decode launches
 *     (latency target < 10 us is met by launch + 12 butterfly stages)
 *     and smem allows 2 CTAs/SM at the 32 KiB segment (K=12288);
 *   * global traffic: K scalars in + K scalars out per row + K signs
 *     read once per row-block (segment loop reuses the same smem).
 */

#include <cuda.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "flute/fht.cuh"

namespace {

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Forward: out = ((x @ H) * s) / sqrt(b) per segment  ==  x @ T.
// In-place use: pass out == x (safe — each segment is staged in smem
// before its stores; segments touch disjoint column ranges).
template <typename scalar_t, int THREADS>
__global__ void __launch_bounds__(THREADS) fht_forward_kernel(const scalar_t* __restrict__ x,      // (M, K)
    const float*    __restrict__ signs,  // (K,)  +-1
    scalar_t*       __restrict__ out,    // (M, K)  == x allowed
    int K, flute::FhtSegs segs) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const scalar_t* __restrict__ xrow = x + static_cast<int64_t>(row) * K;
    scalar_t* __restrict__ orow = out + static_cast<int64_t>(row) * K;

    for (int sgi = 0; sgi < segs.n; ++sgi) {
        const int off = segs.off[sgi];
        const int b = segs.len[sgi];
        const float inv_sqrt_b = rsqrtf(static_cast<float>(b));

        for (int j = threadIdx.x; j < b; j += THREADS) {
            smem[j] = static_cast<float>(xrow[off + j]);
        }
        __syncthreads();

        flute::fht_block<THREADS>(smem, b);   // shift/mask + local stages

        // epilogue: output-side sign (T is COLUMN-scaled) + 1/sqrt(b)
        for (int j = threadIdx.x; j < b; j += THREADS) {
            orow[off + j] = static_cast<scalar_t>(smem[j] * signs[off + j] * inv_sqrt_b);
        }
        __syncthreads();   // smem reused by the next segment
    }
}

// Backward / adjoint: gx = ((g * s) @ H) / sqrt(b) per segment  ==  g @ T^T.
// (Hadamard is symmetric and self-inverse up to b; the sign moves to the
// input side because the adjoint of H diag(s) is diag(s) H.)
template <typename scalar_t, int THREADS>
__global__ void __launch_bounds__(THREADS) fht_backward_kernel(const scalar_t* __restrict__ g,      // (M, K)
    const float*    __restrict__ signs,  // (K,)  +-1
    scalar_t*       __restrict__ gx,     // (M, K)
    int K, flute::FhtSegs segs) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const scalar_t* __restrict__ grow = g + static_cast<int64_t>(row) * K;
    scalar_t* __restrict__ gxrow = gx + static_cast<int64_t>(row) * K;

    for (int sgi = 0; sgi < segs.n; ++sgi) {
        const int off = segs.off[sgi];
        const int b = segs.len[sgi];
        const float inv_sqrt_b = rsqrtf(static_cast<float>(b));

        // prologue: input-side sign, no output sign
        for (int j = threadIdx.x; j < b; j += THREADS) {
            smem[j] = static_cast<float>(grow[off + j]) * signs[off + j];
        }
        __syncthreads();

        flute::fht_block<THREADS>(smem, b);   // shift/mask + local stages

        for (int j = threadIdx.x; j < b; j += THREADS) {
            gxrow[off + j] = static_cast<scalar_t>(smem[j] * inv_sqrt_b);
        }
        __syncthreads();
    }
}

// Forward + AWQ compensation : out = ((x * s) @ T) / s — the exact
// compensated transform x @ (D T D^-1) the legacy rotate-then-AWQ
// artifacts require (palettized_modules._rotate_input), folded into ONE
// kernel instead of the five-op eager chain
//     x.float() -> *s -> fht(fp32) -> /s -> .to(x.dtype)
// (item 4). Numerics are BIT-IDENTICAL to that chain:
//   * prologue   smem[j] = float(x[off+j]) * s[off+j]
//     (float(x) is the exact x.float() cast; the multiply is the same
//     fp32 op in the same order);
//   * butterfly  unchanged (same smem staging, same fp32 fht_stage);
//   * epilogue   t = smem[j] * signs[off+j] * inv_sqrt_b; out = t / s
//     — the same multiply-multiply-DIVIDE sequence (the division is kept
//     a division, NOT a reciprocal multiply, so the rounding matches the
//     eager /s bit for bit), then the same final dtype cast.
// The kernel reads x in its NATIVE dtype (fp16/bf16/fp32) — the eager
// chain's fp32 round-trip tensors never exist, halving the transform's
// global traffic.
template <typename scalar_t, int THREADS>
__global__ void __launch_bounds__(THREADS) fht_forward_awq_kernel(const scalar_t* __restrict__ x,      // (M, K)
    const float*    __restrict__ signs,  // (K,)  +-1
    const float*    __restrict__ s,      // (K,)  positive AWQ scales
    scalar_t*       __restrict__ out,    // (M, K)
    int K, flute::FhtSegs segs) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const scalar_t* __restrict__ xrow = x + static_cast<int64_t>(row) * K;
    scalar_t* __restrict__ orow = out + static_cast<int64_t>(row) * K;

    for (int sgi = 0; sgi < segs.n; ++sgi) {
        const int off = segs.off[sgi];
        const int b = segs.len[sgi];
        const float inv_sqrt_b = rsqrtf(static_cast<float>(b));

        for (int j = threadIdx.x; j < b; j += THREADS) {
            smem[j] = static_cast<float>(xrow[off + j]) * s[off + j];
        }
        __syncthreads();

        flute::fht_block<THREADS>(smem, b);   // shift/mask + local stages

        for (int j = threadIdx.x; j < b; j += THREADS) {
            const float t = smem[j] * signs[off + j] * inv_sqrt_b;
            orow[off + j] = static_cast<scalar_t>(t / s[off + j]);
        }
        __syncthreads();   // smem reused by the next segment
    }
}

// ---------------------------------------------------------------------------
// Host-side launch helpers
// ---------------------------------------------------------------------------

int fht_threads_for(int b_max) {
    int threads = 128;
    while (threads < 1024 && threads * 4 < b_max) threads <<= 1;
    return threads;
}

// Launch the (THREADS x scalar_t) instantiation, opting into > 48 KiB
// dynamic smem when the largest segment needs it (production shapes
// never do; the path exists so K up to 16384 stays legal).
template <typename scalar_t, int THREADS, bool kBackward>
void launch_fht(const scalar_t* in, const float* signs, scalar_t* out,
                int M, int K, const flute::FhtSegs& segs, cudaStream_t stream) {
    constexpr int kMaxDefaultSmem = 48 * 1024;
    int b_max = 0;
    for (int i = 0; i < segs.n; ++i) b_max = std::max(b_max, segs.len[i]);
    const int smem_bytes = b_max * static_cast<int>(sizeof(float));

    const void* kfn = kBackward
        ? reinterpret_cast<const void*>(&fht_backward_kernel<scalar_t, THREADS>)
        : reinterpret_cast<const void*>(&fht_forward_kernel<scalar_t, THREADS>);
    if (smem_bytes > kMaxDefaultSmem) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    if (kBackward) {
        fht_backward_kernel<scalar_t, THREADS>
            <<<M, THREADS, smem_bytes, stream>>>(in, signs, out, K, segs);
    } else {
        fht_forward_kernel<scalar_t, THREADS>
            <<<M, THREADS, smem_bytes, stream>>>(in, signs, out, K, segs);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool kBackward>
void fht_dispatch(const scalar_t* in, const float* signs, scalar_t* out,
                  int M, int K, cudaStream_t stream) {
    const flute::FhtSegs segs = flute::fht_segments(K);
    TORCH_CHECK(segs.n > 0 && (segs.off[segs.n - 1] + segs.len[segs.n - 1]) == K,
                "fht: the segment table does not tile K=", K,
                " (n=", segs.n, ") - internal error");
    int b_max = 0;
    for (int i = 0; i < segs.n; ++i) b_max = std::max(b_max, segs.len[i]);
    const int threads = fht_threads_for(b_max);
    switch (threads) {
        case 1024:
            launch_fht<scalar_t, 1024, kBackward>(in, signs, out, M, K, segs, stream);
            break;
        case 512:
            launch_fht<scalar_t, 512, kBackward>(in, signs, out, M, K, segs, stream);
            break;
        case 256:
            launch_fht<scalar_t, 256, kBackward>(in, signs, out, M, K, segs, stream);
            break;
        default:
            launch_fht<scalar_t, 128, kBackward>(in, signs, out, M, K, segs, stream);
            break;
    }
}

// the compensated forward (see fht_forward_awq_kernel above). Same
// THREADS ladder and > 48 KiB opt-in discipline as the plain forward.
template <typename scalar_t, int THREADS>
void launch_fht_awq(const scalar_t* in, const float* signs, const float* s,
                    scalar_t* out, int M, int K, const flute::FhtSegs& segs,
                    cudaStream_t stream) {
    constexpr int kMaxDefaultSmem = 48 * 1024;
    int b_max = 0;
    for (int i = 0; i < segs.n; ++i) b_max = std::max(b_max, segs.len[i]);
    const int smem_bytes = b_max * static_cast<int>(sizeof(float));

    const void* kfn = reinterpret_cast<const void*>(
        &fht_forward_awq_kernel<scalar_t, THREADS>);
    if (smem_bytes > kMaxDefaultSmem) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    fht_forward_awq_kernel<scalar_t, THREADS>
        <<<M, THREADS, smem_bytes, stream>>>(in, signs, s, out, K, segs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void fht_dispatch_awq(const scalar_t* in, const float* signs, const float* s,
                      scalar_t* out, int M, int K, cudaStream_t stream) {
    const flute::FhtSegs segs = flute::fht_segments(K);
    TORCH_CHECK(segs.n > 0 && (segs.off[segs.n - 1] + segs.len[segs.n - 1]) == K,
                "fht: the segment table does not tile K=", K,
                " (n=", segs.n, ") - internal error");
    int b_max = 0;
    for (int i = 0; i < segs.n; ++i) b_max = std::max(b_max, segs.len[i]);
    const int threads = fht_threads_for(b_max);
    switch (threads) {
        case 1024:
            launch_fht_awq<scalar_t, 1024>(in, signs, s, out, M, K, segs, stream);
            break;
        case 512:
            launch_fht_awq<scalar_t, 512>(in, signs, s, out, M, K, segs, stream);
            break;
        case 256:
            launch_fht_awq<scalar_t, 256>(in, signs, s, out, M, K, segs, stream);
            break;
        default:
            launch_fht_awq<scalar_t, 128>(in, signs, s, out, M, K, segs, stream);
            break;
    }
}

void fht_check_args(const torch::Tensor& x, const torch::Tensor& signs) {
    TORCH_CHECK(x.is_cuda(), "fht: x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "fht: x must be 2-D (M, K) - the Python ",
                "wrapper flattens (batch, seq, K)");
    TORCH_CHECK(x.is_contiguous(), "fht: x must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat ||
                x.scalar_type() == at::kHalf ||
                x.scalar_type() == at::kBFloat16,
                "fht: x dtype must be float32/float16/bfloat16, got ",
                x.scalar_type());
    TORCH_CHECK(signs.is_cuda() && signs.dim() == 1 &&
                signs.scalar_type() == at::kFloat &&
                signs.numel() == x.size(1) && signs.is_contiguous(),
                "fht: signs must be a contiguous (K,) float32 CUDA tensor ",
                "matching x's last dim (got numel=", signs.numel(),
                ", K=", x.size(1), ")");
    const int K = static_cast<int>(x.size(1));
    TORCH_CHECK(K >= 32 && K % 32 == 0,
                "fht: K must be >= 32 and divisible by 32 (power-of-two ",
                "segment decomposition), got K=", K);
    TORCH_CHECK(K <= (1 << 16) - 32,
                "fht: K=", K, " exceeds the 8-segment table's range ",
                "(max ", (1 << 16) - 32, ")");
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entrypoints (declared in bindings.cpp)
// ---------------------------------------------------------------------------

torch::Tensor fht_forward(torch::Tensor x, torch::Tensor signs) {
    fht_check_args(x, signs);
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    torch::Tensor out = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "fht_forward", [&] {
            fht_dispatch<scalar_t, false>(x.data_ptr<scalar_t>(), signs.data_ptr<float>(),
                out.data_ptr<scalar_t>(), M, K, stream);
        });
    return out;
}

torch::Tensor fht_backward(torch::Tensor grad_out, torch::Tensor signs) {
    fht_check_args(grad_out, signs);
    const int M = static_cast<int>(grad_out.size(0));
    const int K = static_cast<int>(grad_out.size(1));
    torch::Tensor gx = torch::empty_like(grad_out);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad_out.scalar_type(), "fht_backward", [&] {
            fht_dispatch<scalar_t, true>(grad_out.data_ptr<scalar_t>(), signs.data_ptr<float>(),
                gx.data_ptr<scalar_t>(), M, K, stream);
        });
    return gx;
}

torch::Tensor fht_inplace(torch::Tensor x, torch::Tensor signs) {
    fht_check_args(x, signs);
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "fht_inplace", [&] {
            fht_dispatch<scalar_t, false>(x.data_ptr<scalar_t>(), signs.data_ptr<float>(),
                x.data_ptr<scalar_t>(), M, K, stream);
        });
    return x;
}

// the compensated forward — out = ((x * s) @ T) / s (one kernel;
// bit-identical to the eager five-op chain it replaces; see
// fht_forward_awq_kernel above). `s` must be a (K,) contiguous float32
// CUDA tensor (the module validates finite/positive at load; a zero or
// NaN here produces inf/NaN outputs honestly — the loud failure the
// repo contract prefers over a silent clamp).
torch::Tensor fht_forward_awq(torch::Tensor x, torch::Tensor signs,
                              torch::Tensor s) {
    fht_check_args(x, signs);
    TORCH_CHECK(s.is_cuda() && s.dim() == 1 &&
                s.scalar_type() == at::kFloat &&
                s.numel() == x.size(1) && s.is_contiguous(),
                "fht_forward_awq: s must be a contiguous (K,) float32 CUDA "
                "tensor matching x's last dim (got numel=", s.numel(),
                ", K=", x.size(1), ")");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    torch::Tensor out = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "fht_forward_awq", [&] {
            fht_dispatch_awq<scalar_t>(x.data_ptr<scalar_t>(), signs.data_ptr<float>(),
                s.data_ptr<float>(), out.data_ptr<scalar_t>(), M, K, stream);
        });
    return out;
}
