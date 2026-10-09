/**
 * src/kernel_cutlass_dense.cu
 *
 * CUTLASS dense FP16 GEMM baseline: C = A @ W_dense^T on a FULL FP16
 * weight matrix (not palettized). Purpose:
 *   (a) validate the CUTLASS install and SM_86 tensor cores
 *   (b) provide an upper-bound performance reference
 *   (c) A/B comparison against the streaming dequant kernel
 *
 * The production kernel is kernel_cutlass_streaming.cu; this one is a
 * benchmarking aid and requires CUTLASS headers at build time
 * (setup.py defines FLUTE_HAVE_CUTLASS when it finds them; otherwise this
 * TU compiles to a runtime stub that throws).
 *
 * API/compatibility notes (verified against the CUTLASS 4.8 sources):
 *
 *  - Uses cutlass::gemm::device::GemmUniversal — the "classic" universal
 *    device API that example 47 (47_ampere_gemm_universal_streamk) still
 *    ships in 4.8, unchanged since 2.x. The older device::Gemm type is
 *    the one that rotted: its defaulted template arguments resolve
 *    through DefaultGemmConfiguration, which has NO Sm86 specialization
 *    in ANY release (verified by grep against both the 4.8 main branch
 *    and the v2.11.0 tag: zero Sm86 entries in both), so
 *    device::Gemm<..., arch::Sm86, ...> with defaulted args was never
 *    compilable against any CUTLASS version — the original build break
 *    was a policy-tag error, not a "4.x removed 2.x" regression.
 *
 *  - ArchTag = arch::Sm80 is the canonical POLICY tag for SM_86 (every
 *    Ampere example in the tree does this). The SASS target comes from
 *    the -gencode flags in setup.py (sm_86 primary), not from ArchTag.
 *
 *  - Every defaulted template parameter is spelled out explicitly
 *    (ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
 *    Swizzle, Stages, AlignmentA/B, Operator), so nothing can ever
 *    instantiate DefaultGemmConfiguration again.
 *
 * Layout mapping:
 *   A = X       [M, K] row-major   (RowMajor)
 *   B = W^T     [K, N] col-major   (ColumnMajor; W is [N, K] row-major)
 *   C = Y       [M, N] row-major   (RowMajor)
 */

#if defined(FLUTE_HAVE_CUTLASS)

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace {

// ---------------------------------------------------------------------------
// Type configuration (mirrors example 47's Ampere shape, FP32 accumulate).
// LinearCombination's parameters are
//   <ElementOutput, int Count, ElementAccumulator, ElementCompute>
// Count = 128 / sizeof_bits<half_t> = 8 (the vectorized fp16 epilogue
// width). It requires the output's leading dimension (N) to be a multiple
// of 8 and the C pointer 16 B aligned; can_implement() enforces both with
// a status error (all Qwen3.5-9B layer shapes satisfy N % 8 == 0).
// ---------------------------------------------------------------------------
using ElementA = cutlass::half_t;
using LayoutA  = cutlass::layout::RowMajor;     // X is [M, K] row-major
using ElementB = cutlass::half_t;
using LayoutB  = cutlass::layout::ColumnMajor;  // W^T = W [N,K] row-major seen as [K,N] col-major
using ElementC = cutlass::half_t;
using LayoutC  = cutlass::layout::RowMajor;     // Y is [M, N] row-major

using ElementAccumulator = float;               // FP32 accumulate (reference role)
using OperatorClass      = cutlass::arch::OpClassTensorOp;
using ArchTag            = cutlass::arch::Sm80; // POLICY tag for SM_86 (see header comment)
using ThreadblockShape   = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape          = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;  // mma.m16n8k16

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 8, ElementAccumulator, float>;

constexpr int kStages    = 4;   // example 47's Ampere value; 4 x 16 KB = 64 KB <= 99 KB/block on SM_86
constexpr int kAlignA    = 8;   // 16 B vectorized global loads
constexpr int kAlignB    = 8;

// Classic data-parallel configuration (default).
using GemmBasic = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignA,
    kAlignB>;

// Stream-K configuration: same shapes, ThreadblockSwizzleStreamK. Fixes the
// wave-quantization loss when the tile grid barely covers the SM count
// (e.g. M=1: a 96-block grid on 80 SMs = 1.2 waves). Opt-in at runtime via
// FLUTE_DENSE_STREAMK=1 (read once per process, BEFORE the first call).
using GemmStreamK = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
    kStages,
    kAlignA,
    kAlignB>;

bool dense_use_streamk() {
    static const bool streamk = [] {
        const char* env = std::getenv("FLUTE_DENSE_STREAMK");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    return streamk;
}

// SM count for the Stream-K scheduler (queried once per process).
int device_sm_count() {
    static const int sms = [] {
        int dev = 0, count = 80;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev);
        }
        return count;
    }();
    return sms;
}

}  // namespace

torch::Tensor qgemm_cutlass_dense(
    torch::Tensor A,        // [M, K] FP16
    torch::Tensor W_dense,  // [N, K] FP16  (dequantized weight — benchmarking only)
    int64_t /*bitwidth*/,
    int64_t /*group_size*/
) {
    TORCH_CHECK(A.is_cuda(),        "A must be CUDA");
    TORCH_CHECK(W_dense.is_cuda(), "W must be CUDA");
    TORCH_CHECK(A.dtype()       == torch::kFloat16, "A must be float16");
    TORCH_CHECK(W_dense.dtype() == torch::kFloat16, "W must be float16");
    TORCH_CHECK(A.is_contiguous(),       "A must be contiguous");
    TORCH_CHECK(W_dense.is_contiguous(), "W must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = W_dense.size(0);
    TORCH_CHECK(W_dense.size(1) == K, "W must have shape [N, K]");

    auto C = torch::empty({M, N}, A.options());

    // Degenerate shapes: a zero-sized grid is an invalid launch; K == 0
    // must yield zeros, not uninitialized memory.
    if (M == 0 || N == 0 || K == 0) {
        return (K == 0 && M > 0 && N > 0) ? torch::zeros({M, N}, A.options()) : C;
    }

    const cutlass::half_t* a_ptr =
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr<at::Half>());
    const cutlass::half_t* b_ptr =
        reinterpret_cast<const cutlass::half_t*>(W_dense.data_ptr<at::Half>());
    cutlass::half_t* d_ptr =
        reinterpret_cast<cutlass::half_t*>(C.data_ptr<at::Half>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (!dense_use_streamk()) {
        typename GemmBasic::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
            {M, N, K},                                // problem size
            1,                                        // batch count / split-k slices
            {1.0f, 0.0f},                             // epilogue: alpha, beta
            a_ptr, b_ptr, d_ptr, d_ptr,               // A, B, C (unused), D
            M * K, N * K, M * N, M * N,               // batch strides (irrelevant, batch=1)
            K,                                        // stride_a (row-major [M,K]: ld = K)
            K,                                        // stride_b (col-major [K,N]: ld = K)
            N,                                        // stride_c
            N                                         // stride_d
        );

        GemmBasic gemm_op;
        TORCH_CHECK(gemm_op.can_implement(args) == cutlass::Status::kSuccess,
                    "CUTLASS (basic) cannot implement this problem");
        auto workspace = torch::empty(
            {(int64_t)GemmBasic::get_workspace_size(args)},
            torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
        TORCH_CHECK(gemm_op.initialize(args, workspace.data_ptr(), stream)
                        == cutlass::Status::kSuccess,
                    "CUTLASS (basic) initialize failed");
        TORCH_CHECK(gemm_op(stream) == cutlass::Status::kSuccess,
                    "CUTLASS (basic) run failed");
        return C;
    }

    typename GemmStreamK::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {M, N, K},                                // problem size
        1,                                        // batch count / split-k slices
        {1.0f, 0.0f},                             // epilogue: alpha, beta
        a_ptr, b_ptr, d_ptr, d_ptr,               // A, B, C (unused), D
        M * K, N * K, M * N, M * N,               // batch strides (irrelevant, batch=1)
        K,                                        // stride_a
        K,                                        // stride_b
        N,                                        // stride_c
        N,                                        // stride_d
        device_sm_count()                         // avail_sms (Stream-K scheduler)
    );

    GemmStreamK gemm_op;
    TORCH_CHECK(gemm_op.can_implement(args) == cutlass::Status::kSuccess,
                "CUTLASS (stream-k) cannot implement this problem");
    auto workspace = torch::empty(
        {(int64_t)GemmStreamK::get_workspace_size(args)},
        torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
    TORCH_CHECK(gemm_op.initialize(args, workspace.data_ptr(), stream)
                    == cutlass::Status::kSuccess,
                "CUTLASS (stream-k) initialize failed");
    TORCH_CHECK(gemm_op(stream) == cutlass::Status::kSuccess,
                "CUTLASS (stream-k) run failed");

    return C;
}

#else  // FLUTE_HAVE_CUTLASS

#include <torch/extension.h>

torch::Tensor qgemm_cutlass_dense(
    torch::Tensor /*A*/,
    torch::Tensor /*W_dense*/,
    int64_t /*bitwidth*/,
    int64_t /*group_size*/
) {
    TORCH_CHECK(false,
        "CUTLASS dense kernel not built — rebuild with FLUTE_HAVE_CUTLASS "
        "defined and CUTLASS include path on the build line "
        "(see DEPLOY.md). The production cutlass_streaming kernel does "
        "NOT need CUTLASS and should still work.");
}

#endif  // FLUTE_HAVE_CUTLASS
