/**
 * src/bindings.cpp
 *
 * Unified PyTorch bindings for all GEMM backends + the FHT kernels.
 *
 * GEMM entrypoint:
 *     qgemm_per_group_lut(A, indices, lut, bitwidth, group_size, backend=...)
 *
 * FHT entrypoints (src/kernel_fht.cu, the Hadamard boundary-fold
 * replacement — see include/flute/fht.cuh):
 *     fht_forward(x, signs)    -> x @ T   (out-of-place)
 *     fht_backward(g, signs)   -> g @ T^T (autograd adjoint)
 *     fht_inplace_(x, signs)   -> x <- x @ T (in-place variant)
 *     fht_forward_awq(x, signs, s) -> ((x*s) @ T) / s (W26: the AWQ-
 *                                   compensated fold, one kernel)
 *
 * W26 GEMM entrypoint (src/kernel_cutlass_streaming.cu):
 *     qgemm_dual_stream(A, indices, lut, bitwidth,
 *                      indices2, lut2, bitwidth2,
 *                      resB, resA, bias, group_size)
 *   — the dual-stream fused decode kernel: ONE launch computes both
 *     streams, the rank-<=16 residual and the bias.
 *
 * W27 GEMM entrypoint (src/kernel_cutlass_streaming.cu):
 *     qgemm_gemv_stream(A, indices, lut, bitwidth,
 *                      indices2, lut2, bitwidth2,
 *                      resB, resA, bias, group_size)
 *   — the M=1 decode GEMV: ONE memory-bound launch for the decode shape
 *     (no tensor cores; group_size is runtime, 64..2048).
 *
 * W28 GEMM entrypoint (src/kernel_cutlass_streaming.cu):
 *     qgemm_gemv_fht_stream(A, indices, lut, bitwidth,
 *                         indices2, lut2, bitwidth2,
 *                         resB, resA, bias, group_size, signs, s)
 *   — the FHT-fused M=1 decode GEMV: the boundary-fold rotation runs as
 *     the kernel prologue (ONE launch where W27 needed FHT + GEMV).
 *
 * W29 GEMM entrypoints (src/kernel_cutlass_streaming.cu):
 *     qgemm_gemv2_stream(A, indices, lut, bitwidth,
 *                        indices2, lut2, bitwidth2,
 *                        resB, resA, bias, group_size)
 *     qgemm_gemv2_fht_stream(A, indices, lut, bitwidth,
 *                            indices2, lut2, bitwidth2,
 *                            resB, resA, bias, group_size, signs, s)
 *   — the GEMV v2 decode kernels: split-K grid + double-buffered K loop
 *     + the wide table (every (B1, B2) pair, GS 16..2048, residual rank
 *     <= 32; the plain entry takes the rotated row, the fused entry
 *     runs the boundary fold as the split-CTA prologue).
 *
 * backend is one of {"debug_simple", "cutlass_dense", "cutlass_streaming", "auto"}
 *
 * Notes:
 *   - "debug_simple"     : always available; scalar-load differential twin
 *                         of cutlass_streaming (bring-up/debugging only)
 *   - "cutlass_dense"    : requires FLUTE_HAVE_CUTLASS; takes a DENSE FP16 W
 *                         instead of indices+LUT (baseline, benchmarking)
 *   - "cutlass_streaming": always available (raw mma.m16n8k16 PTX, no
 *                         CUTLASS dependency); production kernel
 *
 * "auto" picks the best available backend for the given inputs:
 *   - If indices+LUT are provided  → cutlass_streaming
 *   - If a dense W is provided      → cutlass_dense
 */

#include <torch/extension.h>
#include <string>

// Forward declarations of the per-backend entrypoints (defined in other TUs).
torch::Tensor qgemm_debug_simple(
    torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth, int64_t group_size);

torch::Tensor qgemm_cutlass_dense(
    torch::Tensor A, torch::Tensor W_dense,
    int64_t bitwidth, int64_t group_size);

torch::Tensor qgemm_cutlass_streaming(
    torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth, int64_t group_size, int64_t q_layout);

// W26: the dual-stream fused decode kernel (src/kernel_cutlass_streaming.cu).
// One launch: C = A@W1^T (+ A@W2^T) + (A@resB^T)@resA^T + bias. Empty
// indices2/lut2 selects the single-stream rider (bitwidth2 = 0); empty
// resB/resA/bias disable those folds.
torch::Tensor qgemm_cutlass_dual_stream(
    torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2, torch::Tensor lut2, int64_t bitwidth2,
    torch::Tensor resB, torch::Tensor resA, torch::Tensor bias,
    int64_t group_size);

// W27: the M=1 decode GEMV (src/kernel_cutlass_streaming.cu). Same
// tensor contract as qgemm_dual_stream with M pinned to 1 — ONE
// memory-bound launch (no tensor cores); group_size is a RUNTIME
// argument here (64..2048, no GS template dimension).
torch::Tensor qgemm_cutlass_gemv_stream(
    torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2, torch::Tensor lut2, int64_t bitwidth2,
    torch::Tensor resB, torch::Tensor resA, torch::Tensor bias,
    int64_t group_size);

// W28: the FHT-fused M=1 decode GEMV (src/kernel_cutlass_streaming.cu).
// Same contract as qgemm_cutlass_gemv_stream plus the boundary-fold
// rotation INSIDE the kernel: A is the UNROTATED [1, K] fp16 row, signs
// the (K,) fp32 fold signs, s the (K,) fp32 AWQ scales (empty tensor =
// the plain rotation — ONE launch replaces the FHT + GEMV pair;
// bit-identical numerics to the kernels it replaces).
torch::Tensor qgemm_cutlass_gemv_fht_stream(
    torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2, torch::Tensor lut2, int64_t bitwidth2,
    torch::Tensor resB, torch::Tensor resA, torch::Tensor bias,
    int64_t group_size,
    torch::Tensor signs, torch::Tensor s);

// W29: the GEMV v2 decode kernels (src/kernel_cutlass_streaming.cu) —
// the split-K grid + double-buffered K loop + the WIDE table (every
// (B1, B2) pair, GS 16..2048, residual rank <= 32). The plain entry
// takes the ROTATED row (the W27 contract); the FHT-fused entry takes
// the UNROTATED row + signs (+ the AWQ s) and runs the boundary fold as
// the split-CTA prologue (the W28 contract on the W29 kernel).
torch::Tensor qgemm_cutlass_gemv2_stream(
    torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2, torch::Tensor lut2, int64_t bitwidth2,
    torch::Tensor resB, torch::Tensor resA, torch::Tensor bias,
    int64_t group_size);

torch::Tensor qgemm_cutlass_gemv2_fht_stream(
    torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth,
    torch::Tensor indices2, torch::Tensor lut2, int64_t bitwidth2,
    torch::Tensor resB, torch::Tensor resA, torch::Tensor bias,
    int64_t group_size,
    torch::Tensor signs, torch::Tensor s);

// FHT entrypoints (defined in src/kernel_fht.cu).
torch::Tensor fht_forward(torch::Tensor x, torch::Tensor signs);
torch::Tensor fht_backward(torch::Tensor grad_out, torch::Tensor signs);
torch::Tensor fht_inplace(torch::Tensor x, torch::Tensor signs);
// W26: the compensated forward — out = ((x*s) @ T) / s, one kernel.
torch::Tensor fht_forward_awq(torch::Tensor x, torch::Tensor signs,
                              torch::Tensor s);

// Unified dispatcher. Accepts either (indices, lut) for the quantized path
// or (W_dense) for the dense CUTLASS baseline. Pass an empty Tensor for the
// unused variants. q_layout selects the packed-indices layout:
//   1 = "fragment_v1" (idx4) — register-direct dequant
// We intentionally keep the Python signature permissive; the Python wrapper
// (flute_extended/__init__.py) handles the user-facing API.
torch::Tensor qgemm_per_group_lut(
    torch::Tensor A,
    torch::Tensor indices,
    torch::Tensor lut,
    torch::Tensor W_dense,        // empty for quantized backends
    int64_t bitwidth,
    int64_t group_size,
    std::string backend,
    int64_t q_layout
) {
    if (backend == "auto") {
        if (W_dense.numel() > 0) backend = "cutlass_dense";
        else                      backend = "cutlass_streaming";
    }

    if (backend == "debug_simple") {
        return qgemm_debug_simple(A, indices, lut, bitwidth, group_size);
    } else if (backend == "cutlass_dense") {
        return qgemm_cutlass_dense(A, W_dense, bitwidth, group_size);
    } else if (backend == "cutlass_streaming") {
        return qgemm_cutlass_streaming(A, indices, lut, bitwidth, group_size,
                                       q_layout);
    }

    TORCH_CHECK(false, "Unknown backend: '", backend, "'. "
                "Expected one of: debug_simple, cutlass_dense, "
                "cutlass_streaming, auto.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qgemm_per_group_lut", &qgemm_per_group_lut,
          "Quantized GEMM with per-group LUT (multi-backend)",
          py::arg("A"),
          py::arg("indices"),
          py::arg("lut"),
          py::arg("W_dense"),
          py::arg("bitwidth"),
          py::arg("group_size"),
          py::arg("backend") = "auto",
          py::arg("q_layout") = 1);

    // Direct per-backend entrypoints (useful for benchmarking)
    m.def("qgemm_debug_simple",     &qgemm_debug_simple);
    m.def("qgemm_cutlass_dense",    &qgemm_cutlass_dense);
    m.def("qgemm_cutlass_streaming", &qgemm_cutlass_streaming,
          py::arg("A"),
          py::arg("indices"),
          py::arg("lut"),
          py::arg("bitwidth"),
          py::arg("group_size"),
          py::arg("q_layout") = 1);

    // W26: the dual-stream fused decode kernel (ONE launch: both streams
    // + rank-16 residual + bias). The Python wrapper
    // (flute_extended/qgemm_dual_stream) owns the deployment gate and the
    // two-launch fallback for unsupported shapes.
    m.def("qgemm_dual_stream", &qgemm_cutlass_dual_stream,
          "Dual-stream fused quantized GEMM: "
          "C = A@W1^T (+ A@W2^T) + (A@resB^T)@resA^T + bias, one launch",
          py::arg("A"),
          py::arg("indices"),
          py::arg("lut"),
          py::arg("bitwidth"),
          py::arg("indices2"),
          py::arg("lut2"),
          py::arg("bitwidth2") = 0,
          py::arg("resB"),
          py::arg("resA"),
          py::arg("bias"),
          py::arg("group_size") = 512);

    // W27: the M=1 decode GEMV — the memory-bound single-launch decode
    // path (no tensor cores: at M == 1 an m16n8k16 atom does 16-32x the
    // necessary row work). The Python wrapper
    // (flute_extended/qgemm_gemv_stream) owns the routing gate: M == 1
    // here, M 2..16 through qgemm_dual_stream, the rest two-launch.
    m.def("qgemm_gemv_stream", &qgemm_cutlass_gemv_stream,
          "Decode GEMV (M=1): "
          "C = A@W1^T (+ A@W2^T) + (A@resB^T)@resA^T + bias, one "
          "memory-bound launch",
          py::arg("A"),
          py::arg("indices"),
          py::arg("lut"),
          py::arg("bitwidth"),
          py::arg("indices2"),
          py::arg("lut2"),
          py::arg("bitwidth2") = 0,
          py::arg("resB"),
          py::arg("resA"),
          py::arg("bias"),
          py::arg("group_size") = 512);

    // W28: the FHT-fused decode GEMV — the same M == 1 contract with the
    // boundary-fold rotation folded into the kernel prologue (ONE launch
    // where W27 needed an FHT kernel + the GEMV). The Python wrapper
    // (flute_extended/qgemm_gemv_stream's rot_signs/awq_scale parameters)
    // owns the routing gate: fused here, the W27 FHT+GEMV pair otherwise.
    m.def("qgemm_gemv_fht_stream", &qgemm_cutlass_gemv_fht_stream,
          "FHT-fused decode GEMV (M=1): "
          "C = fht(A)@W1^T (+ fht(A)@W2^T) + (fht(A)@resB^T)@resA^T + "
          "bias, one launch (A is the UNROTATED input; fht = the plain "
          "or AWQ-compensated boundary fold)",
          py::arg("A"),
          py::arg("indices"),
          py::arg("lut"),
          py::arg("bitwidth"),
          py::arg("indices2"),
          py::arg("lut2"),
          py::arg("bitwidth2") = 0,
          py::arg("resB"),
          py::arg("resA"),
          py::arg("bias"),
          py::arg("group_size") = 512,
          py::arg("signs"),
          py::arg("s"));

    // W29: the GEMV v2 decode kernels — split-K grid + double-buffered
    // K loop + the wide table (every (B1, B2) pair, GS 16..2048,
    // residual rank <= 32 — lm_head's (4,4)+r32 and the QKV composite
    // pairs ride it). The Python wrapper (flute_extended/qgemm_gemv2_
    // stream) owns the routing gate and falls back to the W27/W28/dual/
    // two-launch ladder on a pre-W29 extension build.
    m.def("qgemm_gemv2_stream", &qgemm_cutlass_gemv2_stream,
          "Decode GEMV v2 (M=1): split-K + double-buffer, "
          "C = A@W1^T (+ A@W2^T) + (A@resB^T)@resA^T + bias (A rotated)",
          py::arg("A"),
          py::arg("indices"),
          py::arg("lut"),
          py::arg("bitwidth"),
          py::arg("indices2"),
          py::arg("lut2"),
          py::arg("bitwidth2") = 0,
          py::arg("resB"),
          py::arg("resA"),
          py::arg("bias"),
          py::arg("group_size") = 512);

    m.def("qgemm_gemv2_fht_stream", &qgemm_cutlass_gemv2_fht_stream,
          "FHT-fused decode GEMV v2 (M=1): split-K + double-buffer, "
          "C = fht(A)@W1^T (+ fht(A)@W2^T) + (fht(A)@resB^T)@resA^T + "
          "bias (A is the UNROTATED input)",
          py::arg("A"),
          py::arg("indices"),
          py::arg("lut"),
          py::arg("bitwidth"),
          py::arg("indices2"),
          py::arg("lut2"),
          py::arg("bitwidth2") = 0,
          py::arg("resB"),
          py::arg("resA"),
          py::arg("bias"),
          py::arg("group_size") = 512,
          py::arg("signs"),
          py::arg("s"));

    // Fast Hadamard Transform (the boundary-fold rotation replacement).
    // The Python wrapper (flute_extended/fht.py at the project root, and
    // scripts/palettized_modules.py) owns shape flattening, dtype
    // validation, autograd and the reference fallback; these bindings are
    // the raw 2-D (M, K) CUDA ops.
    m.def("fht_forward", &fht_forward,
          "Fast Hadamard Transform: out = x @ T (sign-scaled, per-segment)",
          py::arg("x"), py::arg("signs"));
    m.def("fht_backward", &fht_backward,
          "FHT adjoint: gx = g @ T^T (input-side signs)",
          py::arg("grad_out"), py::arg("signs"));
    m.def("fht_inplace_", &fht_inplace,
          "In-place FHT: x <- x @ T",
          py::arg("x"), py::arg("signs"));
    // W26: the compensated forward — the legacy rotate-then-AWQ fold's
    // x @ (D T D^-1) = ((x*s) @ T) / s in ONE kernel.
    m.def("fht_forward_awq", &fht_forward_awq,
          "FHT with folded AWQ compensation: out = ((x*s) @ T) / s",
          py::arg("x"), py::arg("signs"), py::arg("s"));
}
