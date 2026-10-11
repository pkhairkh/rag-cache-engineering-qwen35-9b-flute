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
 *     fht_forward_awq(x, signs, s) -> ((x*s) @ T) / s (the AWQ-
 *                                   compensated fold, one kernel)
 *
 *  GEMM entrypoint (src/kernel_streaming.cu):
 *     qgemm_dual_stream(A, indices, lut, bitwidth,
 *                      indices2, lut2, bitwidth2,
 *                      resB, resA, bias, group_size)
 *   — the dual-stream fused decode kernel: ONE launch computes both
 *     streams, the rank-<=16 residual and the bias.
 *
 *  GEMM entrypoint (src/kernel_streaming.cu):
 *     qgemm_gemv_stream(A, indices, lut, bitwidth,
 *                      indices2, lut2, bitwidth2,
 *                      resB, resA, bias, group_size)
 *   — the M=1 decode GEMV: ONE memory-bound launch for the decode shape
 *     (no tensor cores; group_size is runtime, 64..2048).
 *
 *  GEMM entrypoint (src/kernel_streaming.cu):
 *     qgemm_gemv_fht_stream(A, indices, lut, bitwidth,
 *                         indices2, lut2, bitwidth2,
 *                         resB, resA, bias, group_size, signs, s)
 *   — the FHT-fused M=1 decode GEMV: the boundary-fold rotation runs as
 *     the kernel prologue (ONE launch where  needed FHT + GEMV).
 *
 *  GEMM entrypoints (src/kernel_streaming.cu):
 *     qgemm_gemv_splitk_stream(A, indices, lut, bitwidth,
 *                        indices2, lut2, bitwidth2,
 *                        resB, resA, bias, group_size)
 *     qgemm_gemv_splitk_fht_stream(A, indices, lut, bitwidth,
 *                            indices2, lut2, bitwidth2,
 *                            resB, resA, bias, group_size, signs, s)
 *   — the split-K GEMV decode kernels: split-K grid + double-buffered K loop
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

// The single source of truth for every kernel-TU entrypoint (the
// older hand-written forward declarations here were exactly the bug
// class that shipped the `flute::GemvSegTab` namespace error the box
// had to hand-patch in commit 749eef2 — every defining TU includes this
// header, so a signature/namespace mismatch is a compile error in EVERY
// TU at once).
#include "flute/entrypoints.h"

// Entrypoints defined in the vendored TUs (kernel_fht.cu /
// kernel_debug_simple.cu / kernel_cutlass_dense.cu — signatures here).
torch::Tensor qgemm_debug_simple(torch::Tensor A, torch::Tensor indices, torch::Tensor lut,
    int64_t bitwidth, int64_t group_size);

torch::Tensor qgemm_cutlass_dense(torch::Tensor A, torch::Tensor W_dense,
    int64_t bitwidth, int64_t group_size);

// FHT entrypoints (defined in src/kernel_fht.cu).
torch::Tensor fht_forward(torch::Tensor x, torch::Tensor signs);
torch::Tensor fht_backward(torch::Tensor grad_out, torch::Tensor signs);
torch::Tensor fht_inplace(torch::Tensor x, torch::Tensor signs);
// the compensated forward — out = ((x*s) @ T) / s, one kernel.
torch::Tensor fht_forward_awq(torch::Tensor x, torch::Tensor signs,
                              torch::Tensor s);

// Large-K FHT entrypoints (defined in src/kernel_tq.cu — GLOBAL-scope
// plain torch::Tensor signatures, the kernel_fht.cu surface pattern;
// NOT in flute/entrypoints.h because kernel_tq.cu declares them at
// global scope. Contract: include/flute/fht_large.cuh — the multi-block
// butterfly for rows one CTA's shared memory cannot hold; the small-K
// path (kernel_fht.cu, K <= 16384) keeps everything it accepts).
torch::Tensor fht_forward_large(torch::Tensor x, torch::Tensor signs);
torch::Tensor fht_backward_large(torch::Tensor g, torch::Tensor signs);

// Test-only scalar debug twins of the large-K pair (also defined in
// src/kernel_tq.cu -- W20.6.1): host-side plain-C++/ATen assembly of
// the same math, the same stage sequence, the same pairing order --
// the differential reference G20.P2 compares the real kernels against.
// NEVER on the production dispatch path: fht.py routes only to
// fht_forward_large / fht_backward_large (verified -- no fht.py
// reference to the twins exists or is needed); the W20.6.3 test file
// and the G20.P2 gate function are the only callers.
torch::Tensor tq_debug_fht_large_forward(torch::Tensor x, torch::Tensor signs);
torch::Tensor tq_debug_fht_large_backward(torch::Tensor g, torch::Tensor signs);

// W20.7.4 -- the batched TQ encode entry (also defined in src/kernel_tq.cu
// at global scope, this declaration family's pattern): ONE host call over
// a prefill's whole (M, d) fp32 unit matrix -- 7.1's norm chain + 5.2's
// batched large-K butterfly + 7.3's ladder/pack + the zero-norm sweep,
// one stream-ordered kernel chain per prefill (PROPOSAL.md §3.6: "two
// kernel chains per prefill instead of ~4,000 host round trips" -- the
// decode chain is W20.8's tq_dequant_batched). signs are the quantizer's
// (d,) fp32 rotation vector (fht.rotation_signs), passed IN by the
// caller -- never generated here; boundaries_{lo,hi} the codebook device
// mirrors. Returns (codes_lo (M, L_lo) uint8, codes_hi (M, L_hi) uint8,
// norms (M,) fp32); the packed byte layout is fixture-pinned (G20.P1's
// frozen fixtures; G20.P3's kernel-parity gate). The wrapper never
// reads device memory (no D2H).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tq_quant_batched(
    torch::Tensor units, int64_t d, torch::Tensor signs,
    torch::Tensor boundaries_lo, torch::Tensor boundaries_hi,
    int64_t n_lo, int64_t bits_lo, int64_t n_hi, int64_t bits_hi);

// W20.8.1 -- the batched TQ decode entry (also defined in src/kernel_tq.cu
// at global scope, this declaration family's pattern): the fused decode
// mirror of tq_quant_batched -- the bfe unpack + smem LUT gather pass, the
// committed fht_backward_large adjoint (signs BEFORE the butterfly,
// rsqrtf(K) at the final store), the per-unit norm multiply, the single
// fp16/fp32 cast at the final store. ONE host call over a prefill's whole
// code batch (the two-chains-per-prefill design's decode half,
// PROPOSAL.md §3.6); the unpack is pack's EXACT inverse (the
// fixture-pinned little-endian byte layout, turboquant.py's
// pack_bits/unpack_bits canonical -- G20.P1's frozen fixtures, G20.P3's
// kernel-parity gate). Returns the (M, K) reconstruction in `dtype`
// (float32 default; float16 = the single-cast width); signs are the
// quantizer's (K,) fp32 rotation vector, centroids the codebook mirrors;
// the wrapper never reads device memory (no D2H).
torch::Tensor tq_dequant_batched(
    torch::Tensor codes_lo, torch::Tensor codes_hi,
    torch::Tensor norms, torch::Tensor signs,
    torch::Tensor centroids_lo, torch::Tensor centroids_hi,
    int64_t n_lo, int64_t bits_lo, int64_t n_hi, int64_t bits_hi,
    at::ScalarType dtype);

// W20.9.1 -- the QJL chain pair (also defined in src/kernel_tq.cu at
// global scope, this declaration family's pattern): the W9.2 qjl flag's
// kernel legs (turboquant.py:67-111's Alg.-2 contract), flag-off
// complete. tq_qjl_quant computes the residual sketch over the prefill's
// whole batch: r = x minus tq_dequant_batched's fp32 reconstruction of
// the caller's codes (the committed decode entry, launched verbatim),
// gamma = the residual's (M,) fp32 norms (7.1's norm chain), and the
// sketch = sign(FHT(r)) -- the scale invariance
// sign(FHT(r/gamma)) == sign(FHT(r)) means the division never
// materializes (turboquant.py:892-893; PROPOSAL.md:154-155); returns
// (qjl_signs (M, K) int8, gamma (M,) fp32) -- TQCodes' two qjl fields,
// batched. tq_qjl_compensate adds the decode-side Alg.-2 term
// out = g + (QJL_C * gamma[m] / sqrt(K)) * z with z = the sketch's
// fht_backward_large adjoint and QJL_C = sqrt(pi/2) (turboquant.py:268
// / :1193 -- the paper's estimator with the structured-projection
// substitution). signs are the quantizer's FIRST-draw (d,) fp32 rotation
// vector (the residual decode's un-rotation); qjl_rotation is the SECOND
// draw (fht.rotation_signs(d, seed+7777), QJL_SEED_OFFSET at
// turboquant.py:260) -- both passed IN by the caller, never generated
// here. G20.P3's qjl leg (TASKS.md:1431) is the parity gate.
std::tuple<torch::Tensor, torch::Tensor> tq_qjl_quant(
    torch::Tensor x, torch::Tensor signs,
    torch::Tensor codes_lo, torch::Tensor codes_hi, torch::Tensor norms,
    torch::Tensor centroids_lo, torch::Tensor centroids_hi,
    int64_t n_lo, int64_t bits_lo, int64_t n_hi, int64_t bits_hi,
    torch::Tensor qjl_rotation);

torch::Tensor tq_qjl_compensate(
    torch::Tensor g, torch::Tensor qjl_signs, torch::Tensor gamma,
    torch::Tensor qjl_rotation);

// W20.9.2 -- the serving index frame's normalize kernel (also defined in
// src/kernel_tq.cu at global scope, this declaration family's pattern):
// the row L2 norms of an (M, N) fp32/fp16 CUDA matrix as ONE launch --
// grid = M rows x 256 threads, 7.1's per-warp shfl butterfly + the
// fixed-order cross-warp fold + __fsqrt_rn (the same association
// contract at the smaller tile: one CTA per row, no partial axis, no
// second launch, any positive N), the zero row 0.0 branchless. The
// normalize step of the serving index frame's cosine vectors (G20.5's
// INDEX leg, TASKS.md:1443; index.py:29-33's L2-normalized frame, the
// per-row torch.linalg.vector_norm of index.py:790 batched -- the
// W20.10.2 index-frame vectors pass consumes it). Returns the (M,) fp32
// norms on x's device; the wrapper never reads device memory (no D2H).
torch::Tensor tq_l2norm_rows(torch::Tensor x);

// Unified dispatcher. Accepts either (indices, lut) for the quantized path
// or (W_dense) for the dense CUTLASS baseline. Pass an empty Tensor for the
// unused variants. q_layout selects the packed-indices layout:
//   1 = "idx4" (idx4) — register-direct dequant
// We intentionally keep the Python signature permissive; the Python wrapper
// (flute_extended/__init__.py) handles the user-facing API.
torch::Tensor qgemm_per_group_lut(torch::Tensor A,
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

    // the dual-stream fused decode kernel (ONE launch: both streams
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

    // the M=1 decode GEMV — the memory-bound single-launch decode
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

    // the FHT-fused decode GEMV — the same M == 1 contract with the
    // boundary-fold rotation folded into the kernel prologue (ONE launch
    // where  needed an FHT kernel + the GEMV). The Python wrapper
    // (flute_extended/qgemm_gemv_stream's rot_signs/awq_scale parameters)
    // owns the routing gate: fused here, the FHT+GEMV pair otherwise.
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

    // the split-K GEMV decode kernels — split-K grid + double-buffered
    // K loop + the wide table (every (B1, B2) pair, GS 16..2048,
    // residual rank <= 32 — lm_head's (4,4)+r32 and the QKV composite
    // pairs ride it). The Python wrapper (flute_extended/qgemm_gemv_splitk_
    // stream) owns the routing gate and falls back to the plain-GEMV/dual/
    // two-launch ladder on an older extension build.
    m.def("qgemm_gemv_splitk_stream", &qgemm_cutlass_gemv_splitk_stream,
          "Decode split-K GEMV (M=1): split-K + double-buffer, "
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

    m.def("qgemm_gemv_splitk_fht_stream", &qgemm_cutlass_gemv_splitk_fht_stream,
          "FHT-fused decode split-K GEMV (M=1): split-K + double-buffer, "
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
    // : the grouped multi-blob split-K GEMV — the QKV merge.
    // heterogeneous: the segment table [count, 15] carries a
    // per-segment spec (b1, b2, gs, signs, s — the mixed-radix palette's
    // per-tensor allocation); any signs column nonzero selects the
    // FHT-fused path inside the impl. The Python wrapper
    // (flute_extended/qgemm_gemv_multi) owns the mirror gates, the
    // persistent output buffers and the segment-table construction.
    m.def("qgemm_gemv_multi", &qgemm_cutlass_gemv_multi,
          "Grouped decode split-K GEMV (M=1, heterogeneous): 2-4 modules with "
          "PER-SEGMENT bitwidths/group-sizes/rotations, one launch, "
          "per-segment outputs",
          py::arg("A"),
          py::arg("seg_table"));

    // : the merged gate+up split-K GEMV with the SiLU*mul epilogue —
    // heterogeneous (per-blob pair/GS/rotation in the two
    // seg-table rows; the act/mul elementwise kernels disappear).
    m.def("qgemm_gemv_mlp", &qgemm_cutlass_gemv_mlp,
          "Merged gate+up decode split-K GEMV (M=1, heterogeneous): "
          "C = silu(A@Wg^T) * (A@Wu^T), residuals and biases folded, "
          "one launch, one fp16 round",
          py::arg("A"),
          py::arg("seg_table"),
          py::arg("C"));

    m.def("fht_forward", &fht_forward,
          "Fast Hadamard Transform: out = x @ T (sign-scaled, per-segment)",
          py::arg("x"), py::arg("signs"));
    m.def("fht_backward", &fht_backward,
          "FHT adjoint: gx = g @ T^T (input-side signs)",
          py::arg("grad_out"), py::arg("signs"));
    m.def("fht_inplace_", &fht_inplace,
          "In-place FHT: x <- x @ T",
          py::arg("x"), py::arg("signs"));
    // the compensated forward — the legacy rotate-then-AWQ fold's
    // x @ (D T D^-1) = ((x*s) @ T) / s in ONE kernel.
    m.def("fht_forward_awq", &fht_forward_awq,
          "FHT with folded AWQ compensation: out = ((x*s) @ T) / s",
          py::arg("x"), py::arg("signs"), py::arg("s"));

    // Large-K FHT pair (src/kernel_tq.cu, contract: include/flute/
    // fht_large.cuh): the multi-block butterfly for K > 16384.
    // K eligibility (refused loudly inside, "fht_large: " prefix):
    // K > 16384, K % 32 == 0, K a power of two (the 2^15..2^22
    // envelope); 2-D contiguous (M, K) CUDA fp32/fp16/bf16 in and out
    // (identical dtype, dispatch inside), (K,) fp32 signs.
    m.def("fht_forward_large", &fht_forward_large,
          "Large-K FHT (K > 16384, power of two, % 32 == 0): "
          "out = x @ T, out-of-place, fp32 butterfly (contract: "
          "include/flute/fht_large.cuh)",
          py::arg("x"), py::arg("signs"));
    m.def("fht_backward_large", &fht_backward_large,
          "Large-K FHT adjoint (K > 16384, power of two, % 32 == 0): "
          "gx = g @ T^T, out-of-place, signs before the butterfly "
          "(contract: include/flute/fht_large.cuh)",
          py::arg("g"), py::arg("signs"));

    // Test-only scalar twins of the large-K pair (src/kernel_tq.cu,
    // W20.6.1) -- the differential reference for G20.P2. Never on the
    // production path: fht.py never routes here (verified); the
    // W20.6.3 test file and the G20.P2 gate are the only callers.
    m.def("tq_debug_fht_large_forward", &tq_debug_fht_large_forward,
          "test-only scalar twin of fht_forward_large -- the "
          "differential reference for G20.P2 (never on the production "
          "path); same stage sequence and pairing order as the kernel "
          "(low = u + w at the lower index, high = u - w, ascending len_)",
          py::arg("x"), py::arg("signs"));
    m.def("tq_debug_fht_large_backward", &tq_debug_fht_large_backward,
          "test-only scalar twin of fht_backward_large -- the "
          "differential reference for G20.P2 (never on the production "
          "path); signs at the first load, the same butterfly, rsqrtf(K) "
          "at the final store",
          py::arg("g"), py::arg("signs"));

    // W20.7.4 -- the batched TQ encode entry (src/kernel_tq.cu): ONE
    // host call over a prefill's whole unit batch -- the norm chain +
    // the batched large-K butterfly + the ladder/pack + the zero-norm
    // sweep, stream-ordered on the current CUDA stream. The
    // two-chains-per-prefill design (PROPOSAL.md §3.6: "two kernel
    // chains per prefill instead of ~4,000 host round trips"; the
    // decode chain is W20.8's tq_dequant_batched); the packed byte
    // layout is fixture-pinned (turboquant.py's pack_bits canonical,
    // G20.P1's frozen fixtures, G20.P3's kernel-parity gate).
    m.def("tq_quant_batched", &tq_quant_batched,
          "Batched TurboQuant encode: one call over the prefill's (M, d) "
          "fp32 unit matrix -> (codes_lo (M, L_lo) uint8, codes_hi (M, "
          "L_hi) uint8, norms (M,) fp32); signs are the quantizer's (d,) "
          "fp32 rotation vector, boundaries the codebook mirrors; the "
          "wrapper never reads device memory (no D2H)",
          py::arg("units"),
          py::arg("d"),
          py::arg("signs"),
          py::arg("boundaries_lo"),
          py::arg("boundaries_hi"),
          py::arg("n_lo"),
          py::arg("bits_lo"),
          py::arg("n_hi"),
          py::arg("bits_hi"));

    // W20.8.1 -- the batched TQ decode entry (src/kernel_tq.cu): the
    // decode chain's ONE host call -- the bfe unpack + smem LUT gather,
    // the fht_backward_large adjoint (signs BEFORE the butterfly), the
    // per-unit norm multiply, the single cast. The mirror of
    // tq_quant_batched (the two-chains-per-prefill design's other half,
    // PROPOSAL.md §3.6); the packed byte layout is fixture-pinned and the
    // unpack is pack's exact inverse (G20.P1's frozen fixtures; G20.P3's
    // kernel-parity gate counts divergences).
    m.def("tq_dequant_batched", &tq_dequant_batched,
          "Batched TurboQuant decode: one call over the prefill's packed "
          "(M, L_lo)/(M, L_hi) uint8 code rows + the (M,) fp32 norms -> "
          "the (M, K) reconstruction in `dtype` (float32 default; float16 "
          "= the single cast at the final store); signs are the "
          "quantizer's (K,) fp32 rotation vector, centroids the codebook "
          "mirrors; the wrapper never reads device memory (no D2H)",
          py::arg("codes_lo"),
          py::arg("codes_hi"),
          py::arg("norms"),
          py::arg("signs"),
          py::arg("centroids_lo"),
          py::arg("centroids_hi"),
          py::arg("n_lo"),
          py::arg("bits_lo"),
          py::arg("n_hi"),
          py::arg("bits_hi"),
          py::arg("dtype") = at::kFloat);

    // W20.9.1 -- the QJL chain pair (src/kernel_tq.cu): the W9.2 qjl
    // flag's kernel legs, flag-off complete (the runtime keeps --qjl
    // default off; the dispatcher's qjl routing is later work). The
    // scale-invariance identity sign(FHT(r/gamma)) == sign(FHT(r))
    // (turboquant.py:892-893) lets the sketch sign the butterfly's
    // output verbatim; G20.P3's qjl leg (TASKS.md:1431, "the qjl leg
    // completes W20.9") is the parity gate.
    m.def("tq_qjl_quant", &tq_qjl_quant,
          "Batched TurboQuant QJL encode: the residual sketch over the "
          "prefill's (M, d) fp32 unit matrix -> (qjl_signs (M, d) int8, "
          "gamma (M,) fp32); r = x - tq_dequant_batched's fp32 "
          "reconstruction of the caller's codes, gamma = the residual's "
          "row norms, qjl = sign(FHT(r)) -- scale-invariant, the 1/gamma "
          "division never materializes (turboquant.py:892-893); signs are "
          "the quantizer's FIRST-draw (d,) fp32 rotation vector, "
          "qjl_rotation the SECOND draw (fht.rotation_signs(d, "
          "seed+7777)); the wrapper never reads device memory (no D2H)",
          py::arg("x"),
          py::arg("signs"),
          py::arg("codes_lo"),
          py::arg("codes_hi"),
          py::arg("norms"),
          py::arg("centroids_lo"),
          py::arg("centroids_hi"),
          py::arg("n_lo"),
          py::arg("bits_lo"),
          py::arg("n_hi"),
          py::arg("bits_hi"),
          py::arg("qjl_rotation"));

    m.def("tq_qjl_compensate", &tq_qjl_compensate,
          "Batched TurboQuant QJL decode compensation: out = g + "
          "(QJL_C * gamma[m] / sqrt(K)) * fht_adjoint(qjl_signs) with "
          "QJL_C = sqrt(pi/2) (turboquant.py:268 / :1193) -- the paper's "
          "Alg.-2 residual term; g is the (M, K) fp32 MSE "
          "reconstruction (tq_dequant_batched's float32 width -- the "
          "requested output dtype converts at the caller's very end), "
          "qjl_signs the (M, K) int8 sketch, gamma the (M,) fp32 residual "
          "norms, qjl_rotation the SECOND-draw (K,) fp32 rotation vector "
          "(the adjoint's rotation); out-of-place (g is never written); "
          "the wrapper never reads device memory (no D2H)",
          py::arg("g"),
          py::arg("qjl_signs"),
          py::arg("gamma"),
          py::arg("qjl_rotation"));

    // W20.9.2 -- the serving index frame's normalize kernel (src/
    // kernel_tq.cu): the row L2 norms as ONE launch -- 7.1's shfl tree +
    // the fixed-order cross-warp fold per row CTA (256 threads, one CTA
    // per row, any positive N), __fsqrt_rn, the zero row 0.0 branchless;
    // G20.5's INDEX leg (TASKS.md:1443) is the protocol-integrity
    // consumption, W20.9.3's CUDA-gated suite the differential.
    m.def("tq_l2norm_rows", &tq_l2norm_rows,
          "Row L2 norms, one launch: (M, N) fp32/fp16 CUDA matrix -> "
          "(M,) fp32 (per-warp shfl butterfly + fixed-order cross-warp "
          "fold + __fsqrt_rn per row CTA; the zero row is 0.0 "
          "branchless; the serving index frame's normalize step -- "
          "G20.5's INDEX leg, index.py:29-33/790's cosine frame)",
          py::arg("x"));
}
