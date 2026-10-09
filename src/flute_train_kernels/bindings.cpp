/** src/bindings.cpp — pybind11 bindings for flute_train_kernels

idxN family: every entry takes (bitwidth, indices_layout) with 4-bit
defaults — the 4-bit path is the byte-identical legacy one; bitwidth
1/2/3 run the sub-4-bit kernels. indices_layout, when non-empty, must
equal f"idx{bitwidth}" (the flute_extended agreement-gate convention).
*/
#include <torch/extension.h>
#include <string>

torch::Tensor fused_backward_gemm(
    torch::Tensor grad_y, torch::Tensor indices, torch::Tensor lut,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout);

torch::Tensor backward_simple_twin(
    torch::Tensor grad_y, torch::Tensor indices, torch::Tensor lut,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout);

torch::Tensor lut_grad_scatter(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor indices,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout);

torch::Tensor lut_grad_scatter_twin(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor indices,
    int64_t N, int64_t K, int64_t group_size,
    int64_t bitwidth, std::string indices_layout);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_backward_gemm", &fused_backward_gemm,
          "Fused backward GEMM: grad_X = grad_Y @ dequant(idxN, LUT). "
          "Dispatches on grad_y dtype (fp16/bf16); bitwidth 1-4 (the "
          "idxN family; 4 is the classic nibble path, unchanged).",
          py::arg("grad_y"), py::arg("indices"), py::arg("lut"),
          py::arg("N"), py::arg("K"), py::arg("group_size"),
          py::arg("bitwidth") = 4, py::arg("indices_layout") = "");
    m.def("backward_simple_twin", &backward_simple_twin,
          "G-B2 differential twin: same dequant and mma order as "
          "fused_backward_gemm, but fragments are assembled by scalar "
          "reads instead of ldmatrix (must match bit-exactly).",
          py::arg("grad_y"), py::arg("indices"), py::arg("lut"),
          py::arg("N"), py::arg("K"), py::arg("group_size"),
          py::arg("bitwidth") = 4, py::arg("indices_layout") = "");
    m.def("lut_grad_scatter", &lut_grad_scatter,
          "dL/dLUT (W5, docs/KERNEL_SPEC_DLDLUT.md): the fused "
          "GEMM+scatter — dL/dW = grad_Y^T @ X on Tensor Cores, "
          "scatter-added into [n_groups, 2^bitwidth] fp32 keyed by the "
          "idxN codes, deterministic two-pass. Dispatches on the shared "
          "grad_y/x dtype (fp16/bf16).",
          py::arg("grad_y"), py::arg("x"), py::arg("indices"),
          py::arg("N"), py::arg("K"), py::arg("group_size"),
          py::arg("bitwidth") = 4, py::arg("indices_layout") = "");
    m.def("lut_grad_scatter_twin", &lut_grad_scatter_twin,
          "The G-B5 scalar-fragment twin of lut_grad_scatter (same "
          "staging, mma order and scatter walk; fragments assembled by "
          "scalar reads — must match bit-exactly).",
          py::arg("grad_y"), py::arg("x"), py::arg("indices"),
          py::arg("N"), py::arg("K"), py::arg("group_size"),
          py::arg("bitwidth") = 4, py::arg("indices_layout") = "");
}
