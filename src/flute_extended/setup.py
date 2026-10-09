"""
FLUTE-Extended build configuration.

Backends compiled into the extension:
  - debug_simple      : always built (scalar-load differential twin of the
                         streaming kernel; bring-up/debugging only)
  - cutlass_streaming : always built (raw mma.m16n8k16 PTX + cp.async +
                         prmt dequant merge, double-buffered pipeline)
  - fht               : always built (Fast Walsh-Hadamard Transform — the
                         Hadamard boundary-fold rotation replacement; smem
                         butterfly, fp32 accumulation, block-diagonal K;
                         see include/flute/fht.cuh)

Usage:
  python setup.py build_ext --inplace         # build
  pip install -e .                            # editable install

Target SM:
  SM_86 (A10G / RTX 3090 / A6000, Ampere GA102) is the primary target.
  Binaries also include SM_80/89/90 for forward/backward compatibility.
  SM_75 is NOT supported (mma.m16n8k16 requires SM_80+).

First build on the A10G: append -Xptxas -v to nvcc_flags temporarily and
check the register count of flute_kernel_streaming (~200 expected; if
ptxas reports spills, see docs/DEPLOY.md).
"""

from pathlib import Path

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Absolute path to THIS setup.py's directory. nvcc is invoked by PyTorch's
# cpp_extension from a scratch build directory, so relative include paths
# like "-Iinclude" do not resolve. Always use absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parent
INCLUDE_DIR = PROJECT_ROOT / "include"


# ---------------------------------------------------------------------------
# Source list — always compile every .cu/.cpp.
# ---------------------------------------------------------------------------
sources = [
    "src/kernel_debug_simple.cu",
    "src/kernel_cutlass_streaming.cu",
    "src/kernel_fht.cu",           # FLUTE Extension: Fast Hadamard Transform
    "src/bindings.cpp",
]

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    # PyTorch 2.x requires C++20 ("#error C++20 or later compatible compiler
    # is required to use PyTorch" with c++17).
    "-std=c++20",
    "--expt-relaxed-constexpr",
    # ---------------------------------------------------------------------
    # Architecture flags. Primary target: SM_86 (Ampere GA102 — A10G /
    # RTX 3090 / A6000). SM_75 is excluded: per the PTX ISA target notes,
    # ".f16 floating point type mma operation with the .m16n8k16 shape
    # requires sm_80 or higher", so an sm_75 target fails in ptxas with
    # "Unexpected instruction types specified for 'mma'".
    # ---------------------------------------------------------------------
    "-gencode=arch=compute_80,code=sm_80",   # Ampere A100
    "-gencode=arch=compute_86,code=sm_86",   # Ampere GA102  <-- primary (A10G)
    "-gencode=arch=compute_89,code=sm_89",   # Ada Lovelace
    "-gencode=arch=compute_90,code=sm_90",   # Hopper (mma.m16n8k16 still supported)
    # Project-local include dir (for flute/*.cuh headers) — ABSOLUTE path,
    # required because cpp_extension runs nvcc from a scratch build dir.
    f"-I{INCLUDE_DIR}",
]

cxx_flags = [
    "-O3",
    # PyTorch 2.x requires C++20 (kept in sync with nvcc_flags).
    "-std=c++20",
    f"-I{INCLUDE_DIR}",
]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="flute_extended",
    version="1.0.0",
    description="Quantized GEMM with per-group LUT and Tensor Core acceleration",
    packages=["flute_extended"],
    ext_modules=[
        CUDAExtension(
            "flute_extended._C",
            sources,
            # Absolute include dir so "flute/mma.cuh" / "flute/dequant.cuh"
            # resolve regardless of the nvcc/c++ working directory.
            include_dirs=[str(INCLUDE_DIR)],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy"],
)
