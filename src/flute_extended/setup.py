"""
FLUTE-Extended build configuration.

Backends compiled into the extension:
  - debug_simple      : always built (scalar-load differential twin of the
                         streaming kernel; bring-up/debugging only)
  - cutlass_streaming : always built (raw mma.m16n8k16 PTX + cp.async +
                         prmt dequant merge, double-buffered pipeline)
  - cutlass_dense     : active if CUTLASS is found (dense FP16 baseline,
                         device::GemmUniversal classic API, ArchTag=Sm80
                         policy tag, all-explicit template args)
  - fht               : always built (Fast Walsh-Hadamard Transform — the
                         Hadamard boundary-fold rotation replacement; smem
                         butterfly, fp32 accumulation, block-diagonal K;
                         see include/flute/fht.cuh)

CUTLASS discovery order:
  1. $FLUTE_CUTLASS_HOME
  2. /home/ubuntu/cutlass
  3. $CUTLASS_HOME
  4. /opt/cutlass

Usage:
  python setup.py build_ext --inplace         # build
  pip install -e .                            # editable install
  python test_flute.py                         # correctness suite
  python benchmark_kernel.py                  # L2-flush benchmark

Target SM:
  SM_86 (A10G / RTX 3090 / A6000, Ampere GA102) is the primary target.
  Binaries also include SM_80/89/90 for forward/backward compatibility.
  SM_75 is NOT supported (mma.m16n8k16 requires SM_80+).

First build on the A10G: append -Xptxas -v to nvcc_flags temporarily and
check the register count of flute_kernel_streaming (~200 expected; if
ptxas reports spills, see docs/DEPLOY.md).
"""

import os
from pathlib import Path

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Absolute path to THIS setup.py's directory. nvcc is invoked by PyTorch's
# cpp_extension from a scratch build directory, so relative include paths
# like "-Iinclude" do not resolve. Always use absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parent
INCLUDE_DIR = PROJECT_ROOT / "include"


# ---------------------------------------------------------------------------
# CUTLASS discovery
# ---------------------------------------------------------------------------
def find_cutlass() -> Path | None:
    candidates = []
    env = os.environ.get("FLUTE_CUTLASS_HOME")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/home/ubuntu/cutlass"))
    env2 = os.environ.get("CUTLASS_HOME")
    if env2:
        candidates.append(Path(env2))
    candidates.append(Path("/opt/cutlass"))

    for p in candidates:
        if (p / "include" / "cutlass" / "cutlass.h").is_file():
            return p
    return None


CUTLASS_DIR = find_cutlass()
print(f"[flute_extended] CUTLASS: {CUTLASS_DIR or 'NOT FOUND (cutlass_dense backend disabled)'}")
if CUTLASS_DIR is not None:
    print("[flute_extended] cutlass_dense backend: ENABLED "
          "(classic GemmUniversal API, ArchTag=Sm80 policy tag, all-explicit "
          "template args — CUTLASS 2.x through 4.8+)")


# ---------------------------------------------------------------------------
# Source list — always compile every .cu/.cpp; dense kernel self-guards with
# #if defined(FLUTE_HAVE_CUTLASS) and emits a stub when the macro is absent.
# ---------------------------------------------------------------------------
sources = [
    "src/kernel_debug_simple.cu",
    "src/kernel_cutlass_streaming.cu",
    "src/kernel_cutlass_dense.cu",  # Compiles to stub when FLUTE_HAVE_CUTLASS not defined
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

if CUTLASS_DIR is not None:
    cutlass_includes = [
        str(CUTLASS_DIR / "include"),
        str(CUTLASS_DIR / "tools" / "util" / "include"),
    ]
    nvcc_flags += [f"-I{p}" for p in cutlass_includes]
    cxx_flags  += [f"-I{p}" for p in cutlass_includes]
    # Defined globally; only the dense kernel checks it. The dense TU uses
    # the classic device::GemmUniversal API with every template argument
    # explicit (ArchTag=Sm80 policy tag), which compiles against CUTLASS
    # 2.x through 4.8+ alike — no version pinning required.
    nvcc_flags.append("-DFLUTE_HAVE_CUTLASS=1")
    cxx_flags.append("-DFLUTE_HAVE_CUTLASS=1")


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
