"""
FLUTE-Extended build configuration.

Backends compiled into the extension (the kernel monolith is SPLIT
into per-family translation units — docs/KERNELS.md):
  - debug_simple      : always built (scalar-load differential twin of the
                         streaming kernel; bring-up/debugging only)
  - streaming         : always built (the prefill/streaming GEMM family +
                         the dual-stream fused decode)
  - gemv              : always built (the plain-GEMV M=1 decode GEMV pair)
  - gemv_splitk       : always built (the split-K GEMV)
  - gemv_multi        : always built (the heterogeneous QKV merge)
  - gemv_mlp          : always built (the heterogeneous gate+up
                         merge with the SiLU*mul epilogue)
  - gemv_host         : always built (the shared split-K workspace TU)
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

Target SM (the compile-time repair):
  FLUTE_CUDA_ARCHES drives the -gencode list: comma/space/semicolon
  separated bare arch numbers ("86" | "80,86" | "89,90"). DEFAULT "86" —
  the A10G box. Rationale: the older build compiled FOUR arches
  (80/86/89/90) sequentially inside one nvcc invocation of a 6290-line
  monolith — the box measured >30 min per rebuild. The single-arch
  default plus the TU split (one ninja job per family) cuts a full
  rebuild to minutes; add arches explicitly for portable wheels.
  SM_75 is NOT supported (mma.m16n8k16 requires SM_80+).

Parallel compile : torch's cpp_extension parallelizes with NINJA
across translation units when the `ninja` package is importable — the
build prints a LOUD warning and falls back to the serial distutils path
without it (the box's 30-min serial builds were exactly that).
  MAX_JOBS=8 python setup.py build_ext --inplace   (defaults to the
  CPU count, capped 16).

First build on the A10G: append -Xptxas -v to nvcc_flags temporarily and
check the register count of the streaming kernels (~200-238 expected;
ptxas must report 0 spills — the local compile verification
script, scripts/compile_check.sh, asserts this on every TU).
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
# Source list (the monolith split — one TU per kernel family; every
# family compiles as its own ninja job, and an edit to one family no
# longer recompiles the other four).
# ---------------------------------------------------------------------------
sources = [
    "src/kernel_debug_simple.cu",
    "src/kernel_streaming.cu",        # prefill/streaming +  dual (split)
    "src/kernel_gemv.cu",         # plain-GEMV decode GEMV (split)
    "src/kernel_gemv_splitk.cu",            #  split-K GEMV (split)
    "src/kernel_gemv_multi.cu",      #  heterogeneous QKV merge
    "src/kernel_gemv_mlp.cu",        #  heterogeneous MLP merge
    "src/gemv_host.cpp",             # the shared split-K workspace TU
    "src/kernel_cutlass_dense.cu",   # compiles to a stub when FLUTE_HAVE_CUTLASS is not defined
    "src/kernel_fht.cu",              # FLUTE Extension: Fast Hadamard Transform
    "src/bindings.cpp",
]

# ---------------------------------------------------------------------------
# The parallel-compile preflight : without ninja, torch's
# BuildExtension falls back to the SERIAL distutils path — one .cu at a
# time, which costs the whole point of the TU split. Loud, not silent.
# ---------------------------------------------------------------------------
try:
    import ninja  # noqa: F401
    _HAVE_NINJA = True
except ImportError:
    _HAVE_NINJA = False

if not _HAVE_NINJA:
    print("[flute_extended] *** WARNING: the 'ninja' build tool is NOT "
          "importable — torch's extension build falls back to the SERIAL "
          "distutils path (one .cu at a time; the kernel families are six "
          "separate TUs, so this costs the whole point of the "
          "split). Install it and rebuild:  pip install ninja", flush=True)

if "MAX_JOBS" not in os.environ:
    os.environ.setdefault("MAX_JOBS", str(min(16, os.cpu_count() or 4)))

# ---------------------------------------------------------------------------
# Architecture flags : FLUTE_CUDA_ARCHES — default sm_86 (the A10G
# box; the older four-arch list compiled 4x the kernels for machines
# nobody runs).
# ---------------------------------------------------------------------------
def _parse_arches():
    raw = os.environ.get("FLUTE_CUDA_ARCHES", "86")
    arches = []
    for tok in raw.replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        arch = int(tok)
        if arch < 80:
            raise ValueError(f"FLUTE_CUDA_ARCHES: SM_{arch} is not supported "
                "(mma.m16n8k16 requires SM_80+)")
        if arch > 90:
            raise ValueError(f"FLUTE_CUDA_ARCHES: SM_{arch} exceeds the compiled table "
                "(80..90)")
        arches.append(arch)
    return arches or [86]

ARCHES = _parse_arches()

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
    # Template-noise hygiene: the kernel TUs instantiate shared config
    # structs (TileConfig / SubCfg / DualCfg) per code path, and each
    # instantiation legitimately references only a subset of the
    # constexpr members — nvcc's cudafe front end reports every
    # unreferenced member as warning #177 ("declared but never
    # referenced"), ~110 times per otherwise-clean build. The members
    # are alive across instantiations; the warning is per-instantiation
    # noise. Caveat: #177 also covers dead LOCAL variables in .cu TUs
    # (the 2026-10-09 build's one instance was deleted at the source);
    # that small class is on review hygiene. Everything else — division
    # by zero, ptxas spills, host-TU warnings — stays visible.
    # ---------------------------------------------------------------------
    "-diag-suppress", "177",
    # ---------------------------------------------------------------------
    # Architecture flags : FLUTE_CUDA_ARCHES, default sm_86 (the
    # A10G). See the module docstring — the older 80/86/89/90 list
    # compiled 4 arches sequentially per TU.
    # ---------------------------------------------------------------------
]
for _arch in ARCHES:
    nvcc_flags.append(f"-gencode=arch=compute_{_arch},code=sm_{_arch}")
nvcc_flags += [
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
setup(name="flute_extended",
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
