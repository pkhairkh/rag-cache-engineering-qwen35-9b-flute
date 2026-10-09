"""flute_train_kernels build configuration. Target: SM_86 (A10G).

Include policy: the local include/flute/ tree is self-contained (mma.cuh
copied in, mma_bwd.cuh new), so the build no longer depends on
flute_extended/ being present (spec G-B7). The FLUTE_EXT_DIR fallback
stays for out-of-tree checkouts.
"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJECT_ROOT = Path(__file__).resolve().parent
INCLUDE_DIR = PROJECT_ROOT / "include"

include_dirs = [str(INCLUDE_DIR)]

if not (INCLUDE_DIR / "flute" / "mma.cuh").is_file():
    # Local headers absent (e.g. a partial checkout): fall back to an
    # external flute_extended include tree.
    FLUTE_INCLUDE = PROJECT_ROOT.parent / "flute_extended" / "include"
    if (FLUTE_INCLUDE / "flute" / "mma.cuh").is_file():
        include_dirs.append(str(FLUTE_INCLUDE))
    else:
        env_flute = os.environ.get("FLUTE_EXT_DIR")
        if env_flute and (Path(env_flute) / "include" / "flute" / "mma.cuh").is_file():
            include_dirs.append(str(Path(env_flute) / "include"))
        else:
            print("[flute_train_kernels] WARNING: flute/mma.cuh not found. "
                  "Set FLUTE_EXT_DIR to your flute_extended checkout.")

# No --use_fast_math: it can silently flush denormals and alter NaN
# behavior in fp16 paths (spec CS-1.7). -Xptxas -v is always on so the
# 0-spill gate (G-B7) can read the register/spill transcript from the
# build log; FLUTE_TRAIN_PTXAS_V remains as a no-op compatibility knob.
nvcc_flags = [
    "-O3", "-std=c++20", "--expt-relaxed-constexpr",
    "-Xptxas", "-v",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    f"-I{INCLUDE_DIR}",
]

setup(
    name="flute_train_kernels",
    version="2.0.0",
    description="Fused backward GEMM kernel for QLoRA training (fp16/bf16)",
    packages=["flute_train_kernels"],
    ext_modules=[
        CUDAExtension(
            "flute_train_kernels._C",
            ["src/kernel_backward_gemm.cu",
             "src/kernel_lut_grad.cu",
             "src/bindings.cpp"],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": ["-O3", "-std=c++20"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
