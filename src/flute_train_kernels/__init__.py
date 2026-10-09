"""flute_train_kernels — fused backward GEMM + dL/dLUT for QLoRA training.

Computes grad_X = grad_Y @ dequant(idxN, LUT) in one fused CUDA pass, and
(optionally, via lut_grad_scatter) dL/dLUT. W is dequantized on-the-fly
into shared memory, fed to Tensor Cores via mma.sync.m16n8k16 (FP32
accumulate; fp16 and bf16 activations). The validation gates are the
differential tests in tests/ (test_lut_gradients.py).

idxN family (DEQUANT_SPEC section 8): every entry accepts
(bitwidth, indices_layout) with 4-bit defaults. bitwidth 1/2/3 run the
sub-4-bit kernels over the unified idxN blob
(flute_extended/flute_extended/idxN.py — pack_idxn(idx, bits));
bitwidth 4 is the byte-identical legacy nibble path. The blob contract
per width: N*K*bitwidth/8 bytes, LUT [ceil(N/group_size), 2^bitwidth]
fp16. indices_layout, when given, must equal f"idx{bitwidth}" — the
flute_extended agreement-gate convention (a mismatch refuses loudly,
never a silent remap).

Build:
    cd flute_train_kernels && python setup.py build_ext --inplace
"""
from __future__ import annotations
import os, sys, importlib.util, glob
from typing import Optional
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHED_EXT = None
_CACHED_ERR = None

_BITS_SUPPORTED = (1, 2, 3, 4)


def _check_bitwidth(bitwidth) -> int:
    """Validate and normalize a bit width of the idxN family."""
    try:
        b = int(bitwidth)
    except (TypeError, ValueError) as e:
        raise ValueError(f"bitwidth must be one of {list(_BITS_SUPPORTED)}, "
                         f"got {bitwidth!r}") from e
    if b not in _BITS_SUPPORTED:
        raise ValueError(f"bitwidth must be one of {list(_BITS_SUPPORTED)}, "
                         f"got {bitwidth!r}")
    return b


def _check_layout(bitwidth: int, indices_layout) -> str:
    """Layout agreement gate: empty means unspecified (trusted caller —
    the kernels re-validate the blob byte count and LUT width); a given
    name must equal f"idx{bitwidth}"."""
    if indices_layout is None:
        return ""
    layout = str(indices_layout)
    expected = f"idx{bitwidth}"
    if layout != expected:
        raise ValueError(
            f"indices_layout {layout!r} must match bitwidth {bitwidth} "
            f"(expected {expected!r})")
    return layout


def _load():
    global _CACHED_EXT, _CACHED_ERR
    if _CACHED_EXT is not None: return _CACHED_EXT
    if _CACHED_ERR is not None: return None
    try:
        from . import _C
        _CACHED_EXT = _C
        return _C
    except ImportError as e:
        _CACHED_ERR = str(e)
    for so_path in sorted(glob.glob(os.path.join(_HERE, "_C*.so"))):
        try:
            spec = importlib.util.spec_from_file_location("flute_train_kernels._C", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _CACHED_EXT = mod
            return _C
        except Exception as e:
            _CACHED_ERR = str(e)
    return None

def available() -> bool:
    return _load() is not None and torch.cuda.is_available()

def import_error() -> Optional[str]:
    if _CACHED_ERR is not None: return _CACHED_ERR
    if _load() is None:
        return "extension not built; run: cd flute_train_kernels && python setup.py build_ext --inplace"
    return None

def fused_backward_gemm(grad_y, indices, lut, N, K, group_size,
                        bitwidth=4, indices_layout=None):
    """Compute grad_X = grad_Y @ dequant(idxN, LUT) in one fused GPU pass.

    Dispatches on grad_y.dtype (fp16 or bf16); the LUT stays fp16 per the
    artifact contract. bitwidth 1/2/3 select the sub-4-bit kernels over
    the unified idxN blob (N*K*bitwidth/8 bytes, LUT [N/GS, 2^bitwidth]);
    4 (the default) is the classic nibble path, unchanged.
    """
    ext = _load()
    if ext is None:
        raise RuntimeError(f"flute_train_kernels not built: {import_error()}")
    b = _check_bitwidth(bitwidth)
    layout = _check_layout(b, indices_layout)
    return ext.fused_backward_gemm(grad_y, indices, lut, int(N), int(K),
                                   int(group_size), b, layout)

def backward_simple_twin(grad_y, indices, lut, N, K, group_size,
                         bitwidth=4, indices_layout=None):
    """G-B2 differential twin: scalar fragment assembly, no ldmatrix.

    Must agree with fused_backward_gemm bit-exactly; a mismatch isolates
    the ldmatrix+swizzle fragment path from the staging path. Carries
    the same bitwidth/indices_layout contract as the production entry.
    """
    ext = _load()
    if ext is None:
        raise RuntimeError(f"flute_train_kernels not built: {import_error()}")
    if not hasattr(ext, "backward_simple_twin"):
        raise RuntimeError("backward_simple_twin missing from the built extension; rebuild")
    b = _check_bitwidth(bitwidth)
    layout = _check_layout(b, indices_layout)
    return ext.backward_simple_twin(grad_y, indices, lut, int(N), int(K),
                                    int(group_size), b, layout)

def lut_grad_scatter(grad_y, x, indices, N, K, group_size,
                     bitwidth=4, indices_layout=None):
    """Compute dL/dLUT (W5, docs/KERNEL_SPEC_DLDLUT.md) in one fused
    GPU pass: the dW GEMM on Tensor Cores plus the code-keyed
    scatter-add into [n_groups, 2^bitwidth] fp32. Deterministic
    two-pass.

    Dispatches on the shared grad_y/x dtype (fp16 or bf16); the LUT is
    not an input (this kernel never reads it). bitwidth 1/2/3 select the
    sub-4-bit kernels over the unified idxN blob.
    """
    ext = _load()
    if ext is None:
        raise RuntimeError(f"flute_train_kernels not built: {import_error()}")
    if not hasattr(ext, "lut_grad_scatter"):
        raise RuntimeError("lut_grad_scatter missing from the built "
                           "extension; rebuild (W5 kernel not in this _C)")
    b = _check_bitwidth(bitwidth)
    layout = _check_layout(b, indices_layout)
    return ext.lut_grad_scatter(grad_y, x, indices, int(N), int(K),
                                int(group_size), b, layout)

def lut_grad_scatter_twin(grad_y, x, indices, N, K, group_size,
                          bitwidth=4, indices_layout=None):
    """The scalar-fragment differential twin of lut_grad_scatter (the
    G-B5 bit-exactness pair; same staging/mma/scatter order, fragments
    assembled by scalar reads instead of ldmatrix).
    """
    ext = _load()
    if ext is None:
        raise RuntimeError(f"flute_train_kernels not built: {import_error()}")
    if not hasattr(ext, "lut_grad_scatter_twin"):
        raise RuntimeError("lut_grad_scatter_twin missing from the built "
                           "extension; rebuild (W5 kernel not in this _C)")
    b = _check_bitwidth(bitwidth)
    layout = _check_layout(b, indices_layout)
    return ext.lut_grad_scatter_twin(grad_y, x, indices, int(N), int(K),
                                     int(group_size), b, layout)

def lut_grad_available() -> bool:
    """True when the built extension carries the W5 scatter kernel."""
    ext = _load()
    return ext is not None and hasattr(ext, "lut_grad_scatter")


def idxn_available() -> bool:
    """True when the built extension carries the idxN parameters (the
    bitwidth/indices_layout pybind args on every backward entry) — a
    stale pre-idxN build serves the 4-bit path only. The probe reads
    the pybind signature docstrings (py::arg names), so it is exact for
    this build system and degrades to False for any older one."""
    ext = _load()
    if ext is None or not torch.cuda.is_available():
        return False
    for name in ("fused_backward_gemm", "backward_simple_twin",
                 "lut_grad_scatter", "lut_grad_scatter_twin"):
        fn = getattr(ext, name, None)
        if fn is None or "bitwidth" not in (getattr(fn, "__doc__", "") or ""):
            return False
    return True


__all__ = ["fused_backward_gemm", "backward_simple_twin",
           "lut_grad_scatter", "lut_grad_scatter_twin",
           "lut_grad_available", "idxn_available", "available",
           "import_error"]
