#!/usr/bin/env python3
"""fht.py — Fast Hadamard Transform: the boundary-fold rotation, in O(K log K).

FLUTE Extension (EXTENSION_REQUIREMENT.md). This module replaces the
explicit K x K rotation-matrix multiply of the Hadamard boundary fold

    x_rot = x @ T,     T = blockdiag_b( H_b @ diag(s_b) / sqrt(b) )

with a butterfly, eliminating the rotation-matrix memory (the 1.5 GB of
K x K fp32 / 0.75 GB fp16 the loader used to cache) and the per-forward
CPU -> GPU matrix copy (576 MB for K=12288), and turning O(K^2) compute
into O(K log K).

THE MATH (pinned to this repo's conventions — verify against
`build_rotation_matrix`, the loader-side twin of the palettizer's
`_rot_matrix`):

  * T is COLUMN-scaled:  T[i, j] = H[i, j] * s[j] / sqrt(b)  per
    power-of-two segment b (the production `_rot_matrix` builds
    `hadamard(k) * s.view(1, k) / sqrt(k)`). Therefore

        x @ T    = ((x @ H) * s) / sqrt(b)      signs on the OUTPUT

    (NOTE: the requirement document sketches "(x * s) @ H" — that is the
    TRANSPOSED convention diag(s) @ H, which is T^T, not T. This
    implementation pins the repo's column-scaled T; the unit tests
    compare against the explicit `x @ T` ground truth.)

  * H is symmetric and H @ H = b * I, so the adjoint (autograd backward,
    and the weight-space un-rotation W_rot @ T^T) is

        y @ T^T  = ((y * s) @ H) / sqrt(b)      signs on the INPUT

    — the SAME butterfly with the sign moved to the input side.

  * Block-diagonal K (production down_proj: K = 12288 = 8192 + 4096):
    each power-of-two segment transforms independently (the
    off-diagonal blocks of T are zero); `segments(k)` mirrors
    `_rot_matrix`'s descending binary decomposition.

  * The sign vector is the deterministic per-tensor draw
    `rotation_signs(K, seed)`: generator seeded `seed + 4242`,
    `randint(0, 2, (K,)) * 2 - 1` — bitwise the palettizer's.

Public API
----------
    rotation_signs(K, seed)          -> (K,) float32 +-1 tensor
    segments(K)                      -> [(offset, size), ...]
    build_rotation_matrix(K, seed)   -> the explicit (K, K) T (tests /
                                        the "matmul" fallback backend)
    fht_apply(x, signs, backend=...) -> x @ T     (autograd-aware)
    fht_adjoint(v, signs, backend=...) -> v @ T^T (autograd-aware)
    fht_apply_(x, signs, backend=...) -> in-place x <- x @ T (no-grad)

backends: "auto" (CUDA kernel when built + CUDA input, torch butterfly
otherwise), "kernel" (the CUDA extension — REFUSES loudly when absent),
"reference" (the pure-torch butterfly; CPU-legal, fp32 accumulation),
"matmul" (explicit x @ T — differential testing / debugging only).

Why this file lives at the PROJECT ROOT (flute_extended/fht.py, not
inside the `flute_extended` package): the package __init__ imports
`flute_extended._C` at module import time, which requires the built CUDA
extension. The FHT must stay importable on CPU boxes (the reference
path serves reference-mode eval and the unit tests), so this module is
standalone: it imports `_C` lazily and only for the kernel backend.
`scripts/palettized_modules.py` imports it by path (the repo's
sys.path-insert convention already puts this directory on the path).
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch

__all__ = [
    "rotation_signs",
    "segments",
    "build_rotation_matrix",
    "hadamard_matrix",
    "fht_apply",
    "fht_apply_awq",
    "fht_adjoint",
    "fht_apply_",
    "FhtBackend",
    "kernel_available",
]

FhtBackend = str  # "auto" | "kernel" | "reference" | "matmul"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
# The actual Python package is nested: flute_extended/flute_extended/
# This is the path that must be on sys.path for `import flute_extended` to find the package
_PACKAGE_ROOT = _HERE


# ---------------------------------------------------------------------------
# The sign draw + segment decomposition (the palettizer's conventions)
# ---------------------------------------------------------------------------

def rotation_signs(K: int, seed: int) -> torch.Tensor:
    """The per-tensor +-1 sign vector of the Hadamard boundary fold.

    Deterministic in (K, seed); bitwise-identical to the palettizer's
    `s = randint(0, 2, (K,), generator=manual_seed(seed + 4242)) * 2 - 1`
    (palettize_qwen3_5_9b.py::_rot_matrix and the loader twin). Storage:
    (K,) float32 — ~16 KB at K=4096 instead of the ~64 MB K x K matrix.
    """
    g = torch.Generator().manual_seed(int(seed) + 4242)
    return torch.randint(0, 2, (int(K),), generator=g).float().mul(2).sub(1)


def segments(K: int) -> List[Tuple[int, int]]:
    """Descending power-of-two decomposition [(offset, size), ...].

    Mirrors `_rot_matrix`'s block-diagonal construction: 12288 ->
    [(0, 8192), (8192, 4096)], 4096 -> [(0, 4096)], 6144 ->
    [(0, 4096), (4096, 2048)], 3072 -> [(0, 2048), (2048, 1024)].
    """
    K = int(K)
    assert K >= 1
    out: List[Tuple[int, int]] = []
    off, rem = 0, K
    while rem > 0:
        b = 1 << (rem.bit_length() - 1)
        out.append((off, b))
        off += b
        rem -= b
    return out


def hadamard_matrix(K: int) -> torch.Tensor:
    """The Sylvester Hadamard H_K (entries +-1, symmetric, H @ H = K I).

    Reference/test materialization only — the transforms never build it.
    """
    K = int(K)
    assert K >= 1 and (K & (K - 1)) == 0, "hadamard_matrix needs pow2 K"
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < K:
        H = torch.cat([torch.cat([H, H], 1),
                       torch.cat([H, -H], 1)], 0)
    return H


def build_rotation_matrix(K: int, seed: int) -> torch.Tensor:
    """The explicit (K, K) T (block-diagonal over `segments(K)`).

    Ground truth for the differential tests and the "matmul" fallback
    backend. Byte-semantics identical to palettize's `_rot_matrix`.
    """
    K = int(K)
    s = rotation_signs(K, seed)
    if K >= 1 and (K & (K - 1)) == 0:
        return hadamard_matrix(K) * s.view(1, K) / math.sqrt(K)
    T = torch.zeros(K, K, dtype=torch.float32)
    for off, b in segments(K):
        T[off:off + b, off:off + b] = (hadamard_matrix(b) * s[off:off + b].view(1, b) / math.sqrt(b))
    return T


# ---------------------------------------------------------------------------
# The torch reference (the Phase-1 deliverable): pure butterfly, fp32
# ---------------------------------------------------------------------------

def _butterfly(v: torch.Tensor) -> torch.Tensor:
    """In-place-semantics Walsh-Hadamard butterfly on the last dim.

    v: (R, b) fp32 with b a power of two. Computes v @ H_b via
    stage-length doubling — the EXACT pairing of the CUDA kernel's
    fht_stage (sums into the low element, differences into the high
    element of each 2*len group), so reference and kernel agree to
    fp32 rounding order.
    """
    R, b = v.shape
    n = b
    len_ = 1
    while len_ < n:
        v = v.reshape(R, n // (2 * len_), 2, len_)
        u, w = v[..., 0, :], v[..., 1, :]
        # within each 2*len group: [0, len) <- u + w, [len, 2len) <- u - w
        v = torch.cat([u + w, u - w], dim=-1).reshape(R, n)
        len_ <<= 1
    return v


def _check_shape(x: torch.Tensor, signs: torch.Tensor) -> Tuple[int, int]:
    if signs.dim() != 1 or signs.numel() != x.shape[-1]:
        raise ValueError(f"fht: signs must be a (K,) vector matching x's last dim "
            f"(got signs {tuple(signs.shape)}, x {tuple(x.shape)})")
    if not torch.is_floating_point(x):
        raise TypeError(f"fht: x must be floating point, got {x.dtype}")
    if x.shape[-1] < 1:
        raise ValueError("fht: x's last dim must be >= 1")
    return x.numel() // x.shape[-1], int(x.shape[-1])


def fht_reference(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """((x @ H) * s) / sqrt(b) per segment — the torch butterfly.

    fp32 accumulation regardless of input dtype; returns x's dtype.
    Valid on CPU and CUDA; the differential twin of the CUDA kernel.
    """
    M, K = _check_shape(x, signs)
    lead = x.shape[:-1]
    v = x.reshape(M, K).float()
    s = signs.float().reshape(1, K) if signs.is_floating_point() else \
        signs.float().reshape(1, K)
    out = torch.empty_like(v)
    for off, b in segments(K):
        seg = _butterfly(v[:, off:off + b].contiguous())
        inv = 1.0 / math.sqrt(b)
        out[:, off:off + b] = (seg * s[:, off:off + b]) * inv
    return out.reshape(*lead, K).to(x.dtype)


def fht_reference_adjoint(v: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """((v * s) @ H) / sqrt(b) per segment — the adjoint butterfly (v @ T^T).

    Same staging discipline: fp32 accumulation, signs on the input side.
    """
    M, K = _check_shape(v, signs)
    lead = v.shape[:-1]
    g = v.reshape(M, K).float()
    s = signs.float().reshape(1, K)
    out = torch.empty_like(g)
    for off, b in segments(K):
        seg = _butterfly((g[:, off:off + b] * s[:, off:off + b]).contiguous())
        out[:, off:off + b] = seg * (1.0 / math.sqrt(b))
    return out.reshape(*lead, K).to(v.dtype)


def fht_matmul(x: torch.Tensor, signs: torch.Tensor,
               K: Optional[int] = None, seed: Optional[int] = None) -> torch.Tensor:
    """Explicit x @ T ground truth (builds the K x K matrix).

    Backend "matmul" (differential testing / debugging). When (K, seed)
    is omitted, T is rebuilt from `signs` alone — the sign vector plus
    the segment structure determine T completely.
    """
    M, Kdim = _check_shape(x, signs)
    if K is None:
        K = Kdim
    if seed is not None:
        s = rotation_signs(K, seed)
    else:
        s = signs.float()
    T = torch.zeros(K, K, dtype=torch.float32)
    for off, b in segments(K):
        T[off:off + b, off:off + b] = (hadamard_matrix(b) * s[off:off + b].view(1, b) / math.sqrt(b))
    return (x.reshape(M, K).float() @ T).reshape(*x.shape[:-1], K).to(x.dtype)


# ---------------------------------------------------------------------------
# The CUDA extension (lazy import; loud only on backend="kernel")
# ---------------------------------------------------------------------------

_kernel_module = None
_kernel_tried = False


def _load_kernel():
    """Import flute_extended._C (the built extension) or return None.

    Soft by design: backend="auto" falls back to the torch reference
    when the extension is absent (CPU boxes, pre-build). backend
    "kernel" turns the None into a loud refusal at call time.
    """
    global _kernel_module, _kernel_tried
    if _kernel_tried:
        return _kernel_module
    _kernel_tried = True
    try:
        import flute_extended._C as _C  # noqa: F401  (type: ignore)
        _kernel_module = _C
    except Exception:
        # the package import needs the package root on sys.path (the
        # repo convention: flute_extended/ is the root, and the actual
        # Python package lives at flute_extended/flute_extended/);
        # add it and retry.
        #
        # CRITICAL: the initial failed import may have left a stale
        # namespace package in sys.modules['flute_extended'] with
        # __file__=None. Must remove it before retry, otherwise Python
        # will keep using the broken namespace package.
        try:
            if _PACKAGE_ROOT not in sys.path:
                sys.path.insert(0, _PACKAGE_ROOT)
            # Remove stale namespace package if present
            if 'flute_extended' in sys.modules:
                mod = sys.modules['flute_extended']
                if getattr(mod, '__file__', None) is None:
                    # Namespace package - remove it
                    del sys.modules['flute_extended']
                    # Also remove any submodules
                    for key in list(sys.modules.keys()):
                        if key.startswith('flute_extended.'):
                            del sys.modules[key]
            import flute_extended._C as _C  # type: ignore
            _kernel_module = _C
        except Exception:
            _kernel_module = None
    return _kernel_module


def kernel_available(x: Optional[torch.Tensor] = None) -> bool:
    """True when the CUDA extension is importable (and x, if given, is
    on CUDA)."""
    C = _load_kernel()
    if C is None:
        return False
    if x is None:
        return True
    return bool(x.is_cuda)


def _resolve_backend(backend: FhtBackend, x: torch.Tensor) -> str:
    if backend not in ("auto", "kernel", "reference", "matmul"):
        raise ValueError(f"fht: unknown backend {backend!r} (expected 'auto', 'kernel', "
            f"'reference' or 'matmul')")
    if backend in ("reference", "matmul"):
        return backend
    if backend == "kernel":
        if _load_kernel() is None or not x.is_cuda:
            raise RuntimeError(
                "fht(backend='kernel'): the CUDA extension is unavailable "
                "(not built, or the input is not on CUDA). Rebuild with "
                "`cd flute_extended && python setup.py build_ext --inplace`, "
                "or use backend='auto'/'reference'. Never a silent fallback.")
        return "kernel"
    # auto
    if x.is_cuda and _load_kernel() is not None:
        return "kernel"
    return "reference"


def _raw_kernel_apply(x2d: torch.Tensor, signs: torch.Tensor,
                      transpose: bool) -> torch.Tensor:
    """2-D contiguous dispatch to the extension (forward or adjoint)."""
    C = _load_kernel()
    assert C is not None
    K = int(x2d.shape[1])
    if K < 32 or K % 32 != 0 or K > (1 << 16) - 32:
        # the kernel's segment table + thread geometry contract; fall to
        # the reference rather than refusing (auto mode only reaches
        # here for exotic K; "kernel" mode is validated above)
        return (fht_reference_adjoint if transpose else fht_reference)(x2d, signs)
    signs_f = signs if (signs.is_cuda and signs.dtype == torch.float32
                        and signs.is_contiguous()) else \
        signs.detach().to(x2d.device, torch.float32).contiguous()
    if transpose:
        return C.fht_backward(x2d, signs_f)
    return C.fht_forward(x2d, signs_f)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------

class _FhtFn(torch.autograd.Function):
    """autograd.Function for both orientations.

    forward:   y = x @ T          (transpose=False)
    adjoint:   y = x @ T^T        (transpose=True)
    backward:  the opposite orientation (T is orthogonal per segment:
    T^T = T^{-1}, so the adjoint of the forward is the adjoint op and
    vice versa).
    """

    @staticmethod
    def forward(ctx, x2d: torch.Tensor, signs: torch.Tensor,
                transpose: bool, backend: str):
        ctx.save_for_backward(signs)
        ctx.transpose = bool(transpose)
        ctx.backend = backend
        if backend == "kernel":
            return _raw_kernel_apply(x2d, signs, transpose)
        if backend == "matmul":
            K = int(x2d.shape[1])
            T = torch.zeros(K, K, dtype=torch.float32,
                            device=x2d.device)
            s = signs.float()
            for off, b in segments(K):
                T[off:off + b, off:off + b] = (hadamard_matrix(b).to(x2d.device)
                    * s[off:off + b].view(1, b) / math.sqrt(b))
            return (x2d.float() @ (T.t() if transpose else T)).to(x2d.dtype)
        # reference
        return (fht_reference_adjoint if transpose else fht_reference)(x2d, signs)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        signs, = ctx.saved_tensors
        g = grad_out.contiguous()
        # the adjoint of the adjoint is the forward and vice versa
        gx = _FhtFn.apply(g, signs, not ctx.transpose, ctx.backend)
        return gx, None, None, None


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

def fht_apply(x: torch.Tensor, signs: torch.Tensor,
              backend: FhtBackend = "auto") -> torch.Tensor:
    """out = x @ T — the Hadamard boundary-fold input rotation.

    x: (..., K) any leading dims (the (batch, K) and (batch, seq, K)
    layouts of the requirement both work — leading dims flatten).
    signs: (K,) — `rotation_signs(K, seed)` for the production fold.
    dtype: fp16/bf16/fp32 in and out; accumulation is fp32 on every
    backend. Autograd: gradient flows through (backward = the adjoint
    FHT — one more butterfly, signs on the input side).

    backend: "auto" (default; CUDA kernel when built + CUDA input,
    torch butterfly otherwise), "kernel" (loud refusal when the
    extension is missing), "reference", "matmul".
    """
    M, K = _check_shape(x, signs)
    resolved = _resolve_backend(backend, x)
    x2d = x.reshape(M, K)
    need_contig = x2d.stride(-1) != 1
    x2d = x2d.contiguous() if need_contig else x2d
    signs = signs.detach()
    if signs.device != x2d.device:
        signs = signs.to(x2d.device)
    if x2d.requires_grad:
        out = _FhtFn.apply(x2d, signs, False, resolved)
    else:
        if resolved == "kernel":
            out = _raw_kernel_apply(x2d, signs, False)
        elif resolved == "matmul":
            out = fht_matmul(x2d, signs)
        else:
            out = fht_reference(x2d, signs)
    return out.reshape(*x.shape[:-1], K)


def _check_scale(x: torch.Tensor, s: torch.Tensor) -> None:
    """The AWQ scale vector contract for fht_apply_awq: (K,) contiguous
    float32 on x's device (finite/positive is validated by the module at
    load; the kernel divides honestly — a bad entry yields inf/NaN
    outputs, never a silent clamp)."""
    if s.dim() != 1 or s.numel() != x.shape[-1]:
        raise ValueError(f"fht_awq: s must be a (K,) vector matching x's last dim "
            f"(got s {tuple(s.shape)}, x last dim {x.shape[-1]})")
    if not torch.is_floating_point(s):
        raise TypeError(f"fht_awq: s must be floating point, got {s.dtype}")


def _awq_kernel_available() -> bool:
    """True when the built extension exports fht_forward_awq (a +
    build). A older extension keeps the compensated fold on the torch
    chain — a performance fallback, never a correctness one."""
    C = _load_kernel()
    return C is not None and hasattr(C, "fht_forward_awq")


def fht_apply_awq(x: torch.Tensor, signs: torch.Tensor, s: torch.Tensor,
                  backend: FhtBackend = "auto") -> torch.Tensor:
    """out = ((x * s) @ T) / s — the AWQ-compensated boundary-fold rotation
    (the legacy rotate-then-AWQ artifacts' x @ (D T D^-1); ).

    ONE CUDA kernel on the "kernel"/"auto" path (fht_forward_awq: the
    scale multiply folded into the butterfly's prologue, the unscale
    division into its epilogue) — replacing the older five-op eager
    chain `fht_apply(x.float() * s, signs) / s` (INSPECTION §3.3 item 5,
    the 99 AWQ-consumer modules' fp32 cast-multiply-divide chains). The
    kernel is bit-identical to that chain: same fp32 cast, same
    multiply-multiply-divide order, same final dtype cast (see
    fht_forward_awq_kernel in src/kernel_fht.cu).

    x: (..., K) fp16/bf16/fp32 (read natively — the fp32 round-trip
    tensors of the eager chain never exist). s: (K,) float32, x's device.
    Autograd flows through (the eager chain's backward). backends:
    "auto" (kernel when the extension + CUDA input, the torch chain
    otherwise), "kernel" (loud refusal when either the extension or the
     binding is missing), "reference" (the torch chain, CPU-legal),
    "matmul" (the explicit (x*s) @ T / s — differential testing only).
    """
    M, K = _check_shape(x, signs)
    _check_scale(x, s)
    if backend not in ("auto", "kernel", "reference", "matmul"):
        raise ValueError(f"fht_awq: unknown backend {backend!r} (expected 'auto', "
            f"'kernel', 'reference' or 'matmul')")

    x2d = x.reshape(M, K)
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    signs = signs.detach()
    s = s.detach()
    if signs.device != x2d.device:
        signs = signs.to(x2d.device)
    if s.device != x2d.device or s.dtype != torch.float32 \
            or not s.is_contiguous():
        s = s.to(x2d.device, torch.float32).contiguous()

    use_kernel = (backend in ("auto", "kernel")
        and x2d.is_cuda
        and K >= 32 and K % 32 == 0 and K <= (1 << 16) - 32
        and _awq_kernel_available()
    )
    if backend == "kernel" and not use_kernel:
        raise RuntimeError(
            "fht_apply_awq(backend='kernel'): the compensated-FHT CUDA "
            "kernel is unavailable (not built, an older extension, a "
            "non-CUDA input, or K outside the kernel contract). Rebuild "
            "with `cd flute_extended && python setup.py build_ext "
            "--inplace`, or use backend='auto'/'reference'. Never a "
            "silent fallback.")

    if use_kernel:
        C = _load_kernel()
        out = C.fht_forward_awq(x2d, signs, s)
    elif backend == "matmul":
        T = torch.zeros(K, K, dtype=torch.float32, device=x2d.device)
        sf = signs.float()
        for off, b in segments(K):
            T[off:off + b, off:off + b] = (hadamard_matrix(b).to(x2d.device)
                * sf[off:off + b].view(1, b) / math.sqrt(b))
        out = ((x2d.float() * s) @ T / s).to(x2d.dtype)
    else:
        # reference: the exact older torch chain (bit-identical math —
        # fp32 in/out butterfly, the /s as an fp32 division, the final
        # cast to x.dtype). Autograd flows through it.
        out = fht_reference(x2d.float() * s, signs).float() / s
        out = out.to(x2d.dtype)
    return out.reshape(*x.shape[:-1], K)


def fht_adjoint(v: torch.Tensor, signs: torch.Tensor,
                backend: FhtBackend = "auto") -> torch.Tensor:
    """out = v @ T^T — the weight-space un-rotation / gradient adjoint.

    ((v * s) @ H) / sqrt(b) per segment. Used by
    PalettizedLinear._quantized_weight (the original-space effective
    weight) and PalettizedEmbedding's tied-row un-rotation; the
    autograd backward of fht_apply routes through here.
    """
    M, K = _check_shape(v, signs)
    resolved = _resolve_backend(backend, v)
    v2d = v.reshape(M, K)
    v2d = v2d.contiguous() if v2d.stride(-1) != 1 else v2d
    signs = signs.detach()
    if signs.device != v2d.device:
        signs = signs.to(v2d.device)
    if v2d.requires_grad:
        out = _FhtFn.apply(v2d, signs, True, resolved)
    else:
        if resolved == "kernel":
            out = _raw_kernel_apply(v2d, signs, True)
        elif resolved == "matmul":
            out = _matmul_adjoint(v2d, signs)
        else:
            out = fht_reference_adjoint(v2d, signs)
    return out.reshape(*v.shape[:-1], K)


def _matmul_adjoint(v: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    K = int(v.shape[-1])
    T = build_rotation_matrix_from_signs(signs, K, v.device)
    return (v.float() @ T.t()).to(v.dtype)


def build_rotation_matrix_from_signs(signs: torch.Tensor, K: int,
                                     device=None) -> torch.Tensor:
    T = torch.zeros(K, K, dtype=torch.float32)
    s = signs.detach().float().cpu()
    for off, b in segments(K):
        T[off:off + b, off:off + b] = (hadamard_matrix(b) * s[off:off + b].view(1, b) / math.sqrt(b))
    return T.to(device) if device is not None else T


def fht_apply_(x: torch.Tensor, signs: torch.Tensor,
               backend: FhtBackend = "auto") -> torch.Tensor:
    """In-place variant: x <- x @ T. No-grad only (loud refusal when x
    requires grad — in-place autograd is a silent-corruption hazard).
    Contiguity of the flattened (M, K) view is required."""
    if x.requires_grad:
        raise RuntimeError(
            "fht_apply_: in-place FHT on a tensor that requires grad — "
            "use fht_apply (out-of-place) for autograd correctness")
    M, K = _check_shape(x, signs)
    x2d = x.reshape(M, K)
    if x2d.stride(-1) != 1 or (M > 1 and x2d.stride(0) != K):
        x2d = x2d.contiguous()
        # NOTE: a non-contiguous x gets a contiguous COPY transformed
        # in place; the caller's view of x is only updated when x2d
        # aliases x. Refuse instead? No: copy-back keeps semantics.
        out = fht_apply(x2d, signs, backend)
        x.copy_(out.reshape(*x.shape))
        return x
    resolved = _resolve_backend(backend, x)
    signs = signs.detach()
    if resolved == "kernel":
        C = _load_kernel()
        K_ = int(x2d.shape[1])
        signs_f = signs if (signs.is_cuda and signs.dtype == torch.float32
                            and signs.is_contiguous()) else \
            signs.to(x2d.device, torch.float32).contiguous()
        if K_ >= 32 and K_ % 32 == 0 and K_ <= (1 << 16) - 32:
            return C.fht_inplace_(x2d, signs_f).reshape(*x.shape)
        # exotic K: fall through to the reference overwriting in place
        ref = fht_reference(x2d, signs)
        x2d.copy_(ref)
        return x
    ref = fht_apply(x2d, signs, backend=resolved)
    x2d.copy_(ref.reshape(M, K))
    return x


# ---------------------------------------------------------------------------
# Self-check (python fht.py --selfcheck)
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    torch.manual_seed(0)
    ok = True
    for K in (64, 256, 4096, 12288):
        signs = rotation_signs(K, seed=K)
        x = torch.randn(3, 7, K)
        T = build_rotation_matrix(K, seed=K)
        ref = fht_reference(x, signs)
        exp = (x.float() @ T).to(x.dtype)
        err = (ref - exp).abs().max().item()
        adj = fht_reference_adjoint(x, signs)
        exp_adj = (x.float() @ T.t()).to(x.dtype)
        err_adj = (adj - exp_adj).abs().max().item()
        # fp16 roundtrip
        xh = x.half()
        refh = fht_reference(xh, signs)
        errh = (refh.float() - exp).abs().max().item()
        # autograd
        xa = torch.randn(5, K, requires_grad=True)
        ya = fht_apply(xa, signs)
        ga = torch.randn_like(ya)
        ya.backward(ga)
        gx_manual = fht_reference_adjoint(ga, signs)
        grad_err = (xa.grad - gx_manual).abs().max().item()
        line_ok = (err < 1e-4 and err_adj < 1e-4 and errh < 0.05
                   and grad_err < 1e-4)
        ok &= line_ok
        print(f"K={K:6d}: fwd {err:.2e} adj {err_adj:.2e} "
              f"fp16 {errh:.2e} grad {grad_err:.2e} "
              f"{'OK' if line_ok else 'FAIL'}")
    print("selfcheck:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
