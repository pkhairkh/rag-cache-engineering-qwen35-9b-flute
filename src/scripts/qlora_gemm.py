#!/usr/bin/env python3
"""
qlora_gemm.py — fused dequant + matmul autograd.Function for QLoRA training.

Forward:  Y = X @ W^T  via flute_extended.qgemm_per_group_lut (FLUTE kernel)
Backward: dL/dX = dL/dY @ W  via flute_train_kernels.fused_backward_gemm
          (fused CUDA kernel: W dequantized on-the-fly into shared memory,
           fed to Tensor Cores via mma.sync.m16n8k16 FP32-acc; W never in DRAM)

The frozen buffers (indices, lut) are stored as ctx attributes rather than
save_for_backward: as non-autograd buffers they need no tracking, and
attributes survive gradient-checkpoint recomputation (saved tensors may
not).

Backward-path control:
  FLUTE_FUSED_BWD=0   disable the fused backward kernel (P0 escape hatch;
                      gradients then flow through the cached reference path
                      in qlora_fallback.py)
  FLUTE_DEBUG_NAN=1   raise on NaN in the fused backward output (debug only;
                      the check forces a device sync)

W5 (docs/KERNEL_SPEC_DLDLUT.md): FusedQLoRAGEMMTrainLUT adds dL/dLUT —
the trainable-codebook Function (forward = the FLUTE kernel on the fp16
operand cast of the fp32 master; backward = the existing fused backward
(grad_x) + lut_grad_scatter (grad_lut); the CPU reference twin is the
G-B5 oracle).
"""

from __future__ import annotations
import os, sys
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

_flute_checked = False
_flute_ok = False
_FLUTE_IMPORT_ERROR = None      # the verbatim reason the forward kernel is unavailable
_FLUTE_REPORTED = False         # one-shot banner guard (once per process)


def flute_import_error():
    """The verbatim import failure of flute_extended, or None when the
    forward kernel is importable. Surfaced everywhere the fused path is
    refused so the box never has to guess WHY the kernel is missing
    (the old code swallowed the exception silently)."""
    _check_flute_kernel()
    return _FLUTE_IMPORT_ERROR


def _report_flute_unavailable_once():
    """ONE loud line per process when the FLUTE forward kernel is not
    importable: the verbatim error + the build command + the toolchain
    hints (nvcc must match torch.version.cuda; CUDA_HOME must point at
    the toolkit, not the driver)."""
    global _FLUTE_REPORTED
    if _FLUTE_REPORTED or _flute_ok:
        return
    _FLUTE_REPORTED = True
    print("[qlora_gemm] FLUTE forward kernel UNAVAILABLE — "
          f"{flute_import_error()}", flush=True)
    print("[qlora_gemm]   build: cd flute_extended && python setup.py "
          "build_ext --inplace", flush=True)
    print("[qlora_gemm]   toolchain: the build needs nvcc + CUDA headers "
          "whose major matches torch.version.cuda "
          f"({getattr(torch.version, 'cuda', None)}); CUDA_HOME must point "
          "at the CUDA toolkit (not just the driver). No nvcc on the box "
          "-> install the cuda-toolkit matching the torch CUDA version, "
          "or keep the FLUTE_FROZEN_PATH=torch opt-out (cuBLAS on a "
          "materialized W16, same GEMM shapes).", flush=True)


def _check_flute_kernel() -> bool:
    global _flute_checked, _flute_ok, _FLUTE_IMPORT_ERROR
    if _flute_checked: return _flute_ok
    _flute_checked = True
    try:
        p = os.path.join(_HERE, "..", "flute_extended")
        if p not in sys.path: sys.path.insert(0, p)
        import flute_extended
        if hasattr(flute_extended, "qgemm_per_group_lut"):
            _flute_ok = True
        else:
            _FLUTE_IMPORT_ERROR = ("flute_extended imported but "
                                    "qgemm_per_group_lut is absent (a "
                                    "partial/broken build — rebuild)")
    except Exception as e:
        _flute_ok = False
        _FLUTE_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    _report_flute_unavailable_once()
    return _flute_ok

_backward_checked = False
_backward_ok = False
_backward_escape_banner = False
_BACKWARD_IMPORT_ERROR = None   # verbatim reason the fused backward is down


def backward_import_error():
    """The verbatim reason the fused backward kernel is unavailable
    (FLUTE_FUSED_BWD opt-out, import failure, or CUDA down), or None."""
    _check_backward_kernel()
    return _BACKWARD_IMPORT_ERROR


def _check_backward_kernel() -> bool:
    global _backward_checked, _backward_ok, _backward_escape_banner
    global _BACKWARD_IMPORT_ERROR
    if _backward_checked: return _backward_ok
    _backward_checked = True
    if os.environ.get("FLUTE_FUSED_BWD", "1") in ("", "0"):
        _backward_ok = False
        _BACKWARD_IMPORT_ERROR = ("FLUTE_FUSED_BWD is set to the explicit "
                                  "opt-out (0 or empty)")
        if not _backward_escape_banner:
            _backward_escape_banner = True
            print("[qlora_gemm] FLUTE_FUSED_BWD=0: fused backward kernel disabled; "
                  "gradients use the cached reference path (qlora_fallback.py)",
                  flush=True)
        return False
    try:
        p = os.path.join(_HERE, "..", "flute_train_kernels")
        if p not in sys.path: sys.path.insert(0, p)
        import flute_train_kernels as ftk
        _backward_ok = ftk.available()
        if not _backward_ok:
            _BACKWARD_IMPORT_ERROR = ftk.import_error() or \
                "flute_train_kernels._C loads but CUDA is unavailable"
            if not _backward_escape_banner:
                _backward_escape_banner = True
                print("[qlora_gemm] fused backward kernel UNAVAILABLE — "
                      f"{_BACKWARD_IMPORT_ERROR}", flush=True)
                print("[qlora_gemm]   build: cd flute_train_kernels && "
                      "python setup.py build_ext --inplace", flush=True)
    except Exception as e:
        _backward_ok = False
        _BACKWARD_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    return _backward_ok

def fused_backward_available() -> bool:
    return _check_backward_kernel()

def _debug_nan_enabled() -> bool:
    return os.environ.get("FLUTE_DEBUG_NAN", "") == "1"


_lutgrad_checked = False
_lutgrad_ok = False
_LUTGRAD_IMPORT_ERROR = None


def _check_lut_grad_kernel() -> bool:
    """True when the built flute_train_kernels carries the W5
    lut_grad_scatter kernel (and CUDA is up)."""
    global _lutgrad_checked, _lutgrad_ok, _LUTGRAD_IMPORT_ERROR
    if _lutgrad_checked:
        return _lutgrad_ok
    _lutgrad_checked = True
    try:
        import flute_train_kernels as ftk
        _lutgrad_ok = ftk.lut_grad_available()
        if not _lutgrad_ok:
            _LUTGRAD_IMPORT_ERROR = (
                "the built flute_train_kernels carries no "
                "lut_grad_scatter (rebuild: cd flute_train_kernels && "
                "python setup.py build_ext --inplace)")
    except Exception as e:
        _lutgrad_ok = False
        _LUTGRAD_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    return _lutgrad_ok


def lut_grad_import_error():
    """The verbatim reason the W5 scatter kernel is unavailable, or
    None."""
    _check_lut_grad_kernel()
    return _LUTGRAD_IMPORT_ERROR


# the idx4 flat-position arithmetic mirrors qlora_fallback's
# dequant_idx4_torch (the normative tile/chunk encoding); bounded
# temporaries per chunk.
_SCATTER_CHUNK = 1 << 20


def _lut_grad_scatter_reference(grad_y, x, indices_blob, N, K,
                                 group_size, bitwidth=4):
    """The closed-form dL/dLUT (G-B5a's CPU oracle — the Function's
    reference arm): dW = grad_Y^T @ X in fp32, scattered into
    (n_groups, 2^bitwidth) fp32 keyed by the idxN codes, accumulated in
    the CANONICAL flat pair-position order (bitwidth=4: the byte order —
    bit-exact against the legacy walk) so any equally-ordered oracle is
    bit-exact against it.

    bitwidth 1/2/3 walk the unified idxN layout's PAIR positions (the
    same flat-position lineage as qlora_fallback._idxn_pair_positions,
    restated here so the oracle stays one self-contained file).

    Works on any device (CPU: the gates; CUDA: the Function's fallback
    when the W5 kernel is unavailable)."""
    if bitwidth not in (1, 2, 3, 4):
        raise ValueError(f"idxN bitwidth must be 1-4, got {bitwidth}")
    dW = grad_y.t().float() @ x.float()                  # (N, K) fp32
    n_groups = (N + group_size - 1) // group_size
    codes = 1 << bitwidth
    grad_lut = torch.zeros(n_groups, codes, dtype=torch.float32,
                           device=dW.device)
    blob = indices_blob.reshape(-1)
    g_row = (torch.arange(N, device=dW.device) // group_size)
    g_row = g_row.unsqueeze(1).expand(N, K).reshape(-1)
    dW_flat = dW.reshape(-1)

    if bitwidth == 4:
        # the frozen b=4 byte walk (unchanged)
        total = blob.numel()
        if total != N * (K // 2):
            raise ValueError(f"idx4 blob has {total} bytes, "
                             f"expected {N * (K // 2)}")
        tiles_k = K // 64
        idx_flat = torch.empty(N * K, dtype=torch.long, device=dW.device)
        for start in range(0, total, _SCATTER_CHUNK):
            end = min(start + _SCATTER_CHUNK, total)
            p = torch.arange(start, end, dtype=torch.int64,
                             device=dW.device)
            tile = p >> 12
            t = torch.div(tile, tiles_k, rounding_mode="floor")
            g = tile - t * tiles_k
            within = p & 4095
            wx = within >> 11
            r2 = within & 2047
            chunk = r2 >> 6
            r3 = r2 & 63
            seg = r3 >> 4
            v = (r3 >> 2) & 3
            d = (r3 >> 1) & 1
            s2 = r3 & 1
            n = (t << 7) + (wx << 6) + (v << 4) + (d << 3) + (chunk >> 2)
            k = (g << 6) + (seg << 4) + ((chunk & 3) << 1) + (s2 << 3)
            b = blob[start:end].long()
            nK = n * K
            idx_flat[nK + k] = b & 15
            idx_flat[nK + k + 1] = b >> 4
        grad_lut.index_put_((g_row, idx_flat), dW_flat, accumulate=True)
        return grad_lut

    # sub-4-bit: the flat PAIR walk (bits 2*b per pair, LSB-first) fills
    # the logical index array; ONE final index_put_ in flat (n*K + k)
    # order — the same accumulation order as the b=4 walk.
    b = int(bitwidth)
    field_mask = (1 << (2 * b)) - 1
    expected = (N * K * b) // 8
    if blob.numel() != expected:
        raise ValueError(f"idx{b} blob has {blob.numel()} bytes, "
                         f"expected {expected}")
    tiles_k = K // 64
    total_pairs = (N * K) // 2
    total = blob.numel()
    idx_flat = torch.empty(N * K, dtype=torch.long, device=dW.device)
    for start in range(0, total_pairs, _SCATTER_CHUNK):
        end = min(start + _SCATTER_CHUNK, total_pairs)
        p = torch.arange(start, end, dtype=torch.int64,
                         device=dW.device)
        tile_pair = p >> 12
        t = torch.div(tile_pair, tiles_k, rounding_mode="floor")
        g = tile_pair - t * tiles_k
        within = p & 4095
        wx = within >> 11
        r2 = within & 2047
        lane = r2 >> 6
        r3 = r2 & 63
        seg = r3 >> 4
        j = r3 & 15
        v = j >> 2
        d = (j >> 1) & 1
        s2 = j & 1
        n = (t << 7) + (wx << 6) + (v << 4) + (d << 3) + (lane >> 2)
        k = (g << 6) + (seg << 4) + ((lane & 3) << 1) + (s2 << 3)
        chunk_base = ((t * tiles_k + g) * (1024 * b)
                      + wx * (512 * b) + lane * (16 * b))
        bit = (chunk_base * 8) + 2 * b * r3
        byte0 = bit >> 3
        shift = bit & 7
        span = (shift + 2 * b) > 8
        safe1 = (byte0 + 1).clamp(max=total - 1)
        w0 = blob[byte0].long()
        w1 = torch.where(span, blob[safe1].long(),
                         torch.zeros((), dtype=torch.int64,
                                     device=dW.device))
        field = ((w0 | (w1 << 8)) >> shift) & field_mask
        v0 = field & (codes - 1)
        v1 = field >> b
        nK = n * K
        idx_flat[nK + k] = v0
        idx_flat[nK + k + 1] = v1
    grad_lut.index_put_((g_row, idx_flat), dW_flat, accumulate=True)
    return grad_lut


def _ftk_backward_bitwidth(fn, *args, bitwidth, indices_layout):
    """Call a flute_train_kernels entry with the idxN parameters,
    translating a stale pre-idxN extension's TypeError (no bitwidth
    parameter in the pybind signature) into the loud rebuild message —
    never a silent wrong-width fallback."""
    try:
        return fn(*args, bitwidth=bitwidth, indices_layout=indices_layout)
    except TypeError as e:
        if "argument" in str(e).lower():
            raise RuntimeError(
                "the built flute_train_kernels predates the idxN "
                "extension (its pybind entries carry no bitwidth "
                "parameter): rebuild with cd flute_train_kernels && "
                "python setup.py build_ext --inplace") from e
        raise


def train_lut_reference_forward(x, indices, lut, N, K, group_size,
                                bitwidth=4):
    """The pure-torch forward twin (the G-B5 CPU oracle): W = the
    reference dequant of the fp16 operand, Y = X @ W^T in fp32 (any
    idxN width)."""
    import qlora_fallback
    lut16 = lut if lut.dtype == torch.float16 else lut.to(torch.float16)
    W = qlora_fallback.dequant_idxn_torch(indices, lut16, N, K,
                                           group_size, bitwidth,
                                           dtype=torch.float32)
    return x.float() @ W.t()


def train_lut_reference_backward(grad_y, x, indices, lut, N, K,
                                  group_size, bitwidth=4):
    """The pure-torch backward twin: grad_x = grad_Y @ W (the reference
    dequant) and grad_lut = the closed-form scatter — the analytic
    G-B5a/G-B5d target on CPU (any idxN width)."""
    import qlora_fallback
    lut16 = lut if lut.dtype == torch.float16 else lut.to(torch.float16)
    W = qlora_fallback.dequant_idxn_torch(indices, lut16, N, K,
                                           group_size, bitwidth,
                                           dtype=torch.float32)
    grad_x = grad_y.float() @ W
    grad_lut = _lut_grad_scatter_reference(grad_y, x, indices, N, K,
                                            group_size, bitwidth)
    return grad_x, grad_lut


# --------------------------------------------------------------------------- #
# W10 (HANDOVER "Two-Stream Training"): the two-stream extensions. The
# analytic contract is the per-stream decomposition of the ordered add
#     y = y1 + y2 = x @ W1^T + x @ W2^T
#     dL/dx    = dL/dy @ (W1 + W2)     — two backward GEMMs, summed
#     dL/dlut1 = scatter(dL/dy, x, indices1)   — INDEPENDENT of stream 2
#     dL/dlut2 = scatter(dL/dy, x, indices2)   — INDEPENDENT of stream 1
# so NO new CUDA kernels are required: the W9 idxN backward family
# already serves any width per stream, and the two-stream backward is
# exactly two single-stream backwards (grad_x summed; the LUT
# scatters independent). The reference twins below are the CPU oracles
# (the G-B pattern of the single-stream path, restated for both
# streams); the Functions are the kernel-path arms.
# --------------------------------------------------------------------------- #

def train_lut_two_streams_reference_forward(x, indices1, lut1, bitwidth1,
                                            indices2, lut2, bitwidth2,
                                            N, K, group_size):
    """The pure-torch two-stream forward twin: the ordered add
    y = x @ W1^T + x @ W2^T in fp32 (the reference arm of the
    deployment shape — TWO GEMMs plus ONE add, stream 1 pinned first;
    NOT x @ (W1 + W2)^T)."""
    import qlora_fallback
    lut16_1 = lut1 if lut1.dtype == torch.float16 else lut1.to(torch.float16)
    lut16_2 = lut2 if lut2.dtype == torch.float16 else lut2.to(torch.float16)
    W1 = qlora_fallback.dequant_idxn_torch(indices1, lut16_1, N, K,
                                           group_size, bitwidth1,
                                           dtype=torch.float32)
    W2 = qlora_fallback.dequant_idxn_torch(indices2, lut16_2, N, K,
                                           group_size, bitwidth2,
                                           dtype=torch.float32)
    xf = x.float()
    return xf @ W1.t() + xf @ W2.t()


def train_lut_two_streams_reference_backward(grad_y, x, indices1, lut1,
                                              bitwidth1, indices2, lut2,
                                              bitwidth2, N, K, group_size):
    """The pure-torch two-stream backward twin — the analytic CPU
    oracle: grad_x = dL/dy @ (W1 + W2) (ONE GEMM on the summed
    reference weight — the same values as the two-GEMM sum, the
    cheap oracle form), grad_lut1/grad_lut2 = the per-stream
    closed-form scatters (independent of the other stream)."""
    import qlora_fallback
    lut16_1 = lut1 if lut1.dtype == torch.float16 else lut1.to(torch.float16)
    lut16_2 = lut2 if lut2.dtype == torch.float16 else lut2.to(torch.float16)
    W1 = qlora_fallback.dequant_idxn_torch(indices1, lut16_1, N, K,
                                           group_size, bitwidth1,
                                           dtype=torch.float32)
    W2 = qlora_fallback.dequant_idxn_torch(indices2, lut16_2, N, K,
                                           group_size, bitwidth2,
                                           dtype=torch.float32)
    grad_x = grad_y.float() @ (W1 + W2)
    grad_lut1 = _lut_grad_scatter_reference(grad_y, x, indices1, N, K,
                                            group_size, bitwidth1)
    grad_lut2 = _lut_grad_scatter_reference(grad_y, x, indices2, N, K,
                                            group_size, bitwidth2)
    return grad_x, grad_lut1, grad_lut2


class FusedQLoRAGEMMTrainLUT(torch.autograd.Function):
    """Y = X @ dequant(idxN, lut16)^T with dL/dLUT — the W5 trainable
    codebook path (docs/KERNEL_SPEC_DLDLUT.md §5), any bit width 1-4.

    The _LoRABranchFn pattern applied to the codebook: the trainable LUT
    is an fp32 master; the fp16 kernel operand is its cast, RECOMPUTED
    in backward (never saved — the save-contract rule); x is saved in
    its fp16 operand form. Backward: grad_x via the EXISTING
    fused_backward_gemm (W from the recomputed operand), grad_lut via
    lut_grad_scatter (the deterministic two-pass kernel) or the closed-
    form reference when the kernel is unavailable; grad_indices None
    (frozen integers)."""

    @staticmethod
    def forward(ctx, x, indices, lut_master, bitwidth, group_size, N, K):
        import flute_extended
        original_dtype = x.dtype
        xh = x.half() if x.dtype != torch.float16 else x
        lut16 = lut_master.detach().to(torch.float16)
        y = flute_extended.qgemm_per_group_lut(
            xh, indices, lut16,
            bitwidth=int(bitwidth), group_size=int(group_size),
            indices_layout=f"idx{int(bitwidth)}")
        ctx.indices = indices
        ctx.bitwidth = int(bitwidth)
        ctx.group_size = int(group_size)
        ctx.N = int(N)
        ctx.K = int(K)
        ctx.x_dtype = original_dtype
        ctx.save_for_backward(xh, lut_master)
        return y.to(original_dtype)

    @staticmethod
    def backward(ctx, grad_y):
        xh, lut_master = ctx.saved_tensors
        indices = ctx.indices
        lut16 = lut_master.detach().to(torch.float16)
        need_x = ctx.needs_input_grad[0]
        need_lut = ctx.needs_input_grad[2]
        grad_x = None
        if need_x:
            if _check_backward_kernel():
                import flute_train_kernels as ftk
                grad_y16 = grad_y.half() if grad_y.dtype != \
                    torch.float16 else grad_y
                grad_x = _ftk_backward_bitwidth(
                    ftk.fused_backward_gemm,
                    grad_y16.contiguous(), indices.contiguous(),
                    lut16.contiguous(), ctx.N, ctx.K, ctx.group_size,
                    bitwidth=ctx.bitwidth,
                    indices_layout=f"idx{ctx.bitwidth}")
                grad_x = grad_x.to(ctx.x_dtype)
            else:
                import qlora_fallback
                W = qlora_fallback.dequant_idxn_torch(
                    indices, lut16, ctx.N, ctx.K, ctx.group_size,
                    ctx.bitwidth, dtype=torch.float32)
                grad_x = (grad_y.float() @ W).to(ctx.x_dtype)
        grad_lut = None
        if need_lut:
            if _check_lut_grad_kernel():
                import flute_train_kernels as ftk
                grad_y16 = grad_y.half() if grad_y.dtype != \
                    torch.float16 else grad_y
                grad_lut = _ftk_backward_bitwidth(
                    ftk.lut_grad_scatter,
                    grad_y16.contiguous(), xh, indices.contiguous(),
                    ctx.N, ctx.K, ctx.group_size,
                    bitwidth=ctx.bitwidth,
                    indices_layout=f"idx{ctx.bitwidth}")
                grad_lut = grad_lut.to(lut_master.dtype)
            else:
                grad_lut = _lut_grad_scatter_reference(
                    grad_y, xh, indices, ctx.N, ctx.K,
                    ctx.group_size, ctx.bitwidth).to(lut_master.dtype)
        return grad_x, None, grad_lut, None, None, None, None


def fused_qlora_gemm_train_lut(x, indices, lut_master, bitwidth,
                               group_size, N, K):
    """Y = X @ W^T through the trainable-LUT Function. Returns None when
    the FLUTE forward kernel is unavailable (the caller falls back to
    the reference path)."""
    if not _check_flute_kernel():
        return None
    original_shape = x.shape
    if x.dim() == 3:
        b, s, k = x.shape
        x2 = x.reshape(b * s, k)
    else:
        x2 = x
    y = FusedQLoRAGEMMTrainLUT.apply(x2, indices, lut_master, bitwidth,
                                     group_size, N, K)
    if len(original_shape) == 3:
        y = y.view(original_shape[0], original_shape[1], N)
    return y


class FusedQLoRAGEMMTrainLUTTwoStreams(torch.autograd.Function):
    """Y = X @ (dequant(idxN1, lut1) + dequant(idxN2, lut2))^T — the W10
    two-stream trainable path (HANDOVER W10 Gap 1): the deployment shape
    (two FLUTE qgemms + ONE ordered add, stream 1 pinned first) with
    gradients through BOTH fp32 LUT masters.

    Backward (the analytic contract above):
      * grad_x = dL/dY @ (W1 + W2) — two fused_backward_gemm calls
        (one per stream, each W dequantized on-the-fly from ITS OWN
        lut16 operand — never materialized, never summed in DRAM),
        summed; the reference arm materializes W1 + W2 once.
      * grad_lut{1,2} = lut_grad_scatter(dL/dY, x, indices{1,2}) —
        the per-stream closed-form scatter, INDEPENDENT of the other
        stream (y1 does not depend on lut2 and vice versa), so each
        arm is byte-identical to the single-stream W5/W9 path.
      * grad_indices{1,2} None (frozen integers).

    The streams may carry DIFFERENT idxN widths (mixed:4,2 = idx4 base
    + idx2 refinement; hybrid422 = idx4 + idx4.2 pair-composite) —
    each stream's kernel call pins its own bitwidth/layout explicitly.
    """

    @staticmethod
    def forward(ctx, x, indices1, lut1_master, bitwidth1,
                indices2, lut2_master, bitwidth2, group_size, N, K):
        import flute_extended
        original_dtype = x.dtype
        xh = x.half() if x.dtype != torch.float16 else x
        lut16_1 = lut1_master.detach().to(torch.float16)
        lut16_2 = lut2_master.detach().to(torch.float16)
        y1 = flute_extended.qgemm_per_group_lut(
            xh, indices1, lut16_1,
            bitwidth=int(bitwidth1), group_size=int(group_size),
            indices_layout=f"idx{int(bitwidth1)}")
        y2 = flute_extended.qgemm_per_group_lut(
            xh, indices2, lut16_2,
            bitwidth=int(bitwidth2), group_size=int(group_size),
            indices_layout=f"idx{int(bitwidth2)}")
        y = y1 + y2                      # ordered add, stream 1 first
        ctx.indices1 = indices1
        ctx.indices2 = indices2
        ctx.bitwidth1 = int(bitwidth1)
        ctx.bitwidth2 = int(bitwidth2)
        ctx.group_size = int(group_size)
        ctx.N = int(N)
        ctx.K = int(K)
        ctx.x_dtype = original_dtype
        ctx.save_for_backward(xh, lut1_master, lut2_master)
        return y.to(original_dtype)

    @staticmethod
    def backward(ctx, grad_y):
        xh, lut1_master, lut2_master = ctx.saved_tensors
        lut16_1 = lut1_master.detach().to(torch.float16)
        lut16_2 = lut2_master.detach().to(torch.float16)
        need_x = ctx.needs_input_grad[0]
        need_lut1 = ctx.needs_input_grad[2]
        need_lut2 = ctx.needs_input_grad[5]
        grad_x = None
        if need_x:
            if _check_backward_kernel():
                import flute_train_kernels as ftk
                grad_y16 = grad_y.half() if grad_y.dtype != \
                    torch.float16 else grad_y
                gx1 = _ftk_backward_bitwidth(
                    ftk.fused_backward_gemm,
                    grad_y16.contiguous(), ctx.indices1.contiguous(),
                    lut16_1.contiguous(), ctx.N, ctx.K, ctx.group_size,
                    bitwidth=ctx.bitwidth1,
                    indices_layout=f"idx{ctx.bitwidth1}")
                gx2 = _ftk_backward_bitwidth(
                    ftk.fused_backward_gemm,
                    grad_y16.contiguous(), ctx.indices2.contiguous(),
                    lut16_2.contiguous(), ctx.N, ctx.K, ctx.group_size,
                    bitwidth=ctx.bitwidth2,
                    indices_layout=f"idx{ctx.bitwidth2}")
                grad_x = (gx1 + gx2).to(ctx.x_dtype)
            else:
                import qlora_fallback
                W1 = qlora_fallback.dequant_idxn_torch(
                    ctx.indices1, lut16_1, ctx.N, ctx.K, ctx.group_size,
                    ctx.bitwidth1, dtype=torch.float32)
                W2 = qlora_fallback.dequant_idxn_torch(
                    ctx.indices2, lut16_2, ctx.N, ctx.K, ctx.group_size,
                    ctx.bitwidth2, dtype=torch.float32)
                grad_x = (grad_y.float() @ (W1 + W2)).to(ctx.x_dtype)
        grad_lut1 = None
        if need_lut1:
            if _check_lut_grad_kernel():
                import flute_train_kernels as ftk
                grad_y16 = grad_y.half() if grad_y.dtype != \
                    torch.float16 else grad_y
                grad_lut1 = _ftk_backward_bitwidth(
                    ftk.lut_grad_scatter,
                    grad_y16.contiguous(), xh, ctx.indices1.contiguous(),
                    ctx.N, ctx.K, ctx.group_size,
                    bitwidth=ctx.bitwidth1,
                    indices_layout=f"idx{ctx.bitwidth1}")
                grad_lut1 = grad_lut1.to(lut1_master.dtype)
            else:
                grad_lut1 = _lut_grad_scatter_reference(
                    grad_y, xh, ctx.indices1, ctx.N, ctx.K,
                    ctx.group_size, ctx.bitwidth1).to(lut1_master.dtype)
        grad_lut2 = None
        if need_lut2:
            if _check_lut_grad_kernel():
                import flute_train_kernels as ftk
                grad_y16 = grad_y.half() if grad_y.dtype != \
                    torch.float16 else grad_y
                grad_lut2 = _ftk_backward_bitwidth(
                    ftk.lut_grad_scatter,
                    grad_y16.contiguous(), xh, ctx.indices2.contiguous(),
                    ctx.N, ctx.K, ctx.group_size,
                    bitwidth=ctx.bitwidth2,
                    indices_layout=f"idx{ctx.bitwidth2}")
                grad_lut2 = grad_lut2.to(lut2_master.dtype)
            else:
                grad_lut2 = _lut_grad_scatter_reference(
                    grad_y, xh, ctx.indices2, ctx.N, ctx.K,
                    ctx.group_size, ctx.bitwidth2).to(lut2_master.dtype)
        return (grad_x, None, grad_lut1, None,
                None, grad_lut2, None, None, None, None)


def fused_qlora_gemm_train_lut_two_streams(
        x, indices1, lut1_master, bitwidth1, indices2, lut2_master,
        bitwidth2, group_size, N, K):
    """Y = X @ (W1 + W2)^T through the two-stream trainable-LUT
    Function (W10). Returns None when the FLUTE forward kernel is
    unavailable (the caller falls back to the reference path)."""
    if not _check_flute_kernel():
        return None
    original_shape = x.shape
    if x.dim() == 3:
        b, s, k = x.shape
        x2 = x.reshape(b * s, k)
    else:
        x2 = x
    y = FusedQLoRAGEMMTrainLUTTwoStreams.apply(
        x2, indices1, lut1_master, bitwidth1, indices2, lut2_master,
        bitwidth2, group_size, N, K)
    if len(original_shape) == 3:
        y = y.view(original_shape[0], original_shape[1], N)
    return y


class FusedQLoRAGEMMTwoStreams(torch.autograd.Function):
    """Fused dequant + matmul for the QLoRA FROZEN two-stream branch —
    the W10 fix for the silent stream-2 drop: the plain FusedQLoRAGEMM
    path served only stream 1, so a two-stream module under the
    QLoRALinear wrapper trained against a WRONG forward (y2 missing).
    This Function mirrors FusedQLoRAGEMM (frozen buffers as ctx
    attributes — they survive checkpoint recompute) with the
    deployment shape: TWO qgemm calls + ONE ordered fp16 add, and
    grad_x = dL/dY @ (W1 + W2) via two fused backward GEMMs summed.
    The LUTs carry no gradient on this path (frozen fp16 buffers)."""

    @staticmethod
    def forward(ctx, x, indices1, lut1, bitwidth1, indices2, lut2,
                bitwidth2, group_size, N, K):
        import flute_extended
        original_dtype = x.dtype
        xh = x.half() if x.dtype != torch.float16 else x
        y1 = flute_extended.qgemm_per_group_lut(
            xh, indices1, lut1,
            bitwidth=int(bitwidth1), group_size=int(group_size),
            indices_layout=f"idx{int(bitwidth1)}")
        y2 = flute_extended.qgemm_per_group_lut(
            xh, indices2, lut2,
            bitwidth=int(bitwidth2), group_size=int(group_size),
            indices_layout=f"idx{int(bitwidth2)}")
        y = y1 + y2                      # ordered add, stream 1 first
        ctx.indices1 = indices1
        ctx.lut1 = lut1
        ctx.indices2 = indices2
        ctx.lut2 = lut2
        ctx.bitwidth1 = int(bitwidth1)
        ctx.bitwidth2 = int(bitwidth2)
        ctx.group_size = int(group_size)
        ctx.N = int(N)
        ctx.K = int(K)
        ctx.x_dtype = original_dtype
        return y.to(original_dtype)

    @staticmethod
    def backward(ctx, grad_y):
        if _check_backward_kernel():
            import flute_train_kernels as ftk
            grad_y_h = grad_y.half() if grad_y.dtype != torch.float16 \
                else grad_y
            gx1 = _ftk_backward_bitwidth(
                ftk.fused_backward_gemm,
                grad_y_h.contiguous(), ctx.indices1.contiguous(),
                ctx.lut1.contiguous(), ctx.N, ctx.K, ctx.group_size,
                bitwidth=ctx.bitwidth1,
                indices_layout=f"idx{ctx.bitwidth1}")
            gx2 = _ftk_backward_bitwidth(
                ftk.fused_backward_gemm,
                grad_y_h.contiguous(), ctx.indices2.contiguous(),
                ctx.lut2.contiguous(), ctx.N, ctx.K, ctx.group_size,
                bitwidth=ctx.bitwidth2,
                indices_layout=f"idx{ctx.bitwidth2}")
            grad_x = gx1 + gx2
            if _debug_nan_enabled() and torch.isnan(grad_x).any():
                raise RuntimeError(
                    f"fused_backward_gemm (two-stream) produced NaN "
                    f"(N={ctx.N}, K={ctx.K}, group_size={ctx.group_size})")
            return (grad_x.to(ctx.x_dtype), None, None, None,
                    None, None, None, None, None, None)
        # Reference path: the summed weight materialized once (the
        # single-GEMM form of dL/dY @ (W1 + W2)).
        import qlora_fallback
        w_dtype = grad_y.dtype
        if w_dtype == torch.float32 and ctx.x_dtype != torch.float32:
            w_dtype = ctx.x_dtype
        W1 = qlora_fallback.materialize_weight(
            ctx.indices1, ctx.lut1, ctx.N, ctx.K, ctx.group_size, w_dtype,
            bitwidth=ctx.bitwidth1)
        W2 = qlora_fallback.materialize_weight(
            ctx.indices2, ctx.lut2, ctx.N, ctx.K, ctx.group_size, w_dtype,
            bitwidth=ctx.bitwidth2)
        grad_x = torch.matmul(grad_y.to(w_dtype), (W1 + W2).to(w_dtype))
        return (grad_x.to(ctx.x_dtype), None, None, None,
                None, None, None, None, None, None)


def fused_qlora_gemm_two_streams(x, indices1, lut1, bitwidth1, indices2,
                                  lut2, bitwidth2, group_size, N, K):
    """Y = X @ (W1 + W2)^T with fused dequant+matmul on BOTH streams (the
    frozen two-stream branch, W10). Returns None if the FLUTE kernel is
    not available (caller falls back to reference)."""
    if not _check_flute_kernel():
        return None
    original_shape = x.shape
    if x.dim() == 3:
        b, s, k = x.shape
        x2 = x.reshape(b * s, k)
    else:
        x2 = x
    y = FusedQLoRAGEMMTwoStreams.apply(x2, indices1, lut1, bitwidth1,
                                       indices2, lut2, bitwidth2,
                                       group_size, N, K)
    if len(original_shape) == 3:
        y = y.view(original_shape[0], original_shape[1], N)
    return y


class FusedQLoRAGEMM(torch.autograd.Function):
    """Fused dequant + matmul for the QLoRA frozen branch.

    Forward:  Y = X @ W^T  via the FLUTE kernel (W never in DRAM)
    Backward: dL/dX = dL/dY @ W  via the fused backward kernel (W never in DRAM)

    indices/lut are ctx attributes (frozen buffers, untracked by autograd,
    survive checkpoint recompute) — see the module docstring.
    """

    @staticmethod
    def forward(ctx, x, indices, lut, bitwidth, group_size, N, K):
        import flute_extended

        original_dtype = x.dtype
        xh = x.half() if x.dtype != torch.float16 else x

        y = flute_extended.qgemm_per_group_lut(
            xh, indices, lut,
            bitwidth=int(bitwidth),
            group_size=int(group_size),
            indices_layout=f"idx{int(bitwidth)}",
        )

        # Frozen buffers as ctx attributes: they survive checkpoint
        # recompute, unlike save_for_backward tensors.
        ctx.indices = indices
        ctx.lut = lut
        ctx.bitwidth = int(bitwidth)
        ctx.group_size = int(group_size)
        ctx.N = int(N)
        ctx.K = int(K)
        ctx.x_dtype = original_dtype

        return y.to(original_dtype)

    @staticmethod
    def backward(ctx, grad_y):
        indices = ctx.indices
        lut = ctx.lut

        if _check_backward_kernel():
            import flute_train_kernels as ftk
            grad_y_h = grad_y.half() if grad_y.dtype != torch.float16 else grad_y
            grad_x = _ftk_backward_bitwidth(
                ftk.fused_backward_gemm,
                grad_y_h.contiguous(), indices.contiguous(),
                lut.contiguous(), ctx.N, ctx.K, ctx.group_size,
                bitwidth=ctx.bitwidth,
                indices_layout=f"idx{ctx.bitwidth}")
            if _debug_nan_enabled() and torch.isnan(grad_x).any():
                raise RuntimeError(
                    f"fused_backward_gemm produced NaN "
                    f"(N={ctx.N}, K={ctx.K}, group_size={ctx.group_size})")
            return grad_x.to(ctx.x_dtype), None, None, None, None, None, None

        # Reference path: cached materialized W + cuBLAS (qlora_fallback.py).
        import qlora_fallback
        w_dtype = grad_y.dtype
        if w_dtype == torch.float32 and ctx.x_dtype != torch.float32:
            w_dtype = ctx.x_dtype
        W = qlora_fallback.materialize_weight(
            indices, lut, ctx.N, ctx.K, ctx.group_size, w_dtype,
            bitwidth=ctx.bitwidth)
        grad_x = torch.matmul(grad_y.to(w_dtype), W.to(w_dtype))
        return grad_x.to(ctx.x_dtype), None, None, None, None, None, None


def fused_qlora_gemm(x, indices, lut, bitwidth, group_size, N, K):
    """Compute Y = X @ W^T with fused dequant+matmul. Returns None if
    the FLUTE kernel is not available (caller falls back to reference)."""
    if not _check_flute_kernel():
        return None
    original_shape = x.shape
    if x.dim() == 3:
        b, s, k = x.shape
        x2 = x.reshape(b * s, k)
    else:
        x2 = x
    y = FusedQLoRAGEMM.apply(x2, indices, lut, bitwidth, group_size, N, K)
    if len(original_shape) == 3:
        y = y.view(original_shape[0], original_shape[1], N)
    return y


def fused_gemm_eligible(module, lut_trainable: bool = False) -> bool:
    """The kernel-path eligibility. lut_trainable=True (the W5 mode,
    selected by trainer --lut-path kernel): additionally requires the
    lut_grad_scatter kernel — a trainable LUT is servable on the kernel
    path ONLY when its gradient exists there (the W4 idx4-refusal
    lifted, not bypassed). Sub-4-bit bitwidths additionally require the
    idxN-capable build (the bitwidth pybind parameter) — a stale
    pre-idxN extension resolves the module to the reference path at
    attach instead of failing mid-backward.

    W10: two-stream modules (has_stream2) extend the SAME contract to
    stream 2 — its bitwidth2 needs the idxN build when sub-4, a
    trainable lut2 needs the scatter kernel, and the frozen mode needs
    the fp16/CUDA residency for BOTH streams. (The stream-2 geometry —
    blob numel and LUT palette/groups — is validated at construction
    by PalettizedLinear; eligibility re-checks only the kernel-build
    gates.)"""
    if not _check_flute_kernel(): return False
    if not torch.cuda.is_available(): return False
    if module.N % 128 != 0 or module.K % 64 != 0: return False
    has2 = bool(getattr(module, "has_stream2", False))
    widths = [int(getattr(module, "bitwidth", 4))]
    if has2:
        widths.append(int(getattr(module, "bitwidth2", 4)))
    for w in widths:
        if w < 4:
            try:
                import flute_train_kernels as ftk
                if not ftk.idxn_available():
                    return False
            except Exception:
                return False
    lut1_trainable = bool(module.lut_trainable)
    lut2_trainable = bool(has2 and getattr(module, "lut2_trainable", False))
    if lut1_trainable or lut2_trainable:
        if not lut_trainable: return False
        if not _check_lut_grad_kernel(): return False
    if not lut_trainable:
        if module.lut.dtype != torch.float16: return False
        if has2 and module.lut2.dtype != torch.float16: return False
    if not module.indices.is_cuda: return False
    if has2 and not module.indices2.is_cuda: return False
    if not module.lut_trainable and not module.lut.is_cuda: return False
    if has2 and not lut2_trainable and not module.lut2.is_cuda: return False
    return True


__all__ = ["FusedQLoRAGEMM", "FusedQLoRAGEMMTrainLUT",
           "FusedQLoRAGEMMTwoStreams", "FusedQLoRAGEMMTrainLUTTwoStreams",
           "fused_qlora_gemm", "fused_qlora_gemm_train_lut",
           "fused_qlora_gemm_two_streams",
           "fused_qlora_gemm_train_lut_two_streams",
           "fused_gemm_eligible", "fused_backward_available",
           "flute_import_error", "backward_import_error",
           "lut_grad_import_error", "_lut_grad_scatter_reference",
           "train_lut_reference_forward", "train_lut_reference_backward",
           "train_lut_two_streams_reference_forward",
           "train_lut_two_streams_reference_backward"]
