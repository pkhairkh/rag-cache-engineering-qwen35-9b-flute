"""
FLUTE-Extended: quantized GEMM with per-group LUT and Tensor Core
acceleration.

Backends:
  - "debug_simple"     : scalar-load Tensor Core differential twin of
                         cutlass_streaming (bring-up/debugging ONLY;
                         consumes the LOGICAL packed layout through _C)
  - "cutlass_streaming": production kernel. Consumes the idxN indices
                         layouts (register-direct dequant: no sW round
                         trip, 3-stage A pipeline, paired-LUT, m-major
                         grid) at bitwidths 1/2/3/4

Public API:
    qgemm_per_group_lut(
        A, indices, lut, *, bitwidth=4, group_size,
        backend="auto", indices_layout="idx4"
    ) -> Tensor

indices_layout:
    "idxN" (N in {1,2,3,4}, written "idx1"/"idx2"/"idx3"/"idx4"):
             the canonical on-disk artifact layout family — a flat tensor
             of N*K*b/8 bytes produced by flute_extended.idxN.pack_idxN
             (see docs/DEQUANT_SPEC.md section 8 and
             flute_extended/idxN.py). The kernel dequantizes straight
             into MMA registers. Requires N % 128 == 0 and K % 64 == 0 —
             the packer refuses other shapes and the kernel rejects the
             combination loudly. "idx4" is byte-identical to the
             historical fragment_v1 layout; the sub-4-bit layouts are
             the same chunk tiling at 16*b bytes per thread chunk.
             indices_layout must match bitwidth (idx2 <-> bitwidth=2).

Environment:
    FLUTE_GS32_BK=64   opt group_size=32 layers into BK=64 K-tiles
                        (deep-tile experiment; default is BK=32).
"""

from __future__ import annotations
from typing import Optional, Literal

import torch

from ._C import (
    qgemm_per_group_lut as _qgemm_per_group_lut,
    qgemm_debug_simple,
    qgemm_cutlass_streaming,
)

# W26: the dual-stream fused decode kernel. Imported defensively so an
# un-rebuilt extension (pre-W26 _C) keeps the package importable — the
# module then reports dual_stream_available() == False and
# palettized_modules routes to the two-launch path.
try:
    from ._C import qgemm_dual_stream as _qgemm_dual_stream
except ImportError:            # pragma: no cover - pre-W26 extension
    _qgemm_dual_stream = None

# W27: the M=1 decode GEMV. Same defensive-import discipline — a pre-W27
# build (W26's dual kernel present, GEMV absent) keeps the package
# importable, reports gemv_stream_available() == False, and the module
# routes M == 1 to the dual kernel / two-launch path.
try:
    from ._C import qgemm_gemv_stream as _qgemm_gemv_stream
except ImportError:            # pragma: no cover - pre-W27 extension
    _qgemm_gemv_stream = None

# W28: the FHT-fused decode GEMV (the boundary-fold rotation runs as the
# kernel's prologue). Same defensive-import discipline once more — a
# pre-W28 build (W27's GEMV present, the fused entry absent) keeps the
# package importable, reports gemv_fht_stream_available() == False, and
# the module routes M == 1 to the W27 FHT-then-GEMV pair (a performance
# fallback, never a correctness one).
try:
    from ._C import qgemm_gemv_fht_stream as _qgemm_gemv_fht_stream
except ImportError:            # pragma: no cover - pre-W28 extension
    _qgemm_gemv_fht_stream = None

# W29: the GEMV v2 decode kernel (split-K grid + double-buffered K loop +
# the wide table: every (B1, B2) pair, GS 16..2048, residual rank <= 32).
# Same defensive-import discipline as W26/W27/W28 — a pre-W29 build keeps
# the package importable, reports gemv2_*_available() == False, and the
# module routes M == 1 to the W27/W28/dual/two-launch ladder (a
# performance fallback, never a correctness one).
try:
    from ._C import qgemm_gemv2_stream as _qgemm_gemv2_stream
except ImportError:            # pragma: no cover - pre-W29 extension
    _qgemm_gemv2_stream = None

try:
    from ._C import qgemm_gemv2_fht_stream as _qgemm_gemv2_fht_stream
except ImportError:            # pragma: no cover - pre-W29 extension
    _qgemm_gemv2_fht_stream = None

Backend = Literal[
    "debug_simple",
    "cutlass_streaming",
    "auto",
]

# The supported indices layouts of the idxN family. Kept as a parameter
# (instead of being derived from bitwidth) so call sites state the format
# explicitly and the artifact format is auditable from every invocation;
# the wrapper enforces layout/bitwidth consistency.
IndicesLayout = Literal["idx1", "idx2", "idx3", "idx4"]

_BITWIDTH_BY_LAYOUT = {"idx1": 1, "idx2": 2, "idx3": 3, "idx4": 4}


def qgemm_per_group_lut(
    A: torch.Tensor,
    indices: torch.Tensor,
    lut: torch.Tensor,
    bitwidth: int = 4,
    group_size: int = 32,
    backend: Backend = "auto",
    indices_layout: IndicesLayout = "idx4",
) -> torch.Tensor:
    """Compute Y = X @ W^T where W is palettized.

    Args:
        A        : [M, K] FP16, the input activations X.
        indices  : flat uint8 tensor in the idxN layout (N*K*b/8 bytes;
                   produced by flute_extended.idxN.pack_idxn; a 2-D view
                   with the same byte count is accepted).
        lut      : [num_groups, 2**b] FP16, the lookup table.
        bitwidth : 1, 2, 3 or 4; must match indices_layout (idx2 <-> 2).
        group_size: 32 (MLP) or 64 (attention). LUT group size along N. The
                    streaming kernel's K-tile depth BK is decoupled from it
                    (see the module docstring's environment section).
        backend  : see module docstring. "auto" picks "cutlass_streaming".
        indices_layout: "idx1"/"idx2"/"idx3"/"idx4" (register-direct
                    dequant; requires N % 128 == 0 and K % 64 == 0 — all
                    Qwen3.5-9B layers qualify).

    Returns:
        C : [M, N] FP16.
    """
    if backend == "auto":
        backend = "cutlass_streaming"

    if indices_layout not in _BITWIDTH_BY_LAYOUT:
        raise ValueError(
            f"indices_layout={indices_layout!r} is not supported; the "
            f"supported layouts are {sorted(_BITWIDTH_BY_LAYOUT)} "
            f"(q_layout=1, the idxN family).")
    layout_bits = _BITWIDTH_BY_LAYOUT[indices_layout]
    if int(bitwidth) != layout_bits:
        raise ValueError(
            f"indices_layout={indices_layout!r} implies bitwidth="
            f"{layout_bits}, but bitwidth={bitwidth} was passed; the two "
            f"must agree.")
    q_layout = 1

    # The C++ dispatcher takes (A, indices, lut, W_dense, ...).
    # For the quantized backends W_dense is an empty tensor.
    W_empty = torch.empty(0, dtype=torch.float16, device=A.device)
    return _qgemm_per_group_lut(
        A, indices, lut, W_empty,
        int(bitwidth), int(group_size), backend, q_layout,
    )


# ---------------------------------------------------------------------------
# W26: the dual-stream fused decode kernel
# ---------------------------------------------------------------------------

# The compiled (bitwidth, bitwidth2) pairs — the five two-stream composites
# the deployed artifacts contain (parsed from the layer metadata census:
# (1,4) x45, (4,1) x24, (3,3) x61, (2,3) x3, (2,4) x6) plus the
# single-stream riders (b, 0). This must mirror the C++ FLUTE_DUAL_PAIR
# table in src/kernel_cutlass_streaming.cu exactly.
DUAL_STREAM_PAIRS = frozenset({
    (1, 4), (4, 1), (2, 3), (2, 4), (3, 3), (3, 0), (4, 0),
})

# The compiled group sizes (the deployed range: 240/248 modules; other GS
# values keep the two-launch route). Mirrors FLUTE_DISPATCH_DUAL_GS.
DUAL_STREAM_GROUP_SIZES = frozenset({64, 128, 256, 512})

# The deployment M gate (the decode shape; PERFORMANCE.md §8's regime).
# Above it the module keeps the pre-W26 route so prefill/PPL numerics stay
# byte-identical to the two-launch path.
DUAL_STREAM_MAX_M = 16


def dual_stream_available() -> bool:
    """True when the built extension exports qgemm_dual_stream (a W26+
    build). False on an un-rebuilt extension — the caller routes to the
    two-launch path (a performance fallback, never a correctness one)."""
    return _qgemm_dual_stream is not None


def dual_stream_supported(bitwidth: int, bitwidth2: int, group_size: int,
                           N: int, K: int) -> bool:
    """The pure-Python mirror of the C++ dual dispatch contract — the
    deployment gate palettized_modules consults BEFORE calling
    qgemm_dual_stream (keeping the fused entry loud-only: anything that
    reaches _C.qgemm_dual_stream and fails a check is a real contract
    violation, not a shape the wrapper should have routed elsewhere)."""
    if _qgemm_dual_stream is None:
        return False
    b1 = int(bitwidth)
    b2 = int(bitwidth2) if bitwidth2 else 0
    return (
        (b1, b2) in DUAL_STREAM_PAIRS
        and int(group_size) in DUAL_STREAM_GROUP_SIZES
        and int(N) % 128 == 0
        and int(K) % 64 == 0
    )


def qgemm_dual_stream(
    A: torch.Tensor,
    indices: torch.Tensor,
    lut: torch.Tensor,
    bitwidth: int,
    group_size: int,
    indices2: Optional[torch.Tensor] = None,
    lut2: Optional[torch.Tensor] = None,
    bitwidth2: int = 0,
    resB: Optional[torch.Tensor] = None,
    resA: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The W26 fused decode GEMM — ONE launch for the whole per-module op
    chain of a two-stream (Route-A pair-composite) module:

        C = A @ W1^T  (+ A @ W2^T)  +  (A @ resB^T) @ resA^T  +  bias

    Both streams share the mma A fragments; the stream sum, the rank-<=16
    residual and the bias all accumulate in fp32 inside the kernel and
    round to fp16 ONCE (the module reference path's association — see the
    kernel numerics note in src/kernel_cutlass_streaming.cu).

    Args:
        A        : [M, K] FP16 CUDA contiguous (the ROTATED input).
        indices  : stream-1 idxN blob (uint8, flat or 2-D view).
        lut      : [ceil(N/GS), 2^bitwidth] FP16.
        bitwidth : stream-1 width, 1..4.
        group_size: 64, 128, 256 or 512 (the compiled dual range).
        indices2 / lut2 / bitwidth2: the stream-2 set (None/0 = single).
        resB     : (R, K) FP16, 1 <= R <= 16, or None.
        resA     : (N, R) FP16, or None (must pair with resB).
        bias     : (N,) FP16, or None.

    Returns:
        C : [M, N] FP16.

    Raises:
        RuntimeError loudly when the extension predates W26 (rebuild:
        `cd flute_extended && python setup.py build_ext --inplace`) or
        when a contract is violated — callers gate on
        dual_stream_supported() first and route other shapes to the
        two-launch qgemm_per_group_lut path.
    """
    if _qgemm_dual_stream is None:
        raise RuntimeError(
            "flute_extended: qgemm_dual_stream is unavailable — the built "
            "extension predates W26. Rebuild with `cd flute_extended && "
            "python setup.py build_ext --inplace`, or route the call "
            "through the two-launch qgemm_per_group_lut path.")
    dev = A.device
    empty_u8 = torch.empty(0, dtype=torch.uint8, device=dev)
    empty_f16 = torch.empty(0, dtype=torch.float16, device=dev)
    i2 = indices2 if indices2 is not None else empty_u8
    l2 = lut2 if lut2 is not None else empty_f16
    rb = resB if resB is not None else empty_f16
    ra = resA if resA is not None else empty_f16
    bi = bias if bias is not None else empty_f16
    return _qgemm_dual_stream(
        A, indices, lut, int(bitwidth),
        i2, l2, int(bitwidth2),
        rb, ra, bi,
        int(group_size),
    )


# ---------------------------------------------------------------------------
# W27: the M=1 decode GEMV
# ---------------------------------------------------------------------------

# Same width-pair table as the dual kernel (the GEMV compiles ONE
# instantiation per pair — GS is runtime, not a template dimension).
# Must mirror the C++ FLUTE_GEMV_PAIR table in
# src/kernel_cutlass_streaming.cu exactly.
GEMV_STREAM_PAIRS = frozenset({
    (1, 4), (4, 1), (2, 3), (2, 4), (3, 3), (3, 0), (4, 0),
})

# The runtime-GS range — WIDER than the dual table: the GEMV has no
# GS-shaped smem or tile geometry (the group only indexes the LUT rows
# for the per-warp palette), so GS 1024/2048 ride the same kernel.
# Modules the dual route never covered (its GS table stops at 512) now
# fuse at M == 1. GS 16/32 stay on the two-launch route.
GEMV_STREAM_GROUP_SIZES = frozenset({64, 128, 256, 512, 1024, 2048})


def gemv_stream_available() -> bool:
    """True when the built extension exports qgemm_gemv_stream (a W27+
    build). False on a W26 build or older — the caller routes M == 1 to
    the dual kernel (M <= 16) or the two-launch path (a performance
    fallback, never a correctness one)."""
    return _qgemm_gemv_stream is not None


def gemv_stream_supported(bitwidth: int, bitwidth2: int, group_size: int,
                          N: int, K: int, M: int) -> bool:
    """The pure-Python mirror of the C++ GEMV dispatch contract — the
    deployment gate palettized_modules consults BEFORE calling
    qgemm_gemv_stream (keeping the fused entry loud-only: anything that
    reaches _C.qgemm_gemv_stream and fails a check is a real contract
    violation, not a shape the wrapper should have routed elsewhere)."""
    if _qgemm_gemv_stream is None:
        return False
    b1 = int(bitwidth)
    b2 = int(bitwidth2) if bitwidth2 else 0
    return (
        int(M) == 1
        and (b1, b2) in GEMV_STREAM_PAIRS
        and int(group_size) in GEMV_STREAM_GROUP_SIZES
        and int(N) % 128 == 0
        and int(K) % 64 == 0
        # the smem ceiling of the x-staging tile (the C++ TORCH_CHECK
        # bounds K the same way): 2*K + 2112 <= 99*1024
        and 2 * int(K) + 2112 <= 99 * 1024
    )


# The fused smem tail: the FHT's fp32 staging of the largest segment
# (b_max = the largest power of two <= K = the first fht segment), with
# red/xBf overlaid in its dead tail (>= 2112 B whenever the shape is
# admitted). Mirrors the C++ launcher's formula bit for bit.
def _gemv_fht_smem(K: int) -> int:
    b_max = 1 << (int(K).bit_length() - 1)
    tail = 4 * b_max if 4 * b_max >= 2112 else 2112
    return 2 * int(K) + tail


def gemv_fht_stream_available() -> bool:
    """True when the built extension exports qgemm_gemv_fht_stream (a
    W28+ build). False on a W27 build or older — the caller routes
    M == 1 to the W27 FHT+GEMV pair (a performance fallback, never a
    correctness one)."""
    return _qgemm_gemv_fht_stream is not None


def gemv_fht_stream_supported(bitwidth: int, bitwidth2: int,
                              group_size: int, N: int, K: int, M: int) -> bool:
    """The pure-Python mirror of the C++ FHT-fused GEMV dispatch
    contract — the deployment gate palettized_modules consults BEFORE
    calling the fused entry (same loud-only discipline as
    gemv_stream_supported). Identical to the W27 GEMV gate plus the
    fused kernel's bigger smem bound: 2*K + max(4*b_max, 2112) <= 99 KB
    (b_max = the largest power of two <= K). K=4096 needs 24576 B,
    K=12288 (8192+4096 segments) needs 57344 B — both admitted; K=32768
    would need 2*32768 + 131072 B and keeps the explicit-FHT route."""
    if _qgemm_gemv_fht_stream is None:
        return False
    b1 = int(bitwidth)
    b2 = int(bitwidth2) if bitwidth2 else 0
    return (
        int(M) == 1
        and (b1, b2) in GEMV_STREAM_PAIRS
        and int(group_size) in GEMV_STREAM_GROUP_SIZES
        and int(N) % 128 == 0
        and int(K) % 64 == 0
        and _gemv_fht_smem(int(K)) <= 99 * 1024
    )


def qgemm_gemv_stream(
    A: torch.Tensor,
    indices: torch.Tensor,
    lut: torch.Tensor,
    bitwidth: int,
    group_size: int,
    indices2: Optional[torch.Tensor] = None,
    lut2: Optional[torch.Tensor] = None,
    bitwidth2: int = 0,
    resB: Optional[torch.Tensor] = None,
    resA: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    rot_signs: Optional[torch.Tensor] = None,
    awq_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The W27 decode GEMV — ONE memory-bound launch for the whole
    per-module op chain at the M == 1 decode shape:

        C = A @ W1^T  (+ A @ W2^T)  +  (A @ resB^T) @ resA^T  +  bias

    No tensor cores: at M == 1 an m16n8k16 atom does 16-32x the
    necessary row work, so this kernel is shaped to the CODE STREAM
    instead — the idxN chunk reads stay exactly the coalesced warp
    pattern the layout was designed for, the A row is staged to shared
    memory once, and the per-group LUT palette is held in warp registers
    and served by shfl.sync.idx (see the W27 block comment in
    src/kernel_cutlass_streaming.cu for the full design and its doc
    citations).

    Numerics (documented, decode-only): the streams, the rank-<=16
    residual and the bias accumulate in fp32 and round to fp16 ONCE —
    but in the GEMV's own fixed order (ascending-k per lane, the 4
    i-lane butterfly, the 4 K-quarter adds), which is deterministic yet
    NOT bit-identical to either the two-launch path or the W26 dual
    kernel. Routed only at M == 1, so prefill/PPL numerics stay
    byte-identical to the pre-W27 route; decode greedy tokens may flip
    on near-ties (first-divergence is the aggregate to read).

    W28: pass rot_signs (the module's (K,) fp32 fold sign vector) to run
    the FHT-FUSED variant — A is then the UNROTATED raw [1, K] fp16 row,
    and the boundary-fold rotation x @ T runs as the kernel's prologue
    (ONE launch where W27 needed an FHT kernel + the GEMV; the prologue
    is a line-for-line transcription of fht_forward / fht_forward_awq,
    so the decode outputs are bit-identical to the two-kernel chain).
    awq_scale (the (K,) fp32 legacy rotate-then-AWQ compensation vector)
    selects the compensated transform ((A*s) @ T)/s — pass None for the
    plain rotation. rot_signs=None keeps the W27 contract (A already
    rotated; awq_scale without rot_signs is a contract error).

    Args:
        A        : [1, K] FP16 CUDA contiguous — the ROTATED input when
                   rot_signs is None, the UNROTATED raw input when the
                   FHT is fused. M MUST be 1 (the C++ entry refuses
                   other M loudly).
        indices  : stream-1 idxN blob (uint8, flat or 2-D view).
        lut      : [ceil(N/GS), 2^bitwidth] FP16.
        bitwidth : stream-1 width, 1..4.
        group_size: 64, 128, 256, 512, 1024 or 2048 (the runtime-GS
                    range — wider than the dual kernel's).
        indices2 / lut2 / bitwidth2: the stream-2 set (None/0 = single).
        resB     : (R, K) FP16, 1 <= R <= 16, or None.
        resA     : (N, R) FP16, or None (must pair with resB).
        bias     : (N,) FP16, or None.
        rot_signs: (K,) FP32 CUDA — the fold's sign vector. None keeps
                   the W27 unfused contract; given, the FHT runs inside
                   the kernel (requires a W28+ build — loud refusal
                   otherwise; callers gate on gemv_fht_stream_supported).
        awq_scale: (K,) FP32 CUDA — the rotate-then-AWQ compensation
                   vector, or None for the plain rotation (only valid
                   with rot_signs; coerced to fp32/contiguous like
                   fht.py's s).

    Returns:
        C : [1, N] FP16.

    Raises:
        RuntimeError loudly when the extension predates the needed
        round (W27/W28; rebuild: `cd flute_extended && python setup.py
        build_ext --inplace`) or when a contract is violated — callers
        gate on gemv_stream_supported() / gemv_fht_stream_supported()
        first and route M 2..16 through qgemm_dual_stream, larger
        shapes through the two-launch qgemm_per_group_lut path.
    """
    if rot_signs is None and awq_scale is not None:
        raise ValueError(
            "qgemm_gemv_stream: awq_scale only exists as a companion to "
            "rot_signs (the AWQ compensation is a correction to the "
            "rotated fold) — pass rot_signs too, or drop awq_scale")

    if rot_signs is None:
        # ---- the W27 unfused contract (A already rotated) ----------------
        if _qgemm_gemv_stream is None:
            raise RuntimeError(
                "flute_extended: qgemm_gemv_stream is unavailable — the "
                "built extension predates W27. Rebuild with `cd "
                "flute_extended && python setup.py build_ext --inplace`, "
                "or route the call through the dual/two-launch paths.")
        dev = A.device
        empty_u8 = torch.empty(0, dtype=torch.uint8, device=dev)
        empty_f16 = torch.empty(0, dtype=torch.float16, device=dev)
        i2 = indices2 if indices2 is not None else empty_u8
        l2 = lut2 if lut2 is not None else empty_f16
        rb = resB if resB is not None else empty_f16
        ra = resA if resA is not None else empty_f16
        bi = bias if bias is not None else empty_f16
        return _qgemm_gemv_stream(
            A, indices, lut, int(bitwidth),
            i2, l2, int(bitwidth2),
            rb, ra, bi,
            int(group_size),
        )

    # ---- the W28 FHT-fused contract (A is the UNROTATED row) -------------
    if _qgemm_gemv_fht_stream is None:
        raise RuntimeError(
            "flute_extended: the FHT-fused GEMV (qgemm_gemv_stream with "
            "rot_signs) is unavailable — the built extension predates "
            "W28. Rebuild with `cd flute_extended && python setup.py "
            "build_ext --inplace`, or call without rot_signs (the W27 "
            "FHT-then-GEMV pair).")
    if _qgemm_gemv_stream is None:      # pragma: no cover - W28 carries W27
        raise RuntimeError(
            "flute_extended: qgemm_gemv_stream is unavailable — the "
            "built extension predates W27 (a W28 build always carries "
            "both symbols; this is an import-time inconsistency).")

    K = int(A.shape[1])
    if rot_signs.dim() != 1 or int(rot_signs.numel()) != K:
        raise ValueError(
            f"qgemm_gemv_stream: rot_signs must be a (K,) vector "
            f"matching A's last dim (got {tuple(rot_signs.shape)}, "
            f"K={K})")
    signs = rot_signs.detach()
    if (signs.device != A.device or signs.dtype != torch.float32
            or not signs.is_contiguous()):
        signs = signs.to(A.device, torch.float32).contiguous()

    dev = A.device
    s = torch.empty(0, dtype=torch.float32, device=dev)
    if awq_scale is not None:
        if awq_scale.dim() != 1 or int(awq_scale.numel()) != K:
            raise ValueError(
                f"qgemm_gemv_stream: awq_scale must be a (K,) vector "
                f"matching A's last dim (got {tuple(awq_scale.shape)}, "
                f"K={K})")
        s = awq_scale.detach()
        if (s.device != A.device or s.dtype != torch.float32
                or not s.is_contiguous()):
            s = s.to(A.device, torch.float32).contiguous()

    empty_u8 = torch.empty(0, dtype=torch.uint8, device=dev)
    empty_f16 = torch.empty(0, dtype=torch.float16, device=dev)
    i2 = indices2 if indices2 is not None else empty_u8
    l2 = lut2 if lut2 is not None else empty_f16
    rb = resB if resB is not None else empty_f16
    ra = resA if resA is not None else empty_f16
    bi = bias if bias is not None else empty_f16
    return _qgemm_gemv_fht_stream(
        A, indices, lut, int(bitwidth),
        i2, l2, int(bitwidth2),
        rb, ra, bi,
        int(group_size),
        signs, s,
    )



# ---------------------------------------------------------------------------
# W29: the GEMV v2 — split-K + double-buffer + the wide table
# ---------------------------------------------------------------------------

# The compiled width-pair table: EVERY (B1, B2) with B1 in 1..4, B2 in
# 0..4 — 20 pairs (the W27/W28 tables carried the 7 pairs the layer
# metadata census listed; the QKV composite components deploy as
# (4,3)/(3,4)/(4,4)/(2,2) pairs and the heads are (4,4), so the v2 table
# closes every pair hole in ONE kernel family). Must mirror the C++
# FLUTE_GEMV2_PAIR table in src/kernel_cutlass_streaming.cu exactly.
GEMV2_STREAM_PAIRS = frozenset({
    (b1, b2) for b1 in range(1, 5) for b2 in range(0, 5)
})

# The runtime-GS range: GS 16/32 join (the W27/W28 tables refused them —
# 6 deployed modules fell to the pre-W26 chain). GS < 64 switches the
# kernel's palette serving from the per-warp shfl.idx registers to a
# per-CTA shared palette.
GEMV2_STREAM_GROUP_SIZES = frozenset({16, 32, 64, 128, 256, 512, 1024, 2048})

# The residual-rank cap: 32 (W27/W28 capped at 16 — lm_head's rank-32
# residual fell out of every fused table and cost the whole pre-W26
# chain on the single largest module).
GEMV2_MAX_RESIDUAL_RANK = 32


def gemv2_split_hint(N: int, K: int) -> int:
    """The pure-Python mirror of the C++ split-K policy
    (gemv2_pick_split): SPLIT so that (N/128)*SPLIT >= 160 (2 waves of
    the 80 SMs), a power of two 1..16 with G % (4*SPLIT) == 0 (every j4
    warp keeps >= 1 g-tile). The probe prints it next to the route so the
    box census can show the grid shape W29 chose per module."""
    tiles = int(N) // 128
    G = int(K) // 64
    if tiles >= 160:
        return 1
    best = 1
    s = 2
    while s <= 16:
        if G < 4 * s or G % (4 * s) != 0:
            break
        best = s
        if tiles * s >= 160:
            break
        s <<= 1
    return best


def _gemv2_tail_need(gs: int, b1: int, b2: int) -> int:
    """red[128][4] + xBf[32] + the GS<64 palettes + the ticket slot,
    16-B aligned — mirrors the C++ tail_need formula bit for bit."""
    pal = 0
    if gs < 64:
        ngrp = 128 // gs
        pal = (ngrp << b1) * 2 + (ngrp << b2) * 2
    return (2048 + 128 + pal + 4 + 15) & ~15


def _gemv2_smem(N: int, K: int, gs: int, b1: int, b2: int,
                fht: bool) -> int:
    """The v2 smem bill: the split's x slice (2*Kc, Kc = K/SPLIT) + the
    tail (max(4*b_max, tail_need) when the FHT is fused — the fp32
    staging — else tail_need). Mirrors the C++ launcher's formula."""
    kc = int(K) // gemv2_split_hint(N, K)
    tail_need = _gemv2_tail_need(gs, b1, b2)
    if fht:
        b_max = 1 << (int(K).bit_length() - 1)
        tail = max(4 * b_max, tail_need)
    else:
        tail = tail_need
    return 2 * kc + tail


def gemv2_stream_available() -> bool:
    """True when the built extension exports qgemm_gemv2_stream (a W29+
    build). False on a W28 build or older — the caller routes M == 1 to
    the W27/W28 GEMV pair, the dual kernel or the two-launch path (a
    performance fallback, never a correctness one)."""
    return _qgemm_gemv2_stream is not None


def gemv2_fht_stream_available() -> bool:
    """True when the built extension exports qgemm_gemv2_fht_stream (a
    W29+ build). Same fallback discipline as gemv2_stream_available."""
    return _qgemm_gemv2_fht_stream is not None


def gemv2_stream_supported(bitwidth: int, bitwidth2: int, group_size: int,
                           N: int, K: int, M: int,
                           residual_rank: int = 0) -> bool:
    """The pure-Python mirror of the C++ GEMV v2 dispatch contract — the
    deployment gate palettized_modules consults BEFORE calling
    qgemm_gemv2_stream (same loud-only discipline as the W27 gate): the
    wide pair table (all 20 pairs), GS 16..2048, residual rank <= 32, the
    idxN layout gates, and the split-K smem bound."""
    if _qgemm_gemv2_stream is None:
        return False
    b1 = int(bitwidth)
    b2 = int(bitwidth2) if bitwidth2 else 0
    if int(M) != 1:
        return False
    if (b1, b2) not in GEMV2_STREAM_PAIRS:
        return False
    if int(group_size) not in GEMV2_STREAM_GROUP_SIZES:
        return False
    if int(N) % 128 != 0 or int(K) % 64 != 0:
        return False
    if int(residual_rank) > GEMV2_MAX_RESIDUAL_RANK:
        return False
    return _gemv2_smem(int(N), int(K), int(group_size), b1, b2,
                       fht=False) <= 99 * 1024


def gemv2_fht_stream_supported(bitwidth: int, bitwidth2: int,
                               group_size: int, N: int, K: int, M: int,
                               residual_rank: int = 0) -> bool:
    """The pure-Python mirror of the C++ FHT-fused GEMV v2 dispatch
    contract — identical to gemv2_stream_supported plus the fused smem
    bound (the FHT fp32 staging: 2*Kc + max(4*b_max, tail_need) <= 99
    KB; K=12288 at SPLIT>=4 fits in ~41 KB — down_proj returns to 2
    CTAs/SM)."""
    if _qgemm_gemv2_fht_stream is None:
        return False
    b1 = int(bitwidth)
    b2 = int(bitwidth2) if bitwidth2 else 0
    if int(M) != 1:
        return False
    if (b1, b2) not in GEMV2_STREAM_PAIRS:
        return False
    if int(group_size) not in GEMV2_STREAM_GROUP_SIZES:
        return False
    if int(N) % 128 != 0 or int(K) % 64 != 0:
        return False
    if int(residual_rank) > GEMV2_MAX_RESIDUAL_RANK:
        return False
    return _gemv2_smem(int(N), int(K), int(group_size), b1, b2,
                       fht=True) <= 99 * 1024


def qgemm_gemv2_stream(
    A: torch.Tensor,
    indices: torch.Tensor,
    lut: torch.Tensor,
    bitwidth: int,
    group_size: int,
    indices2: Optional[torch.Tensor] = None,
    lut2: Optional[torch.Tensor] = None,
    bitwidth2: int = 0,
    resB: Optional[torch.Tensor] = None,
    resA: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    rot_signs: Optional[torch.Tensor] = None,
    awq_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The W29 decode GEMV v2 — ONE memory-bound launch (split-K grid +
    double-buffered K loop) for the whole per-module op chain at the
    M == 1 decode shape:

        C = A @ W1^T  (+ A @ W2^T)  +  (A @ resB^T) @ resA^T  +  bias

    The W27 contract, widened: every (bitwidth, bitwidth2) pair, GS
    16..2048 (GS < 64 switches to the per-CTA shared palette), residual
    rank 1..32 (the heads' r32 fuses). The SPLIT-K grid fills the 80 SMs
    on narrow modules (k/v_proj's 8 tiles run 128 CTAs), the double
    register buffer hides the code-stream DRAM latency, and the
    deterministic fixed-order split fold keeps every replay bit-stable
    (decode greedy tokens may flip on near-ties vs the W27/W28 kernels —
    the documented near-tie contract; PFL/M>1 never routes here).

    Args:
        A        : [1, K] FP16 CUDA contiguous — the ROTATED input when
                   rot_signs is None, the UNROTATED raw input when the
                   FHT is fused.
        indices  : stream-1 idxN blob (uint8, flat or 2-D view).
        lut      : [ceil(N/GS), 2^bitwidth] FP16.
        bitwidth : stream-1 width, 1..4.
        group_size: 16, 32, 64, 128, 256, 512, 1024 or 2048.
        indices2 / lut2 / bitwidth2: the stream-2 set (None/0 = single).
        resB     : (R, K) FP16, 1 <= R <= 32, or None.
        resA     : (N, R) FP16, or None (must pair with resB).
        bias     : (N,) FP16, or None.
        rot_signs: (K,) FP32 CUDA — the fold's sign vector. None keeps
                   the plain contract (A already rotated); given, the
                   boundary-fold FHT runs as the split-CTA prologue
                   (bit-identical to the standalone fht_forward).
        awq_scale: (K,) FP32 CUDA — the rotate-then-AWQ compensation
                   vector, or None for the plain rotation (only valid
                   with rot_signs).

    Returns:
        C : [1, N] FP16.

    Raises:
        RuntimeError loudly when the extension predates W29 (rebuild:
        `cd flute_extended && python setup.py build_ext --inplace`) or
        when a contract is violated — callers gate on
        gemv2_stream_supported() / gemv2_fht_stream_supported() first
        and route M 2..16 through qgemm_dual_stream, larger shapes
        through the two-launch qgemm_per_group_lut path.
    """
    if rot_signs is None and awq_scale is not None:
        raise ValueError(
            "qgemm_gemv2_stream: awq_scale only exists as a companion to "
            "rot_signs (the AWQ compensation is a correction to the "
            "rotated fold) — pass rot_signs too, or drop awq_scale")

    dev = A.device
    empty_u8 = torch.empty(0, dtype=torch.uint8, device=dev)
    empty_f16 = torch.empty(0, dtype=torch.float16, device=dev)
    i2 = indices2 if indices2 is not None else empty_u8
    l2 = lut2 if lut2 is not None else empty_f16
    rb = resB if resB is not None else empty_f16
    ra = resA if resA is not None else empty_f16
    bi = bias if bias is not None else empty_f16

    if rot_signs is None:
        # ---- the plain contract (A already rotated) ----------------------
        if _qgemm_gemv2_stream is None:
            raise RuntimeError(
                "flute_extended: qgemm_gemv2_stream is unavailable — the "
                "built extension predates W29. Rebuild with `cd "
                "flute_extended && python setup.py build_ext --inplace`, "
                "or route the call through the W27 GEMV / dual / "
                "two-launch paths.")
        return _qgemm_gemv2_stream(
            A, indices, lut, int(bitwidth),
            i2, l2, int(bitwidth2),
            rb, ra, bi,
            int(group_size),
        )

    # ---- the FHT-fused contract (A is the UNROTATED row) -------------
    if _qgemm_gemv2_fht_stream is None:
        raise RuntimeError(
            "flute_extended: the FHT-fused GEMV v2 (qgemm_gemv2_stream "
            "with rot_signs) is unavailable — the built extension "
            "predates W29. Rebuild with `cd flute_extended && python "
            "setup.py build_ext --inplace`, or call without rot_signs.")
    K = int(A.shape[1])
    if rot_signs.dim() != 1 or int(rot_signs.numel()) != K:
        raise ValueError(
            f"qgemm_gemv2_stream: rot_signs must be a (K,) vector "
            f"matching A's last dim (got {tuple(rot_signs.shape)}, "
            f"K={K})")
    signs = rot_signs.detach()
    if (signs.device != A.device or signs.dtype != torch.float32
            or not signs.is_contiguous()):
        signs = signs.to(A.device, torch.float32).contiguous()

    s = torch.empty(0, dtype=torch.float32, device=dev)
    if awq_scale is not None:
        if awq_scale.dim() != 1 or int(awq_scale.numel()) != K:
            raise ValueError(
                f"qgemm_gemv2_stream: awq_scale must be a (K,) vector "
                f"matching A's last dim (got {tuple(awq_scale.shape)}, "
                f"K={K})")
        s = awq_scale.detach()
        if (s.device != A.device or s.dtype != torch.float32
                or not s.is_contiguous()):
            s = s.to(A.device, torch.float32).contiguous()

    return _qgemm_gemv2_fht_stream(
        A, indices, lut, int(bitwidth),
        i2, l2, int(bitwidth2),
        rb, ra, bi,
        int(group_size),
        signs, s,
    )

__all__ = [
    "qgemm_per_group_lut",
    "qgemm_debug_simple",
    "qgemm_cutlass_streaming",
    "qgemm_dual_stream",
    "dual_stream_available",
    "dual_stream_supported",
    "DUAL_STREAM_PAIRS",
    "DUAL_STREAM_GROUP_SIZES",
    "DUAL_STREAM_MAX_M",
    "qgemm_gemv_stream",
    "gemv_stream_available",
    "gemv_stream_supported",
    "GEMV_STREAM_PAIRS",
    "GEMV_STREAM_GROUP_SIZES",
    "gemv_fht_stream_available",
    "gemv_fht_stream_supported",
    "qgemm_gemv2_stream",
    "gemv2_stream_available",
    "gemv2_stream_supported",
    "gemv2_fht_stream_available",
    "gemv2_fht_stream_supported",
    "gemv2_split_hint",
    "GEMV2_STREAM_PAIRS",
    "GEMV2_STREAM_GROUP_SIZES",
    "GEMV2_MAX_RESIDUAL_RANK",
]
