#!/usr/bin/env python3
"""
palettized_modules.py — the shared palettized-module layer.

Every consumer of idxN artifacts (activation capture, the strict energy
harness, the greedy-equivalence and perplexity evaluators, and the
sequential LUT distillation) imports its PalettizedLinear / SplitQKV /
loaders from here — one implementation, no consumer-local drift.

Two execution paths:
  * kernel path (default): flute_extended.qgemm_per_group_lut with
    indices_layout=f"idx{bitwidth}" — explicit at every call site;
  * CPU reference path (reference=True): torch dequantization through
    flute_extended/idxN.py (loaded standalone, no CUDA extension
    needed), used by CPU debugging. The reference path
    is mathematically identical to the kernel's dequantization (same LUT
    gather, src/docs/QUANTIZATION.md sections 3 and 4).

Residual branch (LQER serving form):
    y = qgemm_idx4(x) + (x B^T) A^T
with A (N, r) and B (r, K) in fp16 — two small dense GEMMs after the
quantized GEMM, rank r in {16, 32}.

Second stream (Route A, optional): the module and the loader carry
a second canonical idx4 stream (the pair-composite of the hybrid422
recipe) — two dequant GEMMs plus ONE ordered add (stream 1 first),
then the residual branch; indices2=None keeps the legacy single-stream
module byte-identical.

 (the kernel-fusion round): the kernel path gained a
FUSED DECODE ROUTE — at M <= 16 with a compiled (bitwidth, bitwidth2,
group_size) combination, PalettizedLinear.forward issues ONE
flute_extended.qgemm_dual_stream launch (both streams + residual + bias,
fp32-fused), and the legacy rotate-then-AWQ fold runs as ONE
fht.fht_apply_awq kernel. Everything else (prefill/PPL shapes,
unlisted pairs, older extensions, CPU, reference, training) keeps the
older paths byte-identically — see PalettizedLinear._dual_stream_
decode for the gate mirror.

 (the decode-GEMV round): the M == 1 shape — the one the
whole-forward CUDA-graph replays dominate with — routes to ONE
memory-bound flute_extended.qgemm_gemv_stream launch instead (no
tensor cores; runtime GS 64..2048, so GS 1024/2048 modules fuse at M == 1
too). M 2..16 keeps the dual-stream kernel, everything else keeps the
two-launch path — see PalettizedLinear._gemv_decode for that gate
mirror.

 (the FHT-fusion round): at M == 1 the boundary-fold rotation moves
INSIDE the GEMV — PalettizedLinear.forward passes the UNROTATED x plus
the module's rot_signs (and the rotate-then-AWQ scale, when the
artifacts are legacy) to flute_extended.qgemm_gemv_stream, whose kernel
performs the FHT as a prologue bit-identical to the standalone
fht/fht_awq kernels it replaces. The per-module decode chain becomes ONE
launch (281 total — the dense arm's own count, vs its FHT+GEMV 562);
M 2..16 keeps the explicit rotation + dual kernel, everything else is
unchanged. FLUTE_NO_FHT_FUSE=1 forces the pair (the box A/B
switch) — see PalettizedLinear._gemv_fht_decode for that gate mirror.
"""

from __future__ import annotations

import os
import sys
from importlib import util as _importlib_util
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Standalone idxN producer (no CUDA extension import; safe on CPU-only
# machines). The idx4.py twin was removed in the unification (main
# project — not part of this repo); idxN.py is the sole
# producer (pack_idxn(·, 4) is byte-identical to the legacy pack_idx4,
# asserted by its self_test) — so every 4-bit unpack goes through it too.
# --------------------------------------------------------------------------- #
# Path resolution: FLUTE_EXT_DIR env var (if set), then sibling-dir candidates.

_IDXN_CANDIDATES = [
    # this repo's layout first (nested package dir exists after the v1.3
    # wholesale sync; flat kept for FLUTE_EXT_DIR-style trees), then the
    # upstream nested-only candidates
    os.path.join(_HERE, "..", "flute_extended", "flute_extended", "idxN.py"),
    os.path.join(_HERE, "flute_extended", "flute_extended", "idxN.py"),
    os.path.join(_HERE, "..", "..", "flute_extended", "flute_extended", "idxN.py"),
    os.path.join(_HERE, "..", "flute_extended", "idxN.py"),
    os.path.join(_HERE, "flute_extended", "idxN.py"),
    os.path.join(_HERE, "..", "..", "flute_extended", "idxN.py"),
]
_env_flute = os.environ.get("FLUTE_EXT_DIR")
if _env_flute:
    _IDXN_CANDIDATES.insert(0, os.path.join(_env_flute, "idxN.py"))
    _IDXN_CANDIDATES.insert(1, os.path.join(_env_flute, "flute_extended", "idxN.py"))
_idxn_module = None


def _get_idxn():
    """Load flute_extended/idxN.py as a standalone module (the unified
    idx1..idx4 producer)."""
    global _idxn_module
    if _idxn_module is None:
        last_err = None
        for path in _IDXN_CANDIDATES:
            if not os.path.exists(path):
                continue
            try:
                spec = _importlib_util.spec_from_file_location(
                    "flute_idxn_standalone", os.path.abspath(path))
                mod = _importlib_util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _idxn_module = mod
                return _idxn_module
            except Exception as e:
                last_err = e
        raise ImportError(f"flute_extended/idxN.py not found (tried: {_IDXN_CANDIDATES}). "
            f"Set FLUTE_EXT_DIR to your flute_extended checkout."
            + (f" Last error: {last_err}" if last_err else ""))
    return _idxn_module


def reference_dequant(indices_blob, lut, N: int, K: int, group_size: int,
                      bitwidth: int = 4) -> torch.Tensor:
    """CPU-reference dequantization: W[n, k] = LUT[n // gs, idx(n, k)].

    Returns an (N, K) fp32 tensor on `lut`'s device. Uses the canonical
    unpackers, so it is exactly the kernel's semantics (src/docs/QUANTIZATION.md
    sections 2-4): LSB-first fields, the idxN tile permutation undone by
    idx4.unpack_idx4 (b=4) / idxN.unpack_idxn (b in 1..3).

    The unpack runs on CPU (numpy, canonical producer logic) but the
    LUT gather runs on the LUT's device, so a GPU LUT is never staged as a
    full fp32 dense weight in host RAM (~200 MB per matrix at the 9B
    scale). When `lut` is an nn.Parameter with requires_grad, the gather is
    differentiable (straight-through: indices are constants) — this is the
    training path used by the layerwise trainer.
    """
    if bitwidth not in (1, 2, 3, 4):
        raise ValueError(f"reference_dequant supports bitwidths 1-4, got {bitwidth}")
    blob = np.ascontiguousarray(indices_blob.detach().cpu().numpy() if torch.is_tensor(indices_blob)
        else indices_blob, dtype=np.uint8).reshape(-1)
    # The canonical unpacker at every width (b=4 is byte-identical to the
    # legacy idx4 nibble walk — the unification's asserted identity).
    idx = _get_idxn().unpack_idxn(blob, N, K, int(bitwidth))

    lut_t = lut if torch.is_tensor(lut) else torch.from_numpy(np.asarray(lut))
    palette = 1 << bitwidth
    if lut_t.dim() != 2:
        lut_t = lut_t.view(-1, palette)
    n_groups = lut_t.shape[0]

    rg = torch.div(torch.arange(N, device=lut_t.device), int(group_size),
                   rounding_mode="floor").clamp_(max=n_groups - 1)
    idx_t = torch.from_numpy(idx.astype(np.int64))
    if lut_t.device != torch.device("cpu"):
        idx_t = idx_t.to(lut_t.device)
    # W[n, k] = lut[n // group_size, idx[n, k]] (spec §3; differentiable
    # when lut_t is a Parameter — .float is the fp16->fp32 master cast)
    return lut_t[rg].float().gather(1, idx_t)


# --------------------------------------------------------------------------- #
# The FLUTE-extension round: the Fast Hadamard Transform loader.
#
# The Hadamard boundary fold's rotation is served by fht.py (the
# requirement's Phase-3 integration): the (K,) sign vector replaces the
# K x K rotation matrix the old class-level _rot_cache held (~1.5 GB
# fp16 across the 8 unique (seed, k) pairs of the 9B model — plus a
# pageable CPU -> GPU copy of it on EVERY forward, 576 MB for the
# K=12288 down_proj). fht.py is loaded STANDALONE (the idxN pattern:
# spec_from_file_location) so the module layer stays importable on
# CPU-only boxes — the FHT reference butterfly serves the CPU route,
# the CUDA kernel serves the kernel route.
# --------------------------------------------------------------------------- #
_FHT_CANDIDATES = [
    os.path.join(_HERE, "..", "flute_extended", "fht.py"),
    os.path.join(_HERE, "flute_extended", "fht.py"),
    os.path.join(_HERE, "..", "..", "flute_extended", "fht.py"),
]
_env_flute_fht = os.environ.get("FLUTE_EXT_DIR")
if _env_flute_fht:
    _FHT_CANDIDATES.insert(0, os.path.join(_env_flute_fht, "fht.py"))
_fht_module = None


def _get_fht():
    """Load flute_extended/fht.py as a standalone module (lazy, cached)."""
    global _fht_module
    if _fht_module is None:
        last_err = None
        for path in _FHT_CANDIDATES:
            if not os.path.exists(path):
                continue
            try:
                spec = _importlib_util.spec_from_file_location(
                    "flute_fht_standalone", os.path.abspath(path))
                mod = _importlib_util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _fht_module = mod
                return _fht_module
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise ImportError(f"flute_extended/fht.py not found (tried: {_FHT_CANDIDATES}). "
            f"Set FLUTE_EXT_DIR to your flute_extended checkout."
            + (f" Last error: {last_err}" if last_err else ""))
    return _fht_module


def _rotation_signs_for(K: int, seed: int) -> torch.Tensor:
    """The boundary-fold sign vector — the loader-side twin of the
    palettizer's draw (generator seed+4242, randint(0,2,(K,))*2-1).

    Delegates to fht.rotation_signs so the two never drift; bitwise
    identical to the K x K builder's column signs the old code used.
    """
    return _get_fht().rotation_signs(int(K), int(seed))


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

LAYOUT_AUDIT = {"idx4": 0, "idxN": 0}


def _bits_of_layout(layout: str) -> int:
    """The bit width an indices_layout name declares ("idx4" -> 4,
    "idx2" -> 2, ...). Raises on anything else."""
    if isinstance(layout, str) and layout.startswith("idx") \
            and layout[3:].isdigit() and int(layout[3:]) in (1, 2, 3, 4):
        return int(layout[3:])
    raise ValueError(f"not an idxN layout name: {layout!r}")


def _is_idx4_artifact_name(index_file: str) -> bool:
    """True for canonical '<san>.idx4' and the stream-tagged variant
    '<san>.idx4.<tag>' (the '.2' pair-composite stream). Tag must be a
    single non-empty token (no dots, no path separators)."""
    if index_file.endswith(".idx4"):
        return True
    if ".idx4." in index_file:
        tag = index_file.split(".idx4.", 1)[1]
        return tag != "" and "." not in tag and "/" not in tag
    return False


def _is_idxn_artifact_name(index_file: str) -> bool:
    """True for canonical '<san>.idx{b}' (b in 1..4) and the stream-tagged
    variant '<san>.idx{b}.<tag>'. Same single-token tag rule as idx4."""
    if not isinstance(index_file, str):
        return False
    import re
    m = re.match(r"^.*\.idx[1-4](\.[^.]+)?$", index_file)
    return m is not None and "/" not in index_file


def _bits_of_artifact_name(index_file: str) -> int:
    """The idxN width an artifact filename declares ('<san>.idx3.2' -> 3;
    '<san>.idx4' -> 4). Raises on anything else."""
    import re
    m = re.match(r"^.*\.idx([1-4])(\.[^.]+)?$", str(index_file))
    if m is None:
        raise ValueError(f"not an idxN artifact name: {index_file!r}")
    return int(m.group(1))


def _lut_name_of(index_file):
    """The LUT file name paired with an idxN index file, following the
    writer's naming convention (main project: palettize_qwen3_5_9b.py::_lut_name_of,
    mirrored verbatim): '<san>.idx{b}' -> '<san>.lut_scalar',
    '<san>.idx{b}.<tag>' -> '<san>.lut_scalar.<tag>'."""
    if not index_file:
        return None
    import re
    m = re.match(r"^(?P<base>.*)\.idx[1-4](?P<tag>\..*)?$", str(index_file))
    if m is None:
        return None
    return m.group("base") + ".lut_scalar" + (m.group("tag") or "")


def validate_indices_layout(meta: Dict, ctx: str) -> None:
    """Refuse anything that is not an idxN artifact with a matching
    indices_layout ('idx1'..'idx4').

    A misread blob produces garbage with no shape error (byte count is
    identical across layouts), so every mismatch fails loudly here —
    including legacy-era metadata that carries the old 'layout' key.
    The stream-tagged names ('<san>.idx4.2', the pair-composite
    stream) validate through the same gate — every member of a
    declared streams set is layout-checked (no member skips).
    sub-4-bit layouts (.idx1/.idx2/.idx3) validate through the same
    gate, and the layout string, the file name and the metadata's
    bitwidth field must AGREE.
    """
    raw = meta.get("indices_layout") or meta.get("layout")
    try:
        layout_bits = _bits_of_layout(raw)
    except ValueError:
        raise ValueError(f"{ctx}: indices_layout={raw!r} is not an idxN layout "
            f"('idx1'..'idx4'). Legacy-era artifacts (idx4 / "
            f".idx4.fd / unpacked legacy) are not readable; re-run "
            f"palettization with the current scripts.")
    index_file = str(meta.get("index_file", ""))
    if not _is_idxn_artifact_name(index_file):
        raise ValueError(f"{ctx}: index file {index_file!r} is not an idxN artifact; "
            f"re-run palettization with the current scripts.")
    name_bits = _bits_of_artifact_name(index_file)
    if name_bits != layout_bits:
        raise ValueError(f"{ctx}: index file {index_file!r} declares idx{name_bits} but "
            f"indices_layout={raw!r}; the two must agree (no silent "
            f"cross-width reads)")
    meta_bw = meta.get("bitwidth")
    if meta_bw is not None and int(meta_bw) != layout_bits:
        raise ValueError(f"{ctx}: metadata bitwidth {meta_bw} != the layout's "
            f"{layout_bits} bits ({index_file!r})")
    if layout_bits == 4:
        LAYOUT_AUDIT["idx4"] += 1
    else:
        LAYOUT_AUDIT["idxN"] += 1


def _check_sha(path: str, expected: Optional[str], ctx: str,
               verify: bool = True) -> None:
    if not verify or not expected:
        return
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise ValueError(f"{ctx}: SHA256 mismatch for {path} "
                         f"(metadata {expected[:16]}..., got {got[:16]}...)")


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #

class PalettizedLinear(nn.Module):
    """Linear layer with on-the-fly FLUTE dequantization (idx4 layout).

    indices/lut (and the optional residual factors resA/resB) are registered
    as non-persistent buffers so that
    (a) model.to(device) moves them, and
    (b) any size accounting sees them (plain attributes would under-report
        the palettized model).

    Residual branch (LQER serving form):
        y = qgemm_idx4(x) + (x resB^T) resA^T

    Second stream : stream 2 is
    another canonical idx4+LUT16 set (the pair-composite stream of the
    hybrid422 recipe). The forward is TWO dequant GEMMs plus ONE
    ordered add (stream 1 pinned first), then the residual branch; the
    kernel path is two qgemm calls plus one fp16 add — zero kernel
    change. indices2=None (the default) registers no stream-2 buffers
    and keeps the legacy single-stream module byte-identical.

     (the fused decode route): at the M <= 16 decode shape, when the
    deployed (bitwidth, bitwidth2, group_size) sits in the compiled
    dual-kernel table, forward collapses the WHOLE chain — both
    streams' qgemm, the stream add, the (x @ resB^T) @ resA^T residual
    pair and the bias — into ONE qgemm_dual_stream launch (fp32-fused
    accumulation, the reference path's association; see
    _dual_stream_decode and flute_extended/flute_extended/__init__.py).
    Prefill/PPL shapes and unlisted pairs keep the two-launch route
    byte-identically. Legacy rotate-then-AWQ modules additionally fold
    their compensation into the FHT (fht.fht_apply_awq, one kernel).

    Trainable LUT (straight-through, AQLM convention):
        make_trainable promotes the fp16 LUT buffer(s) to fp32
        Parameters — BOTH streams become independent masters — and
        caches the unpacked logical indices (int64, on the LUT's
        device). The reference forward then backprops into codebook entries
        only — indices remain frozen constants. freeze_lut demotes back
        to the deployment buffer (fp16-snapped). The kernel path refuses to
        run while ANY LUT is trainable (loud error, never silent fp32
        numerics through a contractually-fp16 artifact).

    Rotation (Hadamard boundary fold,): the palettizer folds
    W_rot = W @ T (T = blockdiag H_b diag(s_b)/sqrt(b), (seed, k)
    derived) and this module consumes the FOLD-SPACE input x_rot =
    x @ T before every GEMM — on BOTH execution paths — via the Fast
    Hadamard Transform (flute_extended/fht.py): the (K,) sign vector
    (self.rot_signs, ~16 KB) replaces the K x K matrix the old
    class-level _rot_cache materialized (~1.5 GB fp16 across the 8
    unique (seed, k) pairs; also a per-forward CPU->GPU copy of up to
    576 MB). The adjoint (weight-space un-rotation,
    _quantized_weight's Wq_rot @ T^T) is the same butterfly with the
    signs on the input side. The explicit matrix survives only as the
    opt-in FLUTE_ROTATION=matmul fallback backend.

    AWQ composition (fold_order): artifacts record which transform
    came first. "awq_then_rotate" (+ producers: W' = (W diag(s))
    @ T) composes exactly with the deployed pipeline (norm folds
    diag(s)^-1 into the gain, the module rotates by T) — no
    compensation. "rotate_then_awq" (the LEGACY producer order of the
    2026-10-04 box run: W' = (W @ T) diag(s)) does NOT: the deployed
    composition is x @ (D^-1 T D T^T) @ W^T — the root cause of the
    bad-greedy-decode round (see scripts/diagnose_greedy_bug.py,
    main project: 0.999 gates, garbage decode). The loader compensates EXACTLY by
    serving the input rotation M = D T D^-1 (scale -> FHT -> unscale)
    with the per-channel s recovered from norm_gain_edits.json; the
    per-layer cos gates stay honest because they measured Q vs W'
    (the fold-space weight) all along.
    """

    def __init__(self, indices, lut, bitwidth, group_size, N, K, bias=None,
                 resA=None, resB=None, indices2=None, lut2=None,
                 bitwidth2=None, reference: bool = False,
                 rotation_seed: int = None, rotation_k: int = None,
                 awq_scale=None, fold_order: str = None):
        super().__init__()
        self.bitwidth = int(bitwidth)
        self.group_size = int(group_size)
        # : the second stream may carry its OWN idxN width (e.g.
        # mixed:4,2 = base idx4 + refinement idx2.2). Default: the
        # stream-1 width (the hybrid422 two-idx4-streams layout).
        self.bitwidth2 = int(bitwidth2) if bitwidth2 is not None \
            else self.bitwidth
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("lut", lut, persistent=False)
        self.N = int(N)
        self.K = int(K)
        self.reference = bool(reference)
        # straight-through caches (only while a LUT is a Parameter)
        self._row_groups: Optional[torch.Tensor] = None
        self._idx_logical: Optional[torch.Tensor] = None
        self._row_groups2: Optional[torch.Tensor] = None
        self._idx_logical2: Optional[torch.Tensor] = None
        # idxN kernel contract (loud, no fallback)
        if self.N % 128 != 0 or self.K % 64 != 0:
            if not self.reference:
                raise ValueError(f"idxN layout requires N%128==0 and K%64==0, got "
                    f"N={self.N}, K={self.K}")
        for _b_w, _b in (("stream-1", self.bitwidth),
                         ("stream-2", self.bitwidth2)):
            if _b not in (1, 2, 3, 4):
                raise ValueError(f"PalettizedLinear: {_b_w} bitwidth must be 1-4, "
                    f"got {_b}")
        expected = (self.N * self.K * self.bitwidth) // 8
        if indices.numel() != expected:
            raise ValueError(f"idx{self.bitwidth} blob numel {indices.numel()} != "
                f"N*K*{self.bitwidth}/8 {expected} (N={self.N}, K={self.K})")
        if resA is not None and resB is not None:
            if resA.shape != (self.N, resB.shape[0]) or \
                    resB.shape[1] != self.K:
                raise ValueError(f"residual factors shape mismatch: A {tuple(resA.shape)}, "
                    f"B {tuple(resB.shape)} for N={self.N}, K={self.K}")
            self.register_buffer("resA", resA, persistent=False)
            self.register_buffer("resB", resB, persistent=False)
        else:
            self.resA = None
            self.resB = None
        # -- second stream (None/None = legacy, no new buffers) ----- #
        if indices2 is None and lut2 is None:
            self.indices2 = None
            self.lut2 = None
        elif indices2 is None or lut2 is None:
            raise ValueError(
                "PalettizedLinear: a second stream needs BOTH indices2 "
                f"and lut2 (got indices2={'None' if indices2 is None else 'set'},"
                f" lut2={'None' if lut2 is None else 'set'}); refusing a "
                "half-constructed two-stream module")
        else:
            expected2 = (self.N * self.K * self.bitwidth2) // 8
            if indices2.numel() != expected2:
                raise ValueError(f"idx{self.bitwidth2} stream-2 blob numel "
                    f"{indices2.numel()} != N*K*{self.bitwidth2}/8 "
                    f"{expected2} (N={self.N}, K={self.K})")
            palette = 1 << self.bitwidth
            palette2 = 1 << self.bitwidth2
            lut2_t = lut2 if torch.is_tensor(lut2) \
                else torch.from_numpy(np.asarray(lut2))
            if lut2_t.dim() != 2:
                lut2_t = lut2_t.view(-1, palette2)
            lut1_t = self.lut if torch.is_tensor(self.lut) \
                else torch.from_numpy(np.asarray(self.lut))
            if lut1_t.dim() != 2:
                lut1_t = lut1_t.view(-1, palette)
            if lut2_t.shape[1] != palette2:
                raise ValueError(f"stream-2 LUT palette width {lut2_t.shape[1]} != "
                    f"2**bitwidth2 {palette2}; stream 2 is an "
                    f"idx{self.bitwidth2} codebook")
            if lut2_t.shape[0] != lut1_t.shape[0]:
                raise ValueError(f"stream-2 LUT groups {lut2_t.shape[0]} != stream-1 "
                    f"{lut1_t.shape[0]}; both streams must share the "
                    f"group grid (same N, K, group_size)")
            self.register_buffer("indices2", indices2, persistent=False)
            self.register_buffer("lut2", lut2, persistent=False)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)
        # -- rotation (Hadamard boundary fold; None = unrotated) ----------- #
        self.rotation_seed = (int(rotation_seed)
                              if rotation_seed is not None else None)
        self.rotation_k = (int(rotation_k)
                           if rotation_k is not None else None)
        self.fold_order = fold_order if fold_order in (
            "awq_then_rotate", "rotate_then_awq") else None
        self.rot_signs = None
        self.awq_scale = None
        self._signs_cache = {}          # device -> (K,) fp32 sign tensor
        if self.rotation_seed is not None and self.rotation_k is not None:
            if self.rotation_k != self.K:
                raise ValueError(f"PalettizedLinear: rotation_k {self.rotation_k} != the "
                    f"input dim K {self.K} — the boundary fold is defined "
                    f"on the module input axis (corrupt metadata)")
            self.rot_signs = _rotation_signs_for(self.rotation_k,
                                                 self.rotation_seed)
            if awq_scale is not None:
                st = (awq_scale if torch.is_tensor(awq_scale)
                      else torch.as_tensor(awq_scale, dtype=torch.float32))
                if st.dim() != 1 or st.numel() != self.K:
                    raise ValueError(f"PalettizedLinear: awq_scale must be a (K,) vector "
                        f"for K={self.K}, got {tuple(st.shape)}")
                st = st.detach().to(torch.float32).cpu()
                if not torch.isfinite(st).all() or bool((st <= 0).any()):
                    raise ValueError(
                        "PalettizedLinear: awq_scale must be finite and > 0 "
                        "per channel (the norm fold divides by it)")
                self.awq_scale = st
                if self.fold_order is None:
                    # s present but no recorded order: the artifacts
                    # predate the fold-order provenance key, so they were
                    # produced by the LEGACY order (rotation first — the
                    # composition the loader must compensate).
                    self.fold_order = "rotate_then_awq"
        elif awq_scale is not None:
            raise ValueError(
                "PalettizedLinear: awq_scale passed without rotation "
                "metadata — the AWQ compensation only exists as a "
                "correction to the rotated fold")
        # The explicit K x K matrix is NOT built (that was the 1.5 GB
        # _rot_cache). _explicit_rot_T materializes it on demand for
        # the FLUTE_ROTATION=matmul fallback and the differential tests.

    def _signs_on(self, device) -> torch.Tensor:
        """The (K,) sign tensor on `device` (per-device cache; ~16 KB)."""
        key = str(device)
        cached = self._signs_cache.get(key)
        if cached is None:
            cached = self.rot_signs.to(device)
            self._signs_cache[key] = cached
        return cached

    def _awq_scale_on(self, device) -> Optional[torch.Tensor]:
        """The (K,) AWQ scale tensor on `device` (None when absent)."""
        if self.awq_scale is None:
            return None
        key = ("awq", str(device))
        cached = self._signs_cache.get(key)
        if cached is None:
            cached = self.awq_scale.to(device)
            self._signs_cache[key] = cached
        return cached

    def _explicit_rot_T(self) -> torch.Tensor:
        """The explicit (K, K) fold matrix T — fp32, CPU.

        Opt-in materialization (the FLUTE_ROTATION=matmul fallback and
        the differential tests). Byte-identical to the older
        _rot_cache entry's construction. NOT registered as a buffer —
        a plain attribute keeps it out of .to(device) accounting, same
        as the old cache.
        """
        if self.rot_signs is None:
            raise ValueError("_explicit_rot_T: module is unrotated")
        fht = _get_fht()
        return fht.build_rotation_matrix(self.rotation_k, self.rotation_seed)

    def _rotation_backend(self) -> str:
        """Resolve the FLUTE_ROTATION env knob to an fht backend.

        FLUTE_ROTATION: unset/"auto"/"fht" (default — CUDA kernel when
        built + CUDA input, torch butterfly otherwise), "reference"
        (the torch butterfly; CPU-legal), "matmul" (the explicit x @ T
        — debugging/differential testing only; materializes K x K).
        """
        choice = os.environ.get("FLUTE_ROTATION", "auto").strip().lower()
        if choice in ("", "auto", "fht"):
            return "auto"
        if choice in ("reference", "matmul"):
            return choice
        raise ValueError(f"FLUTE_ROTATION={choice!r} is not one of auto/fht/reference/"
            f"matmul")

    def _rotate_input(self, x: torch.Tensor) -> torch.Tensor:
        """The boundary-fold input rotation: x <- x @ T (exact for the
        module's artifacts), served by the FHT.

        Legacy rotate-then-AWQ artifacts (self.fold_order ==
        "rotate_then_awq" with a live awq_scale) get the compensated
        transform x @ (D T D^-1) = fht(x * s) / s — the exact inverse
        of the fold the legacy producer applied (proof:
        main project: scripts/diagnose_greedy_bug.py, claim 4). The compensation runs
        in fp32 (the 1/s unscale can push small activations through
        fp16 subnormals).

        the compensated branch is ONE kernel on the fused path —
        fht.fht_apply_awq folds the *s prologue and the /s epilogue
        into the butterfly (bit-identical math to the chain below; see
        flute_extended/fht.py). The older five-op torch chain stays
        as the reference/CPU fallback, verbatim.

        Autograd flows through on every backend.
        """
        fht = _get_fht()
        backend = self._rotation_backend()
        signs = self._signs_on(x.device)
        if backend == "matmul":
            T = self._explicit_rot_T().to(x.device, torch.float32)
            if self.awq_scale is not None and \
                    self.fold_order == "rotate_then_awq":
                s = self._awq_scale_on(x.device)
                return ((x.float() * s) @ T / s).to(x.dtype)
            return (x.float() @ T).to(x.dtype)
        if self.awq_scale is not None and \
                self.fold_order == "rotate_then_awq":
            s = self._awq_scale_on(x.device)
            if hasattr(fht, "fht_apply_awq"):
                # the compensated fold in ONE kernel .
                return fht.fht_apply_awq(x, signs, s, backend=backend)
            # older fht module (defensive — repo code ships together,
            # but the fold must never depend on that): the original
            # eager chain, verbatim.
            xf = x.float()
            return (fht.fht_apply(xf * s, signs, backend=backend)
                    / s).to(x.dtype)
        return fht.fht_apply(x, signs, backend=backend)

    def has_residual(self) -> bool:
        return self.resA is not None and self.resB is not None

    @property
    def has_stream2(self) -> bool:
        """True when the optional second stream (Route A pair-composite)
        is present — False for the legacy single-stream module."""
        return self.indices2 is not None and self.lut2 is not None

    # -- straight-through LUT training API -------------------------------- #

    @property
    def lut_trainable(self) -> bool:
        return isinstance(self.lut, nn.Parameter) and self.lut.requires_grad

    @property
    def lut2_trainable(self) -> bool:
        return (self.lut2 is not None and isinstance(self.lut2, nn.Parameter)
                and self.lut2.requires_grad)

    def _logical_indices_numpy(self, blob=None, bits=None) -> np.ndarray:
        """(N, K) uint8 logical indices from a packed idxN blob (the
        module's stream-1 blob by default; pass the stream-2 blob — and
        its width — for the second stream's cache)."""
        b = int(bits) if bits is not None else self.bitwidth
        if blob is None:
            blob = self.indices
        arr = np.ascontiguousarray(blob.detach().cpu().numpy(), dtype=np.uint8).reshape(-1)
        return _get_idxn().unpack_idxn(arr, self.N, self.K, b)

    @torch.no_grad()
    def _cache_logical_indices(self) -> "PalettizedLinear":
        """Cache (row_groups, idx_logical) on the LUT's device for the
        reference forward path — BOTH streams when stream 2 is present.
        Idempotent. Does not promote the LUTs to Parameters (unlike
        make_trainable)."""
        if self._idx_logical is not None and (not self.has_stream2 or self._idx_logical2 is not None):
            return self
        if self._idx_logical is None:
            lut_t = self.lut if torch.is_tensor(self.lut) \
                else torch.from_numpy(np.asarray(self.lut))
            if lut_t.dim() != 2:
                lut_t = lut_t.view(-1, 1 << self.bitwidth)
            rg = torch.div(torch.arange(self.N, device=lut_t.device),
                           self.group_size, rounding_mode="floor")
            rg = rg.clamp_(max=lut_t.shape[0] - 1)
            idx_np = self._logical_indices_numpy()
            idx_t = torch.from_numpy(idx_np.astype(np.int64))
            if lut_t.device != torch.device("cpu"):
                idx_t = idx_t.to(lut_t.device)
            self._row_groups = rg
            self._idx_logical = idx_t
        if self.has_stream2 and self._idx_logical2 is None:
            lut2_t = self.lut2 if torch.is_tensor(self.lut2) \
                else torch.from_numpy(np.asarray(self.lut2))
            if lut2_t.dim() != 2:
                lut2_t = lut2_t.view(-1, 1 << self.bitwidth2)
            rg2 = torch.div(torch.arange(self.N, device=lut2_t.device),
                            self.group_size, rounding_mode="floor")
            rg2 = rg2.clamp_(max=lut2_t.shape[0] - 1)
            idx_np2 = self._logical_indices_numpy(self.indices2,
                                                  self.bitwidth2)
            idx_t2 = torch.from_numpy(idx_np2.astype(np.int64))
            if lut2_t.device != torch.device("cpu"):
                idx_t2 = idx_t2.to(lut2_t.device)
            self._row_groups2 = rg2
            self._idx_logical2 = idx_t2
        return self

    def make_trainable(self) -> "PalettizedLinear":
        """Promote BOTH LUT buffers (stream 1 and the optional stream 2)
        to independent fp32 nn.Parameters (in place).

        Idempotent. Allocates the int64 logical-index caches (N, K) per
        stream, on each LUT's device — only call this for modules you
        actually train (the trainer calls it per-scope, per-layer).
        """
        if not isinstance(self.lut, nn.Parameter):
            lut32 = self.lut.detach().float().clone()
            idx_np = self._logical_indices_numpy()
            rg = torch.div(torch.arange(self.N, device=lut32.device),
                           self.group_size, rounding_mode="floor")
            rg = rg.clamp_(max=lut32.shape[0] - 1)
            idx_t = torch.from_numpy(idx_np.astype(np.int64))
            if lut32.device != torch.device("cpu"):
                idx_t = idx_t.to(lut32.device)
            del self.lut
            self._row_groups = rg
            self._idx_logical = idx_t
            self.lut = nn.Parameter(lut32)
        if self.has_stream2 and not isinstance(self.lut2, nn.Parameter):
            lut2_t = self.lut2 if torch.is_tensor(self.lut2) \
                else torch.from_numpy(np.asarray(self.lut2))
            if lut2_t.dim() != 2:
                lut2_t = lut2_t.view(-1, 1 << self.bitwidth2)
            lut32b = lut2_t.detach().float().clone()
            idx_np2 = self._logical_indices_numpy(self.indices2,
                                                  self.bitwidth2)
            rg2 = torch.div(torch.arange(self.N, device=lut32b.device),
                            self.group_size, rounding_mode="floor")
            rg2 = rg2.clamp_(max=lut32b.shape[0] - 1)
            idx_t2 = torch.from_numpy(idx_np2.astype(np.int64))
            if lut32b.device != torch.device("cpu"):
                idx_t2 = idx_t2.to(lut32b.device)
            del self.lut2
            self._row_groups2 = rg2
            self._idx_logical2 = idx_t2
            self.lut2 = nn.Parameter(lut32b)
        return self

    def freeze_lut(self, snap_fp16: bool = True) -> "PalettizedLinear":
        """Demote BOTH LUTs back to the deployment buffers (fp16-snapped).

        Idempotent; releases BOTH int64 index caches.
        """
        if isinstance(self.lut, nn.Parameter):
            lut16 = self.lut.detach()
            if snap_fp16:
                lut16 = lut16.to(torch.float16)
            del self.lut
            self.register_buffer("lut", lut16.contiguous(), persistent=False)
        self._row_groups = None
        self._idx_logical = None
        if self.has_stream2:
            if isinstance(self.lut2, nn.Parameter):
                lut16b = self.lut2.detach()
                if snap_fp16:
                    lut16b = lut16b.to(torch.float16)
                del self.lut2
                self.register_buffer("lut2", lut16b.contiguous(),
                                     persistent=False)
            self._row_groups2 = None
            self._idx_logical2 = None
        return self

    def snapped_lut(self) -> torch.Tensor:
        """fp16-grid copy of the current LUT (train master -> deploy grid)."""
        return self.lut.detach().to(torch.float16)

    def snapped_lut2(self) -> Optional[torch.Tensor]:
        """fp16-grid copy of the stream-2 LUT (None when stream 2 is
        absent) — the train master -> deploy grid cast, stream 2."""
        if self.lut2 is None:
            return None
        return self.lut2.detach().to(torch.float16)

    def deploy_clone(self) -> "PalettizedLinear":
        """Deployment-parity copy: frozen fp16 (snapped) LUTs, reference
        path. With a second stream present BOTH streams are carried
        (indices shared — they are frozen anyway).

        the rotation record travels with the clone (seed/k/fold
        order/AWQ scale) — a rotated module cloned without it silently
        served un-rotated numerics (the older clone dropped the
        rotation arguments).

        Used by the distillation loop to propagate the student trajectory
        with exactly the numerics that will be served (indices shared —
        they are frozen anyway).
        """
        return PalettizedLinear(self.indices, self.snapped_lut(), self.bitwidth, self.group_size,
            self.N, self.K,
            bias=None if self.bias is None else self.bias.detach().clone(),
            resA=None if self.resA is None else self.resA.detach().clone(),
            resB=None if self.resB is None else self.resB.detach().clone(),
            indices2=self.indices2, lut2=self.snapped_lut2(),
            bitwidth2=self.bitwidth2,
            reference=True,
            rotation_seed=self.rotation_seed, rotation_k=self.rotation_k,
            awq_scale=self.awq_scale, fold_order=self.fold_order)

    # -- forward ----------------------------------------------------------- #

    def _quantized_weight_stream(self, blob, lut, idx_logical, row_groups,
                                 bits=None) -> torch.Tensor:
        """Dequantize ONE stream (fp32; differentiable while that
        stream's LUT is a Parameter). The single dequant expression
        shared by both streams: the cached-index gather when the
        straight-through cache is live, the canonical CPU unpack
        (reference_dequant) otherwise. `bits` is the stream's idxN width
        (stream 2 may differ from stream 1)."""
        if idx_logical is not None:
            # straight-through path: cached int64 indices, gather on device
            return lut[row_groups].float().gather(1, idx_logical)
        return reference_dequant(blob, lut, self.N, self.K,
                                 self.group_size,
                                 self.bitwidth if bits is None else bits)

    def _quantized_weight(self) -> torch.Tensor:
        """Effective dequantized weight in the ORIGINAL (un-rotated)
        space — Wq_eff ~= W — fp32, differentiable while the LUTs train.

        With stream 2 present this accessor returns the summed
        effective weight W1 + W2 (validation parity — the loader-side
        checks and the trainer's teacher/student weight comparison
        consume the effective matrix); the FORWARD instead computes the
        two-GEMM ordered add y = y1 + y2 (deployment parity with the
        two-qgemm kernel shape, the design notes 1.3/1.4) — the two expressions
        are mathematically equal but not bit-equal.

         (rotation): the streams dequantize the FOLD-SPACE weight
        Wq_rot (≈ W @ T, or the AWQ-composed fold); the un-rotation is
        the ADJOINT FHT, Wq_rot @ T^T — NOT the older `w @ rot_T`
        (that computed W @ T @ T = W * s_column: the sign-scaled
        original, wrong for every rotated module). With AWQ in the fold
        the original-space weight sheds the scale on the correct side:
            legacy order W' = W @ T @ D -> W = adjoint(W' / s)
             order W' = (W @ D) @ T -> W = adjoint(W') / s
        (the norm-gain edit makes the PRE-module input x/s, so the
        module-space effective weight differs from the original-space
        one by that same column scale — consumers comparing against a
        pristine teacher should pair this accessor with the pristine
        teacher's norm).
        """
        w = self._quantized_weight_stream(self.indices, self.lut, self._idx_logical, self._row_groups)
        if self.has_stream2:
            w = w + self._quantized_weight_stream(self.indices2, self.lut2, self._idx_logical2,
                self._row_groups2, bits=self.bitwidth2)
        if self.rot_signs is not None:
            fht = _get_fht()
            signs = self._signs_on(w.device)
            if self.awq_scale is not None and \
                    self.fold_order == "rotate_then_awq":
                s = self._awq_scale_on(w.device)
                w = fht.fht_adjoint(w / s, signs)
            elif self.awq_scale is not None:      # awq_then_rotate
                s = self._awq_scale_on(w.device)
                w = fht.fht_adjoint(w, signs) / s
            else:
                w = fht.fht_adjoint(w, signs)
        return w

    def _gemv_decode(self, xh):
        """The decode-GEMV route (see forward): ONE memory-bound
        qgemm_gemv_stream launch for the M == 1 decode shape — both
        streams, the rank-16 residual and the bias, no tensor cores (the
        whole-forward CUDA-graph replays of scripts/eval_greedy_match.py
        (main project)
        are M == 1; the box report: 13.514 tok/s = 74.0 ms/token of
        GPU-side kernel time, the dual mma kernel ~12x above the code-
        stream memory floor at that shape).

        Returns (output, residual_fused, bias_fused) or None — every gate
        below is a contract mirror of the C++ GEMV dispatch, so a None is
        a PERFORMANCE fallback to the dual-stream route (M 2..16) or the
        two-launch path, never a correctness one. The M == 1 pin keeps
        prefill/PPL numerics byte-identical to the pre-plain-GEMV route (the
        GEMV's fixed fp32 accumulation order is a DECODE-shape contract —
        see the numerics note in kernel_streaming.cu).
        """
        if xh.shape[0] != 1:
            return None
        try:
            import flute_extended
        except ImportError:
            return None
        if xh.dtype != torch.float16 or not xh.is_contiguous():
            return None
        # NOTE: no is_cuda gate here, deliberately — same rationale as
        # the dual route's note (the _C entry refuses loudly; keeping
        # the gate device-free keeps the routing unit-testable on CPU
        # boxes; main project: tests/test_gemv.py).
        b2 = self.bitwidth2 if self.has_stream2 else 0

        def _ok(t, *dtypes):
            return (t is not None and t.is_contiguous()
                    and t.dtype in dtypes)

        if not _ok(self.indices, torch.uint8) or \
                not _ok(self.lut, torch.float16):
            return None
        if self.has_stream2 and (not _ok(self.indices2, torch.uint8)
                or not _ok(self.lut2, torch.float16)):
            return None

        # the fusable residual: fp16/contiguous, rank within the FUSED
        # cap (the kernel's table decides the cap; the caller's residual
        # stays on the cuBLAS post-add route when it exceeds it)
        res_rank = 0
        if self.has_residual() and _ok(self.resA, torch.float16) \
                and _ok(self.resB, torch.float16):
            res_rank = int(self.resB.shape[0])
        bias = self.bias if _ok(self.bias, torch.float16) else None

        # the split-K GEMV route FIRST — the split-K grid + double-buffer
        # kernel with the WIDE table (every (b1, b2) pair, GS 16..2048 —
        # the QKV composite pairs (4,3)/(3,4)/(4,4)/(2,2) and the GS 16/32
        # stragglers ride it — and the residual cap raised to 256 in # the box census' 75 modules at R=64/128/256 fuses instead of
        # falling to the /two-launch routes). A None (older
        # extension, an unusual GS, FLUTE_NO_SPLITK=1) falls to the GEMV
        # below, byte-identically.
        # the WIDE preference — at N/128 >= 160 tiles the split-K GEMV's split
        # policy returns SPLIT=1 and the kernel's 12+ wave deep
        # pipeline is the proven faster streamer (the box probe: lm_head
        # 488.2 GB/s on w27 vs the split-K GEMV's 210-240 GB/s band on its best
        # multi-wave modules); only lm_head (N=248320, 1940 tiles)
        # deploys up there — the next widest module is 12288 rows = 96
        # tiles. FLUTE_NO_WIDE_PREF=1 disables the preference (the box A/B).
        _plain_gemv_first = False
        if self.N // 128 >= 160 and \
                os.environ.get("FLUTE_NO_WIDE_PREF", "").strip() not in (
                    "1", "true", "True"):
            _plain_gemv_first = getattr(flute_extended, "gemv_stream_available",
                                 lambda: False)() and \
                flute_extended.gemv_stream_supported(self.bitwidth, b2, self.group_size, self.N, self.K,
                    xh.shape[0])
        if not _plain_gemv_first and \
                os.environ.get("FLUTE_NO_SPLITK", "").strip() not in ("1", "true",
                                                                  "True") and \
                getattr(flute_extended, "gemv_splitk_available",
                        lambda: False)() and \
                flute_extended.gemv_splitk_supported(self.bitwidth, b2, self.group_size, self.N, self.K,
                    xh.shape[0], res_rank):
            resB = resA = None
            if 1 <= res_rank <= 256:
                resB, resA = self.resB, self.resA
            y = flute_extended.qgemm_gemv_splitk_stream(xh, self.indices, self.lut, self.bitwidth,
                self.group_size,
                indices2=self.indices2 if self.has_stream2 else None,
                lut2=self.lut2 if self.has_stream2 else None,
                bitwidth2=b2,
                resB=resB, resA=resA, bias=bias,
)
            return y, resB is not None, bias is not None

        # the original GEMV (the 7-pair table, GS 64..2048, R <= 16).
        # getattr with a default: a -era fake/stub package without the
        # surface (tests) and an older extension build both take the
        # same graceful exit.
        if not getattr(flute_extended, "gemv_stream_available",
                       lambda: False)():
            return None
        if not flute_extended.gemv_stream_supported(self.bitwidth, b2, self.group_size, self.N, self.K,
                xh.shape[0]):
            return None
        resB = resA = None
        if 1 <= res_rank <= 16:
            resB, resA = self.resB, self.resA
        # else: the residual stays on the cuBLAS post-add route below
        y = flute_extended.qgemm_gemv_stream(xh, self.indices, self.lut, self.bitwidth, self.group_size,
            indices2=self.indices2 if self.has_stream2 else None,
            lut2=self.lut2 if self.has_stream2 else None,
            bitwidth2=b2,
            resB=resB, resA=resA, bias=bias,
)
        return y, resB is not None, bias is not None

    def _gemv_fht_decode(self, x):
        """The decode route (see forward): ONE qgemm_gemv_stream launch
        with the boundary-fold rotation FUSED INTO the kernel prologue —
        the standalone FHT kernel (the last per-module launch outside the
        GEMV) disappears from the decode path. The 281 modules drop from
        FHT+GEMV (562 launches) to ONE launch each — the dense arm's own
        count; the box report put the remaining 5.5% gap exactly
        there (0.945x dense).

        Receives the UNROTATED x (forward calls this BEFORE
        _rotate_input) and passes x + self's cached rot_signs (and, for
        legacy rotate-then-AWQ artifacts, the awq scale — the same
        tensors _rotate_input would hand fht.fht_apply / fht.fht_apply_
        awq) to flute_extended.qgemm_gemv_stream's rot_signs/awq_scale
        parameters. The kernel's prologue is a line-for-line
        transcription of fht_forward / fht_forward_awq (same fp32
        staging, same butterflies, same epilogue rounding), so the
        decode outputs are BIT-IDENTICAL to the FHT-then-GEMV chain
        — the greedy texts and the PPL (M > 1, never routed here) both
        stay exactly where the run left them.

        Returns (output, residual_fused, bias_fused) or None — every
        gate below is a contract mirror of the C++ FHT-fused dispatch,
        so a None is a PERFORMANCE fallback to the plain-GEMV route (explicit
        rotation, then GEMV/dual/two-launch), never a correctness one.
        FLUTE_NO_FHT_FUSE=1 forces that fallback (the box A/B switch).
        """
        if x.shape[0] != 1:
            return None
        if os.environ.get("FLUTE_NO_FHT_FUSE", "").strip() in ("1", "true",
                                                                "True"):
            return None
        if self._rotation_backend() != "auto":
            # FLUTE_ROTATION=reference/matmul pins the explicit rotation
            return None
        if self.lut_trainable or self.lut2_trainable:
            # the train route needs the rotated x through qlora_gemm
            return None
        if x.requires_grad:
            return None
        try:
            import flute_extended
        except ImportError:
            return None
        if x.dtype != torch.float16 or not x.is_contiguous():
            # the fused kernel takes the RAW row in fp16 — anything else
            # keeps the explicit rotation (whose numerics match the
            # input dtype) and the GEMV
            return None
        b2 = self.bitwidth2 if self.has_stream2 else 0

        def _ok(t, *dtypes):
            return (t is not None and t.is_contiguous()
                    and t.dtype in dtypes)

        if not _ok(self.indices, torch.uint8) or \
                not _ok(self.lut, torch.float16):
            return None
        if self.has_stream2 and (not _ok(self.indices2, torch.uint8)
                or not _ok(self.lut2, torch.float16)):
            return None
        # the rotation operands: the module's cached (K,) fp32 sign
        # vector, and for legacy rotate_then_awq artifacts the (K,) fp32
        # AWQ scale (awq_then_rotate folds s into the weights at load
        # and passes none here — the same branch _rotate_input takes)
        signs = self._signs_on(x.device)
        if not _ok(signs, torch.float32) or signs.numel() != self.K \
                or signs.dim() != 1:
            return None
        awq = None
        if self.awq_scale is not None and \
                self.fold_order == "rotate_then_awq":
            awq = self._awq_scale_on(x.device)
            if not _ok(awq, torch.float32) or awq.numel() != self.K \
                    or awq.dim() != 1:
                return None

        res_rank = 0
        if self.has_residual():
            # all-or-nothing: the fused route has NO rotated x for a
            # cuBLAS residual post-add (computing one would re-run the
            # FHT the fusion just eliminated), so a residual beyond the
            # FUSED cap falls back to the explicit-rotation route
            if not (_ok(self.resA, torch.float16)
                    and _ok(self.resB, torch.float16)):
                return None
            res_rank = int(self.resB.shape[0])
        bias = self.bias
        if bias is not None and not _ok(bias, torch.float16):
            return None      # same all-or-nothing contract as the residual

        # the FHT-fused split-K GEMV FIRST — the split-K grid +
        # double-buffer kernel with the WIDE table (every pair, GS
        # 16..2048, the residual cap raised to 256 in the 75
        # R=64/128/256 modules of the box census join the fused
        # epilogue). The prologue is the transcription; the split fold
        # is the split-K GEMV deterministic contract. A None (older extension,
        # R > 256, FLUTE_NO_SPLITK=1) falls to the route below.
        # the WIDE preference (see _gemv_decode) — at >= 160 tiles
        # with a plain-GEMV-compatible pair/GS, decline the FHT-fused route so
        # forward falls to the explicit FHT + GEMV chain (lm_head's
        # proven 488.2 GB/s; the fused split-K GEMV at SPLIT=1 streams ~half of
        # that). The one extra FHT launch costs ~3 us against ~1 ms.
        if self.N // 128 >= 160 and \
                os.environ.get("FLUTE_NO_WIDE_PREF", "").strip() not in (
                    "1", "true", "True") and \
                getattr(flute_extended, "gemv_stream_available",
                        lambda: False)() and \
                flute_extended.gemv_stream_supported(self.bitwidth, b2, self.group_size, self.N, self.K,
                    x.shape[0]):
            return None
        if os.environ.get("FLUTE_NO_SPLITK", "").strip() not in ("1", "true",
                                                              "True") and \
                getattr(flute_extended, "gemv_splitk_fht_available",
                        lambda: False)() and \
                flute_extended.gemv_splitk_fht_supported(self.bitwidth, b2, self.group_size, self.N, self.K,
                    x.shape[0], res_rank):
            resB = resA = None
            if 1 <= res_rank <= 256:
                resB, resA = self.resB, self.resA
            y = flute_extended.qgemm_gemv_splitk_stream(x, self.indices, self.lut, self.bitwidth,
                self.group_size,
                indices2=self.indices2 if self.has_stream2 else None,
                lut2=self.lut2 if self.has_stream2 else None,
                bitwidth2=b2,
                resB=resB, resA=resA, bias=bias,
                rot_signs=signs, awq_scale=awq,
)
            return y, resB is not None, bias is not None

        # the original fused kernel (the 7-pair table, GS 64..2048,
        # R <= 16). getattr with a default: a -era fake/stub package
        # without the surface (tests) and an older extension build
        # both take the same graceful exit.
        if not getattr(flute_extended, "gemv_fht_stream_available",
                       lambda: False)():
            return None
        if not flute_extended.gemv_fht_stream_supported(self.bitwidth, b2, self.group_size, self.N, self.K,
                x.shape[0]):
            return None
        if res_rank > 16:
            return None      # the residual cap; the split-K GEMV (R <= 256) took it above
        resB = resA = None
        if res_rank >= 1:
            resB, resA = self.resB, self.resA
        y = flute_extended.qgemm_gemv_stream(x, self.indices, self.lut, self.bitwidth, self.group_size,
            indices2=self.indices2 if self.has_stream2 else None,
            lut2=self.lut2 if self.has_stream2 else None,
            bitwidth2=b2,
            resB=resB, resA=resA, bias=bias,
            rot_signs=signs, awq_scale=awq,
)
        return y, resB is not None, bias is not None

    def _dual_stream_decode(self, xh):
        """The fused decode route (see forward): ONE
        qgemm_dual_stream launch computing both streams, the rank-16
        residual and the bias.

        Returns (output, residual_fused, bias_fused) or None — every
        gate below is a contract mirror of the C++ dual dispatch, so a
        None is a PERFORMANCE fallback to the two-launch path, never a
        correctness one. The M <= DUAL_STREAM_MAX_M gate keeps prefill/
        PPL numerics byte-identical to the older route (the fused
        kernel's fp32 stream/residual/bias accumulation is a DECODE-shape
        contract — see the numerics note in kernel_streaming.cu).
        """
        try:
            import flute_extended
        except ImportError:
            return None
        if not getattr(flute_extended, "dual_stream_available",
                       lambda: False)():
            return None
        if xh.dtype != torch.float16 or not xh.is_contiguous():
            return None
        # NOTE: no is_cuda gate here, deliberately. A CPU box never gets
        # this far (the flute_extended import above fails without the
        # CUDA extension -> the pre-import fallback returned None), and
        # a CUDA-extension build handed a CPU tensor fails LOUDLY inside
        # _C.qgemm_dual_stream — the same "All tensors must be CUDA"
        # refusal the two-launch qgemm_per_group_lut path has always
        # given. Keeping the gate CUDA-free also keeps the routing
        # unit-testable on CPU boxes (main project: tests/test_dual_stream.py).
        if xh.shape[0] > flute_extended.DUAL_STREAM_MAX_M:
            return None
        b2 = self.bitwidth2 if self.has_stream2 else 0
        if not flute_extended.dual_stream_supported(self.bitwidth, b2, self.group_size, self.N, self.K):
            return None

        def _ok(t, *dtypes):
            # no is_cuda check, deliberately — the _C entry TORCH_CHECKs
            # CUDA loudly (the same refusal the two-launch path gives a
            # CPU tensor), and keeping the gate device-free keeps the
            # routing unit-testable on CPU boxes.
            return (t is not None and t.is_contiguous()
                    and t.dtype in dtypes)

        if not _ok(self.indices, torch.uint8) or \
                not _ok(self.lut, torch.float16):
            return None
        if self.has_stream2 and (not _ok(self.indices2, torch.uint8)
                or not _ok(self.lut2, torch.float16)):
            return None
        resB = resA = None
        if self.has_residual() and _ok(self.resA, torch.float16) \
                and _ok(self.resB, torch.float16) \
                and self.resB.shape[0] <= 16:
            resB, resA = self.resB, self.resA
        # else: the residual stays on the cuBLAS post-add route below
        bias = self.bias if _ok(self.bias, torch.float16) else None
        y = flute_extended.qgemm_dual_stream(xh, self.indices, self.lut, self.bitwidth, self.group_size,
            indices2=self.indices2 if self.has_stream2 else None,
            lut2=self.lut2 if self.has_stream2 else None,
            bitwidth2=b2,
            resB=resB, resA=resA, bias=bias,
)
        return y, resB is not None, bias is not None

    def forward(self, x):
        original_shape = x.shape
        original_dtype = x.dtype
        if x.dim() == 3:
            batch, seq, K = x.shape
            x = x.reshape(batch * seq, K)   # reshape (not view): the input
            # may be a non-contiguous slice, which .view cannot handle
        # Rotation (Hadamard boundary fold): the palettizer folded
        # W_rot = W @ T (the AWQ composition per self.fold_order) and the
        # module must consume the FOLD-SPACE input x <- x @ T — on BOTH
        # execution paths. The older code rotated only the kernel
        # path: the reference path GEMMed the UNROTATED x against the
        # fold-space weight (a rotated-space output — garbage for every
        # rotated artifact; its docstring claimed _quantized_weight
        # handled the rotation, but forward calls
        # _quantized_weight_stream, the per-stream FOLD-SPACE dequant).
        # One rotation, both paths, via the FHT (autograd-transparent).
        # at the M == 1 decode shape the rotation moves INSIDE the
        # GEMV — try the FHT-fused route FIRST, on the UNROTATED x (it
        # consumes x + rot_signs + the legacy AWQ scale in ONE launch;
        # bit-identical to the FHT-then-GEMV chain it replaces). A None
        # (every other shape, GS 16/32, non-fp16, FLUTE_NO_FHT_FUSE=1,
        # an older extension) falls through to the explicit rotation
        # and the /two-launch ladder below, byte-identically.
        if self.rot_signs is not None and not self.reference:
            fused_fht = self._gemv_fht_decode(x)
            if fused_fht is not None:
                output, res_done, bias_done = fused_fht
                if len(original_shape) == 3:
                    output = output.view(original_shape[0], original_shape[1], self.N)
                return output
        if self.rot_signs is not None:
            x = self._rotate_input(x)
        # flags for the fused decode route (set when the dual
        # kernel already folded the residual/bias into its epilogue —
        # the post-adds below then skip them).
        res_done = bias_done = False
        if self.reference:
            x_f = x.float()
            y = x_f @ self._quantized_weight_stream(self.indices, self.lut, self._idx_logical,
                self._row_groups).t()
            if self.has_stream2:
                # ordered add: stream 1 pinned first, then stream 2 —
                # the reference arm of the deployment shape (two GEMMs
                # plus ONE fp32 add; NOT x @ (W1 + W2).t)
                y = y + x_f @ self._quantized_weight_stream(self.indices2, self.lut2, self._idx_logical2,
                    self._row_groups2, bits=self.bitwidth2).t()
            output = y.to(x.dtype)
        else:
            trainable = ["lut"] if self.lut_trainable else []
            if self.lut2_trainable:
                trainable.append("lut2")
            if trainable and x.is_cuda:
                # (): the trainable-LUT
                # kernel route — the qlora_gemm train Functions carry
                # dL/dLUT (the fp16 operand is RECOMPUTED in backward
                # from the fp32 master, never saved). Both LUTs
                # trainable (or lut alone on a two-stream module with a
                # frozen lut2) route through the two-stream Function;
                # the single-stream Function otherwise. A None return
                # (FLUTE kernel unavailable) falls to the loud refusal
                # below — never a silent reference fallback.
                _qg_path = _HERE      # scripts/ — qlora_gemm.py's home
                if _qg_path not in sys.path:
                    sys.path.insert(0, _qg_path)
                import qlora_gemm     # lazy, same style as qlora.py
                if self.has_stream2:
                    y_train = qlora_gemm.fused_qlora_gemm_train_lut_two_streams(x, self.indices, self.lut, self.bitwidth,
                        self.indices2, self.lut2, self.bitwidth2,
                        self.group_size, self.N, self.K)
                else:
                    y_train = qlora_gemm.fused_qlora_gemm_train_lut(x, self.indices, self.lut, self.bitwidth,
                        self.group_size, self.N, self.K)
                if y_train is not None:
                    if self.resA is not None and self.resB is not None:
                        y_train = y_train + (
                            (x.half() @ self.resB.t())
                            @ self.resA.t()).to(y_train.dtype)
                    # NO post-GEMM rotation here. The input was
                    # already rotated once at the top of forward
                    # (x <- x @ T, the fold-space input the GEMM and the
                    # residual branch above both consume). The older
                    # `y_train @ self.rot_T` applied T a SECOND time —
                    # to the OUTPUT (batch, N) — which is wrong for
                    # every shape and a hard shape error whenever
                    # N != K (down_proj: (M, 4096) @ (12288, 12288)).
                    if self.bias is not None:
                        y_train = y_train + self.bias
                    if len(original_shape) == 3:
                        y_train = y_train.view(original_shape[0], original_shape[1], self.N)
                    return y_train
                # else: the loud refusal below (kernel went away).
            if trainable:
                raise ValueError(
                    "PalettizedLinear(kernel path): trainable LUT "
                    f"({', '.join(trainable)}) — the idxN kernel contract "
                    "requires a frozen fp16 LUT, and the FLUTE forward "
                    "kernel is unavailable for the train route. Use "
                    "reference=True for training (CPU or kernel-less "
                    "CUDA), freeze_lut() first, or rebuild: cd "
                    "flute_extended && python setup.py build_ext "
                    "--inplace.")
            bad_dtype = []
            if self.lut.dtype != torch.float16:
                bad_dtype.append(("lut", self.lut.dtype))
            if self.has_stream2 and self.lut2.dtype != torch.float16:
                bad_dtype.append(("lut2", self.lut2.dtype))
            if bad_dtype:
                names = ", ".join(f"{n} ({d})" for n, d in bad_dtype)
                raise ValueError(
                    "PalettizedLinear(kernel path): LUT dtype must be "
                    f"float16, got {names}")
            # Ensure flute_extended is importable (add parent dir to path)
            _flute_path = os.path.join(_HERE, "..", "flute_extended")
            if _flute_path not in sys.path:
                sys.path.insert(0, _flute_path)
            import flute_extended  # lazy: CUDA extension only on the kernel path
            xh = x.half()  # FLUTE kernel contract: FP16 A-operand
            # the GEMV route first — M == 1 (the decode shape the
            # whole-forward CUDA-graph replays dominate with) takes ONE
            # memory-bound qgemm_gemv_stream launch; the dual-stream kernel
            # takes M 2..16; every other shape keeps the two-launch path
            # below byte-identically.
            fused = self._gemv_decode(xh)
            if fused is None:
                # : the fused decode route
                # — ONE qgemm_dual_stream launch replaces the per-module
                # chain of two qgemm launches + the (M, N) add + the two
                # residual GEMMs + the bias add. Gated to the M <= 16
                # decode shape and the compiled (bitwidth, bitwidth2, GS)
                # table; every other shape keeps the two-launch path
                # below byte-identically.
                fused = self._dual_stream_decode(xh)
            if fused is not None:
                output, res_done, bias_done = fused
            else:
                y1 = flute_extended.qgemm_per_group_lut(xh, self.indices, self.lut,
                    bitwidth=self.bitwidth,
                    group_size=self.group_size,
                    indices_layout=f"idx{self.bitwidth}",   # explicit, never defaulted
)
                if self.has_stream2:
                    # the design notes 1.3/1.4 deployment shape: TWO qgemm calls
                    # (stream 1, then stream 2) + ONE ordered fp16 add; the
                    # streams may carry DIFFERENT idxN widths (mixed:4,2 =
                    # idx4 base + idx2 refinement). the add is in-place
                    # on the no-grad path (see _accum_streams) — the (M, N)
                    # transient drops from 3 tensors to 2.
                    y2 = flute_extended.qgemm_per_group_lut(xh, self.indices2, self.lut2,
                        bitwidth=self.bitwidth2,
                        group_size=self.group_size,
                        indices_layout=f"idx{self.bitwidth2}",  # explicit
)
                    output = _accum_streams(y1, y2).to(original_dtype)
                else:
                    output = y1.to(original_dtype)
        if self.resA is not None and self.resB is not None and not res_done:
            # Reference path: use original dtype. Kernel path: use half.
            x_for_res = x.half() if not self.reference else x.to(self.resA.dtype)
            output = output + ((x_for_res @ self.resB.t())
                               @ self.resA.t()).to(output.dtype)
        if self.bias is not None and not bias_done:
            output = output + self.bias
        if len(original_shape) == 3:
            output = output.view(original_shape[0], original_shape[1], self.N)
        return output


def _accum_streams(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """The two-stream sum (stream 1 pinned first): in-place add when
    neither stream carries autograd history, out-of-place otherwise.

     : the eager `y1 + y2` kept THREE
    (M, N) fp16 tensors live (y1, y2, the sum) — at the PPL arm's
    M=8192 on lm_head (N=248320) that is 3 x 4.07 GiB, the dominant
    ask of the quant batch-4 OOM the ladder kept healing. The
    no-grad in-place `y1.add_(y2)` keeps TWO. Numerically identical
    (same op); the kernel path's y1/y2 never require grad (the raw
    pybind qgemm has no autograd), and the guard keeps the trainable
    plane on the out-of-place path.
    """
    if y1.requires_grad or y2.requires_grad:
        return y1 + y2
    return y1.add_(y2)


class SplitQKV(nn.Module):
    def __init__(self, q_proj, k_proj, v_proj):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = q_proj, k_proj, v_proj
        # the merged-GEMV state (built lazily per device at the
        # first M == 1 forward — see _gemv_multi_decode). None until
        # then; {"ok": False, ...} after a refused build.
        self._merge_state = None

    def forward(self, x):
        fused = self._gemv_multi_decode(x)
        if fused is not None:
            return fused
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)],
                         dim=-1)

    # ------------------------------------------------------------------ #
    # the grouped decode route — the QKV merge.
    #
    # ONE qgemm_gemv_multi(_fht) launch replaces the three per-component
    # M == 1 GEMV launches + the torch.cat (main project:
    # docs/A10G_DECODE_INVESTIGATION.md §12.4 item 1): the components share one input row, so the merged
    # grid stacks their fixed paths on the machine together (the
    # the latency-bound split-K GEMV's barriers hide behind each other) and the outputs
    # land STRAIGHT in one persistent [1, N_total] buffer — the cat and
    # its 3 copy kernels disappear.
    #
    # Same discipline as every decode route in this file: a None is a
    # PERFORMANCE fallback to the per-module chain (the cat above),
    # never a correctness one. FLUTE_NO_MERGE=1 is the box A/B switch.
    # The state (the fusion verdict, the segment table, the persistent
    # output buffer) is built ONCE per device on the first eligible call
    # — all pointers are stable module weights, so the table stays valid
    # across CUDA-graph replays (the kernel fully overwrites every
    # output row each call — the P-workspace reuse pattern).
    # ------------------------------------------------------------------ #
    def _gemv_multi_decode(self, x):
        if os.environ.get("FLUTE_NO_MERGE", "").strip() in ("1", "true",
                                                          "True"):
            return None
        # the M == 1 shape (the same reshape PalettizedLinear.forward
        # does; anything else takes the per-module chain)
        orig3d = None
        if x.dim() == 3:
            b, s, k = x.shape
            if b * s != 1:
                return None
            orig3d = (b, s)
            x = x.reshape(1, k)
        if x.shape[0] != 1:
            return None
        if x.dtype != torch.float16 or not x.is_contiguous() \
                or x.requires_grad:
            return None
        if not x.is_cuda:
            return None      # CPU boxes keep the per-module reference path
        try:
            import flute_extended
        except ImportError:
            return None

        state = self._merge_state
        if state is None or state.get("device") != x.device:
            state = self._build_multi_state(x.device, flute_extended)
            self._merge_state = state
        if not state["ok"]:
            return None

        # the heterogeneous call — the segment table carries the
        # per-segment specs (bitwidths, GS, signs, AWQ); the impl picks
        # the FHT path from the signs columns (A is the RAW row then).
        flute_extended.qgemm_gemv_multi(x, state["seg_table"])
        out = state["out"]
        if orig3d is not None:
            out = out.view(orig3d[0], orig3d[1],
                           out.shape[1])
        return out

    def _build_multi_state(self, device, flute_extended):
        """The one-time fusion verdict + buffers (per device). THE
        HETEROGENEOUS CONTRACT: every component must be a fusable
        PalettizedLinear with the SAME K (the one spec the merged kernel
        shares); the (bitwidth, bitwidth2, group_size), the rotation
        signs, the AWQ scales, the residuals (each 0..256) and the biases
        may ALL differ per component — the deployed mixed-radix palette
        gives every tensor its own allocation, and the kernel serves
        each segment's spec at runtime . The rotation
        PRESENCE must be all-or-none across the group (the plain
        contract needs one pre-rotated row for every segment)."""
        mods = [self.q_proj, self.k_proj, self.v_proj]
        state = {"ok": False, "device": device, "mods": mods}
        import flute_extended as fx
        if os.environ.get("FLUTE_NO_MERGE", "").strip() in (
                "1", "true", "True"):
            state["reason"] = "FLUTE_NO_MERGE=1"
            return state
        for m in mods:
            if not isinstance(m, PalettizedLinear):
                state["reason"] = "non-PalettizedLinear component"
                return state
        for m in mods:
            if m.reference or m.lut_trainable or m.lut2_trainable:
                state["reason"] = "reference/trainable component"
                return state

        def _ok(t, *dtypes):
            return (t is not None and t.is_contiguous()
                    and t.dtype in dtypes)

        for m in mods:
            if not _ok(m.indices, torch.uint8) or \
                    not _ok(m.lut, torch.float16):
                state["reason"] = "indices/lut dtype|contiguity"
                return state
            if m.has_stream2 and (not _ok(m.indices2, torch.uint8)
                    or not _ok(m.lut2, torch.float16)):
                state["reason"] = "stream-2 dtype|contiguity"
                return state
        K = mods[0].K
        for m in mods[1:]:
            if m.K != K:
                state["reason"] = "K mismatch across components"
                return state
        for m in mods:
            if m.N % 128 != 0:
                state["reason"] = f"N={m.N} not a multiple of 128"
                return state

        # residuals: all-or-nothing per component, each 1..256
        ranks = []
        for m in mods:
            r = 0
            if m.has_residual():
                if not (_ok(m.resA, torch.float16)
                        and _ok(m.resB, torch.float16)):
                    state["reason"] = "residual dtype|contiguity"
                    return state
                r = int(m.resB.shape[0])
                if r > 256:
                    state["reason"] = f"residual rank {r} > 256"
                    return state
            ranks.append(r)
        biases = []
        for m in mods:
            bias = m.bias if _ok(m.bias, torch.float16) else None
            biases.append(bias)

        # the rotation: PRESENCE all-or-none (per-segment signs/AWQ may
        # differ — each CTA's FHT prologue applies its own segment's
        # fold; the fold ORDER stays uniform, the math class per group)
        rotated = [m.rot_signs is not None for m in mods]
        fused_fht = all(rotated)
        if fused_fht:
            for m in mods:
                if m._rotation_backend() != "auto":
                    state["reason"] = "pinned rotation backend"
                    return state
                signs = m._signs_on(device)
                if signs is None or not _ok(signs, torch.float32) \
                        or signs.numel() != K \
                        or signs.data_ptr() % 16 != 0:
                    state["reason"] = "signs contract"
                    return state
            fold_orders = {m.fold_order for m in mods}
            if len(fold_orders) != 1:
                state["reason"] = "fold_order mismatch"
                return state
            if mods[0].fold_order == "rotate_then_awq":
                for m in mods:
                    if m.awq_scale is not None:
                        awq_t = m._awq_scale_on(device)
                        if awq_t is None or not _ok(awq_t, torch.float32) \
                                or awq_t.numel() != K \
                                or awq_t.data_ptr() % 16 != 0:
                            state["reason"] = "awq contract"
                            return state
        elif any(rotated):
            state["reason"] = "mixed rotation presence"
            return state

        # the per-segment specs (the heterogeneous surface)
        specs = []
        for m in mods:
            b1 = int(m.bitwidth)
            b2 = int(m.bitwidth2) if m.has_stream2 else 0
            specs.append((b1, b2, int(m.group_size)))
        tiles_total = sum(m.N for m in mods) // 128
        ranks_total = sum(ranks)

        if not getattr(fx, "gemv_multi_available",
                       lambda: False)() or \
                not fx.gemv_multi_supported(
                    K, tiles_total, specs, ranks_total, fused_fht):
            state["reason"] = "extension/gate: multi"
            return state

        # the persistent output + the CPU segment table (all pointers
        # stable; C per segment = one buffer + the row offset — the
        # offsets are multiples of 128 rows = 256 B, 16-B aligned).
        # columns: [q1, lut1, q2, lut2, resB, resA, bias, C, N, R,
        # b1, b2, gs, signs, s]
        n_total = sum(m.N for m in mods)
        out = torch.zeros(1, n_total, dtype=torch.float16,
                          device=device)
        sign_tensors = []      # keep-alive refs (the modules own them)
        awq_tensors = []
        rows = []
        off = 0
        for m, r, bias in zip(mods, ranks, biases):
            i2 = m.indices2 if m.has_stream2 else None
            l2 = m.lut2 if m.has_stream2 else None
            resB = m.resB if 1 <= r <= 256 else None
            resA = m.resA if 1 <= r <= 256 else None
            b1 = int(m.bitwidth)
            b2 = int(m.bitwidth2) if m.has_stream2 else 0
            signs_ptr = 0
            awq_ptr = 0
            if fused_fht:
                st = m._signs_on(device)
                sign_tensors.append(st)
                signs_ptr = st.data_ptr()
                if mods[0].fold_order == "rotate_then_awq" \
                        and m.awq_scale is not None:
                    at = m._awq_scale_on(device)
                    awq_tensors.append(at)
                    awq_ptr = at.data_ptr()
            rows.append([
                m.indices.data_ptr(), m.lut.data_ptr(),
                (i2.data_ptr() if i2 is not None else 0),
                (l2.data_ptr() if l2 is not None else 0),
                (resB.data_ptr() if resB is not None else 0),
                (resA.data_ptr() if resA is not None else 0),
                (bias.data_ptr() if bias is not None else 0),
                out.data_ptr() + off * 2,
                m.N, r,
                b1, b2, int(m.group_size),
                signs_ptr, awq_ptr,
            ])
            off += m.N
        seg_table = torch.tensor(rows, dtype=torch.int64)  # CPU

        state.update(ok=True, out=out, seg_table=seg_table,
                     fused_fht=fused_fht, specs=specs,
                     tiles_total=tiles_total,
                     keepalive=(sign_tensors, awq_tensors))
        if os.environ.get("FLUTE_MERGE_DEBUG", "").strip() in (
                "1", "true", "True"):
            print(f"  [merge] SplitQKV merge ON (heterogeneous): {tiles_total} "
                  f"tiles, specs {specs}, K {K}, "
                  f"fht={'yes' if fused_fht else 'no'}, "
                  f"ranks {ranks}", flush=True)
        return state

class FusedPalettizedMLP(nn.Module):
    """the MLP decode fusion — a drop-in replacement for the dense
    Qwen3_5MLP wrapper installed by _install_fused_mlp when gate_proj
    and up_proj are fusable PalettizedLinear twins (same pair, GS, K, N,
    equal rotation operands). The DECODE shape (M == 1) collapses

        act_fn(gate_proj(x)) * up_proj(x)

    into ONE qgemm_gemv_mlp(_fht) launch (the kernel's epilogue computes
    silu in fp32 on the folded partials and writes the product with ONE
    fp16 round — the act/mul elementwise kernels and one output write
    disappear, and the two modules' fixed paths stack on the machine
    together; main project: docs/A10G_DECODE_INVESTIGATION.md §12.4 item 2).

    Every other shape, an un-fusable pair, FLUTE_NO_MERGE=1 or an older
    extension takes the ORIGINAL per-module chain below verbatim (a
    performance fallback, never a correctness one — PPL/prefill route
    M > 1 and stay byte-identical). The attribute names are preserved
    (gate_proj/up_proj/down_proj/act_fn) so path-based introspection
    (the probe, resolve_module) keeps working.
    """

    def __init__(self, gate_proj, up_proj, down_proj, act_fn):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn
        self._merge_state = None      # built lazily per device

    def forward(self, x):
        y = self._fused_gate_up(x)
        if y is not None:
            out = self.down_proj(y)
            if x.dim() == 3:
                out = out.view(x.shape[0], x.shape[1], out.shape[-1])
            return out
        # the original chain, verbatim
        return self.down_proj(self.act_fn(self.gate_proj(x))
                              * self.up_proj(x))

    def _fused_gate_up(self, x):
        if os.environ.get("FLUTE_NO_MERGE", "").strip() in ("1", "true",
                                                          "True"):
            return None
        orig3d = None
        if x.dim() == 3:
            b, s, k = x.shape
            if b * s != 1:
                return None
            orig3d = (b, s)
            x = x.reshape(1, k)
        if x.shape[0] != 1:
            return None
        if x.dtype != torch.float16 or not x.is_contiguous() \
                or x.requires_grad:
            return None
        # NOTE: no is_cuda gate here, deliberately — same rationale as
        # the per-module _gemv_decode: the extension entry refuses
        # loudly on CPU and the fake-surface unit tests (main project:
        # tests/test_gemv_merge.py) drive the wiring with CPU tensors.
        try:
            import flute_extended as fx
        except ImportError:
            return None

        state = self._merge_state
        if state is None or state.get("device") != x.device:
            state = self._build_state(x.device, fx)
            self._merge_state = state
        if not state["ok"]:
            return None

        # the heterogeneous call — the two seg-table rows carry the
        # per-blob specs (bitwidths, GS, signs, AWQ); the impl picks the
        # FHT path from the signs columns and stages two rotated rows
        # when the folds differ.
        y = fx.qgemm_gemv_mlp(x, state["seg_table"], state["out"])
        if orig3d is not None:
            y = y.view(orig3d[0], orig3d[1], y.shape[1])
        return y

    def _build_state(self, device, fx):
        """The one-time fusion verdict + buffers (per device). THE
        HETEROGENEOUS CONTRACT: gate and up must share K and N (the
        shapes by construction); the (bitwidth, bitwidth2, group_size),
        the rotation signs, the AWQ scales, the residuals and the biases
        may ALL differ (the deployed mixed-radix artifact: layer 0's
        gate (4,1) vs up (1,4); layers 4/15/28's per-tensor seeds — the
         uniform gate refused all 32 MLP groups). The rotation
        PRESENCE must be all-or-none; the fold order uniform."""
        g, u = self.gate_proj, self.up_proj
        state = {"ok": False, "device": device}
        if os.environ.get("FLUTE_NO_MERGE", "").strip() in ("1", "true",
                                                          "True"):
            state["reason"] = "FLUTE_NO_MERGE=1"
            return state
        for m in (g, u):
            if not isinstance(m, PalettizedLinear):
                state["reason"] = "non-PalettizedLinear member"
                return state
            if m.reference or m.lut_trainable or m.lut2_trainable:
                state["reason"] = "reference/trainable member"
                return state

        def _ok(t, *dtypes):
            return (t is not None and t.is_contiguous()
                    and t.dtype in dtypes)

        for m in (g, u):
            if not _ok(m.indices, torch.uint8) or \
                    not _ok(m.lut, torch.float16):
                state["reason"] = "indices/lut dtype|contiguity"
                return state
            if m.has_stream2 and (not _ok(m.indices2, torch.uint8)
                    or not _ok(m.lut2, torch.float16)):
                state["reason"] = "stream-2 dtype|contiguity"
                return state
        K = g.K
        if u.K != K or u.N != g.N:
            state["reason"] = "K/N mismatch (not MLP twins)"
            return state
        if g.N % 128 != 0:
            state["reason"] = f"N={g.N} not a multiple of 128"
            return state

        # residuals/biases: each 0..256 / fp16-contig or absent
        def _res(m):
            if not m.has_residual():
                return None, None, 0
            if not (_ok(m.resA, torch.float16)
                    and _ok(m.resB, torch.float16)):
                return None, None, -1        # the refusal marker
            r = int(m.resB.shape[0])
            if r > 256:
                return None, None, -1
            return m.resB, m.resA, r

        resBg, resAg, rg = _res(g)
        resBu, resAu, ru = _res(u)
        if rg < 0 or ru < 0:
            state["reason"] = "residual contract"
            return state
        biasg = g.bias if _ok(g.bias, torch.float16) else None
        biasu = u.bias if _ok(u.bias, torch.float16) else None

        # the rotation: PRESENCE all-or-none; per-blob signs/AWQ may
        # differ (the kernel stages two rotated rows — the gate row
        # and the up row — when the folds differ)
        rotated = (g.rot_signs is not None, u.rot_signs is not None)
        fused_fht = all(rotated)
        folds_differ = False
        sign_tensors = []
        awq_tensors = []
        if fused_fht:
            for m in (g, u):
                if m._rotation_backend() != "auto":
                    state["reason"] = "pinned rotation backend"
                    return state
                st = m._signs_on(device)
                if st is None or not _ok(st, torch.float32) \
                        or st.numel() != K or st.data_ptr() % 16 != 0:
                    state["reason"] = "signs contract"
                    return state
                sign_tensors.append(st)
            if g.fold_order != u.fold_order:
                state["reason"] = "fold_order mismatch"
                return state
            g_awq = u_awq = None
            if g.fold_order == "rotate_then_awq":
                for m, which in ((g, "g"), (u, "u")):
                    if m.awq_scale is not None:
                        at = m._awq_scale_on(device)
                        if at is None or not _ok(at, torch.float32) \
                                or at.numel() != K \
                                or at.data_ptr() % 16 != 0:
                            state["reason"] = "awq contract"
                            return state
                        awq_tensors.append(at)
                        if which == "g":
                            g_awq = at
                        else:
                            u_awq = at
            if g_awq is not u_awq and (g_awq is None or u_awq is None):
                state["reason"] = "mixed AWQ presence"
                return state
            # canonicalize value-equal folds onto ONE object (the kernel
            # pointer-compares the per-blob operands: equal values with
            # distinct tensors would stage a second rotated row
            # needlessly — same math, extra smem)
            if g_awq is not None and u_awq is not None \
                    and torch.equal(g_awq, u_awq):
                u_awq = g_awq
                awq_tensors = [g_awq]
            if torch.equal(sign_tensors[0], sign_tensors[1]):
                sign_tensors = [sign_tensors[0], sign_tensors[0]]
            # the fold differs when the sign tensors or awq tensors are
            # not the SAME object
            if sign_tensors[0] is not sign_tensors[1] or \
                    g_awq is not u_awq:
                folds_differ = True
        elif any(rotated):
            state["reason"] = "mixed rotation presence"
            return state

        # the per-blob specs (the heterogeneous surface)
        specs = []
        for m in (g, u):
            b1 = int(m.bitwidth)
            b2 = int(m.bitwidth2) if m.has_stream2 else 0
            specs.append((b1, b2, int(m.group_size)))

        if not getattr(fx, "gemv_mlp_available",
                       lambda: False)() or \
                not fx.gemv_mlp_supported(
                    K, g.N, specs, rg, ru, fused_fht, folds_differ):
            state["reason"] = "extension/gate: mlp"
            return state

        # the persistent output + the CPU segment table [2, 15]
        # (row 0 = gate, row 1 = up; the multi kernel's columns; the C
        # column unused — the product lands in state["out"]). The
        # signs/awq columns carry the CANONICALIZED fold tensors (the
        # same object for both blobs when the values are equal — the
        # kernel's pointer compare then skips the second prologue).
        out = torch.zeros(1, g.N, dtype=torch.float16, device=device)
        fold_g = sign_tensors[0] if sign_tensors else None
        fold_u = sign_tensors[1] if len(sign_tensors) > 1 else None
        awq_g = awq_tensors[0] if awq_tensors else None
        awq_u = awq_tensors[-1] if awq_tensors else None
        rows = []
        for m, resB, resA, r, bias, fold_t, awq_t in (
                (g, resBg, resAg, rg, biasg, fold_g, awq_g),
                (u, resBu, resAu, ru, biasu, fold_u, awq_u)):
            i2 = m.indices2 if m.has_stream2 else None
            l2 = m.lut2 if m.has_stream2 else None
            b1 = int(m.bitwidth)
            b2 = int(m.bitwidth2) if m.has_stream2 else 0
            signs_ptr = fold_t.data_ptr() if fold_t is not None else 0
            awq_ptr = awq_t.data_ptr() if awq_t is not None else 0
            rows.append([
                m.indices.data_ptr(), m.lut.data_ptr(),
                (i2.data_ptr() if i2 is not None else 0),
                (l2.data_ptr() if l2 is not None else 0),
                (resB.data_ptr() if resB is not None else 0),
                (resA.data_ptr() if resA is not None else 0),
                (bias.data_ptr() if bias is not None else 0),
                0,      # the C column: unused (state["out"] carries it)
                m.N, r,
                b1, b2, int(m.group_size),
                signs_ptr, awq_ptr,
            ])
        seg_table = torch.tensor(rows, dtype=torch.int64)  # CPU

        state.update(ok=True, out=out, seg_table=seg_table,
                     fused_fht=fused_fht, specs=specs,
                     folds_differ=folds_differ,
                     keepalive=(sign_tensors, awq_tensors))
        if os.environ.get("FLUTE_MERGE_DEBUG", "").strip() in (
                "1", "true", "True"):
            print(f"  [merge] FusedPalettizedMLP ON (heterogeneous): N={g.N}, "
                  f"K={K}, specs {specs}, "
                  f"fht={'yes' if fused_fht else 'no'}, "
                  f"folds_differ={folds_differ}, Rg/Ru {rg}/{ru}",
                  flush=True)
        return state

def _install_fused_mlp(model):
    """The loader post-pass: wrap every dense Qwen3_5MLP whose
    gate_proj/up_proj are PalettizedLinear in a FusedPalettizedMLP (the
    decode fusion; the wrapper falls back per-call when the pair is not
    fusable — see its class docstring). Called at the END of
    replace_linear_with_palettized, after every linear has been swapped.
    FLUTE_NO_MERGE=1 skips the wrap entirely (the box A/B switch).

    Matched by CLASS NAME (no modeling import — this file stays
    import-clean for CPU boxes); anything that is not the dense MLP
    wrapper with palettized gate/up members is left alone.
    """
    if os.environ.get("FLUTE_NO_MERGE", "").strip() in ("1", "true",
                                                      "True"):
        return 0
    targets = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if type(child).__name__ != "Qwen3_5MLP":
                continue
            gate = getattr(child, "gate_proj", None)
            up = getattr(child, "up_proj", None)
            down = getattr(child, "down_proj", None)
            act = getattr(child, "act_fn", None)
            if (isinstance(gate, PalettizedLinear)
                    and isinstance(up, PalettizedLinear)
                    and down is not None and act is not None):
                targets.append((parent, child_name, child, gate, up,
                                down, act))
    for parent, child_name, child, gate, up, down, act in targets:
        setattr(parent, child_name,
                FusedPalettizedMLP(gate, up, down, act))
    return len(targets)


def _unpack_logical_indices(blob, N: int, K: int, bits: int = 4) -> np.ndarray:
    """(N, K) uint8 logical indices from a packed idxN blob (LSB-first
    through the canonical unpacker, src/docs/QUANTIZATION.md sections 2 and 4)."""
    arr = np.ascontiguousarray(blob.detach().cpu().numpy() if torch.is_tensor(blob) else blob,
        dtype=np.uint8).reshape(-1)
    return _get_idxn().unpack_idxn(arr, N, K, int(bits))


# --------------------------------------------------------------------------- #
# the gather consumer 
# --------------------------------------------------------------------------- #

def _embedding_ids_range_check(ids: torch.Tensor, N: int) -> None:
    """The PalettizedEmbedding ids range gate (loud refusal, never a
    silent wrap through advanced indexing).

    CUDA ids route to torch._assert_async — a DEVICE-side check.
    The older `int(ids.min)/int(ids.max)` was a per-call D2H
    sync: it serialized the host dispatch pipeline every decode step
    AND is an illegal host sync under CUDA-graph capture (the graphed decode would die at the embedding). The device-side assert
    keeps the contract (an out-of-range id aborts loudly on the
    device, never silently) with zero sync. CPU ids keep the ValueError
    path (the error message names the offending range).
    """
    if ids.is_cuda:
        if ids.numel():
            torch._assert_async((ids.min() >= 0) & (ids.max() < N))
        return
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= N):
        raise ValueError(f"PalettizedEmbedding.forward: ids out of range "
            f"[0, {N}) (min {int(ids.min())}, max "
            f"{int(ids.max())}); refusing a silent wrap")


class PalettizedEmbedding(nn.Module):
    """Embedding module with on-the-fly idxN dequantization — a GATHER.

    WHY NO KERNEL (recorded, the design notes section 3): embed_tokens' runtime
    op is a row gather (out[..., :] = W[ids, :]), not a GEMM — the
    FLUTE qgemm kernel does not apply, and no embedding gather kernel
    is written: the dequant for a gather is itself a gather, which pure
    torch expresses directly; this reference path IS the only path. The
    lut_grad_scatter training kernel is likewise NOT applicable (its
    contraction is a GEMM over activation tokens; the embedding
    gradient is a scatter-reduction over selected rows) — embedding LUTs
    train through this path only, and any future CUDA gather kernel
    must refuse while a LUT is trainable, exactly like PalettizedLinear.

    Storage: the PalettizedLinear conventions — non-persistent uint8
    idx blobs, fp16 (G, 2^b) LUTs (fp32 while training), fp16 residual
    factors; the optional second stream (possibly at its OWN idxN width)
    adds ONE ordered fp16 add, stream 1 pinned first. NO bias.

    forward(ids) — the PINNED dtype ladder (mirrored bitwise by T-H6):
        w_s = lut_s[ids // gs].gather(-1, idx_logical_s[ids]) # fp16
        w = w1 + w2 # fp16 add, stream 1 first
        r = resA[ids].float @ resB.float # fp32 rank-r GEMM
        out = (w.float + r).to(lut.dtype) # ONE final cast
    Output dtype = the stream-1 LUT's dtype (fp16 deployed — the
    embedding-lookup convention; fp32 while the master trains).

    Rotation (, the tied-head case): when the embedding shares the
    lm_head's ROTATED artifact set (tie_word_embeddings — one set,
    "tied": true, produced through the lm_head step with
    rotate=True), the gathered rows are FOLD-SPACE vectors and must be
    un-rotated per row: out <- out @ T^T — the ADJOINT FHT (the same
    butterfly, signs on the input side; O(K log K) per token, no K x K
    matrix). The un-rotation applies to the SUM (w + r) — linearity —
    before the final cast. Untied embed artifacts are UNROTATED (the
    head pass never folds them; no input transform exists to fold
    into) and carry no rotation record — zero behavior change.

    Trainable LUT (straight-through): make_trainable promotes BOTH
    LUTs to fp32 Parameters — the gather is differentiable w.r.t. the
    LUT, indices frozen; freeze_lut snaps both; deploy_clone same.
    """

    def __init__(self, indices, lut, bitwidth, group_size, N, K,
                 indices2=None, lut2=None, bitwidth2=None, resA=None,
                 resB=None, num_embeddings=None, embedding_dim=None,
                 rotation_seed: int = None, rotation_k: int = None):
        """N/K are the PalettizedLinear spellings (rows, columns) and
        the nn.Embedding spellings num_embeddings/embedding_dim are
        accepted as cross-checked aliases (a mismatch is a hard error;
        both are exposed back as read-only properties)."""
        super().__init__()
        self.bitwidth = int(bitwidth)
        if self.bitwidth not in (1, 2, 3, 4):
            raise ValueError(f"PalettizedEmbedding: bitwidth {self.bitwidth} not in "
                f"1..4 — the idxN artifact family is the contract")
        # : the second stream may carry its own idxN width
        self.bitwidth2 = int(bitwidth2) if bitwidth2 is not None \
            else self.bitwidth
        if self.bitwidth2 not in (1, 2, 3, 4):
            raise ValueError(f"PalettizedEmbedding: bitwidth2 {self.bitwidth2} not in "
                f"1..4 — the idxN artifact family is the contract")
        self.group_size = int(group_size)
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("lut", lut, persistent=False)
        self.N = int(N)
        self.K = int(K)
        if num_embeddings is not None and int(num_embeddings) != self.N:
            raise ValueError(f"PalettizedEmbedding: num_embeddings {num_embeddings} != N "
                f"{self.N}; the two spellings must agree")
        if embedding_dim is not None and int(embedding_dim) != self.K:
            raise ValueError(f"PalettizedEmbedding: embedding_dim {embedding_dim} != K "
                f"{self.K}; the two spellings must agree")
        # idxN artifact contract — unconditional (no kernel path whose
        # reference arm could excuse a lazy check)
        if self.N % 128 != 0 or self.K % 64 != 0:
            raise ValueError(f"idxN layout requires N%128==0 and K%64==0, got "
                f"N={self.N}, K={self.K}")
        expected = (self.N * self.K * self.bitwidth) // 8
        if indices.numel() != expected:
            raise ValueError(f"idx{self.bitwidth} blob numel {indices.numel()} != "
                f"N*K*{self.bitwidth}/8 {expected} (N={self.N}, K={self.K})")
        palette = 1 << self.bitwidth
        palette2 = 1 << self.bitwidth2
        lut1_t = self.lut if torch.is_tensor(self.lut) \
            else torch.from_numpy(np.asarray(self.lut))
        if lut1_t.dim() != 2:
            lut1_t = lut1_t.view(-1, palette)
        if lut1_t.shape[1] != palette:
            raise ValueError(f"PalettizedEmbedding: LUT palette width "
                f"{lut1_t.shape[1]} != 2**bitwidth {palette}")
        if resA is not None and resB is not None:
            if resA.shape != (self.N, resB.shape[0]) or \
                    resB.shape[1] != self.K:
                raise ValueError(f"residual factors shape mismatch: A "
                    f"{tuple(resA.shape)}, B {tuple(resB.shape)} for "
                    f"N={self.N}, K={self.K}")
            self.register_buffer("resA", resA, persistent=False)
            self.register_buffer("resB", resB, persistent=False)
        else:
            self.resA = None
            self.resB = None
        # -- second stream (None/None = legacy, no new buffers) ----- #
        if indices2 is None and lut2 is None:
            self.indices2 = None
            self.lut2 = None
        elif indices2 is None or lut2 is None:
            raise ValueError(
                "PalettizedEmbedding: a second stream needs BOTH indices2 "
                f"and lut2 (got indices2={'None' if indices2 is None else 'set'},"
                f" lut2={'None' if lut2 is None else 'set'}); refusing a "
                "half-constructed two-stream module")
        else:
            expected2 = (self.N * self.K * self.bitwidth2) // 8
            if indices2.numel() != expected2:
                raise ValueError(f"idx{self.bitwidth2} stream-2 blob numel "
                    f"{indices2.numel()} != N*K*{self.bitwidth2}/8 "
                    f"{expected2} (N={self.N}, K={self.K})")
            lut2_t = lut2 if torch.is_tensor(lut2) \
                else torch.from_numpy(np.asarray(lut2))
            if lut2_t.dim() != 2:
                lut2_t = lut2_t.view(-1, palette2)
            if lut2_t.shape[1] != palette2:
                raise ValueError(f"stream-2 LUT palette width {lut2_t.shape[1]} != "
                    f"2**bitwidth2 {palette2}; stream 2 is an "
                    f"idx{self.bitwidth2} codebook")
            if lut2_t.shape[0] != lut1_t.shape[0]:
                raise ValueError(f"stream-2 LUT groups {lut2_t.shape[0]} != stream-1 "
                    f"{lut1_t.shape[0]}; both streams must share the "
                    f"group grid (same N, K, group_size)")
            self.register_buffer("indices2", indices2, persistent=False)
            self.register_buffer("lut2", lut2, persistent=False)
        # the (N, K) logical-index caches — the gather forward's STANDING
        # state (unlike PalettizedLinear, which caches only while a LUT
        # trains): built lazily, released by freeze_lut and by a device
        # move (rebuilt on the next forward).
        # the caches are UINT8 (the unpacked indices are uint8 by
        # construction, values < 2^b <= 16). At the head geometry
        # (N=248320, K=4096) the old int64 cache was 7.58 GiB PER STREAM
        # on the LUT's device — 8x the artifact it serves (the R2
        # two-stream head pair took 15.16 GiB standing, which is what
        # pushed the greedy-eval VRAM to the ceiling). uint8 is 0.95 GiB
        # per stream; _gather_stream promotes the per-token ROW SLICE to
        # int64 for torch.gather (a (n_tokens, K) promotion, not (N, K)).
        self._idx_logical: Optional[torch.Tensor] = None
        self._idx_logical2: Optional[torch.Tensor] = None
        # -- rotation (the tied-head case;) ---------------------------- #
        self.rotation_seed = (int(rotation_seed)
                              if rotation_seed is not None else None)
        self.rotation_k = (int(rotation_k)
                           if rotation_k is not None else None)
        self.rot_signs = None
        self._signs_cache = {}
        if self.rotation_seed is not None and self.rotation_k is not None:
            if self.rotation_k != self.K:
                raise ValueError(f"PalettizedEmbedding: rotation_k {self.rotation_k} != "
                    f"the embedding dim K {self.K} — corrupt rotation "
                    f"record on a tied artifact set")
            self.rot_signs = _rotation_signs_for(self.rotation_k,
                                                 self.rotation_seed)

    def _signs_on(self, device) -> torch.Tensor:
        """The (K,) sign tensor on `device` (per-device cache)."""
        key = str(device)
        cached = self._signs_cache.get(key)
        if cached is None:
            cached = self.rot_signs.to(device)
            self._signs_cache[key] = cached
        return cached

    # -- the nn.Embedding spellings (read-only aliases) ------------------ #

    @property
    def num_embeddings(self) -> int:
        return self.N

    @property
    def embedding_dim(self) -> int:
        return self.K

    def has_residual(self) -> bool:
        return self.resA is not None and self.resB is not None

    @property
    def has_stream2(self) -> bool:
        """True when the optional second stream (Route A pair-composite)
        is present — False for the legacy single-stream module."""
        return self.indices2 is not None and self.lut2 is not None

    # -- straight-through LUT training API -------------------------------- #

    @property
    def lut_trainable(self) -> bool:
        return isinstance(self.lut, nn.Parameter) and self.lut.requires_grad

    @property
    def lut2_trainable(self) -> bool:
        return (self.lut2 is not None and isinstance(self.lut2, nn.Parameter)
                and self.lut2.requires_grad)

    def _logical_indices_numpy(self, blob=None, bits=None) -> np.ndarray:
        """(N, K) uint8 logical indices of one packed blob (stream 1 by
        default; pass the stream-2 blob — and its width — for the second
        stream)."""
        return _unpack_logical_indices(self.indices if blob is None else blob, self.N, self.K,
            self.bitwidth if bits is None else int(bits))

    @torch.no_grad()
    def _cache_logical_indices(self) -> "PalettizedEmbedding":
        """Build the (N, K) uint8 logical-index caches (BOTH streams) on
        the LUT's device (uint8, 1 B/element — see the class
        docstring). Idempotent; a device move (module.to) that left a
        cache behind invalidates it first. Indices are frozen constants
        — the straight-through convention."""
        lut_dev = self.lut.device
        if self._idx_logical is not None and \
                self._idx_logical.device != lut_dev:
            self._idx_logical = None
        if self._idx_logical2 is not None and \
                self._idx_logical2.device != lut_dev:
            self._idx_logical2 = None
        if self._idx_logical is None:
            # uint8 — the unpacker's native dtype (values < 2^b). The
            # int64 promotion happens per gathered ROW in _gather_stream.
            idx_t = torch.from_numpy(self._logical_indices_numpy())
            if lut_dev != torch.device("cpu"):
                idx_t = idx_t.to(lut_dev)
            self._idx_logical = idx_t
        if self.has_stream2 and self._idx_logical2 is None:
            idx_t2 = torch.from_numpy(self._logical_indices_numpy(self.indices2, self.bitwidth2))
            if lut_dev != torch.device("cpu"):
                idx_t2 = idx_t2.to(lut_dev)
            self._idx_logical2 = idx_t2
        return self

    def make_trainable(self) -> "PalettizedEmbedding":
        """Promote BOTH LUT buffers (stream 1 and the optional stream 2)
        to independent fp32 nn.Parameters (in place, idempotent). The
        forward gather is differentiable w.r.t. the LUT (the gradient
        is a scatter-reduction over the selected rows — autograd's
        index/gather backward); the indices stay frozen constants."""
        if not isinstance(self.lut, nn.Parameter):
            lut32 = self.lut.detach().float().clone()
            del self.lut
            self.lut = nn.Parameter(lut32)
        if self.has_stream2 and not isinstance(self.lut2, nn.Parameter):
            lut32b = self.lut2.detach().float().clone()
            del self.lut2
            self.lut2 = nn.Parameter(lut32b)
        return self

    def freeze_lut(self, snap_fp16: bool = True) -> "PalettizedEmbedding":
        """Demote BOTH LUTs back to the deployment buffers
        (fp16-snapped). Idempotent; releases BOTH logical-index caches
        (rebuilt lazily by the next forward)."""
        if isinstance(self.lut, nn.Parameter):
            lut16 = self.lut.detach()
            if snap_fp16:
                lut16 = lut16.to(torch.float16)
            del self.lut
            self.register_buffer("lut", lut16.contiguous(), persistent=False)
        self._idx_logical = None
        if self.has_stream2:
            if isinstance(self.lut2, nn.Parameter):
                lut16b = self.lut2.detach()
                if snap_fp16:
                    lut16b = lut16b.to(torch.float16)
                del self.lut2
                self.register_buffer("lut2", lut16b.contiguous(),
                                     persistent=False)
            self._idx_logical2 = None
        return self

    def snapped_lut(self) -> torch.Tensor:
        """fp16-grid copy of the current LUT (train master -> deploy
        grid)."""
        return self.lut.detach().to(torch.float16)

    def snapped_lut2(self) -> Optional[torch.Tensor]:
        """fp16-grid copy of the stream-2 LUT (None when stream 2 is
        absent) — the train master -> deploy grid cast, stream 2."""
        if self.lut2 is None:
            return None
        return self.lut2.detach().to(torch.float16)

    def deploy_clone(self) -> "PalettizedEmbedding":
        """Deployment-parity copy: frozen fp16 (snapped) LUTs, both
        streams carried, indices shared (frozen anyway), residual
        factors cloned — the PalettizedLinear convention (the
        rotation record travels with the clone)."""
        return PalettizedEmbedding(self.indices, self.snapped_lut(), self.bitwidth,
            self.group_size, self.N, self.K, indices2=self.indices2,
            lut2=self.snapped_lut2(), bitwidth2=self.bitwidth2,
            resA=None if self.resA is None else self.resA.detach().clone(),
            resB=None if self.resB is None else self.resB.detach().clone(),
            rotation_seed=self.rotation_seed, rotation_k=self.rotation_k)

    # -- forward ----------------------------------------------------------- #

    def _gather_stream(self, ids, lut, idx_logical, bits=None) -> torch.Tensor:
        """One stream's per-token rows : the (…, 2^b)
        LUT row per token group, gathered along the palette axis at the
        token's K logical indices (out[..., k] = lut_row[..., idx[..., k]])
        — differentiable w.r.t. `lut` (straight-through). `bits` is the
        stream's idxN width (stream 2 may differ from stream 1).

        `idx_logical` is the uint8 cache; the ROW SLICE
        idx_logical[ids] — (n_tokens, K), not (N, K) — is promoted to
        int64 here for torch.gather (uint8 values < 2^b are exact in
        int64; the promotion cost is per-token, O(n_tokens * K), never
        model-scale)."""
        lut_t = lut if torch.is_tensor(lut) \
            else torch.from_numpy(np.asarray(lut))
        if lut_t.dim() != 2:
            lut_t = lut_t.view(-1, 1 << (self.bitwidth if bits is None
                                         else int(bits)))
        return lut_t[ids // self.group_size].gather(
            -1, idx_logical[ids].long())

    def forward(self, ids):
        """ids: any integer tensor shape (…); returns (…, K) rows, dtype
        = the stream-1 LUT's dtype (the pinned ladder, class
        docstring). Ids outside [0, N) are refused loudly — a negative
        id would otherwise wrap silently through advanced indexing."""
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids)
        if ids.dtype not in (torch.int64, torch.int32, torch.int16,
                             torch.int8, torch.uint8):
            raise ValueError(f"PalettizedEmbedding.forward: ids must be an integer "
                f"tensor, got dtype {ids.dtype}")
        ids = ids.long()
        _embedding_ids_range_check(ids, self.N)
        self._cache_logical_indices()
        # stream 1 first — the pinned order
        w = self._gather_stream(ids, self.lut, self._idx_logical)
        if self.has_stream2:
            w = w + self._gather_stream(ids, self.lut2, self._idx_logical2,
                                        bits=self.bitwidth2)
        if self.resA is not None and self.resB is not None:
            # the residual branch: fp32 GEMM on the gathered rows, the
            # fp32 add, ONE final cast (the class-docstring ladder)
            w = w.float() + (self.resA[ids].float() @ self.resB.float())
        else:
            w = w.float()
        # (tied-head rotation): the gathered rows live in the FOLD
        # space — un-rotate each row before the final cast: out @ T^T,
        # the adjoint FHT (signs on the input side). Linearity makes
        # one adjoint on (w + r) exact; fp32 in, ONE cast out.
        if self.rot_signs is not None:
            fht = _get_fht()
            backend_choice = os.environ.get("FLUTE_ROTATION", "auto") \
                .strip().lower()
            backend = "auto" if backend_choice in ("", "auto", "fht") \
                else backend_choice
            w = fht.fht_adjoint(w, self._signs_on(w.device),
                                backend=backend)
        return w.to(self.lut.dtype)


# --------------------------------------------------------------------------- #
# Artifact loaders
# --------------------------------------------------------------------------- #

def fold_input_gram(signs, awq_scale, fold_order, H):
    """The (K, K) Gram of a fold module's FOLD-SPACE input, from the
    captured PRISTINE-input Gram H = E[x x^T] (the persisted
    grams/<name>.gram.npy is pre-fold — the capture hooks tap the
    pristine module input; the producer transforms it internally through
    the same congruences, makes them explicit for the consumers that
    re-fit weights in the fold frame: the trainer's export polish and
    qlora_merge's re-palettization).

      legacy order (W' = W @ T @ D): fold(x) = x T D^-1
          -> D^-1 (T^T H T) D^-1
       order (W' = (W @ D) @ T): fold(x) = x D^-1 T
          -> T^T (D^-1 H D^-1) T
      rotation only: T^T H T

    Mirrors the producer's own _apply_awq_transform/_apply_boundary_fold
    congruences (the trace-loss invariant). `signs` is the (K,) rotation
    sign tensor; `awq_scale` the (K,) frozen scale (None when the fold
    carries no AWQ). Returns fp32 CPU."""
    fht = _get_fht()
    Hd = H.float().cpu()
    if awq_scale is None:
        A = fht.fht_apply(Hd, signs)
        return fht.fht_apply(A.t().contiguous(), signs).t()
    s = awq_scale.detach().to(Hd.device, torch.float32).reshape(-1)
    if fold_order == "rotate_then_awq":
        A = fht.fht_apply(Hd, signs)
        G = fht.fht_apply(A.t().contiguous(), signs).t()
        return G / s.view(-1, 1) / s.view(1, -1)
    B = Hd / s.view(-1, 1) / s.view(1, -1)
    return fht.fht_apply(B.t().contiguous(), signs).t()


# the stale-group_size mismatch ledger. The older loader printed
# one loud line PER mismatching tensor component — 132 lines on the
# 2026-10-07 artifacts dump (all in_proj_qkv Q/K/V components, both
# streams), noise that hides real signals. The loader's POLICY is
# unchanged (trust the LUT geometry, derive the gs); only the
# reporting is aggregated: the FIRST mismatch keeps its detailed line
# (single-mismatch behavior is byte-identical), the rest are counted,
# and replace_linear_with_palettized prints one [GS] summary line.
_GS_MISMATCHES = []


def _note_gs_mismatch(ctx: str, meta_gs: int, n_groups: int, N: int,
                       derived: int) -> None:
    """Record a stale metadata group_size (loader trusts the LUT geometry)."""
    _GS_MISMATCHES.append((ctx, meta_gs, n_groups, N, derived))
    if len(_GS_MISMATCHES) == 1:
        print(f"[palettized_modules] {ctx}: metadata group_size "
              f"{meta_gs} is inconsistent with the LUT geometry "
              f"(n_groups={n_groups}, N={N} -> derived {derived}); "
              f"using the derived value", flush=True)


def _summarize_gs_mismatches() -> None:
    """The one-line aggregate (call at the end of a full module swap)."""
    if len(_GS_MISMATCHES) > 1:
        first = _GS_MISMATCHES[0]
        print(f"  [GS] group_size metadata mismatch on "
              f"{len(_GS_MISMATCHES)} tensor component(s) — the recorded "
              f"sweep field is stale (first: {first[0]}: gs {first[1]} vs "
              f"derived {first[4]}); the LUT geometry is trusted for every "
              f"one of them (policy;  aggregation). Fix the writer: "
              f"the deployed composition's gs is what the metadata should "
              f"record.", flush=True)
    _GS_MISMATCHES.clear()


def _read_indices_and_lut(meta: Dict, artifacts_dir: str, ctx: str,
                          shape=None, verify_sha: bool = True
) -> Tuple[torch.Tensor, torch.Tensor,
                                     int, int, int, int]:
    """Read + validate one (index_file, lut_file) pair. Returns
    (indices_blob, lut, bitwidth, group_size, N, K).

    `shape=(N, K)` overrides metadata dense_shape — required for QKV
    components, whose metadata carries no dense_shape of its own (the
    parent tensor does; the component N is derived from packed_len_bytes).
    """
    idx_path = os.path.join(artifacts_dir, meta["index_file"])
    lut_path = os.path.join(artifacts_dir, meta["lut_file"])
    validate_indices_layout(meta, ctx)
    _check_sha(idx_path, meta.get("sha256_idx"), ctx, verify=verify_sha)
    _check_sha(lut_path, meta.get("sha256_lut"), ctx, verify=verify_sha)

    with open(idx_path, "rb") as f:
        idx_bytes = f.read()
    lut_np = np.fromfile(lut_path, dtype=np.float16)

    bitwidth = int(meta["bitwidth"])
    meta_gs = int(meta["group_size"])
    palette_size = 1 << bitwidth
    n_groups = lut_np.size // palette_size
    if lut_np.size % palette_size:
        raise ValueError(f"{ctx}: LUT size {lut_np.size} not divisible by "
                         f"palette size {palette_size}")
    lut_t = torch.from_numpy(lut_np.copy()).view(n_groups, palette_size)

    N, K = shape if shape is not None else meta.get("dense_shape",
                                                    [None, None])
    if N is None:
        raise ValueError(f"{ctx}: metadata missing dense_shape")
    
    # honor the RECORDED group_size whenever it is geometrically
    # consistent with the LUT (n_groups == ceil(N / meta_gs)). The
    # quantizer groups rows by r // gs with exactly n_groups = ceil(N/gs)
    # LUT rows, and the kernel's own contract is that same ceil check
    # (lut rows == ceil(N/group_size); N itself may be ragged) — so a
    # consistent recorded gs is the EXACT grouping the artifact was
    # quantized under, including GS 1024/2048 at N=151936 where the
    # derived ceil(N/n_groups) lands on 1020/2026: a value outside the
    # kernel's dispatch set that would fail the GS TORCH_CHECK and, on
    # the reference path, silently REGROUP the dequant. The derive-first
    # policy existed because the writer under-recorded gs by one (a
    # floor-division bug, fixed at the source — the resolver's
    # chosen gs is now recorded verbatim); a genuinely stale or
    # inconsistent field still falls back to the derived value, loudly
    # (the per-tensor lines are aggregated — first detailed line +
    # a [GS] count summary at the end of the swap; see
    # _note_gs_mismatch).
    if n_groups > 0:
        derived = (int(N) + n_groups - 1) // n_groups   # ceil for safety
        if meta_gs > 0 and n_groups == (int(N) + meta_gs - 1) // meta_gs:
            group_size = meta_gs
        else:
            group_size = derived
            if meta_gs > 0 and meta_gs != derived:
                _note_gs_mismatch(ctx, meta_gs, n_groups, int(N), derived)
    else:
        group_size = meta_gs  # fallback to metadata if LUT is empty
    # idxN: K*bitwidth/8 bytes per row (exact — K % 64 == 0 in every
    # artifact the writer emits; b=3's 3K/8 is integral there). The
    # legacy ceil((K*8/ipb)/8) form broke at b=3 (8//3 == 2).
    K_packed = (int(K) * bitwidth + 7) // 8
    expected_bytes = N * K_packed
    if len(idx_bytes) != expected_bytes:
        raise ValueError(f"{ctx}: index file {meta['index_file']} has {len(idx_bytes)} B, "
            f"expected N*K_packed = {expected_bytes} B "
            f"(N={N}, K={K}, bitwidth={bitwidth})")
    indices = torch.from_numpy(np.frombuffer(idx_bytes, dtype=np.uint8).copy())
    return indices, lut_t, bitwidth, group_size, int(N), int(K)


def _read_residual(meta: Dict, artifacts_dir: str, ctx: str, N: int, K: int):
    """Read fp16 residual factors when the metadata carries them."""
    res = meta.get("residual")
    if not res:
        return None, None
    a_path = os.path.join(artifacts_dir, res["resA_file"])
    b_path = os.path.join(artifacts_dir, res["resB_file"])
    _check_sha(a_path, res.get("sha256_resA"), ctx)
    _check_sha(b_path, res.get("sha256_resB"), ctx)
    A = torch.from_numpy(np.fromfile(a_path, dtype=np.float16).copy())
    B = torch.from_numpy(np.fromfile(b_path, dtype=np.float16).copy())
    rank = int(res["rank"])
    A = A.view(N, rank)
    B = B.view(rank, K)
    return A, B


def _read_stream2(meta: Dict, artifacts_dir: str, ctx: str, N: int, K: int,
                  bitwidth: int, group_size: int, verify_sha: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Read + validate the SECOND member of a declared "streams" set
    (, Route A; single tensors and QKV components share this).

    The writer's pinned entry schema (main project: palettize_qwen3_5_9b.py::
    _stream_entry): {file, sha256, sha256_lut, n_groups, gs,
    derivation} — `file` is the stream-2 idxN blob name
    ("<san>.idx4.2" for hybrid422, "<san>.idx2.2" for a mixed 4,2
    spec); the paired LUT file follows the writer's
    "<san>.lut_scalar<tag>" convention and is derived from the blob
    name. Stream 1 stays the meta's own index_file/lut_file pair, so
    only the second member is read here. The stream's idxN width is
    taken from the entry's recorded bitwidth (the writer records
    it), falling back to the FILE NAME's declared width, falling back
    to the stream-1 width (legacy hybrid422 entries record nothing).

    T-H4 at load time: a missing/unreadable stream-2 member for a
    tensor whose metadata declares streams is a HARD ValueError naming
    the file and the tensor — never a silent single-stream fallback.

    Returns (indices2, lut2, bitwidth2).
    """
    streams = meta.get("streams")
    if not isinstance(streams, list) or len(streams) < 2:
        raise ValueError(f"{ctx}: metadata 'streams' must be a list with at least 2 "
            f"entries, got {streams!r}; refusing an ambiguous stream set")
    base = streams[0] if isinstance(streams[0], dict) else {}
    if base.get("file") != meta.get("index_file"):
        raise ValueError(f"{ctx}: streams[0] file {base.get('file')!r} != the meta's "
            f"index_file {meta.get('index_file')!r}; the first entry must "
            f"describe the base stream")
    entry = streams[1]
    if not isinstance(entry, dict) or "file" not in entry:
        raise ValueError(f"{ctx}: stream-2 entry must be a dict carrying 'file' "
            f"(got {entry!r})")
    idx2_name = str(entry["file"])
    if not _is_idxn_artifact_name(idx2_name):
        raise ValueError(f"{ctx}: stream-2 file {idx2_name!r} is not a (tagged) idxN "
            f"artifact name")
    # the stream's width: the entry's recorded bitwidth, else the file
    # name's declaration, else the stream-1 width (legacy entries)
    if entry.get("bitwidth") is not None:
        bits2 = int(entry["bitwidth"])
        if bits2 != _bits_of_artifact_name(idx2_name):
            raise ValueError(f"{ctx}: stream-2 entry bitwidth {bits2} != the file name's "
                f"idx{_bits_of_artifact_name(idx2_name)} ({idx2_name!r})")
    else:
        try:
            bits2 = _bits_of_artifact_name(idx2_name)
        except ValueError:
            bits2 = int(bitwidth)
    lut2_name = _lut_name_of(idx2_name)
    if lut2_name is None:
        raise ValueError(f"{ctx}: cannot derive the stream-2 LUT name from "
            f"{idx2_name!r}")
    idx2_path = os.path.join(artifacts_dir, idx2_name)
    lut2_path = os.path.join(artifacts_dir, lut2_name)
    for path, fname in ((idx2_path, idx2_name), (lut2_path, lut2_name)):
        if not os.path.exists(path):
            raise ValueError(f"{ctx}: metadata declares {len(streams)} streams but "
                f"stream-2 member {fname!r} is missing (expected {path}); "
                f"the two-stream artifact set is incomplete — re-run "
                f"palettization (never a silent single-stream fallback)")
    gs2 = int(entry.get("gs", group_size))
    # the stream-2 entry's RECORDED gs rides the member metadata into
    # _read_indices_and_lut, which honors it when geometrically consistent
    # (n_groups == ceil(N/gs)) and otherwise falls back to the derived
    # value with a loud note — both streams of a two-stream artifact share
    # one group grid at write time, so the entries stay consistent.
    member = {
        "index_file": idx2_name,
        "lut_file": lut2_name,
        "sha256_idx": entry.get("sha256"),
        "sha256_lut": entry.get("sha256_lut"),
        "bitwidth": bits2,
        "group_size": gs2,
        "indices_layout": f"idx{bits2}",
    }
    indices2, lut2_t, _bw2, _gs2, _N2, _K2 = _read_indices_and_lut(member, artifacts_dir, ctx=f"{ctx}:stream2", shape=(N, K),
        verify_sha=verify_sha)
    if entry.get("n_groups") is not None and \
            int(entry["n_groups"]) != lut2_t.shape[0]:
        raise ValueError(f"{ctx}: stream-2 n_groups {entry['n_groups']} != the LUT's "
            f"{lut2_t.shape[0]} rows (corrupt stream-2 metadata)")
    return indices2, lut2_t, bits2


def _awq_scale_for_tensor(tensor_meta: Dict, awq_scales: Optional[Dict]):
    """The (K,) AWQ scale for one tensor, from the loader's recovered
    per-consumer map — None when the tensor is unscaled.

    `awq_scales` maps CONSUMER tensor names (the norm_gain_edits.json
    "consumers" list — fused names like
    model.layers.0.linear_attn.in_proj_qkv.weight) to (K,) tensors.
    QKV components inherit the FUSED tensor's scale (they share the
    module input and the one fold). fold_order comes from the tensor's
    own "awq" record when present (+ artifacts), else None (the
    constructor then assumes the legacy rotate-then-AWQ order).
    """
    if not awq_scales:
        return None, None
    var = tensor_meta.get("var") or tensor_meta.get("name")
    s = awq_scales.get(var)
    if s is None:
        # a QKV fused meta's consumers entry matches the fused name; a
        # component meta (no var of its own) is resolved by the caller
        # passing the FUSED meta here
        return None, None
    awq = tensor_meta.get("awq") or {}
    fold_order = awq.get("fold_order") or \
        (tensor_meta.get("rotation") or {}).get("order")
    return s, fold_order


def load_palettized_weight(tensor_meta: Dict, artifacts_dir: str, bias=None,
                           residual: bool = False, reference: bool = False,
                           verify_sha: bool = True,
                           awq_scales: Optional[Dict] = None):
    """Returns (PalettizedLinear | SplitQKV).

    `bias`: the original module's bias Parameter (may be None). For QKV
    splits it is sliced per component (each component carries its own N).
    `residual`: attach the whitened-SVD residual branch when the artifacts
    carry one.
    `awq_scales` : the recovered per-consumer AWQ scale map (see
    _recover_awq_scales) — tensors whose fold carries BOTH the rotation
    and an AWQ scale get the exact compensation for the legacy
    rotate-then-AWQ producer order (or the recorded new order's clean
    composition). QKV components read the FUSED meta's awq record.

    Route A: a tensor meta (or, symmetrically, a QKV component
    meta) carrying a "streams" list constructs the module with BOTH
    streams — stream 1 from index_file/lut_file, stream 2 from
    streams[1] (validated by _read_stream2; a missing member is a hard
    error). The construction is symmetric across the two branches —
    no component ever silently drops its declared stream 2.
    """
    if "components" in tensor_meta:
        K = tensor_meta["dense_shape"][1]
        # the FUSED meta's AWQ record governs every component (the fold
        # was applied to the fused tensor before the split — one fold,
        # one scale, shared K axis)
        fused_s, fused_fold = _awq_scale_for_tensor(tensor_meta, awq_scales)
        fused_fold = fused_fold or \
            ((tensor_meta.get("awq") or {}).get("fold_order"))
        comps, biases = [], []
        for comp_name in ("Q", "K", "V"):
            cm = tensor_meta["components"][comp_name]
            comp_ctx = f"{tensor_meta.get('var', '?')}:{comp_name}"
            # idxN: N = packed_len_bytes * 8 // (K * bitwidth) (the
            # legacy 8//bitwidth form broke at b=3)
            comp_N = int(cm["packed_len_bytes"]) * 8 \
                // (int(K) * int(cm["bitwidth"]))
            indices, lut_t, bw, gs, N_, K_ = _read_indices_and_lut(cm, artifacts_dir, ctx=comp_ctx,
                shape=(comp_N, K), verify_sha=verify_sha)
            indices2 = lut2_t = None
            bw2 = None
            if cm.get("streams"):
                # symmetric per-component construction : a
                # component's own streams list rides the same helper —
                # the small additive shape, so implemented rather than
                # refused (never a silent stream-2 drop)
                indices2, lut2_t, bw2 = _read_stream2(cm, artifacts_dir, ctx=comp_ctx, N=N_, K=K_,
                    bitwidth=bw, group_size=gs, verify_sha=verify_sha)
            resA = resB = None
            if residual:
                resA, resB = _read_residual(cm, artifacts_dir, ctx=comp_ctx, N=N_, K=K_)
            # Rotation for QKV component (same fold for every
            # component: one seed, shared K axis — the split happened
            # AFTER the fold)
            comp_rot = cm.get("rotation")
            rot_seed = comp_rot["seed"] if comp_rot else None
            rot_k = comp_rot["k"] if comp_rot else None
            comps.append(PalettizedLinear(indices, lut_t, bw, gs, N_, K_, resA=resA, resB=resB,
                indices2=indices2, lut2=lut2_t, bitwidth2=bw2,
                reference=reference, rotation_seed=rot_seed, rotation_k=rot_k,
                awq_scale=fused_s, fold_order=fused_fold))
            biases.append(None)
        if bias is not None:
            off = 0
            for i, m in enumerate(comps):
                n = m.N
                biases[i] = bias[off:off + n].clone()
                off += n
            if off != bias.shape[0]:
                raise ValueError(f"{tensor_meta.get('var', '?')}: bias length "
                    f"{bias.shape[0]} != sum of component N ({off})")
        for m, b in zip(comps, biases):
            if b is not None:
                m.bias = nn.Parameter(b)
        return SplitQKV(*comps)
    indices, lut_t, bw, gs, N, K = _read_indices_and_lut(tensor_meta, artifacts_dir, ctx=tensor_meta.get("var", "?"),
        verify_sha=verify_sha)
    resA = resB = None
    if residual:
        resA, resB = _read_residual(tensor_meta, artifacts_dir, ctx=tensor_meta.get("var", "?"),
            N=N, K=K)
    indices2 = lut2_t = None
    bw2 = None
    if tensor_meta.get("streams"):
        indices2, lut2_t, bw2 = _read_stream2(tensor_meta, artifacts_dir, ctx=tensor_meta.get("var", "?"),
            N=N, K=K, bitwidth=bw, group_size=gs, verify_sha=verify_sha)
    # Rotation + AWQ composition for a regular (non-QKV) tensor
    rot = tensor_meta.get("rotation")
    rot_seed = rot["seed"] if rot else None
    rot_k = rot["k"] if rot else None
    awq_s, fold_order = _awq_scale_for_tensor(tensor_meta, awq_scales)
    if bias is not None:
        return PalettizedLinear(indices, lut_t, bw, gs, N, K,
                                bias=bias.clone(), resA=resA, resB=resB,
                                indices2=indices2, lut2=lut2_t,
                                bitwidth2=bw2, reference=reference,
                                rotation_seed=rot_seed, rotation_k=rot_k,
                                awq_scale=awq_s, fold_order=fold_order)
    return PalettizedLinear(indices, lut_t, bw, gs, N, K,
                            resA=resA, resB=resB, indices2=indices2,
                            lut2=lut2_t, bitwidth2=bw2,
                            reference=reference, rotation_seed=rot_seed,
                            rotation_k=rot_k, awq_scale=awq_s,
                            fold_order=fold_order)


# --------------------------------------------------------------------------- #
# the embedding + tied-pair loader entries 
# --------------------------------------------------------------------------- #

def load_palettized_embedding(tensor_meta: Dict, artifacts_dir: str,
                              residual: bool = True,
                              verify_sha: bool = True):
    """Returns PalettizedEmbedding (gather consumer).

    Mirrors load_palettized_weight's single-tensor branch: the declared
    "streams" set constructs BOTH streams (a missing/unreadable
    stream-2 member is a hard error, never a single-stream fallback),
    every member's sha256 is validated against the file and the idx4
    layout gate runs; `residual=True` (the default — the head-pass
    embed set always carries the rank-r pair) attaches .resA/.resB
    when the artifacts have them. An embedding meta never carries QKV
    "components" (embed_tokens is a single matrix) — refused loudly.
    """
    if "components" in tensor_meta:
        raise ValueError(f"{tensor_meta.get('var', '?')}: an embedding meta carries no "
            f"QKV components (embed_tokens is a single matrix, never a "
            f"fused projection); refusing to guess a component split")
    ctx = tensor_meta.get("var", "?")
    indices, lut_t, bw, gs, N, K = _read_indices_and_lut(tensor_meta, artifacts_dir, ctx=ctx, verify_sha=verify_sha)
    resA = resB = None
    if residual:
        resA, resB = _read_residual(tensor_meta, artifacts_dir, ctx=ctx,
                                    N=N, K=K)
    indices2 = lut2_t = None
    bw2 = None
    if tensor_meta.get("streams"):
        indices2, lut2_t, bw2 = _read_stream2(tensor_meta, artifacts_dir, ctx=ctx, N=N, K=K, bitwidth=bw,
            group_size=gs, verify_sha=verify_sha)
    # an embed meta MAY carry a rotation record (the TIED head case
    # serves the lm_head's rotated set to the embedding consumer); the
    # untied head-pass embed is produced UNROTATED and carries none.
    rot = tensor_meta.get("rotation")
    return PalettizedEmbedding(indices, lut_t, bw, gs, N, K, indices2=indices2, lut2=lut2_t,
        bitwidth2=bw2, resA=resA, resB=resB,
        rotation_seed=rot["seed"] if rot else None,
        rotation_k=rot["k"] if rot else None)


def _tied_member_rows(meta: Dict):
    """(member, file, expected sha256) rows for every artifact member a
    consumer's metadata describes — the base idx/lut pair, every
    declared streams member (blob digest AND paired LUT digest), and
    the residual factor pair. The T-H7 comparison table: two consumers
    of one tied artifact set must produce EQUAL rows."""
    rows = [("index_file", meta.get("index_file"), meta.get("sha256_idx")),
            ("lut_file", meta.get("lut_file"), meta.get("sha256_lut"))]
    for i, entry in enumerate(meta.get("streams") or []):
        rows.append((f"streams[{i}].idxN", entry.get("file"),
                     entry.get("sha256")))
        rows.append((f"streams[{i}].lut_scalar", entry.get("file"),
                     entry.get("sha256_lut")))
    res = meta.get("residual")
    if res:
        rows.append(("resA_file", res.get("resA_file"),
                     res.get("sha256_resA")))
        rows.append(("resB_file", res.get("resB_file"),
                     res.get("sha256_resB")))
    return rows


def load_tied_pair(lm_head_meta: Dict, embed_meta: Dict, artifacts_dir: str,
                   residual: bool = True, reference: bool = False,
                   verify_sha: bool = True,
                   awq_scales: Optional[Dict] = None):
    """One artifact set, TWO consumers — the T-H7 tied rule.

    Returns (PalettizedLinear, PalettizedEmbedding) whose buffers are
    the SAME tensors (data_ptr equality on indices, lut, indices2,
    lut2, resA, resB): the artifact is read ONCE through the lm_head
    consumer's metadata (the writer's name; every member's sha256 is
    verified against the file), and the embedding consumer is
    constructed from the same tensors — one read serves both.

    T-H7, enforced BEFORE any file is read: the two consumers' metadata
    must describe the SAME member set (same files, same expected
    sha256s, same streams and residual declarations). A mismatch — one
    consumer expects a different sha or file, or declares a member the
    other does not — is a HARD ValueError naming BOTH consumers.
    lm_head_meta must carry the writer's "tied": true record. Note:
    make_trainable/freeze_lut replace a consumer's LUT tensor by
    design (training is per-consumer) — the sharing contract covers
    the as-loaded serving state.

    `embed_meta=None` is accepted for the tied artifacts of the
    head pass (ONE set under the lm_head name — no embed entry exists;
    the member rows are trivially the lm_head's own). The embedding
    consumer inherits the lm_head's ROTATION record: the shared
    artifact was folded (rotate=True on the lm_head step), so the
    gather forward un-rotates rows through the adjoint FHT — the
    older consumer returned FOLD-SPACE embedding rows (scrambled
    input embeddings for every token). AWQ never rides a head fold
    (the head pass runs without awq_scales), so no compensation here.
    """
    lm_var = str(lm_head_meta.get("var", "?"))
    if embed_meta is None:
        embed_meta = lm_head_meta
        emb_var = f"{lm_var}(tied)"
    else:
        emb_var = str(embed_meta.get("var", "?"))
    if lm_head_meta.get("tied") is not True:
        raise ValueError(f"load_tied_pair({lm_var}, {emb_var}): the lm_head metadata "
            f"does not carry the writer's \"tied\": true record — this is "
            f"not a one-artifact-set tied entry; load each consumer "
            f"separately (load_palettized_weight / load_palettized_embedding)")
    if "components" in lm_head_meta:
        raise ValueError(f"load_tied_pair({lm_var}, {emb_var}): a tied head is a single "
            f"matrix, never a QKV component split")
    rows_l = _tied_member_rows(lm_head_meta)
    rows_e = _tied_member_rows(embed_meta)
    if rows_l != rows_e:
        detail = "the member sets differ"
        for i in range(max(len(rows_l), len(rows_e))):
            a = rows_l[i] if i < len(rows_l) else None
            b = rows_e[i] if i < len(rows_e) else None
            if a != b:
                detail = (f"member {a[0] if a else b[0]}: lm_head expects "
                          f"{a[1:] if a else None}, embed expects "
                          f"{b[1:] if b else None}")
                break
        raise ValueError(f"load_tied_pair({lm_var}, {emb_var}): the two consumers' "
            f"metadata describe different artifact sets (T-H7) — {detail}; "
            f"refusing to serve one artifact set to disagreeing consumers")
    lm = load_palettized_weight(lm_head_meta, artifacts_dir,
                                residual=residual, reference=reference,
                                verify_sha=verify_sha,
                                awq_scales=awq_scales)
    rot = lm_head_meta.get("rotation")
    emb = PalettizedEmbedding(lm.indices, lm.lut, lm.bitwidth, lm.group_size, lm.N, lm.K,
        indices2=lm.indices2, lut2=lm.lut2, resA=lm.resA, resB=lm.resB,
        rotation_seed=rot["seed"] if rot else None,
        rotation_k=rot["k"] if rot else None)
    return lm, emb


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    raise AttributeError("Cannot find decoder layers in model structure")


def resolve_module(model, tensor_name: str):
    """Resolve the module that holds the Linear for a metadata tensor name.

    Layer tensors (`model.layers.<i>.<path>.weight`) resolve inside the
    layer list. the head-pass names `model.lm_head.weight` and
    `model.embed_tokens.weight` also resolve — the older code died on
    int("weight") and head artifacts could NEVER be swapped in.
    lm_head lives on the CausalLM wrapper (model.lm_head), embed_tokens
    on the text model (model.model.embed_tokens — the palettizer's
    _LM_HEAD_PATHS/_EMBED_PATHS convention); an ambiguous resolution
    (two different modules) is refused loudly, never guessed.
    Shared with replace_linear_with_palettized so trainers and the swap
    agree on paths.
    """
    parts = tensor_name.split(".")
    if parts[0] == "model" and len(parts) == 3 and parts[2] == "weight" \
            and parts[1] in ("lm_head", "embed_tokens"):
        name = parts[1]
        wrapper = model
        text_model = getattr(model, "model", model)
        hits = []
        for base, base_label in ((wrapper, "model"),
                                 (text_model, "model.model")):
            if hasattr(base, name):
                hits.append((f"{base_label}.{name}", base, getattr(base, name)))
        if not hits:
            raise ValueError(f"resolve_module({tensor_name}): no head module named "
                f"{name!r} on the model or its text model — the head-pass "
                f"artifact cannot be swapped into this architecture")
        first = hits[0][2]
        for path, base, mod in hits[1:]:
            if mod is not first:
                raise ValueError(f"resolve_module({tensor_name}): ambiguous — "
                    f"{hits[0][0]} and {path} are different modules; "
                    f"refusing to guess the text head")
        holder = hits[0][1]
        return holder, name
    layer_idx = int(parts[2])
    module_path = ".".join(parts[3:-1])
    layer = get_layers(model)[layer_idx]
    parent = layer
    attrs = module_path.split(".")
    for attr in attrs[:-1]:
        parent = getattr(parent, attr)
    return parent, attrs[-1]


def replace_linear_with_palettized(model, metadata, artifacts_dir: str,
                                   residual: bool = False,
                                   reference: bool = False,
                                   verify_sha: bool = True,
                                   awq_scales: Optional[Dict] = None,
                                   heads_dir: Optional[str] = None):
    """Swap every palettized Linear/Embedding for its palettized module.

    NOTE: norm-gain edits are not applied here — they must land before the
    swap. Call apply_norm_gain_edits first, or use load_palettized_model,
    which owns the correct order (and the AWQ-scale recovery that this
    function's `awq_scales` argument consumes).

    head-pass entries swap too. `model.lm_head.weight` becomes a
    PalettizedLinear (rotated — the head pass folds the lm_head);
    `model.embed_tokens.weight` becomes a PalettizedEmbedding (UNLESS
    the lm_head entry carries "tied": true, in which case ONE artifact
    set serves both consumers through load_tied_pair and no separate
    embed entry exists). Previously both names crashed resolve_module
    (int("weight")) — head artifacts were unloadable.

    heads_dir: when set, loads embed_tokens and lm_head from this directory
    instead of artifacts_dir. The heads metadata is loaded from heads_dir
    and merged with the layer metadata.
    """
    # Merge heads metadata if heads_dir is specified
    if heads_dir is not None:
        heads_meta = load_metadata(heads_dir)
        # Remove head entries from layer metadata, then add from heads
        for head_key in ["model.lm_head.weight", "model.embed_tokens.weight"]:
            if head_key in metadata["tensors"]:
                del metadata["tensors"][head_key]
        for head_key, head_val in heads_meta["tensors"].items():
            if head_key in ["model.lm_head.weight", "model.embed_tokens.weight"]:
                metadata["tensors"][head_key] = head_val
        print(f"  Merged heads metadata from {heads_dir}", flush=True)

    # start each swap with a clean stale-group_size ledger (the
    # summary printed at the end counts THIS swap's mismatches only)
    _GS_MISMATCHES.clear()
    n_compensated = 0
    for tensor_name, tensor_meta in metadata["tensors"].items():
        # Determine which artifacts dir to use for this tensor
        is_head = tensor_name in ["model.lm_head.weight", "model.embed_tokens.weight"]
        use_dir = heads_dir if (heads_dir is not None and is_head) else artifacts_dir

        if tensor_name == "model.lm_head.weight" \
                and tensor_meta.get("tied") is True:
            lm, emb = load_tied_pair(tensor_meta, None, use_dir,
                                     residual=residual, reference=reference,
                                     verify_sha=verify_sha,
                                     awq_scales=awq_scales)
            lm_holder, lm_leaf = resolve_module(model, tensor_name)
            setattr(lm_holder, lm_leaf, lm)
            emb_holder, emb_leaf = resolve_module(model, "model.embed_tokens.weight")
            setattr(emb_holder, emb_leaf, emb)
            if getattr(lm, "fold_order", None) == "rotate_then_awq":
                n_compensated += 1
            continue
        parent, leaf = resolve_module(model, tensor_name)
        old_module = getattr(parent, leaf)
        if tensor_name == "model.embed_tokens.weight":
            new_module = load_palettized_embedding(tensor_meta, use_dir, residual=residual,
                verify_sha=verify_sha)
        else:
            bias = getattr(old_module, "bias", None)
            new_module = load_palettized_weight(tensor_meta, use_dir,
                bias=bias.data if bias is not None else None,
                residual=residual, reference=reference,
                verify_sha=verify_sha, awq_scales=awq_scales)
            if isinstance(new_module, SplitQKV):
                # Q/K/V share one fused fold; count the group once per
                # component that carries the compensated order
                n_compensated += sum(
                    1 for m in (new_module.q_proj, new_module.k_proj,
                                new_module.v_proj)
                    if getattr(m, "fold_order", None) == "rotate_then_awq")
            elif getattr(new_module, "fold_order", None) == \
                    "rotate_then_awq":
                n_compensated += 1
        setattr(parent, leaf, new_module)
        del old_module
    if n_compensated:
        print(f"  [ROT] legacy rotate-then-AWQ fold order compensated on "
              f"{n_compensated} tensor group(s) — exact loader-side fix "
              f"(M = D T D^-1; main project: scripts/diagnose_greedy_bug.py claim 4)",
              flush=True)
    # the stale-group_size aggregate (see _note_gs_mismatch — the
    # per-tensor lines are now first-only + count)
    _summarize_gs_mismatches()
    # the MLP decode fusion post-pass — every dense Qwen3_5MLP with
    # palettized gate/up twins becomes a FusedPalettizedMLP (the M == 1
    # route fuses gate+up+silu*mul into ONE GEMV; every other shape and
    # an un-fusable pair keeps the original chain verbatim).
    # FLUTE_NO_MERGE=1 skips the wrap (the box A/B switch).
    n_mlp = _install_fused_mlp(model)
    if n_mlp:
        print(f"  [merge] FusedPalettizedMLP installed on {n_mlp} MLP "
              f"group(s) — the M == 1 decode route merges gate+up+SiLU*mul "
              f"into one split-K GEMV launch (FLUTE_NO_MERGE=1 disables)",
              flush=True)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def apply_norm_gain_edits(model, artifacts_dir: str,
                          verify_sha: bool = True,
                          strict: bool = True) -> Dict:
    """Apply AWQ norm-gain edits (norm_gain_edits.json + norm_edits/*.npy).

    Artifact contract (main project: palettize_qwen3_5_9b.py::_write_norm_gain_edit): each
    .npy stores the complete edited RMSNorm gain parameter; it is copied
    into the named parameter (cast to that parameter's dtype/device) after
    from_pretrained and before swapping palettized modules. Idempotent —
    re-application writes the same values. Returns the applied entries.

    Correctness note (zero-centered norms): Qwen3_5RMSNorm computes
    gain = 1 + w, and the producer folds the AWQ inverse scale exactly:
    w' = (1 + w)/s - 1 (the .npy stores the COMPLETE edited parameter,
    provenance note included). The legacy w/s form (artifacts from
    pre-fix producers) would deviate by (1 - 1/s) per channel — the
    recovery helper below refuses such sets loudly instead of serving
    approximate gains. With AWQ off there is nothing to apply and the
    model is exact.
    """
    path = os.path.join(artifacts_dir, "norm_gain_edits.json")
    if not os.path.exists(path):
        return {}
    import json
    with open(path) as f:
        doc = json.load(f)
    edits = doc.get("edits", {})
    named = dict(model.named_parameters())
    applied = {}
    for pname, entry in edits.items():
        if pname not in named:
            msg = (f"apply_norm_gain_edits: parameter {pname!r} not found "
                   f"in model")
            if strict:
                raise KeyError(msg)
            print(f"  WARNING: {msg}; skipping", flush=True)
            continue
        edit_path = os.path.join(artifacts_dir, entry["file"])
        _check_sha(edit_path, entry.get("sha256"), f"norm_edit:{pname}",
                   verify=verify_sha)
        arr = np.load(edit_path)
        param = named[pname]
        if tuple(arr.shape) != tuple(param.shape):
            raise ValueError(f"norm_edit:{pname}: shape {arr.shape} != parameter shape "
                f"{tuple(param.shape)}")
        param.data.copy_(torch.from_numpy(arr).to(param.dtype))
        applied[pname] = entry
    return applied


def iter_palettized_linears(root) -> Iterator[Tuple[str, "PalettizedLinear"]]:
    """Yield (dotted_name, PalettizedLinear) for every palettized module."""
    for name, mod in root.named_modules():
        if isinstance(mod, PalettizedLinear):
            yield name, mod


def iter_palettized_top(root):
    """Yield (dotted_name, parent, attr_name, child) for every palettized
    top-level module. SplitQKV is yielded as a whole; PalettizedLinear as-is.
    PalettizedLinear children inside a SplitQKV are not yielded separately
    (the parent SplitQKV is yielded instead)."""
    seen = set()
    # First collect all SplitQKV instances to know which children to skip
    split_qkv_children = set()
    for parent_name, parent in root.named_modules():
        for attr_name, child in list(parent._modules.items()):
            if isinstance(child, SplitQKV):
                for sub_name, sub_mod in child.named_modules():
                    if sub_mod is not child:
                        split_qkv_children.add(id(sub_mod))
    
    for parent_name, parent in root.named_modules():
        for attr_name, child in list(parent._modules.items()):
            if isinstance(child, (PalettizedLinear, SplitQKV)):
                if isinstance(child, PalettizedLinear) and id(child) in split_qkv_children:
                    continue
                dotted = f"{parent_name}.{attr_name}" if parent_name else attr_name
                mod_id = id(child)
                if mod_id in seen:
                    continue
                seen.add(mod_id)
                yield dotted, parent, attr_name, child


def load_metadata(artifacts_dir: str) -> Dict:
    import json
    with open(os.path.join(artifacts_dir, "metadata.json")) as f:
        return json.load(f)


def _read_norm_gain_doc(artifacts_dir: str) -> Optional[Dict]:
    """The raw norm_gain_edits.json document, or None when absent."""
    import json
    path = os.path.join(artifacts_dir, "norm_gain_edits.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _recover_awq_scales(model, artifacts_dir: str, metadata: Dict,
                        norm_doc: Optional[Dict],
                        verify_sha: bool = True) -> Dict:
    """Recover the per-channel AWQ scales from norm_gain_edits.json —
    the compensation input for legacy rotate-then-AWQ artifacts.

    THE PROBLEM THIS SOLVES: the 2026-10-04 box run folded the
    rotation BEFORE the AWQ scale (W' = W @ T @ diag(s)). The deployed
    pipeline (norm-gain folds diag(s)^-1, the module rotates by T)
    then computes x @ (D^-1 T D T^T) @ W^T — an O(1) input scrambling
    on every alpha>0 group (in_proj_qkv / in_proj_z / gate_proj /
    up_proj) that the per-layer cosine gates are blind to (both gate
    operands carry the same scrambler). Proof + numbers:
    main project: scripts/diagnose_greedy_bug.py.

    THE RECOVERY: the producer folded the inverse scale into the
    zero-centered RMSNorm gain exactly — w' = (1 + w)/s - 1 — and the
    .npy stores the COMPLETE edited parameter. Capturing w from the
    pristine from_pretrained model BEFORE apply_norm_gain_edits touches
    it gives s = (1 + w_orig) / (1 + w_edited) per channel.

     (trained norms): once a joint-trainer export replaces the edited
    gain with the TRAINED gain, the diff (1 + w_orig)/(1 + w_trained)
    no longer equals the compensation the training actually used — the
    deployed student would silently diverge from the trained one (the
    norm delta is exactly canceled by the renormalized product). The
    export therefore PINS the frozen s alongside the trained gain
    (entry["awq_scale_file"], sha-pinned); when that record exists it is
    PREFERRED over the diff inference, and the shape/range/meta
    cross-checks below still apply to it.

    Validation (loud, never silent):
      * s finite and 0 < s <= 1 + 2e-3 (production s = (m/max)^alpha
        is in (0, 1]);
      * the (1 + w_edited) >= 0 (zero-centered gains are non-negative
        on real models) — diff path only;
      * cross-check against the tensor meta's own "awq" stats
        (s_min/s_max/s_mean) within 2% — a mismatch means the
        norm edits and the tensor metadata disagree (corrupt or
        mixed artifact set) and the run stops.

    Returns {consumer_tensor_name: (K,) fp32 tensor} for every edit
    entry with alpha > 0; empty dict when there are no edits or no
    positive-alpha groups (the run was AWQ-free — nothing to
    compensate). Callers pass the map to replace_linear_with_palettized
    -> load_palettized_weight -> PalettizedLinear(awq_scale=...).
    """
    if not norm_doc:
        return {}
    edits = norm_doc.get("edits", {}) or {}
    if not edits:
        return {}
    named = dict(model.named_parameters())
    tensors_meta = metadata.get("tensors", {}) or {}
    out: Dict = {}
    n_groups = 0
    for pname, entry in edits.items():
        try:
            alpha = float(entry.get("alpha", 0.0) or 0.0)
        except (TypeError, ValueError):
            alpha = 0.0
        if alpha <= 0.0:
            continue
        if pname not in named:
            raise KeyError(f"_recover_awq_scales: parameter {pname!r} (norm edit, "
                f"alpha={alpha}) not found in the loaded model — the "
                f"artifacts and the model disagree")
        w0 = named[pname].detach().float().cpu()
        rec_file = entry.get("awq_scale_file")
        if rec_file:
            # the export pinned the FROZEN compensation — prefer it
            # over the diff inference (which would read the TRAINED gain
            # and silently move the compensation the training used).
            spath = os.path.join(artifacts_dir, str(rec_file))
            _check_sha(spath, entry.get("awq_scale_sha256"),
                       f"awq_recover:{pname}", verify=verify_sha)
            try:
                s = torch.from_numpy(np.ascontiguousarray(np.load(spath))).float().reshape(-1)
            except Exception as e:
                raise ValueError(f"awq_recover:{pname}: cannot read the recorded AWQ "
                    f"scale {rec_file!r} ({e}) — the pinned compensation "
                    f"file is corrupt; refusing") from e
            if tuple(s.shape) != tuple(w0.reshape(-1).shape):
                raise ValueError(f"awq_recover:{pname}: recorded AWQ scale shape "
                    f"{tuple(s.shape)} != the norm parameter's "
                    f"{tuple(w0.reshape(-1).shape)} — the pinned "
                    f"compensation and the norm edit disagree")
        else:
            edit_path = os.path.join(artifacts_dir, entry["file"])
            _check_sha(edit_path, entry.get("sha256"), f"awq_recover:{pname}",
                       verify=verify_sha)
            gamma = torch.from_numpy(np.ascontiguousarray(np.load(edit_path))).float()
            if tuple(gamma.shape) != tuple(w0.shape):
                raise ValueError(f"awq_recover:{pname}: edited-gain shape "
                    f"{tuple(gamma.shape)} != the model's {tuple(w0.shape)}")
            gain_new = 1.0 + gamma
            if bool((gain_new < 0).any()):
                raise ValueError(f"awq_recover:{pname}: 1 + w_edited has negative entries — "
                    f"the edit is not a zero-centered RMSNorm gain fold "
                    f"(legacy w/s producer?); refusing the compensation")
            gain_new = gain_new.clamp(min=1e-6)
            s = (1.0 + w0) / gain_new
        if not torch.isfinite(s).all() or bool((s <= 0).any()) \
                or float(s.max()) > 1.0 + 2e-3:
            raise ValueError(f"awq_recover:{pname}: recovered AWQ scales out of the "
                f"production range (0, 1] — min {float(s.min()):.4g}, "
                f"max {float(s.max()):.4g}; the norm edit and the model "
                f"checkpoint disagree")
        n_groups += 1
        for consumer in entry.get("consumers", []) or []:
            cname = str(consumer)
            # cross-check vs the tensor's own recorded stats
            cmeta = tensors_meta.get(cname) or {}
            awq = cmeta.get("awq") or {}
            if awq.get("applied"):
                for key, val in (("s_min", float(s.min())),
                                 ("s_max", float(s.max())),
                                 ("s_mean", float(s.mean()))):
                    rec = awq.get(key)
                    if rec is None:
                        continue
                    if abs(val - float(rec)) > 0.02 * max(1.0, abs(float(rec))):
                        raise ValueError(f"awq_recover:{cname}: recovered {key}={val:.4g} "
                            f"!= the metadata's {float(rec):.4g} (>2%) — "
                            f"the norm edits and the tensor metadata "
                            f"describe different scales; refusing")
            out.setdefault(cname, s)
    if out:
        print(f"  [ROT] recovered AWQ scales for {n_groups} norm group(s) / "
              f"{len(out)} consumer tensor(s) from norm_gain_edits.json",
              flush=True)
    return out


def load_palettized_model(artifacts_dir: str, model_name: str,
                          device: str = "cuda:0",
                          dtype=torch.float16, residual: bool = False,
                          reference: bool = False,
                          apply_norm_edits: bool = True,
                          awq_compensation: bool = True,
                          heads_dir: Optional[str] = None):
    """Load the base model and swap in the idx4 artifacts.

    Shared by the evaluators (main project: eval_greedy_match, eval_ppl)
    and reusable by
    the capture/energy harnesses.

    Applies norm_gain_edits.json (when present) between from_pretrained and
    the module swap — the order the artifact contract mandates. Pass
    apply_norm_edits=False if you applied the edits yourself (application
    is idempotent, so double application is also safe).

    while the model is still PRISTINE (pre-edit), the AWQ scales of
    every alpha>0 norm group are recovered from the edits
    (_recover_awq_scales: s = (1+w_orig)/(1+w_edited)) and handed to the
    module swap, which compensates the legacy rotate-then-AWQ fold order
    EXACTLY (input rotation M = D T D^-1). This is what makes the
    2026-10-04 box artifacts (0.999 gates, garbage greedy decode) serve
    correctly without re-palettizing. Pass awq_compensation=False only
    for differential debugging (the deployed math then reproduces the
    broken composition deliberately).

    heads_dir: when set, loads embed_tokens and lm_head from this directory
    instead of artifacts_dir. Useful for combining a separate head-pass
    (higher quality LUT with compensation streams + residual) with the
    layer artifacts from the main pass.
    """
    from transformers import AutoModelForCausalLM
    metadata = load_metadata(artifacts_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True)
    awq_scales: Dict = {}
    if apply_norm_edits:
        norm_doc = _read_norm_gain_doc(artifacts_dir)
        if norm_doc and awq_compensation:
            # PRISTINE-model recovery must run BEFORE the edits land
            awq_scales = _recover_awq_scales(model, artifacts_dir, metadata, norm_doc)
        applied = apply_norm_gain_edits(model, artifacts_dir)
        if applied:
            print(f"  Applied {len(applied)} AWQ norm-gain edits "
                  f"(norm_gain_edits.json)", flush=True)
    replace_linear_with_palettized(model, metadata, artifacts_dir,
                                   residual=residual, reference=reference,
                                   awq_scales=awq_scales,
                                   heads_dir=heads_dir)
    model = model.to(device)
    model.eval()
    return model, metadata
