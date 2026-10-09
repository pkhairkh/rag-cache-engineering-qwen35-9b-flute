#!/usr/bin/env python3
"""
qlora.py — QLoRA for the FLUTE-palettized Qwen3.5-9B stack.

Forward: Y = frozen_branch(x) + (alpha/r) * (dropout(x) @ A^T) @ B^T
  - frozen branch: THREE-WAY policy resolved ONCE at attach:
    fused-flute (FLUTE kernel, W never in DRAM) | torch-gpu-cached
    (explicit FLUTE_FROZEN_PATH=torch opt-out: W16 materialized once +
    cuBLAS) | reference-cpu (CPU only; forbidden on CUDA)
  - LoRA branch: standard torch.matmul (autograd-aware, always builds a graph)
"""

from __future__ import annotations
import json, math, os, re, sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import palettized_modules as pmod
from palettized_modules import PalettizedLinear, SplitQKV


# Adapter init contract: parameters are born from torch.zeros (never
# torch.empty) and an unknown init string is a hard, named error —
# uninitialized memory in lora_A is amplified by the first optimizer step
# (dL/dB = scale * (dL/dy) @ (x @ A^T)^T), so it must never exist.
_INIT_A_MODES = ("kaiming_uniform", "normal")
_INIT_B_MODES = ("zero", "zeros")


def _validate_init_modes(init_a: str, init_b: str, ctx: str) -> None:
    """Loud contract check for the adapter init strings. Fails fast at
    attach entry, before any module is wrapped, so lora_A never holds
    uninitialized memory."""
    if init_a not in _INIT_A_MODES:
        raise ValueError(
            f"{ctx}: unknown init_a {init_a!r} (expected one of "
            f"{_INIT_A_MODES}); lora_A is born from torch.zeros, and an "
            f"unknown name is refused before allocation.")
    if init_b not in _INIT_B_MODES:
        raise ValueError(
            f"{ctx}: unknown init_b {init_b!r} (expected one of "
            f"{_INIT_B_MODES}; B is always zero-initialized — the standard "
            f"LoRA contract that keeps the adapter contribution exactly "
            f"zero at step 0).")


@dataclass
class QLoRAConfig:
    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    scope: str = "all"
    include_residual_branch: bool = True
    init_a: str = "kaiming_uniform"
    init_b: str = "zero"
    base_model: str = "Qwen/Qwen3.5-9B"
    artifacts_dir: str = ""
    # Per-module rank map: {dotted module path: r}; a SplitQKV is
    # addressed per component ("<path>.q", ".k", ".v");
    # r = 0 leaves that module unwrapped. None = uniform-r attach for
    # every module (from_json also defaults it for configs that simply
    # lack the key).
    rank_map: Optional[Dict[str, int]] = None
    # "proportional": alpha_i = r_i // 4 (kappa = 0.25 — every wrapped
    # module trains at scale 0.25). "global": fixed alpha (scale then
    # varies per module). Only consulted when rank_map is set.
    alpha_mode: str = "proportional"
    tensors: Dict[str, Dict] = None

    def to_json(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f: json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f: d = json.load(f)
        # Keys absent from the JSON (old configs predate rank_map and
        # alpha_mode) fall back to the dataclass defaults — a plain
        # d.get(k) would force None over alpha_mode's "proportional".
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# QKV component order shared by the wrapper, the rank-map resolution and
# the per-component geometry stored in QLoRAConfig.tensors.
_QKV_COMPONENTS = ("q", "k", "v")
_ALPHA_MODES = ("proportional", "global")


def _check_rank(key, value):
    """Validate one rank_map value: a non-negative integer, never
    silently coerced (a float 8.5 or a string "8" is an error, not r=8)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"attach_qlora: rank_map[{key!r}] must be an "
                         f"integer rank, got {value!r}")
    rank = int(value)
    if rank < 0:
        raise ValueError(f"attach_qlora: rank_map[{key!r}] must be >= 0, "
                         f"got {rank}")
    return rank


def _resolve_alpha(key, rank, alpha, alpha_mode):
    """alpha_i for one wrapped module (or QKV component) of rank `rank`.

    proportional: alpha_i = r_i // 4 — every rank must be a positive
       multiple of 4 so alpha stays a positive integer (loud ValueError
       naming the module otherwise).
    global: the legacy fixed alpha (scale then varies per module)."""
    if alpha_mode == "proportional":
        if rank <= 0 or rank % 4 != 0:
            raise ValueError(
                f"attach_qlora: alpha_mode='proportional' requires every "
                f"wrapped rank to be a positive multiple of 4 (alpha = "
                f"r // 4 must be a positive integer); {key!r} has r={rank}")
        return rank // 4
    return alpha


def _qkv_component_values(field, value):
    """Normalize a QLoRASplitQKV r/alpha argument (int or per-component
    dict {"q": .., "k": .., "v": ..}) to a full per-component dict.
    A dict missing components or carrying unknown keys is an error —
    never a silent default."""
    if not isinstance(value, dict):
        return {c: value for c in _QKV_COMPONENTS}
    missing = [c for c in _QKV_COMPONENTS if c not in value]
    unknown = [c for c in value if c not in _QKV_COMPONENTS]
    if missing or unknown:
        raise ValueError(
            f"QLoRASplitQKV: per-component {field} dict must have exactly "
            f"the keys q/k/v (missing: {missing}, unknown: {unknown})")
    out = {}
    for c in _QKV_COMPONENTS:
        v = value[c]
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or (isinstance(v, float) and not v.is_integer()):
            raise ValueError(f"QLoRASplitQKV: per-component {field}[{c!r}] "
                             f"must be an integer, got {v!r}")
        out[c] = int(v)
        if out[c] < 0:
            raise ValueError(f"QLoRASplitQKV: per-component {field}[{c!r}] "
                             f"must be >= 0, got {out[c]}")
    return out


# ---------------------------------------------------------------------------
# The three-way frozen-branch policy, resolved ONCE at attach time —
# never per forward, never silently.
#
#   fused-flute       FLUTE kernel forward + fused backward kernel — the
#                     DEFAULT on CUDA; an unavailable kernel/eligibility is
#                     a hard attach-time error, not a fallback
#   torch-gpu-cached  the EXPLICIT opt-out (FLUTE_FROZEN_PATH=torch): W16
#                     materialized once per module through qlora_fallback's
#                     WeightCache, cuBLAS matmul, autograd backward
#   reference-cpu     CPU only (always fine); requesting it on CUDA is a
#                     hard error
#
# The only escape hatches are FLUTE_FROZEN_PATH (this policy, forward) and
# FLUTE_FUSED_BWD (the fused backward, honored inside qlora_gemm.py).
# ---------------------------------------------------------------------------

_FROZEN_PATHS = ("fused-flute", "torch-gpu-cached", "reference-cpu")
_LEGAL_FROZEN_ENV = ("fused", "torch", "reference")

# One-shot guard for the torch-gpu-cached opt-out banner (rule d): the
# resolver runs once per attach (per layer job in the engine), so the line
# prints once per process per module name — never per forward.
_TORCH_PATH_REPORTED = set()


def _fused_backward_ok() -> bool:
    """Fused-backward availability for the error messages only (guarded
    import: a broken/missing qlora_gemm must not mask the error it is
    about to report). Honors the FLUTE_FUSED_BWD escape hatch exactly like
    qlora_gemm.fused_backward_available()."""
    try:
        import qlora_gemm
        return bool(qlora_gemm.fused_backward_available())
    except Exception:
        return False


def _resolve_frozen_path(on_cuda, flute_ok, eligible, env, ctx) -> str:
    """Pure three-way frozen-branch resolver.

    Returns one of "fused-flute" | "torch-gpu-cached" | "reference-cpu".
    A PURE function of its arguments — no device probing, no os.environ
    reads (env is injected), no prints; the loud reporting lives in the
    wrapper (QLoRALinear.__init__ / _report_torch_optout_once).

    Rules:
      (a) not on_cuda -> "reference-cpu" (always fine — CPU runs/tests;
          the CPU path stays silent and identical);
      (b) on_cuda: FLUTE_FROZEN_PATH (read from `env`) in {"fused"
          (default, also for unset/empty), "torch", "reference"}; any
          other value is a loud ValueError listing the legal values;
      (c) on_cuda + fused + (not flute_ok or not eligible) -> hard
          RuntimeError naming the offending module (`ctx`) and the build
          commands — never a silent fallback;
      (d) on_cuda + torch -> "torch-gpu-cached" (allowed; the wrapper
          prints ONE loud opt-out line at resolution);
      (e) on_cuda + reference -> hard RuntimeError: reference-cpu is
          forbidden on CUDA runs.
    """
    if not on_cuda:
        return "reference-cpu"
    raw = env.get("FLUTE_FROZEN_PATH")
    choice = raw if raw not in (None, "") else "fused"
    if choice not in _LEGAL_FROZEN_ENV:
        raise ValueError(
            f"[qlora] FLUTE_FROZEN_PATH={raw!r} is not a legal frozen path "
            f"(module {ctx!r}); legal values: "
            f"fused (default) | torch | reference")
    if choice == "reference":
        raise RuntimeError(
            f"[qlora] frozen path=reference-cpu is forbidden on a CUDA run "
            f"(FLUTE_FROZEN_PATH=reference was set explicitly for module "
            f"{ctx!r}). T10 always-fused: reference numerics are CPU-only "
            f"(the eval harnesses use eval_reference()); take the default "
            f"fused-flute, or the explicit FLUTE_FROZEN_PATH=torch opt-out.")
    if choice == "torch":
        return "torch-gpu-cached"
    # choice == "fused" (the default): the always-fused mandate. An
    # unavailable FLUTE forward kernel or an ineligible module geometry
    # is a hard attach-time error naming the remediation — never a
    # fallback.
    if (not flute_ok) or (not eligible):
        bwd = _fused_backward_ok()
        bwd_note = "" if bwd else (
            "The fused backward kernel is also unavailable — build it too:\n"
            "      cd flute_train_kernels && python setup.py build_ext "
            "--inplace\n")
        why = []
        if not flute_ok:
            reason = ""
            try:
                import qlora_gemm as _qg
                reason = _qg.flute_import_error() or ""
            except Exception:
                pass
            why.append("flute_extended.qgemm_per_group_lut is not "
                       "importable/available"
                       + (f" ({reason})" if reason else ""))
        if not eligible:
            why.append("fused_gemm_eligible(base) is False (geometry "
                       "N%128==0/K%64==0, frozen fp16 LUT, CUDA-resident "
                       "indices/lut)")
        raise RuntimeError(
            f"[qlora] frozen path=fused-flute unavailable for module "
            f"{ctx!r} on a CUDA run: {'; '.join(why)}. T10 always-fused: "
            f"never a silent fallback — build the FLUTE forward kernel:\n"
            f"      cd flute_extended && python setup.py build_ext "
            f"--inplace\n"
            f"{bwd_note}"
            f"or take the EXPLICIT opt-out (loud, recorded): "
            f"FLUTE_FROZEN_PATH=torch (torch-gpu-cached: W16 materialized "
            f"once via qlora_fallback + cuBLAS matmul).")
    return "fused-flute"


def _report_torch_optout_once(name: str) -> None:
    """ONE loud line per process per module name for the explicit
    torch-gpu-cached opt-out (rule d). Called by the wrapper at resolution
    time so the resolver itself stays pure; the module-level guard makes
    the line once-per-process-per-module, not per forward (and not even
    per re-attach within the same process). The line carries the verbatim
    reason the fused path is unavailable (never a mystery) and
    the structural note that this opt-out bypasses the fused backward
    kernel — gradients flow through autograd's matmul on the cached W16."""
    if name in _TORCH_PATH_REPORTED:
        return
    _TORCH_PATH_REPORTED.add(name)
    why = ""
    try:
        import qlora_gemm
        ferr = qlora_gemm.flute_import_error()
        berr = qlora_gemm.backward_import_error()
        bits = []
        if ferr:
            bits.append(f"forward kernel: {ferr}")
        if berr:
            bits.append(f"fused backward: {berr}")
        if bits:
            why = " [why the fused path is off: " + "; ".join(bits) + "]"
    except Exception:
        pass
    print("[qlora] frozen path=torch-gpu-cached (explicit "
          f"FLUTE_FROZEN_PATH=torch opt-out){why} — the fused backward "
          "kernel is BYPASSED on this path (autograd matmul on the cached "
          "W16); build both kernels and drop the env to run the "
          "deployment-faithful fused path", flush=True)


# ---------------------------------------------------------------------------
# The memory-lean LoRA branch (see _LoRABranchFn): the fp32 input cast is
# recomputed in backward instead of saved, so no (n, K) fp32 copy of the
# module input is kept alive between forward and backward. With ~200
# wrapped components at the training geometry this is the difference
# between ~3.3 GiB and ~6.7 GiB of saved-for-backward memory.
# ---------------------------------------------------------------------------

class _LoRABranchFn(torch.autograd.Function):
    """lora = (x.float() @ A^T) @ B^T — the QLoRA non-autocast branch with
    the fp32 x-cast recomputed in backward instead of saved.

    Saves: x (any dtype — typically the fp16 module input), A, B (the
    adapter parameters; non-fp32 adapters are cast once per call, small)
    and the (n, r) mid. Never keeps the (n, K) fp32 cast nor the (n, N)
    fp32 product alive past the op. Degenerate r=0 shapes (empty adapters
    inside mixed-rank SplitQKV) produce exact zeros, like the naive chain.
    """

    @staticmethod
    def forward(ctx, x, A, B):
        # A.t()/B.t() on fp32 parameters are views, so the GEMMs see
        # standard operand layouts.
        A32 = A if A.dtype == torch.float32 else A.float()
        B32 = B if B.dtype == torch.float32 else B.float()
        x32 = x.float() if x.dtype != torch.float32 else x
        mid = x32 @ A32.t()          # (n, r)
        out = mid @ B32.t()          # (n, N) fp32 — NOT saved
        ctx.save_for_backward(x, A, B, mid)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, A, B, mid = ctx.saved_tensors
        A32 = A if A.dtype == torch.float32 else A.float()
        B32 = B if B.dtype == torch.float32 else B.float()
        need_x = ctx.needs_input_grad[0]
        need_A = ctx.needs_input_grad[1]
        need_B = ctx.needs_input_grad[2]

        dmid = None
        if need_x or need_A:
            dmid = grad_out @ B32                     # (n, r)
        dB = None
        if need_B:
            dB = grad_out.t() @ mid                   # (N, r)
        dA = None
        if need_A:
            x32 = x.float() if x.dtype != torch.float32 else x
            dA = dmid.t() @ x32                       # (r, K)
            if A.dtype != torch.float32:
                dA = dA.to(A.dtype)
        dx = None
        if need_x:
            dx = (dmid @ A32)                         # (n, K) fp32
            dx = dx.to(x.dtype)
        return dx, dA, dB


class QLoRALinear(nn.Module):
    """Wraps a PalettizedLinear; adds a trainable LoRA branch.

    The frozen branch is routed by `self._frozen_path`, resolved ONCE in
    __init__ via `_resolve_frozen_path` — never re-resolved per forward,
    never a silent fallback."""

    def __init__(self, pal_linear, r, alpha, dropout=0.0,
                 init_a="kaiming_uniform", init_b="zero"):
        super().__init__()
        self.base = pal_linear
        # Kernel path on CUDA (packed blob, no materialized (N,K) tensors);
        # reference path on CPU. eval_reference()/eval_kernel() switch modes.
        lut_dev = self.base.lut.device if torch.is_tensor(self.base.lut) else None
        self.base.reference = (lut_dev is None or lut_dev.type != "cuda")

        K, N = pal_linear.K, pal_linear.N
        self.r = int(r)
        self.alpha = int(alpha)
        # r = 0 is the explicit no-adapter marker (rank-map tier 0): the
        # empty A/B below and scale 0.0 make the LoRA branch contribute
        # exactly zero, so forward is bit-identical to the frozen base.
        self.scale = float(alpha) / float(r) if self.r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA parameters follow the frozen branch's device.
        device = pal_linear.indices.device if hasattr(pal_linear, 'indices') and torch.is_tensor(pal_linear.indices) else None
        
        # lora_A: (r, K), lora_B: (N, r) — B zero-init so LoRA contributes
        # zero at step 0 (model is bit-identical to the un-LoRA'd base).
        # r = 0 gives (0, K)/(N, 0) empties: keys still round-trip through
        # save/load, but the branch is identically zero. Both parameters
        # start from torch.zeros (never torch.empty; see
        # _validate_init_modes). B is always zeros.
        _validate_init_modes(init_a, init_b,
                             f"QLoRALinear(N={N}, K={K}, r={self.r})")
        self.lora_A = nn.Parameter(torch.zeros(self.r, K, device=device))
        self.lora_B = nn.Parameter(torch.zeros(N, self.r, device=device))
        if self.r > 0:
            if init_a == "kaiming_uniform":
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            elif init_a == "normal":
                nn.init.normal_(self.lora_A, std=1.0 / math.sqrt(self.r))
            if not bool(torch.isfinite(self.lora_A.detach()).all()):
                raise RuntimeError(
                    f"QLoRALinear(N={N}, K={K}, r={self.r}): lora_A is "
                    f"non-finite after init_a={init_a!r} — refusing to train "
                    f"on a broken initialization")

        # ---- resolve the frozen branch ONCE, at attach time -------------
        # The path is frozen for the module's lifetime (never re-checked
        # per forward): CPU -> reference-cpu silently; CUDA -> fused-flute
        # by default (a missing/ineligible kernel raises HERE), and
        # torch-gpu-cached only via the explicit FLUTE_FROZEN_PATH=torch
        # opt-out (announced once per process per module name).
        idx = getattr(self.base, "indices", None)
        idx_dev = idx.device if torch.is_tensor(idx) else None
        on_cuda = ((lut_dev is not None and lut_dev.type == "cuda")
                   or (idx_dev is not None and idx_dev.type == "cuda"))
        try:
            import qlora_gemm
            flute_ok = qlora_gemm._check_flute_kernel()
        except Exception:
            flute_ok = False
        try:
            eligible = bool(flute_ok
                            and qlora_gemm.fused_gemm_eligible(self.base))
        except Exception:
            eligible = False
        self._frozen_ctx = (f"PalettizedLinear(N={self.base.N}, "
                            f"K={self.base.K}, "
                            f"group_size={self.base.group_size}, "
                            f"bitwidth={self.base.bitwidth})")
        self._frozen_path = _resolve_frozen_path(
            on_cuda, flute_ok, eligible, os.environ, self._frozen_ctx)
        if self._frozen_path == "torch-gpu-cached":
            _report_torch_optout_once(self._frozen_ctx)
        # W16 for the torch-gpu-cached branch: materialized lazily on the
        # FIRST forward (through qlora_fallback.materialize_weight, whose
        # WeightCache dedupes by blob identity) and then kept on self —
        # a detached constant, so backward flows through autograd's
        # matmul (dL/dX = dL/dY @ W16) with no fused backward kernel.
        self._torch_cached_W = None
        # Cached fp32 casts of the FROZEN residual factors: casting two
        # weight-sized constants per forward would be pure overhead for
        # values that never change. Values are bit-identical to per-call
        # casts (same frozen fp16 source, single cast); _apply()
        # invalidates on any device/dtype move so a stale cache is
        # impossible.
        self._resB32 = None
        self._resA32 = None

    def _apply(self, fn):
        # any .to()/.float()/.cuda() move invalidates the fp32 residual
        # caches (they are derived constants of the base buffers)
        self._resB32 = None
        self._resA32 = None
        return super()._apply(fn)

    def _residual32(self):
        """(resB32 [K, r], resA32 [r, N]) — lazily-cast fp32 views of the
        frozen residual factors for the y += (x @ resB^T) @ resA^T branch.
        Bit-identical to per-call casts; cached because the factors are
        frozen for the module's lifetime."""
        if self._resB32 is None or self._resA32 is None:
            self._resB32 = self.base.resB.t().float()
            self._resA32 = self.base.resA.t().float()
        return self._resB32, self._resA32

    @torch.no_grad()
    def eval_kernel(self):
        """O-1/eval harness switch: force the BASE's kernel mode
        (base.reference=False). This flips only the base module's
        internal routing — it overrides the frozen path solely
        on the reference-cpu branch (the `self.base(x2)` forward); the
        fused-flute and torch-gpu-cached branches never consult
        base.reference."""
        self.base.reference = False
        return self

    @torch.no_grad()
    def eval_reference(self):
        """O-1/eval harness switch: force the BASE's reference mode
        (base.reference=True) — the reference-numics harnesses only (see
        eval_kernel's note: only the reference-cpu frozen branch consults
        it)."""
        self.base.reference = True
        return self

    def _use_fused(self):
        """Read-only status helper: True exactly when the module's frozen
        path is fused-flute. Not consulted by forward; the status
        reporters (data._report_status) are the consumers."""
        return self._frozen_path == "fused-flute"

    def forward(self, x):
        x_shape = x.shape
        if x.dim() == 3:
            b, s, k = x.shape
            x2 = x.reshape(b * s, k)
        else:
            x2 = x

        # ---- W14: the FOLD input ----------------------------------------
        # The base module's rotation (+ AWQ compensation) applied ONCE:
        # every CUDA branch below and the LoRA branch consume the
        # FOLD-SPACE input — the frame the base's artifacts, its resA/
        # resB residual factors, and the merge's fold-space W +
        # scale*(B@A) all live in. Pre-W14 the fused-flute and
        # torch-cached branches GEMMed the RAW x2 against the fold-space
        # codebook (garbage for every rotated artifact — the same class
        # as the pre-W13 reference-path bug), the residual branch mixed
        # frames, and the lora branch trained in a frame the merge
        # cannot reproduce. The reference branch keeps handing the RAW
        # x2 to self.base (PalettizedLinear.forward rotates internally
        # on BOTH of its paths) — only the lora branch needs x_fold
        # there, so the fold is computed once above and reused.
        if getattr(self.base, "rot_signs", None) is not None:
            x_fold = self.base._rotate_input(x2)
        else:
            x_fold = x2

        # ---- Frozen branch, routed by the path frozen at attach.
        # Never re-resolved per forward; never a silent fallback.
        if self._frozen_path == "fused-flute":
            # FLUTE kernel forward (W never in DRAM), fused backward for
            # dL/dX (escape hatch FLUTE_FUSED_BWD=0 lives in qlora_gemm).
            import qlora_gemm
            # W10: a two-stream base (refinement artifacts, palette > 16)
            # routes through the two-stream arms — the trainable case
            # through the W10 Function (dL/dLUT for BOTH masters), the
            # frozen case through the two-qgemm deployment shape. The
            # plain single-stream arms below are unchanged.
            has2 = bool(getattr(self.base, "has_stream2", False))
            if isinstance(self.base.lut, nn.Parameter) \
                    and self.base.lut.requires_grad:
                # W5 (docs/KERNEL_SPEC_DLDLUT.md §5): the trainable
                # codebook on the kernel path — the Function carries
                # dL/dLUT (lut_grad_scatter, or its closed-form
                # reference arm when the kernel is down). Selected by
                # the trainer's --lut-path kernel; the plain frozen
                # path below is unchanged.
                if has2:
                    y_q = qlora_gemm.fused_qlora_gemm_train_lut_two_streams(
                        x_fold, self.base.indices, self.base.lut,
                        self.base.bitwidth, self.base.indices2,
                        self.base.lut2, self.base.bitwidth2,
                        self.base.group_size, self.base.N, self.base.K)
                else:
                    y_q = qlora_gemm.fused_qlora_gemm_train_lut(
                        x_fold, self.base.indices, self.base.lut,
                        self.base.bitwidth, self.base.group_size,
                        self.base.N, self.base.K)
            elif has2:
                # W10 frozen two-stream: TWO qgemms + ONE ordered add
                # (stream 1 pinned first) + the summed fused backward —
                # the plain single-stream call would silently DROP y2
                # and train against a wrong forward.
                y_q = qlora_gemm.fused_qlora_gemm_two_streams(
                    x_fold, self.base.indices, self.base.lut,
                    self.base.bitwidth, self.base.indices2,
                    self.base.lut2, self.base.bitwidth2,
                    self.base.group_size, self.base.N, self.base.K)
            else:
                y_q = qlora_gemm.fused_qlora_gemm(
                    x_fold, self.base.indices, self.base.lut,
                    self.base.bitwidth, self.base.group_size,
                    self.base.N, self.base.K)
            if y_q is None:
                # Eligibility was pre-validated at attach; a None here
                # means the FLUTE kernel went away AFTER the path was
                # frozen — refuse rather than silently fall back.
                raise RuntimeError(
                    f"[qlora] frozen path=fused-flute: fused_qlora_gemm "
                    f"returned None for {self._frozen_ctx} although the "
                    f"path was resolved and pre-validated at attach time — "
                    f"the FLUTE forward kernel became unavailable after "
                    f"attach (T10: never a silent fallback). Rebuild: "
                    f"cd flute_extended && python setup.py build_ext "
                    f"--inplace; or re-attach under the explicit "
                    f"FLUTE_FROZEN_PATH=torch opt-out.")
            if self.base.bias is not None:
                y_q = y_q + self.base.bias
            if self.base.resA is not None and self.base.resB is not None:
                if torch.is_autocast_enabled():
                    y_q = y_q + (x_fold @ self.base.resB.t()) \
                        @ self.base.resA.t()
                else:
                    # Cached fp32 residual factors (see _residual32) —
                    # the factors are frozen constants, not per-step work.
                    # W14: the residual consumes the FOLD input (the
                    # ladder was fit on the fold-space weight).
                    rB32, rA32 = self._residual32()
                    y_q = y_q + ((x_fold.float() @ rB32)
                                 @ rA32).to(y_q.dtype)
        elif self._frozen_path == "torch-gpu-cached":
            # Explicit FLUTE_FROZEN_PATH=torch opt-out: W16 materialized
            # ONCE per module (first forward; the WeightCache dedupes by
            # blob identity), then y_q = x2 @ W16.t() — cuBLAS fp16 with
            # fp32 accumulation, the same numerics class as the FLUTE
            # kernel (fp16 operands, fp32 accumulate), + bias and the
            # resA/resB residual branch with the same fp16/fp32 handling
            # as the fused branch. Backward flows through autograd's
            # matmul naturally (W16 is a detached constant).
            W16 = self._torch_cached_W
            if W16 is None:
                import qlora_fallback
                with torch.no_grad():
                    W16 = qlora_fallback.materialize_weight(
                        self.base.indices, self.base.lut,
                        self.base.N, self.base.K,
                        self.base.group_size, torch.float16,
                        bitwidth=self.base.bitwidth)
                    # W10: a two-stream base materializes BOTH streams
                    # (the ordered-add values W1 + W2 — the single-GEMM
                    # form of the deployment shape); stream 2 carries
                    # its OWN idxN width
                    if getattr(self.base, "has_stream2", False):
                        W16 = W16 + qlora_fallback.materialize_weight(
                            self.base.indices2, self.base.lut2,
                            self.base.N, self.base.K,
                            self.base.group_size, torch.float16,
                            bitwidth=self.base.bitwidth2)
                    if W16.requires_grad:   # never true for frozen
                        W16 = W16.detach()  # indices/lut, but stay safe
                self._torch_cached_W = W16
            in_dtype = x_fold.dtype
            # fp16 operands like the FLUTE kernel (identical to a plain
            # x_fold @ W16.t() for the fp16 inputs of a CUDA run; the cast
            # pair only keeps fp32/bf16 inputs legal), output back in the
            # input's dtype exactly like FusedQLoRAGEMM.
            xh = x_fold if in_dtype == W16.dtype else x_fold.to(W16.dtype)
            y_q = xh @ W16.t()
            if in_dtype != y_q.dtype:
                y_q = y_q.to(in_dtype)
            if self.base.bias is not None:
                y_q = y_q + self.base.bias
            if self.base.resA is not None and self.base.resB is not None:
                if torch.is_autocast_enabled():
                    y_q = y_q + (x_fold @ self.base.resB.t()) \
                        @ self.base.resA.t()
                else:
                    # Cached fp32 residual factors (see _residual32).
                    # W14: the residual consumes the FOLD input.
                    rB32, rA32 = self._residual32()
                    y_q = y_q + ((x_fold.float() @ rB32)
                                 @ rA32).to(y_q.dtype)
        else:
            # reference-cpu — resolved only when the base is off-CUDA;
            # eval_kernel()/eval_reference() steer the base's internal
            # routing on exactly this branch.
            y_q = self.base(x2)

        # Trainable LoRA branch via standard matmul: it is always on the
        # autograd graph, so dL/dA and dL/dB depend only on dL/dY and x —
        # independent of the frozen branch's backward.
        #
        # W14 (frame): the branch consumes the FOLD input on EVERY path —
        # the merge folds scale*(B@A) into the FOLD-SPACE weight
        # (qlora_merge._materialize_weight), so an adapter trained on the
        # raw frame would deploy rotated (a silent train/deploy gap on
        # every rotated artifact). On the reference path the fold is a
        # second application of the base's own rotation (computed above;
        # self.base(x2) rotates internally) — cheap next to the GEMM.
        if torch.is_autocast_enabled():
            x_drop = self.dropout(x_fold)
            lora = (x_drop @ self.lora_A.t()) @ self.lora_B.t()
            y_lora = lora * self.scale
        else:
            # Memory-lean non-autocast branch (see _LoRABranchFn): fp32
            # math with the (n, K) input cast recomputed in backward —
            # never saved. Dropout applies on the input-dtype side of the
            # cast (identical at p=0; for p>0 the mask multiplies the
            # input-dtype x, matching the autocast branch's placement).
            x_drop = self.dropout(x_fold)
            lora = _LoRABranchFn.apply(x_drop, self.lora_A, self.lora_B)
            y_lora = lora.to(y_q.dtype) * self.scale

        y = y_q + y_lora
        if len(x_shape) == 3:
            y = y.view(x_shape[0], x_shape[1], -1)
        return y

    def trainable_params(self):
        return [("lora_A", self.lora_A), ("lora_B", self.lora_B)]


class QLoRASplitQKV(nn.Module):
    """Wraps a SplitQKV; q/k/v each get their own QLoRALinear.

    `r`/`alpha` are either ints (uniform over q/k/v — today's behavior)
    or per-component dicts {"q": rq, "k": rk, "v": rv} (rank-map mode;
    a component with r_c = 0 gets an empty, exactly-zero LoRA branch).

    The frozen path mirrors itself — q/k/v are QLoRALinear
    instances that each resolve `self._frozen_path` in their own
    __init__ (this wrapper adds nothing and breaks nothing); the
    eval_kernel()/eval_reference() switches propagate per component.
    """

    def __init__(self, split_qkv, r, alpha, dropout=0.0,
                 init_a="kaiming_uniform", init_b="zero"):
        super().__init__()
        r_c = _qkv_component_values("r", r)
        a_c = _qkv_component_values("alpha", alpha)
        self.q = QLoRALinear(split_qkv.q_proj, r_c["q"], a_c["q"],
                             dropout, init_a, init_b)
        self.k = QLoRALinear(split_qkv.k_proj, r_c["k"], a_c["k"],
                             dropout, init_a, init_b)
        self.v = QLoRALinear(split_qkv.v_proj, r_c["v"], a_c["v"],
                             dropout, init_a, init_b)
        if isinstance(r, dict) or isinstance(alpha, dict):
            # Per-component geometry: store the dicts and scale as the
            # per-component scales — never an invented scalar for mixed
            # ranks (a uniform-looking dict stays a dict: the caller asked
            # for per-component control).
            self.r = dict(r_c)
            self.alpha = dict(a_c)
            self.scale = {c: (a_c[c] / r_c[c] if r_c[c] > 0 else 0.0)
                          for c in _QKV_COMPONENTS}
        else:
            self.r = int(r)
            self.alpha = int(alpha)
            self.scale = float(alpha) / float(r)

    def forward(self, x):
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)

    @torch.no_grad()
    def eval_kernel(self):
        self.q.eval_kernel(); self.k.eval_kernel(); self.v.eval_kernel()
        return self

    @torch.no_grad()
    def eval_reference(self):
        self.q.eval_reference(); self.k.eval_reference(); self.v.eval_reference()
        return self

    def trainable_params(self):
        out = []
        for tag, m in (("q", self.q), ("k", self.k), ("v", self.v)):
            for n, p in m.trainable_params():
                out.append((f"{tag}.{n}", p))
        return out


def _scope_matches(name, scope):
    if scope == "all": return True
    if scope == "mlp":
        return ".mlp." in name and any(s in name for s in ("gate_proj", "up_proj", "down_proj"))
    if scope == "attn":
        return any(s in name for s in ("q_proj", "k_proj", "v_proj", "o_proj",
                                        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"))
    return re.search(scope, name) is not None


def _tensor_metadata(pal_or_split):
    if isinstance(pal_or_split, SplitQKV):
        return {"is_qkv": True, "components": {
            "q": {"N": pal_or_split.q_proj.N, "K": pal_or_split.q_proj.K,
                  "group_size": pal_or_split.q_proj.group_size, "bitwidth": pal_or_split.q_proj.bitwidth},
            "k": {"N": pal_or_split.k_proj.N, "K": pal_or_split.k_proj.K,
                  "group_size": pal_or_split.k_proj.group_size, "bitwidth": pal_or_split.k_proj.bitwidth},
            "v": {"N": pal_or_split.v_proj.N, "K": pal_or_split.v_proj.K,
                  "group_size": pal_or_split.v_proj.group_size, "bitwidth": pal_or_split.v_proj.bitwidth}}}
    return {"is_qkv": False, "N": pal_or_split.N, "K": pal_or_split.K,
            "group_size": pal_or_split.group_size, "bitwidth": pal_or_split.bitwidth}


def attach_qlora(model, metadata=None, r=64, alpha=16, dropout=0.05,
                 scope="all", include_residual_branch=True,
                 init_a="kaiming_uniform", init_b="zero",
                 base_model="Qwen/Qwen3.5-9B", artifacts_dir="",
                 rank_map=None, alpha_mode="proportional"):
    """Wrap every in-scope palettized module with a trainable LoRA branch.

    rank_map (optional): {dotted module path: r} — plain modules use their
    own path, a SplitQKV is addressed per component ("<path>.q", ".k",
    ".v"; a bare QKV path is invalid and lands in the unknown-key guard).
    Modules absent from the map fall back to `r`. r_i = 0 leaves the module
    unwrapped (the frozen base stays in the model) and records an explicit
    {r: 0, alpha: 0, scale: 0.0} marker in cfg.tensors. rank_map=None takes
    the legacy uniform-r path, byte-identically.

    alpha_mode (consulted only when rank_map is set): "proportional"
    (default) sets alpha_i = r_i // 4 — kappa = 0.25, the proven stack's
    exact scale at r=64/alpha=16, so every wrapped module keeps scale 0.25;
    "global" keeps the legacy fixed alpha (scale then varies per module).
    """
    if rank_map is not None and alpha_mode not in _ALPHA_MODES:
        raise ValueError(f"attach_qlora: unknown alpha_mode {alpha_mode!r} "
                          f"(expected one of {_ALPHA_MODES})")
    # Validate the init contract ONCE, before any module is wrapped — the
    # per-module QLoRALinear check stays as defense in depth, but the
    # attach entry point fails fast with ONE loud error (not one per module).
    _validate_init_modes(init_a, init_b, "attach_qlora")
    cfg = QLoRAConfig(r=r, alpha=alpha, dropout=dropout, scope=scope,
                      include_residual_branch=include_residual_branch,
                      init_a=init_a, init_b=init_b,
                      base_model=base_model, artifacts_dir=artifacts_dir,
                      rank_map=rank_map, alpha_mode=alpha_mode, tensors={})
    n_wrapped, n_zero = 0, 0
    consumed = set()
    for dotted, parent, attr_name, child in pmod.iter_palettized_top(model):
        if not _scope_matches(dotted, scope): continue
        if isinstance(child, SplitQKV):
            # Per-component ranks; a bare `dotted` key is deliberately
            # never consumed here (a QKV rank must name its component).
            r_c, a_c = {}, {}
            for comp in _QKV_COMPONENTS:
                key = f"{dotted}.{comp}"
                if rank_map is not None and key in rank_map:
                    r_c[comp] = _check_rank(key, rank_map[key])
                    consumed.add(key)
                else:
                    r_c[comp] = r
                if r_c[comp] == 0:
                    a_c[comp] = 0          # explicit no-adapter marker
                elif rank_map is None:
                    a_c[comp] = alpha
                else:
                    a_c[comp] = _resolve_alpha(key, r_c[comp], alpha,
                                               alpha_mode)
            meta = _tensor_metadata(child)
            for comp in _QKV_COMPONENTS:
                cmeta = meta["components"][comp]
                cmeta["r"] = r_c[comp]
                cmeta["alpha"] = a_c[comp]
                cmeta["scale"] = a_c[comp] / r_c[comp] if r_c[comp] > 0 else 0.0
            if all(r_c[comp] == 0 for comp in _QKV_COMPONENTS):
                # Whole QKV at r=0: the original SplitQKV stays in place.
                cfg.tensors[dotted] = dict(meta, r=0, alpha=0, scale=0.0)
                n_zero += 1
                continue
            if len(set(r_c.values())) == 1 and len(set(a_c.values())) == 1:
                # Uniform geometry: scalar wrapper args + top-level summary.
                wrapper = QLoRASplitQKV(child, r_c["q"], a_c["q"],
                                        dropout, init_a, init_b)
                meta["r"] = r_c["q"]
                meta["alpha"] = a_c["q"]
                meta["scale"] = a_c["q"] / r_c["q"]
            else:
                # Mixed ranks: per-component dicts; no invented top-level
                # scalar r/alpha/scale (the components carry the geometry).
                wrapper = QLoRASplitQKV(child, r_c, a_c, dropout,
                                        init_a, init_b)
            setattr(parent, attr_name, wrapper)
            cfg.tensors[dotted] = meta
            n_wrapped += 1
        else:
            r_i = r
            if rank_map is not None and dotted in rank_map:
                r_i = _check_rank(dotted, rank_map[dotted])
                consumed.add(dotted)
            if r_i == 0:
                # r_i = 0: the original PalettizedLinear stays in the model
                # (forward is the frozen base); no lora_A/lora_B keys.
                cfg.tensors[dotted] = dict(_tensor_metadata(child),
                                           r=0, alpha=0, scale=0.0)
                n_zero += 1
                continue
            a_i = alpha if rank_map is None \
                else _resolve_alpha(dotted, r_i, alpha, alpha_mode)
            wrapper = QLoRALinear(child, r_i, a_i, dropout, init_a, init_b)
            setattr(parent, attr_name, wrapper)
            cfg.tensors[dotted] = dict(_tensor_metadata(child), r=r_i,
                                       alpha=a_i, scale=a_i / r_i)
            n_wrapped += 1
    if rank_map is not None:
        # Never-silently-no-op: every key must have named a wrapped (or
        # r=0-skipped) plain module or QKV component path.
        unknown = sorted(k for k in rank_map if k not in consumed)
        if unknown:
            shown = ", ".join(unknown[:10]) + \
                (", ..." if len(unknown) > 10 else "")
            raise RuntimeError(
                f"attach_qlora: {len(unknown)} rank_map key(s) matched no "
                f"palettized module or QKV component "
                f"(QKV keys are '<path>.q/.k/.v'): {shown}")
    if n_wrapped == 0:
        if rank_map is None:
            raise RuntimeError("attach_qlora: no PalettizedLinear/SplitQKV found")
        raise RuntimeError("attach_qlora: no PalettizedLinear/SplitQKV wrapped "
                           f"(scope={scope!r} matched no module or every "
                           f"mapped rank is 0)")
    print(f"  [qlora] attached: {n_wrapped} modules wrapped", flush=True)
    if rank_map is not None:
        print(f"  [qlora] rank map: {len(consumed)}/{len(rank_map)} keys, "
              f"alpha_mode={alpha_mode}, {n_zero} module(s) at r=0",
              flush=True)
    return model, cfg


def iter_qlora_modules(root):
    for name, mod in root.named_modules():
        if isinstance(mod, (QLoRALinear, QLoRASplitQKV)):
            yield name, mod


def report_frozen_paths(root):
    """The frozen dequant-path fact report.

    Walks every QLoRALinear under `root` (a QLoRASplitQKV contributes its
    q/k/v components — each an QLoRALinear yielded in its own right by
    iter_qlora_modules, so QKV components are reported like any other
    module) and reports, per module, the path FROZEN at attach time
    (`self._frozen_path` — read here, never re-resolved), the base
    geometry (N, K) and the torch-gpu-cached branch's materialized W16
    footprint (0 until the first forward materializes it).

    PURELY observational — a fact-printing instrument: no state is
    changed, no module is touched, no path is re-resolved or asserted
    (start-of-run CUDA enforcement is a separate guard).

    Returns a dict:
      {
        "n_modules": int,      # total QLoRALinear components under root
        "counts": {path: n},   # aggregate counts per path value, with an
                               # explicit 0 for every unused legal path
        "bwd": "fused-kernel" | "cached-cuBLAS",
        "modules": [ {"name": str, "N": int, "K": int, "path": str,
                      "cached_w_bytes": int}, ... ],
      }

    `bwd` mirrors qlora_gemm's escape semantics exactly: "fused-kernel"
    only when the fused backward kernel is available AND FLUTE_FUSED_BWD
    is not set to the explicit opt-out ("0", or the empty value
    qlora_gemm also treats as disabled); otherwise "cached-cuBLAS"
    (gradients flow through qlora_fallback's cached reference path).
    """
    modules = []
    counts = {p: 0 for p in _FROZEN_PATHS}
    for name, mod in iter_qlora_modules(root):
        if not isinstance(mod, QLoRALinear):
            continue  # a QLoRASplitQKV: its q/k/v QLoRALinear children
                      # are yielded below in their own right
        path = getattr(mod, "_frozen_path", None)
        if path not in counts:
            # an unknown path value is a broken invariant, not a fact to
            # silently mis-bucket — never a silent no-op
            raise RuntimeError(
                f"report_frozen_paths: module {name!r} carries an unknown "
                f"frozen path {path!r} (legal values: {_FROZEN_PATHS})")
        counts[path] += 1
        W = getattr(mod, "_torch_cached_W", None)
        cached_w_bytes = int(W.numel() * W.element_size()) \
            if torch.is_tensor(W) else 0
        modules.append({"name": name, "N": int(mod.base.N),
                        "K": int(mod.base.K), "path": path,
                        "cached_w_bytes": cached_w_bytes})
    # bwd: fused only if the kernel is available AND not explicitly
    # escaped (FLUTE_FUSED_BWD "" / "0" — the same values qlora_gemm's
    # _check_backward_kernel treats as the opt-out).
    escaped = os.environ.get("FLUTE_FUSED_BWD", "1") in ("", "0")
    bwd = "fused-kernel" if (not escaped and _fused_backward_ok()) \
        else "cached-cuBLAS"
    return {"n_modules": len(modules), "counts": counts, "bwd": bwd,
            "modules": modules}


def _flatten_qlora_state_dict(model):
    sd = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QLoRALinear):
            sd[f"{name}.lora_A"] = mod.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = mod.lora_B.detach().cpu()
        elif isinstance(mod, QLoRASplitQKV):
            for tag, sub in (("q", mod.q), ("k", mod.k), ("v", mod.v)):
                sd[f"{name}.{tag}.lora_A"] = sub.lora_A.detach().cpu()
                sd[f"{name}.{tag}.lora_B"] = sub.lora_B.detach().cpu()
    return sd


def save_qlora(model, out_dir, config):
    os.makedirs(out_dir, exist_ok=True)
    sd = _flatten_qlora_state_dict(model)
    if not sd: raise RuntimeError("save_qlora: no QLoRA modules found")
    torch.save(sd, os.path.join(out_dir, "qlora_adapters.pt"))
    config.to_json(os.path.join(out_dir, "qlora_config.json"))
    print(f"  [qlora] saved {len(sd)} tensors -> {out_dir}", flush=True)


def load_qlora(model, adapters_dir, strict=True, config=None):
    if config is None:
        config = QLoRAConfig.from_json(os.path.join(adapters_dir, "qlora_config.json"))
    sd = torch.load(os.path.join(adapters_dir, "qlora_adapters.pt"), map_location="cpu")
    model_sd = dict(model.named_parameters())
    n_loaded = 0
    missing = []
    for k, v in sd.items():
        if k not in model_sd:
            missing.append(k)
            continue
        param = model_sd[k]
        if v.shape != param.shape:
            raise RuntimeError(f"load_qlora: shape mismatch for {k}")
        param.data.copy_(v.to(param.device, dtype=param.dtype))
        n_loaded += 1
    if missing:
        # A silently skipped key is a silently un-applied adapter —
        # fail loudly instead.
        preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        if strict:
            raise RuntimeError(
                f"load_qlora: {len(missing)}/{len(sd)} adapter keys not found "
                f"in the model — refusing to silently no-op. "
                f"First missing: {preview}")
        print(f"  [qlora] WARNING: {len(missing)}/{len(sd)} adapter keys "
              f"skipped (not in model): {preview}", flush=True)
    print(f"  [qlora] loaded {n_loaded}/{len(sd)} tensors from {adapters_dir}", flush=True)
    return model, config


def load_qlora_model(artifacts_dir, adapters_dir=None, model_name="Qwen/Qwen3.5-9B",
                     device="cuda:0", dtype=torch.float16, residual=False,
                     r=64, alpha=16, dropout=0.05, scope="all", eval_kernel=True):
    model, metadata = pmod.load_palettized_model(
        artifacts_dir, model_name, device=device, dtype=dtype,
        residual=residual, reference=False)
    if adapters_dir is not None:
        cfg = QLoRAConfig.from_json(os.path.join(adapters_dir, "qlora_config.json"))
        model, cfg = attach_qlora(
            model, metadata, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout,
            scope=cfg.scope, include_residual_branch=cfg.include_residual_branch,
            init_a=cfg.init_a, init_b=cfg.init_b,
            base_model=model_name, artifacts_dir=artifacts_dir,
            rank_map=cfg.rank_map, alpha_mode=cfg.alpha_mode)
        load_qlora(model, adapters_dir, strict=True, config=cfg)
    else:
        model, cfg = attach_qlora(
            model, metadata, r=r, alpha=alpha, dropout=dropout, scope=scope,
            base_model=model_name, artifacts_dir=artifacts_dir)
    model.eval()
    if eval_kernel:
        for _, mod in iter_qlora_modules(model):
            mod.eval_kernel()
    return model, cfg


def count_trainable_params(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total - n_train, n_total


def freeze_all_non_qlora(model):
    for name, p in model.named_parameters():
        is_lora = name.endswith(".lora_A") or name.endswith(".lora_B")
        p.requires_grad_(is_lora)
    n_train, n_frozen, n_total = count_trainable_params(model)
    print(f"  [qlora] trainable={n_train/1e6:.2f}M frozen={n_frozen/1e6:.2f}M "
          f"total={n_total/1e6:.2f}M ({100.0*n_train/max(n_total,1):.4f}%)", flush=True)
    return model


__all__ = ["QLoRAConfig", "QLoRALinear", "QLoRASplitQKV",
           "attach_qlora", "save_qlora", "load_qlora", "load_qlora_model",
           "iter_qlora_modules", "report_frozen_paths",
           "count_trainable_params",
           "freeze_all_non_qlora", "_resolve_frozen_path"]
