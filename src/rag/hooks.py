"""hooks.py — the 9-hook capture harness (SPECIFICATION.md §1; the snapshot
object spec §5 consumes).

WHY CAPTURE AT HOOKS: each linear-attention layer's recurrent state is
written exactly once per forward (`update_recurrent_state` inside
`Qwen3_5GatedDeltaNet.forward`, W2's interception point). By the time
hook j fires — after the boundary layer's forward RETURNS — every layer
mapped to hook j has already run and its TurboQuant codes never change
again within that forward. Hook capture therefore equals post-forward
capture for the FINAL codes, but the harness also PROVES the layer stack
ran the full pattern (every hook fired: layer 0 + all 8 full-attention
boundaries) and keeps mid-forward capture points for diagnostics. The
snapshot is what spec §5 ingests: 24 S codes + 24 conv codes + M1 + M2,
all TQCodes — codes only, never fp tensors.

THE MAP (the enumerated contract — the authority):

    hook 0: after layer 0  -> S_0
    hook 1: after layer 3  -> S_1, S_2
    hook 2: after layer 7  -> S_4, S_5, S_6
    hook 3: after layer 11 -> S_8, S_9, S_10
    hook 4: after layer 15 -> S_12, S_13, S_14
    hook 5: after layer 19 -> S_16, S_17, S_18
    hook 6: after layer 23 -> S_20, S_21, S_22
    hook 7: after layer 27 -> S_24, S_25, S_26
    hook 8: after layer 31 -> S_28, S_29, S_30

DIAGRAM-ARITHMETIC NOTE: spec §1's inline "1 + 8×3 = 24" line miscounts
its own diagram — hook 0 takes S_0 out of the first group, so hook 1
captures only S_1, S_2 and the remaining seven hooks capture three layers
each. The true total is 1 + 2 + 7×3 = 24, which is what this module and
its tests pin.

GENERAL RULE (any [L,L,L,F]-style layer_types list): hook 0 fires after
layer 0 and captures it when it is linear; each subsequent hook fires
after a FULL-ATTENTION layer and captures the linear layers that ran
since the previous boundary (excluding already-captured). Total captured
MUST equal the number of linear layers — validated at construction with
a loud error otherwise: the harness refuses patterns it cannot fully
capture (an all-linear stack beyond layer 0, or trailing linear layers
after the last full-attention boundary, leave layers whose codes no hook
would ever see).

EMPTY CAPTURE POINTS (decision): a full-attention boundary with no
uncaptured linear layers behind it yields a capture point with EMPTY
linear_layers — a pure boundary witness: it still fires and still proves
that full-attention layer ran. Kept, NOT skipped — the production map is
unchanged under this decision (it contains no empty points; every hook
in the enumerated table captures at least one layer), and firing at
every boundary is what makes `capture(require_all_fired=True)` a proof
that the WHOLE pattern ran, not just the layers that carry codes.

CODES ONLY: the forward hooks never touch fp tensors — they copy the
TQCodes references out of the bound `TQCache` (`cache.s_codes[L]` /
`cache.conv_codes[L]`), which are current at that moment (a later write
replaces the layer's codes OBJECT, so an already-captured reference is
immutable-in-practice and never mutates under the hook). Dequantization
happens once, in `CacheSnapshot.capture_vector()` (spec §4), through the
kind's quantizer (`tq_cache.resolve_quantizer`, which keeps the kind's
D3 seed at every scale).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

import _paths  # noqa: F401  (anchors src/rag, src/scripts, src/flute_extended)
from tq_cache import TQCache, resolve_quantizer
from turboquant import TQCodes

__all__ = [
    "SPEC_LAYER_TYPES", "CapturePoint", "CacheSnapshot", "CaptureHooks",
    "hook_map",
]

# ------------------------------------------------------------ constants ---
_LINEAR = "linear_attention"
_FULL = "full_attention"
_VALID_LAYER_TYPES = (_LINEAR, _FULL)

# The Qwen3.5 32-layer plan (SPECIFICATION §1): [L, L, L, F] x 8.
SPEC_LAYER_TYPES: List[str] = [_LINEAR, _LINEAR, _LINEAR, _FULL] * 8


# ----------------------------------------------------------- the map -----
@dataclass(frozen=True)
class CapturePoint:
    """One firing boundary of the harness (spec §1).

    hook_idx        0..K-1, sequential
    after_layer     the layer whose forward RETURN triggers the hook
    linear_layers   the S layers captured at that moment (may be empty —
                    a pure boundary witness; see the module docstring)
    """

    hook_idx: int
    after_layer: int
    linear_layers: tuple[int, ...]


def hook_map(layer_types: Sequence[str]) -> List[CapturePoint]:
    """The capture map for a layer_types plan, per the general rule.

    Loud validation: unknown layer types, empty plans, and plans that
    cannot be fully captured (linear layers behind the LAST capture
    boundary — the all-linear stack beyond layer 0, or trailing linear
    layers after the last full-attention layer) all raise ValueError at
    construction: the harness refuses patterns it cannot fully capture.
    """
    types = list(layer_types)
    if not types:
        raise ValueError(
            "hook_map: layer_types is empty — nothing to capture (spec §1 "
            "expects the 32-layer [L,L,L,F]x8 plan)")
    for i, lt in enumerate(types):
        if lt not in _VALID_LAYER_TYPES:
            raise ValueError(
                f"hook_map: layer_types[{i}] = {lt!r} is not a known layer "
                f"type (valid: {list(_VALID_LAYER_TYPES)})")

    # hook 0: after layer 0, capturing it when it is linear
    points: List[CapturePoint] = [
        CapturePoint(hook_idx=0, after_layer=0,
                     linear_layers=(0,) if types[0] == _LINEAR else ())]
    pending: List[int] = []
    for i in range(1, len(types)):
        if types[i] == _LINEAR:
            pending.append(i)
        else:  # a full-attention layer is a firing boundary
            points.append(CapturePoint(hook_idx=len(points), after_layer=i,
                                       linear_layers=tuple(pending)))
            pending = []
    if pending:
        n_linear = sum(1 for lt in types if lt == _LINEAR)
        captured = [L for p in points for L in p.linear_layers]
        raise ValueError(
            f"hook_map: this layer plan cannot be fully captured — linear "
            f"layer(s) {pending} sit behind the last capture boundary "
            f"(after layer {points[-1].after_layer}) and no later hook "
            f"would ever fire to capture them ({len(captured)} of "
            f"{n_linear} linear layers captured). The harness refuses "
            f"patterns it cannot fully capture; the spec §1 plan ends on "
            f"a full-attention layer and never hits this.")

    # loud invariant: every linear layer is captured exactly once
    captured = [L for p in points for L in p.linear_layers]
    n_linear = sum(1 for lt in types if lt == _LINEAR)
    if len(captured) != n_linear or len(set(captured)) != len(captured):
        # unreachable given the construction above — kept loud anyway
        raise RuntimeError(
            f"hook_map: internal invariant violated — captured {captured} "
            f"for {n_linear} linear layers")
    return points


# ---------------------------------------------------------- dequant ------
def _codes_bits(codes: TQCodes) -> float:
    """The float bit-width a TQCodes unit was quantized at (uniform b ->
    b; split -> floor(x) + hi-set fraction), for resolving the kind's
    quantizer at capture_vector time."""
    if codes.bits_lo == codes.bits_hi:
        return float(codes.bits_lo)
    return float(codes.bits_lo) + (codes.n_hi / codes.d)


def _dequant_codes(kind: str, codes: TQCodes) -> torch.Tensor:
    """Dequantize one unit through the KIND's quantizer (the D3-seeded
    frame, at whatever scale the codes carry), fp32, shape (d,)."""
    quantizer = resolve_quantizer(kind, codes.d, _codes_bits(codes))
    return quantizer.dequant(codes, dtype=torch.float32)


# --------------------------------------------------------- the snapshot --
@dataclass
class CacheSnapshot:
    """The §5 snapshot: 24 S codes + 24 conv codes + M1 + M2 (all TQCodes).

    S/conv codes were copied by the hooks at their firing boundaries
    (latest wins — decode steps re-fire hooks and refresh); M1/M2 are
    read from the cache at capture time.
    """

    s_codes: Dict[int, TQCodes]
    conv_codes: Dict[int, TQCodes]
    m1_codes: Optional[TQCodes]
    m2_codes: Optional[TQCodes]

    def capture_vector(self) -> torch.Tensor:
        """The §4 retrieval vector: concat(dequant(S) for S layers
        ascending, dequant(M1), dequant(M2)) — a 1-D fp32 tensor.

        Production dims: 24 x 524,288 + 2 x 524,288 = 13,631,488. Each
        unit dequantizes through its kind's quantizer
        (`tq_cache.resolve_quantizer(kind, d)` — the D3 frame contract
        holds at every scale).
        """
        parts: List[torch.Tensor] = []
        for layer_idx in sorted(self.s_codes):
            codes = self.s_codes[layer_idx]
            if codes is None:
                raise ValueError(
                    f"CacheSnapshot.capture_vector: no S codes for layer "
                    f"{layer_idx} — the snapshot is incomplete")
            parts.append(_dequant_codes("S", codes))
        for kind, codes in (("M1", self.m1_codes), ("M2", self.m2_codes)):
            if codes is not None:
                parts.append(_dequant_codes(kind, codes))
        if not parts:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(parts).to(torch.float32)


# ----------------------------------------------------------- the hooks ---
def _resolve_stack(target) -> List[nn.Module]:
    """Normalize an attach() target to a plain list of nn.Module.

    Accepts an nn.ModuleList, a list/tuple of nn.Module, or a model with
    `.layers` (the real `Qwen3_5TextModel` route). Loud TypeErrors
    otherwise.
    """
    if isinstance(target, nn.ModuleList):
        seq: List[nn.Module] = list(target)
    elif isinstance(target, (list, tuple)):
        seq = list(target)
    elif hasattr(target, "layers"):
        raw = target.layers
        if isinstance(raw, nn.ModuleList):
            seq = list(raw)
        elif isinstance(raw, (list, tuple)):
            seq = list(raw)
        else:
            raise TypeError(
                f"CaptureHooks.attach: target.layers is "
                f"{type(raw).__name__} — expected an nn.ModuleList or a "
                f"list/tuple of nn.Module")
    else:
        raise TypeError(
            f"CaptureHooks.attach: target must be an nn.ModuleList, a "
            f"list/tuple of nn.Module, or a model with .layers — got "
            f"{type(target).__name__}")
    if not seq:
        raise ValueError("CaptureHooks.attach: the layer stack is empty")
    for i, module in enumerate(seq):
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"CaptureHooks.attach: stack layer {i} is "
                f"{type(module).__name__}, not an nn.Module")
    return seq


class CaptureHooks:
    """Registers the pattern's forward hooks and collects the §5 snapshot.

    Lifecycle: `bind(cache)` and `attach(stack)` in either order, run the
    forward(s), `capture()`. `attach` is idempotent (detaches old handles
    first and resets the capture session); re-running the stack re-fires
    the hooks and refreshes the captured codes — latest wins, the decode
    loop contract.

    The hooks never touch fp tensors: they copy `cache.s_codes[L]` /
    `cache.conv_codes[L]` references, current at the moment the boundary
    layer's forward returns. NB: the single-token decode path's conv
    state is mutated in place by `causal_conv1d_update` and lazily
    re-captured by the cache on its NEXT read (W2 design); ingestion
    prefill (T > 1) writes through `update_conv_state`, so its codes are
    current at hook time — the harness's target regime.
    """

    def __init__(self, layer_types: Sequence[str] = SPEC_LAYER_TYPES):
        self._layer_types = list(layer_types)
        # loud construction-time validation of the capture map
        self._points: List[CapturePoint] = hook_map(self._layer_types)
        self._cache: Optional[TQCache] = None
        self._handles: List = []
        self._fired: Dict[int, int] = {}
        self._captured_s: Dict[int, Optional[TQCodes]] = {}
        self._captured_conv: Dict[int, Optional[TQCodes]] = {}

    # ------------------------------------------------------ introspection -
    @property
    def points(self) -> List[CapturePoint]:
        """The validated capture map (a copy — diagnostics/tests)."""
        return list(self._points)

    @property
    def fired_counts(self) -> Dict[int, int]:
        """Fire count per hook_idx (0 = never fired) — the proof the full
        pattern ran; diagnostics."""
        return {p.hook_idx: self._fired.get(p.hook_idx, 0)
                for p in self._points}

    # ------------------------------------------------------------- binding -
    def bind(self, cache: TQCache) -> None:
        """Bind the TQCache the hooks read codes from (before running the
        forward). Loudly refuses cache/hook plan mismatches."""
        for attr in ("s_codes", "conv_codes", "m1_codes", "m2_codes"):
            if not hasattr(cache, attr):
                raise TypeError(
                    f"CaptureHooks.bind: expected a TQCache-like cache "
                    f"(missing .{attr}) — got {type(cache).__name__}")
        mapped = {L for p in self._points for L in p.linear_layers}
        known = set(getattr(cache, "linear_layer_indices", lambda: [])())
        if mapped != known:
            raise ValueError(
                f"CaptureHooks.bind: the cache's linear-attention plan "
                f"{sorted(known)} disagrees with the hook map's captured "
                f"layers {sorted(mapped)} — bind a cache built from the "
                f"same layer_types")
        n = len(getattr(cache, "layers", []) or [])
        if n and n != len(self._layer_types):
            raise ValueError(
                f"CaptureHooks.bind: the cache carries {n} layers but the "
                f"hook map is for {len(self._layer_types)} (layer_types "
                f"mismatch)")
        self._cache = cache

    # ------------------------------------------------------------- attach --
    def attach(self, target) -> None:
        """Register forward hooks on the boundary layers (after_layer).
        `target`: an nn.ModuleList / list of nn.Module / a model with
        `.layers`. Idempotent: old handles are detached first and the
        capture session (fired counts, captured codes) resets."""
        layers = _resolve_stack(target)
        if len(layers) != len(self._layer_types):
            raise ValueError(
                f"CaptureHooks.attach: the stack has {len(layers)} layers "
                f"but the hook map is for {len(self._layer_types)} "
                f"(layer_types mismatch)")
        # soft consistency check when the modules self-describe (the real
        # Qwen3_5DecoderLayer carries .block_type; stubs may not)
        for p in self._points:
            block_type = getattr(layers[p.after_layer], "block_type", None)
            if block_type is not None and block_type != \
                    self._layer_types[p.after_layer]:
                raise ValueError(
                    f"CaptureHooks.attach: hook {p.hook_idx} fires after "
                    f"layer {p.after_layer}, but that module's block_type "
                    f"is {block_type!r} while the hook plan says "
                    f"{self._layer_types[p.after_layer]!r}")
        self.detach()
        self._fired = {}
        self._captured_s = {}
        self._captured_conv = {}
        for p in self._points:
            handle = layers[p.after_layer].register_forward_hook(
                self._make_hook(p))
            self._handles.append(handle)

    def detach(self) -> None:
        """Remove all registered forward hooks (captured codes are kept —
        a snapshot taken afterwards still reflects what fired)."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------ capture --
    def capture(self, require_all_fired: bool = True) -> CacheSnapshot:
        """Collect the §5 snapshot from the fired hooks (latest wins).

        Raises loudly when `require_all_fired` and some hook never fired
        (the layer stack did not run the full pattern), or whenever a
        mapped linear layer has no codes (a mapped layer ran but wrote
        nothing — codes capture is the proof the pattern ran).
        """
        cache = self._cache
        if cache is None:
            raise RuntimeError(
                "CaptureHooks.capture: no cache bound — call bind(cache) "
                "first")
        if require_all_fired:
            unfired = [p.hook_idx for p in self._points
                       if self._fired.get(p.hook_idx, 0) == 0]
            if unfired:
                raise RuntimeError(
                    f"CaptureHooks.capture: hook(s) {unfired} never fired — "
                    f"the layer stack did not run the full pattern (hook "
                    f"map: {self._describe_points()})")
        missing_s = sorted(
            L for p in self._points for L in p.linear_layers
            if self._captured_s.get(L) is None)
        missing_conv = sorted(
            L for p in self._points for L in p.linear_layers
            if self._captured_conv.get(L) is None)
        if missing_s or missing_conv:
            raise RuntimeError(
                f"CaptureHooks.capture: mapped linear layers have no codes "
                f"(S: {missing_s}, conv: {missing_conv}) — a mapped layer "
                f"ran but wrote nothing to the bound cache, or its hook "
                f"fired before any write; the snapshot would be incomplete")
        s_codes = {L: self._captured_s[L]
                   for p in self._points for L in p.linear_layers}
        conv_codes = {L: self._captured_conv[L]
                      for p in self._points for L in p.linear_layers}
        # M1/M2 are read at capture time (spec §5: globals, not per-hook)
        return CacheSnapshot(s_codes=s_codes, conv_codes=conv_codes,
                             m1_codes=cache.m1_codes,
                             m2_codes=cache.m2_codes)

    # ------------------------------------------------------------ plumbing -
    def _make_hook(self, point: CapturePoint):
        """The forward hook for one capture point: marks it fired and
        copies the mapped layers' CURRENT codes (references — no fp
        tensors touched)."""
        def _hook(_module, _args, _output):
            self._fired[point.hook_idx] = self._fired.get(
                point.hook_idx, 0) + 1
            cache = self._cache
            if cache is None:
                raise RuntimeError(
                    f"CaptureHooks: hook {point.hook_idx} (after layer "
                    f"{point.after_layer}) fired with no cache bound — "
                    f"call bind(cache) before running the forward")
            s_view = cache.s_codes
            conv_view = cache.conv_codes
            for L in point.linear_layers:
                try:
                    self._captured_s[L] = s_view[L]
                    self._captured_conv[L] = conv_view[L]
                except KeyError as exc:
                    raise RuntimeError(
                        f"CaptureHooks: hook {point.hook_idx} maps linear "
                        f"layer {L}, but the bound cache has no such layer "
                        f"(cache plan vs hook layer_types mismatch)") \
                        from exc
        return _hook

    def _describe_points(self) -> str:
        return "; ".join(
            f"h{p.hook_idx}@{p.after_layer}->{list(p.linear_layers)}"
            for p in self._points)
