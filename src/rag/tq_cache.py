"""tq_cache.py — the online TurboQuant cache wrapper (SPECIFICATION.md §3.2).

Quantize-on-write / dequantize-on-read around the transformers
`DynamicCache` protocol, so the model's forward never knows it is using
quantized caches:

  READ   cache.layers[L].conv_states[0] / .recurrent_states[0]
         → dequantized fp16 tensor (transient — the read path itself)
  WRITE  cache.update_conv_state(x, L, ...) / .update_recurrent_state(s, L)
         → TurboQuant codes stored; the layer dicts hold NO state tensors

Interception points are the exact call sites of
`src/scripts/modeling.py::Qwen3_5GatedDeltaNet.forward`:

  * `cache_params.layers[L].conv_states[0]`      (single-token decode read)
  * `cache_params.update_conv_state(mixed_qkv, L, conv_kernel_size=...)` (write)
  * `cache_params.layers[L].recurrent_states[0]` (decode read)
  * `cache_params.update_recurrent_state(last_recurrent_state, L)`      (write)
  * `cache_params.has_previous_state(L)` / `.layers[L].record_past`

The conv decode path has one subtlety the wrapper must honor: on
single-token steps `modeling.py` calls `causal_conv1d_update(...)` which
mutates the handed-out conv state IN PLACE without calling
`update_conv_state`. The wrapper therefore re-captures (requantizes) the
handed-out tensor at the NEXT access of that layer's conv state — the
lazy sync below. The codes always reflect the latest known state; the
cache still never PERSISTS fp16.

D4 fallback: `online=False` selects the quantize-on-snapshot regime (raw
tensors during forward, `snapshot_codes()` quantizes at chunk boundaries)
— spec §3.2's "ALWAYS codes" relaxed to "always on disk", decided by the
Phase-2 gate on the GPU box, never by default here.

M1/M2 slots (spec §2.2/§5): the two global memories are cache STATE —
`update_m1/read_m1/m1_codes` (+ M2) quantize/dequantize with their own
kinds and D3 seeds, exactly like S.

Quantizer resolution: the production shapes hit the canonical kinds
(S: 1×32×128×128 = 524,288; conv: 1×8192×4 = 32,768). Smaller
power-of-two shapes (tests) resolve to a custom-size quantizer that keeps
the KIND's seed, so the D3 frame contract holds at every scale.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from transformers.cache_utils import (
    DYNAMIC_LAYER_TYPE_MAPPING,
    DynamicCache,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
)
import _paths  # noqa: F401
import turboquant as tq
from turboquant import KINDS, SEEDS, TQCodes, TurboQuant

__all__ = ["TQCache", "TQLinearAttentionLayer", "resolve_quantizer"]

_ONLINE_ERR = (
    "TQCache(online): the cache never holds fp16 state tensors "
    "(SPECIFICATION §3.2) — writes go through update_conv_state / "
    "update_recurrent_state, which quantize; use online=False for the "
    "quantize-on-snapshot fallback (PROPOSAL D4)")


def resolve_quantizer(kind: str, numel: int, bits: float = 3.5) -> TurboQuant:
    """The quantizer for a kind at a given flattened size.

    Production sizes hit the canonical kinds (shared rotation per kind,
    D3). Other power-of-two sizes (tests) get a custom-d quantizer that
    KEEPS the kind's seed — same frame contract, smaller unit.
    """
    if (numel & (numel - 1)) != 0 or numel < 1:
        raise ValueError(
            f"resolve_quantizer({kind}): flattened size {numel} is not a "
            f"power of two — the FHT single-block contract (PROPOSAL D2) "
            f"requires power-of-two units")
    canonical_d, seed = KINDS[kind]
    if numel == canonical_d:
        return tq.get_quantizer(kind, bits)
    key = f"{kind}:{bits}:{numel}"
    reg = tq._REGISTRY
    if key not in reg:
        reg[key] = TurboQuant(kind="custom", bits=bits, d=numel, seed=seed)
    return reg[key]


# ------------------------------------------------------------ state view ---
class _StateView(dict):
    """The dict-like the model reads states through (`layer.conv_states[0]`).

    Online mode: __getitem__ dequantizes the codes (and, for conv states,
    first re-captures any in-place mutation of the previously handed-out
    tensor — the `causal_conv1d_update` decode path). __setitem__ refuses:
    the cache never stores fp16.
    Offline mode (D4 snapshot fallback): plain dict passthrough.
    """

    def __init__(self, layer: "TQLinearAttentionLayer", which: str):
        super().__init__()
        self._layer = layer
        self._which = which  # "conv" | "s"

    def __getitem__(self, state_idx: int):
        layer = self._layer
        if layer.online:
            if self._which == "conv":
                layer._sync_conv()
                layer.reads["conv"] += 1
                codes = layer._conv_codes
                if codes is None:
                    return None
                t = layer._tq_conv.dequant(codes, dtype=layer._conv_dtype)
                t = t.reshape(layer._conv_shape)
                layer._handed_conv = t
                return t
            layer.reads["s"] += 1
            codes = layer._s_codes
            if codes is None:
                return None
            t = layer._tq_s.dequant(codes, dtype=layer._s_dtype)
            return t.reshape(layer._s_shape)
        return super().__getitem__(state_idx)

    def __setitem__(self, state_idx: int, value) -> None:
        if self._layer.online:
            raise RuntimeError(_ONLINE_ERR)
        super().__setitem__(state_idx, value)


# ------------------------------------------------------- TQ linear layer ---
class TQLinearAttentionLayer(LinearAttentionLayer):
    """A linear-attention cache layer whose store is TurboQuant codes.

    Mirrors `LinearAttentionLayer`'s exact update semantics (the conv
    windowing contract `modeling.py` depends on) while substituting the
    persistent store: codes, never state tensors.
    """

    def __init__(self, number_of_states: int = 1, bits: float = 3.5,
                 online: bool = True):
        super().__init__(number_of_states=number_of_states)
        self.bits = float(bits)
        self.online = bool(online)
        self._tq_s: Optional[TurboQuant] = None
        self._tq_conv: Optional[TurboQuant] = None
        self._s_codes: Optional[TQCodes] = None
        self._conv_codes: Optional[TQCodes] = None
        self._s_shape: Optional[torch.Size] = None
        self._s_dtype: torch.dtype = torch.float16
        self._conv_shape: Optional[torch.Size] = None
        self._conv_dtype: torch.dtype = torch.float16
        self._handed_conv: Optional[torch.Tensor] = None
        self.reads = {"conv": 0, "s": 0}
        self.writes = {"conv": 0, "s": 0}
        # replace the plain state dicts with the dequantizing views
        self.conv_states = _StateView(self, "conv")
        self.recurrent_states = _StateView(self, "s")

    # ------------------------------------------------------------ helpers -
    def _sync_conv(self) -> None:
        """Re-capture the handed-out conv tensor (in-place mutation by
        causal_conv1d_update) into codes — the lazy quantize-on-write."""
        if self.online and self._handed_conv is not None:
            if self._conv_codes is not None:
                t = self._handed_conv
                self._conv_codes = self._tq_conv.quant(t.reshape(-1))
            self._handed_conv = None

    def _init_s(self, tensor: torch.Tensor, state_idx: int) -> None:
        n = tensor.numel()
        self._tq_s = resolve_quantizer("S", n, self.bits)
        self._s_shape = tuple(tensor.shape)
        self._s_dtype = tensor.dtype
        self.is_recurrent_states_initialized[state_idx] = True

    def _init_conv(self, conv_states: torch.Tensor, state_idx: int,
                   conv_kernel_size: Optional[int]) -> int:
        kernel = int(conv_kernel_size or conv_states.shape[-1])
        window_shape = (*conv_states.shape[:-1], kernel)
        n = 1
        for s in window_shape:
            n *= s
        self._tq_conv = resolve_quantizer("conv", n, self.bits)
        self._conv_shape = tuple(window_shape)
        self._conv_dtype = conv_states.dtype
        self.conv_kernel_size[state_idx] = kernel
        self.is_conv_states_initialized[state_idx] = True
        return kernel

    # ------------------------------------------------------ cache-layer API -
    def lazy_initialization(self, conv_states=None, recurrent_states=None,
                            state_idx: int = 0,
                            conv_kernel_size: Optional[int] = None) -> None:
        """Online: capture shapes only — NEVER allocate fp16 state tensors.
        Offline: parent behavior (raw tensors, snapshot fallback)."""
        if not self.online:
            return super().lazy_initialization(
                conv_states=conv_states, recurrent_states=recurrent_states,
                state_idx=state_idx, conv_kernel_size=conv_kernel_size)
        if recurrent_states is not None and not self.is_recurrent_states_initialized[state_idx]:
            self._init_s(recurrent_states, state_idx)
        if conv_states is not None and not self.is_conv_states_initialized[state_idx]:
            self._init_conv(conv_states, state_idx, conv_kernel_size)

    def update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0,
                          conv_kernel_size: Optional[int] = None,
                          **kwargs) -> torch.Tensor:
        """The conv windowing contract of `LinearAttentionLayer`, with a
        code store: returns cat([old_window, new]) for the causal conv,
        persists quant(last kernel) only."""
        if not self.online:
            return super().update_conv_state(
                conv_states, state_idx=state_idx,
                conv_kernel_size=conv_kernel_size, **kwargs)
        self._sync_conv()
        if not self.is_conv_states_initialized[state_idx]:
            kernel = self._init_conv(conv_states, state_idx, conv_kernel_size)
        else:
            kernel = int(self.conv_kernel_size[state_idx])
        self.writes["conv"] += 1
        if not self.has_previous_state[state_idx]:
            # first (prefill) call: full = new states, left-padded to kernel
            full = conv_states
            if full.shape[-1] < kernel:
                pad = kernel - full.shape[-1]
                full = torch.nn.functional.pad(full, (pad, 0), value=0.0)
            self.has_previous_state[state_idx] = True
        else:
            old = self._tq_conv.dequant(self._conv_codes,
                                        dtype=self._conv_dtype)
            old = old.reshape(self._conv_shape)
            full = torch.cat([old, conv_states], dim=-1)
        window = full[..., -kernel:]
        self._conv_codes = self._tq_conv.quant(window.reshape(-1))
        return full

    def update_recurrent_state(self, recurrent_states: torch.Tensor,
                               state_idx: int = 0,
                               **kwargs) -> torch.Tensor:
        """Quantize-on-write for S; returns the dequantized state (the
        read path — a transient, never persisted)."""
        if not self.online:
            return super().update_recurrent_state(
                recurrent_states, state_idx=state_idx, **kwargs)
        if not self.is_recurrent_states_initialized[state_idx]:
            self._init_s(recurrent_states, state_idx)
        self.writes["s"] += 1
        self._s_codes = self._tq_s.quant(recurrent_states.reshape(-1))
        out = self._tq_s.dequant(self._s_codes, dtype=self._s_dtype)
        return out.reshape(self._s_shape)

    # --------------------------------------------------------- code access -
    @property
    def s_codes(self) -> Optional[TQCodes]:
        return self._s_codes

    @s_codes.setter
    def s_codes(self, codes: Optional[TQCodes]) -> None:
        self._s_codes = codes
        if codes is not None:
            if not self.is_recurrent_states_initialized.get(0):
                self._tq_s = resolve_quantizer("S", codes.d, self.bits)
                if self._s_shape is None:
                    self._s_shape = (codes.d,)  # flat until a real shape is known
                self._s_dtype = torch.float16
                self.is_recurrent_states_initialized[0] = True
            self.has_previous_state[0] = True

    @property
    def conv_codes(self) -> Optional[TQCodes]:
        return self._conv_codes

    @conv_codes.setter
    def conv_codes(self, codes: Optional[TQCodes]) -> None:
        self._conv_codes = codes
        if codes is not None:
            if not self.is_conv_states_initialized.get(0):
                self._tq_conv = resolve_quantizer("conv", codes.d, self.bits)
                if self._conv_shape is None:
                    self._conv_shape = (codes.d,)  # flat until a real shape is known
                self._conv_dtype = torch.float16
                self.is_conv_states_initialized[0] = True
            self.has_previous_state[0] = True

    def snapshot_codes(self) -> Dict[str, Optional[TQCodes]]:
        """D4 fallback: quantize the currently-held raw tensors (offline
        mode). Online mode: the codes as they stand."""
        if self.online:
            self._sync_conv()
            return {"s": self._s_codes, "conv": self._conv_codes}
        out: Dict[str, Optional[TQCodes]] = {"s": None, "conv": None}
        s = dict.get(self.recurrent_states, 0)
        c = dict.get(self.conv_states, 0)
        if s is not None and self._tq_s is None:
            self._init_s(s, 0)
        if s is not None:
            out["s"] = self._tq_s.quant(s.reshape(-1))
        if c is not None and self._tq_conv is None:
            self._init_conv(c, 0, None)
        if c is not None:
            out["conv"] = self._tq_conv.quant(c.reshape(-1))
        return out

    # -------------------------------------------------- inherited overrides -
    def reset(self) -> None:
        """Codes-aware reset (parent's zero_() path assumes raw tensors)."""
        self._sync_conv()
        self._s_codes = None
        self._conv_codes = None
        self._handed_conv = None
        for i in range(self.number_of_states):
            self.has_previous_state[i] = False
            self.is_conv_states_initialized[i] = False
            self.is_recurrent_states_initialized[i] = False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError(
            "TQLinearAttentionLayer.reorder_cache: beam search over "
            "TurboQuant code stores is out of scope for the RAG build "
            "(single-query decode contract, SPECIFICATION §6)")


# ---------------------------------------------------------------- TQCache ---
class TQCache(DynamicCache):
    """The monkey-patched cache the model receives as `past_key_values`.

    Construction:
      * TQCache(config=model.config) — wraps the linear-attention layers
        the config builds (production path; full-attn layers untouched,
        spec §2.4).
      * TQCache(layer_types=[...]) — builds layers directly from a
        layer-type list (tests / stub stacks).

    M1/M2 (spec §2.2): global cache STATE on this object —
    update_m1/read_m1/m1_codes (+M2 twins), quantized with their own
    D3-seeded kinds.
    """

    def __init__(self, config=None, layer_types: Optional[Iterable[str]] = None,
                 bits: float = 3.5, online: bool = True):
        self._tq_bits = float(bits)
        self._online = bool(online)
        self._m1_codes: Optional[TQCodes] = None
        self._m2_codes: Optional[TQCodes] = None
        self._tq_m1: Optional[TurboQuant] = None
        self._tq_m2: Optional[TurboQuant] = None
        self._m1_shape = None
        self._m2_shape = None
        if config is not None:
            super().__init__(config=config)
            self._wrap_linear_layers()
        else:
            types = list(layer_types or [])
            layers = []
            for lt in types:
                if lt == "linear_attention":
                    layers.append(TQLinearAttentionLayer(
                        bits=self._tq_bits, online=self._online))
                else:
                    cls = DYNAMIC_LAYER_TYPE_MAPPING.get(lt)
                    if cls is None:
                        raise ValueError(
                            f"TQCache: unknown layer type {lt!r} (known: "
                            f"{sorted(DYNAMIC_LAYER_TYPE_MAPPING)})")
                    layers.append(cls())
            # DynamicCache.__init__ builds layers itself only from config/ddp
            # data; the layer list route lives on the Cache base.
            super(DynamicCache, self).__init__(layers=layers)

    # -------------------------------------------------------------- wrap ---
    def _wrap_linear_layers(self) -> None:
        for i, layer in enumerate(self.layers):
            if (isinstance(layer, LinearAttentionCacheLayerMixin)
                    and not isinstance(layer, TQLinearAttentionLayer)):
                self.layers[i] = TQLinearAttentionLayer(
                    number_of_states=layer.number_of_states,
                    bits=self._tq_bits, online=self._online)

    def linear_layer_indices(self) -> List[int]:
        return [i for i, l in enumerate(self.layers)
                if isinstance(l, TQLinearAttentionLayer)]

    # --------------------------------------------------------- code views --
    @property
    def s_codes(self) -> Dict[int, Optional[TQCodes]]:
        """spec §5: `tq_cache.s_codes[layer_idx]` — the S code store."""
        return {i: l.s_codes for i, l in enumerate(self.layers)
                if isinstance(l, TQLinearAttentionLayer)}

    @property
    def conv_codes(self) -> Dict[int, Optional[TQCodes]]:
        return {i: l.conv_codes for i, l in enumerate(self.layers)
                if isinstance(l, TQLinearAttentionLayer)}

    def set_s_codes(self, layer_idx: int, codes: Optional[TQCodes]) -> None:
        self.layers[layer_idx].s_codes = codes

    def set_conv_codes(self, layer_idx: int, codes: Optional[TQCodes]) -> None:
        self.layers[layer_idx].conv_codes = codes

    # ------------------------------------------------------------- M1 / M2 --
    def _resolve_m(self, which: str, tensor: torch.Tensor) -> None:
        tqm = resolve_quantizer(which, tensor.numel(), self._tq_bits)
        setattr(self, f"_tq_{which.lower()}", tqm)
        setattr(self, f"_{which.lower()}_shape", tuple(tensor.shape))

    def update_m1(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize-on-write for the global key-memory (spec §2.2/§5)."""
        if (self._tq_m1 is None or self._m1_codes is None
                or tensor.numel() != self._m1_codes.d):
            self._resolve_m("M1", tensor)
        self._m1_codes = self._tq_m1.quant(tensor.reshape(-1))
        out = self._tq_m1.dequant(self._m1_codes, dtype=tensor.dtype)
        return out.reshape(tuple(tensor.shape))

    def read_m1(self, dtype: torch.dtype = torch.float16) -> Optional[torch.Tensor]:
        if self._m1_codes is None:
            return None
        return self._tq_m1.dequant(self._m1_codes, dtype=dtype).reshape(self._m1_shape)

    @property
    def m1_codes(self) -> Optional[TQCodes]:
        return self._m1_codes

    @m1_codes.setter
    def m1_codes(self, codes: Optional[TQCodes]) -> None:
        self._m1_codes = codes
        if codes is not None:
            self._tq_m1 = resolve_quantizer("M1", codes.d, self._tq_bits)
            self._m1_shape = (codes.d,)

    def update_m2(self, tensor: torch.Tensor) -> torch.Tensor:
        if (self._tq_m2 is None or self._m2_codes is None
                or tensor.numel() != self._m2_codes.d):
            self._resolve_m("M2", tensor)
        self._m2_codes = self._tq_m2.quant(tensor.reshape(-1))
        out = self._tq_m2.dequant(self._m2_codes, dtype=tensor.dtype)
        return out.reshape(tuple(tensor.shape))

    def read_m2(self, dtype: torch.dtype = torch.float16) -> Optional[torch.Tensor]:
        if self._m2_codes is None:
            return None
        return self._tq_m2.dequant(self._m2_codes, dtype=dtype).reshape(self._m2_shape)

    @property
    def m2_codes(self) -> Optional[TQCodes]:
        return self._m2_codes

    @m2_codes.setter
    def m2_codes(self, codes: Optional[TQCodes]) -> None:
        self._m2_codes = codes
        if codes is not None:
            self._tq_m2 = resolve_quantizer("M2", codes.d, self._tq_bits)
            self._m2_shape = (codes.d,)

    # ------------------------------------------------------------ snapshot --
    def snapshot_codes(self) -> Dict[str, object]:
        """All codes of this cache (the §5 snapshot: 24 S + 24 conv + M1/M2
        at production shapes)."""
        self._sync_all()
        return {
            "s": self.s_codes,
            "conv": self.conv_codes,
            "m1": self._m1_codes,
            "m2": self._m2_codes,
        }

    def _sync_all(self) -> None:
        for l in self.layers:
            if isinstance(l, TQLinearAttentionLayer):
                l._sync_conv()
