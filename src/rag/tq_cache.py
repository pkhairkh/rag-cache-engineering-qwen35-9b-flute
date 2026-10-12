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

THE W29 PER-HEAD S MODE (the F2b repair — FINDINGS W28 item 9, the box
matrix's worst-block 7.65 at 38x head-norm spread): a 4-D recurrent
state (B, H, K, V) is quantized as B*H PER-HEAD units of d=K*V (the
production geometry: 32 units of 16,384, the SAME kind seed 101 at the
per-head scale — the D3 frame contract at every scale), each unit with
its OWN fp32 norm (the paper's per-vector norm recipe — Theorem 1
applied per unit), through TurboQuant.quant_units' vectorized chain
(ONE norm reduction + ONE batched fht + ONE batched searchsorted per
bit-width + one pack per stream — no per-unit Python loop). The frame
is the PERMUTED rotation y = FHT(P x) (P from seed+4545: the 2^14
Hadamard's Kronecker alignment with the 128x128 matrix layout makes
the plain rotation mismatch the codebook on rank-1/low-rank heads —
measured 0.087/0.056 vs the 0.022 floor; the permutation restores the
CLT — 0.0221 structure-independent; scripts/w28_tq_sim.py). Sim
worst-head rel-MSE: 10.69 -> 0.023 at 55x spread, 297 -> 0.023 at
525x. The layer's read path, snapshot codes and every §4 consumer see
the SAME flat (B*H*K*V,) geometry (dequant returns the flat vector).

s_flat=True (DEFAULT False) restores the W28 flat regime — ONE 524,288-
dim unit, ONE norm, the legacy bit-exact path (the A/B flag; also the
fallback for non-4-D states). Codes are self-describing: units codes
decode through dequant's partition routing, flat codes through the
legacy route — a mixed-regime install is REFUSED loudly by the frame
guards (re-ingest the corpus at the per-head regime to install).

M1/M2 stay on the flat single-unit regime this wave (documented:
the memory subspace is INERT on the ingested corpora — zero-init
gates, exact-zero codes — so the hazard is unmeasured there; the
split decision rides the W21+ gates retrain, HANDOVER §5).

Quantizer resolution: the production S shape hits the canonical kind
(1×32×128×128 = 524,288). The conv unit resolves at the window's flat
size rounded up to a multiple of 32 (the Qwen3.5 in_proj Q+V geometry:
1×6144×4 = 24,576 — exactly a multiple) through the W11 full-window
policy: no truncation, no power-of-two padding of the UNIT (the pow2
pads live INSIDE the W15 split sub-units — see below). Custom sizes
keep the KIND's seed (D3 frame contract at every scale).

THE W15 OUTLIER SPLIT (the paper's §LongBench recipe, conv default):
`_init_conv` resolves the conv quantizer with `group = kernel` — the
window's (channel, tap) flat layout is one group per channel, and
TurboQuant(group=kernel) quantizes the top-round(frac·n_ch) energy
channels at bits_hi with their OWN power-of-two full rotation + norm
(the paper: "two independent instances of TurboQuant"). The codes carry
the channel mask / group / second norm and are SELF-DESCRIBING — the
conv_codes setter and every generic reader (hooks, index, bisect)
resolve from the codes' own group, and dequant routes on
codes.partition, so pre-W15 partition="half" snapshots decode
unchanged. Measured: ~3x lower conv write-path distortion than the
legacy fixed coordinate split at the same effective bits (the W15 wave
— see turboquant.py's module docstring).

THE W9.2 `graph_safe` FLAG (P7 CUDA-graph capture hooks — DEFAULT OFF):
TQCache(..., graph_safe=True) / TQLinearAttentionLayer(..., graph_safe=True)
is a passthrough contract marker for the GPU box's P7 hardening (the
cuda-graph capture of the decode step — the quantize-on-write IS in the
hot loop, PROPOSAL P7). On CPU it is PROVABLY a no-op: the flag gates
nothing in the update/read paths (the W9.2 parity gate: flag-on ==
flag-off outputs, bit-identical), and the two hook methods it exposes,
before_graph_capture()/after_graph_capture(), are empty (see their
docstrings). torch._dynamo.mark_static_address is NEVER called by this
module (zero dynamo references — the only call sites in the process are
the PARENT class's offline lazy_initialization, which the online path
replaces entirely; see TQLinearAttentionLayer.lazy_initialization).

W20.4.1 (the wiring audit): this module was authored as the TORCH SURFACE
of the NumPy-era core — torch tensors in (quant), TQCodes out, W18's
dequant `device` parameter doing the one-move-at-the-source placement —
so the W20.2/W20.3 landings beneath it (TQCodes the torch-native
container; turboquant's quant/dequant device-native) complete the
interception with zero further code motion here: every read dequantizes
through the device parameter onto the tracked/writer device, every write
stores the torch TQCodes as-is (the npz edge lives in snapshot.py's
to_arrays/from_arrays — never here). The zero audit (the brief's
host-array / `.item` residue grep) reads 0 module-wide; no DISK-EDGE
lines exist — this module has no disk edge. The 120-192 per-prefill
round trips ride device tensors end-to-end statically; the behavioral
proof is G20.P1's on the A10G box.
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


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1) — legacy helper (the W10
    zero-pad-to-pow2 conv policy; superseded by `_conv_quant_dim`)."""
    if n < 1:
        raise ValueError(f"_next_pow2: n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


_MAX_KERNEL_SEGMENT = 16384


def _conv_quant_dim(numel: int) -> int:
    """The conv unit's quantizer dimension: the FULL flat window rounded up
    to a multiple of 32 (the FHT kernel's thread geometry).

    No truncation, no power-of-two padding: the segmented FHT transforms the
    window block-diagonally (fht.segments — descending powers of two), and
    Qwen3.5's 24,576-element window segments as 16,384 + 8,192 — every
    segment fits the 64 KiB kernel tile, so the whole window rides ONE code
    unit in the kind's shared D3 frame (seed 202). Rounding up to 32
    zero-pads at most 31 coordinates (energy-free: the stored norm is the
    real window's). Windows whose FIRST segment exceeds 16,384 (numel >=
    32,768) still quantize correctly — the FHT then runs the torch reference
    on-device instead of the CUDA kernel (fht._kernel_eligible).
    """
    if numel < 1:
        raise ValueError(f"_conv_quant_dim: numel must be >= 1, got {numel}")
    return ((numel + 31) // 32) * 32


def resolve_quantizer(kind: str, numel: int, bits: float = 3.5,
                       group: int = 1, qjl: bool = False) -> TurboQuant:
    """The quantizer for a kind at a given flattened size.

    Production sizes hit the canonical kinds (shared rotation per kind,
    D3). Other sizes get a custom-d quantizer that KEEPS the kind's seed —
    same frame contract (PROPOSAL D3), different unit length.

    The segmented FHT (fht.segments) makes the old power-of-two
    requirement obsolete: any d is quantizable (the reference butterfly
    handles every segmentable d; the CUDA kernel auto-engages for d it
    can tile — fht._kernel_eligible: multiple of 32, every segment
    <= 16,384). The CALLING layer picks the d (the conv policy rounds
    the window up to a multiple of 32; S/M1/M2 use their canonical d).

    W15 `group` (default 1): the paper's §LongBench outlier split for
    channel-grouped units — the conv window's (channel, tap) layout, one
    group per channel (group = the conv kernel width). group > 1 and
    non-integer bits quantize the top-round(frac·n_ch) energy channels
    at bits_hi with their own frame/norm (see TurboQuant). Codes are
    self-describing: dequant routes on codes.partition, so a
    split-configured quantizer still decodes legacy partition="half"
    codes exactly (through the flat twin).

    W9.2 `qjl` (default False): the Alg.-2 residual-sketch A/B flag,
    plumbed through the cache stack (TQCache(qjl=...)) so the GPU box
    can run the paper's two-stage estimator end-to-end.
    """
    if numel < 1:
        raise ValueError(f"resolve_quantizer({kind}): numel must be >= 1, got {numel}")
    canonical_d, seed = KINDS[kind]
    if numel == canonical_d and group == 1 and not qjl:
        return tq.get_quantizer(kind, bits)
    key = f"{kind}:{bits}:{numel}:{group}:{int(qjl)}"
    reg = tq._REGISTRY
    if key not in reg:
        reg[key] = TurboQuant(kind="custom", bits=bits, d=numel, seed=seed,
                               qjl=qjl, group=group)
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
                # W18: dequant's device param carries the placement (the
                # tracked device — the write-time tensor's device)
                t = layer._dequant_conv(layer._conv_dtype)
                layer._handed_conv = t
                return t
            layer.reads["s"] += 1
            codes = layer._s_codes
            if codes is None:
                return None
            t = layer._tq_s.dequant(codes, dtype=layer._s_dtype,
                                     device=layer._device)
            t = t.reshape(layer._s_shape)
            return t
        # offline mode: plain passthrough (None before the first write —
        # the mixin's dicts are lazily filled, read-before-write is legal)
        return dict.get(self, state_idx)

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
                 online: bool = True, graph_safe: bool = False,
                 qjl: bool = False, s_flat: bool = False):
        super().__init__(number_of_states=number_of_states)
        self.bits = float(bits)
        self.online = bool(online)
        # W9.2 P7 flag (DEFAULT OFF — pure contract marker on CPU; see the
        # module docstring): carried on the layer, passed through by TQCache.
        self.graph_safe = bool(graph_safe)
        # W15: the Alg.-2 qjl A/B flag, plumbed through (default OFF).
        self.qjl = bool(qjl)
        # W29: the per-head S mode (DEFAULT ON — the F2b repair, FINDINGS
        # W28 item 9; s_flat=True restores the W28 flat single-unit regime
        # for the A/B and the non-4-D fallback).
        self.s_flat = bool(s_flat)
        if self.qjl and not self.s_flat:
            raise ValueError(
                "TQLinearAttentionLayer: the W9.2 qjl A/B flag and the W29 "
                "per-head S mode are mutually exclusive this wave (the "
                "residual sketch is defined per FLAT unit; the units A/B "
                "lands with the fused kernel chain) — pass s_flat=True to "
                "run the qjl A/B on the legacy flat regime")
        self._tq_s: Optional[TurboQuant] = None
        self._tq_conv: Optional[TurboQuant] = None
        self._s_codes: Optional[TQCodes] = None
        self._conv_codes: Optional[TQCodes] = None
        self._s_shape: Optional[torch.Size] = None
        self._s_dtype: torch.dtype = torch.float16
        # W29: the per-head S geometry — None in the flat regime; in the
        # per-head regime (_s_units, _s_unit_d) = (B*H, K*V) of the
        # learned state shape, and the quantizer is the PER-UNIT frame.
        self._s_units: Optional[int] = None
        self._s_unit_d: Optional[int] = None
        self._conv_shape: Optional[torch.Size] = None
        self._conv_dtype: torch.dtype = torch.float16
        self._conv_numel: Optional[int] = None   # real window size (pre-pad)
        self._conv_d: Optional[int] = None       # quantizer unit size (>= numel)
        self._handed_conv: Optional[torch.Tensor] = None
        # track device for correct dequant placement
        self._device: Optional[torch.device] = None
        self.reads = {"conv": 0, "s": 0}
        self.writes = {"conv": 0, "s": 0}
        self._m1m2_tokens = 0  # tokens seen at this layer (M1/M2 write positions)
        # replace the plain state dicts with the dequantizing views
        self.conv_states = _StateView(self, "conv")
        self.recurrent_states = _StateView(self, "s")

    # ------------------------------------------------------------ helpers -
    def _pad_conv_flat(self, t: torch.Tensor) -> torch.Tensor:
        """The (numel,) flat conv window, zero-padded UP to the (d,) quantizer
        unit — a no-op for multiple-of-32 geometries (`_conv_d ==
        _conv_numel`; Qwen3.5's 24,576 window). NEVER truncates: a window
        longer than the unit is geometry drift (raise, never silently drop
        channels). Padding adds no energy: the stored norm (and the
        relative-MSE budget on the REAL coordinates) is unchanged."""
        flat = t.reshape(-1)
        target = self._conv_d if self._conv_d is not None else flat.numel()
        if flat.numel() > target:
            raise ValueError(
                f"conv window numel {flat.numel()} exceeds the layer's "
                f"quantizer unit d={target} — geometry drift (the unit is "
                f"sized by _conv_quant_dim at init; it never truncates)")
        pad = target - flat.numel()
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad))
        return flat

    def _dequant_conv(self, dtype: torch.dtype) -> torch.Tensor:
        """Codes -> the shaped window: dequant the (d,) unit, STRIP the <= 31
        round-up pad, reshape to `_conv_shape`. The tracked device (when
        known — set at the first write) is the output's device (W18: the
        dequant device parameter, one move at the source)."""
        t = self._tq_conv.dequant(self._conv_codes, dtype=dtype,
                                  device=self._device)
        n = self._conv_numel if self._conv_numel is not None else t.numel()
        if t.numel() > n:
            t = t[:n]
        elif t.numel() < n:
            # a codes-installed flat unit short of the learned window: pad
            # (the setter keeps numel == d; this only fires on mixed flows)
            t = torch.nn.functional.pad(t, (0, n - t.numel()))
        return t.reshape(self._conv_shape)

    def _sync_conv(self) -> None:
        """Re-capture the handed-out conv tensor (in-place mutation by
        causal_conv1d_update) into codes — the lazy quantize-on-write."""
        if self.online and self._handed_conv is not None:
            if self._conv_codes is not None:
                t = self._handed_conv
                self._conv_codes = self._tq_conv.quant(self._pad_conv_flat(t))
            self._handed_conv = None

    def _init_s(self, tensor: torch.Tensor, state_idx: int) -> None:
        """Resolve the S quantizer for the learned state geometry.

        W29: a 4-D (B, H, K, V) state in the per-head mode (the default)
        resolves B*H units of d=K*V — the per-unit quantizer KEEPS the
        kind's seed (resolve_quantizer's custom-d path, the D3 frame
        contract at every scale) and quant_units applies the per-unit
        norms + the permuted rotation. Non-4-D states and s_flat=True
        resolve the legacy flat unit at numel (bit-identical W28 path).
        The 4-D shape must have H >= 1 and K*V == the resolved d — a
        degenerate (K*V == 0) state falls back to the flat regime
        (nothing to split)."""
        n = tensor.numel()
        per_head = (not self.s_flat) and tensor.dim() == 4 \
            and tensor.shape[1] >= 1 and tensor.shape[2] * tensor.shape[3] > 0
        if per_head:
            self._s_units = int(tensor.shape[0]) * int(tensor.shape[1])
            self._s_unit_d = int(tensor.shape[2]) * int(tensor.shape[3])
            self._tq_s = resolve_quantizer(
                "S", self._s_unit_d, self.bits, qjl=self.qjl)
        else:
            self._s_units = None
            self._s_unit_d = None
            self._tq_s = resolve_quantizer("S", n, self.bits, qjl=self.qjl)
        self._s_shape = tuple(tensor.shape)
        self._s_dtype = tensor.dtype
        if self._device is None:
            self._device = tensor.device
        self.is_recurrent_states_initialized[state_idx] = True

    def _init_conv(self, conv_states: torch.Tensor, state_idx: int,
                   conv_kernel_size: Optional[int]) -> int:
        kernel = int(conv_kernel_size or conv_states.shape[-1])
        window_shape = (*conv_states.shape[:-1], kernel)
        n = 1
        for s in window_shape:
            n *= s
        # The full-window policy (SPECIFICATION N16.1): d = the flat window
        # rounded up to a multiple of 32 — NO truncation (the GPU session's
        # 16,384 cap silently zeroed 8,192 of the 24,576 coordinates at every
        # read — the ingestion deltas and every installed conv window were
        # built on a one-third-blinded conv state), NO power-of-two padding
        # (the W10 policy's 32,768 unit has a single 32,768-segment FHT the
        # CUDA kernel cannot tile: 128 KiB > the ~99 KiB consumer opt-in;
        # 24,576 segments as 16,384 + 8,192 — every tile fits 64 KiB).
        d = _conv_quant_dim(n)
        # W15: the paper's outlier split — the conv window's flat layout is
        # (channel, tap), so ONE GROUP per channel (group = kernel). The
        # split routes through TurboQuant(group=kernel): top-k energy
        # channels at bits_hi with their own frame/norm (class docstring).
        self._tq_conv = resolve_quantizer("conv", d, self.bits,
                                          group=kernel, qjl=self.qjl)
        self._conv_shape = tuple(window_shape)
        self._conv_numel = n
        self._conv_d = d
        self._conv_dtype = conv_states.dtype
        self._device = conv_states.device
        self.conv_kernel_size[state_idx] = kernel
        self.is_conv_states_initialized[state_idx] = True
        return kernel

    # ------------------------------------------------------ cache-layer API -
    def lazy_initialization(self, conv_states=None, recurrent_states=None,
                            state_idx: int = 0,
                            conv_kernel_size: Optional[int] = None) -> None:
        """Online: capture shapes only — NEVER allocate fp16 state tensors.
        Offline: parent behavior (raw tensors, snapshot fallback).

        W9.2 `graph_safe` VERIFICATION (the no-marking guarantee): the only
        torch._dynamo.mark_static_address call sites reachable from this
        class are inside the PARENT's lazy_initialization (transformers'
        cache_utils — it marks the RAW conv/recurrent tensors "to be able
        to use cudagraphs"). The ONLINE branch below never calls the
        parent and this module has ZERO torch._dynamo references, so the
        online path provably makes no mark_static_address calls — there is
        nothing for graph_safe to skip. The offline (D4) branch keeps the
        parent's marks VERBATIM (they are transformers' own cudagraph
        enabler on raw tensors, and the offline regime is not the P7
        capture target), so graph_safe=True changes nothing there either.
        """
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
            old = self._dequant_conv(self._conv_dtype)
            # ensure device consistency with new input
            if old.device != conv_states.device:
                old = old.to(conv_states.device)
            full = torch.cat([old, conv_states], dim=-1)
        window = full[..., -kernel:]
        self._conv_codes = self._tq_conv.quant(self._pad_conv_flat(window))
        return full

    def update_recurrent_state(self, recurrent_states: torch.Tensor,
                               state_idx: int = 0,
                               **kwargs) -> torch.Tensor:
        """Quantize-on-write for S; returns the dequantized state (the
        read path — a transient, never persisted).

        W29: a per-head layer (4-D state, s_flat=False) quantizes the
        (B, H, K, V) state as B*H PER-HEAD units — quant_units' ONE
        vectorized pass, each unit its OWN norm (the paper's per-vector
        recipe — the dynamic-range repair; the OLD path's single 524,288-
        dim norm set the per-coordinate noise floor at the GLOBAL energy
        and crushed the small heads at every decode step). The flat
        regime is bit-identical to W28."""
        if not self.online:
            return super().update_recurrent_state(
                recurrent_states, state_idx=state_idx, **kwargs)
        if not self.is_recurrent_states_initialized[state_idx]:
            self._init_s(recurrent_states, state_idx)
        self.writes["s"] += 1
        if self._s_units is not None:
            # the per-head regime: ONE vectorized quant over (B*H, K*V)
            x2 = recurrent_states.reshape(self._s_units, self._s_unit_d)
            self._s_codes = self._tq_s.quant_units(x2)
        else:
            self._s_codes = self._tq_s.quant(recurrent_states.reshape(-1))
        # W18: the dequant device parameter — the round-trip state lands
        # on the writer's device directly.
        out = self._tq_s.dequant(self._s_codes, dtype=self._s_dtype,
                                 device=recurrent_states.device)
        out = out.reshape(self._s_shape)
        return out

    # ------------------------------------------------- P7 graph hooks ---
    # W9.2 `graph_safe` (P7): the CUDA-graph capture hook points. They are
    # exposed on every TQLinearAttentionLayer (callable no-ops), but they
    # are only part of the contract when the layer was built with
    # graph_safe=True — the GPU box overrides or wraps them when it
    # CUDA-graph-captures the decode step (the quantize-on-write —
    # update_recurrent_state/update_conv_state — is in the captured hot
    # loop, PROPOSAL P7).
    def before_graph_capture(self) -> None:
        """Called by the GPU box immediately BEFORE cuda-graph capture of
        the decode step. NO-OP ON CPU — provable: this method returns None
        and reads/writes no state (the codes, shapes, counters are
        untouched; assert away in the W9.2 parity gate). Intended GPU-side
        use: warm up / pin the static buffers the captured region will
        read (e.g. materialize the dequantized read shapes once) so the
        capture sees stable addresses.
        """
        return None

    def after_graph_capture(self) -> None:
        """Called by the GPU box immediately AFTER cuda-graph capture (and
        typically once per replay batch). NO-OP ON CPU (same proof as
        before_graph_capture). Intended GPU-side use: re-arm anything the
        capture froze — e.g. the LAZY CONV RE-CAPTURE (_sync_conv: the
        in-place causal_conv1d_update mutation must be requantized into
        codes between replays) and any side-channel bookkeeping that must
        stay OUTSIDE the captured region.
        """
        return None

    # --------------------------------------------------------- code access -
    @property
    def s_codes(self) -> Optional[TQCodes]:
        return self._s_codes

    @s_codes.setter
    def s_codes(self, codes: Optional[TQCodes]) -> None:
        self._s_codes = codes
        if codes is not None:
            units = int(codes.units) if getattr(codes, "units", None) is not None \
                and codes.partition == "units" else None
            if not self.is_recurrent_states_initialized.get(0):
                # the codes' OWN cardinals resolve the frame (self-
                # describing): d is the PER-UNIT d on units codes — the
                # kind's seed at the per-head scale (D3 at every scale).
                self._tq_s = resolve_quantizer("S", codes.d, self.bits)
                if units is not None:
                    self._s_units = units
                    self._s_unit_d = int(codes.d)
                    if self._s_shape is None:
                        # flat until a real shape is known — the units
                        # container's flat length (the §4 geometry)
                        self._s_shape = (units * int(codes.d),)
                elif self._s_shape is None:
                    self._s_shape = (codes.d,)  # flat until a real shape is known
                self._s_dtype = torch.float16
                # placement default for code-installed layers that never ran
                # a forward: the ACTIVE CUDA context (a loaded model implies
                # an initialized context; a CPU-only box implies none)
                if self._device is None and torch.cuda.is_initialized():
                    self._device = torch.device("cuda")
                self.is_recurrent_states_initialized[0] = True
            elif self._tq_s is not None and codes.d != self._tq_s.d:
                raise ValueError(
                    f"s_codes setter: codes.d={codes.d} != the layer's "
                    f"quantizer unit d={self._tq_s.d} (geometry/frame "
                    f"drift — the codes and the layer must share the S frame; "
                    f"a flat<->units regime mix is exactly this: re-ingest "
                    f"the corpus at the layer's regime, W29)")
            elif units is not None and self._s_units is not None \
                    and units != self._s_units:
                raise ValueError(
                    f"s_codes setter: codes carry units={units} but the "
                    f"layer's per-head geometry is {self._s_units} units of "
                    f"d={self._s_unit_d} — a batch/head geometry drift (the "
                    f"codes and the layer must share the per-head layout)")
            self.has_previous_state[0] = True

    @property
    def conv_codes(self) -> Optional[TQCodes]:
        return self._conv_codes

    @conv_codes.setter
    def conv_codes(self, codes: Optional[TQCodes]) -> None:
        self._conv_codes = codes
        if codes is not None:
            if not self.is_conv_states_initialized.get(0):
                # W15: the group comes from the CODES (self-describing):
                # partition="outlier" units carry it; legacy "half" units
                # are flat (group=1).
                grp = int(codes.group) if codes.group is not None else 1
                self._tq_conv = resolve_quantizer("conv", codes.d, self.bits,
                                                  group=grp, qjl=self.qjl)
                if self._conv_shape is None:
                    self._conv_shape = (codes.d,)  # flat until a real shape is known
                # flat (d,) unit: numel == d until a forward/reseed teaches
                # the real window shape (same relation by construction —
                # _conv_quant_dim never truncates)
                self._conv_numel = codes.d
                self._conv_d = codes.d
                self._conv_dtype = torch.float16
                # placement default for code-installed layers that never ran
                # a forward: the ACTIVE CUDA context (see s_codes setter)
                if self._device is None and torch.cuda.is_initialized():
                    self._device = torch.device("cuda")
                self.is_conv_states_initialized[0] = True
            elif self._conv_d is not None and codes.d != self._conv_d:
                raise ValueError(
                    f"conv_codes setter: codes.d={codes.d} != the layer's "
                    f"quantizer unit d={self._conv_d} (geometry/frame "
                    f"drift — the codes and the layer must share the conv "
                    f"frame; snapshots built before the full-window policy "
                    f"(pre-W11, d=16,384-truncated) must be RE-INGESTED)")
            # kernel size from the window shape when known (reseeded layers
            # skip lazy init — conv_kernel_size must not stay None)
            if self._conv_shape is not None and len(self._conv_shape) >= 1 \
                    and self.conv_kernel_size.get(0) is None:
                self.conv_kernel_size[0] = int(self._conv_shape[-1])
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
            if self._s_units is not None:
                # W29: the per-head regime — quant_units over (B*H, K*V)
                out["s"] = self._tq_s.quant_units(
                    s.reshape(self._s_units, self._s_unit_d))
            else:
                out["s"] = self._tq_s.quant(s.reshape(-1))
        if c is not None and self._tq_conv is None:
            self._init_conv(c, 0, None)
        if c is not None:
            out["conv"] = self._tq_conv.quant(self._pad_conv_flat(c))
        return out

    # -------------------------------------------------- inherited overrides -
    def reset(self) -> None:
        """Codes-aware reset (parent's zero_() path assumes raw tensors)."""
        self._sync_conv()
        self._s_codes = None
        self._conv_codes = None
        self._handed_conv = None
        self._m1m2_tokens = 0
        for i in range(self.number_of_states):
            self.has_previous_state[i] = False
            self.is_conv_states_initialized[i] = False
            self.is_recurrent_states_initialized[i] = False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError(
            "TQLinearAttentionLayer.reorder_cache: beam search over "
            "TurboQuant code stores is out of scope for the RAG build "
            "(single-query decode contract, SPECIFICATION §6)")

    # ---------------------------------------- W28: the protocol guards ----
    def crop(self, *args, **kwargs) -> None:
        """W28 — LOUD refusal (never a silent corruption): the parent's
        crop assumes RAW conv/recurrent tensors in the state dicts; the
        online TQ layer holds CODES (the dicts are _StateView shims over
        code stores). A generate()-path call that reaches here (e.g.
        DynamicCache.crop at max-length) would silently no-op or corrupt
        the code stores through the parent's raw-tensor arithmetic.
        If your run hits this, the call came from the transformers
        generate machinery — paste the traceback and the
        tq_gen_matrix.py --trace output (the dev box owns the fix)."""
        raise NotImplementedError(
            "TQLinearAttentionLayer.crop: cropping a TurboQuant code "
            "store is not implemented — the online layer's conv/recurrent "
            "state is CODES, not raw tensors (W28 guard; run "
            "scripts/gpu/tq_gen_matrix.py --trace and paste the "
            "backtrace)")

    def batch_repeat_interleave(self, *args, **kwargs) -> None:
        """W28 — LOUD refusal (the same contract as crop): beam/repeat
        expansion of code stores is not defined for the single-query RAG
        decode path; a silent parent fallback would corrupt the codes."""
        raise NotImplementedError(
            "TQLinearAttentionLayer.batch_repeat_interleave: beam "
            "expansion over TurboQuant code stores is not implemented "
            "(W28 guard; single-query decode contract, SPECIFICATION §6)"
        )


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

    graph_safe (W9.2, DEFAULT False): the P7 flag — passed through to
    every TQLinearAttentionLayer (see that class + the module docstring;
    provably a no-op on CPU).

    s_flat (W29, DEFAULT False = per-head ON): the F2b repair flag —
    False quantizes the 4-D (B, H, K, V) recurrent state as B*H
    per-head units (own norms, the paper's per-vector recipe);
    True restores the W28 flat single-unit regime (the A/B).
    """

    def __init__(self, config=None, layer_types: Optional[Iterable[str]] = None,
                 bits: float = 3.5, online: bool = True,
                 graph_safe: bool = False, qjl: bool = False,
                 s_flat: bool = False):
        self._tq_bits = float(bits)
        self._online = bool(online)
        # W9.2 P7 flag (DEFAULT OFF) — passed through to every wrapped /
        # built TQLinearAttentionLayer; a no-op on CPU (module docstring).
        self._graph_safe = bool(graph_safe)
        # W29: the per-head S mode (DEFAULT ON — the F2b repair, FINDINGS
        # W28 item 9; s_flat=True restores the W28 flat regime for the A/B).
        self._tq_s_flat = bool(s_flat)
        # W15: the paper's Alg.-2 qjl A/B flag, plumbed to every layer's
        # quantizers (DEFAULT OFF — the D1 A/B contract; the GPU box's
        # Phase-5 decision measures it end-to-end with TQCache(qjl=True)).
        self._tq_qjl = bool(qjl)
        self._m1_codes: Optional[TQCodes] = None
        self._m2_codes: Optional[TQCodes] = None
        self._tq_m1: Optional[TurboQuant] = None
        self._tq_m2: Optional[TurboQuant] = None
        self._m1_shape = None
        self._m2_shape = None
        # track device for M1/M2 dequant
        self._m_device: Optional[torch.device] = None
        if config is not None:
            super().__init__(config=config)
            self._wrap_linear_layers()
        else:
            types = list(layer_types or [])
            layers = []
            for lt in types:
                if lt == "linear_attention":
                    layers.append(TQLinearAttentionLayer(
                        bits=self._tq_bits, online=self._online,
                        graph_safe=self._graph_safe, qjl=self._tq_qjl,
                        s_flat=self._tq_s_flat))
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
                    bits=self._tq_bits, online=self._online,
                    graph_safe=self._graph_safe, qjl=self._tq_qjl,
                    s_flat=self._tq_s_flat)

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
        # track device for read path
        if self._m_device is None:
            self._m_device = tensor.device

    def _m_read_device(self) -> Optional[torch.device]:
        """W18: where a codes-only read (no live write yet — e.g. right
        after install_from_disk) should land. Resolution order: the
        device seen at the last M write; else the first tracked layer
        device; else None (CPU — a codes-only cache with no device
        signal at all)."""
        if self._m_device is None:
            for layer in self.layers:
                d = getattr(layer, "_device", None)
                if d is not None:
                    self._m_device = d
                    break
        return self._m_device

    def update_m1(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize-on-write for the global key-memory (spec §2.2/§5)."""
        if (self._tq_m1 is None or self._m1_codes is None
                or tensor.numel() != self._m1_codes.d):
            self._resolve_m("M1", tensor)
        self._m1_codes = self._tq_m1.quant(tensor.reshape(-1))
        # W18: the dequant device parameter — the round-trip state lands
        # on the writer's device directly.
        out = self._tq_m1.dequant(self._m1_codes, dtype=tensor.dtype,
                                   device=tensor.device)
        out = out.reshape(tuple(tensor.shape))
        return out

    def read_m1(self, dtype: torch.dtype = torch.float16) -> Optional[torch.Tensor]:
        if self._m1_codes is None:
            return None
        self._check_m_geometry("read_m1", "M1", self._m1_codes.d,
                               self._m1_shape)
        t = self._tq_m1.dequant(self._m1_codes, dtype=dtype,
                                 device=self._m_read_device()).reshape(self._m1_shape)
        return t

    @property
    def m1_codes(self) -> Optional[TQCodes]:
        return self._m1_codes

    @m1_codes.setter
    def m1_codes(self, codes: Optional[TQCodes]) -> None:
        self._m1_codes = codes
        if codes is not None:
            self._tq_m1 = resolve_quantizer("M1", codes.d, self._tq_bits)
            if self._m1_shape is None:
                self._m1_shape = (codes.d,)  # flat until a real shape is known

    def update_m2(self, tensor: torch.Tensor) -> torch.Tensor:
        if (self._tq_m2 is None or self._m2_codes is None
                or tensor.numel() != self._m2_codes.d):
            self._resolve_m("M2", tensor)
        self._m2_codes = self._tq_m2.quant(tensor.reshape(-1))
        out = self._tq_m2.dequant(self._m2_codes, dtype=tensor.dtype,
                                   device=tensor.device)
        out = out.reshape(tuple(tensor.shape))
        return out

    def read_m2(self, dtype: torch.dtype = torch.float16) -> Optional[torch.Tensor]:
        if self._m2_codes is None:
            return None
        self._check_m_geometry("read_m2", "M2", self._m2_codes.d,
                               self._m2_shape)
        t = self._tq_m2.dequant(self._m2_codes, dtype=dtype,
                                 device=self._m_read_device()).reshape(self._m2_shape)
        return t

    @staticmethod
    def _check_m_geometry(op: str, which: str, d: int, shape) -> None:
        """W17: the M1/M2 codes' flat dim must match the recorded state
        shape — a mismatch (codes installed from another mem_size) is a
        LOUD, actionable error, never a bare torch reshape failure."""
        if shape is None:
            return
        numel = 1
        try:
            for s in shape:
                numel *= int(s)
        except TypeError:
            return  # unknown geometry — the flat read is the contract
        if int(d) != numel:
            raise ValueError(
                f"TQCache.{op}: the {which} codes carry d={int(d)} but "
                f"the recorded state shape {tuple(shape)} holds {numel} "
                f"elements — an M1/M2 geometry mismatch (the codes and "
                f"the module/state were built at different mem_size; "
                f"re-ingest or reload with the matching geometry)")

    @property
    def m2_codes(self) -> Optional[TQCodes]:
        return self._m2_codes

    @m2_codes.setter
    def m2_codes(self, codes: Optional[TQCodes]) -> None:
        self._m2_codes = codes
        if codes is not None:
            self._tq_m2 = resolve_quantizer("M2", codes.d, self._tq_bits)
            if self._m2_shape is None:
                self._m2_shape = (codes.d,)  # flat until a real shape is known

    # ------------------------------------------------------------ snapshot --
    def snapshot_codes(self) -> Dict[str, object]:
        """All codes of this cache (the §5 snapshot: 24 S + 24 conv + M1/M2
        at production shapes). Online: the standing codes. Offline (D4
        fallback): quantizes the held raw tensors per layer."""
        out: Dict[str, object] = {
            "s": {}, "conv": {}, "m1": self._m1_codes, "m2": self._m2_codes}
        for i, l in enumerate(self.layers):
            if isinstance(l, TQLinearAttentionLayer):
                layer_codes = l.snapshot_codes()  # syncs + (offline) quantizes
                out["s"][i] = layer_codes["s"]
                out["conv"][i] = layer_codes["conv"]
        return out

    def _sync_all(self) -> None:
        for l in self.layers:
            if isinstance(l, TQLinearAttentionLayer):
                l._sync_conv()
