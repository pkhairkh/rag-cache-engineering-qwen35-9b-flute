"""m1m2.py — the two global memory matrices M1/M2 (SPECIFICATION.md §2.2).

TWO tensors for the whole model — shared across ALL 24 linear-attention
layers, NOT per-layer (spec §2.2: "TWO for the whole model"):

  M1 (global key-memory):    (num_heads, mem_size, head_dim) = (32, 128, 128)
  M2 (global value-memory):  same shape

  read  (spec §2.2):  softmax(q @ M1ᵀ) @ M2  — folded into the layer's
                      output path by the W3.2 wiring (modeling.py).
  write (spec §2.2):  gated ADDITIVE per-token scatter — slot = position
                      mod mem_size. delta_M1/M2 are therefore
                      path-independent and chunk deltas sum losslessly at
                      install (PROPOSAL D4 / SPEC §6: install sums
                      DEQUANTIZED M1/M2 deltas, never the codes).

CRITICAL ARCHITECTURAL FACT: M1/M2 are RUNTIME CACHE STATE, not weights.
They start zero, accumulate during prefill, and are snapshotted /
quantized / installed as TurboQuant codes through `TQCache.update_m1 /
read_m1 / m1_codes` (+M2 twins) — `src/rag/tq_cache.py`, kinds "M1"/"M2"
(D3 seeds 303/404). At the default geometry the state numel is
32·128·128 = 524,288 = 2^19 — exactly the canonical M1/M2 quantization
unit d (a power of two, as the FHT single-block contract requires); at
1.0 MiB fp16 this matches spec §2.2's "1.0 MiB" per memory. This module
therefore carries NO m1/m2 buffers; its WEIGHTS are the GATES ONLY —
spec §7 / PROPOSAL Phase 4 train "the M1/M2 read/write gates".

Zero-init (PROPOSAL P3): the write gates start at 0.0 — M1/M2 start as
no-ops and the untrained model's behavior is bit-unchanged (write
returns the state bit-identical; read of zero memories returns exact
zeros). The write is deliberately ONE branch-free gated add
(`m + g · Σ_t k_t`): a `if gate == 0: skip` short-circuit would zero the
gradient ∂m_new/∂g = Σ_t k_t exactly where P3's fine-tune has to start
opening the gates.

Online loop (SPEC §3.2, PROPOSAL D4): `m1m2_from_cache` reads the
cache's m1/m2 codes, falling back to `init_state()` zeros when the cache
has none (the first forward sees zeros; the writes open the state);
`push_to_cache` quantize-on-writes through `cache.update_m1/update_m2`
and hands back the dequantized round-trip — the state the NEXT forward
must see (the codes are the persisted truth; the cache never holds
fp16).

Numerics: the softmax path runs in fp32 (scores scaled by 1/√D over the
mem dim, per the read contract); everything else keeps the input dtype —
read returns q.dtype, write returns the STATE's (m1/m2) dtype. k/v are
cast up only when their dtype is wider than the state's; the gated add
accumulates in the promoted dtype and casts back, so uniform-dtype calls
(the tests, the production fp16 path) do no hidden rounding beyond the
native dtype.

Composability notes (spec §2.2 "Summing is lossless"): the additive
write is exact in real arithmetic for any regrouping of the multiset of
(position→token) pairs. In floating point it is BIT-exact whenever the
per-slot token groupings match — which covers the W3.3 contracts:
sequential writes at positions [0,1] then [2,3] vs one [0,1,2,3] call
(each slot lands once), and the two-token slot collision
`m1[h, 0] = g·(k0 + k1)` (the per-slot token sum is formed BEFORE the
single gated add). Re-grouping the same multiset differently across
calls (many tokens in one slot, split differently) is exact only up to
fp re-association (~1 ulp), which the D4 install path absorbs by
requantizing the dequantized SUM once (never summing codes).

This module imports torch only at module top (NO transformers) —
`TQCache` is imported lazily inside the cache helpers; `import _paths`
first (house sys.path anchor, see src/rag/_paths.py).

Refs: SPECIFICATION.md §2.2 (the memories), §3.2 (online loop),
§5 (M1/M2 codes in the snapshot), §6 (install sums dequantized deltas),
§7 (the fine-tune trains the gates); PROPOSAL.md P3 (zero-init write
gates), D4 (the delta protocol), Phase 3/4.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import _paths  # noqa: F401  (house sys.path anchor — must precede sibling imports)

import torch
from torch import nn

if TYPE_CHECKING:  # never executed at runtime — keeps the module transformers-free
    from tq_cache import TQCache

__all__ = ["M1M2", "m1m2_from_cache", "push_to_cache"]


def _pos_int(name: str, value) -> int:
    """Loud positive-int config validation."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"M1M2: {name} must be a python int, got "
            f"{type(value).__name__} ({value!r})")
    if value < 1:
        raise ValueError(f"M1M2: {name} must be >= 1, got {value}")
    return value


class M1M2(nn.Module):
    """The two global memories (SPECIFICATION.md §2.2).

    The WEIGHTS here are the GATES ONLY (spec §7 / PROPOSAL Phase 4
    fine-tune trains exactly these, as per-layer scalars):

      * ``write_gate_k`` / ``write_gate_v`` — (num_linear_layers,) each,
        ZERO-init (PROPOSAL P3: M1/M2 start as no-ops — the untrained
        model's behavior is bit-unchanged; the fine-tune opens them).
      * ``read_gate`` — (num_linear_layers,), ONE-init (reads of the zero
        memories are exact zeros anyway, so P3's bit-unchanged property
        holds from both the read and the write side).

    There are NO m1/m2 buffers on this module: the memories are RUNTIME
    CACHE STATE — zero at start, accumulating during prefill, quantized /
    snapshotted / installed through the TQCache (spec §3.2/§5; see
    `m1m2_from_cache` / `push_to_cache`).

    Geometry: the spec writes the memories as (1, 32, mem_size, 128) —
    the leading 1 is the vacuous batch dim (the memories are GLOBAL, one
    per model, read by every batch row); this module's state shape drops
    it: (num_heads, mem_size, head_dim).
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128,
                 mem_size: int = 128, num_linear_layers: int = 24):
        num_heads = _pos_int("num_heads", num_heads)
        head_dim = _pos_int("head_dim", head_dim)
        mem_size = _pos_int("mem_size", mem_size)
        num_linear_layers = _pos_int("num_linear_layers", num_linear_layers)
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mem_size = mem_size
        self.num_linear_layers = num_linear_layers
        self._sqrt_d = math.sqrt(head_dim)
        # GATES — the only weights (the §7 fine-tune trains these).
        self.write_gate_k = nn.Parameter(torch.zeros(num_linear_layers))
        self.write_gate_v = nn.Parameter(torch.zeros(num_linear_layers))
        self.read_gate = nn.Parameter(torch.ones(num_linear_layers))

    # ------------------------------------------------------------ geometry --
    def state_shape(self) -> torch.Size:
        """(num_heads, mem_size, head_dim) — the M1/M2 state shape."""
        return torch.Size((self.num_heads, self.mem_size, self.head_dim))

    def init_state(self, dtype: torch.dtype = torch.float16,
                   device=None) -> torch.Tensor:
        """A fresh zero M (or M2) state tensor of state_shape()."""
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(
                f"M1M2.init_state: dtype must be a floating torch dtype, "
                f"got {dtype!r}")
        # Infer device from module parameters if not specified
        if device is None:
            device = self.write_gate_k.device
        return torch.zeros(self.state_shape(), dtype=dtype, device=device)

    def extra_repr(self) -> str:
        return (f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"mem_size={self.mem_size}, "
                f"num_linear_layers={self.num_linear_layers}")

    # ---------------------------------------------------------- validation --
    def _check_layer_idx(self, op: str, layer_idx) -> None:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise TypeError(
                f"M1M2.{op}: layer_idx must be a python int, got "
                f"{type(layer_idx).__name__} ({layer_idx!r})")
        if not 0 <= layer_idx < self.num_linear_layers:
            raise ValueError(
                f"M1M2.{op}: layer_idx {layer_idx} out of range for "
                f"num_linear_layers={self.num_linear_layers} "
                f"(valid: [0, {self.num_linear_layers}))")

    def _check_qkv(self, op: str, name: str, x) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"M1M2.{op}: {name} must be a torch.Tensor, got "
                f"{type(x).__name__}")
        if not x.is_floating_point():
            raise TypeError(
                f"M1M2.{op}: {name} must be floating point, got dtype "
                f"{x.dtype}")
        want = f"(B, {self.num_heads}, T, {self.head_dim})"
        if x.dim() != 4 or x.shape[1] != self.num_heads \
                or x.shape[3] != self.head_dim:
            raise ValueError(
                f"M1M2.{op}: {name} must be {want}, got {tuple(x.shape)}")

    def _check_state(self, op: str, name: str, m) -> None:
        if not isinstance(m, torch.Tensor):
            raise TypeError(
                f"M1M2.{op}: {name} must be a torch.Tensor, got "
                f"{type(m).__name__}")
        if not m.is_floating_point():
            raise TypeError(
                f"M1M2.{op}: {name} must be floating point, got dtype "
                f"{m.dtype}")
        want = tuple(self.state_shape())
        if tuple(m.shape) != want:
            raise ValueError(
                f"M1M2.{op}: {name} must be {want} (= state_shape()), got "
                f"{tuple(m.shape)}")

    # ---------------------------------------------------------------- read --
    def read(self, q: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor,
             layer_idx: int) -> torch.Tensor:
        """softmax(q @ M1ᵀ) @ M2, scaled by read_gate[layer_idx] (spec §2.2).

        q: (B, H, T, D); m1/m2: (H, mem, D) -> returns (B, H, T, D) in
        q.dtype. Softmax over the mem dim, scores scaled by 1/√D, softmax
        computed in fp32 for stability (scores cast to float; the result
        comes back to q.dtype). With zero memories the read returns exact
        zeros (softmax(uniform) @ 0 = 0) — the P3 no-op property.
        """
        self._check_layer_idx("read", layer_idx)
        self._check_qkv("read", "q", q)
        self._check_state("read", "m1", m1)
        self._check_state("read", "m2", m2)
        if m1.device != q.device or m2.device != q.device:
            raise ValueError(
                f"M1M2.read: q/m1/m2 must live on one device, got "
                f"{q.device} / {m1.device} / {m2.device}")
        # fp32 score path: (B, H, T, D) @ (H, D, mem) -> (B, H, T, mem)
        scores = torch.matmul(q.to(torch.float32),
                              m1.to(torch.float32).transpose(-1, -2))
        scores = scores / self._sqrt_d
        probs = torch.softmax(scores, dim=-1)          # over the mem dim
        out = torch.matmul(probs, m2.to(torch.float32))   # (B, H, T, D)
        out = out * self.read_gate[layer_idx]          # per-layer scalar gate
        return out.to(q.dtype)

    # --------------------------------------------------------------- write --
    def write(self, k: torch.Tensor, v: torch.Tensor, m1: torch.Tensor,
              m2: torch.Tensor, layer_idx: int,
              positions: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gated ADDITIVE per-token write — path-independent deltas (spec
        §2.2, PROPOSAL D4): for the token at position p
        (slot = p mod mem_size),

            m1_new[h, slot, :] += write_gate_k[layer_idx] * k[b, h, t, :]
            m2_new[h, slot, :] += write_gate_v[layer_idx] * v[b, h, t, :]

        (realized as ONE gated add over the per-slot token SUM — g·Σ_t k_t,
        the same sum in exact arithmetic — so the two-token slot collision
        is bit-exactly g·(k0 + k1)).

        k/v: (B, H, T, D); positions: (T,) integer tensor (default
        arange(T)); returns (m1_new, m2_new), each (H, mem, D) in the
        STATE's dtype. MUST NOT mutate m1/m2 in place — returns fresh
        tensors (functional; the wiring quantizes and stores them via the
        cache). Zero gates ⇒ the states come back bit-identical (P3
        no-op), and the gradient w.r.t. a zero gate still flows (the
        gated add is branch-free — ∂m_new/∂g = Σ_t k_t ≠ 0, which the
        fine-tune needs to open the gates).
        """
        self._check_layer_idx("write", layer_idx)
        self._check_qkv("write", "k", k)
        self._check_qkv("write", "v", v)
        self._check_state("write", "m1", m1)
        self._check_state("write", "m2", m2)
        if k.shape != v.shape:
            raise ValueError(
                f"M1M2.write: k and v must share (B, H, T, D), got "
                f"{tuple(k.shape)} vs {tuple(v.shape)}")
        if (k.device != m1.device or v.device != m1.device
                or m2.device != m1.device):
            raise ValueError(
                f"M1M2.write: k/v/m1/m2 must live on one device, got "
                f"{k.device} / {v.device} / {m1.device} / {m2.device}")
        B, H, T, D = tuple(k.shape)
        if positions is None:
            pos = torch.arange(T, device=k.device)
        else:
            if not isinstance(positions, torch.Tensor):
                raise TypeError(
                    f"M1M2.write: positions must be a torch.Tensor, got "
                    f"{type(positions).__name__}")
            if positions.dim() != 1 or positions.numel() != T:
                raise ValueError(
                    f"M1M2.write: positions must be a (T,) = ({T},) "
                    f"tensor, got shape {tuple(positions.shape)}")
            if (positions.dtype.is_floating_point
                    or positions.dtype.is_complex
                    or positions.dtype == torch.bool):
                raise TypeError(
                    f"M1M2.write: positions must be an integer tensor, "
                    f"got dtype {positions.dtype}")
            if positions.numel() > 0 and bool((positions < 0).any()):
                raise ValueError(
                    f"M1M2.write: positions must be non-negative (token "
                    f"positions), got min "
                    f"{int(positions.min().item())}")
            pos = positions.to(device=k.device, dtype=torch.long)
        slots = torch.remainder(pos, self.mem_size)    # (T,) slot = p mod mem
        m1_new = self._gated_scatter(k, m1, self.write_gate_k[layer_idx],
                                     slots, B, "k")
        m2_new = self._gated_scatter(v, m2, self.write_gate_v[layer_idx],
                                     slots, B, "v")
        return m1_new, m2_new

    def _gated_scatter(self, x: torch.Tensor, state: torch.Tensor,
                       gate: torch.Tensor, slots: torch.Tensor, B: int,
                       name: str) -> torch.Tensor:
        """state + gate · scatter(x) — the additive write for one memory.

        delta[s] = Σ_{tokens at slot s} x[b, :, t, :] (vectorized
        torch.index_add — the additive scatter, a token-major src with a
        (B·T,) index; NO python loop over tokens), then ONE gated add.
        Collision order within a slot is the token order (deterministic
        on CPU; ≤2-token collisions are bit-stable everywhere — fp add
        commutativity).
        """
        H, D = self.num_heads, self.head_dim
        T = x.shape[2]
        dtype = torch.promote_types(x.dtype, state.dtype)
        # token-major (B·T, H, D): token i = b·T + t — matches the index
        src = x.permute(0, 2, 1, 3).reshape(B * T, H, D).to(dtype)
        idx = slots.repeat(B)                             # (B·T,) slot/token
        delta = torch.zeros(self.mem_size, H, D, dtype=dtype,
                            device=state.device)
        delta = torch.index_add(delta, 0, idx, src)       # (mem, H, D)
        delta = delta.permute(1, 0, 2)                    # -> (H, mem, D)
        # one branch-free gated add — gradients flow at gate == 0 (P3)
        return (state.to(dtype) + gate * delta).to(state.dtype)

    # ------------------------------------------------------------- forward --
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                m1: torch.Tensor, m2: torch.Tensor, layer_idx: int,
                positions: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: (read_out, m1_new, m2_new) = read + write.

        The read sees the PRE-write memories (causal: this chunk's
        queries attend to what the memory already held; the write then
        folds this chunk's k/v in for the next forward).
        """
        out = self.read(q, m1, m2, layer_idx)
        m1_new, m2_new = self.write(k, v, m1, m2, layer_idx,
                                    positions=positions)
        return out, m1_new, m2_new


# ------------------------------------------------------- cache glue (§3.2) --
def m1m2_from_cache(cache: "TQCache", module: "M1M2",
                    dtype: torch.dtype = torch.float16
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The §3.2 online loop's read side: (m1, m2) from the cache's codes.

    Reads `cache.read_m1/read_m2` (dequantized TurboQuant codes, kinds
    M1/M2); falls back to `module.init_state()` ZEROS when the cache has
    none — the first forward of the online loop sees zeros and the
    writes open the state (spec §3.2). The returned tensors are
    validated against `module.state_shape()` — a geometry mismatch
    between the installed cache state and the module is a loud error,
    never a silent reshape.

    TQCache is imported lazily (function body) so this module stays
    importable without transformers.
    """
    from tq_cache import TQCache  # lazy: tq_cache pulls transformers
    if not isinstance(cache, TQCache):
        raise TypeError(
            f"m1m2_from_cache: cache must be a TQCache, got "
            f"{type(cache).__name__}")
    if not isinstance(module, M1M2):
        raise TypeError(
            f"m1m2_from_cache: module must be an M1M2, got "
            f"{type(module).__name__}")
    m1 = cache.read_m1(dtype=dtype)
    m2 = cache.read_m2(dtype=dtype)
    device = None
    if m1 is not None:
        device = m1.device
    elif m2 is not None:
        device = m2.device
    want = tuple(module.state_shape())
    out = []
    for name, t in (("m1", m1), ("m2", m2)):
        if t is None:
            t = module.init_state(dtype=dtype, device=device)
        elif tuple(t.shape) != want:
            raise ValueError(
                f"m1m2_from_cache: cached {name} has shape "
                f"{tuple(t.shape)}, but this M1M2's state_shape is {want} "
                f"— the cache state and the module geometry disagree "
                f"(num_heads/mem_size/head_dim)")
        out.append(t)
    return out[0], out[1]


def push_to_cache(cache: "TQCache", m1: torch.Tensor, m2: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The §3.2 online loop's write side: quantize-on-write.

    `cache.update_m1(m1)` / `cache.update_m2(m2)` quantize the full
    memories to TurboQuant codes (the persisted truth — the cache never
    holds fp16). Returns the dequantized ROUND-TRIP states (the state
    the next forward must see; what you wrote is what you read back).

    The D4 delta protocol never sums these codes — deltas are taken
    between dequantized states and the install path requantizes the SUM.
    """
    from tq_cache import TQCache  # lazy: tq_cache pulls transformers
    if not isinstance(cache, TQCache):
        raise TypeError(
            f"push_to_cache: cache must be a TQCache, got "
            f"{type(cache).__name__}")
    for name, t in (("m1", m1), ("m2", m2)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"push_to_cache: {name} must be a torch.Tensor, got "
                f"{type(t).__name__}")
        if not t.is_floating_point():
            raise TypeError(
                f"push_to_cache: {name} must be floating point, got dtype "
                f"{t.dtype}")
    if tuple(m1.shape) != tuple(m2.shape):
        raise ValueError(
            f"push_to_cache: M1/M2 are twins — shapes must match, got "
            f"{tuple(m1.shape)} vs {tuple(m2.shape)}")
    m1_rt = cache.update_m1(m1)
    m2_rt = cache.update_m2(m2)
    return m1_rt, m2_rt
