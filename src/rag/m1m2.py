"""m1m2.py — the two global memory matrices M1/M2 (SPECIFICATION.md §2.2).

TWO tensors for the whole model — shared across ALL 24 linear-attention
layers, NOT per-layer (spec §2.2: "TWO for the whole model"):

  M1 (global key-memory):    (num_heads, mem_size, head_dim) = (32, 128, 128)
  M2 (global value-memory):  same shape

  read  (spec §2.2):  softmax(q @ M1ᵀ) @ M2  — folded into the layer's
                      output path by the W3.2 wiring (modeling.py).
  write (spec §2.2):  gated ADDITIVE per-token scatter. delta_M1/M2 are
                      path-independent per CALL and chunk deltas sum
                      losslessly at install (PROPOSAL D4 / SPEC §6:
                      install sums DEQUANTIZED M1/M2 deltas, never the
                      codes).

W21 — THE ADDRESS (the position-lock, decomposed): the v1 write chose
the slot by POSITION (slot = p mod mem_size), which made a pair-side's
state delta simply its (key/value) SEQUENCE: at T < mem_size every slot
holds exactly one token, so the §7 / §4 state-cosine between two sides
is the POSITION-ALIGNED key correlation — a surface-form lock, not a
content similarity. The box's W20.15 instruments predicted the ceiling
(cos_pos riding a stable ~+0.36 common-direction pedestal, spread
~0.12-0.13, no trend; the low-lexical-overlap buckets flat) and the
W20.15 close note carried exactly this wave. Every reference design for
key-value memories addresses by CONTENT, never by position —
Product-Key Memory (Lample et al. 2019, arXiv:1907.05242: the slot is
chosen by key-prototype similarity), Memorizing Transformers (Wu et
al. 2022, arXiv:2203.08913: kNN retrieval into a (key, value) store by
the key itself), the fast-weight/linear-attention lineage the model's
own 24 S-states follow (the delta-rule write is content-addressed by
k). The v1 position-addressed scatter is the odd one out; W21 moves
the WRITE ADDRESS to the key's content and keeps everything else.

THE W21 WRITE (default `write_addr="content"`):

  1. CENTER (default `center_write=True`): k̃ = k − mean_t k (per
     (batch, head), over the call's tokens; v likewise). The v1 box run
     measured a stable ~+0.36 cosine between RANDOM cross pairs — the
     well-known representation-anisotropy pedestal (the keys' shared
     "typical direction"; the canonical fix is centering/whitening, Su
     et al. 2021, "Whitening Sentence Representations for Better
     Semantics"). The W19 delta-loss already removes the SYSTEM state's
     common component; this removes the TOKEN-key common component at
     the source, train and serve identically (it is inside the write).
     A centered bag has zero slot-sum BY CONSTRUCTION, so the state
     cosine becomes a CONTENT covariance, not a length × pedestal
     artifact. Side effect (deliberate): a T=1 (decode) call centers to
     the zero vector and writes nothing — the memory freezes at
     prefill end, which is what the retrieval state should be (the v1
     counter hack only stopped decode overwriting slot 0).
  2. ADDRESS: slot = simhash(k̃) — the sign pattern of the
     Hadamard-mixed centered key (Charikar 2002 simhash over a FIXED,
     mathematically-deterministic Sylvester Hadamard — no learned
     prototypes, no RNG, bit-identical on every box). `addr_blocks`
     (κ, default 4) independent sign-blocks give each token κ slots:
     the collision probability of two keys with cosine c is
     1 − (1 − p(c)^b)^κ with p(c) = 1 − arccos(c)/π, b = log2(mem/κ) —
     near-identical contextual tokens collide with high probability,
     unrelated ones almost never. The state cosine over the resulting
     content-bags is a token-pair kernel: shared CONTENT matches
     anywhere in the two texts (position-free), which is exactly the
     spec §7 objective ("similar texts -> similar states, retrieval by
     content overlap") and what the v1 position lock could not
     express at any gate setting.
  3. GATED ADD (unchanged shape): m_new = m + g[L, h] · scatter(k̃) —
     ONE branch-free gated add per call; the write gates stay the only
     trained weights.

The write gates are now PER-(LAYER, HEAD) — (num_linear_layers,
num_heads) each. The v1 per-layer scalar could only reweight layers;
the flattened state cosine weights heads by ‖k̃_h‖², and a 2-5×
cross-head scale spread (the natural variation of the per-head key
deviations) is a 4-25× weight imbalance the layer scalar cannot touch.
Per-head gates are the granularity the memory literature uses
(Memorizing Transformers' read gate is per-dimension); they remain
"the gates" — the M1M2 module's only parameters, 3 vectors of
(L·H, L·H, L) floats. read_gate stays per-LAYER (it scales the layer's
read output, one scalar per output path). Gates .npz artifacts carry
the shapes; a v1 (L,) artifact is REFUSED loudly by load_gates (the
addressing changed the semantics — a v1-trained gate file is
meaningless under content addressing and must be retrained).

`write_addr="positional"` restores the v1 slot rule EXACTLY (the same
token-major scatter code path, bit-identical, kept as the A/B and the
pinned reference the W3.3 tests assert); `center_write=False` restores
raw (uncentered) keys/values. The two knobs are orthogonal on purpose
— the driver's held-out ladder sweeps the four corners plus the
κ ladder, all offline on the captured k/v (no re-prefill per config).

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
returns the state bit-identical in BOTH addressing modes; read of zero
memories returns exact zeros). The write is deliberately ONE
branch-free gated add (`m + g · Σ_t k̃_t`): a `if gate == 0: skip`
short-circuit would zero the LOCAL Jacobian ∂m_new/∂g = Σ_t k̃_t
exactly where P3's fine-tune has to start opening the gates. W19
caveat (measured on the box, proven in m1m2_finetune's docstring): the
P3 point is nonetheless an EXACT SADDLE of the §7 contrastive
objective — at g=0 every pair-side replays to the SAME (zero) state,
so the LOSS-level gradient dL/dg = <dL/dm, Δ> is identically zero and
the gates never open from there BY THEMSELVES; the trainer
(GatesTrainConfig.gate_init) opens them explicitly before step 0. The
P3 deployment default stays — untrained ingest is a no-op.

Online loop (SPEC §3.2, PROPOSAL D4): `m1m2_from_cache` reads the
cache's m1/m2 codes, falling back to `init_state()` zeros when the
cache has none (the first forward sees zeros; the writes open the
state); `push_to_cache` quantize-on-writes through
`cache.update_m1/update_m2` and hands back the dequantized round-trip
— the state the NEXT forward must see (the codes are the persisted
truth; the cache never holds fp16).

Determinism contract (W21): the slot is a pure function of the input
k bits — center (fp32) -> fixed Sylvester Hadamard mixing (fp32) ->
sign bits -> integer slot. The capture/replay pair recomputes it from
the SAME captured k (bit-identical inputs, same device, same shapes)
so the W18 exact-gradient replay stays exact: write_deltas reproduces
the forward's scatter bit-for-bit (shared code). The mixing matrix is
built from the Sylvester recursion (±1 entries, exact in any dtype) —
NOT a seeded RNG — so every box derives the same matrix; it is a plain
attribute, never a registered buffer (it must not enter state_dict or
the gates artifact; geometry pins it).

Numerics: the softmax read path runs in fp32 (scores scaled by 1/√D
over the mem dim, per the read contract); the write's centering and
mixing run in fp32 (the scatter promotes with the state's dtype and
casts back, so uniform-dtype calls do no hidden rounding beyond the
native dtype). k/v are cast up only when their dtype is wider than the
state's; the gated add accumulates in the promoted dtype and casts
back.

Composability notes (spec §2.2 "Summing is lossless"): the additive
write is exact in real arithmetic for any regrouping of the CALL's
multiset of (token -> slot) pairs, and per-CALL deltas (the D4 install
unit) are deterministic functions of that call's k/v in BOTH addressing
modes — chunk deltas sum losslessly at install exactly as before.
Within one call, the v1 positional mode is BIT-exact under regrouping
(per-slot sums formed before the single gated add); the content mode
re-groups tokens by content hash instead of position — the same
per-slot-sum property holds (index_add over the call's tokens), and
the per-call centering means merging two CALLS into one longer call
changes each call's own center (call-granular, documented — install
sums CALL deltas, never merged calls).

This module imports torch only at module top (NO transformers) —
`TQCache` is imported lazily inside the cache helpers; `import _paths`
first (house sys.path anchor, see src/rag/_paths.py).

Refs: SPECIFICATION.md §2.2 (the memories), §3.2 (online loop),
§5 (M1/M2 codes in the snapshot), §6 (install sums dequantized deltas),
§7 (the fine-tune trains the gates); PROPOSAL.md P3 (zero-init write
gates), D4 (the delta protocol), Phase 3/4; W21 (this wave — the
content address + centering + per-head gates); Lample et al. 2019
(arXiv:1907.05242), Wu et al. 2022 (arXiv:2203.08913), Su et al. 2021
(whitening/centering), Charikar 2002 (simhash).
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

_ADDR_MODES = ("content", "positional")


def _pos_int(name: str, value) -> int:
    """Loud positive-int config validation."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"M1M2: {name} must be a python int, got "
            f"{type(value).__name__} ({value!r})")
    if value < 1:
        raise ValueError(f"M1M2: {name} must be >= 1, got {value}")
    return value


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1)."""
    return 1 << (n - 1).bit_length()


def _sylvester_hadamard(n: int) -> torch.Tensor:
    """The (n, n) Sylvester Hadamard matrix, n a power of two, EXACT ±1
    entries (deterministic on every box — no RNG, no version drift)."""
    if n < 1 or n != _next_pow2(n):
        raise ValueError(f"_sylvester_hadamard: n must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h


class M1M2(nn.Module):
    """The two global memories (SPECIFICATION.md §2.2).

    The WEIGHTS here are the GATES ONLY (spec §7 / PROPOSAL Phase 4
    fine-tunes exactly these):

      * ``write_gate_k`` / ``write_gate_v`` — (num_linear_layers,
        num_heads) each, ZERO-init (PROPOSAL P3: M1/M2 start as no-ops
        — the untrained model's behavior is bit-unchanged in both
        addressing modes; the fine-tune opens them). Per-(layer, head)
        since W21: the flattened state cosine weights heads by
        |k̃_h|^2 and only a per-head gate can rebalance them.
      * ``read_gate`` — (num_linear_layers,), ONE-init (reads of the
        zero memories are exact zeros anyway, so P3's bit-unchanged
        property holds from both the read and the write side).

    Addressing knobs (plain attributes, settable post-init for the
    driver's A/B + held-out ladder; validated at every write):

      * ``write_addr`` — "content" (W21 default: slot = simhash of the
        centered key) or "positional" (the v1 slot = p mod mem_size,
        bit-identical v1 semantics).
      * ``center_write`` — True (W21 default: write k − mean_t k, the
        anisotropy-pedestal removal) or False (raw keys/values, v1).
      * ``addr_blocks`` — κ, the number of independent sign-blocks
        (each token writes κ slots; 4 by default at mem 1024 ->
        b = 8 sign bits per block).

    There are NO m1/m2 buffers on this module: the memories are RUNTIME
    CACHE STATE — zero at start, accumulating during prefill, quantized
    / snapshotted / installed through the TQCache (spec §3.2/§5; see
    `m1m2_from_cache` / `push_to_cache`).

    Geometry: the spec writes the memories as (1, 32, mem_size, 128) —
    the leading 1 is the vacuous batch dim (the memories are GLOBAL, one
    per model, read by every batch row); this module's state shape drops
    it: (num_heads, mem_size, head_dim).
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128,
                 mem_size: int = 128, num_linear_layers: int = 24,
                 write_addr: str = "content", center_write: bool = True,
                 addr_blocks: int = 4):
        num_heads = _pos_int("num_heads", num_heads)
        head_dim = _pos_int("head_dim", head_dim)
        mem_size = _pos_int("mem_size", mem_size)
        num_linear_layers = _pos_int("num_linear_layers", num_linear_layers)
        if not isinstance(write_addr, str) or write_addr not in _ADDR_MODES:
            raise ValueError(
                f"M1M2: write_addr must be one of {_ADDR_MODES}, got "
                f"{write_addr!r}")
        if not isinstance(center_write, bool):
            raise TypeError(
                f"M1M2: center_write must be a bool, got "
                f"{type(center_write).__name__} ({center_write!r})")
        addr_blocks = _pos_int("addr_blocks", addr_blocks)
        if addr_blocks > 8:
            raise ValueError(
                f"M1M2: addr_blocks must be <= 8 (the sign-block budget; "
                f"the ladder tops out at 8), got {addr_blocks}")
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mem_size = mem_size
        self.num_linear_layers = num_linear_layers
        self.write_addr = write_addr
        self.center_write = center_write
        self.addr_blocks = addr_blocks
        self._sqrt_d = math.sqrt(head_dim)
        # GATES — the only weights (the §7 fine-tune trains these).
        self.write_gate_k = nn.Parameter(
            torch.zeros(num_linear_layers, num_heads))
        self.write_gate_v = nn.Parameter(
            torch.zeros(num_linear_layers, num_heads))
        self.read_gate = nn.Parameter(torch.ones(num_linear_layers))
        # The fixed sign-mixing for the content address: the first
        # head_dim rows of the Sylvester Hadamard at width
        # next_pow2(head_dim) — a plain attribute, NEVER a registered
        # buffer (it must not enter state_dict / the gates artifact;
        # the geometry pins it bit-exactly on every box). Per-device
        # copies cache lazily in _mix_dev.
        dp = _next_pow2(head_dim)
        self._mix_full = _sylvester_hadamard(dp)[:head_dim, :].contiguous()
        self._mix_dev = {}
        # fail fast on impossible address geometry (loud, at init too)
        self._addr_geometry()

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
                f"num_linear_layers={self.num_linear_layers}, "
                f"write_addr={self.write_addr!r}, "
                f"center_write={self.center_write}, "
                f"addr_blocks={self.addr_blocks}")

    def _addr_geometry(self) -> Tuple[int, int, int]:
        """Validate the addressing config and derive (blocks, stride,
        bits): stride = mem // blocks (the per-block slot range), bits =
        sign bits per block (capped by the stride's log2 and by the
        mixing width // blocks). Loud on impossible configs."""
        if not isinstance(self.write_addr, str) \
                or self.write_addr not in _ADDR_MODES:
            raise ValueError(
                f"M1M2: write_addr must be one of {_ADDR_MODES}, got "
                f"{self.write_addr!r}")
        if not isinstance(self.center_write, bool):
            raise TypeError(
                f"M1M2: center_write must be a bool, got "
                f"{type(self.center_write).__name__}")
        blocks = _pos_int("addr_blocks", self.addr_blocks)
        if blocks > 8:
            raise ValueError(
                f"M1M2: addr_blocks must be <= 8, got {blocks}")
        dp = _next_pow2(self.head_dim)
        if blocks > dp:
            raise ValueError(
                f"M1M2: addr_blocks {blocks} needs >= {blocks} mixed "
                f"coordinates, head_dim {self.head_dim} gives {dp} — "
                f"reduce addr_blocks or raise head_dim")
        stride = max(1, self.mem_size // blocks)
        bits = max(1, min(stride.bit_length() - 1, dp // blocks))
        return blocks, stride, bits

    def _get_mix(self, device) -> torch.Tensor:
        """The (head_dim, next_pow2(head_dim)) sign-mixing on `device`
        (cached per device; the CPU master is the pinned ±1 truth)."""
        key = str(device)
        cached = self._mix_dev.get(key)
        if cached is None:
            cached = self._mix_full.to(device)
            self._mix_dev[key] = cached
        return cached

    def _centered(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, D) -> fp32, per-(b, h) token-mean removed when
        center_write (the anisotropy-pedestal removal, W21); the raw
        fp32 cast otherwise."""
        xc = x.to(torch.float32)
        if self.center_write:
            xc = xc - xc.mean(dim=2, keepdim=True)
        return xc

    def _content_slots(self, kc: torch.Tensor) -> torch.Tensor:
        """(B, H, T, D) centered fp32 keys -> (B, H, T, K) slot indices.

        The W21 content address: y = kc @ Hadamard (fixed ±1 mixing);
        block j reads the sign bits of y[..., j*b:(j+1)*b] as an
        integer bucket; slot_j = (j*stride + bucket) mod mem_size. A
        pure deterministic function of the key bits (simhash — two keys
        with cosine c share a block's bucket with probability
        1 - (1 - p^b)^K, p = 1 - arccos(c)/pi)."""
        blocks, stride, bits = self._addr_geometry()
        B, H, T, _D = kc.shape
        y = torch.matmul(kc, self._get_mix(kc.device))     # (B, H, T, dp)
        weights = (1 << torch.arange(bits, device=kc.device,
                                     dtype=torch.int64))   # (bits,)
        slots = torch.empty(B, H, T, blocks, dtype=torch.int64,
                            device=kc.device)
        for j in range(blocks):
            bits_t = (y[..., j * bits:(j + 1) * bits] > 0).to(torch.int64)
            bucket = (bits_t * weights).sum(dim=-1)        # (B, H, T)
            slots[..., j] = (j * stride + bucket) % self.mem_size
        return slots

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
        zeros (softmax(uniform) @ 0 = 0) — the P3 no-op property. W21
        note: with content-addressed M1 the slots hold content clusters
        (sums of similar-context keys), so the softmax matches a query
        against CONTENT, the Memorizing-Transformers retrieval semantics;
        with a centered M2 the retrieved value is that cluster's centered
        deviation — the q's own common direction dots to ~0 against the
        zero-slot-mean state, so the read self-centers.
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
        """Gated ADDITIVE per-token write — path-independent per-call
        deltas (spec §2.2, PROPOSAL D4).

        W21 CONTENT mode (default): per (b, h), per token t,
        k̃_t = k_t − mean_t k (center_write) and its slot is
        simhash(k̃_t) (addr_blocks independent sign-blocks of the
        Hadamard mixing); the scatter sums per slot, then ONE gated add

            m1_new[h, s, :] += write_gate_k[layer_idx, h] * Σ_{t→s} k̃_t
            m2_new[h, s, :] += write_gate_v[layer_idx, h] * Σ_{t→s} ṽ_t

        — the delta is a function of the call's OWN tokens only (the
        D4 per-call unit). A T=1 call centers to zero and writes
        nothing (decode freezes the memory — deliberate).

        POSITIONAL mode (`write_addr="positional"`, the v1 A/B): slot =
        p mod mem_size, the bit-exact v1 token-major scatter.

        k/v: (B, H, T, D); positions: (T,) integer tensor (validated
        in both modes; USED only by the positional mode); returns
        (m1_new, m2_new), each (H, mem, D) in the STATE's dtype. MUST
        NOT mutate m1/m2 in place — returns fresh tensors (functional;
        the wiring quantizes and stores them via the cache). Zero gates
        ⇒ the states come back bit-identical (P3 no-op, both modes),
        and the LOCAL gradient w.r.t. a zero gate still flows (the
        gated add is branch-free — ∂m_new/∂g = Σ_t k̃_t ≠ 0); NOTE
        (W19): the §7 InfoNCE sits at an exact saddle at g=0 (dL/dm = 0
        on the identical states), so the TRAINER opens the gates via
        GatesTrainConfig.gate_init — see m1m2_finetune's docstring.
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
        pos = self._resolve_positions(positions, T, k.device)
        self._addr_geometry()                       # loud on bad knobs
        if self.write_addr == "content":
            kc = self._centered(k)
            vc = self._centered(v)
            slots = self._content_slots(kc)         # keys address BOTH
            m1_new = self._gated_scatter_content(
                kc, m1, self.write_gate_k[layer_idx], slots)
            m2_new = self._gated_scatter_content(
                vc, m2, self.write_gate_v[layer_idx], slots)
        else:
            slots = torch.remainder(pos, self.mem_size)    # (T,) slot=p mod mem
            m1_new = self._gated_scatter(k, m1, self.write_gate_k[layer_idx],
                                         slots, B, "k")
            m2_new = self._gated_scatter(v, m2, self.write_gate_v[layer_idx],
                                         slots, B, "v")
        return m1_new, m2_new

    def write_deltas(self, k: torch.Tensor, v: torch.Tensor,
                     layer_idx: int,
                     positions: Optional[torch.Tensor] = None,
                     state: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(Δ1, Δ2) — the RAW per-call deltas, state-independent (the §2.2
        additive property: m_new = m + g·Δ with Δ the pure scatter of
        the centered/raw k/v). W18: the memory-safe replay's backward
        recomputes these ONE CALL AT A TIME instead of retaining the
        24-boundary-state chain — bit-identical to the deltas the write
        path itself adds (the same scatter code, shared, no drift; the
        content mode recomputes the SAME deterministic slots from the
        same captured k — the exactness contract holds).

        `state` (any state-shaped tensor, e.g. the m_init) fixes the
        promotion dtype + device of the returned deltas; positions
        resolve exactly as in write() (used by the positional mode
        only)."""
        self._check_layer_idx("write_deltas", layer_idx)
        self._check_qkv("write_deltas", "k", k)
        self._check_qkv("write_deltas", "v", v)
        if state is not None:
            self._check_state("write_deltas", "state", state)
        if k.shape != v.shape:
            raise ValueError(
                f"M1M2.write_deltas: k and v must share (B, H, T, D), got "
                f"{tuple(k.shape)} vs {tuple(v.shape)}")
        ref = state if state is not None else k
        B, H, T, D = tuple(k.shape)
        pos = self._resolve_positions(positions, T, k.device)
        self._addr_geometry()                       # loud on bad knobs
        if self.write_addr == "content":
            kc = self._centered(k)
            vc = self._centered(v)
            slots = self._content_slots(kc)
            d1 = self._scatter_delta_content(kc, slots, ref)
            d2 = self._scatter_delta_content(vc, slots, ref)
        else:
            slots = torch.remainder(pos, self.mem_size)
            d1 = self._scatter_delta(k, slots, B, ref)
            d2 = self._scatter_delta(v, slots, B, ref)
        return d1, d2

    def _resolve_positions(self, positions, T: int,
                           device: torch.device) -> torch.Tensor:
        """The write-positions contract (shared by write/write_deltas):
        default arange(T); loud validation otherwise. W20.4.4: the
        min-position guard is the tensor-space (positions >= 0).all()
        check — gather-free, no host-scalar read; the one host value on
        the failure path is the raise's own message edge. W21: the
        CONTENT mode ignores the positions for addressing (the slot is
        the key's content hash) — the validation still runs (the v1
        contract, and the wiring keeps passing real positions).
        """
        if positions is None:
            return torch.arange(T, device=device)
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
        if positions.numel() > 0 and not (positions >= 0).all():
            # W20.4.4: tensor-space guard (the legacy any() bool plus the
            # min host-scalar read are gone) — on integer dtypes
            # "not all >= 0" is exactly "any < 0", the legacy semantic;
            # the message edge below is the only host read, and only on
            # the failure path.
            raise ValueError(
                f"M1M2.write: positions must be non-negative (token "
                f"positions), got min "
                f"{int(positions.min())}")
        return positions.to(device=device, dtype=torch.long)

    def _scatter_delta(self, x: torch.Tensor, slots: torch.Tensor, B: int,
                       state: torch.Tensor) -> torch.Tensor:
        """delta (H, mem, D) in promote(x, state) — the v1 POSITIONAL
        additive scatter (token-major index_add, NO python loop).
        Bit-identical to the pre-W21 write path (the pinned reference).
        Shared by the gated write and the W18 write_deltas (the replay
        backward's recompute is the SAME code, hence bit-identical)."""
        H, D = self.num_heads, self.head_dim
        T = x.shape[2]
        dtype = torch.promote_types(x.dtype, state.dtype)
        src = x.permute(0, 2, 1, 3).reshape(B * T, H, D).to(dtype)
        idx = slots.repeat(B)                             # (B·T,) slot/token
        delta = torch.zeros(self.mem_size, H, D, dtype=dtype,
                            device=x.device)
        delta = torch.index_add(delta, 0, idx, src)       # (mem, H, D)
        return delta.permute(1, 0, 2)                    # -> (H, mem, D)

    def _gated_scatter(self, x: torch.Tensor, state: torch.Tensor,
                       gate: torch.Tensor, slots: torch.Tensor, B: int,
                       name: str) -> torch.Tensor:
        """state + gate · scatter(x) — the v1 POSITIONAL additive write
        for one memory (gate now the (H,) per-head row; a uniform row
        reproduces the v1 per-layer scalar bit-exactly).

        delta[s] = Σ_{tokens at slot s} x[b, :, t, :] (the shared
        _scatter_delta), then ONE gated add. Collision order within a
        slot is the token order (deterministic on CPU; ≤2-token
        collisions are bit-stable everywhere — fp add commutativity).
        """
        dtype = torch.promote_types(x.dtype, state.dtype)
        delta = self._scatter_delta(x, slots, B, state)
        # one branch-free gated add — gradients flow at gate == 0 (P3)
        return (state.to(dtype)
                + gate.to(dtype).view(-1, 1, 1) * delta).to(state.dtype)

    def _scatter_delta_content(self, xc: torch.Tensor, slots: torch.Tensor,
                               state: torch.Tensor) -> torch.Tensor:
        """delta (H, mem, D) in promote(xc, state) — the W21 CONTENT
        scatter: per (head, block) index_add into the (H·mem, D) flat
        layout (head_offset + slot), the per-slot token sums formed by
        index_add exactly as in the positional path. Bit-identical
        between write and write_deltas (shared code — the W18
        exactness contract)."""
        B, H, T, D = xc.shape
        blocks = slots.shape[-1]
        dtype = torch.promote_types(xc.dtype, state.dtype)
        src = xc.reshape(B * H * T, D).to(dtype)
        head_off = (torch.arange(H, device=xc.device,
                                 dtype=slots.dtype)
                    * self.mem_size).view(1, H, 1)
        delta = torch.zeros(H * self.mem_size, D, dtype=dtype,
                            device=xc.device)
        for j in range(blocks):
            idx = (slots[..., j] + head_off).reshape(-1)   # (B·H·T,)
            delta = torch.index_add(delta, 0, idx, src)
        return delta.view(H, self.mem_size, D)

    def _gated_scatter_content(self, xc: torch.Tensor, state: torch.Tensor,
                               gate: torch.Tensor,
                               slots: torch.Tensor) -> torch.Tensor:
        """state + gate · scatter(xc) — the W21 CONTENT additive write
        for one memory (xc the CENTERED fp32 operand, gate the (H,)
        per-head row). One branch-free gated add (P3 gradients flow at
        gate == 0)."""
        delta = self._scatter_delta_content(xc, slots, state)
        dtype = delta.dtype
        return (state.to(dtype)
                + gate.to(dtype).view(-1, 1, 1) * delta).to(state.dtype)

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
    # W18: the states follow the MODULE's device (the wiring's consumer) —
    # a codes-only cache may dequant to CPU while the module lives on
    # CUDA; init_state already defaults there, the read side moves now.
    want_dev = module.write_gate_k.device
    if m1 is not None and m1.device != want_dev:
        m1 = m1.to(want_dev)
    if m2 is not None and m2.device != want_dev:
        m2 = m2.to(want_dev)
    want = tuple(module.state_shape())
    out = []
    for name, t in (("m1", m1), ("m2", m2)):
        if t is None:
            t = module.init_state(dtype=dtype, device=want_dev)
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
