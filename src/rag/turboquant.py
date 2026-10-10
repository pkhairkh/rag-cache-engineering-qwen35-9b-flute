"""turboquant.py — the TurboQuant quantizer core (SPECIFICATION.md §3).

Online, data-oblivious vector quantization (arXiv:2504.19874), mapped to
this repo's cache tensors per PROPOSAL.md §1–§2:

  quant(x):    norm (fp, stored) → r = x/‖x‖ → FHT rotation (y = r @ T,
               the repo's `flute_extended/fht.py`, O(d log d)) → per-set
               scalar Lloyd-Max encode (`codebooks.py`) → packed uint8 codes
  dequant:     unpack → centroids → adjoint FHT (y @ Tᵀ) → × norm

Quantization units (PROPOSAL D2 / SPECIFICATION §3.3–§5):

  kind    d (dims)   units per chunk   FHT stages
  S       524,288    24 (per layer)    19 (2^19, single segment)
  conv    32,768     24 (per layer)    15
  M1      524,288    1                 19
  M2      524,288    1                 19

The 3.5-bit recipe (§3.3, PROPOSAL §1.3/D2): a fixed 50/50 coordinate
split — the first half of coordinates at 3 bits, the second half at 4 —
(no calibration, data-oblivious; the paper's outlier-split variant is the
W9 flag, never the default). Per unit one fp32 norm scalar is stored and
never quantized (the paper's recipe).

Rotation sharing (PROPOSAL D3): ONE rotation instance per tensor kind —
every quantization of a given kind (system prompt, chunk deltas, query,
install sums) uses the same FHT sign vector, fixed seed per kind, so
dequantized deltas live in one shared rotated frame and the IVFADC index
sees a stable original-frame vector. Seeds persist in the index
side-metadata.

Entropy coding is deliberately skipped (paper: ~5% at b=4 — not worth it).
Codes are b-bit indices + one fp norm. Lloyd-Max is not additive:
Q(a+b) ≠ Q(a)+Q(b) — install math (SPECIFICATION §6) therefore requantizes
SUMS of dequantized vectors (see rag/install.py); this module intentionally
offers no code-plus-code operation.

THE W15 OUTLIER SPLIT (the paper's OWN non-integer-bit recipe — the
conv kind's default; see TurboQuant(group=...)): arXiv:2504.19874 §5.3
quantizes KV-cache-like tensors by "splitting channels into outlier and
non-outlier sets, and applying two independent instances of TurboQuant
to each, allocating higher bit precision to outliers". The conv window
(channel, tap) layout quantizes through TWO sub-instances: the top-k
energy channels (k = round(frac·n_ch)) at bits_hi with their OWN
power-of-two full rotation (seed+555) and own fp norm; the rest at
bits_lo with the kind's seed. The membership mask (packed channel
bits), the group and the second norm ride in TQCodes — the codes are
self-describing and dequant routes on codes.partition, so pre-W15
partition="half" snapshots decode UNCHANGED (through the flat twin) and
legacy-only paths (hooks/index) read split codes exactly (through the
split twin). Measured at production conv geometry (channel-structured
windows): ~3x lower write-path rel-MSE than the fixed coordinate
half-split at the SAME effective bits — the fixed split hands the 4-bit
half to a fixed coordinate range that ignores the channel-energy
structure, while the paper's split spends them where the energy is.
group=1 (S/M1/M2 — no channel structure; the full 2^19 rotation already
mixes everything) keeps the legacy flat split, BIT-IDENTICAL.

THE W9.2 `qjl` FLAG (P7 / D1's flagged prod-variant A/B — DEFAULT OFF):
`TurboQuant(..., qjl=True)` additionally computes the structured QJL
residual sketch (PROPOSAL §1.2, the paper's Alg. 2 "TurboQuant_prod"):

  r   = x − dequant_mse(codes)          (ORIGINAL-frame residual; the
                                         rotated-frame form fht(x/‖x‖) − y
                                         is identical by linearity — sign()
                                         is scale-invariant either way)
  γ   = ‖r‖₂                            (one fp32 scalar, never quantized)
  qjl = sign(FHT(r/γ, seed+7777))       (d int8 ±1 — the SECOND
                                         deterministic sign draw, own seed
                                         = the kind's seed + QJL_SEED_OFFSET,
                                         same d as the rotation)
  dequant: x̃ = x̃_mse + (√(π/2)/√d)·γ·FHT_adjoint(qjl)

EXACT FORMULATION (documented per the A/B contract): PROPOSAL §1.2 writes
the paper's estimator as x̃ = x̃_mse + (√(π/2)/d)·γ·Sᵀ·qjl with S iid
N(0,1) d×d — unit-variance entries, rows of norm ~√d. Our substitution
(PROPOSAL §1.5 item 4: "the QJL projection is likewise substituted with a
structured sketch") is S := √d·FHT(d, seed+7777): the sign-flipped
Hadamard scaled to the SAME per-entry variance, whose second moments
match the dense Gaussian exactly (E[S_ik·S_jl] = δ_ij·δ_kl over the ±1
draw). Since sign() is scale-invariant the SKETCH is computed on the
unscaled orthogonal FHT, and the adjoint carries the √d:

  (√(π/2)/d)·γ·Sᵀ·qjl  ==  (√(π/2)/√d)·γ·FHT_adjoint(qjl)

Unbiasedness of ⟨y, x̃⟩ (E = ⟨y, x⟩) survives any JL-valid projection with
the right moments to CLT accuracy — the FHT's projection coordinates are
equal-weight ±1 sums, Gaussian-marginal for generic x — and the VARIANCE
CONSTANT of the structured substitute is exactly what the GPU Phase-5 gate
re-measures (PROPOSAL §1.5(4), D1's A/B decision rule). Two deliberate A/B
simplifications vs. the paper's Alg. 2, both pinned by PROPOSAL D1: (i)
the MSE base layer keeps the SAME bit budget (the paper drops it to b-1;
here the A/B isolates the residual bit at the production budget — "one
extra bit per coordinate", so the effective rate is b+1); (ii) the
retrieval-plane consumer (IVFADC-side IP estimation) is the GPU box's
Phase-5 decision — index.py is deliberately untouched.

Round-trip note: with qjl=True the MSE typically IMPROVES (the residual
is compensated in expectation: E‖r−c‖² ≈ (π/2−1)·γ² vs. γ², with
‖c‖ = √(π/2)·γ exactly); the parity gate covers qjl=False only.
Serialization is BACKWARD-COMPATIBLE: codes quantized without the flag
carry qjl_signs/gamma == None and write NONE of the new keys; from_arrays
loads old npz/snapshots (no qjl keys) with the fields None.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple
import math

import numpy as np
import torch

import _paths  # noqa: F401  (anchors src/rag, src/scripts, src/flute_extended)
import codebooks
import fht

__all__ = [
    "KINDS", "SEEDS", "TQCodes", "TurboQuant", "get_quantizer",
    "pack_bits", "unpack_bits",
]

# ---------------------------------------------------------------- kinds ----
# (kind -> (d, seed)); the seeds are the D3 rotation keys — persisted in the
# index side-metadata by rag/index.py; do not change them once an index
# exists (frame drift would silently corrupt retrieval).
# NOTE (W15): KINDS["conv"]'s canonical d (32,768) is the HISTORICAL
# power-of-two unit. The production conv window (Qwen3.5 in_proj Q+V:
# 1×6144×4 = 24,576) resolves through the CUSTOM-d path (kind="custom",
# seed 202 preserved — the D3 frame contract), and quantizes through the
# paper's outlier split (TurboQuant(group=kernel); see the module
# docstring). The canonical entry is kept UNCHANGED so pre-W15
# 24,576-dim "custom" codes still pass _check_codes against the custom
# path (flipping the canonical d would silently change the resolved
# kind string and strand every existing snapshot).
KINDS: Dict[str, Tuple[int, int]] = {
    "S":   (524_288, 101),   # per-layer recurrent state (spec §2.1)
    "conv": (32_768, 202),   # per-layer conv state (spec §2.3; see NOTE)
    "M1":  (524_288, 303),   # global key-memory (spec §2.2)
    "M2":  (524_288, 404),   # global value-memory (spec §2.2)
}
SEEDS: Dict[str, int] = {k: v[1] for k, v in KINDS.items()}

# W9.2 qjl flag: the QJL sketch's own sign-draw seed offset, added to the
# kind's D3 rotation seed (a SECOND deterministic draw — same d).
QJL_SEED_OFFSET = 7777
# The paper's §LongBench outlier-split recipe ("splitting channels into
# outlier and non-outlier sets, two independent instances of TurboQuant"):
# the OUTLIER sub-quantizer's own D3 frame seed offset, added to the kind's
# seed (the regular sub-quantizer keeps the kind's own seed).
SPLIT_SEED_OFFSET = 555
# The paper's Alg.-2 IP-estimator constant (PROPOSAL §1.2: (√(π/2)/d)·γ·Sᵀ·qjl
# with S unit-variance; folded with S = √d·FHT -> (√(π/2)/√d)·γ·FHT_adjoint).
QJL_C = math.sqrt(math.pi / 2.0)


# ------------------------------------------------------------- bit packs ---
def pack_bits(idx: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned indices (each < 2**bits, bits in {1..8}) into a uint8
    bit-stream (little-endian bit order). Lossless; length =
    ceil(n * bits / 8) bytes."""
    if bits not in range(1, 9):
        raise ValueError(f"pack_bits: bits must be 1..8, got {bits}")
    idx = np.asarray(idx)
    if idx.size and idx.dtype != np.uint8:
        # range-check BEFORE any dtype cast (wide dtypes would wrap mod 256)
        if int(idx.max()) >= (1 << bits) or int(idx.min()) < 0:
            raise ValueError(
                f"pack_bits: index out of {bits}-bit range "
                f"[{int(idx.min())}, {int(idx.max())}]")
    idx = np.ascontiguousarray(idx, dtype=np.uint8)
    if idx.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if int(idx.max(initial=0)) >= (1 << bits):
        raise ValueError(
            f"pack_bits: index {int(idx.max())} exceeds {bits}-bit range")
    if bits == 4:  # fast path: two nibbles per byte
        if idx.size % 2:
            idx = np.concatenate([idx, np.zeros(1, np.uint8)])
        return (idx[0::2] | (idx[1::2] << 4)).astype(np.uint8)
    bits_mat = ((idx[:, None] >> np.arange(bits, dtype=np.uint8)) & 1)
    return np.packbits(bits_mat.reshape(-1), bitorder="little")


def unpack_bits(packed: np.ndarray, bits: int, n: int) -> np.ndarray:
    """Inverse of pack_bits: returns exactly n indices."""
    if bits not in range(1, 9):
        raise ValueError(f"unpack_bits: bits must be 1..8, got {bits}")
    if bits == 4:
        p = np.ascontiguousarray(packed, dtype=np.uint8)
        out = np.empty(p.size * 2, dtype=np.uint8)
        out[0::2] = p & 0xF
        out[1::2] = p >> 4
        return out[:n]
    raw = np.unpackbits(np.ascontiguousarray(packed, dtype=np.uint8),
                        bitorder="little")[: n * bits].reshape(n, bits)
    weights = (1 << np.arange(bits, dtype=np.uint8))
    return (raw * weights).sum(axis=-1, dtype=np.uint8)


# ----------------------------------------------------------------- codes ---
@dataclass
class TQCodes:
    """Quantized codes for ONE unit (one S layer / one conv layer / M1 / M2).

    Layout: the coordinate split is contiguous by construction — the first
    `n_lo` coordinates use `bits_lo`, the rest use `bits_hi` (uniform
    bit-width: bits_hi == bits_lo, n_hi == 0). The split is data-oblivious
    (PROPOSAL D2 default); the W9 outlier A/B would carry an explicit
    partition — recorded in `partition` when not the default halves.
    """
    kind: str
    d: int
    norm: np.float32                       # fp norm, never quantized (paper recipe)
    bits_lo: int                           # bits for the lo set (1..8)
    bits_hi: int                           # bits for the hi set (== bits_lo if uniform)
    n_lo: int                              # coordinate count in the lo set
    n_hi: int                              # coordinate count in the hi set
    idx_lo: np.ndarray                     # packed uint8 bit-stream
    idx_hi: np.ndarray                     # packed uint8 bit-stream
    seed: int                              # the D3 rotation seed (persisted)
    partition: str = "half"                # "half" (legacy) | "outlier" (paper §LongBench)
    # W15 — the paper's outlier split (partition="outlier", conv kind):
    # group   = coords per channel (the conv kernel width; 1 = flat).
    #           Present IFF partition == "outlier" (backward compatible).
    # mask    = packed per-channel outlier membership (ceil(n_ch/8) uint8;
    #           bit i = 1 -> channel i is an OUTLIER (hi set). Present IFF
    #           partition == "outlier".
    # norm_hi = the outlier sub-set's fp32 norm (the regular set's norm is
    #           `norm`, parallel to idx_lo/idx_hi). Present IFF partition ==
    #           "outlier". For "half" codes the single `norm` field holds
    #           the whole unit's norm (unchanged legacy contract).
    # NOTE: for "outlier" codes, n_lo/n_hi are the PADDED sub-quantizer
    # unit sizes (power-of-two per sub-set — see TurboQuant._quant_split);
    # the real-coordinate counts are (n_ch - k)*group and k*group with
    # k = mask bit-count, n_ch = 8*len(mask).
    group: Optional[int] = None
    mask: Optional[np.ndarray] = None
    norm_hi: Optional[np.float32] = None
    # W9.2 qjl A/B (DEFAULT None = flag OFF / legacy codes): the structured
    # QJL residual sketch — (d,) int8 ±1 = sign(FHT(r/γ, seed+7777)) — and
    # the fp32 residual norm γ. Present IFF quantized with qjl=True;
    # serialization writes them ONLY when present (backward compatible).
    qjl_signs: Optional[np.ndarray] = None
    gamma: Optional[np.float32] = None
    # ---- serialization ---------------------------------------------------
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Flat array dict for embedding into a larger npz (snapshot.py).

        The qjl fields are written ONLY when present, so codes quantized
        without the flag produce the EXACT pre-W9.2 key set (old readers
        and old digests are unaffected)."""
        out = {
            "kind": np.array(self.kind),
            "d": np.array(self.d, dtype=np.int64),
            "norm": np.array(self.norm, dtype=np.float32),
            "bits_lo": np.array(self.bits_lo, dtype=np.int64),
            "bits_hi": np.array(self.bits_hi, dtype=np.int64),
            "n_lo": np.array(self.n_lo, dtype=np.int64),
            "n_hi": np.array(self.n_hi, dtype=np.int64),
            "seed": np.array(self.seed, dtype=np.int64),
            "partition": np.array(self.partition),
            "idx_lo": self.idx_lo,
            "idx_hi": self.idx_hi,
        }
        if self.partition == "outlier":
            if self.group is not None:
                out["group"] = np.array(int(self.group), dtype=np.int64)
            if self.mask is not None:
                out["mask"] = np.ascontiguousarray(self.mask, dtype=np.uint8)
            if self.norm_hi is not None:
                out["norm_hi"] = np.array(self.norm_hi, dtype=np.float32)
        if self.qjl_signs is not None:
            out["qjl_signs"] = np.ascontiguousarray(self.qjl_signs,
                                                   dtype=np.int8)
        if self.gamma is not None:
            out["gamma"] = np.array(self.gamma, dtype=np.float32)
        return out

    @classmethod
    def from_arrays(cls, a: Dict[str, np.ndarray]) -> "TQCodes":
        """Inverse of to_arrays. BACKWARD COMPATIBLE: an array dict
        written WITHOUT the qjl keys (pre-W9.2 npz/snapshots) loads with
        qjl_signs/gamma == None — the flag-off dequant path."""
        qjl = a.get("qjl_signs") if hasattr(a, "get") else None
        gam = a.get("gamma") if hasattr(a, "get") else None
        part = str(a.get("partition", "half")) if hasattr(a, "get") else "half"
        grp = a.get("group") if hasattr(a, "get") else None
        msk = a.get("mask") if hasattr(a, "get") else None
        nhi = a.get("norm_hi") if hasattr(a, "get") else None
        return cls(
            kind=str(a["kind"]),
            d=int(a["d"]),
            norm=np.float32(a["norm"]),
            bits_lo=int(a["bits_lo"]),
            bits_hi=int(a["bits_hi"]),
            n_lo=int(a["n_lo"]),
            n_hi=int(a["n_hi"]),
            idx_lo=np.asarray(a["idx_lo"], dtype=np.uint8),
            idx_hi=np.asarray(a["idx_hi"], dtype=np.uint8),
            seed=int(a["seed"]),
            partition=part,
            group=(None if grp is None else int(grp)),
            mask=(None if msk is None
                  else np.asarray(msk, dtype=np.uint8)),
            norm_hi=(None if nhi is None else np.float32(nhi)),
            qjl_signs=(None if qjl is None
                       else np.asarray(qjl, dtype=np.int8)),
            gamma=(None if gam is None else np.float32(gam)),
        )

    def nbytes(self) -> int:
        n = self.idx_lo.nbytes + self.idx_hi.nbytes + 4  # + fp32 norm
        if self.partition == "outlier":   # W15: mask + second fp32 norm
            if self.mask is not None:
                n += int(self.mask.nbytes)
            n += 4
        if self.qjl_signs is not None:      # W9.2 A/B: +1 bit/coordinate
            n += int(self.qjl_signs.nbytes)  # (d,) int8 sketch
        if self.gamma is not None:
            n += 4                           # fp32 gamma
        return n


# ------------------------------------------------------------ quantizer ----
class TurboQuant:
    """One quantizer instance per tensor KIND (PROPOSAL D3).

    Parameters
    ----------
    kind : "S" | "conv" | "M1" | "M2" — or "custom" with explicit d/seed.
    bits : float in (0, 8] or int. 3.5 (default) → 50/50 split at (3, 4)
        bits (spec §3.3). An int b → uniform b-bit. A non-integer x is
        realized as (floor(x), ceil(x)) with the fractional part choosing
        the lo/hi coordinate ratio (0.5 → half/half).
    seed : overrides the kind's D3 seed (tests only — never in production).
    qjl : bool, default False — the W9.2 P7/D1 prod-variant A/B flag. When
        True, quant() additionally computes the structured QJL residual
        sketch (see the module docstring for the exact formulation) and
        the returned TQCodes carry qjl_signs/gamma; dequant() then adds the
        compensation term (√(π/2)/√d)·γ·FHT_adjoint(qjl) whenever the CODES
        carry the fields (a qjl=False instance dequantizes qjl codes
        correctly — the codes are self-describing). Flag OFF is
        bit-identical to the pre-W9.2 behavior (parity gate).
    group : int, default 1 — W15, the paper's §LongBench outlier-split
        recipe ("splitting channels into outlier and non-outlier sets, and
        applying two independent instances of TurboQuant to each, allocating
        higher bit precision to outliers"). group > 1 selects it: the d
        coordinates are CHANNEL GROUPS of `group` coords (the conv window's
        (channel, tap) layout — group = the kernel width), k = round(frac·d/group)
        highest-energy channels quantize at bits_hi with their OWN
        power-of-two full rotation (frame seed = kind seed + 555) and their
        OWN fp norm; the remaining channels at bits_lo with the kind's own
        seed frame. The membership mask (packed, ceil(n_ch/8) bytes) and the
        second norm ride in the codes (TQCodes.mask / norm_hi) — measured
        at production conv geometry this is ~3x lower write-path rel-MSE
        than the fixed coordinate half-split at the SAME effective bits
        (the fixed split puts the 4 bits on a fixed coordinate half that
        ignores the channel-energy structure; the paper's split puts them
        on the channels that carry the energy). group == 1 (S/M1/M2 — and
        any flat vector) keeps the legacy coordinate half-split exactly
        (partition="half", bit-identical to pre-W15). dequant() routes on
        the CODES' partition field, so old "half" snapshots load and
        decode unchanged through a split-configured quantizer.
    """

    def __init__(self, kind: str = "S", bits: float = 3.5,
                 d: Optional[int] = None, seed: Optional[int] = None,
                 qjl: bool = False, group: int = 1):
        if kind not in KINDS:
            if d is None or seed is None:
                raise ValueError(
                    f"TurboQuant: unknown kind {kind!r}; pass d and seed "
                    f"explicitly for custom kinds (tests only)")
        self.kind = kind
        self.d = int(d if d is not None else KINDS[kind][0])
        self.seed = int(seed if seed is not None else KINDS[kind][1])
        # FHT kernel supports non-power-of-two via segmentation (as long as
        # K is a multiple of 32 and each segment fits in shared memory).
        # Only validate that d is a positive integer.
        if self.d < 1:
            raise ValueError(f"TurboQuant: d must be >= 1, got {self.d}")
        if not (0 < bits <= 8):
            raise ValueError(f"TurboQuant: bits must be in (0, 8], got {bits}")
        # codebooks are solved per integer bit-width 1..4 (the paper's range)
        eff_bits = (int(bits), int(bits) + 1) if not float(bits).is_integer() \
            else (int(bits), int(bits))
        if eff_bits[0] < 1 or eff_bits[1] > 4:
            raise ValueError(
                f"TurboQuant: realized bit-widths {eff_bits} exceed the "
                f"Lloyd-Max codebook range (1..4) — see codebooks.py")
        self.bits_spec = float(bits)
        if float(bits).is_integer():
            self.bits_lo = self.bits_hi = int(bits)
            frac = 0.0
        else:
            lo, hi = int(bits), int(bits) + 1
            if hi > 8:
                raise ValueError(f"TurboQuant: bits {bits} exceeds 8")
            self.bits_lo, self.bits_hi = lo, hi
            frac = float(bits) - lo
        # coordinate split: frac is the fraction at the hi bit-width
        self.group = int(group)
        if self.group < 1:
            raise ValueError(f"TurboQuant: group must be >= 1, got {group}")
        if self.d % self.group != 0:
            raise ValueError(
                f"TurboQuant({kind}): d={self.d} is not a multiple of the "
                f"channel group size {self.group}")
        self.n_channels = self.d // self.group
        self.split = self.group > 1 and not float(bits).is_integer()
        if self.split:
            k = int(round(frac * self.n_channels))
            if not (1 <= k < self.n_channels):
                raise ValueError(
                    f"TurboQuant({kind}): the outlier split needs 1 <= k < "
                    f"n_channels, got k={k} of {self.n_channels} channels "
                    f"(d={self.d}, group={self.group}, bits={bits})")
            self.k_out = k
            reg_real = (self.n_channels - k) * self.group
            out_real = k * self.group
            self.reg_d, self.out_d = _next_pow2(reg_real), _next_pow2(out_real)
            # W15: for a SPLIT quantizer n_lo/n_hi are the PADDED sub-unit
            # sizes (the codes' idx streams cover the padded sub-d's — the
            # pad carries quantization noise that the un-rotation spreads,
            # and the strip at dequant discards; measured: the padded full
            # rotation beats the exact-size segmented rotation, P1).
            self.n_lo, self.n_hi = self.reg_d, self.out_d
            # the sub-codebooks resolve lazily through _sub_quantizer
            self._cb_lo = self._cb_hi = None
            self._signs = None
        else:
            n_hi = int(round(frac * self.d))
            self.n_lo, self.n_hi = self.d - n_hi, n_hi
            self.k_out = 0
            self.reg_d = self.out_d = self.d
            self._cb_lo = codebooks.get_codebook(self.bits_lo, self.d)
            self._cb_hi = (self._cb_lo if self.bits_hi == self.bits_lo
                           else codebooks.get_codebook(self.bits_hi, self.d))
            # the D3 rotation: one sign vector per kind, generated once
            self._signs = fht.rotation_signs(self.d, self.seed)
        # W9.2 qjl flag (DEFAULT OFF — the fields below are computed
        # unconditionally but USED only on qjl paths, so flag-off output
        # is bit-identical to the pre-W9.2 behavior):
        self.qjl = bool(qjl)
        # the SECOND deterministic sign draw (kind seed + 7777, same d) —
        # the structured QJL projection S = √d·FHT(d, seed+7777)
        self._qjl_signs_vec = fht.rotation_signs(self.d,
                                                 self.seed + QJL_SEED_OFFSET)

    # ------------------------------------------------------ sub-quantizers --
    def _sub_quantizer(self, role: str) -> "TurboQuant":
        """One of the two paper-recipe sub-instances ("reg" | "out") for a
        SPLIT quantizer: uniform integer bits, power-of-two d (its own FULL
        single-segment rotation — the paper's invariant), deterministic
        frame seed (regular keeps the kind's seed; outlier = seed + 555).
        Resolved through the process registry so instances are shared."""
        assert self.split
        bits = self.bits_lo if role == "reg" else self.bits_hi
        d = self.reg_d if role == "reg" else self.out_d
        seed = self.seed if role == "reg" else self.seed + SPLIT_SEED_OFFSET
        key = f"{self.kind}:{bits}:{d}:{seed}:sub"
        reg = _REGISTRY
        if key not in reg:
            reg[key] = TurboQuant(kind="custom", bits=bits, d=d, seed=seed)
        return reg[key]

    # ---------------------------------------------------------------- API --
    def quant(self, x: torch.Tensor) -> TQCodes:
        """Quantize ONE unit: x is a (d,) tensor (any float dtype).

        qjl=True (the W9.2 flag) additionally attaches the residual sketch:
        the ORIGINAL-frame residual r = x − dequant_mse(codes), its fp32
        norm γ, and qjl = sign(FHT(r/γ, seed+7777)) (module docstring has
        the exact formulation). The MSE layer (idx/norm) is computed
        IDENTICALLY with and without the flag — the A/B is purely additive.

        group > 1 (the constructor's W15 paper-recipe flag) routes to the
        outlier-channel split: see the class docstring."""
        x = torch.as_tensor(x)
        if x.dim() != 1 or x.shape[0] != self.d:
            raise ValueError(
                f"TurboQuant({self.kind}): expected a ({self.d},) tensor, "
                f"got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError(f"TurboQuant: x must be floating point, got {x.dtype}")
        x32 = x.detach().to(torch.float32)
        if self.split:
            return self._quant_split(x32)
        norm = float(x32.norm().item())
        if norm == 0.0:  # degenerate: all-zero unit (codes of zero)
            codes = TQCodes(self.kind, self.d, np.float32(0.0),
                            self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
                            np.zeros(_packed_len(self.n_lo, self.bits_lo), np.uint8),
                            np.zeros(_packed_len(self.n_hi, self.bits_hi), np.uint8),
                            self.seed)
        else:
            r = (x32 / norm).reshape(1, self.d)
            y = fht.fht_apply(r, self._signs).reshape(self.d)      # y = r @ T
            y_np = y.cpu().numpy() if y.is_cuda else y.numpy()
            lo, hi = y_np[: self.n_lo], y_np[self.n_lo:]
            idx_lo = np.searchsorted(self._cb_lo.boundaries, lo).astype(np.uint8)
            idx_hi = np.searchsorted(self._cb_hi.boundaries, hi).astype(np.uint8)
            codes = TQCodes(self.kind, self.d, np.float32(norm),
                            self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
                            pack_bits(idx_lo, self.bits_lo),
                            pack_bits(idx_hi, self.bits_hi),
                            self.seed)
        if self.qjl:
            self._attach_qjl(codes, x32)
        return codes

    def _attach_qjl(self, codes: TQCodes, x32: torch.Tensor) -> None:
        """Compute + attach the structured QJL residual sketch (W9.2 flag).

        r = x32 − dequant_mse(codes) in the ORIGINAL frame (the rotated
        form is identical by linearity); γ = ‖r‖; qjl =
        sign(FHT(r/γ, seed+7777)) — a (d,) int8 ±1 vector. A zero residual
        (γ == 0: the degenerate zero-norm unit, or an exact reconstruction)
        stores zeros and γ = 0 — the compensation term is then exactly 0.

        W16 DEVICE FIX: _dequant_mse returns a CPU tensor (its codebook
        path is numpy/torch-CPU) while x32 can be CUDA — the GPU box's
        --qjl A/B crashed exactly here ("Expected all tensors to be on
        the same device"). The reconstruction is moved onto x32's device
        BEFORE the residual subtraction (CPU behavior unchanged —
        same-device is a no-op)."""
        x_mse = self._dequant_mse(codes, dtype=torch.float32)
        if x_mse.device != x32.device:
            x_mse = x_mse.to(x32.device)
        resid = x32 - x_mse
        gamma = float(resid.norm().item())
        if gamma > 0.0:
            rn = (resid / gamma).reshape(1, self.d)
            proj = fht.fht_apply(rn, self._qjl_signs_vec).reshape(self.d)
            # ±1 exactly (np.sign; 0 only on an exact-zero projection —
            # measure-zero for generic residuals, degrades that coordinate
            # by 1/√d of its weight)
            signs = np.sign(proj.cpu().numpy() if proj.is_cuda
                            else proj.numpy()).astype(np.int8)
        else:
            signs = np.zeros(self.d, dtype=np.int8)
        codes.qjl_signs = signs
        codes.gamma = np.float32(gamma)

    # ------------------------------------------- W15: the outlier split ----
    def _quant_split(self, x32: torch.Tensor) -> TQCodes:
        """The paper's §LongBench recipe at work (see the class docstring):
        top-k channels by energy → the outlier sub-set (bits_hi, own frame
        seed+555, own norm, pow2-padded own rotation); the rest → the
        regular sub-set (bits_lo, the kind's seed, own norm). Deterministic
        (stable argsort on the channel energies)."""
        n_ch, g = self.n_channels, self.group
        x_np = x32.cpu().numpy() if x32.is_cuda else x32.numpy()
        xg = x_np.reshape(n_ch, g)
        energy = np.einsum("ij,ij->i", xg, xg)
        order = np.argsort(-energy, kind="stable")
        out_ch = np.sort(order[: self.k_out])
        reg_ch = np.sort(order[self.k_out:])
        mask = np.zeros(n_ch, dtype=bool)
        mask[out_ch] = True
        xreg = np.ascontiguousarray(xg[reg_ch].reshape(-1))
        xout = np.ascontiguousarray(xg[out_ch].reshape(-1))
        q_reg, q_out = self._sub_quantizer("reg"), self._sub_quantizer("out")
        c_reg = q_reg.quant(_pad_to(xreg, self.reg_d))
        c_out = q_out.quant(_pad_to(xout, self.out_d))
        codes = TQCodes(
            self.kind, self.d, np.float32(c_reg.norm),
            self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
            c_reg.idx_lo, c_out.idx_lo, self.seed,
            partition="outlier", group=g,
            mask=np.packbits(mask),
            norm_hi=np.float32(c_out.norm))
        if self.qjl:
            self._attach_qjl(codes, x32)
        return codes

    def _split_channels(self, codes: TQCodes) -> tuple:
        """(reg_ch, out_ch) from the codes' packed mask — validated."""
        if codes.mask is None or codes.group is None:
            raise ValueError(
                f"TurboQuant({self.kind}): outlier-partition codes without a "
                f"mask/group — corrupted serialization?")
        n_ch = self.d // int(codes.group)
        bits_ = np.unpackbits(np.ascontiguousarray(codes.mask, dtype=np.uint8))
        if bits_.shape[0] < n_ch:
            raise ValueError(
                f"TurboQuant({self.kind}): mask covers {bits_.shape[0]} "
                f"channels, the unit needs {n_ch}")
        mask = bits_[:n_ch].astype(bool)
        out_ch = np.nonzero(mask)[0]
        reg_ch = np.nonzero(~mask)[0]
        if out_ch.shape[0] != self.k_out:
            raise ValueError(
                f"TurboQuant({self.kind}): mask says {out_ch.shape[0]} outlier "
                f"channels, the geometry says k={self.k_out} — frame drift?")
        return reg_ch, out_ch

    def _dequant_split_mse(self, codes: TQCodes,
                           dtype: torch.dtype) -> torch.Tensor:
        """The outlier-partition MSE reconstruction: two sub-dequants
        (each through its own frame + norm), pad-strip, scatter to the
        channel layout."""
        reg_ch, out_ch = self._split_channels(codes)
        q_reg, q_out = self._sub_quantizer("reg"), self._sub_quantizer("out")
        c_reg = TQCodes(q_reg.kind, q_reg.d, codes.norm,
                        q_reg.bits_lo, q_reg.bits_hi, q_reg.n_lo, q_reg.n_hi,
                        codes.idx_lo, np.zeros(0, np.uint8), q_reg.seed)
        c_out = TQCodes(q_out.kind, q_out.d, codes.norm_hi,
                        q_out.bits_lo, q_out.bits_hi, q_out.n_lo, q_out.n_hi,
                        codes.idx_hi, np.zeros(0, np.uint8), q_out.seed)
        reg_real, out_real = reg_ch.shape[0] * int(codes.group), \
            out_ch.shape[0] * int(codes.group)
        x = np.zeros(self.d, dtype=np.float32)
        xg = x.reshape(self.n_channels, int(codes.group))
        if float(codes.norm) != 0.0:
            v = q_reg.dequant(c_reg, dtype=torch.float32) \
                .numpy()[: reg_real]
            xg[reg_ch] = v.reshape(-1, int(codes.group))
        if codes.norm_hi is not None and float(codes.norm_hi) != 0.0:
            v = q_out.dequant(c_out, dtype=torch.float32) \
                .numpy()[: out_real]
            xg[out_ch] = v.reshape(-1, int(codes.group))
        return torch.from_numpy(x).to(dtype)

    def _flat_twin(self) -> "TurboQuant":
        """The legacy flat-semantics twin of a SPLIT quantizer (same kind /
        bits / d / seed, group=1) — dequantizing pre-W15 partition="half"
        codes through it keeps every legacy contract exact (the codes are
        self-describing; old snapshots load and decode unchanged)."""
        key = f"{self.kind}:{self.bits_spec}:{self.d}:{self.seed}:flat"
        if key not in _REGISTRY:
            _REGISTRY[key] = TurboQuant(
                kind=self.kind, bits=self.bits_spec, d=self.d,
                seed=self.seed, qjl=self.qjl)
        return _REGISTRY[key]

    def _split_twin(self, group: int) -> "TurboQuant":
        """The split-semantics twin at `group` coords per channel — the
        reverse of _flat_twin: a FLAT-configured quantizer (group=1, e.g.
        resolved by a generic codes-only path) dequantizing
        partition="outlier" codes routes here so the sub-quantizer frames
        resolve from the codes' own geometry."""
        key = f"{self.kind}:{self.bits_spec}:{self.d}:{self.seed}:split{group}"
        if key not in _REGISTRY:
            _REGISTRY[key] = TurboQuant(
                kind=self.kind, bits=self.bits_spec, d=self.d,
                seed=self.seed, qjl=self.qjl, group=group)
        return _REGISTRY[key]

    def _route_for(self, codes: TQCodes) -> "TurboQuant":
        """The quantizer that owns CODES' partition (self, or the twin with
        the right group semantics). Codes are self-describing: outlier
        codes need split semantics (self's group must equal the codes'
        group); half codes need flat semantics."""
        part = getattr(codes, "partition", "half")
        if part == "outlier":
            grp = int(codes.group) if codes.group is not None else 1
            if self.split and self.group == grp:
                return self
            return self._split_twin(grp)
        if self.split:
            return self._flat_twin()
        return self

    def dequant(self, codes: TQCodes,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize ONE unit back to a (d,) tensor of `dtype`.

        Routing (W15): the CODES are self-describing — partition="outlier"
        dequantizes through the split sub-quantizers; partition="half"
        (all pre-W15 codes, and every S/M1/M2 unit) dequantizes through the
        flat frame — through the legacy twin when self is split-configured,
        so old snapshots load unchanged.

        When the codes carry the QJL sketch (qjl_signs + gamma, the W9.2
        flag — present IFF quantized with qjl=True), the reconstruction is
        the paper's Alg.-2 estimator: x̃ = x̃_mse + (√(π/2)/√d)·γ·
        FHT_adjoint(qjl)  ==  x̃_mse + (√(π/2)/d)·γ·Sᵀ·qjl with the
        structured projection S = √d·FHT(d, codes.seed+7777) (module
        docstring). Codes WITHOUT the fields: exactly the pre-W9.2 path."""
        if getattr(codes, "partition", "half") == "outlier":
            q = self._route_for(codes)
            q._check_codes(codes)
            out = q._dequant_split_mse(codes, dtype=torch.float32)
            # NOTE: no whole-unit zero-norm early return here — `norm` is
            # the REGULAR sub-set's norm; a zero regular set with a live
            # outlier set (or vice versa) is handled per sub-set above.
            out = self._qjl_compensation_term(codes, out)
            return out.to(dtype)
        q = self._route_for(codes)
        return q._dequant_flat(codes, dtype)

    def _qjl_compensation_term(self, codes: TQCodes,
                               out: torch.Tensor) -> torch.Tensor:
        """The Alg.-2 residual term, shared by both partitions (the sketch
        lives in the ORIGINAL frame — partition-agnostic)."""
        if codes.qjl_signs is None and codes.gamma is None:
            return out
        if codes.qjl_signs is None or codes.gamma is None:
            raise ValueError(
                f"TurboQuant({self.kind}, bits={self.bits_spec}, "
                f"d={self.d}, seed={self.seed}): codes carry a PARTIAL "
                f"QJL sketch (qjl_signs={'set' if codes.qjl_signs is not None else 'None'}, "
                f"gamma={'set' if codes.gamma is not None else 'None'}) — "
                f"corrupted serialization?")
        s = np.asarray(codes.qjl_signs)
        if s.dtype != np.int8 or s.ndim != 1 or s.shape[0] != codes.d:
            raise ValueError(
                f"TurboQuant({self.kind}): qjl_signs must be a "
                f"({codes.d},) int8 ±1 sketch, got shape={s.shape}, "
                f"dtype={s.dtype}")
        gamma = float(codes.gamma)
        if gamma != 0.0:
            # _check_codes pins codes.seed == self.seed, so the cached
            # second draw is the codes' own QJL sign vector.
            # np.array(...) = a WRITABLE fp32 copy: torch.from_numpy on
            # a read-only view (e.g. the zero-copy views onto mmap'd
            # snapshot members) would warn + be UB on write.
            s_t = torch.from_numpy(
                np.array(s, dtype=np.float32)).reshape(1, self.d)
            z = fht.fht_adjoint(s_t, self._qjl_signs_vec).reshape(self.d)
            out = out + (QJL_C * gamma / math.sqrt(self.d)) * z
        return out

    def _dequant_flat(self, codes: TQCodes,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The legacy (pre-W15) dequant body, VERBATIM — the flat path."""
        self._check_codes(codes)
        if float(codes.norm) == 0.0:
            # zero-norm unit ⇒ the quant-side residual was exactly zero ⇒
            # γ == 0 ⇒ the compensation term is exactly 0 — plain zeros.
            return torch.zeros(self.d, dtype=dtype)
        out = self._dequant_mse(codes, dtype=torch.float32)
        out = self._qjl_compensation_term(codes, out)
        return out.to(dtype)

    def _dequant_mse(self, codes: TQCodes,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The MSE-layer reconstruction. W15: dispatches on the codes'
        partition through _route_for (the split path through the
        sub-quantizers; the flat path through the twin when self carries
        the other semantics)."""
        if getattr(codes, "partition", "half") == "outlier":
            q = self._route_for(codes)
            q._check_codes(codes)
            return q._dequant_split_mse(codes, dtype)
        q = self._route_for(codes)
        if q is not self:
            return q._dequant_mse(codes, dtype)
        idx_lo = unpack_bits(codes.idx_lo, codes.bits_lo, codes.n_lo)
        idx_hi = unpack_bits(codes.idx_hi, codes.bits_hi, codes.n_hi)
        y = np.empty(self.d, dtype=np.float32)
        y[: codes.n_lo] = self._cb_lo.centroids[idx_lo]
        y[codes.n_lo:] = self._cb_hi.centroids[idx_hi]
        y_t = torch.from_numpy(y).reshape(1, self.d)
        r = fht.fht_adjoint(y_t, self._signs).reshape(self.d)   # r = y @ T^T
        return (r * float(codes.norm)).to(dtype)

    # round-trip helper (evals/Phase-1 measurement harness)
    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve device: dequant returns CPU tensors (codebook path is numpy/CPU)
        out = self.dequant(self.quant(x), dtype=x.dtype)
        return out.to(x.device) if out.device != x.device else out

    # ------------------------------------------------------------ plumbing -
    def _check_codes(self, codes: TQCodes) -> None:
        if (codes.kind != self.kind or codes.d != self.d
                or codes.bits_lo != self.bits_lo
                or codes.bits_hi != self.bits_hi
                or codes.n_lo != self.n_lo or codes.n_hi != self.n_hi
                or codes.seed != self.seed):
            raise ValueError(
                f"TurboQuant({self.kind}, bits={self.bits_spec}, d={self.d}, "
                f"seed={self.seed}): codes are "
                f"({codes.kind}, d={codes.d}, bits=({codes.bits_lo},"
                f"{codes.bits_hi}), n=({codes.n_lo},{codes.n_hi}), "
                f"seed={codes.seed}) — mismatch (frame drift? see PROPOSAL D3)")


def _packed_len(n: int, bits: int) -> int:
    return (n * bits + 7) // 8


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1) — the W15 split sub-units."""
    if n < 1:
        raise ValueError(f"_next_pow2: n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def _pad_to(x: np.ndarray, d: int) -> torch.Tensor:
    """Zero-pad the flat (n,) numpy vector UP to (d,) and return a
    contiguous fp32 torch tensor (n <= d). The pad adds no energy — the
    sub-quantizer's stored norm is the real sub-set's."""
    if x.shape[0] > d:
        raise ValueError(f"_pad_to: n={x.shape[0]} exceeds d={d}")
    if x.shape[0] < d:
        x = np.concatenate(
            [x, np.zeros(d - x.shape[0], dtype=np.float32)])
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


# -------------------------------------------------------------- registry ---
_REGISTRY: Dict[str, TurboQuant] = {}


def get_quantizer(kind: str, bits: float = 3.5) -> TurboQuant:
    """Process-wide registry: one TurboQuant per (kind, bits) — the shared
    rotation instance per tensor kind (D3)."""
    key = f"{kind}:{bits}"
    if key not in _REGISTRY:
        _REGISTRY[key] = TurboQuant(kind=kind, bits=bits)
    return _REGISTRY[key]
