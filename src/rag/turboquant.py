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
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

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
KINDS: Dict[str, Tuple[int, int]] = {
    "S":   (524_288, 101),   # per-layer recurrent state (spec §2.1)
    "conv": (32_768, 202),   # per-layer conv state (spec §2.3)
    "M1":  (524_288, 303),   # global key-memory (spec §2.2)
    "M2":  (524_288, 404),   # global value-memory (spec §2.2)
}
SEEDS: Dict[str, int] = {k: v[1] for k, v in KINDS.items()}


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
    partition: str = "half"                # "half" (default) | "outlier" (W9 A/B)

    # ---- serialization ---------------------------------------------------
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Flat array dict for embedding into a larger npz (snapshot.py)."""
        return {
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

    @classmethod
    def from_arrays(cls, a: Dict[str, np.ndarray]) -> "TQCodes":
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
            partition=str(a["partition"]),
        )

    def nbytes(self) -> int:
        return self.idx_lo.nbytes + self.idx_hi.nbytes + 4  # + fp32 norm


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
    """

    def __init__(self, kind: str = "S", bits: float = 3.5,
                 d: Optional[int] = None, seed: Optional[int] = None):
        if kind not in KINDS:
            if d is None or seed is None:
                raise ValueError(
                    f"TurboQuant: unknown kind {kind!r}; pass d and seed "
                    f"explicitly for custom kinds (tests only)")
        self.kind = kind
        self.d = int(d if d is not None else KINDS[kind][0])
        self.seed = int(seed if seed is not None else KINDS[kind][1])
        if self.d < 1 or (self.d & (self.d - 1)) != 0:
            raise ValueError(
                f"TurboQuant: d must be a power of two (FHT single-block "
                f"contract), got {self.d}")
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
        n_hi = int(round(frac * self.d))
        self.n_lo, self.n_hi = self.d - n_hi, n_hi
        self._cb_lo = codebooks.get_codebook(self.bits_lo, self.d)
        self._cb_hi = (self._cb_lo if self.bits_hi == self.bits_lo
                       else codebooks.get_codebook(self.bits_hi, self.d))
        # the D3 rotation: one sign vector per kind, generated once
        self._signs = fht.rotation_signs(self.d, self.seed)

    # ---------------------------------------------------------------- API --
    def quant(self, x: torch.Tensor) -> TQCodes:
        """Quantize ONE unit: x is a (d,) tensor (any float dtype)."""
        x = torch.as_tensor(x)
        if x.dim() != 1 or x.shape[0] != self.d:
            raise ValueError(
                f"TurboQuant({self.kind}): expected a ({self.d},) tensor, "
                f"got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError(f"TurboQuant: x must be floating point, got {x.dtype}")
        x32 = x.detach().to(torch.float32)
        norm = float(x32.norm().item())
        if norm == 0.0:  # degenerate: all-zero unit (codes of zero)
            return TQCodes(self.kind, self.d, np.float32(0.0),
                           self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
                           np.zeros(_packed_len(self.n_lo, self.bits_lo), np.uint8),
                           np.zeros(_packed_len(self.n_hi, self.bits_hi), np.uint8),
                           self.seed)
        r = (x32 / norm).reshape(1, self.d)
        y = fht.fht_apply(r, self._signs).reshape(self.d)      # y = r @ T
        y_np = y.numpy()
        lo, hi = y_np[: self.n_lo], y_np[self.n_lo:]
        idx_lo = np.searchsorted(self._cb_lo.boundaries, lo).astype(np.uint8)
        idx_hi = np.searchsorted(self._cb_hi.boundaries, hi).astype(np.uint8)
        return TQCodes(self.kind, self.d, np.float32(norm),
                       self.bits_lo, self.bits_hi, self.n_lo, self.n_hi,
                       pack_bits(idx_lo, self.bits_lo),
                       pack_bits(idx_hi, self.bits_hi),
                       self.seed)

    def dequant(self, codes: TQCodes,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize ONE unit back to a (d,) tensor of `dtype`."""
        self._check_codes(codes)
        if float(codes.norm) == 0.0:
            return torch.zeros(self.d, dtype=dtype)
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
        return self.dequant(self.quant(x), dtype=x.dtype)

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


# -------------------------------------------------------------- registry ---
_REGISTRY: Dict[str, TurboQuant] = {}


def get_quantizer(kind: str, bits: float = 3.5) -> TurboQuant:
    """Process-wide registry: one TurboQuant per (kind, bits) — the shared
    rotation instance per tensor kind (D3)."""
    key = f"{kind}:{bits}"
    if key not in _REGISTRY:
        _REGISTRY[key] = TurboQuant(kind=kind, bits=bits)
    return _REGISTRY[key]
