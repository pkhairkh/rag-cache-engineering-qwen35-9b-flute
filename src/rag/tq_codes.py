"""TQCodes — the rev-1 torch-native quantized-codes container (W20.2.1).

THE CONTRACT
------------
Codes are torch tensors on the quantizing device. The packed bit-streams
(idx_lo/idx_hi — and the mask, when present) are torch uint8 tensors;
every norm (norm — and norm_hi/gamma, when present) is a 0-dim float32
TENSOR — never a python float, never a host-array scalar. The `.item`
era ends here: nothing in this container reads a value back off the
device.

W29 — THE "units" PARTITION (the per-head S repair, FINDINGS W28 item
9 / the box matrix's worst-block 7.65 measurement): a units codes
object carries M quantization units of the SAME per-unit cardinals
(d, bits, split, seed — the frame) in ONE container — the production
S case: 32 per-head units of d=16,384, own norm EACH (the (M,) fp32
`norm` tensor; the paper's per-vector norm recipe, Theorem 1 applied
per unit). The idx streams are the row-major concatenation of the M
per-unit packed streams (units*L bytes each — quant_units guarantees
the per-unit stream is byte-aligned); n_lo/n_hi stay PER-UNIT counts.
Dequantization routes on codes.partition (the W15 self-describing
discipline) and returns the flat (M*d,) vector — the layer's flat
layout, so every §4 consumer sees the same flat geometry. The frame
at the units scale is the PERMUTED rotation y = FHT(P x) with P drawn
deterministically from the kind's seed (turboquant.UNITS_PERM_OFFSET)
— seed-addressed, so NO new serialization fields beyond `units`.

NumPy exists ONLY inside to_arrays()/from_arrays() — the npz disk edge
(to_arrays flattens one unit into the array dict that snapshot.py embeds
into a larger npz; from_arrays is the inverse hop). The import itself is
function-local to those two methods, so the confinement is structural
rather than conventional: nothing at module scope can even name a host
array by accident.

Snapshot-npz compatibility is the HARD constraint: to_arrays() emits the
EXACT pre-W20 key set, dtypes and conditional-key discipline — kind, d,
norm, bits_lo, bits_hi, n_lo, n_hi, seed, partition, idx_lo, idx_hi
always; group/mask/norm_hi only when partition == "outlier" (each still
only when the field is present); units only when partition == "units"
(the norm member is then the (M,) per-unit vector — a 0-dim norm on a
units unit is a corruption the reader refuses); qjl_signs/gamma only
when present. Old readers and old digests are unaffected.

Field semantics are UNCHANGED from the host-array-era class this module
replaces (W20.2.1 moved the CONTAINER to torch; W20.3 moves the
quantizer's construction sites onto it — until then the NumPy-typed
construction sites in turboquant.py fail this class's TypeError checks
by design: the staged hand-off).

Construction discipline: __post_init__ raises TypeError (never a bare
assert) on wrong types — pure metadata checks (type/dtype/shape; no
device sync, no value reads); scalar fields stay python scalars.
__repr__ dumps no tensor values, so it is sync-free by construction.

Self-contained by law (PROMPT.md §1): no repo imports — dataclasses,
typing and torch at module scope; NumPy only at the disk edge.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Union

import torch

__all__ = ["TQCodes"]


# ---------------------------------------------------------- type cops ----
# Pure metadata checks (class / dtype / shape — no device sync, no value
# reads); TypeError with a precise message, NEVER a bare assert, so a
# wrong-typed construction fails loudly and legibly at the boundary.
def _chk_str(name: str, val: Any) -> None:
    if not isinstance(val, str):
        raise TypeError(f"TQCodes.{name} must be a python str, "
                        f"got {type(val).__name__}")


def _chk_int(name: str, val: Any) -> None:
    # python int ONLY (bool passes isinstance — harmless); a NumPy scalar
    # here means a caller leaked the disk edge INTO the container.
    if not isinstance(val, int):
        raise TypeError(f"TQCodes.{name} must be a python int, "
                        f"got {type(val).__name__}")


def _chk_tensor(name: str, val: Any, dtype: torch.dtype,
                ndim: Optional[int] = None) -> None:
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"TQCodes.{name} must be a torch.Tensor, "
                        f"got {type(val).__name__}")
    if val.dtype != dtype:
        raise TypeError(f"TQCodes.{name} must have dtype {dtype}, "
                        f"got {val.dtype}")
    if ndim is not None and val.dim() != ndim:
        raise TypeError(f"TQCodes.{name} must be {ndim}-dim, got "
                        f"{val.dim()}-dim of shape {tuple(val.shape)}")


def _tnbytes(t: torch.Tensor) -> int:
    """Byte count from metadata only — numel() * element_size(), valid on
    any device, no sync, and no dependence on Tensor nbytes attributes
    across torch versions."""
    return t.numel() * t.element_size()


# ---------------------------------------------------------------- codes --
@dataclass
class TQCodes:
    """Quantized codes for ONE unit (one S layer / one conv layer / M1 /
    M2) — the rev-1 torch-native container (module docstring = the
    contract).

    Layout: the coordinate split is contiguous by construction — the
    first `n_lo` coordinates use `bits_lo`, the rest use `bits_hi`
    (uniform bit-width: bits_hi == bits_lo, n_hi == 0). The split is
    data-oblivious (PROPOSAL D2 default); the W9 outlier A/B would carry
    an explicit partition — recorded in `partition` when not the default
    halves.

    Field vocabulary is IDENTICAL to the host-array-era class (names,
    order, defaults — snapshot.py derives its serialization surface from
    the dataclass fields); only the TYPES moved: host arrays -> device
    tensors, host scalar norms -> 0-dim float32 tensors; python scalars
    stay python scalars.
    """

    kind: str
    d: int
    # 0-dim float32 TENSOR for partition in {"half", "outlier"} — the fp
    # norm, never quantized (paper recipe); never a python float, never a
    # host scalar (the `.item` era is over). W29 partition="units": a
    # (units,) 1-D float32 TENSOR — the PER-UNIT norms vector (one fp32
    # norm per quantization unit — the paper's per-vector recipe applied
    # per unit; the dynamic-range repair, FINDINGS W28 item 9).
    norm: torch.Tensor
    bits_lo: int                           # bits for the lo set (1..8)
    bits_hi: int                           # bits for the hi set (== bits_lo if uniform)
    n_lo: int                              # coordinate count in the lo set
    n_hi: int                              # coordinate count in the hi set
    idx_lo: torch.Tensor                   # packed uint8 bit-stream (device-resident)
    idx_hi: torch.Tensor                   # packed uint8 bit-stream (device-resident)
    seed: int                              # the D3 rotation seed (persisted)
    partition: str = "half"                # "half" (legacy) | "outlier" (W15) | "units" (W29)
    # W15 — the paper's outlier split (partition="outlier", conv kind):
    # group   = coords per channel (the conv kernel width; 1 = flat).
    #           Present IFF partition == "outlier" (backward compatible).
    # mask    = packed per-channel outlier membership (ceil(n_ch/8) uint8
    #           tensor; bit i = 1 -> channel i is an OUTLIER (hi set).
    #           Present IFF partition == "outlier".
    # norm_hi = the outlier sub-set's norm (the regular set's norm is
    #           `norm`, parallel to idx_lo/idx_hi) — a 0-dim float32
    #           tensor. Present IFF partition == "outlier". For "half"
    #           codes the single `norm` field holds the whole unit's norm
    #           (unchanged legacy contract).
    # NOTE: for "outlier" codes, n_lo/n_hi are the PADDED sub-quantizer
    # unit sizes (power-of-two per sub-set — see TurboQuant._quant_split);
    # the real-coordinate counts are (n_ch - k)*group and k*group with
    # k = mask bit-count, n_ch = 8*len(mask).
    group: Optional[int] = None
    mask: Optional[torch.Tensor] = None      # packed membership (uint8)
    norm_hi: Optional[torch.Tensor] = None   # 0-dim float32
    # W29 — the per-head/multi-unit mode (partition="units", the S kind):
    # units = the number of quantization units carried (the production S
    #         case: 32 per-head units of d=16,384). Present IFF
    #         partition == "units". In that mode `norm` is the (units,)
    #         per-unit norms vector, n_lo/n_hi are PER-UNIT counts, and
    #         idx_lo/idx_hi are the row-major concatenation of the M
    #         per-unit packed streams (each _packed_len(n, bits) bytes —
    #         quant_units guarantees the per-unit byte alignment).
    units: Optional[int] = None
    # W9.2 qjl A/B (DEFAULT None = flag OFF / legacy codes): the
    # structured QJL residual sketch — (d,) int8 ±1 = sign(FHT(r/γ,
    # seed+7777)) — and the fp32 residual norm γ. Present IFF quantized
    # with qjl=True; serialization writes them ONLY when present
    # (backward compatible).
    qjl_signs: Optional[torch.Tensor] = None  # (d,) int8 ±1 sketch
    gamma: Optional[torch.Tensor] = None      # 0-dim float32

    # ---- construction discipline -----------------------------------------
    def __post_init__(self) -> None:
        """Type cops at the boundary: pure metadata checks — no device
        sync, no value reads; cheap enough for the per-unit hot path.
        The staged W20.2/W20.3 hand-off means the NumPy-typed
        construction sites in turboquant.py fail HERE by design until
        W20.3 rewrites them onto torch."""
        _chk_str("kind", self.kind)
        _chk_int("d", self.d)
        if self.partition == "units":
            # W29: the per-unit norms vector — (units,) fp32, one norm per
            # quantization unit. A 0-dim norm on a units unit (or a
            # mismatched length) is a corruption, refused loudly HERE.
            if self.units is None:
                raise TypeError(
                    "TQCodes.units: partition='units' requires the units "
                    "count (got None) — the multi-unit codes are "
                    "self-describing (W29)")
            _chk_int("units", self.units)
            if self.units < 1:
                raise TypeError(
                    f"TQCodes.units must be >= 1, got {self.units}")
            _chk_tensor("norm", self.norm, torch.float32, ndim=1)
            if self.norm.shape[0] != self.units:
                raise TypeError(
                    f"TQCodes.norm must be a ({self.units},) fp32 "
                    f"per-unit norms vector, got shape "
                    f"{tuple(self.norm.shape)} (partition='units', W29)")
        else:
            _chk_tensor("norm", self.norm, torch.float32, ndim=0)
            if self.units is not None:
                raise TypeError(
                    f"TQCodes.units: present ({self.units}) but "
                    f"partition={self.partition!r} — the units count rides "
                    f"ONLY on partition='units' codes (W29; a stray units "
                    f"count on a {self.partition!r} unit is a corruption)")
        _chk_int("bits_lo", self.bits_lo)
        _chk_int("bits_hi", self.bits_hi)
        _chk_int("n_lo", self.n_lo)
        _chk_int("n_hi", self.n_hi)
        _chk_tensor("idx_lo", self.idx_lo, torch.uint8)
        _chk_tensor("idx_hi", self.idx_hi, torch.uint8)
        _chk_int("seed", self.seed)
        _chk_str("partition", self.partition)
        if self.group is not None:
            _chk_int("group", self.group)
        if self.mask is not None:
            _chk_tensor("mask", self.mask, torch.uint8)
        if self.norm_hi is not None:
            _chk_tensor("norm_hi", self.norm_hi, torch.float32, ndim=0)
        if self.qjl_signs is not None:
            _chk_tensor("qjl_signs", self.qjl_signs, torch.int8)
        if self.gamma is not None:
            _chk_tensor("gamma", self.gamma, torch.float32, ndim=0)

    # ---- the torch container surface --------------------------------------
    def to(self, device: Union[str, torch.device, int],
           **kwargs: Any) -> "TQCodes":
        """torch.Tensor.to semantics for the tensor fields: returns a NEW
        TQCodes with every tensor field moved (a same-device move shares
        storage, exactly like the tensor op); scalar fields carry over
        unchanged; kwargs (e.g. non_blocking=True) pass straight through.
        __post_init__ re-runs on the result, so a dtype-changing kwarg
        fails loudly instead of corrupting the contract."""
        return replace(
            self,
            norm=self.norm.to(device, **kwargs),
            idx_lo=self.idx_lo.to(device, **kwargs),
            idx_hi=self.idx_hi.to(device, **kwargs),
            mask=(None if self.mask is None
                  else self.mask.to(device, **kwargs)),
            norm_hi=(None if self.norm_hi is None
                     else self.norm_hi.to(device, **kwargs)),
            qjl_signs=(None if self.qjl_signs is None
                       else self.qjl_signs.to(device, **kwargs)),
            gamma=(None if self.gamma is None
                   else self.gamma.to(device, **kwargs)),
        )

    @property
    def device(self) -> torch.device:
        """The resident device — idx_lo's (every tensor field shares it
        by construction). Metadata only; no sync."""
        return self.idx_lo.device

    def nbytes(self) -> int:
        """Total bytes of the tensor payload — the SAME accounting as the
        host-array-era class (both idx streams + 4 bytes per fp32 norm;
        the mask and the qjl sketch only when present). W29: a units
        container carries `units` fp32 norms (4 bytes each — the per-unit
        norms vector). Metadata only."""
        n = _tnbytes(self.idx_lo) + _tnbytes(self.idx_hi)
        n += 4 * (self.units if self.partition == "units" else 1)
        if self.partition == "outlier":   # W15: mask + second fp32 norm
            if self.mask is not None:
                n += _tnbytes(self.mask)
            n += 4
        if self.qjl_signs is not None:      # W9.2 A/B: +1 bit/coordinate
            n += _tnbytes(self.qjl_signs)   # (d,) int8 sketch
        if self.gamma is not None:
            n += 4                           # fp32 gamma
        return n

    def __repr__(self) -> str:
        # sync-free by construction: metadata only — NO tensor value dumps
        return (f"TQCodes(kind={self.kind!r}, d={self.d}, "
                f"bits=({self.bits_lo}, {self.bits_hi}), "
                f"n=({self.n_lo}, {self.n_hi}), device={self.device}, "
                f"nbytes={self.nbytes()})")

    # ---- the npz disk edge (NumPy lives HERE and ONLY here) --------------
    def to_arrays(self) -> "Dict[str, np.ndarray]":
        """Flat array dict for embedding into a larger npz (snapshot.py).

        The qjl fields are written ONLY when present, so codes quantized
        without the flag produce the EXACT pre-W9.2 key set (old readers
        and old digests are unaffected). Key set, dtypes, insertion order
        and conditional-key discipline replicate the pre-W20 class
        1:1 — device tensors take ONE hop to host here, at the edge."""
        import numpy as np  # the disk edge — NumPy exists only here (+ from_arrays)

        def _np(t):  # torch -> host array: one hop, no value reads
            return t.detach().to("cpu").numpy()

        out = {
            "kind": np.array(self.kind),
            "d": np.array(self.d, dtype=np.int64),
            "norm": np.array(_np(self.norm), dtype=np.float32),
            "bits_lo": np.array(self.bits_lo, dtype=np.int64),
            "bits_hi": np.array(self.bits_hi, dtype=np.int64),
            "n_lo": np.array(self.n_lo, dtype=np.int64),
            "n_hi": np.array(self.n_hi, dtype=np.int64),
            "seed": np.array(self.seed, dtype=np.int64),
            "partition": np.array(self.partition),
            "idx_lo": np.ascontiguousarray(_np(self.idx_lo),
                                           dtype=np.uint8),
            "idx_hi": np.ascontiguousarray(_np(self.idx_hi),
                                           dtype=np.uint8),
        }
        if self.partition == "outlier":
            if self.group is not None:
                out["group"] = np.array(int(self.group), dtype=np.int64)
            if self.mask is not None:
                out["mask"] = np.ascontiguousarray(_np(self.mask),
                                                   dtype=np.uint8)
            if self.norm_hi is not None:
                out["norm_hi"] = np.array(_np(self.norm_hi),
                                          dtype=np.float32)
        if self.partition == "units":
            # W29: the units count rides the array dict — the norm member
            # is ALREADY the (units,) per-unit vector above; readers that
            # do not know partition='units' never meet the key (old
            # readers/digests unaffected).
            out["units"] = np.array(int(self.units), dtype=np.int64)
        if self.qjl_signs is not None:
            out["qjl_signs"] = np.ascontiguousarray(_np(self.qjl_signs),
                                                    dtype=np.int8)
        if self.gamma is not None:
            out["gamma"] = np.array(_np(self.gamma), dtype=np.float32)
        return out

    @classmethod
    def from_arrays(cls, a: "Dict[str, np.ndarray]") -> "TQCodes":
        """Inverse of to_arrays (host arrays -> torch tensors, resident
        on CPU; move them with .to(device)). BACKWARD COMPATIBLE: an
        array dict written WITHOUT the qjl keys (pre-W9.2 npz/snapshots)
        loads with qjl_signs/gamma == None — the flag-off dequant path.
        Scalars are int()-ed here AT the disk edge (legal: the inputs
        are host arrays by contract — never a device tensor)."""
        import numpy as np  # the disk edge (inverse direction)

        qjl = a.get("qjl_signs") if hasattr(a, "get") else None
        gam = a.get("gamma") if hasattr(a, "get") else None
        part = str(a.get("partition", "half")) if hasattr(a, "get") else "half"
        grp = a.get("group") if hasattr(a, "get") else None
        msk = a.get("mask") if hasattr(a, "get") else None
        nhi = a.get("norm_hi") if hasattr(a, "get") else None
        nunits = a.get("units") if hasattr(a, "get") else None
        # W29: the norm member is the (units,) per-unit vector on units
        # codes (the 0-dim scalar otherwise); np.asarray handles both
        # shapes — __post_init__ validates against the partition.
        return cls(
            kind=str(a["kind"]),
            d=int(a["d"]),
            norm=torch.as_tensor(np.asarray(a["norm"], dtype=np.float32)),
            bits_lo=int(a["bits_lo"]),
            bits_hi=int(a["bits_hi"]),
            n_lo=int(a["n_lo"]),
            n_hi=int(a["n_hi"]),
            idx_lo=torch.as_tensor(
                np.ascontiguousarray(a["idx_lo"], dtype=np.uint8)),
            idx_hi=torch.as_tensor(
                np.ascontiguousarray(a["idx_hi"], dtype=np.uint8)),
            seed=int(a["seed"]),
            partition=part,
            units=(None if nunits is None else int(nunits)),
            group=(None if grp is None else int(grp)),
            mask=(None if msk is None
                  else torch.as_tensor(
                      np.ascontiguousarray(msk, dtype=np.uint8))),
            norm_hi=(None if nhi is None
                     else torch.as_tensor(
                         np.asarray(nhi, dtype=np.float32))),
            qjl_signs=(None if qjl is None
                       else torch.as_tensor(
                           np.ascontiguousarray(qjl, dtype=np.int8))),
            gamma=(None if gam is None
                   else torch.as_tensor(
                       np.asarray(gam, dtype=np.float32))),
        )
