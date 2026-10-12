"""install.py — the §6 install math (SPECIFICATION §6; PROPOSAL D4).

Install = sum the retrieved chunks' TurboQuant code DELTAS on top of the
system reset point, per S layer and per global memory, then requantize the
SUM ONCE:

    install(sys, d1..dn) = TQ.quant( dequant(sys) + Σ_i dequant(d_i) )

This is exactly "sum in the rotated space" from spec §6's note: the FHT is
linear and every delta of a kind shares the D3 rotation, so dequant → sum →
single requant is the one-round implementation (never per-addend). Lloyd-Max
is NOT additive — Q(a+b) ≠ Q(a)+Q(b) — which is why raw codes are never
summed and this module deliberately offers no code-plus-code operation.

conv_state is NEVER summed (spec §6: "use the last retrieved chunk's" —
installed verbatim).

W16 — THE ABSOLUTE-PROTOCOL INSTALL (protocol="absolute" snapshots, the
new default): the chunks carry the cache's OWN end-of-chunk codes (zero
extra quantization at ingest). The §6 sum contract translates exactly:

    target = sys + Σ_i (abs_i − sys) = Σ_i abs_i − (n−1)·sys

  n == 1   the target IS abs_1 — installed VERBATIM (bit-exact, zero
           requant rounds). The W16 noise decomposition measured the
           delta path's capture+install rounds at ~70% of the write-path
           distortion (0.0699 of 0.0994 at the rig); the verbatim install
           removes both rounds — the installed state equals the ingest
           end state exactly.
  n >= 2   quant( Σ dequant(abs_i) − (n−1)·dequant(sys) ) — ONE requant,
           the same N23 contract, one fewer addend-noise round than the
           delta path (which dequants sys + n deltas and requants).
  legacy   protocol="delta-v1" snapshots keep the original path below,
           bit-identical (backward compat for every pre-W16 corpus).

W20.4.2 — THE TORCH-NATIVE INSTALL: TQCodes are device tensors now
(tq_codes rev-1 — NumPy lives only in to_arrays/from_arrays, the npz
edge) and this module never names a host array: every dequant is
device-parametric (W20.3.3 — each codes decodes ON ITS OWN DEVICE; no
forced transfer, no host round trip, no `.item` read); the §6 fold
keeps the legacy accumulation order and dtype EXACTLY (fp32:
dequant(sys) first, then + dequant(addend_i) in retrieval order — one
left fold); the single requant rides TurboQuant.quant on the
accumulation device and returns a torch TQCodes. The npz loads ride
snapshot.py's edge (load_chunk rebuilds every unit through the
from_arrays hop; ingest's load_system_state delegates to the same
loader), so install.py opens no npz itself — the NumPy zero-audit
holds by construction. The W16 verbatim install is preserved EXACTLY:
CODE-COPY, never recompute — bit-exact within a backend (the n==0/n==1
paths hand the codes OBJECT over; zero tensor traffic). One new
torch-era guard: a unit's frame must be co-resident on ONE device —
the device analogue of the d/seed frame guards (a mixed-device fold
would die deep inside torch's add; it dies HERE, loudly, naming the
addend).

W20.10.1 — THE BATCHED REQUANT CHAIN (PROPOSAL §3.6, the K4 serving
side): on the kernel path (CUDA codes + the extension's TQ pair + an
eligible geometry — the dispatcher's own predicate) a fold's units
dequantize through ONE tq_dequant_batched call (the units' packed idx
streams stack into the entry's (M, L) rows, the 0-dim norms into the
(M,) vector), the sum is ONE torch reduction over the batched rows,
and the single requant rides the dispatcher's q.quant → tq_quant_batched
(W20.8.2: CUDA routes to the kernel ALWAYS when eligible). One kernel
chain per unit-kind fold instead of one launch per addend. On every
other box (CPU, pre-build, ineligible geometry, sketch-carrying codes)
the fold is the legacy left fold VERBATIM — the torch twin path,
bit-identical to the pre-W20.10 install, which the CPU gate suite pins.
The W16 verbatim install (n<=1 hand-offs) is untouched: CODE-COPY,
never recompute, bit-exact within a backend.

W26 — THE GDN BEST-CHUNK INSTALL (FINDINGS.md F1/F4, the E-ladder's
fix arm, opt-in): the GatedDeltaNet recurrence
S ← λ(I−βkkᵀ)S + βkvᵀ carries erase+decay, so independently-captured
chunk states are NOT additive — the n>=2 §6 S sum installs a state no
trajectory ever produced (scripts/demo_install_algebra.py: n=1
rel-MSE 0.0000, n=3 rel-MSE 2.63 vs the 0.15 corruption budget).
install_snapshot(..., gdn_install="best") installs the S codes
VERBATIM from the highest-scoring retrieved chunk (the n==1 path's
bit-exact hand-off — zero requant rounds) and takes conv from the
SAME chunk (F4: retrieval order is score-descending, so the sum
path's "last" chunk is the LOWEST-scoring one). M1/M2 stay summed
either way — their write is additive and the superposition is exact
(m1m2.py D4; the READ side is FINDINGS F3's business, not the
install's). gdn_install="sum" (the default) keeps the §6 contract
bit-identical — the E1-E3 ladder (FINDINGS.md §E) measures the
broken path on it; flip to "best" only after E1 confirms.

Frame guard: every code must carry the kind's D3 seed — a mismatched frame
would silently corrupt the sum (PROPOSAL D3); it raises loudly here.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

import _paths  # noqa: F401
from ingest import SystemState, load_system_state
from snapshot import ChunkSnapshot
from tq_cache import TQCache, TQLinearAttentionLayer, resolve_quantizer
from turboquant import TQCodes, _load_tq_kernel, _tq_kernel_available

__all__ = ["sum_turboquant_codes", "install_snapshot", "install_from_disk",
           "sum_absolute_codes"]


def _dequant_units_batched(q, units: Sequence[TQCodes]):
    """W20.10.1 — ONE tq_dequant_batched over a fold's units (the serving
    batch, PROPOSAL §3.6): the units' packed idx streams stack into the
    entry's (M, L) rows, the 0-dim fp32 norms into the (M,) vector, and
    the whole frame decode rides ONE stream-ordered kernel chain — the
    same call turboquant's _dequant_kernel makes at M=1 (row ownership
    is per-CTA and fixed-order, so each unit's reconstruction is the
    M=1 decode exactly; no host round trip of any kind).

    Returns the (M, d) fp32 rows on the units' device, or None when the
    dispatcher would NOT route this decode to the kernel — the caller
    then keeps the legacy left fold (the torch twin path, bit-identical
    to the pre-W20.10 install). The routing decision is the
    dispatcher's own predicate (q._dequant_kernel_eligible on the first
    unit: CUDA codes + the kernel pair through the ONE
    _tq_kernel_available probe + flat partition + no QJL sketch +
    eligible geometry), evaluated with the dispatcher's call order —
    the per-unit _check_codes belt runs FIRST (metadata-only, loud
    frame drift), mirroring _dequant_flat. Every unit must sit on the
    first unit's device (the mixed-device belt, _check_kernel_devices'
    discipline — the install guards already pin this; the belt names
    both devices loudly if a future caller forgets)."""
    C = _load_tq_kernel()
    if C is None or not units:
        return None
    first = units[0]
    if not _tq_kernel_available(first.idx_lo):
        return None
    for i, u in enumerate(units):
        q._check_codes(u)      # metadata-only; loud frame drift, the
        # dispatcher's own belt (_dequant_flat calls it before routing)
        if u.device != first.device:
            raise RuntimeError(
                f"install.py batched decode: unit {i} sits on {u.device} "
                f"but the fold's first unit sits on {first.device} — a "
                f"mixed-device frame; move the unit's codes to ONE device "
                f"(one sanctioned .to) before installing")
    if not q._dequant_kernel_eligible(first, torch.float32):
        return None
    dev = first.device
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)
    codes_lo = torch.stack([u.idx_lo for u in units]).contiguous()
    codes_hi = torch.stack([u.idx_hi for u in units]).contiguous()
    norms = torch.stack([u.norm for u in units])
    return C.tq_dequant_batched(
        codes_lo, codes_hi, norms, signs,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi,
        torch.float32)


def sum_turboquant_codes(system: TQCodes, deltas: Iterable[TQCodes],
                         kind: str, bits: float = 3.5,
                         qjl: bool = False) -> TQCodes:
    """dequant-sum-requant ONCE: quant(dequant(sys) + Σ dequant(deltas)).

    `kind` is the unit's kind ("S" | "M1" | "M2" — the caller knows it; the
    codes' own `.kind` may be "custom" at test scales but the SEED still
    identifies the frame). Raises on frame drift (seed mismatch).

    W15 `qjl`: when the cache runs the Alg.-2 A/B (TQCache(qjl=True)), the
    requantized SUM re-attaches the residual sketch (the deltas' own
    sketches are consumed by their dequants; the sum's fresh sketch
    covers the sum's residual).

    W20.4.2 torch: each codes dequantizes on ITS OWN device (the
    device-parametric decode — no forced transfer, no host hop); the
    requant lands on the accumulation device as a torch TQCodes. The
    frame must be co-resident — a delta on another device raises HERE
    (a metadata-only check, the device analogue of the d/seed guards),
    not deep inside torch's add.

    W20.10.1 — the batched requant chain: on the kernel path (CUDA +
    eligible geometry — the dispatcher's own predicate) the fold's
    units dequantize through ONE tq_dequant_batched call, the sum is
    ONE torch reduction over the batched rows, and the requant rides
    q.quant → tq_quant_batched (the dispatcher routes CUDA ALWAYS when
    eligible — verified, no re-plumbing needed). On every other box
    the fold below is the legacy left fold VERBATIM (fp32, system
    first, then the deltas in retrieval order — the torch twin path
    the CPU gate suite pins bit-exact). The signature is unchanged."""
    q = resolve_quantizer(kind, system.d, bits, qjl=qjl)
    if system.seed != q.seed:
        raise ValueError(
            f"sum_turboquant_codes({kind}): system codes seed {system.seed} "
            f"!= the kind's D3 seed {q.seed} — frame drift (PROPOSAL D3)")
    deltas = list(deltas)
    for i, d in enumerate(deltas):
        if d.d != system.d:
            raise ValueError(
                f"sum_turboquant_codes({kind}): delta {i} has d={d.d}, "
                f"system has d={system.d} — unit mismatch")
        if d.seed != system.seed:
            raise ValueError(
                f"sum_turboquant_codes({kind}): delta {i} seed {d.seed} != "
                f"frame seed {system.seed} — frame drift (PROPOSAL D3)")
        if d.device != system.device:
            raise ValueError(
                f"sum_turboquant_codes({kind}): delta {i} sits on {d.device} "
                f"but the frame's system codes sit on {system.device} — a "
                f"mixed-device frame; move the unit's codes to ONE device "
                f"(one sanctioned .to) before installing")
    rows = _dequant_units_batched(q, [system, *deltas])
    if rows is not None:
        # the batched chain: ONE torch sum over the stacked rows (the
        # stack order IS the legacy addend order — system first, then
        # the deltas in retrieval order; an empty fold is the (1, d)
        # sum, the system row exactly)
        acc = rows.sum(dim=0)
    else:
        # the legacy left fold (the torch twin path), verbatim
        acc = q.dequant(system)
        for d in deltas:
            acc = acc + q.dequant(d)
    # ONE requant of the sum — q.quant routes CUDA+eligible to
    # tq_quant_batched through the dispatcher (W20.8.2)
    return q.quant(acc)


def _last_conv(retrieved: Sequence[ChunkSnapshot], layer_idx: int
               ) -> Optional[TQCodes]:
    """spec §6: conv_state = the LAST retrieved chunk's codes (never summed)."""
    for snap in reversed(retrieved):
        if layer_idx in snap.conv_codes and snap.conv_codes[layer_idx] is not None:
            return snap.conv_codes[layer_idx]
    return None


def _best_conv(retrieved: Sequence[ChunkSnapshot], layer_idx: int
               ) -> Optional[TQCodes]:
    """W26 (FINDINGS F4): conv_state from the BEST (highest-scoring =
    retrieval-order FIRST) retrieved chunk's codes, never summed.

    Retrieval order is score-descending, so the §6 "last retrieved
    chunk" rule hands the answer the LOWEST-scoring chunk's tail
    window — and with gdn_install="best" the S state comes from the
    top-scoring chunk, so the conv must come from the SAME chunk or
    the first query tokens read a window belonging to a state that was
    never installed. Per-layer fallback mirrors _last_conv: the first
    snapshot in retrieval order that carries the layer's conv codes."""
    for snap in retrieved:
        if layer_idx in snap.conv_codes and snap.conv_codes[layer_idx] is not None:
            return snap.conv_codes[layer_idx]
    return None


def sum_absolute_codes(system: TQCodes, absolutes: Sequence[TQCodes],
                       kind: str, bits: float = 3.5,
                       qjl: bool = False) -> TQCodes:
    """The W16 absolute-protocol §6 sum: the D4 algebra with the chunks'
    OWN end codes as the addends.

        target = sys + Σ_i (abs_i − sys) = Σ_i abs_i − (n−1)·sys

    n == 0 → the system codes VERBATIM (the reseed identity — no requant,
    unlike the delta path's one-round requant of the empty sum).
    n == 1 → abs_0 VERBATIM (the target IS the chunk's end state — the
    bit-exact install; zero requant rounds).
    n >= 2 → ONE requant of the sum (the N23 contract).

    Frame guards (loud): every abs must carry the kind's D3 seed and the
    same unit d as the system codes; the bits must match the frame's bits
    (the codes are dequantized through the kind's quantizer — a bits
    mismatch would raise in _check_codes at READ time, so it is caught
    HERE, at the boundary).

    W20.4.2 torch: the verbatim paths are pure OBJECT hand-offs — the
    codes' tensor streams are neither copied nor recomputed (CODE-COPY,
    never recompute; bit-exact within a backend by construction); the
    n>=2 fold dequantizes each codes on its own device and keeps the
    legacy order (fp32, (1-n)·dequant(sys) first, then the absolutes
    in order), one requant on the accumulation device. Co-residency is
    guarded like the delta path (the n<=1 hand-offs are placement-
    agnostic — the cache's setters own placement).

    W20.10.1 — the n>=2 fold is the batched requant chain on the kernel
    path (ONE tq_dequant_batched over [sys, absolutes...], the (1-n)
    scale on the sys row, ONE torch sum over the absolutes' rows, the
    requant through the dispatcher's q.quant → tq_quant_batched); every
    other box keeps the legacy left fold VERBATIM (the torch twin,
    bit-identical). The n<=1 verbatim hand-offs are untouched."""
    q = resolve_quantizer(kind, system.d, bits, qjl=qjl)
    if system.seed != q.seed:
        raise ValueError(
            f"sum_absolute_codes({kind}): system codes seed {system.seed} "
            f"!= the kind's D3 seed {q.seed} — frame drift (PROPOSAL D3)")
    if not absolutes:
        return system
    absolutes = list(absolutes)
    if len(absolutes) == 1:
        a = absolutes[0]
        if a.d != system.d or a.seed != system.seed:
            raise ValueError(
                f"sum_absolute_codes({kind}): absolute codes "
                f"(d={a.d}, seed={a.seed}) != the frame "
                f"(d={system.d}, seed={system.seed}) — frame drift "
                f"(PROPOSAL D3)")
        if a.bits_lo != q.bits_lo or a.bits_hi != q.bits_hi:
            raise ValueError(
                f"sum_absolute_codes({kind}): verbatim install refused — "
                f"absolute codes bits ({a.bits_lo}, {a.bits_hi}) != the "
                f"frame's ({q.bits_lo}, {q.bits_hi}); a bits mismatch would "
                f"raise at READ time (dequant _check_codes); re-ingest the "
                f"corpus at the frame's bits")
        return a
    acc = q.dequant(system) * (1 - len(absolutes))
    for i, a in enumerate(absolutes):
        if a.d != system.d:
            raise ValueError(
                f"sum_absolute_codes({kind}): absolute {i} has d={a.d}, "
                f"system has d={system.d} — unit mismatch")
        if a.seed != system.seed:
            raise ValueError(
                f"sum_absolute_codes({kind}): absolute {i} seed {a.seed} != "
                f"frame seed {system.seed} — frame drift (PROPOSAL D3)")
        if a.device != system.device:
            raise ValueError(
                f"sum_absolute_codes({kind}): absolute {i} sits on "
                f"{a.device} but the frame's system codes sit on "
                f"{system.device} — a mixed-device frame; move the unit's "
                f"codes to ONE device (one sanctioned .to) before "
                f"installing")
    rows = _dequant_units_batched(q, [system, *absolutes])
    if rows is not None:
        # the batched chain: (1-n)·dequant(sys) + Σ dequant(abs_i) —
        # the sys row scales, the absolutes' stacked rows take ONE
        # torch sum (n == 2 is the single-row sum, the row exactly)
        acc = rows[0] * (1 - len(absolutes)) + rows[1:].sum(dim=0)
    else:
        # the legacy left fold (the torch twin path), verbatim
        acc = q.dequant(system) * (1 - len(absolutes))
        for a in absolutes:
            acc = acc + q.dequant(a)
    return q.quant(acc)


def install_snapshot(cache: TQCache, system: SystemState,
                     retrieved: Sequence[ChunkSnapshot],
                     gdn_install: str = "sum") -> dict:
    """The §6 install into `cache` (a live query cache — shapes known):

      S per layer:      sum_turboquant_codes(sys, [snap deltas], kind="S")
                        (delta-v1 snapshots) OR sum_absolute_codes(sys,
                        [snap absolutes], kind="S") (absolute snapshots —
                        VERBATIM when a single chunk is retrieved)
      M1 / M2:          same, kind="M1"/"M2"
      conv per layer:   the LAST retrieved chunk's codes, verbatim

    W26 (FINDINGS F1/F4): `gdn_install="best"` — S codes VERBATIM from
    the highest-scoring retrieved chunk (retrieval order is
    score-descending; the per-layer addend lists below filter None in
    that order, so their head IS the best chunk), conv from the SAME
    chunk (_best_conv), M1/M2 still summed (additive write, exact
    superposition). "sum" (the default) is the §6 contract above,
    bit-identical to the pre-W26 install — the E1-E3 ladder's baseline.

    Returns a report {layer: {"mode": ..., ...}} for the evals ledger
    ("mode" gains "verbatim"/"sum-abs" for the W16 absolute path and
    "best-verbatim"/"best-delta" for the W26 best path; best-mode S
    entries also carry "s_source_chunk" — the chunk id the S state
    came from, so the box log can verify it equals the top-1 retrieved
    id). The full-attn layers are untouched (spec §2.4).

    W16: the requant inherits the CACHE's qjl A/B setting (the codes are
    self-describing; a qjl cache requantizes the sum WITH the sketch so
    the installed state keeps the Alg.-2 compensation).

    W20.4.2 torch: the summed units are torch TQCodes (the requant
    rides the frame's device); the conv / M1 / M2 verbatim installs are
    OBJECT hand-offs — CODE-COPY, never recompute, bit-exact within a
    backend (the W16 gate contract, unchanged); the cache's setters own
    any placement (the W20.4.1 wiring moves codes at READ time). The
    report shape is the pinned W16 surface, byte-preserved."""
    bits = system.bits
    # the cache's Alg.-2 setting (any TQ linear layer's S quantizer; the
    # default False is bit-identical to the pre-W15 install)
    qjl = False
    for l in cache.layers:
        if isinstance(l, TQLinearAttentionLayer) and l._tq_s is not None:
            qjl = bool(l._tq_s.qjl)
            break
    # W16: dispatch on the snapshots' protocol — ALL the same (a mixed
    # retrieval would mean a mixed-protocol corpus: refuse loudly, the
    # manifest's chunk_protocol pins one layout per corpus)
    protocols = {s.protocol for s in retrieved}
    if len(protocols) > 1:
        raise ValueError(
            f"install_snapshot: mixed snapshot protocols {sorted(protocols)} "
            f"— a corpus carries ONE chunk protocol (re-ingest consistently)")
    absolute = bool(protocols) and "absolute" in protocols
    if gdn_install not in ("sum", "best"):
        raise ValueError(
            f"install_snapshot: gdn_install={gdn_install!r} — expected "
            f"'sum' (the §6 contract) or 'best' (the FINDINGS F1 remedy: "
            f"S+conv verbatim from the highest-scoring chunk, M1/M2 "
            f"summed)")
    report: dict = {}
    for L in sorted(system.s_codes):
        if absolute:
            absolutes = [s.s_codes[L] for s in retrieved
                         if L in s.s_codes and s.s_codes[L] is not None]
            # W26 (FINDINGS F1): the GDN S state is NON-ADDITIVE under
            # the delta-rule recurrence — in "best" mode the S codes
            # install VERBATIM from the highest-scoring chunk (the
            # n==1 hand-off: bit-exact, zero requant). ONLY the S
            # addends are sliced; M1/M2 below keep the full D4 sum.
            s_absolutes = absolutes
            best_mode = gdn_install == "best" and len(absolutes) > 1
            if best_mode:
                s_absolutes = absolutes[:1]
            summed = sum_absolute_codes(system.s_codes[L], s_absolutes,
                                        kind="S", bits=bits, qjl=qjl)
            if not absolutes:
                mode = "sys-verbatim"
            elif best_mode:
                mode = "best-verbatim"
            else:
                mode = "verbatim" if len(absolutes) == 1 else "sum-abs"
            report[L] = {"mode": mode, "n_chunks": len(retrieved),
                         "protocol": "absolute"}
            if best_mode:
                report[L]["s_source_chunk"] = next(
                    s.chunk_id for s in retrieved
                    if L in s.s_codes and s.s_codes[L] is not None)
        else:
            deltas = [s.s_codes[L] for s in retrieved
                      if L in s.s_codes and s.s_codes[L] is not None]
            # W26: the delta-protocol twin — in "best" mode only the
            # BEST chunk's delta installs (quant(sys + delta_best), one
            # requant); M1/M2 below keep the full sum.
            s_deltas = deltas
            best_mode = gdn_install == "best" and len(deltas) > 1
            if best_mode:
                s_deltas = deltas[:1]
            summed = sum_turboquant_codes(system.s_codes[L], s_deltas,
                                          kind="S", bits=bits, qjl=qjl)
            mode = "best-delta" if best_mode else "sum"
        cache.set_s_codes(L, summed)

        conv = _best_conv(retrieved, L) if gdn_install == "best" \
            else _last_conv(retrieved, L)
        conv_tag = "best-conv" if gdn_install == "best" else "last-conv"
        if conv is not None:
            cache.set_conv_codes(L, conv)
            if absolute:
                report[L]["mode"] = f"{mode}+{conv_tag}"
                report[L]["conv_partition"] = conv.partition
            else:
                # the legacy report shape, VERBATIM (the tests pin it)
                report[L] = {"mode": f"{mode}+{conv_tag}",
                             "n_deltas": len(deltas),
                             "conv_partition": conv.partition}
        elif not absolute:
            report[L] = {"mode": mode, "n_deltas": len(deltas),
                         "conv": "system"}
    if system.m1_codes is not None:
        if absolute:
            # W26: M1/M2 KEEP the full sum in "best" mode too — the
            # write is additive (m1m2.py D4); F1 owns only the GDN S.
            m1_abs = [s.m1_codes for s in retrieved if s.m1_codes is not None]
            cache.m1_codes = sum_absolute_codes(
                system.m1_codes, m1_abs, kind="M1", bits=bits, qjl=qjl)
            report["M1"] = {"mode": "verbatim" if len(m1_abs) == 1 else "sum-abs",
                            "n_chunks": len(m1_abs)}
        else:
            m1_deltas = [s.m1_codes for s in retrieved if s.m1_codes is not None]
            cache.m1_codes = sum_turboquant_codes(
                system.m1_codes, m1_deltas, kind="M1", bits=bits, qjl=qjl)
            report["M1"] = {"mode": "sum", "n_deltas": len(m1_deltas)}
    if system.m2_codes is not None:
        if absolute:
            m2_abs = [s.m2_codes for s in retrieved if s.m2_codes is not None]
            cache.m2_codes = sum_absolute_codes(
                system.m2_codes, m2_abs, kind="M2", bits=bits, qjl=qjl)
            report["M2"] = {"mode": "verbatim" if len(m2_abs) == 1 else "sum-abs",
                            "n_chunks": len(m2_abs)}
        else:
            m2_deltas = [s.m2_codes for s in retrieved if s.m2_codes is not None]
            cache.m2_codes = sum_turboquant_codes(
                system.m2_codes, m2_deltas, kind="M2", bits=bits, qjl=qjl)
            report["M2"] = {"mode": "sum", "n_deltas": len(m2_deltas)}
    return report


def install_from_disk(cache: TQCache, disk_dir: str,
                      retrieved_ids: Sequence[int],
                      bits: float = 3.5,
                      gdn_install: str = "sum") -> dict:
    """Convenience: load the reset point + chunk snapshots from disk and
    install (spec §6 steps 5–6). gdn_install passes through to
    install_snapshot (W26: "best" = the F1 remedy; default "sum").

    W20.4.2: the loads ride snapshot.py's disk edge — load_system_state
    and load_chunk both rebuild torch TQCodes through the from_arrays
    hop (the only NumPy in the chain, and it is NOT in this module);
    this function opens no npz itself."""
    import os
    system = load_system_state(disk_dir)
    if system.bits != bits:
        # the reset point's own bits win — a mismatch is a caller error
        raise ValueError(
            f"install_from_disk: bits={bits} but the persisted system state "
            f"was built at bits={system.bits}")
    snaps = []
    for cid in retrieved_ids:
        path = os.path.join(disk_dir, "snapshots", f"chunk_{int(cid):05d}.npz")
        if not os.path.exists(path):
            raise ValueError(f"install_from_disk: {path} not found")
        snaps.append(load_snapshot(path))
    return install_snapshot(cache, system, snaps, gdn_install=gdn_install)


def load_snapshot(path: str) -> ChunkSnapshot:
    """The disk-edge delegation (kept for the historical import surface):
    snapshot.py owns the npz read and the from_arrays hop — this module
    never touches a host array."""
    from snapshot import load_chunk
    return load_chunk(path)
