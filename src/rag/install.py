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

Frame guard: every code must carry the kind's D3 seed — a mismatched frame
would silently corrupt the sum (PROPOSAL D3); it raises loudly here.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import _paths  # noqa: F401
from ingest import SystemState, load_system_state
from snapshot import ChunkSnapshot
from tq_cache import TQCache, TQLinearAttentionLayer, resolve_quantizer
from turboquant import TQCodes

__all__ = ["sum_turboquant_codes", "install_snapshot", "install_from_disk",
           "sum_absolute_codes"]


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
    covers the sum's residual)."""
    q = resolve_quantizer(kind, system.d, bits, qjl=qjl)
    if system.seed != q.seed:
        raise ValueError(
            f"sum_turboquant_codes({kind}): system codes seed {system.seed} "
            f"!= the kind's D3 seed {q.seed} — frame drift (PROPOSAL D3)")
    acc = q.dequant(system)
    for i, d in enumerate(deltas):
        if d.d != system.d:
            raise ValueError(
                f"sum_turboquant_codes({kind}): delta {i} has d={d.d}, "
                f"system has d={system.d} — unit mismatch")
        if d.seed != system.seed:
            raise ValueError(
                f"sum_turboquant_codes({kind}): delta {i} seed {d.seed} != "
                f"frame seed {system.seed} — frame drift (PROPOSAL D3)")
        acc = acc + q.dequant(d)
    return q.quant(acc)


def _last_conv(retrieved: Sequence[ChunkSnapshot], layer_idx: int
               ) -> Optional[TQCodes]:
    """spec §6: conv_state = the LAST retrieved chunk's codes (never summed)."""
    for snap in reversed(retrieved):
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
    HERE, at the boundary)."""
    q = resolve_quantizer(kind, system.d, bits, qjl=qjl)
    if system.seed != q.seed:
        raise ValueError(
            f"sum_absolute_codes({kind}): system codes seed {system.seed} "
            f"!= the kind's D3 seed {q.seed} — frame drift (PROPOSAL D3)")
    if not absolutes:
        return system
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
        acc = acc + q.dequant(a)
    return q.quant(acc)


def install_snapshot(cache: TQCache, system: SystemState,
                     retrieved: Sequence[ChunkSnapshot]) -> dict:
    """The §6 install into `cache` (a live query cache — shapes known):

      S per layer:      sum_turboquant_codes(sys, [snap deltas], kind="S")
                        (delta-v1 snapshots) OR sum_absolute_codes(sys,
                        [snap absolutes], kind="S") (absolute snapshots —
                        VERBATIM when a single chunk is retrieved)
      M1 / M2:          same, kind="M1"/"M2"
      conv per layer:   the LAST retrieved chunk's codes, verbatim

    Returns a report {layer: {"mode": ..., ...}} for the evals ledger
    ("mode" gains "verbatim"/"sum-abs" for the W16 absolute path). The
    full-attn layers are untouched (spec §2.4).

    W16: the requant inherits the CACHE's qjl A/B setting (the codes are
    self-describing; a qjl cache requantizes the sum WITH the sketch so
    the installed state keeps the Alg.-2 compensation)."""
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
    report: dict = {}
    for L in sorted(system.s_codes):
        if absolute:
            absolutes = [s.s_codes[L] for s in retrieved
                         if L in s.s_codes and s.s_codes[L] is not None]
            summed = sum_absolute_codes(system.s_codes[L], absolutes,
                                        kind="S", bits=bits, qjl=qjl)
            mode = ("verbatim" if len(absolutes) == 1 else "sum-abs") \
                if absolutes else "sys-verbatim"
            report[L] = {"mode": mode, "n_chunks": len(retrieved),
                         "protocol": "absolute"}
        else:
            deltas = [s.s_codes[L] for s in retrieved
                      if L in s.s_codes and s.s_codes[L] is not None]
            summed = sum_turboquant_codes(system.s_codes[L], deltas,
                                          kind="S", bits=bits, qjl=qjl)
            mode = "sum"
        cache.set_s_codes(L, summed)

        conv = _last_conv(retrieved, L)
        if conv is not None:
            cache.set_conv_codes(L, conv)
            if absolute:
                report[L]["mode"] = f"{mode}+last-conv"
                report[L]["conv_partition"] = conv.partition
            else:
                # the legacy report shape, VERBATIM (the tests pin it)
                report[L] = {"mode": f"{mode}+last-conv",
                             "n_deltas": len(deltas),
                             "conv_partition": conv.partition}
        elif not absolute:
            report[L] = {"mode": mode, "n_deltas": len(deltas),
                         "conv": "system"}
    if system.m1_codes is not None:
        if absolute:
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
                      bits: float = 3.5) -> dict:
    """Convenience: load the reset point + chunk snapshots from disk and
    install (spec §6 steps 5–6)."""
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
    return install_snapshot(cache, system, snaps)


def load_snapshot(path: str) -> ChunkSnapshot:
    from snapshot import load_chunk
    return load_chunk(path)
