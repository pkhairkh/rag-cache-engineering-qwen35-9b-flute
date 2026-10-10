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

__all__ = ["sum_turboquant_codes", "install_snapshot", "install_from_disk"]


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


def install_snapshot(cache: TQCache, system: SystemState,
                     retrieved: Sequence[ChunkSnapshot]) -> dict:
    """The §6 install into `cache` (a live query cache — shapes known):

      S per layer:      sum_turboquant_codes(sys, [snap deltas], kind="S")
      M1 / M2:          same, kind="M1"/"M2"
      conv per layer:   the LAST retrieved chunk's codes, verbatim

    Returns a report {layer: {"mode": "sum"|"last"|"none", ...}} for the
    evals ledger. The full-attn layers are untouched (spec §2.4).

    W15: the requant inherits the CACHE's qjl A/B setting (the codes are
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
    report: dict = {}
    for L in sorted(system.s_codes):
        deltas = [s.s_codes[L] for s in retrieved
                  if L in s.s_codes and s.s_codes[L] is not None]
        summed = sum_turboquant_codes(system.s_codes[L], deltas,
                                      kind="S", bits=bits, qjl=qjl)
        cache.set_s_codes(L, summed)

        conv = _last_conv(retrieved, L)
        if conv is not None:
            cache.set_conv_codes(L, conv)
            report[L] = {"mode": "sum+last-conv", "n_deltas": len(deltas),
                         "conv_partition": conv.partition}
        else:
            report[L] = {"mode": "sum", "n_deltas": len(deltas),
                         "conv": "system"}
    if system.m1_codes is not None:
        m1_deltas = [s.m1_codes for s in retrieved if s.m1_codes is not None]
        cache.m1_codes = sum_turboquant_codes(
            system.m1_codes, m1_deltas, kind="M1", bits=bits, qjl=qjl)
        report["M1"] = {"mode": "sum", "n_deltas": len(m1_deltas)}
    if system.m2_codes is not None:
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
