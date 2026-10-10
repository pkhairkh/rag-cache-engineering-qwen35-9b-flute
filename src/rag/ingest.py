"""ingest.py — the delta-protocol ingestion + batch driver (SPECIFICATION
§5; PROPOSAL D4).

The protocol (D4, pinned to the toy-validated path-independent convention):

    reset point:   S_sys <- prefill(system) codes                (quantized once)
    per chunk i:   fresh cache, reseed linear codes from S_sys;
                   prefill(chunk_i)  (online TQ: every write requantizes)
                   delta_i = dequant(codes_i) - dequant(S_sys)   (per S layer,
                                                                M1, M2)
                   store TQ.quant(delta_i)                       (~6 MiB/chunk)
    retrieval:     the vector is dequantized from the ABSOLUTE codes in the
                   spec §4 order (S ascending, M1, M2) — computed on demand,
                   never stored (PROPOSAL D5: exact vectors exist only as
                   TQ codes on disk).
    install (§6):  dequant(S_sys) + sum_i dequant(delta_i), requantized ONCE
                   (rag/install.py) — never sums raw codes.
    conv_state:    ABSOLUTE codes stored per chunk (the §6 install rule:
                   "use the last retrieved chunk's" — conv is never summed).

Full-attention layers (spec §2.4): every chunk prefill starts with EMPTY
full-attn KV (fresh cache) — chunk states depend only on chunk tokens and
the reseeded linear state; the full-attn state converges within the prefill
and is never snapshotted.

The driver is restartable: a JSON manifest (out_dir/ingest_manifest.json)
records completed chunk ids; re-running skips them. Ingestion is
embarrassingly parallel over chunks (spec §5); this driver is the
sequential reference — the GPU box may shard it by chunk range.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import snapshot as snap_mod
import turboquant as tq
from snapshot import ChunkSnapshot
from turboquant import TQCodes
from tq_cache import TQCache, TQLinearAttentionLayer, resolve_quantizer

__all__ = [
    "SystemState", "ChunkRecord", "prefill_system", "reseed_cache",
    "ingest_chunk", "IngestDriver", "MANIFEST_NAME", "SYSTEM_FILE",
    "load_system_state", "m1m2_mem_size_from_system", "check_m1m2_geometry",
]

MANIFEST_NAME = "ingest_manifest.json"
SYSTEM_FILE = "system_state.npz"  # the persisted reset point (absolute protocol)


# ------------------------------------------------------------- system ------
@dataclass
class SystemState:
    """The D4 reset point: the system-prefill cache as TurboQuant codes.

    Carries the per-layer tensor SHAPES too: codes are flat (d,) — a fresh
    cache's layers have no shape metadata until a forward runs, so reseeding
    a fresh cache needs the geometry restored alongside the codes (else
    dequantized reads come back flat).
    """

    s_codes: Dict[int, TQCodes]
    conv_codes: Dict[int, TQCodes]
    m1_codes: Optional[TQCodes]
    m2_codes: Optional[TQCodes]
    system_ref: str = ""
    bits: float = 3.5
    s_shapes: Dict[int, tuple] = field(default_factory=dict)
    conv_shapes: Dict[int, tuple] = field(default_factory=dict)
    s_dtype: Optional[str] = None
    conv_dtype: Optional[str] = None
    m1_shape: Optional[tuple] = None
    m2_shape: Optional[tuple] = None

    @classmethod
    def from_cache(cls, cache: TQCache, system_ref: str = "",
                   bits: float = 3.5) -> "SystemState":
        codes = cache.snapshot_codes()
        missing = [L for L, c in codes["s"].items() if c is None]
        if missing:
            raise ValueError(
                f"SystemState.from_cache: layers {missing} have no S codes — "
                f"the system prefill did not run through the full stack")
        s_shapes, conv_shapes, s_dtype, conv_dtype = {}, {}, None, None
        for i, layer in enumerate(cache.layers):
            if isinstance(layer, TQLinearAttentionLayer):
                if layer._s_shape is not None:
                    s_shapes[i] = tuple(layer._s_shape)
                    s_dtype = str(layer._s_dtype).replace("torch.", "")
                if layer._conv_shape is not None:
                    conv_shapes[i] = tuple(layer._conv_shape)
                    conv_dtype = str(layer._conv_dtype).replace("torch.", "")
        return cls(
            s_codes=dict(codes["s"]),
            conv_codes={L: c for L, c in codes["conv"].items() if c is not None},
            m1_codes=codes["m1"], m2_codes=codes["m2"],
            system_ref=system_ref, bits=bits,
            s_shapes=s_shapes, conv_shapes=conv_shapes,
            s_dtype=s_dtype, conv_dtype=conv_dtype,
            m1_shape=tuple(cache._m1_shape) if cache._m1_shape else None,
            m2_shape=tuple(cache._m2_shape) if cache._m2_shape else None)

    def reference(self) -> str:
        """A stable string identifying this reset point (manifest field)."""
        if self.system_ref:
            return self.system_ref
        h = 0
        for L in sorted(self.s_codes):
            c = self.s_codes[L]
            h = (h * 1000003 + int(float(c.norm) * 1e6) + L) & 0xFFFFFFFF
        for c in (self.m1_codes, self.m2_codes):
            if c is not None:
                h = (h * 1000003 + int(float(c.norm) * 1e6)) & 0xFFFFFFFF
        return f"sysstate-{h:08x}"


def _save_system_state(out_dir: str, system: SystemState) -> str:
    """Persist the reset point as an ABSOLUTE-protocol chunk snapshot (the
    index builder and the install path both need S_sys codes + geometry from
    disk alone). Shapes/dtypes ride in `extra` (JSON-safe)."""
    path = os.path.join(out_dir, SYSTEM_FILE)
    snap = ChunkSnapshot(
        chunk_id=0, protocol="absolute",
        s_codes=system.s_codes, conv_codes=system.conv_codes,
        m1_codes=system.m1_codes, m2_codes=system.m2_codes,
        system_ref=system.reference(),
        extra={"s_shapes": {str(k): list(v) for k, v in system.s_shapes.items()},
               "conv_shapes": {str(k): list(v) for k, v in system.conv_shapes.items()},
               "s_dtype": system.s_dtype, "conv_dtype": system.conv_dtype,
               "m1_shape": list(system.m1_shape) if system.m1_shape else None,
               "m2_shape": list(system.m2_shape) if system.m2_shape else None,
               "bits": system.bits})
    return snap_mod.save_chunk(path, snap)


def load_system_state(disk_dir: str) -> SystemState:
    """Reconstruct the SystemState from disk (manifest's reset point)."""
    path = os.path.join(disk_dir, SYSTEM_FILE)
    if not os.path.exists(path):
        raise ValueError(
            f"load_system_state: {path} not found — run the IngestDriver "
            f"(the reset point is persisted on first run)")
    snap = snap_mod.load_chunk(path)
    if snap.protocol != "absolute":
        raise ValueError(
            f"load_system_state: protocol {snap.protocol!r} != 'absolute'")
    extra = snap.extra or {}
    return SystemState(
        s_codes=dict(snap.s_codes),
        conv_codes=dict(snap.conv_codes),
        m1_codes=snap.m1_codes, m2_codes=snap.m2_codes,
        system_ref=snap.system_ref or "",
        bits=float(extra.get("bits", 3.5)),
        s_shapes={int(k): tuple(v) for k, v in (extra.get("s_shapes") or {}).items()},
        conv_shapes={int(k): tuple(v) for k, v in (extra.get("conv_shapes") or {}).items()},
        s_dtype=extra.get("s_dtype"), conv_dtype=extra.get("conv_dtype"),
        m1_shape=tuple(extra["m1_shape"]) if extra.get("m1_shape") else None,
        m2_shape=tuple(extra["m2_shape"]) if extra.get("m2_shape") else None)


def m1m2_mem_size_from_system(system: SystemState) -> Optional[int]:
    """W17: the corpus's M1/M2 slot count from the persisted reset point
    (system_state.npz's m1_shape[1]) — None when the corpus was ingested
    WITHOUT M1/M2 (the m1/m2 units absent).

    The query-side model must load the SAME mem_size (the module's state
    geometry vs the cache's codes — drift fails loudly at the first
    forward); the GPU tools resolve it from disk so a query run can never
    silently mismatch the ingested geometry."""
    shape = getattr(system, "m1_shape", None)
    if shape is not None and len(shape) == 3:
        return int(shape[1])
    return None


def check_m1m2_geometry(model, system: SystemState,
                        mem_size: Optional[int]) -> None:
    """W17: loud cross-check of the loaded model's M1/M2 module against
    the corpus geometry (the reset point's m1_shape) — an actionable
    error BEFORE the first forward (the deep guard is the M1M2
    write/read shape validation, which fires mid-prefill otherwise)."""
    inner = getattr(model, "model", model)
    module = getattr(inner, "m1m2", None)
    have = system.m1_shape if system.m1_shape is not None else None
    if module is None:
        if have is not None:
            raise ValueError(
                "check_m1m2_geometry: the corpus was ingested WITH M1/M2 "
                f"(m1 shape {tuple(have)}) but this model was loaded "
                "without the wiring — load with use_m1m2=True")
        return
    want = tuple(module.state_shape()) if module is not None else None
    if have is None:
        if mem_size is not None and module.mem_size != int(mem_size):
            pass  # no corpus pin; the explicit mem_size is the contract
        return
    if tuple(have) != tuple(want):
        raise ValueError(
            f"check_m1m2_geometry: the model's M1/M2 state {tuple(want)} "
            f"!= the corpus reset point's {tuple(have)} — reload the "
            f"model with --m1m2-mem-size {int(have[1])} (ingest and "
            f"query MUST agree on the geometry)")


def reseed_cache(cache: TQCache, system: SystemState) -> None:
    """Reset the linear layers of `cache` to the system reset point: codes
    re-installed WITH their geometry (shapes/dtypes — a fresh cache's layers
    have no shape metadata until a forward runs), M1/M2 token counters
    zeroed, full-attn KV untouched (use a fresh cache per chunk for the
    empty-KV contract)."""
    import torch as _torch
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, TQLinearAttentionLayer):
            layer.reset()
            if i in system.conv_codes:
                # restore the window geometry FIRST so the codes setter can
                # infer conv_kernel_size (reseeded layers skip lazy init)
                if i in system.conv_shapes:
                    layer._conv_shape = tuple(system.conv_shapes[i])
                    layer.conv_kernel_size[0] = int(system.conv_shapes[i][-1])
                if system.conv_dtype:
                    layer._conv_dtype = getattr(_torch, system.conv_dtype)
                layer.conv_codes = system.conv_codes[i]
            if i in system.s_codes:
                if i in system.s_shapes:
                    layer._s_shape = tuple(system.s_shapes[i])
                if system.s_dtype:
                    layer._s_dtype = getattr(_torch, system.s_dtype)
                layer.s_codes = system.s_codes[i]
    cache.m1_codes = system.m1_codes
    cache.m2_codes = system.m2_codes
    # the codes setters stamp a flat (d,) shape on fresh caches — restore
    # the real geometry AFTER the assignment (unconditionally: system knows)
    if system.m1_shape is not None:
        cache._m1_shape = tuple(system.m1_shape)
    if system.m2_shape is not None:
        cache._m2_shape = tuple(system.m2_shape)


def prefill_system(model, system_token_ids: torch.Tensor, cache: TQCache,
                   system_ref: str = "", bits: float = 3.5) -> SystemState:
    """Prefill the system prompt on a FRESH cache; return the reset point.

    The fresh-cache contract matters: the system point is the zero-point of
    the delta protocol — a cache that already ran content would offset every
    chunk delta (loudly refuse rather than silently corrupt)."""
    if any(isinstance(l, TQLinearAttentionLayer) and l.s_codes is not None
           for l in cache.layers):
        raise ValueError(
            "prefill_system: the cache already holds S codes — pass a FRESH "
            "cache (the system prefill defines the delta protocol's zero)")
    with torch.no_grad():
        model(input_ids=system_token_ids, past_key_values=cache,
              use_cache=True)
    return SystemState.from_cache(cache, system_ref=system_ref, bits=bits)


# ------------------------------------------------------------- per chunk ----
@dataclass
class ChunkRecord:
    """One ingested chunk: the D4 delta codes + the W16 ABSOLUTE codes.

    W16: `abs_s`/`abs_m1`/`abs_m2` are the cache's OWN end-of-chunk codes
    (references — captured with ZERO extra quantization rounds). Storing
    them (protocol="absolute") removes the delta-capture and install
    requant rounds — the W16 noise decomposition measured those two rounds
    at 70% of the write-path distortion (capture+install 0.0699 of the
    0.0994 total at the rig; scripts/w16_probe_noise.py). The DELTAS are
    still computed and remain the retrieval-frame representation (the
    centered retrieval frame scores dequant(abs) - dequant(sys)).
    """
    chunk_idx: int
    n_tokens: int
    delta_s: Dict[int, TQCodes]
    conv_codes: Dict[int, TQCodes]           # ABSOLUTE (last-chunk-wins, §6)
    delta_m1: Optional[TQCodes]
    delta_m2: Optional[TQCodes]
    cache_vector: np.ndarray                 # fp32, §4 order — transient (D5)
    abs_s: Dict[int, TQCodes] = field(default_factory=dict)
    abs_m1: Optional[TQCodes] = None
    abs_m2: Optional[TQCodes] = None

    def to_snapshot(self, system: SystemState,
                    protocol: str = "delta-v1") -> ChunkSnapshot:
        """protocol "delta-v1" (legacy): the D4 delta codes — the install
        runs the dequant-sum-requant chain (2 extra rounds). protocol
        "absolute" (W16 default): the cache's own end codes — the
        single-chunk install is VERBATIM (bit-exact, 0 rounds); the deltas
        are recoverable at read time as dequant(abs) - dequant(sys)."""
        if protocol == "absolute":
            s_codes = self.abs_s
            m1, m2 = self.abs_m1, self.abs_m2
        elif protocol == "delta-v1":
            s_codes = self.delta_s
            m1, m2 = self.delta_m1, self.delta_m2
        else:
            raise ValueError(
                f"ChunkRecord.to_snapshot: protocol {protocol!r} not in "
                f"('delta-v1', 'absolute')")
        return ChunkSnapshot(
            chunk_id=self.chunk_idx, protocol=protocol,
            s_codes=s_codes, conv_codes=self.conv_codes,
            m1_codes=m1, m2_codes=m2,
            system_ref=system.reference(),
            extra={"n_tokens": self.n_tokens})


def _dequant(kind: str, codes: TQCodes, bits: float) -> torch.Tensor:
    return resolve_quantizer(kind, codes.d, bits).dequant(codes)


def ingest_chunk(model, chunk_token_ids: torch.Tensor, cache: TQCache,
                 system: SystemState) -> ChunkRecord:
    """Spec §5's ingest, D4 protocol (see module docstring). The cache is
    reseeded from the system point, the chunk is prefilled (online TQ —
    every state write requantizes), and the snapshot is read back as CODES
    (never fp16)."""
    bits = system.bits
    reseed_cache(cache, system)
    with torch.no_grad():
        model(input_ids=chunk_token_ids, past_key_values=cache,
              use_cache=True)
    codes = cache.snapshot_codes()  # {"s": {L: codes}, "conv": ..., "m1", "m2"}
    missing = [L for L in system.s_codes if codes["s"].get(L) is None]
    if missing:
        raise ValueError(
            f"ingest_chunk: layers {missing} produced no S codes during the "
            f"chunk prefill — the model did not run the full stack")

    delta_s: Dict[int, TQCodes] = {}
    for L in sorted(system.s_codes):
        q = resolve_quantizer("S", codes["s"][L].d, bits)
        d_vec = (_dequant("S", codes["s"][L], bits)
                 - _dequant("S", system.s_codes[L], bits))
        delta_s[L] = q.quant(d_vec)

    delta_m1 = delta_m2 = None
    if codes["m1"] is not None and system.m1_codes is not None:
        q = resolve_quantizer("M1", codes["m1"].d, bits)
        delta_m1 = q.quant(_dequant("M1", codes["m1"], bits)
                           - _dequant("M1", system.m1_codes, bits))
    if codes["m2"] is not None and system.m2_codes is not None:
        q = resolve_quantizer("M2", codes["m2"].d, bits)
        delta_m2 = q.quant(_dequant("M2", codes["m2"], bits)
                           - _dequant("M2", system.m2_codes, bits))

    # the retrieval vector: dequantized ABSOLUTE codes, §4 order
    pieces = [_dequant("S", codes["s"][L], bits) for L in sorted(system.s_codes)]
    if codes["m1"] is not None:
        pieces.append(_dequant("M1", codes["m1"], bits))
    if codes["m2"] is not None:
        pieces.append(_dequant("M2", codes["m2"], bits))
    cache_vector = torch.cat(pieces).to(torch.float32).numpy()

    return ChunkRecord(
        chunk_idx=-1, n_tokens=int(chunk_token_ids.shape[-1]),
        delta_s=delta_s,
        conv_codes={L: c for L, c in codes["conv"].items() if c is not None},
        delta_m1=delta_m1, delta_m2=delta_m2,
        cache_vector=cache_vector,
        # W16: the cache's own end codes — ZERO extra quantization rounds
        abs_s=dict(codes["s"]),
        abs_m1=codes["m1"], abs_m2=codes["m2"])


# ------------------------------------------------------------- driver -------
class IngestDriver:
    """The restartable batch driver (spec §5): system prefill once, then
    per chunk — fresh cache, reseed, prefill, delta, save, manifest update.

    chunks: an iterable of (chunk_idx, token_ids) OR a list of token-id
    tensors (indices 0..n-1). cache_factory: () -> TQCache — REQUIRED unless
    the model exposes a usable `.config` (then a TQCache(config=...) is
    built per chunk). return_records=True keeps the records (vectors) in
    memory — TEST-ONLY at small dims; at production the vectors are
    re-derived from the codes by the index builder (D5, never stored).
    """

    def __init__(self, model, system_token_ids: torch.Tensor,
                 chunks: Sequence, out_dir: str,
                 cache_factory: Optional[Callable[[], TQCache]] = None,
                 bits: float = 3.5, system_ref: str = "",
                 resume: bool = True, return_records: bool = False,
                 chunk_protocol: str = "absolute"):
        self.model = model
        self.system_token_ids = system_token_ids
        self.chunks = self._normalize(chunks)
        self.out_dir = os.path.abspath(out_dir)
        self.cache_factory = cache_factory or self._default_cache_factory
        self.bits = bits
        self.system_ref = system_ref
        self.resume = resume
        self.return_records = return_records
        # W16: "absolute" (default) — the chunk snapshots carry the cache's
        # own end codes (the verbatim install, 0 extra rounds; the W16
        # noise decomposition: the delta path's 2 rounds own ~70% of the
        # write-path distortion). "delta-v1" keeps the D4 legacy layout.
        if chunk_protocol not in ("absolute", "delta-v1"):
            raise ValueError(
                f"IngestDriver: chunk_protocol {chunk_protocol!r} not in "
                f"('absolute', 'delta-v1')")
        self.chunk_protocol = chunk_protocol
        self.records: List[ChunkRecord] = []

    # ------------------------------------------------------------ plumbing -
    def _default_cache_factory(self) -> TQCache:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            raise ValueError(
                "IngestDriver: no cache_factory and the model has no .config "
                "— pass cache_factory=TQCache(layer_types=[...]) (tests) or "
                "a real model")
        return TQCache(config=cfg, bits=self.bits)

    @staticmethod
    def _normalize(chunks) -> List[tuple]:
        items = list(chunks)
        if items and isinstance(items[0], (tuple, list)) and len(items[0]) == 2:
            return [(int(i), t) for i, t in items]
        return list(enumerate(items))

    def _manifest_path(self) -> str:
        return os.path.join(self.out_dir, MANIFEST_NAME)

    def _load_manifest(self) -> dict:
        path = self._manifest_path()
        if self.resume and os.path.exists(path):
            with open(path) as f:
                man = json.load(f)
            # W16: a resumed run must keep writing the SAME chunk protocol
            # (a mixed corpus would force per-snapshot dispatch at install
            # and strand the manifest's meaning)
            old = man.get("chunk_protocol")
            if old is not None and old != self.chunk_protocol:
                raise ValueError(
                    f"IngestDriver resume: the manifest's chunk_protocol "
                    f"{old!r} != this run's {self.chunk_protocol!r} — a "
                    f"re-protocol re-ingest needs a fresh out_dir or "
                    f"chunk_protocol={old!r}")
            return man
        return {"protocol": "delta-v1", "bits": self.bits,
                "system_ref": None, "done": {}, "vector_dims": None,
                "chunk_protocol": self.chunk_protocol}

    def _save_manifest(self, man: dict) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        tmp = self._manifest_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        os.replace(tmp, self._manifest_path())

    # ---------------------------------------------------------------- run --
    def run(self) -> dict:
        """Ingest all chunks. Returns stats (and keeps .records when
        return_records). Idempotent under resume: completed chunk ids are
        skipped, the manifest is rewritten atomically after every chunk."""
        os.makedirs(self.out_dir, exist_ok=True)
        man = self._load_manifest()
        # the system prefill: once per run (the reset point must be stable
        # across resumes — the manifest pins its reference)
        if man.get("system_ref") is None:
            sys_cache = self.cache_factory()
            system = prefill_system(self.model, self.system_token_ids,
                                    sys_cache, system_ref=self.system_ref,
                                    bits=self.bits)
            man["system_ref"] = system.reference()
            man["system_file"] = os.path.relpath(
                _save_system_state(self.out_dir, system), self.out_dir)
        else:
            # resume: rebuild the reset point from a fresh prefill and
            # CHECK it matches the manifest reference (deterministic
            # prefill => same codes => same reference)
            sys_cache = self.cache_factory()
            system = prefill_system(self.model, self.system_token_ids,
                                    sys_cache, system_ref=self.system_ref,
                                    bits=self.bits)
            if system.reference() != man["system_ref"]:
                raise ValueError(
                    f"IngestDriver resume: system reset point drifted "
                    f"(manifest {man['system_ref']!r} vs now "
                    f"{system.reference()!r}) — the system prompt or model "
                    f"changed; refusing to corrupt the delta protocol")

        stats = {"n_chunks": len(self.chunks), "done_before": len(man["done"]),
                 "ingested": 0, "skipped": 0, "seconds": 0.0}
        t0 = time.time()
        for idx, token_ids in self.chunks:
            if str(idx) in man["done"]:
                stats["skipped"] += 1
                continue
            cache = self.cache_factory()
            record = ingest_chunk(self.model, token_ids, cache, system)
            record.chunk_idx = idx
            snap = record.to_snapshot(system, protocol=self.chunk_protocol)
            path = snap_mod.save_chunk(self.out_dir, snap)
            man["done"][str(idx)] = os.path.relpath(path, self.out_dir)
            man["vector_dims"] = int(record.cache_vector.shape[0])
            man["chunk_protocol"] = self.chunk_protocol
            self._save_manifest(man)
            stats["ingested"] += 1
            if self.return_records:
                self.records.append(record)
        stats["seconds"] = time.time() - t0
        return stats
