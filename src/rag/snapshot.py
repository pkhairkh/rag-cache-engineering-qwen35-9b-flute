"""snapshot.py — the per-chunk TurboQuant-code disk format (SPECIFICATION.md
§5 ingestion + §11 disk layout): ONE uncompressed npz per chunk at
<disk>/snapshots/chunk_XXXXX.npz, ~6 MiB, holding the 24 S codes + 24 conv
codes + M1 codes + M2 codes — TurboQuant CODES ONLY ("NO chunk text. NO
token IDs. NO fp16", spec §5).

THE D5 RESOLUTION (spec-internal tension, resolved by PROPOSAL D5): spec
§5's pseudo-code shows `cache_vector` (24 x 524,288 + 2 x 524,288 =
13,631,488 fp32 = 54.5 MB at production dims) saved INSIDE the npz — but
the §5 disk table, the §11 layout and PROPOSAL D5 pin ~6 MiB per chunk and
"exact vectors exist only as TQ codes on disk, dequantized for the rerank
one candidate at a time". RESOLUTION: the on-disk format is CODES-ONLY;
the retrieval vector is computed on demand from the codes (dequantize +
concat in the §4 order — see hooks.CacheSnapshot.capture_vector) and is
NEVER stored. `include_vector=True` exists strictly as a DEBUG option that
can be flipped explicitly — it writes a ~54.5 MB (52 MiB) fp32 member at
production dims, ~9x the whole 6 MiB chunk budget, and VIOLATES the D5
disk contract; never enable it in ingest/production paths.

THE D4 DELTA PROTOCOL (what ingest stores per chunk): DELTA codes for
S/M1/M2 relative to the system-prompt state and ABSOLUTE codes for
conv_state (spec §6 install: "use the last retrieved chunk's" — never
summed). This codec is PROTOCOL-AGNOSTIC: it stores whatever code dicts it
is given and records the convention in the `protocol` metadata string
("delta-v1" default, "absolute" allowed). The deltas themselves are
computed by the ingestion driver (rag/ingest.py, W5.2) — never here.

NPZ LAYOUT (flat keys; np.savez UNCOMPRESSED — codes are incompressible
bit-streams, entropy coding rejected at ~5% gain for b=4 per PROPOSAL):

    meta            0-d JSON string: {version, chunk_id, protocol,
                    system_ref, extra, sha256, n_s_layers, saved_at}
    s_{L}_{f}       TQCodes.to_arrays() fields f for S layer L
                    (s_0_kind, s_0_d, s_0_norm, ..., s_0_idx_lo, s_0_idx_hi)
    conv_{L}_{f}    same for the conv codes of layer L
    m1_{f}, m2_{f}  same for the global memories (omitted when None)
    vector          1-D fp32 — ONLY when include_vector=True (DEBUG; D5)

INTEGRITY: the meta's `sha256` digests ALL code bytes (the packed idx
streams, the fp32 norms and every scalar field of every unit, in canonical
order S layers ascending -> conv layers ascending -> M1 -> M2, each unit
tagged so reordering/renaming cannot collide). load_chunk/verify_chunk
recompute the digest and compare, raising loudly on mismatch. The digest
deliberately EXCLUDES `saved_at` (and the debug vector): the same inputs
always produce the same sha256, call after call.

DETERMINISM: np.savez pins member timestamps at the ZIP epoch (1980-01-01)
and this module emits members in a fixed order (meta, S layers ascending,
conv layers ascending, m1, m2, vector), so the same inputs give
byte-identical files except the single `saved_at` UTC ISO string inside
the meta JSON (kept for provenance — the one call-varying field, noted
here per the format contract).

SIZE ARITHMETIC (the 6 MiB gate, spec §5 table + §11; MEASURED, this box):
one 3.5-bit unit at d=524,288 costs 262,144*3/8 + 262,144*4/8 + 4 (fp32
norm) = 229,380 B; at d=32,768 it costs 14,340 B. Full chunk code bytes:
24*229,380 + 24*14,340 + 2*229,380 = 6,308,040 B exactly (6.016 MiB — the
spec's "Total per chunk ~6.0 MiB", same number W1.4's gate 6 pins for the
codes). The npz CONTAINER adds a fixed, deterministic 141.5 KiB — the flat
§5 key layout means 551 members (meta + 50 units x 11 fields), each costing
~263 B of .npy header (128 B, 64-byte-aligned) + zip bookkeeping (~135 B:
local header + data descriptor + central-directory entry) — so the measured
FILE is 6,452,958 B = 6.154 MiB. Gate used here: file in [5.9, 6.2] MiB AND
code bytes == 6,308,040 B (the [5.9, 6.1] MiB budget holds for the CODES,
the budget the spec's component table actually sums; the 2.3% container
overhead is format-inherent and does not scale with corpus semantics —
note the spec's OWN §11 line "50,000 x 6 MiB = ~300 GiB" implies
~6.44 MB/chunk, which the measured 6.45 MB matches).

This module is the CODEC only: stdlib + numpy in the codec logic (importing
TQCodes from turboquant pulls torch transitively — the existing
serialization API, used as-is; no torch calls of our own). Dequantization
and retrieval math live in turboquant/tq_cache/hooks; the delta computation
and the 24-layer capture live in W5.2's ingest driver.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, fields as _dc_fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

import _paths  # noqa: F401  (house convention: anchors src/rag, src/scripts, src/flute_extended)
from turboquant import TQCodes

__all__ = [
    "SNAPSHOT_VERSION", "DEFAULT_PROTOCOL", "PROTOCOLS", "ChunkSnapshot",
    "snapshot_path", "save_chunk", "load_chunk", "verify_chunk", "chunk_nbytes",
]

# ------------------------------------------------------------- constants ---
SNAPSHOT_VERSION = 1

# D4 delta-protocol storage labels (the codec stores either; ingest decides).
DEFAULT_PROTOCOL = "delta-v1"            # s/m1/m2 deltas, conv absolute
PROTOCOLS = ("delta-v1", "absolute")

_NPZ_EXT = ".npz"

# The TQCodes serialization surface (exactly the to_arrays()/from_arrays
# keys — derived from the dataclass so a field change fails loudly here).
_CODE_FIELDS: tuple = tuple(f.name for f in _dc_fields(TQCodes))

# meta JSON keys (the §5 format contract).
_META_KEYS = ("chunk_id", "protocol", "system_ref", "extra", "version",
              "sha256", "n_s_layers", "saved_at")

_LAYER_RE = {"s": re.compile(r"^s_(\d+)_"), "conv": re.compile(r"^conv_(\d+)_")}
_CHUNK_NAME_RE = re.compile(r"^chunk_(\d{5,})\.npz$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PathLike = Union[str, "os.PathLike[str]"]


# -------------------------------------------------------------- snapshot ---
@dataclass
class ChunkSnapshot:
    """One chunk's TurboQuant codes as stored on disk (spec §5).

    chunk_id      the chunk index — the 5-digit file name chunk_XXXXX is
                  derived from it (snapshot_path; ids >= 100_000 simply
                  grow past five digits and still round-trip).
    protocol      "delta-v1" (default) — s_codes/m1/m2 are DELTAS relative
                  to the system-prompt state (D4); "absolute" — they are
                  absolute states. conv_codes are ABSOLUTE codes under
                  either protocol (spec §6 install: last retrieved chunk's,
                  never summed). The codec stores whatever codes it is
                  given; the delta computation belongs to W5.2's driver.
    s_codes       linear layer idx -> TQCodes (production: the 24 spec §1
                  linear indices 0,1,2,4,5,6,...,30).
    conv_codes    linear layer idx -> TQCodes (production: 24 entries).
    m1_codes,
    m2_codes      the global memory codes (None allowed — omitted from the
                  npz entirely).
    system_ref    free-form reference to the system reset point the deltas
                  are relative to (hash, path, tag...); informational.
    extra         caller metadata (e.g. token count). MUST be JSON-safe:
                  str keys and JSON scalars/lists/dicts as values (tuples
                  come back as lists through the round-trip).

    Comparison: TQCodes fields contain numpy arrays, so compare snapshots
    field-wise (idx streams with np.array_equal, norms bit-exactly) —
    never with == (array ambiguity).
    """

    chunk_id: int
    protocol: str = DEFAULT_PROTOCOL
    s_codes: Dict[int, TQCodes] = field(default_factory=dict)
    conv_codes: Dict[int, TQCodes] = field(default_factory=dict)
    m1_codes: Optional[TQCodes] = None
    m2_codes: Optional[TQCodes] = None
    system_ref: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        _validate_snapshot(self, "ChunkSnapshot")


def _check_chunk_id(chunk_id: Any, ctx: str) -> None:
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
        raise TypeError(
            f"{ctx}: chunk_id must be a non-negative int (the 5-digit "
            f"chunk_XXXXX index), got {type(chunk_id).__name__}: {chunk_id!r}")
    if chunk_id < 0:
        raise ValueError(f"{ctx}: chunk_id must be >= 0, got {chunk_id}")


def _validate_snapshot(snap: "ChunkSnapshot", ctx: str) -> None:
    """Loud structural validation (runs at construction AND at save —
    dataclass fields are mutable between the two)."""
    _check_chunk_id(snap.chunk_id, ctx)
    if snap.protocol not in PROTOCOLS:
        raise ValueError(
            f"{ctx}: protocol must be one of {PROTOCOLS} (D4 delta-protocol "
            f"storage: 'delta-v1' = s/m1/m2 deltas + absolute conv, "
            f"'absolute' = all absolute), got {snap.protocol!r}")
    for name, codes_map in (("s_codes", snap.s_codes),
                            ("conv_codes", snap.conv_codes)):
        if not isinstance(codes_map, Mapping):
            raise TypeError(
                f"{ctx}: {name} must be a mapping layer_idx -> TQCodes, "
                f"got {type(codes_map).__name__}")
        for layer_idx, codes in codes_map.items():
            if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) \
                    or layer_idx < 0:
                raise TypeError(
                    f"{ctx}: {name} keys must be non-negative int layer "
                    f"indices, got {layer_idx!r}")
            if not isinstance(codes, TQCodes):
                raise TypeError(
                    f"{ctx}: {name}[{layer_idx}] must be a TQCodes object, "
                    f"got {type(codes).__name__} (codes only — never fp "
                    f"tensors, spec §5)")
    for name, codes in (("m1_codes", snap.m1_codes), ("m2_codes", snap.m2_codes)):
        if codes is not None and not isinstance(codes, TQCodes):
            raise TypeError(
                f"{ctx}: {name} must be a TQCodes object or None, got "
                f"{type(codes).__name__}")
    if snap.system_ref is not None and not isinstance(snap.system_ref, str):
        raise TypeError(
            f"{ctx}: system_ref must be a str or None (free-form system "
            f"reset-point reference), got {type(snap.system_ref).__name__}")
    if snap.extra is not None:
        if not isinstance(snap.extra, Mapping):
            raise TypeError(
                f"{ctx}: extra must be a JSON-safe dict or None, got "
                f"{type(snap.extra).__name__}")
        for k in snap.extra:
            if not isinstance(k, str):
                raise TypeError(
                    f"{ctx}: extra keys must be str (JSON object keys), "
                    f"got {k!r}")


# ------------------------------------------------------------- integrity ---
def _unit_digest_bytes(tag: str, codes: TQCodes) -> bytes:
    """Canonical byte encoding of ONE code unit for the integrity digest.

    head = 'tag|kind|d|bits_lo|bits_hi|n_lo|n_hi|seed|partition' (text,
    unambiguous — no field may contain '|' or NUL) + NUL, then the 4 exact
    fp32 norm bytes, then the idx_lo/idx_hi bit-stream bytes (their lengths
    are derivable from the head, so the stream parses unambiguously).
    """
    head = "|".join((
        tag, str(codes.kind), str(int(codes.d)),
        str(int(codes.bits_lo)), str(int(codes.bits_hi)),
        str(int(codes.n_lo)), str(int(codes.n_hi)),
        str(int(codes.seed)), str(codes.partition),
    )).encode("utf-8")
    norm = np.asarray(codes.norm, dtype=np.float32).tobytes()
    lo = np.ascontiguousarray(codes.idx_lo, dtype=np.uint8).tobytes()
    hi = np.ascontiguousarray(codes.idx_hi, dtype=np.uint8).tobytes()
    return head + b"\x00" + norm + lo + hi


def _digest_codes(s_codes: Mapping[int, TQCodes],
                  conv_codes: Mapping[int, TQCodes],
                  m1: Optional[TQCodes],
                  m2: Optional[TQCodes]) -> str:
    """sha256 over ALL code bytes in canonical order (S ascending, conv
    ascending, M1, M2). Covers exactly what the codec stores — never
    saved_at, never the debug vector — so it is call-invariant."""
    h = hashlib.sha256()
    for layer_idx in sorted(s_codes):
        h.update(_unit_digest_bytes(f"s:{layer_idx}", s_codes[layer_idx]))
    for layer_idx in sorted(conv_codes):
        h.update(_unit_digest_bytes(f"conv:{layer_idx}",
                                    conv_codes[layer_idx]))
    if m1 is not None:
        h.update(_unit_digest_bytes("m1", m1))
    if m2 is not None:
        h.update(_unit_digest_bytes("m2", m2))
    return h.hexdigest()


def _packed_len(n: int, bits: int) -> int:
    return (n * bits + 7) // 8


# ----------------------------------------------------------------- paths ---
def snapshot_path(disk_dir: PathLike, chunk_id: int) -> str:
    """The spec §11 layout path: <disk_dir>/snapshots/chunk_{id:05d}.npz.

    Pure function — no directories are created (save_chunk does that).
    """
    _check_chunk_id(chunk_id, "snapshot_path")
    return os.path.join(os.fspath(disk_dir), "snapshots",
                        f"chunk_{chunk_id:05d}{_NPZ_EXT}")


def chunk_nbytes(path: PathLike) -> int:
    """Actual file size of a chunk snapshot (os.path.getsize)."""
    p = os.fspath(path)
    try:
        return os.path.getsize(p)
    except OSError as e:
        raise FileNotFoundError(
            f"chunk_nbytes: cannot stat chunk snapshot {p!r}: {e}") from e


# ------------------------------------------------------------------ save ---
def save_chunk(path_or_disk_dir: PathLike, snap: ChunkSnapshot,
               include_vector: bool = False,
               vector: Optional[np.ndarray] = None) -> str:
    """Write ONE chunk snapshot npz (spec §5/§11) and return its path.

    path_or_disk_dir is either a DISK DIR (the spec §11 root — the file is
    then <dir>/snapshots/chunk_XXXXX.npz, both directories created as
    needed) or an explicit .npz FILE PATH (written verbatim). An existing
    non-npz file is rejected loudly.

    include_vector : bool, default False
        *** DEBUG ONLY — THE D5 RESOLUTION ***  The production format is
        CODES-ONLY: the retrieval vector is recomputed from the codes and
        NEVER stored (spec §5 table + §11 pin ~6 MiB/chunk; PROPOSAL D5:
        "exact vectors exist only as TQ codes on disk").  Setting
        include_vector=True writes the fp32 retrieval vector into the npz
        — at production dims that is 13,631,488 x 4 B ~= 54.5 MB, ~9x the
        entire 6 MiB chunk budget, and violates the disk contract.  Use
        only for offline codec debugging; never in ingest/production.

    vector : np.ndarray | None
        The 1-D fp32 retrieval vector (§4 order: dequantized S layers
        ascending, M1, M2 — conv is NOT part of it). Required when
        include_vector=True, forbidden otherwise (a vector passed with
        include_vector=False is a caller bug — refused loudly, never
        silently dropped).
    """
    _validate_snapshot(snap, "save_chunk")
    p = os.fspath(path_or_disk_dir)
    if os.path.isdir(p):
        path = snapshot_path(p, snap.chunk_id)
    elif p.endswith(_NPZ_EXT):
        path = p
    elif os.path.exists(p):
        raise ValueError(
            f"save_chunk: {p!r} exists and is a file without the {_NPZ_EXT} "
            f"extension — pass either a disk dir (the spec §11 "
            f"<disk>/snapshots/chunk_XXXXX.npz path is derived from "
            f"chunk_id) or an explicit .npz file path")
    else:  # a disk dir that does not exist yet
        path = snapshot_path(p, snap.chunk_id)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # ---- the debug vector (validated BEFORE any disk write) --------------
    vec: Optional[np.ndarray] = None
    if include_vector:
        if vector is None:
            raise ValueError(
                "save_chunk: include_vector=True requires the `vector` "
                "argument (the 1-D fp32 retrieval vector) — DEBUG ONLY, "
                "violates the 6 MiB codes-only budget (D5)")
        try:
            vec = np.asarray(vector, dtype=np.float32)
        except Exception as e:
            raise TypeError(
                f"save_chunk: vector must be a 1-D numeric array: {e}") from e
        if vec.ndim != 1:
            raise ValueError(
                f"save_chunk: vector must be 1-D (§4: concat of dequantized "
                f"S, M1, M2), got shape {vec.shape}")
        want = sum(codes.d for codes in snap.s_codes.values())
        if snap.m1_codes is not None:
            want += snap.m1_codes.d
        if snap.m2_codes is not None:
            want += snap.m2_codes.d
        if vec.shape[0] != want:
            raise ValueError(
                f"save_chunk: vector length {vec.shape[0]} != expected "
                f"{want} (sum of S/M1/M2 code dims; conv codes are not part "
                f"of the §4 retrieval vector)")
    elif vector is not None:
        raise ValueError(
            "save_chunk: vector was passed but include_vector=False — "
            "refusing to silently drop it (the on-disk format is "
            "CODES-ONLY per D5)")

    # ---- meta (JSON; sha256 covers the CODE bytes only) ------------------
    digest = _digest_codes(snap.s_codes, snap.conv_codes,
                           snap.m1_codes, snap.m2_codes)
    meta = {
        "version": SNAPSHOT_VERSION,
        "chunk_id": snap.chunk_id,
        "protocol": snap.protocol,
        "system_ref": snap.system_ref,
        "extra": snap.extra,
        "sha256": digest,
        "n_s_layers": len(snap.s_codes),
        # the ONLY call-varying field (provenance; excluded from the digest)
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"save_chunk: ChunkSnapshot.extra is not JSON-safe: {e}") from e

    # ---- flat code members (deterministic order) --------------------------
    arrays: Dict[str, np.ndarray] = {"meta": np.array(meta_json)}
    for layer_idx in sorted(snap.s_codes):
        for key, val in snap.s_codes[layer_idx].to_arrays().items():
            arrays[f"s_{layer_idx}_{key}"] = val
    for layer_idx in sorted(snap.conv_codes):
        for key, val in snap.conv_codes[layer_idx].to_arrays().items():
            arrays[f"conv_{layer_idx}_{key}"] = val
    for name, codes in (("m1", snap.m1_codes), ("m2", snap.m2_codes)):
        if codes is not None:
            for key, val in codes.to_arrays().items():
                arrays[f"{name}_{key}"] = val
    if vec is not None:
        arrays["vector"] = vec

    np.savez(path, **arrays)
    return path


# ------------------------------------------------------------------ load ---
def _read_meta_json(z: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    """Parse the 'meta' member (JSON object). Raises loudly on anything
    that is not exactly the §5 meta string."""
    if "meta" not in z.files:
        raise ValueError(
            "load_chunk: npz has no 'meta' member — not a chunk snapshot "
            "(spec §5 format)")
    raw = np.asarray(z["meta"])
    if raw.dtype.kind != "U" or raw.size != 1:
        raise ValueError(
            f"load_chunk: 'meta' must be a single JSON string, got "
            f"dtype={raw.dtype}, size={raw.size}")
    try:
        meta = json.loads(str(raw.item()))
    except json.JSONDecodeError as e:
        raise ValueError(f"load_chunk: 'meta' is not valid JSON: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(
            f"load_chunk: 'meta' JSON must be an object, got "
            f"{type(meta).__name__}")
    return meta


def _check_meta(meta: Mapping[str, Any], path: str) -> None:
    """Loud validation of the meta object: keys, version, protocol, types,
    digest shape, and file-name-vs-meta chunk_id consistency."""
    missing = [k for k in _META_KEYS if k not in meta]
    if missing:
        raise ValueError(
            f"load_chunk: 'meta' is missing required keys {missing} "
            f"(spec §5 format)")
    version = meta["version"]
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"load_chunk: snapshot version {version!r} != SNAPSHOT_VERSION "
            f"{SNAPSHOT_VERSION} — on-disk format changed; re-ingest or "
            f"migrate this corpus (spec §5)")
    protocol = meta["protocol"]
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"load_chunk: unknown protocol {protocol!r} — expected one of "
            f"{PROTOCOLS} (D4 delta-protocol storage)")
    chunk_id = meta["chunk_id"]
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) \
            or chunk_id < 0:
        raise ValueError(
            f"load_chunk: meta chunk_id must be a non-negative int, got "
            f"{chunk_id!r}")
    sha = meta["sha256"]
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        raise ValueError(
            f"load_chunk: meta sha256 must be a 64-hex-char digest, got "
            f"{sha!r}")
    n_s = meta["n_s_layers"]
    if isinstance(n_s, bool) or not isinstance(n_s, int) or n_s < 0:
        raise ValueError(
            f"load_chunk: meta n_s_layers must be a non-negative int, got "
            f"{n_s!r}")
    if not (meta["system_ref"] is None or isinstance(meta["system_ref"], str)):
        raise ValueError(
            f"load_chunk: meta system_ref must be a str or None, got "
            f"{meta['system_ref']!r}")
    if not (meta["extra"] is None or isinstance(meta["extra"], dict)):
        raise ValueError(
            f"load_chunk: meta extra must be a JSON object or null, got "
            f"{meta['extra']!r}")
    if not isinstance(meta["saved_at"], str):
        raise ValueError(
            f"load_chunk: meta saved_at must be a str, got "
            f"{meta['saved_at']!r}")
    m = _CHUNK_NAME_RE.fullmatch(os.path.basename(path))
    if m is not None and int(m.group(1)) != chunk_id:
        raise ValueError(
            f"load_chunk: file name says chunk_{int(m.group(1)):05d} but "
            f"meta says chunk_id {chunk_id} — renamed/overwritten file?")


def _layer_ids(z: np.lib.npyio.NpzFile, prefix: str) -> List[int]:
    """Sorted layer indices discovered under 's_'/'conv_' members."""
    pat = _LAYER_RE[prefix]
    out: set = set()
    for key in z.files:
        m = pat.match(key)
        if m is not None:
            out.add(int(m.group(1)))
    return sorted(out)


def _unit_state(z: np.lib.npyio.NpzFile, name: str) -> bool:
    """True iff the m1/m2 unit is fully present; LOUD on partial units
    (a truncated unit must never silently read as 'absent')."""
    prefix = f"{name}_"
    present = [k for k in z.files if k.startswith(prefix)]
    if not present:
        return False
    missing = [f"{prefix}{f}" for f in _CODE_FIELDS
               if f"{prefix}{f}" not in z.files]
    if missing:
        raise ValueError(
            f"load_chunk: {name} codes are incomplete — members {sorted(present)} "
            f"are present but {missing} is/are missing (corrupt npz?)")
    return True


def _check_members(z: np.lib.npyio.NpzFile, s_layers: List[int],
                   conv_layers: List[int], has_m1: bool, has_m2: bool) -> None:
    """The npz member set must be EXACTLY the §5 layout — stray members
    (which the sha256 digest does not cover) are a tampering signal."""
    allowed = {"meta", "vector"}
    for layer_idx in s_layers:
        allowed.update(f"s_{layer_idx}_{f}" for f in _CODE_FIELDS)
    for layer_idx in conv_layers:
        allowed.update(f"conv_{layer_idx}_{f}" for f in _CODE_FIELDS)
    if has_m1:
        allowed.update(f"m1_{f}" for f in _CODE_FIELDS)
    if has_m2:
        allowed.update(f"m2_{f}" for f in _CODE_FIELDS)
    stray = sorted(set(z.files) - allowed)
    if stray:
        raise ValueError(
            f"load_chunk: unknown npz members {stray} — not part of the §5 "
            f"chunk format (tampered or foreign file?)")


def _check_layer_count(meta: Mapping[str, Any], s_layers: List[int]) -> None:
    if int(meta["n_s_layers"]) != len(s_layers):
        raise ValueError(
            f"load_chunk: meta says n_s_layers={meta['n_s_layers']} but the "
            f"npz holds {len(s_layers)} S units — truncated/tampered file?")


def _read_unit(z: np.lib.npyio.NpzFile, prefix: str) -> TQCodes:
    """Rebuild one TQCodes unit from its prefixed members, with structural
    validation BEFORE from_arrays (which would silently cast)."""
    label = prefix[:-1]  # e.g. "s_3", "conv_12", "m1"
    missing = [f for f in _CODE_FIELDS if f"{prefix}{f}" not in z.files]
    if missing:
        raise ValueError(
            f"load_chunk: unit '{label}' is incomplete — missing members "
            f"{missing} (corrupt npz?)")
    raw = {f: z[f"{prefix}{f}"] for f in _CODE_FIELDS}
    for f in ("idx_lo", "idx_hi"):
        a = np.asarray(raw[f])
        if a.dtype != np.uint8 or a.ndim != 1:
            raise ValueError(
                f"load_chunk: unit '{label}' {f} must be a 1-D uint8 "
                f"bit-stream, got dtype={a.dtype}, ndim={a.ndim}")
    n_lo, n_hi = int(raw["n_lo"]), int(raw["n_hi"])
    bits_lo, bits_hi = int(raw["bits_lo"]), int(raw["bits_hi"])
    for f, n, b in (("idx_lo", n_lo, bits_lo), ("idx_hi", n_hi, bits_hi)):
        want = _packed_len(n, b)
        got = np.asarray(raw[f]).shape[0]
        if got != want:
            raise ValueError(
                f"load_chunk: unit '{label}' {f} holds {got} bytes, expected "
                f"ceil({n}*{b}/8) = {want} (corrupt npz?)")
    try:
        return TQCodes.from_arrays(raw)
    except Exception as e:
        raise ValueError(
            f"load_chunk: cannot rebuild TQCodes for unit '{label}': {e}") \
            from e


def load_chunk(path: PathLike) -> ChunkSnapshot:
    """Full reconstruction of a chunk snapshot (spec §5): meta validation
    (version, protocol, keys, file-name consistency), per-unit TQCodes
    rebuild, and the sha256 integrity comparison — raises LOUDLY on any
    mismatch/corruption.

    The debug `vector` member (if one was saved) is intentionally NOT
    reconstructed: the production retrieval vector is recomputed from the
    codes (D5) — access it directly via np.load(path)["vector"] if needed.
    """
    p = os.fspath(path)
    z = np.load(p, allow_pickle=False)
    if not hasattr(z, "files"):
        raise ValueError(
            f"load_chunk: {p!r} is not an npz archive (got "
            f"{type(z).__name__}) — chunk snapshots are "
            f"snapshots/chunk_XXXXX.npz (spec §11)")
    with z:
        meta = _read_meta_json(z)
        _check_meta(meta, p)
        s_layers = _layer_ids(z, "s")
        conv_layers = _layer_ids(z, "conv")
        has_m1 = _unit_state(z, "m1")
        has_m2 = _unit_state(z, "m2")
        _check_members(z, s_layers, conv_layers, has_m1, has_m2)
        _check_layer_count(meta, s_layers)
        s_codes = {layer_idx: _read_unit(z, f"s_{layer_idx}_")
                   for layer_idx in s_layers}
        conv_codes = {layer_idx: _read_unit(z, f"conv_{layer_idx}_")
                      for layer_idx in conv_layers}
        m1 = _read_unit(z, "m1_") if has_m1 else None
        m2 = _read_unit(z, "m2_") if has_m2 else None
        digest = _digest_codes(s_codes, conv_codes, m1, m2)
        if digest != meta["sha256"]:
            raise ValueError(
                f"load_chunk: SHA-256 MISMATCH for chunk {meta['chunk_id']} "
                f"({p}) — the code bytes were corrupted after save (stored "
                f"{meta['sha256'][:16]}..., recomputed {digest[:16]}...); "
                f"refusing to load (spec §5 integrity)")
        return ChunkSnapshot(
            chunk_id=int(meta["chunk_id"]),
            protocol=str(meta["protocol"]),
            s_codes=s_codes,
            conv_codes=conv_codes,
            m1_codes=m1,
            m2_codes=m2,
            system_ref=meta["system_ref"],
            extra=meta["extra"],
        )


# ---------------------------------------------------------------- verify ---
def verify_chunk(path: PathLike) -> Dict[str, Any]:
    """Non-raising health check of a chunk snapshot file.

    Returns {"ok": bool, "chunk_id": ..., "protocol": ..., "n_s": int,
    "n_conv": int, "has_m1": bool, "has_m2": bool, "nbytes": int,
    "sha256_ok": bool} — on any format/corruption problem it REPORTS
    instead of raising: ok=False plus an "error" message (fields that
    could be salvaged before the failure are still filled in).
    """
    out: Dict[str, Any] = {
        "ok": False, "chunk_id": None, "protocol": None,
        "n_s": 0, "n_conv": 0, "has_m1": False, "has_m2": False,
        "nbytes": None, "sha256_ok": False,
    }
    try:
        p = os.fspath(path)
        out["nbytes"] = os.path.getsize(p)
        z = np.load(p, allow_pickle=False)
        if not hasattr(z, "files"):
            raise ValueError(
                f"{p!r} is not an npz archive (got {type(z).__name__})")
        with z:
            meta = _read_meta_json(z)
            out["chunk_id"] = meta.get("chunk_id")
            out["protocol"] = meta.get("protocol")
            s_layers = _layer_ids(z, "s")
            conv_layers = _layer_ids(z, "conv")
            out["n_s"], out["n_conv"] = len(s_layers), len(conv_layers)
            out["has_m1"] = _unit_state(z, "m1")
            out["has_m2"] = _unit_state(z, "m2")
            _check_meta(meta, p)
            _check_members(z, s_layers, conv_layers,
                           out["has_m1"], out["has_m2"])
            _check_layer_count(meta, s_layers)
            s_codes = {layer_idx: _read_unit(z, f"s_{layer_idx}_")
                       for layer_idx in s_layers}
            conv_codes = {layer_idx: _read_unit(z, f"conv_{layer_idx}_")
                          for layer_idx in conv_layers}
            m1 = _read_unit(z, "m1_") if out["has_m1"] else None
            m2 = _read_unit(z, "m2_") if out["has_m2"] else None
            digest = _digest_codes(s_codes, conv_codes, m1, m2)
            out["sha256_ok"] = bool(digest == meta.get("sha256"))
            out["ok"] = bool(out["sha256_ok"])
    except Exception as e:  # never raise — report
        out["ok"] = False
        out["sha256_ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
    return out
