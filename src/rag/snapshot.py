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

W9.2 OPTIONAL qjl MEMBERS: codes quantized with TurboQuant(qjl=True)
additionally carry {s,conv,m1,m2}_{...}_qjl_signs (d int8 ±1) +
{...}_gamma (fp32) per unit — written only when present, so pre-W9.2
files round-trip unchanged (fields load as None) and the integrity digest
of a qjl-free unit is BYTE-IDENTICAL to the v1 formula (the sketch bytes
are appended to the digest only when present).

W9.2 `use_mmap` LOAD OPTION (P7 hardening flag, DEFAULT OFF): a load-time
flag on load_chunk/verify_chunk. OFF: exactly the pre-W9.2 reader
(np.load, eager) — byte-identical loads (the parity gate). ON: members
are returned as READ-ONLY np.memmap views into the snapshot FILE — see
_MmapNpz's docstring for precisely what that does and does NOT mean (the
flag never lies: np.load(npz, mmap_mode=...) itself silently IGNORES the
flag in numpy 2.x, which is why the dedicated reader exists).

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
import struct
import warnings
import zipfile
from dataclasses import dataclass, field, fields as _dc_fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import numpy.lib.format as _npformat

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

# The TQCodes serialization surface: the REQUIRED keys (exactly the
# pre-W9.2 to_arrays() surface) + the OPTIONAL W9.2 qjl sketch keys
# (present only when the codes were quantized with qjl=True; a unit
# carrying exactly ONE of the pair is a corruption signal, refused
# loudly in _read_unit) + the OPTIONAL W15 outlier-split keys
# (group/mask/norm_hi — present IFF partition == "outlier"; the triple
# is all-or-none, refused loudly when partial). Still derived from the
# dataclass so a REQUIRED field change fails loudly here; the optional
# sets are carved out and mirrored by hand.
_OPTIONAL_CODE_FIELDS: tuple = ("qjl_signs", "gamma", "group", "mask",
                                 "norm_hi")
_SPLIT_CODE_FIELDS: tuple = ("group", "mask", "norm_hi")
_CODE_FIELDS: tuple = tuple(f.name for f in _dc_fields(TQCodes)
                            if f.name not in _OPTIONAL_CODE_FIELDS)
assert set(_OPTIONAL_CODE_FIELDS) <= {f.name for f in _dc_fields(TQCodes)}

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
    W9.2: when the unit carries the QJL sketch (BOTH qjl_signs and gamma),
    the d int8 sign bytes + the 4 fp32 gamma bytes are appended — a
    qjl-free unit digests BYTE-IDENTICALLY to the pre-W9.2 formula, so
    v1 files stay verifiable and the digest stays call-invariant.
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
    out = head + b"\x00" + norm + lo + hi
    if codes.qjl_signs is not None and codes.gamma is not None:
        out += np.ascontiguousarray(codes.qjl_signs, dtype=np.int8).tobytes()
        out += np.asarray(codes.gamma, dtype=np.float32).tobytes()
    # W15: the outlier split's channel mask + second norm + group — a
    # tampered mask must change the digest. "half" units (mask is None)
    # digest BYTE-IDENTICALLY to the pre-W15 formula.
    if codes.mask is not None:
        out += np.ascontiguousarray(codes.mask, dtype=np.uint8).tobytes()
        out += np.asarray(codes.norm_hi, dtype=np.float32).tobytes()
        out += np.asarray(int(codes.group or 1), dtype=np.int64).tobytes()
    return out


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
    (which the sha256 digest does not cover) are a tampering signal.
    W9.2: the optional qjl pair is allowed per unit (present or absent,
    never partial — _read_unit refuses the partial case)."""
    allowed = {"meta", "vector"}
    for layer_idx in s_layers:
        allowed.update(f"s_{layer_idx}_{f}" for f in _CODE_FIELDS)
        allowed.update(f"s_{layer_idx}_{f}" for f in _OPTIONAL_CODE_FIELDS)
    for layer_idx in conv_layers:
        allowed.update(f"conv_{layer_idx}_{f}" for f in _CODE_FIELDS)
        allowed.update(f"conv_{layer_idx}_{f}" for f in _OPTIONAL_CODE_FIELDS)
    if has_m1:
        allowed.update(f"m1_{f}" for f in _CODE_FIELDS)
        allowed.update(f"m1_{f}" for f in _OPTIONAL_CODE_FIELDS)
    if has_m2:
        allowed.update(f"m2_{f}" for f in _CODE_FIELDS)
        allowed.update(f"m2_{f}" for f in _OPTIONAL_CODE_FIELDS)
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
    validation BEFORE from_arrays (which would silently cast).

    W9.2: the qjl pair (qjl_signs + gamma) is optional — absent on pre-W9.2
    files (fields load as None), refused loudly when PARTIAL (exactly one
    of the two present is a corrupted sketch)."""
    label = prefix[:-1]  # e.g. "s_3", "conv_12", "m1"
    missing = [f for f in _CODE_FIELDS if f"{prefix}{f}" not in z.files]
    if missing:
        raise ValueError(
            f"load_chunk: unit '{label}' is incomplete — missing members "
            f"{missing} (corrupt npz?)")
    ql_key, gm_key = f"{prefix}qjl_signs", f"{prefix}gamma"
    has_q, has_g = ql_key in z.files, gm_key in z.files
    if has_q != has_g:
        raise ValueError(
            f"load_chunk: unit '{label}' carries a PARTIAL QJL sketch — "
            f"{ql_key if has_q else gm_key} is present but "
            f"{gm_key if has_q else ql_key} is missing (corrupt npz?)")
    # W15: the outlier-split triple (group/mask/norm_hi) — present IFF
    # partition == "outlier" (all-or-none; a partial triple or a stray
    # triple on a "half" unit is a corruption signal).
    split_present = [f for f in _SPLIT_CODE_FIELDS
                     if f"{prefix}{f}" in z.files]
    raw = {f: z[f"{prefix}{f}"] for f in _CODE_FIELDS}
    part = str(np.asarray(raw["partition"]))
    if part == "outlier":
        if len(split_present) != len(_SPLIT_CODE_FIELDS):
            raise ValueError(
                f"load_chunk: unit '{label}' is partition=outlier but the "
                f"split members {split_present} are incomplete (needs "
                f"{list(_SPLIT_CODE_FIELDS)} — corrupt npz?)")
    elif split_present:
        raise ValueError(
            f"load_chunk: unit '{label}' is partition={part!r} but carries "
            f"stray split members {split_present} (corrupt npz?)")
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
    if has_q:
        s = np.asarray(z[ql_key])
        if s.dtype != np.int8 or s.ndim != 1 or s.shape[0] != int(raw["d"]):
            raise ValueError(
                f"load_chunk: unit '{label}' qjl_signs must be a "
                f"({int(raw['d'])},) int8 ±1 sketch, got dtype={s.dtype}, "
                f"shape={s.shape}")
        g = np.asarray(z[gm_key])
        if g.dtype != np.float32 or g.size != 1:
            raise ValueError(
                f"load_chunk: unit '{label}' gamma must be a single fp32 "
                f"scalar, got dtype={g.dtype}, size={g.size}")
        raw["qjl_signs"] = s
        raw["gamma"] = g
    if part == "outlier":
        m = np.asarray(z[f"{prefix}mask"])
        if m.dtype != np.uint8 or m.ndim != 1:
            raise ValueError(
                f"load_chunk: unit '{label}' mask must be a 1-D uint8 "
                f"packed channel mask, got dtype={m.dtype}, ndim={m.ndim}")
        grp = np.asarray(z[f"{prefix}group"])
        if grp.dtype != np.int64 or grp.size != 1 or int(grp) < 1:
            raise ValueError(
                f"load_chunk: unit '{label}' group must be a positive int "
                f"scalar, got dtype={grp.dtype}, value={grp!r}")
        if int(raw["d"]) % int(grp) != 0:
            raise ValueError(
                f"load_chunk: unit '{label}' d={int(raw['d'])} is not a "
                f"multiple of group={int(grp)} (corrupt npz?)")
        nhi = np.asarray(z[f"{prefix}norm_hi"])
        if nhi.dtype != np.float32 or nhi.size != 1:
            raise ValueError(
                f"load_chunk: unit '{label}' norm_hi must be a single fp32 "
                f"scalar, got dtype={nhi.dtype}, size={nhi.size}")
        raw["group"], raw["mask"], raw["norm_hi"] = int(grp), m, nhi
    try:
        return TQCodes.from_arrays(raw)
    except Exception as e:
        raise ValueError(
            f"load_chunk: cannot rebuild TQCodes for unit '{label}': {e}") \
            from e


# ------------------------------------------------------- W9.2 mmap ---
class _MmapNpz:
    """An np.load-style reader whose members are READ-ONLY np.memmap views
    into the snapshot FILE (the `use_mmap=True` side of the W9.2 flag).

    WHAT THE FLAG ACTUALLY DOES — the flag never lies:

    * np.load(npz, mmap_mode=...) itself silently IGNORES mmap_mode on
      npz archives in numpy 2.x (members come back as eager heap arrays;
      verified on 2.1.3) — which is exactly why this dedicated reader
      exists instead of a one-word np.load change.
    * EAGER (cheap): the zip central directory (member names/order) and
      each requested member's .npy header (a 128-byte padded record).
    * LAZY: every ARRAY member (idx_lo/idx_hi bit-streams, qjl_signs,
      the debug vector) is returned as an np.memmap — pages fault in on
      FIRST TOUCH. Through TQCodes.from_arrays the fields come back as
      plain ndarrays that are ZERO-COPY VIEWS onto those memmaps (numpy's
      asarray strips the subclass; the bytes still live in the mapped
      region — inspect .base for the np.memmap root). Either way the
      loaded codes hold NO private heap copies of the streams (eager
      np.load reads each member into a fresh heap array that then stays
      alive inside the codes; here the bytes live once, in the OS page
      cache, shared by every reader of the file).
    * 0-d scalar members (norm/d/bits/n_lo/n_hi/seed — a few bytes each)
      and the 0-d unicode `meta` string are read EAGERLY (nothing to
      page-fault at 0-d; they are parsed immediately for validation
      anyway) — the mmap win is about the big streams only.
    * The sha256 integrity digest is STILL recomputed (it touches every
      code byte once, transiently) — use_mmap changes WHERE the bytes
      live, never WHETHER they are verified.
    * REQUIRES ZIP_STORED members (np.savez's uncompressed format —
      always ours, see the module docstring). A compressed member cannot
      be mmapped at all: it falls back to an eager read WITH a warning
      (never silent). np.memmap holds its own file descriptor, so closing
      this reader does NOT invalidate the views; the FILE must simply not
      be rewritten/truncated while the loaded codes are alive.
    * VALUES are bit-identical to the eager reader (the W9.2 parity gate).

    Mirrors the NpzFile surface this module uses: .files (stripped names,
    zip order), __getitem__, __contains__, close() + context manager.
    """

    def __init__(self, path: str):
        self._path = os.fspath(path)
        try:
            self._fh = open(self._path, "rb")
            self._zf = zipfile.ZipFile(self._fh)
        except (OSError, zipfile.BadZipFile) as e:
            raise ValueError(
                f"_MmapNpz: {self._path!r} is not a readable npz archive: "
                f"{e}") from e
        self._entries: Dict[str, zipfile.ZipInfo] = {}
        for zi in self._zf.infolist():
            name = zi.filename
            if name.endswith(".npy"):
                name = name[:-4]
            self._entries[name] = zi
        self.files = list(self._entries)  # NpzFile semantics: zip order

    # ------------------------------------------------------------ reader --
    def _member(self, name: str):
        zi = self._entries[name]
        if zi.compress_type != zipfile.ZIP_STORED:
            warnings.warn(
                f"use_mmap: npz member {name!r} is compressed "
                f"(compress_type={zi.compress_type}) — cannot be mmapped; "
                f"falling back to an eager in-memory read (this module's "
                f"writer never compresses)", RuntimeWarning, stacklevel=3)
            with self._zf.open(zi) as fp:
                return _npformat.read_array(fp, allow_pickle=False)
        # local file header: 30 fixed bytes + name + extra (the extra-field
        # length in the LOCAL header can differ from the central one —
        # parse it from the file, never trust the ZipInfo copy)
        self._fh.seek(zi.header_offset)
        fixed = self._fh.read(30)
        if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
            raise ValueError(
                f"use_mmap: bad local zip header for member {name!r} "
                f"at offset {zi.header_offset} (corrupt npz?)")
        name_len, extra_len = struct.unpack("<HH", fixed[26:30])
        data_start = zi.header_offset + 30 + name_len + extra_len
        with self._zf.open(zi) as fp:
            version = _npformat.read_magic(fp)
            shape, fortran, dtype = self._read_npy_header(fp, version, name)
            consumed = fp.tell()  # bytes of the .npy header record
        if len(shape) == 0 or dtype.kind in ("U", "S", "O"):
            # 0-d scalars + string dtypes: eager (nothing to lazily page;
            # the meta JSON must be parsed immediately anyway)
            with self._zf.open(zi) as fp:
                return _npformat.read_array(fp, allow_pickle=False)
        return np.memmap(self._path, mode="r", dtype=dtype, shape=shape,
                         order="F" if fortran else "C",
                         offset=data_start + consumed)

    @staticmethod
    def _read_npy_header(fp, version, name: str):
        major, minor = version
        try:
            if (major, minor) == (1, 0):
                return _npformat.read_array_header_1_0(fp)
            if (major, minor) in ((2, 0), (3, 0)):
                # 3.0 is 2.0's record with an explicit utf-8 header
                # encoding — same 4-byte length, same reader.
                return _npformat.read_array_header_2_0(fp)
            return _npformat._read_array_header(fp, version)
        except Exception as e:
            raise ValueError(
                f"use_mmap: cannot parse the .npy header of member {name!r} "
                f"(version {version}): {e}") from e

    # ------------------------------------------------------ dict surface --
    def __getitem__(self, name: str):
        if name not in self._entries:
            raise KeyError(
                f"{name!r} is not a member of the archive "
                f"({os.path.basename(self._path)})")
        return self._member(name)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __enter__(self) -> "_MmapNpz":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._zf.close()
        finally:
            self._fh.close()


def _open_npz(path: str, use_mmap: bool, ctx: str):
    """The chunk-npz reader behind load_chunk/verify_chunk (W9.2 flag).

    use_mmap=False (DEFAULT — the parity path): exactly the pre-W9.2
    reader, np.load(path, allow_pickle=False) — eager, byte-identical.
    use_mmap=True: the _MmapNpz reader (read-only per-member memmaps;
    see its docstring for the honest behavior contract)."""
    if not use_mmap:
        z = np.load(path, allow_pickle=False)
        if not hasattr(z, "files"):
            raise ValueError(
                f"{ctx}: {path!r} is not an npz archive (got "
                f"{type(z).__name__}) — chunk snapshots are "
                f"snapshots/chunk_XXXXX.npz (spec §11)")
        return z
    return _MmapNpz(path)


def load_chunk(path: PathLike, use_mmap: bool = False) -> ChunkSnapshot:
    """Full reconstruction of a chunk snapshot (spec §5): meta validation
    (version, protocol, keys, file-name consistency), per-unit TQCodes
    rebuild, and the sha256 integrity comparison — raises LOUDLY on any
    mismatch/corruption.

    use_mmap : bool, default False — the W9.2 P7 flag. False: the eager
    pre-W9.2 reader (bit-identical loads — the parity gate). True: array
    members come back as read-only np.memmap views into the snapshot file
    (values identical; see _MmapNpz for exactly what that does and does
    not mean — the flag never lies).

    The debug `vector` member (if one was saved) is intentionally NOT
    reconstructed: the production retrieval vector is recomputed from the
    codes (D5) — access it directly via np.load(path)["vector"] if needed.
    """
    p = os.fspath(path)
    z = _open_npz(p, use_mmap, "load_chunk")
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
def verify_chunk(path: PathLike, use_mmap: bool = False) -> Dict[str, Any]:
    """Non-raising health check of a chunk snapshot file.

    Returns {"ok": bool, "chunk_id": ..., "protocol": ..., "n_s": int,
    "n_conv": int, "has_m1": bool, "has_m2": bool, "nbytes": int,
    "sha256_ok": bool} — on any format/corruption problem it REPORTS
    instead of raising: ok=False plus an "error" message (fields that
    could be salvaged before the failure are still filled in).

    use_mmap : bool, default False — the W9.2 flag (same reader choice as
    load_chunk; the verification result is identical either way).
    """
    out: Dict[str, Any] = {
        "ok": False, "chunk_id": None, "protocol": None,
        "n_s": 0, "n_conv": 0, "has_m1": False, "has_m2": False,
        "nbytes": None, "sha256_ok": False,
    }
    try:
        p = os.fspath(path)
        out["nbytes"] = os.path.getsize(p)
        z = _open_npz(p, use_mmap, "verify_chunk")
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
