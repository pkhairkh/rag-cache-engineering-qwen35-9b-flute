"""index.py — the IVFADC retrieval index: build + preselect + rerank
(SPECIFICATION.md §8, §6 steps 3–4, §10; PROPOSAL.md D5).

The spec §8 stack with its verbatim constants (IndexConfig defaults):

    quantizer = faiss.IndexFlatIP(13_631_488)
    index     = faiss.IndexIVFPQ(quantizer, 13_631_488, nlist=224, m=64, nbits=8)
    index.train(cache_vectors_fp32); index.add(cache_vectors_fp32)   # nprobe=8

Query flow (§6): step 3 = `preselect` (IVFADC over the cache vectors →
top-100 candidates); step 4 = `rerank` (exact cos-sim on the full
dequantized vectors → top-3 chunk indices). §10: the index lives in CPU
RAM (faiss.write_index → a single mmap-able file, <path>.meta.json beside
it carries the provenance).

PROPOSAL D5 — the 13.6M-dim vectors are NEVER materialized en masse:
  * TRAIN materializes ONLY the first `train_sample` vectors (default
    4096) as one (n, d) fp32 block — faiss.train wants a dense matrix;
    that block is the only dense corpus materialization this module ever
    makes (4096 x 13.6M x 4 B ~ 222 GiB at production dims — WARNED; the
    GPU box must shard or shrink `train_sample`).
  * ADD streams the rest in batches of 1024 — IVFPQ stores m x nbits/8 =
    64-byte compressed entries per vector, never fp32 rows.
  * RERANK dequantizes the exact vectors ONE CANDIDATE AT A TIME from the
    TurboQuant codes on disk (ChunkVectorLoader; ~54.5 MB fp32 transient
    per candidate at production dims, freed before the next — D5's
    "exact vectors exist only as TQ codes on disk").

THE NORMALIZATION DECISION (documented): spec §4 measures cosine (direction
overlap — "cos-sim measures content overlap"), so vectors are
L2-NORMALIZED before train/add and the metric is INNER PRODUCT: IP over
unit vectors IS cos-sim in both the coarse (IndexFlatIP) and the PQ/ADC
stages. The pre-normalization norms are NOT stored — the normalized frame
is the index's own frame; `rerank` recomputes the exact cosine from the
RAW (dequantized) vectors, so the final ranking never depends on the
compression frame.

THE DOUBLE-ROUND RECONSTRUCTION (ChunkVectorLoader): the on-disk chunks
are D4 DELTA codes (W5), so the ABSOLUTE retrieval vector of chunk i is
dequant(S_sys) + dequant(delta_i) per S layer, then M1, M2 (§4 order; conv
codes are ABSOLUTE per §6 and NOT part of the §4 vector). The ingest-time
record.cache_vector was dequantized from the ABSOLUTE codes (single quant
round from the true state); this disk reconstruction differs by ONE EXTRA
quant round on the delta (quant(dequant(abs) - dequant(sys)) vs abs) —
the documented budget is rel-MSE < 0.10 (W5.3 measured ~0.011 per S
segment; the W6.3 gate measures ~0.013 end-to-end at the stub scale).

Provenance (turboquant.py: "seeds persist in the index side-metadata"):
build_index(path=...) writes <path>.meta.json with the config fields, the
four D3 rotation SEEDS, sha256 digests of the production-frame codebook
arrays (codebooks.get_codebook(3, 524_288) / get_codebook(4, 524_288) —
the 3.5-bit recipe's realized widths at the S/M1/M2 unit dim, D2), the
vector counts, faiss version, created_utc and the caller's
extra_metadata. The hashes pin the QUANTIZER RELEASE FRAME (the committed
codebook artifacts) — corpus-specific frame data (unit dims at test
scale) rides in extra_metadata. load_index restores nprobe from the
metadata (faiss already serializes it — belt and braces).

Guards (loud): config/stream dim mismatch; m not dividing d; zero-norm
corpus vectors (unnormalizable); empty corpus; fewer train vectors than
faiss's 39*nlist rule of thumb (WARNING only — kwarg `min_train_warn`);
a train sample whose dense block would exceed 8 GiB (WARNING only);
ChunkVectorLoader refuses a missing system_state.npz / manifest, protocol
or frame (system_ref) drift, per-unit dim drift, and vector-length drift
vs the manifest.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import faiss
import numpy as np
import torch

import _paths  # noqa: F401  (house rule: anchor BEFORE the sibling imports)
from codebooks import get_codebook
from ingest import MANIFEST_NAME, load_system_state
from snapshot import load_chunk
from tq_cache import resolve_quantizer
from turboquant import SEEDS

__all__ = [
    "IndexConfig",
    "ChunkVectorLoader",
    "build_index",
    "load_index",
    "preselect",
    "rerank",
    "codebook_sha256",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: add-time streaming batch (D5: never a dense (n, d) corpus block)
ADD_BATCH = 1024
#: warn (never fail) when the materialized train block would exceed this
TRAIN_RAM_WARN_BYTES = 8 * 1024 ** 3
#: side-metadata file: <index path> + META_SUFFIX
META_SUFFIX = ".meta.json"
#: the pinned production retrieval frame (D2/D3): the 3.5-bit recipe's
#: realized (3, 4) codebook widths at the S/M1/M2 unit dim 2^19
_FRAME_BITS = (3, 4)
_FRAME_UNIT_D = 524_288

_CODEBOOK_SHA_CACHE: Dict[Tuple[int, int], str] = {}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_int(value, name: str, minimum: int = 1, ctx: str = "") -> int:
    """Loud int validation (bools are ints — refuse them explicitly)."""
    prefix = f"{ctx}: " if ctx else ""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{prefix}{name} must be an int, got {type(value).__name__}: "
            f"{value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{prefix}{name} must be >= {minimum}, got {v}")
    return v


def _validate_config(config, ctx: str = "") -> None:
    """Validate an IndexConfig-like object (dataclasses are mutable —
    build_index re-validates what __post_init__ already checked)."""
    prefix = f"{ctx}" if ctx else "IndexConfig"
    for name in ("d", "nlist", "m", "nbits", "nprobe", "preselect_k",
                 "rerank_k"):
        _check_int(getattr(config, name, None), name, minimum=1,
                   ctx=prefix)
    d, m = int(config.d), int(config.m)
    if d % m != 0:
        raise ValueError(
            f"{prefix}: m={m} must divide d={d} — faiss.IndexIVFPQ splits "
            f"every vector into m sub-quantizers of d/m dims (spec §8: m=64 "
            f"divides d=13,631,488 = 64 x 212,992)")
    if int(config.nbits) > 16:
        raise ValueError(
            f"{prefix}: nbits={config.nbits} exceeds faiss PQ's 16-bit code "
            f"width (spec §8 pins nbits=8)")


# ---------------------------------------------------------------------------
# IndexConfig (spec §8 + §6 constants)
# ---------------------------------------------------------------------------
@dataclass
class IndexConfig:
    """The §8/§6 index constants — the defaults ARE the spec numbers.

    d:           13,631,488 = 24 x 524,288 (S) + 2 x 524,288 (M1/M2), §4/§8
    nlist:       224 IVF lists, §8
    m:           64 PQ sub-quantizers (must divide d), §8
    nbits:       8 bits per PQ code, §8
    nprobe:      8 lists probed per query, §8
    preselect_k: 100 (spec §6 step 3 — the IVFADC candidate cut)
    rerank_k:    3 (spec §6 step 4 — the final top-3)
    """

    d: int = 13_631_488          # spec §4/§8 (24*524288 + 2*524288)
    nlist: int = 224             # spec §8
    m: int = 64                  # spec §8 (must divide d)
    nbits: int = 8               # spec §8
    nprobe: int = 8              # spec §8
    preselect_k: int = 100       # spec §6 step 3
    rerank_k: int = 3            # spec §6 step 4

    def __post_init__(self) -> None:
        _validate_config(self, ctx="IndexConfig()")


# ---------------------------------------------------------------------------
# Codebook provenance
# ---------------------------------------------------------------------------
def codebook_sha256(bits: int, d: int) -> str:
    """sha256 of a Lloyd-Max codebook's arrays (codebooks.get_codebook).

    Digests the identity (bits, d, level count), the centroid and boundary
    arrays (fp32, little-endian) and the mse constant — everything a
    dequantizer needs — so two codebooks hash equal iff they decode equal.
    Memoized: the committed artifacts are immutable, the hash is stable.
    NOTE: an uncached (bits, d) pair makes get_codebook SOLVE and cache a
    new codebook — pass dims you mean.
    """
    key = (_check_int(bits, "bits", minimum=1, ctx="codebook_sha256"),
           _check_int(d, "d", minimum=3, ctx="codebook_sha256"))
    if key in _CODEBOOK_SHA_CACHE:
        return _CODEBOOK_SHA_CACHE[key]
    cb = get_codebook(key[0], key[1])
    h = sha256()
    h.update(f"codebook|bits={cb.bits}|d={cb.d}|levels={cb.centroids.shape[0]}"
             .encode("utf-8"))
    h.update(np.ascontiguousarray(cb.centroids, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(cb.boundaries, dtype=np.float32).tobytes())
    h.update(f"|mse_per_variance={float(cb.mse_per_variance)!r}".encode("utf-8"))
    digest = h.hexdigest()
    _CODEBOOK_SHA_CACHE[key] = digest
    return digest


def _frame_codebook_sha256() -> Dict[str, str]:
    """The production retrieval frame's codebook digests (side metadata)."""
    return {f"b{b}_d{_FRAME_UNIT_D}": codebook_sha256(b, _FRAME_UNIT_D)
            for b in _FRAME_BITS}


# ---------------------------------------------------------------------------
# ChunkVectorLoader (D5: vectors on demand, from the W5 delta codes)
# ---------------------------------------------------------------------------
class ChunkVectorLoader:
    """Streams the §4 retrieval vectors from disk (PROPOSAL D5: never
    materialized en masse).

    The absolute retrieval vector of chunk i (DOUBLE-ROUND reconstruction,
    see the module docstring):

        vector(i) = concat_L( dequant(S_sys[L]) + dequant(delta_i[L]) ,
                              dequant(M1_sys) + dequant(delta_m1_i),
                              dequant(M2_sys) + dequant(delta_m2_i) )

    in the §4 order (S layers ascending, then M1, then M2; conv codes are
    excluded — §4). Every unit is dequantized through
    tq_cache.resolve_quantizer with the D3 kind seeds, so the frame matches
    ingest. Nothing is cached in memory: every .vector() call re-reads the
    chunk npz (sha256-verified by snapshot.load_chunk) and re-dequantizes —
    the rerank's one-candidate-at-a-time contract.
    """

    def __init__(self, disk_dir: str, bits: float = 3.5):
        self.disk_dir = os.path.abspath(os.fspath(disk_dir))
        if not os.path.isdir(self.disk_dir):
            raise ValueError(
                f"ChunkVectorLoader: {self.disk_dir} is not a directory — "
                f"pass the IngestDriver's out_dir (the one holding "
                f"system_state.npz + ingest_manifest.json)")
        # the persisted reset point (ingest.load_system_state is loud when
        # system_state.npz is missing)
        self.system = load_system_state(self.disk_dir)
        self.bits = float(bits)
        if abs(self.bits - float(self.system.bits)) > 1e-9:
            raise ValueError(
                f"ChunkVectorLoader: bits mismatch — loader was given "
                f"{self.bits} but the reset point at {self.disk_dir} was "
                f"ingested at {self.system.bits}; refusing to dequantize "
                f"through the wrong quantizer frame (PROPOSAL D3)")
        man_path = os.path.join(self.disk_dir, MANIFEST_NAME)
        if not os.path.isfile(man_path):
            raise ValueError(
                f"ChunkVectorLoader: {man_path} not found — the ingest "
                f"manifest maps chunk ids to snapshot files; run the "
                f"IngestDriver first")
        with open(man_path, "r") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict):
            raise ValueError(
                f"ChunkVectorLoader: {man_path} is not a JSON object")
        self.manifest = manifest
        if manifest.get("protocol") != "delta-v1":
            raise ValueError(
                f"ChunkVectorLoader: manifest protocol "
                f"{manifest.get('protocol')!r} != 'delta-v1' — this loader "
                f"reconstructs the D4 delta protocol only")
        man_bits = manifest.get("bits")
        if man_bits is not None and abs(float(man_bits) - self.bits) > 1e-9:
            raise ValueError(
                f"ChunkVectorLoader: manifest bits {man_bits} != loader "
                f"bits {self.bits} — frame drift, refusing")
        done = manifest.get("done")
        if done is None or not isinstance(done, dict):
            raise ValueError(
                f"ChunkVectorLoader: {man_path} has no 'done' chunk map")
        self._chunks: Dict[int, str] = {}
        for k, rel in done.items():
            try:
                cid = int(k)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"ChunkVectorLoader: manifest 'done' key {k!r} is not "
                    f"an int chunk id") from e
            if not isinstance(rel, str) or not rel:
                raise ValueError(
                    f"ChunkVectorLoader: manifest 'done'[{k!r}] is not a "
                    f"relative snapshot path: {rel!r}")
            self._chunks[cid] = rel
        # the expected §4 vector length: sum of S dims + M1 + M2
        self._dims = sum(int(c.d) for c in self.system.s_codes.values())
        for c in (self.system.m1_codes, self.system.m2_codes):
            if c is not None:
                self._dims += int(c.d)
        man_dims = manifest.get("vector_dims")
        if man_dims is not None and int(man_dims) != self._dims:
            raise ValueError(
                f"ChunkVectorLoader: manifest vector_dims {man_dims} != the "
                f"reset point's §4 length {self._dims} (sum of S + M1 + M2 "
                f"unit dims) — the corpus and the reset point disagree")

    # ------------------------------------------------------------ surface --
    @property
    def dims(self) -> int:
        """The §4 retrieval-vector length (sum of S + M1 + M2 unit dims)."""
        return self._dims

    def chunk_ids(self) -> List[int]:
        """All ingested chunk ids, ascending (the iter_vectors order)."""
        return sorted(self._chunks)

    def vector(self, chunk_id: int) -> np.ndarray:
        """The chunk's absolute retrieval vector: fp32, shape (dims,).

        DOUBLE-ROUND reconstruction (module docstring): one extra quant
        round on the delta vs the ingest-time record.cache_vector.
        """
        if isinstance(chunk_id, bool) or not isinstance(
                chunk_id, (int, np.integer)):
            raise TypeError(
                f"ChunkVectorLoader.vector: chunk_id must be an int, got "
                f"{type(chunk_id).__name__}: {chunk_id!r}")
        cid = int(chunk_id)
        if cid not in self._chunks:
            known = self.chunk_ids()
            raise ValueError(
                f"ChunkVectorLoader.vector: chunk {cid} is not in the "
                f"manifest — known ids: {known[:8]}"
                f"{'...' if len(known) > 8 else ''} ({len(known)} total)")
        path = os.path.join(self.disk_dir, self._chunks[cid])
        if not os.path.isfile(path):
            raise ValueError(
                f"ChunkVectorLoader.vector: {path} (manifest entry for "
                f"chunk {cid}) does not exist — the snapshot tree is "
                f"incomplete; re-run the IngestDriver")
        snap = load_chunk(path)  # sha256-verified (snapshot.py, W5.1)
        if snap.protocol != "delta-v1":
            raise ValueError(
                f"ChunkVectorLoader.vector: chunk {cid} protocol "
                f"{snap.protocol!r} != 'delta-v1' — this loader reads the "
                f"D4 delta protocol only (the ABSOLUTE protocol is the "
                f"system reset point, not a chunk)")
        if snap.system_ref != self.system.reference():
            raise ValueError(
                f"ChunkVectorLoader.vector: chunk {cid} system_ref "
                f"{snap.system_ref!r} != the reset point's "
                f"{self.system.reference()!r} — frame drift (the chunk was "
                f"ingested against a different system prompt/model), "
                f"refusing (PROPOSAL D3)")
        chunk_layers = set(snap.s_codes)
        sys_layers = set(self.system.s_codes)
        if chunk_layers != sys_layers:
            raise ValueError(
                f"ChunkVectorLoader.vector: chunk {cid} S layers "
                f"{sorted(chunk_layers)} != the reset point's "
                f"{sorted(sys_layers)} — the delta protocol requires every "
                f"S unit to be present")
        pieces = []
        for L in sorted(self.system.s_codes):
            sysc = self.system.s_codes[L]
            delta = snap.s_codes[L]
            if int(delta.d) != int(sysc.d):
                raise ValueError(
                    f"ChunkVectorLoader.vector: chunk {cid} layer {L} delta "
                    f"d={delta.d} != system d={sysc.d} — unit-dim drift")
            q = resolve_quantizer("S", int(sysc.d), self.bits)
            pieces.append(q.dequant(sysc) + q.dequant(delta))
        for name, sysc, delta in (
                ("M1", self.system.m1_codes, snap.m1_codes),
                ("M2", self.system.m2_codes, snap.m2_codes)):
            if (sysc is None) != (delta is None):
                raise ValueError(
                    f"ChunkVectorLoader.vector: chunk {cid} {name} codes "
                    f"{'missing' if delta is None else 'present'} while the "
                    f"reset point's are "
                    f"{'present' if sysc is not None else 'missing'} — the "
                    f"delta protocol requires both sides of every unit")
            if sysc is None:
                continue
            if int(delta.d) != int(sysc.d):
                raise ValueError(
                    f"ChunkVectorLoader.vector: chunk {cid} {name} delta "
                    f"d={delta.d} != system d={sysc.d} — unit-dim drift")
            q = resolve_quantizer(name, int(sysc.d), self.bits)
            pieces.append(q.dequant(sysc) + q.dequant(delta))
        vec = torch.cat(pieces).to(torch.float32).numpy()
        if vec.shape != (self._dims,):
            raise ValueError(
                f"ChunkVectorLoader.vector: chunk {cid} reconstructed "
                f"{vec.shape} vectors, expected ({self._dims},) — §4 order "
                f"length drift (manifest {self.manifest.get('vector_dims')})")
        return vec

    def iter_vectors(self, chunk_ids: Optional[Iterable[int]] = None
                     ) -> Iterator[np.ndarray]:
        """Yield retrieval vectors in chunk-id order (D5: one at a time)."""
        if chunk_ids is None:
            ids = self.chunk_ids()
        else:
            ids = [int(c) for c in chunk_ids]
            unknown = [c for c in ids if c not in self._chunks]
            if unknown:
                raise ValueError(
                    f"ChunkVectorLoader.iter_vectors: unknown chunk ids "
                    f"{unknown[:8]} — known: {self.chunk_ids()[:8]}"
                    f"{'...' if len(self._chunks) > 8 else ''}")
        for cid in ids:
            yield self.vector(cid)


# ---------------------------------------------------------------------------
# build / load
# ---------------------------------------------------------------------------
def _normalize_streamed(v, config: IndexConfig, pos: int) -> np.ndarray:
    """Validate + L2-normalize one streamed corpus vector (fp32, (d,))."""
    row = np.asarray(v)
    if row.ndim != 1:
        raise ValueError(
            f"build_index: stream position {pos}: expected a 1-D (d,) "
            f"vector, got shape {row.shape}")
    if row.shape[0] != int(config.d):
        raise ValueError(
            f"build_index: stream position {pos}: vector dim "
            f"{row.shape[0]} != config.d={int(config.d)} — refusing to "
            f"build an index over a mismatched frame (SPECIFICATION §8 "
            f"fixes d=13,631,488; the loader/stream and IndexConfig must "
            f"agree)")
    row = np.ascontiguousarray(row, dtype=np.float32)
    norm = float(np.linalg.norm(row))
    if not np.isfinite(norm):
        raise ValueError(
            f"build_index: stream position {pos}: non-finite vector "
            f"(norm {norm!r})")
    if norm == 0.0:
        raise ValueError(
            f"build_index: stream position {pos}: zero-norm vector — the "
            f"cosine frame cannot normalize it (a zero §4 vector means "
            f"every unit collapsed; investigate the corpus, don't index "
            f"it)")
    return row / norm


def _index_metadata(config: IndexConfig, n_train: int, ntotal: int,
                    extra_metadata: Optional[dict]) -> dict:
    if extra_metadata is not None and not isinstance(extra_metadata, dict):
        raise TypeError(
            f"build_index: extra_metadata must be a dict (JSON object), "
            f"got {type(extra_metadata).__name__}")
    meta = {
        "module": "src/rag/index.py",
        "spec": "SPECIFICATION §8 (IVFADC constants), §6 steps 3-4 "
                "(preselect/rerank), §10 (CPU-RAM index); PROPOSAL D5 "
                "(streamed build, on-demand rerank)",
        "d": int(config.d),
        "nlist": int(config.nlist),
        "m": int(config.m),
        "nbits": int(config.nbits),
        "nprobe": int(config.nprobe),
        "preselect_k": int(config.preselect_k),
        "rerank_k": int(config.rerank_k),
        "metric": "inner_product",
        "vector_frame": "l2-normalized (cosine; norms not stored — the "
                        "rerank recomputes exact cos from raw vectors)",
        "seeds": {str(k): int(s) for k, s in SEEDS.items()},
        "codebook_sha256": _frame_codebook_sha256(),
        "counts": {"train_vectors": int(n_train),
                   "indexed_vectors": int(ntotal)},
        "faiss_version": str(getattr(faiss, "__version__", "unknown")),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extra_metadata": dict(extra_metadata) if extra_metadata else {},
    }
    try:
        json.dumps(meta)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"build_index: the side metadata is not JSON-serializable "
            f"(extra_metadata must hold JSON-safe values only)") from e
    return meta


def _write_index_and_meta(index: faiss.IndexIVFPQ, path: str,
                          meta: dict) -> None:
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    # atomic writes (house pattern): tmp + os.replace for both files
    tmp_faiss = path + ".tmp"
    faiss.write_index(index, tmp_faiss)
    os.replace(tmp_faiss, path)
    meta_path = path + META_SUFFIX
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    os.replace(tmp_meta, meta_path)


def build_index(vectors_iter, config: IndexConfig = IndexConfig(),
                train_sample: int = 4096, path: str | None = None,
                extra_metadata: dict | None = None,
                min_train_warn: int | None = None
                ) -> faiss.IndexIVFPQ:
    """Build the §8 IVFADC index over a stream of raw §4 vectors.

    vectors_iter: iterable of raw (un-normalized) fp vectors, (d,) each —
    e.g. ChunkVectorLoader(disk_dir).iter_vectors(). Position i in the
    stream becomes faiss id i (the caller owns the id↔chunk mapping; the
    loader yields chunk-id order).

    D5 streaming: the FIRST `train_sample` vectors are materialized as one
    (n, d) fp32 block for index.train (the only dense block); they are
    then re-added (batched) and the REST of the stream is added in batches
    of ADD_BATCH — vectors are never materialized en masse.

    Normalization: every vector is L2-normalized before train/add (the
    index's cosine frame — see the module docstring); norms are not kept.

    path: when given, faiss.write_index to `path` (atomic) + the
    provenance side-metadata JSON at `path` + '.meta.json'.

    min_train_warn: the WARNING (not failure) threshold for faiss's
    39*nlist rule of thumb; None → 39*nlist. Small test corpora warn —
    that is expected and harmless.
    """
    _validate_config(config, ctx="build_index")
    train_sample = _check_int(train_sample, "train_sample", minimum=1,
                              ctx="build_index")
    stream = iter(vectors_iter)

    # a-priori RAM guard: the train block is the one dense materialization
    projected = train_sample * int(config.d) * 4
    if projected > TRAIN_RAM_WARN_BYTES:
        warnings.warn(
            f"build_index: the train sample would materialize "
            f"{projected / 1024 ** 3:.1f} GiB ({train_sample} x "
            f"{int(config.d)} x 4 B) — above this module's "
            f"{TRAIN_RAM_WARN_BYTES / 1024 ** 3:.0f} GiB guard. At "
            f"production dims (§8: d=13,631,488) the GPU box must shard "
            f"the corpus and/or shrink train_sample, or stream the train "
            f"set from disk in shards; PROPOSAL D5 is otherwise violated.",
            RuntimeWarning, stacklevel=2)

    # ---- train: materialize ONLY the first train_sample vectors --------
    train_rows: List[np.ndarray] = []
    for v in stream:
        train_rows.append(_normalize_streamed(v, config, len(train_rows) + 1))
        if len(train_rows) >= train_sample:
            break
    n_train = len(train_rows)
    if n_train == 0:
        raise ValueError(
            "build_index: the vector stream is empty — nothing to index "
            "(run the IngestDriver / pass a non-empty loader)")
    threshold = (39 * int(config.nlist) if min_train_warn is None
                 else int(min_train_warn))
    if n_train < threshold:
        warnings.warn(
            f"build_index: only {n_train} train vectors < {threshold} "
            f"(faiss's 39*nlist rule of thumb at nlist={int(config.nlist)}"
            f"{'; min_train_warn=' + str(min_train_warn) if min_train_warn is not None else ''})"
            f" — the coarse/PQ stages will be under-trained. Expected at "
            f"small test scale; NOT a failure (the gate kwargs exist for "
            f"this).", RuntimeWarning, stacklevel=2)
    train_mat = np.ascontiguousarray(np.stack(train_rows), dtype=np.float32)

    quantizer = faiss.IndexFlatIP(int(config.d))
    index = faiss.IndexIVFPQ(quantizer, int(config.d), int(config.nlist),
                             int(config.m), int(config.nbits),
                             faiss.METRIC_INNER_PRODUCT)
    if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
        raise RuntimeError(
            f"build_index: faiss built metric_type={index.metric_type}, "
            f"expected INNER_PRODUCT — the cosine frame (module docstring) "
            f"is broken; refusing to continue")
    index.train(train_mat)

    # ---- add: the train rows (batched), then the rest of the stream -----
    for i in range(0, n_train, ADD_BATCH):
        index.add(train_mat[i:i + ADD_BATCH])
    batch: List[np.ndarray] = []
    pos = n_train
    for v in stream:
        pos += 1
        batch.append(_normalize_streamed(v, config, pos))
        if len(batch) >= ADD_BATCH:
            index.add(np.ascontiguousarray(np.stack(batch), dtype=np.float32))
            batch.clear()
    if batch:
        index.add(np.ascontiguousarray(np.stack(batch), dtype=np.float32))

    index.nprobe = int(config.nprobe)

    if path is not None:
        meta = _index_metadata(config, n_train, index.ntotal, extra_metadata)
        _write_index_and_meta(index, path, meta)
    return index


def load_index(path: str) -> Tuple[faiss.IndexIVFPQ, dict]:
    """Load a build_index artifact: (index, side metadata).

    Restores nprobe from the metadata (faiss serializes it too — belt and
    braces). Loud on: missing index file, non-IVFPQ file, missing/invalid
    metadata, dim drift between the two.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise ValueError(
            f"load_index: {path} not found — build_index(path=...) writes "
            f"the faiss index there")
    index = faiss.read_index(path)
    if not isinstance(index, faiss.IndexIVFPQ):
        raise ValueError(
            f"load_index: {path} holds a {type(index).__name__}, not an "
            f"IndexIVFPQ — this module reads §8 IVFADC artifacts only")
    meta_path = path + META_SUFFIX
    if not os.path.isfile(meta_path):
        raise ValueError(
            f"load_index: {meta_path} not found — the side metadata is "
            f"written by build_index(path=...) and carries the frame "
            f"provenance (seeds, codebook hashes, config); refusing to "
            f"serve an index of unknown frame")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if not isinstance(meta, dict) or "d" not in meta:
        raise ValueError(
            f"load_index: {meta_path} is not a metadata object (missing "
            f"'d')")
    if int(meta["d"]) != int(index.d):
        raise ValueError(
            f"load_index: metadata d={meta['d']} != index d={index.d} — "
            f"the .index and .meta.json files disagree (mixed artifacts?)")
    if "nprobe" in meta:
        index.nprobe = _check_int(meta["nprobe"], "metadata nprobe",
                                   ctx="load_index")
    return index, meta


# ---------------------------------------------------------------------------
# §6 steps 3–4
# ---------------------------------------------------------------------------
def _query_vector(query_vector, d: Optional[int], ctx: str) -> np.ndarray:
    q = np.asarray(query_vector)
    if q.ndim != 1:
        raise ValueError(
            f"{ctx}: the query must be a 1-D (d,) vector, got shape "
            f"{q.shape}")
    if d is not None and q.shape[0] != int(d):
        raise ValueError(
            f"{ctx}: query dim {q.shape[0]} != {int(d)} — the query's §4 "
            f"vector and the index must share the frame (SPECIFICATION §4: "
            f"'the query's cache vector and the chunk's cache vector are in "
            f"the SAME space')")
    q = np.ascontiguousarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm):
        raise ValueError(f"{ctx}: non-finite query vector (norm {norm!r})")
    if norm == 0.0:
        raise ValueError(
            f"{ctx}: zero-norm query — cosine is undefined; a §4 query "
            f"vector is a live cache state, never all-zero")
    return q


def preselect(index, query_vector: np.ndarray, k: int = 100) -> np.ndarray:
    """§6 step 3: IVFADC preselect → candidate chunk ids.

    The query is L2-normalized (the index's cosine frame) and searched
    with index.search; returns an int64 id array with faiss's -1 padding
    filtered. k defaults to spec §6's 100 (IndexConfig.preselect_k).
    """
    k = _check_int(k, "k", ctx="preselect")
    d = getattr(index, "d", None)
    if d is None:
        raise TypeError(
            "preselect: `index` must be a faiss index (has .d/.search), "
            f"got {type(index).__name__}")
    q = _query_vector(query_vector, d, ctx="preselect")
    q_hat = (q / float(np.linalg.norm(q))).reshape(1, -1)
    _, labels = index.search(q_hat, k)
    ids = np.asarray(labels).reshape(-1)
    ids = ids[ids >= 0]
    return ids.astype(np.int64)


def rerank(loader, query_vector: np.ndarray, candidate_ids,
           k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """§6 step 4: exact cos-sim rerank → (top ids, scores desc).

    D5 LAZY: dequantizes the candidates' full vectors ONE AT A TIME via
    loader.vector(cid) — exactly the ids passed (each exactly once,
    duplicates coalesced), nothing else. The scores are exact cosines of
    the RAW vectors (query and candidates), independent of the index's
    normalized frame. Ties keep candidate order (deterministic). A
    zero-norm candidate scores 0.0 (a legitimate stored zero state —
    cosine against it is 0 by convention, documented). Returns
    min(k, #candidates) results.
    """
    k = _check_int(k, "k", ctx="rerank")
    if not hasattr(loader, "vector"):
        raise TypeError(
            f"rerank: loader must expose .vector(chunk_id) (a "
            f"ChunkVectorLoader or a mock), got {type(loader).__name__}")
    if candidate_ids is None:
        raise TypeError("rerank: candidate_ids must be a sequence of ints")
    q = _query_vector(query_vector, None, ctx="rerank")
    qn = float(np.linalg.norm(q))
    uniq: List[int] = []
    seen = set()
    for c in candidate_ids:
        if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
            raise TypeError(
                f"rerank: candidate ids must be ints, got "
                f"{type(c).__name__}: {c!r}")
        c = int(c)
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        raise ValueError(
            "rerank: no candidates — pass preselect(...)'s ids (spec §6 "
            "step 3 feeds step 4)")
    scores = np.empty(len(uniq), dtype=np.float64)
    for i, cid in enumerate(uniq):
        v = np.asarray(loader.vector(cid))
        if v.ndim != 1 or v.shape[0] != q.shape[0]:
            raise ValueError(
                f"rerank: candidate {cid} vector shape {v.shape} != the "
                f"query's ({q.shape[0]},) — frame drift between the loader "
                f"and the query")
        v = np.ascontiguousarray(v, dtype=np.float32)
        vn = float(np.linalg.norm(v))
        if not np.isfinite(vn):
            raise ValueError(
                f"rerank: candidate {cid} has non-finite norm {vn!r}")
        scores[i] = 0.0 if vn == 0.0 else float(np.dot(q, v) / (qn * vn))
    # stable descending order: score desc, then candidate arrival order
    order = sorted(range(len(uniq)), key=lambda i: (-scores[i], i))
    top = order[: min(k, len(order))]
    ids = np.asarray([uniq[i] for i in top], dtype=np.int64)
    return ids, scores[top].astype(np.float32)
