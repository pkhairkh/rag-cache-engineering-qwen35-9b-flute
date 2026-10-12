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
W16: protocol="absolute" chunks (the new default) carry the cache's own
end codes — the loader dequantizes them DIRECTLY (one round, no
reconstruction) and derives the delta vector on demand.

THE W16 CENTERED RETRIEVAL FRAME (RetrievalFrame, this module): spec N17
says "cos-sim measures content overlap = relevance" — but the ABSOLUTE
vectors on both sides carry a large COMMON component (the system prompt's
state + the model's generic-text response), which the box measured as a
~0.70 cosine floor with a 0.016 top-3 spread (gold doc OUTSIDE the top-3
— ranking degenerates to norm/length effects). The frame centers both
sides: the query by the reset point's own vector + the corpus delta mean,
the candidates by the same mean (see RetrievalFrame's docstring for the
measured discrimination-gap improvement; scripts/w16_probe_retrieval.py
reproduces the box's failure signature and the fix). The frame is
OPTIONAL end-to-end: run_index.py writes retrieval_frame.npz beside the
faiss index and builds the index over the CENTERED stream;
answer_query/preselect/rerank pick it up automatically when the file
exists (the absolute behavior stays bit-identical without it).

W20.4.3: the vector builds, the centered-frame math and the preselect/
rerank plumbing are TORCH-NATIVE fp32 (the device-parametric dequant;
CPU-resident codes decode on CPU — the legacy reader behavior). The one
host crossing per faiss hand-off is the FOREIGN BOUNDARY (PROPOSAL rev
2) — each is marked `FAISS-EDGE` (the train block, the add flush, the
outbound query, the inbound labels). The W16 frame side file rides
snapshot.py's save/load edge (the npz hop lives THERE; this module
stays host-array-free outside the faiss boundary).

W20.10.2 — THE BATCHED FRAME BUILDS (PROPOSAL §3.6, the K4 serving
side): the §4 vector builds (dequant + concat + L2-normalize) are
batched GPU passes. The loader's per-unit dequant loops group their
units by quantizer (resolve_quantizer's registry identity = one
geometry: same d/bits/seed; the S units and M1/M2 are separate
groups), and each group rides ONE tq_dequant_batched call (the kernel
path, CUDA+eligible — the dispatcher's own predicate; every other box
keeps the per-unit q.dequant loop, bit-identical to 4.3's build). The
cosine-frame normalize is ONE batched pass per train/add block —
tq_l2norm_rows when the extension provides it (the availability probe
idiom; torch's .norm(dim=-1) twin otherwise). faiss still receives
its vectors ONCE at the boundary: the same single FAISS-EDGE host hop
per site, now feeding from the GPU-produced rows.

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
import numbers
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import faiss
import torch

import _paths  # noqa: F401  (house rule: anchor BEFORE the sibling imports)
from codebooks import get_codebook
from ingest import MANIFEST_NAME, load_system_state
from snapshot import load_chunk, load_frame_npz, save_frame_npz
from tq_cache import resolve_quantizer
from turboquant import SEEDS, _load_tq_kernel, _tq_kernel_available

__all__ = [
    "IndexConfig",
    "ChunkVectorLoader",
    "RetrievalFrame",
    "build_retrieval_frame",
    "save_retrieval_frame",
    "load_retrieval_frame",
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


# ------------------------------------------- W20.10.2: the batched passes --
def _dequant_rows(q, codes_list):
    """ONE tq_dequant_batched over one geometry group's codes (the serving
    batch, PROPOSAL §3.6): the group's packed idx streams stack into the
    entry's (M, L) rows, the 0-dim fp32 norms into the (M,) vector, and
    the whole group decode rides ONE stream-ordered kernel chain — the
    same call turboquant's _dequant_kernel makes at M=1 (row ownership
    is per-CTA and fixed-order, so each unit's reconstruction is the
    M=1 decode exactly; no host round trip of any kind).

    Returns the (M, d) fp32 rows on the codes' device, or None when the
    dispatcher would NOT route this decode to the kernel (CPU codes,
    pre-build, ineligible geometry, sketch-carrying codes) — the caller
    falls back to the per-unit q.dequant loop, bit-identical to 4.3's
    build. The routing decision is the dispatcher's own predicate
    (q._dequant_kernel_eligible on the first unit), evaluated after the
    per-unit _check_codes belt (metadata-only, loud frame drift — the
    dispatcher's own call order); the co-residency belt names both
    devices loudly (_check_kernel_devices' discipline)."""
    C = _load_tq_kernel()
    if C is None or not codes_list:
        return None
    first = codes_list[0]
    if not _tq_kernel_available(first.idx_lo):
        return None
    for i, c in enumerate(codes_list):
        q._check_codes(c)      # metadata-only; loud frame drift, the
        # dispatcher's own belt (_dequant_flat calls it before routing)
        if c.device != first.device:
            raise RuntimeError(
                f"index.py batched decode: unit {i} sits on {c.device} "
                f"but the group's first unit sits on {first.device} — a "
                f"mixed-device frame; the loader's codes must be "
                f"co-resident before the frame build")
    if not q._dequant_kernel_eligible(first, torch.float32):
        return None
    dev = first.device
    signs = q._device_signs(dev)
    cb_lo, cb_hi = q._device_codebooks(dev)
    codes_lo = torch.stack([c.idx_lo for c in codes_list]).contiguous()
    codes_hi = torch.stack([c.idx_hi for c in codes_list]).contiguous()
    norms = torch.stack([c.norm for c in codes_list])
    return C.tq_dequant_batched(
        codes_lo, codes_hi, norms, signs,
        cb_lo.centroids, cb_hi.centroids,
        q.n_lo, q.bits_lo, q.n_hi, q.bits_hi,
        torch.float32)


def _frame_rows(q, codes_list) -> torch.Tensor:
    """(M, d) fp32 rows for one geometry group — ONE tq_dequant_batched on
    the kernel path (W20.10.2), the per-unit q.dequant dispatcher loop
    stacked otherwise (bit-identical to 4.3's per-unit build; each
    q.dequant routes itself — the M=1 kernel call on CUDA+eligible, the
    torch twin everywhere else)."""
    rows = _dequant_rows(q, codes_list)
    if rows is None:
        rows = torch.stack([q.dequant(c) for c in codes_list])
    return rows


def _l2norm_rows(mat: torch.Tensor) -> torch.Tensor:
    """The (M,) row L2 norms of an (M, N) block — tq_l2norm_rows when the
    extension provides it for these rows (the availability probe idiom:
    the kernel-pair probe plus the W20.9.2 name check, so a pre-W20.9
    build falls back), torch's x.norm(dim=-1) otherwise (the
    dispatcher-level convention — the W20.9.3 differential's own twin
    reference)."""
    C = _load_tq_kernel()
    if (C is not None and hasattr(C, "tq_l2norm_rows")
            and _tq_kernel_available(mat) and mat.dim() == 2
            and mat.is_contiguous()
            and mat.dtype in (torch.float32, torch.float16)):
        return C.tq_l2norm_rows(mat)
    return mat.norm(dim=-1)


def _group_by_quantizer(units):
    """W20.10.2 — the per-geometry grouping: (q, codes) pairs in
    first-appearance order → [(q, positions, codes)] with `positions`
    the pairs' §4 indices. resolve_quantizer's registry identity IS the
    geometry (same instance = same d/bits/seed frame; different kinds or
    dims never share an instance), so id(q) groups exactly the units a
    single tq_dequant_batched call can serve."""
    groups = []
    by_q = {}
    for pos, (q, codes) in enumerate(units):
        g = by_q.get(id(q))
        if g is None:
            g = (q, [], [])
            by_q[id(q)] = g
            groups.append(g)
        g[1].append(pos)
        g[2].append(codes)
    return groups


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_int(value, name: str, minimum: int = 1, ctx: str = "") -> int:
    """Loud int validation (bools are ints — refuse them explicitly)."""
    prefix = f"{ctx}: " if ctx else ""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
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
def _fp32_bytes(t: torch.Tensor) -> bytes:
    """sha256 feed: the little-endian fp32 byte image of a 1-D tensor —
    bit-identical to the host-array-era tobytes() feed (the codebook
    digests in existing .meta.json files stay stable; metadata only —
    the tables are tiny, <= 31 values)."""
    return bytes(t.detach().to("cpu", torch.float32)
                .contiguous().view(torch.uint8).tolist())


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
    tables = cb.device_tensors("cpu")   # the 2.2 mirror — same fp32 bytes
    h = sha256()
    h.update(f"codebook|bits={cb.bits}|d={cb.d}|"
             f"levels={int(tables.centroids.shape[0])}"
             .encode("utf-8"))
    h.update(_fp32_bytes(tables.centroids))
    h.update(_fp32_bytes(tables.boundaries))
    h.update(f"|mse_per_variance={float(cb.mse_per_variance)!r}".encode("utf-8"))
    digest = h.hexdigest()
    _CODEBOOK_SHA_CACHE[key] = digest
    return digest


def _frame_codebook_sha256(unit_dims: Optional[Iterable[int]] = None
                           ) -> Dict[str, str]:
    """The production retrieval frame's codebook digests (side metadata).

    W17: `unit_dims` extends the pinned set beyond the canonical S/M1/M2
    unit (2^19) — the ACTUAL unit dims of the corpus's M1/M2 codes when
    the loader geometry is a mem_size experiment (d = 32*mem*128: 2^22
    at mem 1024, ...). Each (bits, d) solves+cached deterministically
    (codebooks.get_codebook); pass the dims you mean (an uncached pair
    SOLVES a new codebook — the pin documents exactly what decoded the
    corpus)."""
    dims = [_FRAME_UNIT_D]
    if unit_dims:
        for d in unit_dims:
            d = int(d)
            if d > 0 and d not in dims:
                dims.append(d)
    return {f"b{b}_d{d}": codebook_sha256(b, d)
            for d in dims for b in _FRAME_BITS}


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
        if manifest.get("protocol") not in ("delta-v1",):
            raise ValueError(
                f"ChunkVectorLoader: manifest protocol "
                f"{manifest.get('protocol')!r} != 'delta-v1' — this loader "
                f"reconstructs the D4 delta protocol family only (the "
                f"chunk snapshots' own 'protocol' field selects the layout: "
                f"'delta-v1' deltas or 'absolute' end codes — both W16-legit)")
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

    @property
    def device(self) -> torch.device:
        """W20.13.2: the loader's working device — the CORPUS side's
        answer to "which device do we score on". The codes' device (the
        reset point's own units, probed cheaply — metadata only, no
        dequant, no sync): the npz disk edge (from_arrays) is CPU today,
        so this is CPU for every on-disk corpus, and a future
        CUDA-resident loader reports CUDA. vector()/delta_vector()
        reconstruct on the same device (the docstrings' "on the codes'
        device"), so rerank's query hop targets THIS device — the ONE
        small operand moves, never the candidate stream. A
        mixed-device reset point (never producible by from_arrays)
        refuses loudly: never a silent cross-device fallback."""
        devs = {c.device for c in self.system.s_codes.values()}
        for c in (self.system.m1_codes, self.system.m2_codes):
            if c is not None:
                devs.add(c.device)
        if not devs:
            raise RuntimeError(
                "ChunkVectorLoader.device: the reset point carries no "
                "codes — a corrupt system_state.npz (the ingest driver "
                "always writes S layers)")
        if len(devs) > 1:
            raise RuntimeError(
                f"ChunkVectorLoader.device: the reset point's units sit "
                f"on mixed devices {sorted(str(d) for d in devs)} — "
                f"refusing to pick one silently (TQCodes.to(device) is "
                f"the caller's move)")
        return devs.pop()

    def chunk_ids(self) -> List[int]:
        """All ingested chunk ids, ascending (the iter_vectors order)."""
        return sorted(self._chunks)

    def _system_pieces(self) -> List[torch.Tensor]:
        """dequant(sys) per unit in §4 order (S ascending, M1, M2) — the
        shared zero-point vector of the delta protocol.

        W20.10.2: the units group by their quantizer (one geometry per
        group — the S units and M1/M2 are separate groups), each group's
        system codes ride ONE tq_dequant_batched on the kernel path
        (CUDA+eligible — the dispatcher's own predicate); the fallback is
        the per-unit q.dequant loop, bit-identical to 4.3's build."""
        units = []
        for L in sorted(self.system.s_codes):
            sysc = self.system.s_codes[L]
            units.append((resolve_quantizer("S", int(sysc.d), self.bits),
                          sysc))
        for name, sysc in (("M1", self.system.m1_codes),
                           ("M2", self.system.m2_codes)):
            if sysc is not None:
                units.append((resolve_quantizer(
                    name, int(sysc.d), self.bits), sysc))
        pieces: List[Optional[torch.Tensor]] = [None] * len(units)
        for q, positions, codes_group in _group_by_quantizer(units):
            rows = _frame_rows(q, codes_group)
            for j, u in enumerate(positions):
                pieces[u] = rows[j]
        return pieces

    def system_vector(self) -> torch.Tensor:
        """W16: the reset point's own §4 vector (dequantized system codes,
        fp32, (dims,)) — the common component every ABSOLUTE vector in the
        corpus carries (the retrieval frame subtracts it). W20.4.3: a
        torch tensor on the codes' device — the faiss hand-offs own the
        host hop."""
        vec = torch.cat(self._system_pieces()).to(torch.float32)
        if vec.shape != (self._dims,):
            raise ValueError(
                f"ChunkVectorLoader.system_vector: reconstructed "
                f"{vec.shape}, expected ({self._dims},) — §4 order length "
                f"drift")
        return vec

    def _load_snap(self, chunk_id: int, cid: int):
        path = os.path.join(self.disk_dir, self._chunks[cid])
        if not os.path.isfile(path):
            raise ValueError(
                f"ChunkVectorLoader.vector: {path} (manifest entry for "
                f"chunk {cid}) does not exist — the snapshot tree is "
                f"incomplete; re-run the IngestDriver")
        snap = load_chunk(path)  # sha256-verified (snapshot.py, W5.1)
        if snap.protocol not in ("delta-v1", "absolute"):
            raise ValueError(
                f"ChunkVectorLoader: chunk {cid} snapshot protocol "
                f"{snap.protocol!r} not in ('delta-v1', 'absolute') — this "
                f"loader reads the D4 delta-protocol family only")
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
        return snap

    def _unit_pairs(self, snap):
        """(name, kind, sys_codes, chunk_codes) per §4 unit — the shared
        validation for vector()/delta_vector()."""
        pairs = []
        for L in sorted(self.system.s_codes):
            sysc = self.system.s_codes[L]
            delta = snap.s_codes[L]
            if int(delta.d) != int(sysc.d):
                raise ValueError(
                    f"ChunkVectorLoader.vector: chunk {snap.chunk_id} layer "
                    f"{L} codes d={delta.d} != system d={sysc.d} — unit-dim "
                    f"drift")
            pairs.append((f"L{L}", "S", sysc, delta))
        for name, sysc, delta in (
                ("M1", self.system.m1_codes, snap.m1_codes),
                ("M2", self.system.m2_codes, snap.m2_codes)):
            if (sysc is None) != (delta is None):
                raise ValueError(
                    f"ChunkVectorLoader.vector: chunk {snap.chunk_id} "
                    f"{name} codes "
                    f"{'missing' if delta is None else 'present'} while the "
                    f"reset point's are "
                    f"{'present' if sysc is not None else 'missing'} — the "
                    f"delta protocol requires both sides of every unit")
            if sysc is None:
                continue
            if int(delta.d) != int(sysc.d):
                raise ValueError(
                    f"ChunkVectorLoader.vector: chunk {snap.chunk_id} "
                    f"{name} codes d={delta.d} != system d={sysc.d} — "
                    f"unit-dim drift")
            pairs.append((name, name, sysc, delta))
        return pairs

    def vector(self, chunk_id: int) -> torch.Tensor:
        """The chunk's absolute retrieval vector: fp32 torch tensor,
        shape (dims,), on the codes' device.

        protocol-aware (W16): 'delta-v1' snapshots reconstruct the
        DOUBLE-ROUND absolute (dequant(sys) + dequant(delta), the legacy
        contract); 'absolute' snapshots dequant the END CODES directly —
        ONE round, strictly tighter than the double-round reconstruction.
        """
        if isinstance(chunk_id, bool) or not isinstance(
                chunk_id, numbers.Integral):
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
        snap = self._load_snap(chunk_id, cid)
        pairs = self._unit_pairs(snap)
        # W20.10.2 — the batched frame build: the units group by their
        # quantizer (one geometry per group — the S units and M1/M2 are
        # separate groups), each group rides ONE tq_dequant_batched on
        # the kernel path (CUDA+eligible — the dispatcher's own
        # predicate; the fallback is the per-unit q.dequant loop,
        # bit-identical to 4.3's build), and the §4 pieces land in unit
        # order.
        q_groups = []               # (q, positions, sys_group, chunk_group)
        by_q = {}
        for u, (_name, kind, sysc, chunk_codes) in enumerate(pairs):
            q = resolve_quantizer(kind, int(sysc.d), self.bits)
            g = by_q.get(id(q))
            if g is None:
                g = (q, [], [], [])
                by_q[id(q)] = g
                q_groups.append(g)
            g[1].append(u)
            g[2].append(sysc)
            g[3].append(chunk_codes)
        pieces: List[Optional[torch.Tensor]] = [None] * len(pairs)
        for q, positions, sys_group, chunk_group in q_groups:
            if snap.protocol == "absolute":
                rows = _frame_rows(q, chunk_group)
                for j, u in enumerate(positions):
                    pieces[u] = rows[j]
            else:
                # delta-v1: the double-round absolute — the group's sys and
                # chunk codes ride ONE stacked batched decode (sys rows
                # first, then the chunk rows; the add is per unit)
                rows = _frame_rows(q, [*sys_group, *chunk_group])
                m = len(positions)
                for j, u in enumerate(positions):
                    pieces[u] = rows[j] + rows[m + j]
        vec = torch.cat(pieces).to(torch.float32)
        if vec.shape != (self._dims,):
            raise ValueError(
                f"ChunkVectorLoader.vector: chunk {cid} reconstructed "
                f"{vec.shape} vectors, expected ({self._dims},) — §4 order "
                f"length drift (manifest {self.manifest.get('vector_dims')})")
        return vec

    def delta_vector(self, chunk_id: int) -> torch.Tensor:
        """W16: the chunk's DELTA vector (what the chunk's prefill ADDED on
        top of the reset point), fp32 torch tensor, shape (dims,), §4
        order, on the codes' device.

        protocol-aware: 'delta-v1' dequants the stored DELTA codes (the D4
        layout); 'absolute' computes dequant(end codes) - dequant(sys) —
        the same quantity, one dequant round on each side. This is the
        CONTENT vector the centered retrieval frame scores (N17: "cos-sim
        measures content overlap" — the system reset point is not
        content)."""
        if isinstance(chunk_id, bool) or not isinstance(
                chunk_id, numbers.Integral):
            raise TypeError(
                f"ChunkVectorLoader.delta_vector: chunk_id must be an int, "
                f"got {type(chunk_id).__name__}: {chunk_id!r}")
        cid = int(chunk_id)
        if cid not in self._chunks:
            raise ValueError(
                f"ChunkVectorLoader.delta_vector: chunk {cid} is not in the "
                f"manifest — known: {self.chunk_ids()[:8]}")
        snap = self._load_snap(chunk_id, cid)
        pairs = self._unit_pairs(snap)
        # W20.10.2 — the batched frame build (the delta twin of vector():
        # per-geometry groups, ONE tq_dequant_batched per group on the
        # kernel path; the fallback is the per-unit q.dequant loop,
        # bit-identical to 4.3's build; the operand order is the legacy
        # one — chunk first for the absolute subtraction).
        q_groups = []               # (q, positions, sys_group, chunk_group)
        by_q = {}
        for u, (_name, kind, sysc, chunk_codes) in enumerate(pairs):
            q = resolve_quantizer(kind, int(sysc.d), self.bits)
            g = by_q.get(id(q))
            if g is None:
                g = (q, [], [], [])
                by_q[id(q)] = g
                q_groups.append(g)
            g[1].append(u)
            g[2].append(sysc)
            g[3].append(chunk_codes)
        pieces: List[Optional[torch.Tensor]] = [None] * len(pairs)
        for q, positions, sys_group, chunk_group in q_groups:
            if snap.protocol == "absolute":
                # dequant(end codes) - dequant(sys): the chunk rows first,
                # then the sys rows — the legacy subtraction order
                rows = _frame_rows(q, [*chunk_group, *sys_group])
                m = len(positions)
                for j, u in enumerate(positions):
                    pieces[u] = rows[j] - rows[m + j]
            else:
                rows = _frame_rows(q, chunk_group)
                for j, u in enumerate(positions):
                    pieces[u] = rows[j]
        vec = torch.cat(pieces).to(torch.float32)
        if vec.shape != (self._dims,):
            raise ValueError(
                f"ChunkVectorLoader.delta_vector: chunk {cid} reconstructed "
                f"{vec.shape}, expected ({self._dims},) — §4 order drift")
        return vec

    def iter_vectors(self, chunk_ids: Optional[Iterable[int]] = None
                     ) -> Iterator[torch.Tensor]:
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

    def iter_delta_vectors(self, chunk_ids: Optional[Iterable[int]] = None
                           ) -> Iterator[torch.Tensor]:
        """W16: yield DELTA vectors in chunk-id order (the centered
        retrieval frame's corpus stream)."""
        if chunk_ids is None:
            ids = self.chunk_ids()
        else:
            ids = [int(c) for c in chunk_ids]
            unknown = [c for c in ids if c not in self._chunks]
            if unknown:
                raise ValueError(
                    f"ChunkVectorLoader.iter_delta_vectors: unknown chunk "
                    f"ids {unknown[:8]} — known: {self.chunk_ids()[:8]}")
        for cid in ids:
            yield self.delta_vector(cid)


# ------------------------------------------------- W27: ingest regime -----
def corpus_memory_state(loader, n_sample: int = 3, rel_tol: float = 0.01
                        ) -> Optional[bool]:
    """W27 — is the corpus's M1/M2 memory subspace EMPTY, CARRIES CONTENT,
    or ABSENT?  The retrieval-side query prefill must run in the SAME
    write regime the chunk vectors were CAPTURED in; this introspection
    supplies the corpus's side of that contract.

    Returns:
      None   the corpus carries no M1/M2 units at all (or no chunks) —
             there is no memory regime to match.
      True   EMPTY — every sampled chunk's memory delta is ~0 relative to
             its own full delta: the corpus was INGESTED WITH ZERO WRITE
             GATES (the P3 no-op regime — the box's
             ragga_ingested_test100 is exactly this: its chunk m1/m2
             codes dequantize to exact zeros, so the memory subspace is
             inert and the §4 metric rides the S subspace alone).
      False  CARRIES CONTENT — at least one sampled chunk wrote real
             memory content at ingest; the query side needs the SAME
             trained-gates artifact to produce a matching query vector.

    Mechanics: the §4 delta vector's layout is [S layers ascending, M1,
    M2], so the memory subspace is the tail of len = system m1.d + m2.d;
    `rel_tol` separates the EXACT zeros of a no-op write (idempotent
    requant of a norm-0 unit) from real content (which dominates the
    vector — the M1/M2 dims are the large majority of d in production
    geometry).  `loader` only needs `.chunk_ids()` and
    `.delta_vector(cid)` (ChunkVectorLoader has both; tests may stub).
    """
    ids = list(loader.chunk_ids()) if hasattr(loader, "chunk_ids") else []
    if not ids:
        return None
    system = getattr(loader, "system", None)
    d_tail = 0
    for attr in ("m1_codes", "m2_codes"):
        c = getattr(system, attr, None) if system is not None else None
        if c is not None:
            d_tail += int(c.d)
    if d_tail == 0:
        return None
    for cid in ids[:max(1, int(n_sample))]:
        v = loader.delta_vector(int(cid))
        if not torch.is_tensor(v) or v.numel() < d_tail:
            raise ValueError(
                f"corpus_memory_state: chunk {cid} delta vector "
                f"{tuple(v.shape) if torch.is_tensor(v) else type(v)} is "
                f"shorter than the memory tail {d_tail} — the loader and "
                f"the system state disagree on the §4 layout")
        tail = v[v.numel() - d_tail:]
        v_norm = float(torch.linalg.vector_norm(v.float()))
        t_norm = float(torch.linalg.vector_norm(tail.float()))
        if t_norm > rel_tol * max(1.0, v_norm):
            return False
    return True


# ---------------------------------------------------- W16: the frame ------
FRAME_META_KEYS = ("kind", "dims", "system_ref", "n_mean_chunks",
                   "chunk_protocol", "created_utc")


@dataclass
class RetrievalFrame:
    """W16 — the centered retrieval frame (spec N17 done right).

    THE FAILURE IT FIXES (measured, scripts/w16_probe_retrieval.py): both
    the query vector and every chunk vector are ABSOLUTE states (sys +
    content), so plain cos-sim carries a large COMMON term — the system
    prompt's state plus the model's generic-text response. On the box the
    top-3 scores collapse to a ~0.70 floor with a 0.016 spread (top-3
    [0.7232, 0.7072, 0.7071], gold doc outside) — the ranking degenerates
    to norm/length effects. The centered frame scores CONTENT:

        q_centered  = (q_abs - sys_vector) - mean_vector
        c_centered  = (c_delta)             - mean_vector
        score       = cos(q_centered, c_centered)

    sys_vector  = the reset point's own §4 vector (the delta protocol's
                  zero — protocol-principled, no corpus statistics)
    mean_vector = the corpus mean of the chunk DELTAS (removes the residual
                  generic-text direction all English prose shares)

    Everything stays in the §4 cache-state space (N17: no embedder, no
    chunk text, no hidden states) — this is a metric-level centering, and
    the measured discrimination gap (best content-match minus best
    non-match) grows ~3x over the absolute frame in the box-matching
    regime.
    """

    sys_vector: torch.Tensor
    mean_vector: torch.Tensor
    dims: int
    system_ref: str = ""
    n_mean_chunks: int = 0
    chunk_protocol: str = "delta-v1"

    def __post_init__(self) -> None:
        self.sys_vector = torch.as_tensor(
            self.sys_vector, dtype=torch.float32).contiguous()
        self.mean_vector = torch.as_tensor(
            self.mean_vector, dtype=torch.float32).contiguous()
        if self.sys_vector.shape != (self.dims,) \
                or self.mean_vector.shape != (self.dims,):
            raise ValueError(
                f"RetrievalFrame: sys_vector {tuple(self.sys_vector.shape)} / "
                f"mean_vector {tuple(self.mean_vector.shape)} != "
                f"({self.dims},)")
        if not (torch.isfinite(self.sys_vector).all()
                and torch.isfinite(self.mean_vector).all()):
            raise ValueError(
                "RetrievalFrame: non-finite frame vectors — refusing to "
                "center retrieval through a broken frame")

    # ------------------------------------------------------------ centering --
    def center_query(self, qvec_abs) -> torch.Tensor:
        """(q_abs - sys) - mean — the query's content vector (fp32, on the
        frame's device: the query hops there once, W20.4.3)."""
        q = torch.as_tensor(qvec_abs, dtype=torch.float32).to(
            self.sys_vector.device)
        if q.shape != (self.dims,):
            raise ValueError(
                f"RetrievalFrame.center_query: query {tuple(q.shape)} != "
                f"({self.dims},) — the §4 vector and the frame disagree")
        return (q - self.sys_vector) - self.mean_vector

    def center_delta(self, delta_vec) -> torch.Tensor:
        """delta - mean — the chunk's content vector (fp32, on the frame's
        device: the candidate hops there once)."""
        d = torch.as_tensor(delta_vec, dtype=torch.float32).to(
            self.mean_vector.device)
        if d.shape != (self.dims,):
            raise ValueError(
                f"RetrievalFrame.center_delta: delta {tuple(d.shape)} != "
                f"({self.dims},) — §4 order drift")
        return d - self.mean_vector

    def check_loader(self, loader: "ChunkVectorLoader") -> None:
        """Loud frame-vs-corpus guards: dims and the reset-point reference
        (a re-ingestion with a different system prompt must invalidate the
        frame, not silently mis-center every query). W20.13.2 adds the
        device belt: the frame's vectors and the loader's codes must be
        co-resident — rerank centers the query on the FRAME's device and
        scores the LOADER's candidates, so a frame/corpus device split
        would otherwise fail at the first dot with an anonymous
        cross-device error instead of this named one."""
        if int(self.dims) != int(loader.dims):
            raise ValueError(
                f"RetrievalFrame: dims {self.dims} != the corpus's "
                f"{loader.dims} — frame/corpus drift")
        ref = loader.system.reference()
        if self.system_ref and ref and self.system_ref != ref:
            raise ValueError(
                f"RetrievalFrame: system_ref {self.system_ref!r} != the "
                f"corpus's {ref!r} — the reset point changed after the "
                f"frame was built; re-run run_index.py (the centering is "
                f"meaningless against a foreign zero point)")
        loader_dev = getattr(loader, "device", None)
        if loader_dev is not None and self.sys_vector.device != loader_dev:
            raise ValueError(
                f"RetrievalFrame: the frame's vectors sit on "
                f"{self.sys_vector.device} but the corpus's codes sit on "
                f"{loader_dev} — a frame/corpus device split; rebuild "
                f"the frame on the corpus's device "
                f"(build_retrieval_frame does this by construction; a "
                f"loaded frame may need its vectors moved with the "
                f"TQCodes.to(device) discipline)")


def build_retrieval_frame(loader: "ChunkVectorLoader",
                          max_chunks: Optional[int] = None
                          ) -> RetrievalFrame:
    """One pass over the corpus deltas → the RetrievalFrame (sys vector +
    the mean of the chunk deltas). D5: one vector in memory at a time; the
    mean is an fp32 accumulator. max_chunks caps the mean's sample (None =
    the whole corpus — the estimator's noise shrinks as 1/sqrt(n)).
    W20.4.3: torch-native (fp64 accumulator on the sys vector's device,
    widened to fp32 once at the end — the same arithmetic as the
    host-array era)."""
    sys_vector = loader.system_vector()
    ids = loader.chunk_ids()
    if max_chunks is not None:
        ids = ids[:max_chunks]
    if not ids:
        raise ValueError(
            "build_retrieval_frame: the corpus is empty — nothing to "
            "estimate the mean from")
    acc = torch.zeros(loader.dims, dtype=torch.float64,
                      device=sys_vector.device)
    protocols = set()
    for cid in ids:
        # per-snapshot protocol (the loader validates everything else)
        path = os.path.join(loader.disk_dir, loader._chunks[cid])
        snap = load_chunk(path)
        protocols.add(snap.protocol)
        acc = acc + loader.delta_vector(cid)
    if len(protocols) > 1:
        raise ValueError(
            f"build_retrieval_frame: mixed snapshot protocols "
            f"{sorted(protocols)} — a centered frame needs one layout")
    mean = (acc / len(ids)).to(torch.float32)
    return RetrievalFrame(
        sys_vector=sys_vector, mean_vector=mean, dims=int(loader.dims),
        system_ref=loader.system.reference(), n_mean_chunks=len(ids),
        chunk_protocol=next(iter(protocols)))


FRAME_FILE = "retrieval_frame.npz"


def save_retrieval_frame(loader: "ChunkVectorLoader", frame: RetrievalFrame,
                         path: Optional[str] = None) -> str:
    """Persist the frame beside the snapshots (atomic tmp+replace). The
    npz carries the two fp32 vectors + a JSON meta (validated on load).
    W20.4.3: the npz write rides snapshot.save_frame_npz — the disk
    edge lives THERE (this module's vectors stay torch-resident)."""
    if path is None:
        path = os.path.join(loader.disk_dir, FRAME_FILE)
    path = os.fspath(path)
    meta = {
        "kind": "retrieval-frame-v1",
        "dims": int(frame.dims),
        "system_ref": str(frame.system_ref),
        "n_mean_chunks": int(frame.n_mean_chunks),
        "chunk_protocol": str(frame.chunk_protocol),
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    return save_frame_npz(path, frame.sys_vector, frame.mean_vector, meta)


def load_retrieval_frame(disk_dir: str,
                         path: Optional[str] = None) -> RetrievalFrame:
    """Load the frame written by save_retrieval_frame (loud on drift,
    partial files, or a foreign format). `path` overrides the default
    <disk_dir>/retrieval_frame.npz (tests)."""
    path = path if path is not None else os.path.join(
        os.fspath(disk_dir), FRAME_FILE)
    if not os.path.isfile(path):
        raise ValueError(
            f"load_retrieval_frame: {path} not found — run_index.py "
            f"writes it at index-build time")
    sys_vector, mean_vector, meta = load_frame_npz(path)
    if not isinstance(meta, dict) or meta.get("kind") != "retrieval-frame-v1":
        raise ValueError(
            f"load_retrieval_frame: {path} meta kind "
            f"{meta.get('kind')!r} != 'retrieval-frame-v1' — a foreign "
            f"frame file; refusing to center through it")
    dims = int(meta.get("dims", 0))
    if dims <= 0:
        raise ValueError(
            f"load_retrieval_frame: {path} meta dims {meta.get('dims')!r}")
    return RetrievalFrame(
        sys_vector=sys_vector, mean_vector=mean_vector, dims=dims,
        system_ref=str(meta.get("system_ref", "")),
        n_mean_chunks=int(meta.get("n_mean_chunks", 0)),
        chunk_protocol=str(meta.get("chunk_protocol", "delta-v1")))


def iter_centered_vectors(loader: "ChunkVectorLoader", frame: RetrievalFrame
                           ) -> Iterator[torch.Tensor]:
    """The build-side stream: centered content vectors in chunk-id order
    (feed build_index — it L2-normalizes and validates per stream row)."""
    frame.check_loader(loader)
    for cid in loader.chunk_ids():
        yield frame.center_delta(loader.delta_vector(cid))


# ---------------------------------------------------------------------------
# build / load
# ---------------------------------------------------------------------------
def _validate_streamed(v, config: IndexConfig, pos: int) -> torch.Tensor:
    """Validate ONE streamed corpus vector → an fp32 (d,) row (W20.10.2
    split of 4.3's _normalize_streamed: host-array inputs coerce
    zero-copy at the tensor boundary; the per-row norm guards moved to
    the batched _normalize_rows — same messages, same 1-based stream
    positions)."""
    row = torch.as_tensor(v)
    if row.dim() != 1:
        raise ValueError(
            f"build_index: stream position {pos}: expected a 1-D (d,) "
            f"vector, got shape {tuple(row.shape)}")
    if row.shape[0] != int(config.d):
        raise ValueError(
            f"build_index: stream position {pos}: vector dim "
            f"{row.shape[0]} != config.d={int(config.d)} — refusing to "
            f"build an index over a mismatched frame (SPECIFICATION §8 "
            f"fixes d=13,631,488; the loader/stream and IndexConfig must "
            f"agree)")
    return row.to(torch.float32)


def _normalize_rows(mat: torch.Tensor, base: int) -> torch.Tensor:
    """L2-normalize an (M, d) block of validated fp32 stream rows — the
    frame's cosine normalize as ONE batched pass (W20.10.2, PROPOSAL
    §3.6): ONE tq_l2norm_rows launch + ONE broadcast divide on the
    kernel path (via _l2norm_rows' availability probe); torch's
    .norm(dim=-1) twin otherwise. The per-row guards keep 4.3's
    semantics and messages (non-finite first, then zero-norm; `base` is
    the block's first 1-based stream position) — the only host reads
    are the .all()/.any() guard checks and the raise's own message
    edge, one per BLOCK now instead of one per vector."""
    norms = _l2norm_rows(mat)
    if not bool(torch.isfinite(norms).all()):
        bad = (~torch.isfinite(norms)).nonzero().reshape(-1)
        j = int(bad[0])
        raise ValueError(
            f"build_index: stream position {base + j}: non-finite vector "
            f"(norm {float(norms[j])!r})")
    if bool((norms == 0.0).any()):
        bad = (norms == 0.0).nonzero().reshape(-1)
        j = int(bad[0])
        raise ValueError(
            f"build_index: stream position {base + j}: zero-norm vector — "
            f"the cosine frame cannot normalize it (a zero §4 vector means "
            f"every unit collapsed; investigate the corpus, don't index "
            f"it)")
    return mat / norms.unsqueeze(1)


def _index_metadata(config: IndexConfig, n_train: int, ntotal: int,
                    extra_metadata: Optional[dict],
                    unit_dims: Optional[Iterable[int]] = None) -> dict:
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
        "codebook_sha256": _frame_codebook_sha256(unit_dims),
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


def _faiss_add(index, mat: torch.Tensor) -> None:
    """One ADD flush (D5 batching): the normalized (M, d) torch block
    handed to faiss as ONE contiguous host fp32 matrix — the foreign
    boundary, once per flush (W20.4.3; the call sequence itself is
    unchanged). W20.10.2: the block arrives ALREADY normalized from the
    batched frame pass (_normalize_rows — GPU-produced rows; faiss is
    fed once at the boundary, PROPOSAL §3.6)."""
    if mat.shape[0] == 0:
        return
    # FAISS-EDGE: the one sanctioned host crossing — faiss consumes host
    # float32 (PROPOSAL rev 2, the foreign boundary).
    index.add(mat.detach().to("cpu", torch.float32)
              .contiguous().numpy())  # FAISS-EDGE (the host hop)


def build_index(vectors_iter, config: IndexConfig = IndexConfig(),
                train_sample: int = 4096, path: str | None = None,
                extra_metadata: dict | None = None,
                min_train_warn: int | None = None,
                unit_dims: Iterable[int] | None = None
                ) -> faiss.IndexIVFPQ:
    """Build the §8 IVFADC index over a stream of raw §4 vectors.

    vectors_iter: iterable of raw (un-normalized) fp vectors, (d,) each —
    e.g. ChunkVectorLoader(disk_dir).iter_vectors(). Position i in the
    stream becomes faiss id i (the caller owns the id↔chunk mapping; the
    loader yields chunk-id order).

    W17 unit_dims: the corpus's ACTUAL M1/M2 code unit dims (e.g.
    [32*mem*128] at a mem_size experiment) — pinned into the side
    metadata's codebook digests alongside the canonical 2^19 unit (the
    callers with a loader pass [loader.system.m1_codes.d] when present).

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
    train_rows: List[torch.Tensor] = []
    for v in stream:
        train_rows.append(_validate_streamed(v, config, len(train_rows) + 1))
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
    # the ONE dense block (D5), L2-normalized as ONE batched pass
    # (W20.10.2: ONE tq_l2norm_rows + one divide on the kernel path —
    # the faiss train block feeds from the GPU-produced rows)
    train_mat = _normalize_rows(torch.stack(train_rows), base=1)
    # FAISS-EDGE: the one sanctioned host crossing — faiss consumes host
    # float32 (PROPOSAL rev 2, the foreign boundary). The train block
    # crosses ONCE here; the add loop below reuses the same host block.
    train_host = (train_mat.detach().to("cpu", torch.float32)
                  .contiguous().numpy())  # FAISS-EDGE (the host hop)

    quantizer = faiss.IndexFlatIP(int(config.d))
    index = faiss.IndexIVFPQ(quantizer, int(config.d), int(config.nlist),
                             int(config.m), int(config.nbits),
                             faiss.METRIC_INNER_PRODUCT)
    if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
        raise RuntimeError(
            f"build_index: faiss built metric_type={index.metric_type}, "
            f"expected INNER_PRODUCT — the cosine frame (module docstring) "
            f"is broken; refusing to continue")
    index.train(train_host)

    # ---- add: the train rows (batched), then the rest of the stream -----
    for i in range(0, n_train, ADD_BATCH):
        index.add(train_host[i:i + ADD_BATCH])
    batch: List[torch.Tensor] = []
    pos = n_train
    for v in stream:
        pos += 1
        batch.append(_validate_streamed(v, config, pos))
        if len(batch) >= ADD_BATCH:
            # W20.10.2: the flush block normalizes as ONE batched pass
            # (ONE tq_l2norm_rows + one divide on the kernel path) —
            # faiss is fed once at the boundary, from the GPU rows
            _faiss_add(index, _normalize_rows(
                torch.stack(batch), base=pos - len(batch) + 1))
            batch.clear()
    if batch:
        _faiss_add(index, _normalize_rows(
            torch.stack(batch), base=pos - len(batch) + 1))

    index.nprobe = int(config.nprobe)

    if path is not None:
        meta = _index_metadata(config, n_train, index.ntotal, extra_metadata,
                               unit_dims=unit_dims)
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
def _query_vector(query_vector, d: Optional[int], ctx: str) -> torch.Tensor:
    """Validate one raw query vector → a contiguous fp32 torch tensor
    (host-array inputs coerce zero-copy; W20.4.3 — the norm checks stay
    tensor-space, no host reads)."""
    q = torch.as_tensor(query_vector)
    if q.dim() != 1:
        raise ValueError(
            f"{ctx}: the query must be a 1-D (d,) vector, got shape "
            f"{tuple(q.shape)}")
    if d is not None and q.shape[0] != int(d):
        raise ValueError(
            f"{ctx}: query dim {q.shape[0]} != {int(d)} — the query's §4 "
            f"vector and the index must share the frame (SPECIFICATION §4: "
            f"'the query's cache vector and the chunk's cache vector are in "
            f"the SAME space')")
    q = q.to(torch.float32)
    norm = torch.linalg.vector_norm(q)
    if not torch.isfinite(norm):
        raise ValueError(
            f"{ctx}: non-finite query vector (norm {float(norm)!r})")
    if norm == 0.0:
        raise ValueError(
            f"{ctx}: zero-norm query — cosine is undefined; a §4 query "
            f"vector is a live cache state, never all-zero")
    return q


def preselect(index, query_vector, k: int = 100,
              frame: Optional[RetrievalFrame] = None) -> torch.Tensor:
    """§6 step 3: IVFADC preselect → candidate chunk ids.

    The query is L2-normalized (the index's cosine frame) and searched
    with index.search; returns a 1-D int64 id TENSOR with faiss's -1
    padding filtered (W20.4.3: torch-native; the ONE host crossing per
    direction is the faiss boundary, marked FAISS-EDGE). k defaults to
    spec §6's 100 (IndexConfig.preselect_k).

    W16 `frame`: when given (and the index was BUILT on centered vectors —
    run_index.py writes both together), the ABSOLUTE §4 query vector is
    centered (sys + corpus-mean subtraction) before the search; None keeps
    the legacy absolute behavior bit-identical."""
    k = _check_int(k, "k", ctx="preselect")
    d = getattr(index, "d", None)
    if d is None:
        raise TypeError(
            "preselect: `index` must be a faiss index (has .d/.search), "
            f"got {type(index).__name__}")
    q = _query_vector(query_vector, d, ctx="preselect")
    if frame is not None:
        q = frame.center_query(q)
        if q.shape != (int(d),):
            raise ValueError(
                f"preselect: the centered query {tuple(q.shape)} != "
                f"index d={d} — the frame and the index disagree "
                f"(rebuild together)")
    q_hat = (q / torch.linalg.vector_norm(q)).reshape(1, -1)
    # FAISS-EDGE: the one sanctioned host crossing — faiss consumes host
    # float32 (PROPOSAL rev 2, the foreign boundary).
    q_host = (q_hat.detach().to("cpu", torch.float32)
              .contiguous().numpy())  # FAISS-EDGE (the host hop)
    _, labels = index.search(q_host, k)
    ids = torch.from_numpy(labels.reshape(-1))  # FAISS-EDGE (inbound hop)
    ids = ids[ids >= 0].to(torch.int64)
    return ids


def rerank(loader, query_vector, candidate_ids,
           k: int = 3, frame: Optional[RetrievalFrame] = None
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """§6 step 4: exact cos-sim rerank → (top ids, scores desc).

    D5 LAZY: dequantizes the candidates' full vectors ONE AT A TIME via
    loader.vector(cid) (frame=None, the legacy absolute metric) or
    loader.delta_vector(cid) centered through `frame` (W16: the content
    metric — the query vector stays the ABSOLUTE §4 vector either way;
    the frame does the sys+mean subtraction on both sides) — exactly the
    ids passed (each exactly once, duplicates coalesced), nothing else.
    Ties keep candidate order (deterministic). A zero-norm candidate
    scores 0.0 (a legitimate stored zero state — cosine against it is 0
    by convention, documented). Returns min(k, #candidates) results.

    W20.4.3: torch-native end-to-end (shapes unchanged, N30–N32): the
    ids return as a 1-D int64 tensor, the scores as a 1-D fp32 tensor;
    candidate ids accept python ints, Integral host scalars (legacy
    host-array callers) and single-element torch tensors (preselect's
    output). No faiss here — hence no host crossing at all.
    """
    k = _check_int(k, "k", ctx="rerank")
    if not hasattr(loader, "vector"):
        raise TypeError(
            f"rerank: loader must expose .vector(chunk_id) (a "
            f"ChunkVectorLoader or a mock), got {type(loader).__name__}")
    if candidate_ids is None:
        raise TypeError("rerank: candidate_ids must be a sequence of ints")
    q = _query_vector(query_vector, None, ctx="rerank")
    if frame is not None:
        frame.check_loader(loader)
        q = frame.center_query(q)
    else:
        # W20.13.2 — the ONE query hop: the candidates define the
        # working device (the loader's own .device — metadata only,
        # no dequant), and the query is the single small operand, so
        # IT hops there once instead of every torch.dot raising
        # cross-device (the box's live-model CUDA query against the
        # npz-edge CPU corpus — the W20.4.3 regression a CUDA-less
        # dev box could never surface). A loader without .device
        # (the tests' torch-typed mock loaders) keeps q where it is —
        # the per-candidate belt below aligns the OTHER side there
        # (test-scale cost; the real path's belt is a no-op).
        target = getattr(loader, "device", None)
        if target is not None and q.device != target:
            q = q.to(target)
    qn = torch.linalg.vector_norm(q)
    if qn == 0.0 and frame is not None:
        raise ValueError(
            "rerank: the CENTERED query is zero-norm (the query's state "
            "equals the reset point + corpus mean — no content to match); "
            "cosine is undefined")
    uniq: List[int] = []
    seen = set()
    for c in candidate_ids:
        if isinstance(c, bool):
            raise TypeError(
                f"rerank: candidate ids must be ints, got "
                f"{type(c).__name__}: {c!r}")
        if torch.is_tensor(c):
            if c.numel() != 1:
                raise TypeError(
                    f"rerank: candidate ids must be ints, got a "
                    f"{tuple(c.shape)} tensor")
            c = int(c)
        elif not isinstance(c, numbers.Integral):
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
    # the cos-sim is computed in tensor space; the divide happens in fp64
    # (dot in fp32, widened exactly) — the host-array-era arithmetic.
    scores = torch.zeros(len(uniq), dtype=torch.float64, device=q.device)
    for i, cid in enumerate(uniq):
        if frame is not None:
            v = frame.center_delta(loader.delta_vector(cid))
        else:
            v = loader.vector(cid)
        if not torch.is_tensor(v):
            raise TypeError(
                f"rerank: loader.vector({cid}) must return a torch tensor "
                f"(W20.4.3), got {type(v).__name__}")
        if v.dim() != 1 or v.shape[0] != q.shape[0]:
            raise ValueError(
                f"rerank: candidate {cid} vector shape {tuple(v.shape)} != "
                f"the query's ({q.shape[0]},) — frame drift between the "
                f"loader and the query")
        v = v.to(torch.float32)
        # W20.13.2 belt: the candidate lands on q's device — a NO-OP
        # on the real path (the query hop / the frame belt already
        # aligned them; torch.Tensor.to returns self when device AND
        # dtype already match); the device-less mock loaders align
        # here, one small test-scale tensor at a time.
        if v.device != q.device:
            v = v.to(q.device, torch.float32)
        vn = torch.linalg.vector_norm(v)
        if not torch.isfinite(vn):
            raise ValueError(
                f"rerank: candidate {cid} has non-finite norm "
                f"{float(vn)!r}")
        if vn == 0.0:
            continue   # scores[i] stays 0.0 (the documented convention)
        scores[i] = torch.dot(q, v).double() / (qn.double() * vn.double())
    # stable descending order: score desc, then candidate arrival order
    order = torch.argsort(-scores, stable=True)
    top = order[: min(k, len(order))]
    ids = torch.as_tensor(uniq, dtype=torch.int64,
                          device=scores.device)[top]
    return ids, scores[top].to(torch.float32)
