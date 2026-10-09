"""
idxN.py — canonical producer and verifier for the idx1/idx2/idx3/idx4
packed-indices layout family (sub-byte LUT quantization).

This module is the unified pack/unpack implementation for all bit widths (1-4).
producer — to bit widths b in {1, 2, 3, 4}. The design goal is the SAME
approach, the SAME style, at every width: the on-disk artifact is a
permutation of the logically packed indices that places every GPU
thread's share of one 64-k tile CONTIGUOUSLY, so the production kernel
(``qgemm_per_group_lut(..., indices_layout="idxN")``, dispatch flag
q_layout=1) dequantizes straight into the mma.m16n8k16 B-fragment
registers with no shared-memory round trip.

For every width the artifact is ``<name>.idx{b}`` + ``<name>.lut_scalar``
with ``2**b`` fp16 entries per group, and the dequantization semantics
are unchanged: ``W[n, k] = LUT[n // group_size, idx[n, k]]`` (pure
codebook lookup, DEQUANT_SPEC sections 3-4).

Eligibility (all widths): ``N % 128 == 0`` and ``K % 64 == 0``. Anything
else raises — there is no fallback layout and no silent remap.

LOGICAL packing (generalizes DEQUANT_SPEC section 2, LSB-first):

* b=4: two indices per byte (the classic nibble layout; the nibble pair
  is exactly one k-pair ``{W[k], W[k+1]}`` of one output row);
* b=2: four indices per byte, LSB-first;
* b=1: eight indices per byte, LSB-first;
* b=3: eight indices per 3 bytes — a 24-bit little-endian group holding
  8 consecutive 3-bit fields.

The unit that matters for the kernel is the k-PAIR: two consecutive
values ``(idx[n, k], idx[n, k+1])`` with k even, which dequantize into
ONE mma B-fragment u32 ``{W[k], W[k+1]}`` (low half = W[k], high half =
W[k+1]). Every width packs pairs as contiguous b-bit field pairs, so a
pair always occupies ``2b`` consecutive bits of the logical stream.

NORMATIVE blob layout (must match the kernel's fragment-direct path):

* the blob is cut into tiles of ``1024*b`` bytes, one per (128-row tile
  ``t``, 64-k-value tile ``g``), at byte offset
  ``((t * K//64) + g) * 1024*b``;
* within a tile, thread (``wx`` in {0,1}, ``lane`` in [0,32)) owns a
  contiguous ``16*b``-byte chunk at ``wx*512*b + lane*16*b`` (thread
  index ``th = wx*32 + lane``);
* the chunk is the thread's 128-element share of the tile: 8 rows x 16
  k-values, organized as 64 k-PAIRS. Pair ``p`` (``p`` in [0,64)) is at
  bit offset ``2*b*p`` of the chunk, LSB-first, and covers
  ``n  = t*128 + wx*64 + v*16 + d*8 + (lane >> 2)`` and
  ``k  = g*64 + kt*16 + 2*(lane & 3) + 8*s2`` (and ``k+1``), where
  ``p = kt*16 + v*4 + d*2 + s2`` with ``kt`` in [0,4), ``v`` in [0,4),
  ``d`` in [0,2), ``s2`` in [0,2);
* at b=4 this reduces EXACTLY to the idx4 layout (one byte per pair,
  chunk byte ``p`` == ``q[n, kp]`` with ``kp = kt*8 + (lane&3) + 4*s2``):
  ``pack_idxn(idx, 4)`` is byte-identical to ``idx4.pack_idx4(idx)``
  (asserted by self_test and by the test suite).

Paired-LUT dequant style per width (why the layout is shaped this way):

* b=4: 1 byte = 1 pair = 1 fragment u32 — one 256-entry byte-indexed
  paired-LUT LDS (the current E1 path);
* b=2: 1 byte = 2 pairs = 2 fragment u32s — two 256-entry tables (the
  low nibble pair and the high nibble pair), still one LDS per u32;
* b=1: 1 byte = 4 pairs — four 256-entry tables, one LDS per u32;
* b=3: 3 bytes = 4 pairs — one 64-entry table indexed by the 6-bit pair
  field, one LDS + one field extraction per u32.

This module is the ONLY producer of the family in the repository:
scripts/palettize_qwen3_5_9b.py writes artifacts with it, the test
suite synthesizes reference blobs with it, and loaders verify artifacts
with it. This is now the sole implementation for all widths.
4-bit producer (idxN defers to its contract at b=4 and is asserted
byte-identical).
"""

from __future__ import annotations

import functools
import numpy as np

# Kernel-side tile constants (B-side tile configuration of the streaming
# kernel's fragment-direct path; must not change without a matching
# kernel change). Width-independent: the tile is 128 rows x 64 k-values
# at every b; only the byte count per tile/chunk scales with b.
ROWS_PER_TILE = 128
K_PER_GROUP = 64
PAIRS_PER_CHUNK = 64               # 64 k-pairs = 128 elements per thread
CHUNK_BYTES_PER_BIT = 2            # 16*b bytes per chunk per bit width
TILE_THREADS = 64                  # (wx, lane) chunk owners (wy duplicates)
TILE_BYTES_PER_BIT = 1024          # 64 threads * 16*b bytes

BITS_SUPPORTED = (1, 2, 3, 4)


def check_bits(bits: int) -> int:
    """Validate and normalize a bit width of the idxN family."""
    b = int(bits)
    if b not in BITS_SUPPORTED:
        raise ValueError(f"idxN supports bit widths {BITS_SUPPORTED}, got bits={bits}")
    return b


def check_eligible(N: int, K: int) -> None:
    """Raise ValueError unless the shape can be stored in the idxN layout."""
    if N % ROWS_PER_TILE != 0:
        raise ValueError(f"idxN layout requires N % 128 == 0, got N={N}")
    if K % K_PER_GROUP != 0:
        raise ValueError(f"idxN layout requires K % 64 == 0, got K={K}")


def blob_bytes(N: int, K: int, bits: int) -> int:
    """Total byte count of the idxN blob for an (N, K) tensor at `bits`."""
    check_bits(bits)
    return (int(N) * int(K) * bits) // 8


def packed_row_bytes(K: int, bits: int) -> int:
    """Bytes per row of the logical packed stream (K % 64 == 0 makes this
    exact for every supported width, including b=3)."""
    check_bits(bits)
    return (int(K) * int(bits)) // 8


@functools.lru_cache(maxsize=4)
def _chunk_element_map(bits: int):
    """(row_map, col_map) arrays of shape (64 threads, 128 elements).

    Element ``e`` of thread ``th``'s chunk (``e = 2*p + j``, j in {0,1})
    is the logical index matrix entry
    ``I[t*128 + row_map[th, e], g*64 + col_map[th, e]]``.
    ``row_map``/``col_map`` are tile-local (row in [0,128), k in [0,64));
    the tile base (t, g) is applied by the caller.
    """
    check_bits(bits)
    row_map = np.empty((TILE_THREADS, PAIRS_PER_CHUNK * 2), dtype=np.int64)
    col_map = np.empty((TILE_THREADS, PAIRS_PER_CHUNK * 2), dtype=np.int64)
    for wx in range(2):
        for lane in range(32):
            th = wx * 32 + lane
            g_lane = lane >> 2          # B-fragment column group (PTX groupID)
            i = lane & 3                # B-fragment k-offset selector
            pos = 0
            for kt in range(4):
                for v in range(4):
                    for d in range(2):
                        for s2 in range(2):
                            # pair p = kt*16 + v*4 + d*2 + s2 -> (k, k+1)
                            row_map[th, pos] = wx * 64 + v * 16 + d * 8 + g_lane
                            col_map[th, pos] = kt * 16 + 2 * i + 8 * s2
                            pos += 1
                            row_map[th, pos] = row_map[th, pos - 1]
                            col_map[th, pos] = col_map[th, pos - 1] + 1
                            pos += 1
    assert pos == PAIRS_PER_CHUNK * 2
    return row_map, col_map


def _bit_pack_chunk(elems: np.ndarray, bits: int) -> np.ndarray:
    """Pack (..., 128) LSB-first at `bits` bits per value into
    (..., 16*bits) uint8 bytes.

    b=4: byte = v0 | v1<<4 (one k-pair per byte).
    b=2: byte = v0 | v1<<2 | v2<<4 | v3<<6 (two k-pairs).
    b=1: byte = v0 | ... | v7<<7 (four k-pairs).
    b=3: 8 values -> 3 bytes (24-bit little-endian group).
    """
    if bits == 4:
        v = elems.astype(np.uint8)
        return (v[..., 0::2] | (v[..., 1::2] << 4))
    if bits == 2:
        v = elems.astype(np.uint8).reshape(*elems.shape[:-1], 32, 4)
        return (v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4)
                | (v[..., 3] << 6))
    if bits == 1:
        v = elems.astype(np.uint8).reshape(*elems.shape[:-1], 16, 8)
        out = np.zeros(elems.shape[:-1] + (16,), dtype=np.uint8)
        for j in range(8):
            out |= v[..., j] << j
        return out
    # bits == 3: 8 values per 3-byte group, 24-bit little-endian
    v = elems.astype(np.uint32).reshape(*elems.shape[:-1], 16, 8)
    acc = np.zeros(elems.shape[:-1] + (16,), dtype=np.uint32)
    for j in range(8):
        acc |= v[..., j] << (3 * j)
    b0 = (acc & 0xFF).astype(np.uint8)
    b1 = ((acc >> 8) & 0xFF).astype(np.uint8)
    b2 = ((acc >> 16) & 0xFF).astype(np.uint8)
    out = np.empty(elems.shape[:-1] + (48,), dtype=np.uint8)
    out[..., 0::3] = b0
    out[..., 1::3] = b1
    out[..., 2::3] = b2
    return out


def _bit_unpack_chunk(bytes_: np.ndarray, bits: int, n_elem: int) -> np.ndarray:
    """Inverse of _bit_pack_chunk: (..., 16*bits) bytes -> (..., n_elem)
    uint8 values (n_elem = 128 for full chunks; tails are caller-sliced)."""
    if bits == 4:
        b = bytes_.astype(np.uint8)
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        out = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), dtype=np.uint8)
        out[..., 0::2] = lo
        out[..., 1::2] = hi
        return out[..., :n_elem]
    if bits == 2:
        # each byte holds 4 values: value m is byte m//4, bits 2*(m%4)
        b = bytes_.astype(np.uint8)
        v0 = b & 0x3
        v1 = (b >> 2) & 0x3
        v2 = (b >> 4) & 0x3
        v3 = (b >> 6) & 0x3
        out = np.empty(b.shape[:-1] + (b.shape[-1] * 4,), dtype=np.uint8)
        out[..., 0::4] = v0
        out[..., 1::4] = v1
        out[..., 2::4] = v2
        out[..., 3::4] = v3
        return out[..., :n_elem]
    if bits == 1:
        b = bytes_.astype(np.uint8)
        out = np.empty(b.shape[:-1] + (b.shape[-1] * 8,), dtype=np.uint8)
        for j in range(8):
            out[..., j::8] = (b >> j) & 0x1
        return out[..., :n_elem]
    # bits == 3: 3 bytes -> 8 values (24-bit little-endian group)
    b = bytes_.astype(np.uint32).reshape(*bytes_.shape[:-1],
                                         bytes_.shape[-1] // 3, 3)
    acc = (b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16))
    out = np.empty(acc.shape[:-1] + (acc.shape[-1] * 8,), dtype=np.uint8)
    for j in range(8):
        out[..., j::8] = (acc >> (3 * j)) & 0x7
    return out[..., :n_elem]


def pack_idxn(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack logical b-bit indices [N, K] (values < 2**b) into the idxN blob.

    At bits=4 the output is byte-identical to the legacy idx4 format.
    pack_idx4 (same permutation, same nibble order) — asserted by
    self_test() and the test suite.
    """
    b = check_bits(bits)
    idx = np.ascontiguousarray(indices, dtype=np.uint8)
    if idx.ndim != 2:
        raise ValueError(f"expected a 2-D index matrix, got shape {idx.shape}")
    N, K = idx.shape
    check_eligible(N, K)
    max_val = np.iinfo(np.uint8).max if idx.size else 0
    del max_val
    if idx.size and int(idx.max(initial=0)) >= (1 << b):
        raise ValueError(f"idx{b} indices must be < {1 << b}, got max "
            f"{int(idx.max())}")
    row_map, col_map = _chunk_element_map(b)

    T = N // ROWS_PER_TILE
    G = K // K_PER_GROUP
    # [T, G, 64 threads, 128 elements] gather
    rows = row_map[None, None, :, :] + (np.arange(T) * ROWS_PER_TILE)[:, None, None, None]
    cols = col_map[None, None, :, :] + (np.arange(G) * K_PER_GROUP)[None, :, None, None]
    elems = idx[rows, cols]                       # (T, G, 64, 128)
    chunks = _bit_pack_chunk(elems, b)            # (T, G, 64, 16*b)
    # chunk (th) sits at byte offset th*16*b within the tile
    blob = chunks.reshape(T, G, TILE_BYTES_PER_BIT * b)
    return np.ascontiguousarray(blob.reshape(-1))


def unpack_idxn(blob: np.ndarray, N: int, K: int, bits: int) -> np.ndarray:
    """Invert the idxN permutation: blob -> logical indices [N, K].

    Used to verify on-disk artifacts against the reference dequantization
    without going through the kernel.
    """
    b = check_bits(bits)
    check_eligible(N, K)
    blob = np.ascontiguousarray(blob, dtype=np.uint8).reshape(-1)
    expected = blob_bytes(N, K, b)
    if blob.size != expected:
        raise ValueError(f"idx{b} blob has {blob.size} bytes, expected N*K*{b}/8 = "
            f"{expected} (N={N}, K={K})")
    row_map, col_map = _chunk_element_map(b)

    T = N // ROWS_PER_TILE
    G = K // K_PER_GROUP
    tiles = blob.reshape(T, G, TILE_THREADS, 16 * b)
    elems = _bit_unpack_chunk(tiles, b, 128)      # (T, G, 64, 128)

    idx = np.empty((N, K), dtype=np.uint8)
    rows = row_map[None, None, :, :] + (np.arange(T) * ROWS_PER_TILE)[:, None, None, None]
    cols = col_map[None, None, :, :] + (np.arange(G) * K_PER_GROUP)[None, :, None, None]
    idx[rows, cols] = elems
    return idx


def pack_idxn_from_packed(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack a logical packed row stream [N, K*b/8] (LSB-first, the layout
    DEQUANT_SPEC section 2 generalizes) into the idxN blob.

    Kept for parity with idx4.pack_idx4_from_packed: the logical stream is
    unpacked to indices and re-packed through the canonical path.
    """
    b = check_bits(bits)
    q = np.ascontiguousarray(q, dtype=np.uint8)
    if q.ndim != 2:
        raise ValueError(f"expected a 2-D packed matrix, got shape {q.shape}")
    N = q.shape[0]
    K = (q.shape[1] * 8) // b
    idx = _bit_unpack_chunk(q, b, K)
    return pack_idxn(idx.reshape(N, K), b)


def self_test(shapes=((128, 64), (256, 128), (384, 192), (1024, 512)),
    bits_all=BITS_SUPPORTED,
) -> bool:
    """CPU round-trip check: pack -> unpack must reproduce the input
    exactly, at every width; b=4 must equal idx4.pack_idx4 byte-for-byte."""
    rng = np.random.default_rng(0)
    for b in bits_all:
        for N, K in shapes:
            idx = rng.integers(0, 1 << b, size=(N, K), dtype=np.uint8)
            blob = pack_idxn(idx, b)
            assert blob.size == blob_bytes(N, K, b)
            back = unpack_idxn(blob, N, K, b)
            assert np.array_equal(back, idx), (b, N, K)
            # corrupt-width blob must be refused loudly
            try:
                unpack_idxn(blob, N, K, 4 if b != 4 else 2)
                raise AssertionError("width confusion not caught")
            except ValueError:
                pass
    # b=4: byte-identical to the canonical idx4 producer
    try:
        import importlib.util as _ilu
        import os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _p = _os.path.join(_here, "idx4.py")
        _spec = _ilu.spec_from_file_location("flute_idx4_for_idxn_selftest", _p)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
    except FileNotFoundError:
        # idx4.py not co-located (packaged layout) — the equivalence is
        # still enforced by the repo test suite; skip here.
        _m = None
    if _m is not None:
        for N, K in shapes:
            idx = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
            assert np.array_equal(pack_idxn(idx, 4), _m.pack_idx4(idx)), (N, K)
            assert np.array_equal(unpack_idxn(pack_idxn(idx, 4), N, K, 4), idx), (N, K)
    return True


if __name__ == "__main__":
    ok = self_test()
    print("idxN self_test:", "PASS" if ok else "FAIL")
