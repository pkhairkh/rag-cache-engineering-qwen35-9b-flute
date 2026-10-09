#!/usr/bin/env python3
"""
qlora_fallback.py — reference-path weight materialization for the QLoRA
frozen branch.

The fallback computes dL/dX = dL/dY @ W with W materialized once per module
(the weights are frozen during QLoRA, so per-call re-materialization is pure
waste). Two components:

* dequant_idx4_torch — pure-torch inverse of the idx4 blob layout
  (flute_extended.idx4 is the normative producer; the position algebra here
  mirrors its _tile_permutation). Device-agnostic: CPU in tests, GPU in
  production.
* WeightCache — LRU cache keyed on (blob identity, lut identity, dtype),
  byte-capped. A full bf16 cache of all 248 modules would be 13.8 GB and
  must never be attempted; the default cap is 6 GiB with eviction, and a
  single entry larger than the cap is served transiently (never cached).

Cache keys use id(): entries hold references to the key tensors, so the
ids stay valid as long as the entries exist (a freed tensor's id could
otherwise be reused by a new allocation).
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

# Layout constants (flute_extended.idx4 / DEQUANT_SPEC section 7).
_ROWS_PER_TILE = 128
_K_PER_TILE = 64
_TILE_BYTES = 4096

_DEFAULT_CAP_GIB = 6.0
_POSITIONS_PER_CHUNK = 1 << 20


def dequant_idx4_torch(
    indices_blob: torch.Tensor,
    lut: torch.Tensor,
    N: int,
    K: int,
    group_size: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dequantize an idx4 blob to the logical weight W[N, K].

    W[n, k] = lut[n // group_size, idx(n, k)] with LSB-first nibble pairs,
    executed as vectorized scatter/gather over flat byte positions. Works on
    any device; temporaries are bounded by _POSITIONS_PER_CHUNK.
    """
    if N % _ROWS_PER_TILE != 0 or K % _K_PER_TILE != 0:
        raise ValueError(f"idx4 requires N % 128 == 0 and K % 64 == 0, got N={N}, K={K}")
    expected = N * (K // 2)
    blob = indices_blob.reshape(-1)
    if blob.numel() != expected:
        raise ValueError(f"idx4 blob has {blob.numel()} bytes, expected {expected}")
    if blob.dtype != torch.uint8:
        raise ValueError(f"idx4 blob must be uint8, got {blob.dtype}")
    if lut.dtype != torch.float16:
        raise ValueError(f"lut must be float16 (artifact contract), got {lut.dtype}")

    out_dtype = dtype or torch.float16
    device = blob.device
    W = torch.empty(N, K, dtype=torch.float16, device=device)
    Wf = W.view(-1)
    lut_flat = lut.reshape(-1)
    tiles_k = K // _K_PER_TILE
    total = expected

    for start in range(0, total, _POSITIONS_PER_CHUNK):
        end = min(start + _POSITIONS_PER_CHUNK, total)
        p = torch.arange(start, end, dtype=torch.int64, device=device)
        # Flat position -> (n, k) per the normative tile/chunk encoding.
        tile = p >> 12
        t = torch.div(tile, tiles_k, rounding_mode="floor")
        g = tile - t * tiles_k
        within = p & (_TILE_BYTES - 1)
        wx = within >> 11
        r2 = within & 2047
        chunk = r2 >> 6
        r3 = r2 & 63
        seg = r3 >> 4
        v = (r3 >> 2) & 3
        d = (r3 >> 1) & 1
        s2 = r3 & 1
        n = (t << 7) + (wx << 6) + (v << 4) + (d << 3) + (chunk >> 2)
        k = (g << 6) + (seg << 4) + ((chunk & 3) << 1) + (s2 << 3)
        grp = torch.div(n, group_size, rounding_mode="floor")
        base = grp * 16
        b = blob[start:end].long()
        lo = b & 15
        hi = b >> 4
        nK = n * K
        Wf[nK + k] = lut_flat[base + lo]
        Wf[nK + k + 1] = lut_flat[base + hi]

    return W.to(out_dtype) if out_dtype != torch.float16 else W


# The idxN pair walk (DEQUANT_SPEC section 8): flat PAIR positions, the
# width-independent (n, k) decode, and the 2*b-bit field extraction from
# the two-byte little-endian window. At b=4 the walk degenerates to the
# byte walk above (byte0 = the pair's byte, shift 0) — asserted by the
# test suite as a bit-exact identity.
def _idxn_pair_positions(p, tiles_k, K, b):
    """Flat pair index tensor -> (n, k, byte0, shift) tensors."""
    tile_pair = p >> 12                        # 4096 pairs per tile
    t = torch.div(tile_pair, tiles_k, rounding_mode="floor")
    g = tile_pair - t * tiles_k
    within = p & 4095
    wx = within >> 11                          # 2048 pairs per half-tile
    r2 = within & 2047
    lane = r2 >> 6                             # 64 pairs per (wx, lane) chunk
    r3 = r2 & 63                               # pair index within the chunk
    seg = r3 >> 4
    j = r3 & 15
    v = j >> 2
    d = (j >> 1) & 1
    s2 = j & 1
    n = (t << 7) + (wx << 6) + (v << 4) + (d << 3) + (lane >> 2)
    k = (g << 6) + (seg << 4) + ((lane & 3) << 1) + (s2 << 3)
    chunk_bytes = 16 * b
    half_bytes = 512 * b
    tile_bytes = 1024 * b
    chunk_base = ((t * tiles_k + g) * tile_bytes
                  + wx * half_bytes + lane * chunk_bytes)
    bit = (chunk_base * 8) + 2 * b * r3        # the pair field's LSB
    byte0 = bit >> 3
    shift = bit & 7
    return n, k, byte0, shift


def dequant_idxn_torch(
    indices_blob: torch.Tensor,
    lut: torch.Tensor,
    N: int,
    K: int,
    group_size: int,
    bitwidth: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dequantize an idxN blob (bitwidth 1/2/3/4) to W[N, K] in torch.

    W[n, k] = lut[n // group_size, idx_b(n, k)] over the unified idxN
    layout: flat-PAIR position walk (the kernel's segment arithmetic,
    worked backwards — the same lineage discipline as dequant_idx4_torch,
    not the packer's map). Works on any device; temporaries are bounded
    by _POSITIONS_PER_CHUNK pairs per chunk. At bitwidth=4 the result is
    bit-exact against dequant_idx4_torch (the test-suite identity gate).
    """
    if bitwidth not in (1, 2, 3, 4):
        raise ValueError(f"idxN bitwidth must be 1-4, got {bitwidth}")
    if bitwidth == 4:
        return dequant_idx4_torch(indices_blob, lut, N, K, group_size, dtype)
    if N % _ROWS_PER_TILE != 0 or K % _K_PER_TILE != 0:
        raise ValueError(
            f"idxN requires N % 128 == 0 and K % 64 == 0, got N={N}, K={K}")
    expected = (N * K * bitwidth) // 8
    blob = indices_blob.reshape(-1)
    if blob.numel() != expected:
        raise ValueError(
            f"idx{bitwidth} blob has {blob.numel()} bytes, expected "
            f"{expected} (N={N}, K={K})")
    if blob.dtype != torch.uint8:
        raise ValueError(f"idxN blob must be uint8, got {blob.dtype}")
    if lut.dtype != torch.float16:
        raise ValueError(
            f"lut must be float16 (artifact contract), got {lut.dtype}")

    b = int(bitwidth)
    codes = 1 << b
    field_mask = (1 << (2 * b)) - 1
    out_dtype = dtype or torch.float16
    device = blob.device
    W = torch.empty(N, K, dtype=torch.float16, device=device)
    Wf = W.view(-1)
    lut_flat = lut.reshape(-1)
    if lut_flat.numel() != ((N + group_size - 1) // group_size) * codes:
        raise ValueError(
            f"idx{b} lut must hold ceil(N/{group_size})*{codes} fp16 "
            f"entries, got {lut_flat.numel()}")
    tiles_k = K // _K_PER_TILE
    total_pairs = (N * K) // 2

    for start in range(0, total_pairs, _POSITIONS_PER_CHUNK):
        end = min(start + _POSITIONS_PER_CHUNK, total_pairs)
        p = torch.arange(start, end, dtype=torch.int64, device=device)
        n, k, byte0, shift = _idxn_pair_positions(p, tiles_k, K, b)
        # Two-byte little-endian window. The 2*b-bit field (b <= 4 -> at
        # most 8 bits) needs byte0+1 only when shift + 2*b > 8, and a
        # spanning field ends strictly inside the blob (its last bit is
        # <= total_bits - 1), so byte0+1 is in range exactly then; the
        # non-spanning branch reads zero instead (never past the end —
        # the LAST pair's field ends at the blob's bit end).
        span = (shift + 2 * b) > 8
        safe1 = (byte0 + 1).clamp(max=blob.numel() - 1)
        w0 = blob[byte0].long()
        w1 = torch.where(span, blob[safe1].long(),
                         torch.zeros((), dtype=torch.int64, device=device))
        field = ((w0 | (w1 << 8)) >> shift) & field_mask
        v0 = field & (codes - 1)
        v1 = field >> b
        grp = torch.div(n, group_size, rounding_mode="floor")
        base = grp * codes
        nK = n * K
        Wf[nK + k] = lut_flat[base + v0]
        Wf[nK + k + 1] = lut_flat[base + v1]

    return W.to(out_dtype) if out_dtype != torch.float16 else W


class WeightCache:
    """Byte-capped LRU cache of materialized weights.

    Key: (id(indices), id(lut), dtype). Entries pin their key tensors so the
    ids cannot be recycled while the entry lives. get() returns the cached
    tensor or None; put() inserts with eviction, or refuses (returns False)
    when a single entry would exceed the cap.
    """

    def __init__(self, cap_bytes: int):
        if cap_bytes <= 0:
            raise ValueError("cap_bytes must be positive")
        self.cap_bytes = int(cap_bytes)
        self._entries: "OrderedDict[Tuple[int, int, torch.dtype], Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.refusals = 0

    def _key(self, indices: torch.Tensor, lut: torch.Tensor,
             dtype: torch.dtype) -> Tuple[int, int, torch.dtype]:
        return (id(indices), id(lut), dtype)

    def get(self, indices: torch.Tensor, lut: torch.Tensor,
            dtype: torch.dtype) -> Optional[torch.Tensor]:
        key = self._key(indices, lut, dtype)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, indices: torch.Tensor, lut: torch.Tensor,
            W: torch.Tensor) -> bool:
        nbytes = W.numel() * W.element_size()
        if nbytes > self.cap_bytes:
            self.refusals += 1
            return False
        while self.cached_bytes() + nbytes > self.cap_bytes:
            self._entries.popitem(last=True)
            self.evictions += 1
        key = self._key(indices, lut, W.dtype)
        self._entries[key] = (W, nbytes, indices, lut)
        return True

    def cached_bytes(self) -> int:
        return sum(size for _, size, _, _ in self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> Dict[str, float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "refusals": self.refusals,
            "cached_gib": self.cached_bytes() / (1024 ** 3),
        }

    def __len__(self) -> int:
        return len(self._entries)


_LOCK = threading.Lock()
_CACHES = []


def default_cache() -> WeightCache:
    """The process-wide cache, sized by FLUTE_WCACHE_GIB (default 6 GiB)."""
    with _LOCK:
        if not _CACHES:
            gib = float(os.environ.get("FLUTE_WCACHE_GIB", _DEFAULT_CAP_GIB))
            _CACHES.append(WeightCache(int(gib * 1024 ** 3)))
        return _CACHES[0]


def clear_all() -> None:
    """Phase-scoped clear: drop every cache (between train/eval/merge)."""
    with _LOCK:
        for cache in _CACHES:
            cache.clear()


def stats() -> Dict[str, float]:
    return default_cache().stats()


def materialize_weight(
    indices: torch.Tensor,
    lut: torch.Tensor,
    N: int,
    K: int,
    group_size: int,
    dtype: Optional[torch.dtype] = None,
    cache: Optional[WeightCache] = None,
    bitwidth: int = 4,
) -> torch.Tensor:
    """Materialize W (with LUT lookup only — the residual is handled by the
    autograd path in qlora.py) for the reference backward.

    bitwidth 1/2/3 dequantize through the idxN pair walk
    (dequant_idxn_torch); 4 is the frozen nibble path. Cached when
    possible; on cache refusal (single entry over cap) or a CUDA
    out-of-memory during materialization, falls back to a transient
    tensor.
    """
    use_cache = cache if cache is not None else default_cache()
    W = use_cache.get(indices, lut, dtype or torch.float16)
    if W is not None:
        return W
    try:
        W = dequant_idxn_torch(indices, lut, N, K, group_size, bitwidth,
                               dtype)
        use_cache.put(indices, lut, W)
    except torch.cuda.OutOfMemoryError:
        # Evict and retry once; if it still fails, serve transiently.
        use_cache.clear()
        W = dequant_idxn_torch(indices, lut, N, K, group_size, bitwidth,
                               dtype)
    return W


__all__ = ["dequant_idx4_torch", "dequant_idxn_torch", "WeightCache",
           "default_cache", "clear_all", "stats", "materialize_weight"]
