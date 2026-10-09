"""query.py — the eight-step query flow (SPECIFICATION §6).

    [1] tokenize (the CALLER provides token ids — tokenizers are not a
        dependency of this module)
    [2] prefill the query through the online-TQ cache, reseeded from the
        system reset point → the query's S/M1/M2 captured as codes → the
        dequantized §4 query vector
    [3] IVFADC preselect on cache vectors → top-100 candidates
    [4] cos-sim rerank → top-3 chunk indices
    [5] load the top-3 chunks' TurboQuant codes from disk (~18 MiB)
    [6] install: sum the S/M1/M2 deltas on top of the system codes
        (dequant-sum-requant once — rag/install.py); conv = last chunk
    [7] answer: prefill the query AGAIN over the installed cache (spec §6's
        answer step — the delta rule continues from the installed state);
        the full-attn layers run fresh (spec §2.4)
    [8] decode: greedy loop via the model's forward (use_cache=True)

Oracle mode: `retrieved_ids` passed explicitly skips steps 3–4 (the P6
e2e oracle-install variant — isolates retrieval quality from generation
quality, PROPOSAL P6); step 5 (loading the chosen chunks' codes) still
runs — the install needs the codes.

Timing: every step is instrumented (the §9 ledger); QueryResult.timings
carries seconds per step.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import snapshot as snap_mod
from ingest import SystemState, load_system_state, reseed_cache
from install import install_snapshot
from tq_cache import TQCache, resolve_quantizer

__all__ = ["QueryResult", "answer_query", "query_cache_vector"]


@dataclass
class QueryResult:
    retrieved_ids: List[int] = field(default_factory=list)
    cos_scores: List[float] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    new_token_ids: List[int] = field(default_factory=list)
    oracle: bool = False
    install_report: Optional[dict] = None


def query_cache_vector(cache: TQCache, system: SystemState) -> np.ndarray:
    """The §4 query vector: dequantized ABSOLUTE-from-system codes of the
    query prefill, concatenated in §4 order (S ascending, M1, M2), fp32."""
    codes = cache.snapshot_codes()
    pieces = []
    for L in sorted(system.s_codes):
        c = codes["s"].get(L)
        if c is None:
            raise ValueError(
                f"query_cache_vector: layer {L} has no S codes — the query "
                f"prefill did not run the full stack")
        pieces.append(resolve_quantizer("S", c.d, system.bits).dequant(c))
    for kind in ("m1", "m2"):
        c = codes[kind]
        if c is not None:
            pieces.append(resolve_quantizer(
                kind.upper(), c.d, system.bits).dequant(c))
    return torch.cat(pieces).to(torch.float32).numpy()


def answer_query(
    model,
    query_token_ids: torch.Tensor,
    system: SystemState,
    index=None,
    loader=None,
    cache: Optional[TQCache] = None,
    cache_factory: Optional[Callable[[], TQCache]] = None,
    preselect_k: int = 100,
    rerank_k: int = 3,
    retrieved_ids: Optional[Sequence[int]] = None,   # oracle mode
    max_new_tokens: int = 0,
    decode_fn: Optional[Callable] = None,
) -> QueryResult:
    """Run the §6 flow. `cache` (or cache_factory) must produce a cache the
    MODEL accepts (production: TQCache(config=model.config); tests:
    TQCache(layer_types=[...])). `decode_fn(model, cache, last_logits,
    max_new_tokens)` implements step 8 for exotic heads — default: a plain
    greedy loop over model(input_ids=next_id, past_key_values=cache)."""
    res = QueryResult()
    res.oracle = retrieved_ids is not None
    if cache is None:
        if cache_factory is None:
            raise ValueError(
                "answer_query: pass cache= or cache_factory= (production: "
                "TQCache(config=model.config); tests: layer_types)")
        cache = cache_factory()

    t = {}

    # [2] query prefill from the system reset point → the query vector
    t0 = time.perf_counter()
    reseed_cache(cache, system)
    with torch.no_grad():
        model(input_ids=query_token_ids, past_key_values=cache, use_cache=True)
    qvec = query_cache_vector(cache, system)
    t["prefill_snapshot"] = time.perf_counter() - t0

    # [3–4] retrieval (skipped in oracle mode)
    if retrieved_ids is None:
        import index as index_mod
        t0 = time.perf_counter()
        candidates = index_mod.preselect(index, qvec, k=preselect_k)
        ids, scores = index_mod.rerank(loader, qvec, candidates, k=rerank_k)
        t["preselect_rerank"] = time.perf_counter() - t0
        res.retrieved_ids = [int(i) for i in ids]
        res.cos_scores = [float(s) for s in scores]
        retrieved_ids = res.retrieved_ids
    else:
        res.retrieved_ids = [int(i) for i in retrieved_ids]

    # [5] load the retrieved chunks' codes from disk
    t0 = time.perf_counter()
    disk_dir = getattr(loader, "disk_dir", None)
    if disk_dir is None:
        raise ValueError(
            "answer_query: loader must expose .disk_dir (ChunkVectorLoader)")
    snaps = []
    for cid in retrieved_ids:
        path = os.path.join(disk_dir, "snapshots",
                            f"chunk_{int(cid):05d}.npz")
        if not os.path.exists(path):
            raise ValueError(f"answer_query: {path} not found")
        snaps.append(snap_mod.load_chunk(path))
    t["load_codes"] = time.perf_counter() - t0

    # [6] install (dequant-sum-requant; conv = last chunk)
    t0 = time.perf_counter()
    res.install_report = install_snapshot(cache, system, snaps)
    t["install"] = time.perf_counter() - t0

    # [7] answer: prefill the query over the installed cache
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=query_token_ids, past_key_values=cache,
                    use_cache=True)
    logits = out[0] if isinstance(out, tuple) else out
    t["answer_prefill"] = time.perf_counter() - t0

    # [8] decode
    if max_new_tokens > 0:
        t0 = time.perf_counter()
        if decode_fn is not None:
            res.new_token_ids = list(decode_fn(
                model, cache, logits, max_new_tokens))
        else:
            res.new_token_ids = _greedy_decode(
                model, cache, logits, max_new_tokens)
        t["decode"] = time.perf_counter() - t0

    res.timings = t
    return res


def _greedy_decode(model, cache, logits, max_new_tokens: int) -> List[int]:
    """Plain greedy loop (step 8). `logits` is the answer-prefill output of
    the stub/model; the loop feeds each argmax back as the next input."""
    out_ids: List[int] = []
    cur_logits = logits
    next_id = torch.tensor([[int(cur_logits[:, -1, :].argmax(dim=-1))]])
    out_ids.append(int(next_id))
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            r = model(input_ids=next_id, past_key_values=cache,
                      use_cache=True)
        cur_logits = r[0] if isinstance(r, tuple) else r
        if cur_logits is None or cur_logits.dim() < 3:
            break
        next_id = torch.tensor(
            [[int(cur_logits[:, -1, :].argmax(dim=-1))]])
        out_ids.append(int(next_id))
    return out_ids
