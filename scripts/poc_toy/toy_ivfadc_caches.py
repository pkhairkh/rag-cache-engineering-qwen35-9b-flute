#!/usr/bin/env python3
"""toy_ivfadc_caches.py — test the IVFADC + two-global-cache architecture.

The architecture:
- IVFADC on disk: preselects top-100 candidates by approximate distance
  on the chunk's pooled hidden state (the LUT model's own representation)
- Cos sim rerank: top-100 → top-3 exact
- Two global caches (cache A: KV, cache B: state): restore the top-3
  chunks' states into the model. Two entry types:
  - Independent entry (keyed by chunk hash): fast, ~4.1e-3 restoration error
  - Prefix-aware entry (keyed by prefix + chunk hash): lossless, requires
    the exact prefix to be cached

What we test:
1. Does IVFADC preselect + cos sim rerank retrieve relevant chunks?
   (retrieval quality — the vectorDB step)
2. Does the cache restoration (independent entry) preserve accuracy?
   (the restoration error — the cost of caching)
3. Does the prefix-aware entry produce lossless results?
   (the fallback path)
4. Is the full pipeline (IVFADC + caches) faster than no-cache?
   (the latency win)
"""
import argparse, hashlib, json, math, os, random, sys, time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the model from the previous toy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_cache_engineered_rag import (
    TinyHybridModel, TinyGatedDeltaNet, TinyFullAttention,
    gen_corpus, gen_query, hash_tokens,
)


# ---------------------------------------------------------------------------
# The IVFADC index (simplified — coarse quantizer + PQ, on "disk")
# ---------------------------------------------------------------------------

class SimpleIVFADC:
    """A simplified IVFADC index.
    - Coarse quantizer: k-means with nlist clusters (on the exact vectors)
    - PQ: split the vector into m sub-vectors, quantize each to 256 centroids
    - Search: probe nprobe nearest clusters, compute PQ-approx distances

    In the real system this is FAISS IndexIVFPQ. Here it's a toy that
    tests the architecture, not the IVFADC quality.
    """
    def __init__(self, nlist=16, m=8, nprobe=4):
        self.nlist = nlist
        self.m = m
        self.nprobe = nprobe
        self.vectors = None  # the exact vectors (for rerank)
        self.coarse_centroids = None
        self.pq_centroids = None  # (m, 256, sub_dim)
        self.pq_codes = None  # (N, m) uint8
        self.cluster_assignments = None  # (N,) — which cluster each vector belongs to
        self.chunk_ids = None

    def build(self, vectors, chunk_ids):
        """Build the index. vectors: (N, D) float32."""
        N, D = vectors.shape
        self.vectors = vectors
        self.chunk_ids = np.array(chunk_ids)

        # 1. coarse quantizer: k-means (simplified — random init, 5 iterations)
        rng = np.random.RandomState(0)
        idx = rng.choice(N, self.nlist, replace=False)
        self.coarse_centroids = vectors[idx].copy()
        for _ in range(5):
            # assign each vector to nearest centroid
            dists = ((vectors[:, None, :] - self.coarse_centroids[None, :, :])**2).sum(-1)
            self.cluster_assignments = dists.argmin(-1)
            # update centroids
            for c in range(self.nlist):
                mask = self.cluster_assignments == c
                if mask.any():
                    self.coarse_centroids[c] = vectors[mask].mean(0)

        # 2. PQ: split D into m sub-vectors of D/m dims each
        sub_dim = D // self.m
        n_pq = min(256, N)  # PQ centroids capped at the number of vectors
        self.pq_centroids = np.zeros((self.m, n_pq, sub_dim), dtype=np.float32)
        self.pq_codes = np.zeros((N, self.m), dtype=np.uint8)
        for j in range(self.m):
            sub_vectors = vectors[:, j*sub_dim:(j+1)*sub_dim]
            # k-means with n_pq centroids (simplified)
            rng_j = np.random.RandomState(j)
            idx = rng_j.choice(len(sub_vectors), n_pq, replace=False)
            cents = sub_vectors[idx].copy()
            for _ in range(3):
                dists = ((sub_vectors[:, None, :] - cents[None, :, :])**2).sum(-1)
                assigns = dists.argmin(-1)
                for c in range(len(cents)):
                    mask = assigns == c
                    if mask.any():
                        cents[c] = sub_vectors[mask].mean(0)
            self.pq_centroids[j] = cents
            # encode
            dists = ((sub_vectors[:, None, :] - cents[None, :, :])**2).sum(-1)
            self.pq_codes[:, j] = dists.argmin(-1).astype(np.uint8)

    def search(self, query_vec, k=100):
        """Returns top-k chunk IDs by PQ-approximate distance."""
        # 1. find nprobe nearest coarse clusters
        coarse_dists = ((self.coarse_centroids - query_vec)**2).sum(-1)
        probe_clusters = np.argsort(coarse_dists)[:self.nprobe]

        # 2. gather candidates from probed clusters
        candidates = np.where(np.isin(self.cluster_assignments, probe_clusters))[0]

        # 3. PQ-approximate distance for each candidate
        sub_dim = len(query_vec) // self.m
        approx_dists = np.zeros(len(candidates), dtype=np.float32)
        for j in range(self.m):
            q_sub = query_vec[j*sub_dim:(j+1)*sub_dim]
            cents = self.pq_centroids[j]
            # for each candidate, look up its PQ code for sub-vector j
            codes = self.pq_codes[candidates, j]
            approx_sub = cents[codes]
            q_sub_rep = np.broadcast_to(q_sub, approx_sub.shape)
            approx_dists += ((approx_sub - q_sub_rep)**2).sum(-1)

        # 4. top-k by approx distance
        top_k_local = np.argsort(approx_dists)[:k]
        return candidates[top_k_local]

    def rerank(self, candidate_idxs, query_vec, k=3):
        """Cos sim rerank on the exact vectors. Returns top-k chunk IDs."""
        candidate_vectors = self.vectors[candidate_idxs]
        # cos sim (vectors are mean-pooled hidden states, not normalized — normalize first)
        cand_norm = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-8)
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        scores = cand_norm @ q_norm
        top_k_local = np.argsort(scores)[-k:][::-1]
        return candidate_idxs[top_k_local]


# ---------------------------------------------------------------------------
# The two global caches (with independent + prefix-aware entries)
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    linear_state: torch.Tensor
    conv_state: torch.Tensor
    kv: Tuple[torch.Tensor, torch.Tensor]
    size: int


class GlobalCache:
    """The combined global cache. Two entry types:
    - Independent (key = hash(chunk_tokens)): snapshotted from isolated prefill
    - Prefix-aware (key = hash(prefix_tokens + chunk_tokens)): snapshotted in context"""
    def __init__(self, max_entries=256):
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max = max_entries
        self.hits = 0
        self.misses = 0
        self.prefix_hits = 0
        self.indep_hits = 0

    def lookup(self, key):
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        return None

    def install(self, key, linear_state, conv_state, kv):
        size = (linear_state.nelement() * linear_state.element_size() +
                conv_state.nelement() * conv_state.element_size() +
                kv[0].nelement() * kv[0].element_size() * 2)
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        while len(self.entries) >= self.max and self.entries:
            self.entries.popitem(last=False)
        self.entries[key] = CacheEntry(linear_state.clone(), conv_state.clone(),
                                       (kv[0].clone(), kv[1].clone()), size)
        return True


# ---------------------------------------------------------------------------
# The full pipeline: embed → IVFADC + rerank → cache restore
# ---------------------------------------------------------------------------

def embed_corpus(model, chunks, cache, device):
    """Embed all chunks: independent prefill, snapshot states, build IVFADC index."""
    vectors = []
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        with torch.no_grad():
            logits, lin_state, conv_state, kv, _ = model(chunk, return_states=True)
        # the IVFADC vector: mean-pool the hidden state (use logits as proxy for hidden)
        v = logits.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
        vectors.append(v)
        chunk_ids.append(i)
        # install the INDEPENDENT cache entry
        chunk_hash = hash_tokens(chunk)
        cache.install(chunk_hash, lin_state, conv_state, kv)
    vectors = np.stack(vectors)
    # build the IVFADC index
    index = SimpleIVFADC(nlist=min(16, len(chunks)), m=8, nprobe=4)
    index.build(vectors, chunk_ids)
    return index


def retrieve(model, query, index, device, top_k=3):
    """IVFADC preselect + cos sim rerank → top-k chunk IDs."""
    with torch.no_grad():
        logits, _, _, _, _ = model(query, return_states=True)
    v_q = logits.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    # IVFADC preselect
    candidates = index.search(v_q, k=100)
    # cos sim rerank
    top_k_idxs = index.rerank(candidates, v_q, k=top_k)
    return top_k_idxs.tolist()


def augment_and_query(model, system_prompt, query, retrieved_chunks, cache, device,
                      use_prefix_aware=True):
    """Restore system prompt + retrieved chunks' states, then prefill the query.
    Returns (logits, restoration_path)."""
    # restore system prompt
    sys_hash = hash_tokens(system_prompt)
    sys_entry = cache.lookup(sys_hash)
    if sys_entry is None:
        with torch.no_grad():
            _, lin, conv, kv, _ = model(system_prompt, return_states=True)
        cache.install(sys_hash, lin, conv, kv)
        sys_entry = cache.entries[sys_hash]
    kv_cache = (sys_entry.kv[0].clone(), sys_entry.kv[1].clone())
    linear_state = sys_entry.linear_state.clone()
    conv_state = sys_entry.conv_state.clone()

    restoration_path = "system_only"
    # restore retrieved chunks in document order
    for chunk in retrieved_chunks:
        if use_prefix_aware:
            # try prefix-aware entry first (lossless)
            # the prefix is system_prompt + this chunk (what we cached during setup)
            prefix_tokens = torch.cat([system_prompt, chunk], dim=1)
            prefix_hash = hash_tokens(prefix_tokens)
            prefix_entry = cache.lookup(prefix_hash)
            if prefix_entry is not None:
                kv_cache = (torch.cat([kv_cache[0], prefix_entry.kv[0]], dim=2),
                            torch.cat([kv_cache[1], prefix_entry.kv[1]], dim=2))
                linear_state = prefix_entry.linear_state.clone()
                conv_state = prefix_entry.conv_state.clone()
                restoration_path = "prefix_aware"
                cache.prefix_hits += 1
                continue
        # try independent entry (fast, ~4.1e-3 error)
        chunk_hash = hash_tokens(chunk)
        indep_entry = cache.lookup(chunk_hash)
        if indep_entry is not None:
            kv_cache = (torch.cat([kv_cache[0], indep_entry.kv[0]], dim=2),
                        torch.cat([kv_cache[1], indep_entry.kv[1]], dim=2))
            linear_state = indep_entry.linear_state.clone()
            conv_state = indep_entry.conv_state.clone()
            restoration_path = "independent"
            cache.indep_hits += 1
        else:
            # cache miss — re-prefill (lossless)
            with torch.no_grad():
                _, linear_state, conv_state, kv, _ = model(chunk,
                    linear_state=linear_state, conv_state=conv_state,
                    kv_cache=kv_cache, return_states=True)
            kv_cache = kv
            restoration_path = "reprefill"

    # prefill the query on top
    with torch.no_grad():
        logits, _, _, _, _ = model(query, linear_state=linear_state,
                                    conv_state=conv_state, kv_cache=kv_cache)
    return logits, restoration_path


def no_cache_baseline(model, system_prompt, query, retrieved_chunks, device):
    """Full re-prefill of (system + chunks + query). The ground truth."""
    all_tokens = system_prompt
    for chunk in retrieved_chunks:
        all_tokens = torch.cat([all_tokens, chunk], dim=1)
    all_tokens = torch.cat([all_tokens, query], dim=1)
    with torch.no_grad():
        logits, _, _, _, _ = model(all_tokens)
    return logits[:, -query.shape[1]:]


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

def test_retrieval_quality(model, device, n_chunks=32, chunk_len=16, query_len=8,
                            system_len=8, n_queries=40, top_k=3):
    """Test 1: Does IVFADC + rerank retrieve relevant chunks?"""
    torch.manual_seed(0)
    vocab = model.vocab
    system_prompt = torch.randint(0, vocab, (1, system_len))
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)

    # embed the corpus
    cache = GlobalCache(max_entries=n_chunks + 1)
    # also embed the system prompt
    with torch.no_grad():
        _, lin, conv, kv, _ = model(system_prompt, return_states=True)
    cache.install(hash_tokens(system_prompt), lin, conv, kv)
    index = embed_corpus(model, chunks, cache, device)

    # the chunks' topics (chunk i has topic i % 4)
    chunk_topics = [i % 4 for i in range(n_chunks)]

    hits_ivfadc = 0
    hits_random = 0
    for q in range(n_queries):
        query_topic = q % 4
        query = gen_query(query_topic, query_len, vocab, seed=q)
        # IVFADC retrieval
        retrieved_idxs = retrieve(model, query, index, device, top_k=top_k)
        retrieved_topics = [chunk_topics[i] for i in retrieved_idxs]
        if query_topic in retrieved_topics:
            hits_ivfadc += 1
        # random baseline
        random_idxs = random.sample(range(n_chunks), top_k)
        random_topics = [chunk_topics[i] for i in random_idxs]
        if query_topic in random_topics:
            hits_random += 1

    return {
        'ivfadc_hit_rate': hits_ivfadc / n_queries,
        'random_hit_rate': hits_random / n_queries,
        'n_queries': n_queries,
        'n_chunks': n_chunks,
    }


def test_restoration_error(model, device, n_chunks=8, chunk_len=16, query_len=8,
                            system_len=8, n_queries=10, top_k=3):
    """Test 2: What's the restoration error for the independent entry (fast path)?
    And does re-prefill-in-context (the lossless fallback) match no-cache?

    NOTE on prefix-aware: the prefix-aware entry must be keyed by the ACCUMULATED
    prefix (system + chunk1 + chunk2 + ... + chunkN), not (system + chunkN).
    This is the prefix-keying problem: for N chunks, there are O(N) possible
    prefixes per chunk (one per position in the retrieval order). Caching all
    of them is O(N²). The independent entry (one per chunk) is O(N) and is
    the realistic fast path. The prefix-aware entry is only cacheable for
    KNOWN hot prefixes (e.g., the system prompt alone, or system + the
    single most-retrieved chunk).

    This test measures:
    - The independent entry's error (the fast path's cost)
    - The re-prefill-in-context path's error (should be 0 — it's lossless)"""
    torch.manual_seed(0)
    vocab = model.vocab
    system_prompt = torch.randint(0, vocab, (1, system_len))
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)

    cache = GlobalCache(max_entries=n_chunks * 4 + 10)
    # embed system prompt
    with torch.no_grad():
        _, lin, conv, kv, _ = model(system_prompt, return_states=True)
    cache.install(hash_tokens(system_prompt), lin, conv, kv)
    # embed chunks (independent entries only — the realistic fast path)
    index = embed_corpus(model, chunks, cache, device)

    # for each query, retrieve top-3, then compare:
    # - no_cache (ground truth)
    # - independent restore (fast path, ~5e-3 error)
    # - re-prefill in context (the lossless fallback)
    max_diff_indep = 0.0
    max_diff_reprefill = 0.0
    for q in range(n_queries):
        query = gen_query(q % 4, query_len, vocab, seed=q)
        retrieved_idxs = retrieve(model, query, index, device, top_k=top_k)
        retrieved_chunks = [chunks[i] for i in retrieved_idxs]

        # ground truth: no cache (full re-prefill as one sequence)
        logits_no = no_cache_baseline(model, system_prompt, query, retrieved_chunks, device)

        # independent restore (fast path)
        logits_indep, _ = augment_and_query(model, system_prompt, query,
                                            retrieved_chunks, cache, device,
                                            use_prefix_aware=False)
        diff_indep = (logits_no - logits_indep).abs().max().item()
        max_diff_indep = max(max_diff_indep, diff_indep)

        # re-prefill in context (the lossless fallback — what augment_and_query
        # does on a cache MISS, but we force it here by clearing the cache entries)
        # This should match no_cache exactly.
        logits_reprefill, _ = augment_and_query_reprefill(model, system_prompt, query,
                                                           retrieved_chunks, cache, device)
        diff_reprefill = (logits_no - logits_reprefill).abs().max().item()
        max_diff_reprefill = max(max_diff_reprefill, diff_reprefill)

    return {
        'max_diff_independent': max_diff_indep,
        'max_diff_reprefill_in_context': max_diff_reprefill,
        'independent_lossless': max_diff_indep < 1e-4,
        'reprefill_lossless': max_diff_reprefill < 1e-4,
    }


def augment_and_query_reprefill(model, system_prompt, query, retrieved_chunks, cache, device):
    """The lossless fallback: re-prefill each chunk in context (on top of the
    accumulated state). This is what happens on a cache miss. Should match no-cache."""
    # restore system prompt
    sys_hash = hash_tokens(system_prompt)
    sys_entry = cache.lookup(sys_hash)
    if sys_entry is None:
        with torch.no_grad():
            _, lin, conv, kv, _ = model(system_prompt, return_states=True)
        cache.install(sys_hash, lin, conv, kv)
        sys_entry = cache.entries[sys_hash]
    kv_cache = (sys_entry.kv[0].clone(), sys_entry.kv[1].clone())
    linear_state = sys_entry.linear_state.clone()
    conv_state = sys_entry.conv_state.clone()

    # re-prefill each chunk in context (the lossless path)
    for chunk in retrieved_chunks:
        with torch.no_grad():
            _, linear_state, conv_state, kv_cache, _ = model(chunk,
                linear_state=linear_state, conv_state=conv_state,
                kv_cache=kv_cache, return_states=True)

    # prefill the query on top
    with torch.no_grad():
        logits, _, _, _, _ = model(query, linear_state=linear_state,
                                    conv_state=conv_state, kv_cache=kv_cache)
    return logits, "reprefill"


def test_latency(model, device, n_chunks=32, chunk_len=16, query_len=8,
                  system_len=8, n_runs=20, top_k=3):
    """Test 3: Is the IVFADC + cache pipeline faster than no-cache?"""
    torch.manual_seed(0)
    vocab = model.vocab
    system_prompt = torch.randint(0, vocab, (1, system_len))
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)

    cache = GlobalCache(max_entries=n_chunks + 1)
    with torch.no_grad():
        _, lin, conv, kv, _ = model(system_prompt, return_states=True)
    cache.install(hash_tokens(system_prompt), lin, conv, kv)
    index = embed_corpus(model, chunks, cache, device)

    # warm up
    for _ in range(3):
        query = gen_query(0, query_len, vocab, seed=0)
        retrieved_idxs = retrieve(model, query, index, device, top_k=top_k)
        retrieved_chunks = [chunks[i] for i in retrieved_idxs]
        no_cache_baseline(model, system_prompt, query, retrieved_chunks, device)
        augment_and_query(model, system_prompt, query, retrieved_chunks, cache, device)

    # time no-cache
    t0 = time.time()
    for r in range(n_runs):
        query = gen_query(r % 4, query_len, vocab, seed=r)
        retrieved_idxs = retrieve(model, query, index, device, top_k=top_k)
        retrieved_chunks = [chunks[i] for i in retrieved_idxs]
        no_cache_baseline(model, system_prompt, query, retrieved_chunks, device)
    t_no = (time.time() - t0) / n_runs * 1000

    # time IVFADC + cache
    t0 = time.time()
    for r in range(n_runs):
        query = gen_query(r % 4, query_len, vocab, seed=r)
        retrieved_idxs = retrieve(model, query, index, device, top_k=top_k)
        retrieved_chunks = [chunks[i] for i in retrieved_idxs]
        augment_and_query(model, system_prompt, query, retrieved_chunks, cache, device)
    t_cache = (time.time() - t0) / n_runs * 1000

    return {
        'no_cache_ms': t_no,
        'ivfadc_cache_ms': t_cache,
        'speedup': t_no / t_cache if t_cache > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ivfadc_cache_results.json'))
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    model = TinyHybridModel(hidden=32, vocab=256).to(device)
    model.eval()

    print("=" * 70)
    print("TEST 1: Retrieval quality — IVFADC + rerank vs random")
    print("=" * 70)
    retrieval = test_retrieval_quality(model, device, n_chunks=32, chunk_len=16,
                                        query_len=8, system_len=8, n_queries=40, top_k=3)
    print(f"  IVFADC + rerank hit rate: {retrieval['ivfadc_hit_rate']*100:.1f}%")
    print(f"  Random baseline hit rate: {retrieval['random_hit_rate']*100:.1f}%")
    print(f"  (n_queries={retrieval['n_queries']}, n_chunks={retrieval['n_chunks']}, top_k=3)")

    print()
    print("=" * 70)
    print("TEST 2: Restoration error — independent (fast) vs re-prefill (lossless)")
    print("=" * 70)
    restoration = test_restoration_error(model, device, n_chunks=8, chunk_len=16,
                                          query_len=8, system_len=8, n_queries=10, top_k=3)
    print(f"  Independent entry max diff:        {restoration['max_diff_independent']:.6e}  (lossless: {restoration['independent_lossless']})")
    print(f"  Re-prefill-in-context max diff:    {restoration['max_diff_reprefill_in_context']:.6e}  (lossless: {restoration['reprefill_lossless']})")
    print(f"  (Independent = fast path, ~5e-3 error — the cost of caching.)")
    print(f"  (Re-prefill = lossless fallback, used on cache miss or high-error query.)")
    print(f"  (Prefix-aware entries are O(N²) — only for known hot prefixes, not general.)")

    print()
    print("=" * 70)
    print("TEST 3: Latency — IVFADC + cache vs no-cache")
    print("=" * 70)
    latency = test_latency(model, device, n_chunks=32, chunk_len=16,
                            query_len=8, system_len=8, n_runs=20, top_k=3)
    print(f"  No-cache (full re-prefill):      {latency['no_cache_ms']:.3f} ms")
    print(f"  IVFADC + cache (fast path):      {latency['ivfadc_cache_ms']:.3f} ms")
    print(f"  Speedup: {latency['speedup']:.2f}×")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    results = {'retrieval': retrieval, 'restoration': restoration, 'latency': latency}
    print(f"  IVFADC retrieves relevant chunks:      {retrieval['ivfadc_hit_rate'] > retrieval['random_hit_rate']}")
    print(f"  Independent cache has small error:      {restoration['max_diff_independent'] < 1e-2}")
    print(f"  Re-prefill-in-context is lossless:      {restoration['reprefill_lossless']}")
    print(f"  IVFADC + cache is faster than no-cache:  {latency['speedup'] > 1.0}")
    print()
    all_pass = (retrieval['ivfadc_hit_rate'] > retrieval['random_hit_rate']
                and restoration['max_diff_independent'] < 1e-2
                and restoration['reprefill_lossless']
                and latency['speedup'] > 1.0)
    if all_pass:
        print("  ✓ The IVFADC + two-cache architecture WORKS on the toy model.")
    else:
        print("  ✗ Some tests failed — see above.")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
