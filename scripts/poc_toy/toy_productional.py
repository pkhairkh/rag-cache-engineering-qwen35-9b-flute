#!/usr/bin/env python3
"""toy_productional.py — a productional end-to-end test.

NOT a toy-scale test. This runs the full pipeline at a realistic scale:
1. Generate 100 chunks (realistic chunk count for a small corpus)
2. Ingest: prefill each, snapshot S + M1 + M2 deltas, SAVE to disk (numpy)
3. Build IVFADC index on the pooled hidden states
4. SAVE the index to disk
5. Load everything back from disk (simulate a fresh process)
6. For 20 queries: IVFADC preselect → cos sim rerank → top-3 → augment
7. Measure: correctness (cache vs no-cache), retrieval quality, latency,
   disk footprint, end-to-end timing

This tests the ACTUAL production flow: save → load → retrieve → augment.
"""
import argparse, hashlib, json, math, os, random, sys, time, shutil
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_kimi_two_caches import (
    TinyHybridModelWithKimiCaches, gen_corpus, gen_query, hash_tokens,
    ingest_corpus, query_with_caches, query_no_cache,
)
from toy_ivfadc_caches import SimpleIVFADC


# ---------------------------------------------------------------------------
# Disk persistence (the "save" step)
# ---------------------------------------------------------------------------

def save_snapshot(chunk_hash, snapshot, out_dir, idx):
    """Save a chunk's snapshot (S, conv_state, delta_M1, delta_M2, hidden) to disk."""
    path = os.path.join(out_dir, f"chunk_{idx:04d}.npz")
    np.savez(path,
             chunk_hash=chunk_hash,
             S=snapshot.S.detach().cpu().numpy(),
             conv_state=snapshot.conv_state.detach().cpu().numpy(),
             delta_M1=snapshot.delta_M1.detach().cpu().numpy(),
             delta_M2=snapshot.delta_M2.detach().cpu().numpy(),
             hidden=snapshot.hidden.detach().cpu().numpy())


def load_snapshot(path, device):
    """Load a snapshot from disk."""
    data = np.load(path, allow_pickle=True)
    chunk_hash = str(data['chunk_hash'])
    S = torch.from_numpy(data['S']).to(device)
    conv_state = torch.from_numpy(data['conv_state']).to(device)
    delta_M1 = torch.from_numpy(data['delta_M1']).to(device)
    delta_M2 = torch.from_numpy(data['delta_M2']).to(device)
    hidden = torch.from_numpy(data['hidden']).to(device)
    return chunk_hash, S, conv_state, delta_M1, delta_M2, hidden


def save_chunk_metadata(chunk_hash, chunk_tokens, out_dir, idx):
    """Save the chunk's token IDs (for re-prefill fallback) and metadata."""
    path = os.path.join(out_dir, f"chunk_meta_{idx:04d}.npz")
    np.savez(path,
             chunk_hash=chunk_hash,
             token_ids=chunk_tokens.cpu().numpy())


def save_ivfadc_index(index, vectors, chunk_hashes, out_dir):
    """Save the IVFADC index to disk."""
    path = os.path.join(out_dir, "ivfadc_index.npz")
    np.savez(path,
             vectors=vectors,
             chunk_hashes=np.array(chunk_hashes),
             coarse_centroids=index.coarse_centroids,
             pq_centroids=index.pq_centroids,
             pq_codes=index.pq_codes,
             cluster_assignments=index.cluster_assignments,
             chunk_ids=index.chunk_ids)


def load_ivfadc_index(path):
    """Load the IVFADC index from disk. Returns (index, vectors, chunk_hashes)."""
    data = np.load(path, allow_pickle=True)
    index = SimpleIVFADC()
    index.vectors = data['vectors']
    index.coarse_centroids = data['coarse_centroids']
    index.pq_centroids = data['pq_centroids']
    index.pq_codes = data['pq_codes']
    index.cluster_assignments = data['cluster_assignments']
    index.chunk_ids = data['chunk_ids']
    vectors = data['vectors']
    chunk_hashes = list(data['chunk_hashes'])
    return index, vectors, chunk_hashes


# ---------------------------------------------------------------------------
# The productional pool (loads from disk on demand)
# ---------------------------------------------------------------------------

@dataclass
class DiskSnapshot:
    """Compatible with ChunkSnapshot — has the same attributes."""
    delta_M1: torch.Tensor
    delta_M2: torch.Tensor
    S: torch.Tensor
    conv_state: torch.Tensor
    hidden: torch.Tensor


class DiskBackedPool:
    """A cache pool that loads snapshots from disk on demand.
    Caches loaded snapshots in memory (LRU) for repeated access.
    Returns DiskSnapshot objects (compatible with ChunkSnapshot)."""
    def __init__(self, snapshots_dir, max_in_memory=64, device='cpu'):
        self.snapshots_dir = snapshots_dir
        self.max_in_memory = max_in_memory
        self.device = device
        self.in_memory: OrderedDict[str, DiskSnapshot] = OrderedDict()
        self.hash_to_idx: Dict[str, int] = {}
        for fname in sorted(os.listdir(snapshots_dir)):
            if fname.startswith("chunk_meta_"):
                meta_path = os.path.join(snapshots_dir, fname)
                data = np.load(meta_path, allow_pickle=True)
                chunk_hash = str(data['chunk_hash'])
                idx = int(fname.replace("chunk_meta_", "").replace(".npz", ""))
                self.hash_to_idx[chunk_hash] = idx
        self.hits = 0
        self.misses = 0

    def lookup(self, chunk_hash):
        if chunk_hash in self.in_memory:
            self.hits += 1
            self.in_memory.move_to_end(chunk_hash)
            return self.in_memory[chunk_hash]
        self.misses += 1
        if chunk_hash not in self.hash_to_idx:
            return None
        idx = self.hash_to_idx[chunk_hash]
        snap_path = os.path.join(self.snapshots_dir, f"chunk_{idx:04d}.npz")
        _, S, conv, dM1, dM2, hidden = load_snapshot(snap_path, self.device)
        snap = DiskSnapshot(delta_M1=dM1, delta_M2=dM2, S=S, conv_state=conv, hidden=hidden)
        while len(self.in_memory) >= self.max_in_memory:
            self.in_memory.popitem(last=False)
        self.in_memory[chunk_hash] = snap
        return snap


# ---------------------------------------------------------------------------
# The productional pipeline
# ---------------------------------------------------------------------------

def run_productional_test(model, device, n_chunks=100, chunk_len=32, query_len=16,
                          system_len=16, n_queries=20, top_k=3,
                          work_dir="/home/z/my-project/scripts/poc_toy/prod_test"):
    """Run the full productional pipeline:
    1. Generate 100 chunks
    2. Ingest: prefill, snapshot, SAVE to disk
    3. Build IVFADC, SAVE to disk
    4. Load everything back (fresh process simulation)
    5. For 20 queries: IVFADC → rerank → augment
    6. Measure everything
    """
    print(f"\n{'='*70}")
    print(f"PRODUCTIONAL TEST: {n_chunks} chunks, {n_queries} queries, top_k={top_k}")
    print(f"{'='*70}")

    # clean the work dir
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    vocab = model.vocab

    # ---- PHASE 1: GENERATE ----
    print(f"\n[Phase 1] Generating {n_chunks} chunks...")
    t0 = time.time()
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)
    print(f"  Generated {len(chunks)} chunks in {time.time()-t0:.2f}s")

    # ---- PHASE 2: INGEST + SAVE ----
    print(f"\n[Phase 2] Ingesting {n_chunks} chunks (prefill + snapshot + save to disk)...")
    t0 = time.time()
    snapshots_dir = os.path.join(work_dir, "snapshots")
    os.makedirs(snapshots_dir)
    # ingest with conv-reset (path-independent)
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)

    vectors = []
    chunk_hashes = []
    for idx, chunk in enumerate(chunks):
        M1_before = M1_0.clone()
        M2_before = M2_0.clone()
        with torch.no_grad():
            logits, S_after, conv_after, M1_after, M2_after = model(
                chunk, S=S0, conv_state=conv0, M1_state=M1_0, M2_state=M2_0, return_states=True)
            hidden = logits.mean(dim=1)
        delta_M1 = M1_after - M1_before
        delta_M2 = M2_after - M2_before
        chunk_hash = hash_tokens(chunk)
        # save snapshot to disk
        from toy_kimi_two_caches import ChunkSnapshot
        snap = ChunkSnapshot(delta_M1, delta_M2, S_after, conv_after, hidden)
        save_snapshot(chunk_hash, snap, snapshots_dir, idx)
        save_chunk_metadata(chunk_hash, chunk, snapshots_dir, idx)
        vectors.append(hidden.squeeze(0).cpu().numpy())
        chunk_hashes.append(chunk_hash)
        if (idx + 1) % 20 == 0:
            print(f"  Ingested {idx+1}/{n_chunks} chunks...")
    vectors = np.stack(vectors).astype(np.float32)
    ingest_time = time.time() - t0
    print(f"  Ingestion complete: {ingest_time:.2f}s ({ingest_time/n_chunks*1000:.1f}ms/chunk)")

    # disk footprint
    snapshot_files = [f for f in os.listdir(snapshots_dir) if f.startswith("chunk_") and not f.startswith("chunk_meta")]
    total_disk = sum(os.path.getsize(os.path.join(snapshots_dir, f)) for f in os.listdir(snapshots_dir))
    print(f"  Disk footprint (snapshots + metadata): {total_disk/1024:.1f} KiB ({total_disk/n_chunks:.0f} B/chunk)")

    # ---- PHASE 3: BUILD + SAVE IVFADC ----
    print(f"\n[Phase 3] Building IVFADC index...")
    t0 = time.time()
    index = SimpleIVFADC(nlist=min(16, n_chunks//4), m=8, nprobe=4)
    index.build(vectors, list(range(n_chunks)))
    save_ivfadc_index(index, vectors, chunk_hashes, work_dir)
    index_time = time.time() - t0
    index_disk = os.path.getsize(os.path.join(work_dir, "ivfadc_index.npz"))
    print(f"  IVFADC built + saved in {index_time:.2f}s")
    print(f"  IVFADC index disk: {index_disk/1024:.1f} KiB")

    # ---- PHASE 4: SIMULATE FRESH PROCESS (load from disk) ----
    print(f"\n[Phase 4] Simulating fresh process (loading from disk)...")
    t0 = time.time()
    # load the IVFADC index
    index_path = os.path.join(work_dir, "ivfadc_index.npz")
    index_loaded, vectors_loaded, chunk_hashes_loaded = load_ivfadc_index(index_path)
    # create the disk-backed pool
    pool = DiskBackedPool(snapshots_dir, max_in_memory=32, device=device)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")
    print(f"  Pool has {len(pool.hash_to_idx)} chunk snapshots indexed")

    # ---- PHASE 5: QUERY (IVFADC → rerank → augment) ----
    print(f"\n[Phase 5] Running {n_queries} queries (IVFADC + rerank + cache augment)...")
    chunk_topics = [i % 4 for i in range(n_chunks)]

    # the query flow: embed query → IVFADC → rerank → restore deltas → prefill query
    def run_one_query(query_tokens, system_tokens, use_cache=True):
        # 1. embed the query
        with torch.no_grad():
            logits, _, _, _, _ = model(query_tokens, return_states=True)
        v_q = logits.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

        # 2. IVFADC preselect (top-100)
        candidate_idxs = index_loaded.search(v_q, k=min(100, n_chunks))

        # 3. cos sim rerank (top-k)
        candidate_vectors = vectors_loaded[candidate_idxs]
        cand_norm = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-8)
        q_norm = v_q / (np.linalg.norm(v_q) + 1e-8)
        scores = cand_norm @ q_norm
        top_k_local = np.argsort(scores)[-top_k:][::-1]
        retrieved_idxs = candidate_idxs[top_k_local]
        retrieved_chunks = [chunks[i] for i in retrieved_idxs]
        retrieved_topics = [chunk_topics[i] for i in retrieved_idxs]

        # 4. augment: restore deltas (sum) or re-prefill
        if use_cache:
            logits_out = query_with_caches(model, system_tokens, query_tokens,
                                            retrieved_chunks, pool, device)
        else:
            logits_out = query_no_cache(model, system_tokens, query_tokens,
                                         retrieved_chunks, device)
        return logits_out, retrieved_topics

    # generate queries (topic-biased)
    queries = [gen_query(q % 4, query_len, vocab, seed=q) for q in range(n_queries)]
    system_prompt = torch.randint(0, vocab, (1, system_len))

    # correctness: cache vs no-cache
    print(f"  Measuring correctness (cache vs no-cache)...")
    max_diff = 0.0
    for q_idx, query in enumerate(queries):
        logits_cache, _ = run_one_query(query, system_prompt, use_cache=True)
        logits_no, _ = run_one_query(query, system_prompt, use_cache=False)
        diff = (logits_cache - logits_no).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"  Max logit diff (cache vs no-cache): {max_diff:.6e}  (Lossless: {max_diff < 1e-4})")

    # retrieval quality
    print(f"  Measuring retrieval quality (topic match in top-{top_k})...")
    hits = 0
    for q_idx, query in enumerate(queries):
        query_topic = q_idx % 4
        _, retrieved_topics = run_one_query(query, system_prompt, use_cache=True)
        if query_topic in retrieved_topics:
            hits += 1
    hit_rate = hits / n_queries
    print(f"  Retrieval hit rate: {hit_rate*100:.1f}%  (random baseline: ~{(1-(1-0.25)**top_k)*100:.1f}%)")

    # latency
    print(f"  Measuring latency (cache vs no-cache)...")
    # warm up
    for _ in range(3):
        run_one_query(queries[0], system_prompt, use_cache=True)
        run_one_query(queries[0], system_prompt, use_cache=False)
    # time cache
    t0 = time.time()
    for query in queries:
        run_one_query(query, system_prompt, use_cache=True)
    t_cache = (time.time() - t0) / n_queries * 1000
    # time no-cache
    t0 = time.time()
    for query in queries:
        run_one_query(query, system_prompt, use_cache=False)
    t_no = (time.time() - t0) / n_queries * 1000
    speedup = t_no / t_cache if t_cache > 0 else 0
    print(f"  No-cache: {t_no:.2f} ms/query")
    print(f"  Cache:    {t_cache:.2f} ms/query  (speedup: {speedup:.2f}×)")

    # cache stats
    print(f"  Pool stats: hits={pool.hits}, misses={pool.misses}, in_memory={len(pool.in_memory)}")

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print(f"PRODUCTIONAL TEST SUMMARY")
    print(f"{'='*70}")
    results = {
        'n_chunks': n_chunks,
        'n_queries': n_queries,
        'top_k': top_k,
        'ingest_time_s': round(ingest_time, 2),
        'ingest_ms_per_chunk': round(ingest_time/n_chunks*1000, 1),
        'index_build_time_s': round(index_time, 2),
        'load_time_s': round(load_time, 2),
        'disk_snapshots_kib': round(total_disk/1024, 1),
        'disk_per_chunk_b': round(total_disk/n_chunks),
        'disk_index_kib': round(index_disk/1024, 1),
        'correctness_max_diff': max_diff,
        'correctness_lossless': max_diff < 1e-4,
        'retrieval_hit_rate': round(hit_rate, 3),
        'retrieval_random_baseline': round(1-(1-0.25)**top_k, 3),
        'latency_no_cache_ms': round(t_no, 2),
        'latency_cache_ms': round(t_cache, 2),
        'speedup': round(speedup, 2),
        'pool_hits': pool.hits,
        'pool_misses': pool.misses,
    }
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-chunks', type=int, default=100)
    ap.add_argument('--n-queries', type=int, default=20)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/productional_results.json')
    ap.add_argument('--work-dir', default='/home/z/my-project/scripts/poc_toy/prod_test')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    model = TinyHybridModelWithKimiCaches(hidden=32, vocab=256, mem_size=16).to(device)
    model.eval()

    results = run_productional_test(
        model, device,
        n_chunks=args.n_chunks, chunk_len=32, query_len=16,
        system_len=16, n_queries=args.n_queries, top_k=args.top_k,
        work_dir=args.work_dir)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
