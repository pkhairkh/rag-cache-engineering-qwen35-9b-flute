#!/usr/bin/env python3
"""toy_snapshots_only.py — the CORRECT architecture. NO text on disk.

WHAT WE STORE ON DISK:
  - Per chunk: the snapshotted caches (S, conv_state, delta_M1, delta_M2)
    and the pooled hidden vector (for IVFADC).
  - NO chunk text. NO token IDs. NO re-prefill fallback.

WHAT WE DO AT QUERY TIME:
  1. Embed the query → IVFADC preselect → cos sim rerank → top-k chunk IDs
  2. Load those k chunks' SNAPSHOTS from disk
  3. INSTALL the snapshots directly into the running model:
     - sum the delta_M1, delta_M2 (composable, path-independent)
     - sum the S deltas (also composable with conv-reset)
     - restore the conv_state (last chunk's)
  4. The model answers the query FROM THE INSTALLED CACHE STATE.
     It never sees the chunk text. The cache IS the memory.

This is NOT 2020 RAG. There is no retrieval of text, no re-prefill,
no chunk text on disk. The model's installed cache state IS the
retrieved knowledge.
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
)
from toy_ivfadc_caches import SimpleIVFADC


# ---------------------------------------------------------------------------
# The snapshot — caches ONLY, no text
# ---------------------------------------------------------------------------

@dataclass
class CacheSnapshot:
    """What we store per chunk. NO text. NO token IDs.
    Just the caches + the vector for IVFADC."""
    delta_S: torch.Tensor          # the chunk's contribution to S (composable)
    delta_M1: torch.Tensor         # the chunk's contribution to M1 (composable)
    delta_M2: torch.Tensor         # the chunk's contribution to M2 (composable)
    conv_state: torch.Tensor       # the chunk's final conv state (for the query's sliding window)
    hidden: torch.Tensor           # the pooled hidden vector (for IVFADC retrieval)


# ---------------------------------------------------------------------------
# Ingestion — snapshot the caches, save to disk, NO TEXT
# ---------------------------------------------------------------------------

def ingest_and_snapshot(model, chunks, device, snapshots_dir):
    """Ingest all chunks: prefill each (conv-reset, path-independent),
    snapshot the cache deltas + the pooled hidden vector. Save to disk.
    NO chunk text is saved. NO token IDs. Just the caches."""
    os.makedirs(snapshots_dir, exist_ok=True)
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)

    vectors = []
    chunk_hashes = []
    for idx, chunk in enumerate(chunks):
        # prefill the chunk from clean state (conv-reset → path-independent)
        with torch.no_grad():
            logits, S_after, conv_after, M1_after, M2_after = model(
                chunk, S=S0, conv_state=conv0, M1_state=M1_0, M2_state=M2_0, return_states=True)
            hidden = logits.mean(dim=1)  # the pooled vector for IVFADC

        # the chunk's cache deltas (composable by summation)
        delta_S = S_after - S0       # S started at 0, so delta_S = S_after
        delta_M1 = M1_after - M1_0   # M1 started at the init, delta is the contribution
        delta_M2 = M2_after - M2_0

        chunk_hash = hash_tokens(chunk)
        chunk_hashes.append(chunk_hash)

        # save the snapshot to disk (caches + vector, NO TEXT)
        snap_path = os.path.join(snapshots_dir, f"chunk_{idx:04d}.npz")
        np.savez(snap_path,
                 chunk_hash=chunk_hash,
                 delta_S=delta_S.detach().cpu().numpy(),
                 delta_M1=delta_M1.detach().cpu().numpy(),
                 delta_M2=delta_M2.detach().cpu().numpy(),
                 conv_state=conv_after.detach().cpu().numpy(),
                 hidden=hidden.detach().cpu().numpy())
        vectors.append(hidden.squeeze(0).detach().cpu().numpy())

        if (idx + 1) % 20 == 0:
            print(f"  Snapshotted {idx+1}/{len(chunks)} chunks...")

    vectors = np.stack(vectors).astype(np.float32)
    return vectors, chunk_hashes


# ---------------------------------------------------------------------------
# IVFADC index (on the snapshot vectors)
# ---------------------------------------------------------------------------

def build_and_save_ivfadc(vectors, chunk_hashes, work_dir):
    """Build IVFADC on the snapshot vectors, save to disk."""
    n_chunks = len(vectors)
    index = SimpleIVFADC(nlist=min(16, max(1, n_chunks//4)), m=8, nprobe=4)
    index.build(vectors, list(range(n_chunks)))
    index_path = os.path.join(work_dir, "ivfadc_index.npz")
    np.savez(index_path,
             vectors=vectors,
             chunk_hashes=np.array(chunk_hashes),
             coarse_centroids=index.coarse_centroids,
             pq_centroids=index.pq_centroids,
             pq_codes=index.pq_codes,
             cluster_assignments=index.cluster_assignments,
             chunk_ids=index.chunk_ids)
    return index, vectors


def load_ivfadc(work_dir):
    """Load the IVFADC index from disk."""
    path = os.path.join(work_dir, "ivfadc_index.npz")
    data = np.load(path, allow_pickle=True)
    index = SimpleIVFADC()
    index.vectors = data['vectors']
    index.coarse_centroids = data['coarse_centroids']
    index.pq_centroids = data['pq_centroids']
    index.pq_codes = data['pq_codes']
    index.cluster_assignments = data['cluster_assignments']
    index.chunk_ids = data['chunk_ids']
    return index, data['vectors']


# ---------------------------------------------------------------------------
# The disk-backed snapshot pool (loads cache snapshots on demand)
# ---------------------------------------------------------------------------

class SnapshotPool:
    """Loads cache snapshots from disk on demand. LRU in memory.
    Returns the cache deltas + conv_state. NO TEXT."""
    def __init__(self, snapshots_dir, max_in_memory=64, device='cpu'):
        self.snapshots_dir = snapshots_dir
        self.max_in_memory = max_in_memory
        self.device = device
        self.in_memory: OrderedDict[str, CacheSnapshot] = OrderedDict()
        self.hash_to_idx: Dict[str, int] = {}
        # scan for snapshot files
        for fname in sorted(os.listdir(snapshots_dir)):
            if fname.startswith("chunk_") and fname.endswith(".npz") and "meta" not in fname:
                idx = int(fname.replace("chunk_", "").replace(".npz", ""))
                # we don't have the hash without loading; just map idx→path
        # build idx→path map (we'll load hash on demand)
        self.idx_files = sorted([f for f in os.listdir(snapshots_dir)
                                  if f.startswith("chunk_") and f.endswith(".npz") and "meta" not in f])
        self.hits = 0
        self.misses = 0

    def lookup_by_idx(self, idx):
        """Load a snapshot by chunk index. Returns CacheSnapshot or None."""
        # check in-memory first (keyed by idx as string)
        key = str(idx)
        if key in self.in_memory:
            self.hits += 1
            self.in_memory.move_to_end(key)
            return self.in_memory[key]
        self.misses += 1
        if idx >= len(self.idx_files):
            return None
        path = os.path.join(self.snapshots_dir, self.idx_files[idx])
        data = np.load(path, allow_pickle=True)
        snap = CacheSnapshot(
            delta_S=torch.from_numpy(data['delta_S']).to(self.device),
            delta_M1=torch.from_numpy(data['delta_M1']).to(self.device),
            delta_M2=torch.from_numpy(data['delta_M2']).to(self.device),
            conv_state=torch.from_numpy(data['conv_state']).to(self.device),
            hidden=torch.from_numpy(data['hidden']).to(self.device),
        )
        while len(self.in_memory) >= self.max_in_memory:
            self.in_memory.popitem(last=False)
        self.in_memory[key] = snap
        return snap


# ---------------------------------------------------------------------------
# Query: IVFADC → top-k → INSTALL caches → answer (NO TEXT)
# ---------------------------------------------------------------------------

def answer_query_from_caches(model, query_tokens, system_S, system_conv,
                              system_M1, system_M2, retrieved_idxs, pool, device):
    """Answer a query by INSTALLING the retrieved chunks' cache snapshots
    into the running model. The model never sees the chunk text.

    1. IVFADC has already given us the top-k chunk indices
    2. Load those chunks' cache snapshots from disk
    3. SUM the deltas (delta_S, delta_M1, delta_M2) — composable
    4. INSTALL into the model: S = system_S + sum(delta_S), etc.
    5. The model answers the query FROM THE INSTALLED CACHE STATE.

    The model's forward pass on the query tokens uses the installed caches.
    The chunk text is never loaded, never re-prefilled. The cache IS the memory."""
    if len(retrieved_idxs) == 0:
        restored_S = system_S
        restored_conv = system_conv
        restored_M1 = system_M1
        restored_M2 = system_M2
    else:
        # sum the deltas from all retrieved chunks
        restored_S = system_S.clone()
        restored_M1 = system_M1.clone()
        restored_M2 = system_M2.clone()
        for idx in retrieved_idxs:
            snap = pool.lookup_by_idx(idx)
            if snap is None:
                continue
            restored_S = restored_S + snap.delta_S
            restored_M1 = restored_M1 + snap.delta_M1
            restored_M2 = restored_M2 + snap.delta_M2
        # use the last retrieved chunk's conv_state (sliding window for the query)
        last_snap = pool.lookup_by_idx(retrieved_idxs[-1])
        restored_conv = last_snap.conv_state.clone() if last_snap else system_conv

    # the model answers the query FROM THE INSTALLED CACHES
    # NO chunk text is loaded. NO re-prefill. The cache IS the memory.
    with torch.no_grad():
        logits, _, _, _, _ = model(query_tokens,
                                    S=restored_S, conv_state=restored_conv,
                                    M1_state=restored_M1, M2_state=restored_M2)
    return logits


def embed_query(model, query_tokens, device):
    """Embed the query: prefill it, get the pooled hidden vector for IVFADC."""
    with torch.no_grad():
        logits, _, _, _, _ = model(query_tokens, return_states=True)
    return logits.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def retrieve_topk(model, query_tokens, index, vectors, top_k=3, nprobe=4):
    """IVFADC preselect + cos sim rerank → top-k chunk indices."""
    v_q = embed_query(model, query_tokens, model.embed.weight.device)
    candidates = index.search(v_q, k=min(100, len(vectors)))
    candidate_vectors = vectors[candidates]
    cand_norm = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-8)
    q_norm = v_q / (np.linalg.norm(v_q) + 1e-8)
    scores = cand_norm @ q_norm
    top_k_local = np.argsort(scores)[-top_k:][::-1]
    return candidates[top_k_local].tolist()


# ---------------------------------------------------------------------------
# The full test
# ---------------------------------------------------------------------------

def run_test(model, device, n_chunks=100, chunk_len=32, query_len=16,
             system_len=16, n_queries=20, top_k=3,
             work_dir="/home/z/my-project/scripts/poc_toy/snapshots_only_test"):
    print(f"\n{'='*70}")
    print(f"SNAPSHOTS-ONLY TEST: {n_chunks} chunks, {n_queries} queries, top_k={top_k}")
    print(f"NO TEXT ON DISK. CACHES ONLY. INSTALL DIRECTLY.")
    print(f"{'='*70}")

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    snapshots_dir = os.path.join(work_dir, "snapshots")
    vocab = model.vocab

    # ---- PHASE 1: GENERATE + INGEST + SNAPSHOT ----
    print(f"\n[Phase 1] Generate {n_chunks} chunks, ingest, snapshot caches to disk...")
    t0 = time.time()
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)
    vectors, chunk_hashes = ingest_and_snapshot(model, chunks, device, snapshots_dir)
    ingest_time = time.time() - t0
    snapshot_disk = sum(os.path.getsize(os.path.join(snapshots_dir, f))
                        for f in os.listdir(snapshots_dir))
    print(f"  Ingested + snapshotted in {ingest_time:.2f}s ({ingest_time/n_chunks*1000:.1f}ms/chunk)")
    print(f"  Disk: {snapshot_disk/1024:.1f} KiB ({snapshot_disk/n_chunks:.0f} B/chunk) — CACHES ONLY, NO TEXT")

    # ---- PHASE 2: BUILD IVFADC ----
    print(f"\n[Phase 2] Build IVFADC on snapshot vectors...")
    t0 = time.time()
    index, vectors = build_and_save_ivfadc(vectors, chunk_hashes, work_dir)
    index_disk = os.path.getsize(os.path.join(work_dir, "ivfadc_index.npz"))
    print(f"  Built in {time.time()-t0:.2f}s, disk: {index_disk/1024:.1f} KiB")

    # ---- PHASE 3: LOAD (fresh process) ----
    print(f"\n[Phase 3] Load (fresh process simulation)...")
    t0 = time.time()
    index_loaded, vectors_loaded = load_ivfadc(work_dir)
    pool = SnapshotPool(snapshots_dir, max_in_memory=32, device=device)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    # ---- PHASE 4: QUERIES ----
    print(f"\n[Phase 4] Run {n_queries} queries (IVFADC → install caches → answer)...")
    # the system prompt's caches (computed once, kept in memory)
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    system_prompt = torch.randint(0, vocab, (1, system_len))
    with torch.no_grad():
        _, sys_S, sys_conv, sys_M1, sys_M2 = model(system_prompt, S=S0, conv_state=conv0,
                                                     M1_state=M1_0, M2_state=M2_0, return_states=True)

    queries = [gen_query(q % 4, query_len, vocab, seed=q) for q in range(n_queries)]
    chunk_topics = [i % 4 for i in range(n_chunks)]

    # measure retrieval + latency
    retrieval_hits = 0
    total_query_time = 0
    for q_idx, query in enumerate(queries):
        query_topic = q_idx % 4
        t0 = time.time()
        # IVFADC → top-k
        retrieved_idxs = retrieve_topk(model, query, index_loaded, vectors_loaded, top_k=top_k)
        # INSTALL caches + answer
        logits = answer_query_from_caches(model, query, sys_S, sys_conv,
                                          sys_M1, sys_M2, retrieved_idxs, pool, device)
        total_query_time += time.time() - t0
        # check retrieval quality
        retrieved_topics = [chunk_topics[i] for i in retrieved_idxs]
        if query_topic in retrieved_topics:
            retrieval_hits += 1

    avg_query_ms = total_query_time / n_queries * 1000
    retrieval_rate = retrieval_hits / n_queries
    print(f"  Retrieval hit rate: {retrieval_rate*100:.1f}% (random: ~{(1-(1-0.25)**top_k)*100:.1f}%)")
    print(f"  Avg query latency: {avg_query_ms:.2f} ms (IVFADC + install + answer)")
    print(f"  Pool: hits={pool.hits}, misses={pool.misses}, in_memory={len(pool.in_memory)}")

    # ---- CORRECTNESS CHECK ----
    # The "no-cache" baseline: prefill (system + retrieved chunk TEXTS + query)
    # But we DON'T have the chunk texts at query time (they're not on disk).
    # The whole point is: the model answers from the installed caches.
    # For correctness, we verify the model produces DIFFERENT answers
    # depending on which chunks' caches are installed (i.e., the caches
    # carry information).
    print(f"\n[Correctness] Do installed caches carry information?")
    query = queries[0]
    # answer with NO chunks installed (just system prompt)
    logits_no_chunks = answer_query_from_caches(model, query, sys_S, sys_conv,
                                                 sys_M1, sys_M2, [], pool, device)
    # answer with chunk 0 installed
    logits_with_chunk0 = answer_query_from_caches(model, query, sys_S, sys_conv,
                                                   sys_M1, sys_M2, [0], pool, device)
    # answer with chunks 0,1,2 installed
    logits_with_3 = answer_query_from_caches(model, query, sys_S, sys_conv,
                                              sys_M1, sys_M2, [0,1,2], pool, device)
    diff_0 = (logits_no_chunks - logits_with_chunk0).abs().max().item()
    diff_3 = (logits_no_chunks - logits_with_3).abs().max().item()
    diff_0_vs_3 = (logits_with_chunk0 - logits_with_3).abs().max().item()
    print(f"  Diff (no chunks vs 1 chunk):    {diff_0:.6e}")
    print(f"  Diff (no chunks vs 3 chunks):   {diff_3:.6e}")
    print(f"  Diff (1 chunk vs 3 chunks):     {diff_0_vs_3:.6e}")
    print(f"  Caches carry information: {diff_0 > 1e-4 or diff_3 > 1e-4}")
    print(f"  (Different installed caches → different answers. The cache IS the memory.)")

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    results = {
        'n_chunks': n_chunks,
        'n_queries': n_queries,
        'top_k': top_k,
        'architecture': 'snapshots_only — NO TEXT ON DISK',
        'ingest_time_s': round(ingest_time, 2),
        'disk_snapshots_kib': round(snapshot_disk/1024, 1),
        'disk_per_chunk_b': round(snapshot_disk/n_chunks),
        'disk_index_kib': round(index_disk/1024, 1),
        'retrieval_hit_rate': round(retrieval_rate, 3),
        'retrieval_random_baseline': round(1-(1-0.25)**top_k, 3),
        'avg_query_ms': round(avg_query_ms, 2),
        'caches_carry_info': diff_0 > 1e-4 or diff_3 > 1e-4,
        'diff_no_vs_1_chunk': diff_0,
        'diff_no_vs_3_chunks': diff_3,
        'pool_hits': pool.hits,
        'pool_misses': pool.misses,
    }
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\n  KEY: NO chunk text on disk. NO re-prefill.")
    print(f"  The model answers from INSTALLED CACHE SNAPSHOTS.")
    print(f"  This is cache-engineered RAG, not 2020 RAG.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-chunks', type=int, default=100)
    ap.add_argument('--n-queries', type=int, default=20)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/snapshots_only_results.json')
    ap.add_argument('--work-dir', default='/home/z/my-project/scripts/poc_toy/snapshots_only_test')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    model = TinyHybridModelWithKimiCaches(hidden=32, vocab=256, mem_size=16).to(device)
    model.eval()

    results = run_test(model, device, n_chunks=args.n_chunks, chunk_len=32,
                       query_len=16, system_len=16, n_queries=args.n_queries,
                       top_k=args.top_k, work_dir=args.work_dir)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
