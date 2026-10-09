#!/usr/bin/env python3
"""toy_1000_chunks.py — thorough test with 1000 chunks.

Tests the FULL pipeline at realistic scale:
1. SNAPSHOT 1000 chunks (10 topics, 100 chunks each)
2. SAVE snapshots to disk (caches only, NO TEXT)
3. Run 100 queries (topic-biased)
4. For each query: IVFADC preselect → cos sim rerank → top-k=3
5. INSTALL the retrieved snapshots into the running model
6. ANSWER the query from the installed caches (NO REPREFILL)

THE KEY TEST: does installing the IVFADC-retrieved caches produce
a BETTER answer than installing WRONG caches?

We measure:
- Retrieval precision: % of retrieved top-3 that match the query's topic
- Answer quality: does the IVFADC-retrieved answer match the "gold" answer
  (the answer when the correct-topic chunks are installed)?
- Comparison: IVFADC-retrieved vs random-retrieved vs gold-retrieved
- Latency: end-to-end query time
"""
import argparse, hashlib, json, math, os, random, sys, time, shutil
from collections import OrderedDict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_kimi_two_caches import (
    TinyHybridModelWithKimiCaches, gen_query, hash_tokens,
)
from toy_ivfadc_caches import SimpleIVFADC


# ---------------------------------------------------------------------------
# Generate 1000 chunks across 10 topics (100 chunks each)
# ---------------------------------------------------------------------------

N_TOPICS = 10
CHUNKS_PER_TOPIC = 100

def gen_topic_chunks(n_topics, chunks_per_topic, chunk_len, vocab, seed=42):
    """Generate chunks where chunk i has topic (i // chunks_per_topic).
    Each topic has a distinct token bias (topic t uses tokens [10 + t*20 : 10 + t*20 + 20])."""
    rng = random.Random(seed)
    chunks = []
    topics = []
    for t in range(n_topics):
        topic_tokens = [10 + t * 20 + j for j in range(20)]
        for _ in range(chunks_per_topic):
            tokens = []
            for _ in range(chunk_len):
                if rng.random() < 0.4:  # 40% topic-biased
                    tokens.append(rng.choice(topic_tokens))
                else:
                    tokens.append(rng.randint(0, vocab - 1))
            chunks.append(torch.tensor([tokens], dtype=torch.long))
            topics.append(t)
    return chunks, topics


def gen_topic_query(topic, query_len, vocab, seed=0):
    """Generate a query biased toward the given topic.
    Uses the SAME token pattern as gen_topic_chunks: [10 + topic*20 : 10 + topic*20 + 20]."""
    rng = random.Random(seed + topic)
    topic_tokens = [10 + topic * 20 + j for j in range(20)]
    tokens = []
    for _ in range(query_len):
        if rng.random() < 0.5:
            tokens.append(rng.choice(topic_tokens))
        else:
            tokens.append(rng.randint(0, vocab - 1))
    return torch.tensor([tokens], dtype=torch.long)


# ---------------------------------------------------------------------------
# Snapshot ingestion (caches only, NO TEXT on disk)
# ---------------------------------------------------------------------------

@dataclass
class CacheSnapshot:
    delta_S: torch.Tensor
    delta_M1: torch.Tensor
    delta_M2: torch.Tensor
    conv_state: torch.Tensor
    hidden: torch.Tensor


def ingest_and_snapshot(model, chunks, device, snapshots_dir):
    """Prefill each chunk (conv-reset, path-independent), snapshot caches, save to disk.
    NO TEXT saved. Just the cache deltas + the pooled vector."""
    os.makedirs(snapshots_dir, exist_ok=True)
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)

    vectors = []
    for idx, chunk in enumerate(chunks):
        with torch.no_grad():
            logits, S_after, conv_after, M1_after, M2_after = model(
                chunk, S=S0, conv_state=conv0, M1_state=M1_0, M2_state=M2_0, return_states=True)
            hidden = logits.mean(dim=1)
        delta_S = S_after - S0
        delta_M1 = M1_after - M1_0
        delta_M2 = M2_after - M2_0
        snap_path = os.path.join(snapshots_dir, f"chunk_{idx:05d}.npz")
        np.savez(snap_path,
                 delta_S=delta_S.detach().cpu().numpy(),
                 delta_M1=delta_M1.detach().cpu().numpy(),
                 delta_M2=delta_M2.detach().cpu().numpy(),
                 conv_state=conv_after.detach().cpu().numpy(),
                 hidden=hidden.detach().cpu().numpy())
        vectors.append(hidden.squeeze(0).detach().cpu().numpy())
        if (idx + 1) % 200 == 0:
            print(f"  Snapshotted {idx+1}/{len(chunks)} chunks...")
    vectors = np.stack(vectors).astype(np.float32)
    return vectors


# ---------------------------------------------------------------------------
# IVFADC
# ---------------------------------------------------------------------------

def build_ivfadc(vectors, nlist=32, m=8, nprobe=8):
    index = SimpleIVFADC(nlist=min(nlist, len(vectors)), m=m, nprobe=min(nprobe, nlist))
    index.build(vectors, list(range(len(vectors))))
    return index


# ---------------------------------------------------------------------------
# Snapshot pool (disk-backed, LRU in memory)
# ---------------------------------------------------------------------------

class SnapshotPool:
    def __init__(self, snapshots_dir, max_in_memory=128, device='cpu'):
        self.snapshots_dir = snapshots_dir
        self.max_in_memory = max_in_memory
        self.device = device
        self.in_memory: OrderedDict[str, CacheSnapshot] = OrderedDict()
        self.files = sorted([f for f in os.listdir(snapshots_dir)
                              if f.startswith("chunk_") and f.endswith(".npz")])
        self.hits = 0
        self.misses = 0

    def lookup(self, idx):
        key = str(idx)
        if key in self.in_memory:
            self.hits += 1
            self.in_memory.move_to_end(key)
            return self.in_memory[key]
        self.misses += 1
        if idx >= len(self.files):
            return None
        path = os.path.join(self.snapshots_dir, self.files[idx])
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
# Install caches + answer (NO REPREFILL)
# ---------------------------------------------------------------------------

def install_and_answer(model, query_tokens, sys_S, sys_conv, sys_M1, sys_M2,
                      retrieved_idxs, pool, device):
    """Install the retrieved snapshots' caches into the model, answer the query.
    NO REPREFILL. The model answers from the installed caches."""
    if len(retrieved_idxs) == 0:
        restored_S = sys_S
        restored_conv = sys_conv
        restored_M1 = sys_M1
        restored_M2 = sys_M2
    else:
        restored_S = sys_S.clone()
        restored_M1 = sys_M1.clone()
        restored_M2 = sys_M2.clone()
        last_snap = None
        for idx in retrieved_idxs:
            snap = pool.lookup(idx)
            if snap is None:
                continue
            restored_S = restored_S + snap.delta_S
            restored_M1 = restored_M1 + snap.delta_M1
            restored_M2 = restored_M2 + snap.delta_M2
            last_snap = snap
        restored_conv = last_snap.conv_state.clone() if last_snap else sys_conv
    with torch.no_grad():
        logits, _, _, _, _ = model(query_tokens,
                                    S=restored_S, conv_state=restored_conv,
                                    M1_state=restored_M1, M2_state=restored_M2)
    return logits


def embed_query(model, query_tokens, device):
    with torch.no_grad():
        logits, _, _, _, _ = model(query_tokens, return_states=True)
    return logits.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def ivfadc_retrieve(model, query_tokens, index, vectors, top_k=3):
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

def run_test(n_chunks=1000, chunk_len=32, query_len=16, system_len=16,
             n_queries=100, top_k=3,
             work_dir="/home/z/my-project/scripts/poc_toy/test_1000"):
    print(f"\n{'='*70}")
    print(f"THOROUGH TEST: {n_chunks} chunks, {n_queries} queries, top_k={top_k}")
    print(f"NO REPREFILL. INSTALL CACHES ONLY.")
    print(f"{'='*70}")

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    snapshots_dir = os.path.join(work_dir, "snapshots")

    torch.manual_seed(0)
    device = 'cpu'
    model = TinyHybridModelWithKimiCaches(hidden=32, vocab=256, mem_size=16).to(device)
    model.eval()
    vocab = model.vocab

    # ---- PHASE 1: GENERATE 1000 chunks ----
    print(f"\n[Phase 1] Generate {n_chunks} chunks ({N_TOPICS} topics × {CHUNKS_PER_TOPIC} chunks)...")
    chunks, chunk_topics = gen_topic_chunks(N_TOPICS, CHUNKS_PER_TOPIC, chunk_len, vocab, seed=42)
    print(f"  Generated {len(chunks)} chunks, topics: {Counter(chunk_topics)}")

    # ---- PHASE 2: SNAPSHOT + SAVE ----
    print(f"\n[Phase 2] Snapshot {n_chunks} chunks, save caches to disk (NO TEXT)...")
    t0 = time.time()
    vectors = ingest_and_snapshot(model, chunks, device, snapshots_dir)
    ingest_time = time.time() - t0
    snapshot_disk = sum(os.path.getsize(os.path.join(snapshots_dir, f))
                        for f in os.listdir(snapshots_dir))
    print(f"  Done in {ingest_time:.2f}s ({ingest_time/n_chunks*1000:.1f}ms/chunk)")
    print(f"  Disk: {snapshot_disk/1024:.1f} KiB ({snapshot_disk/n_chunks:.0f} B/chunk) — CACHES ONLY")

    # ---- PHASE 3: BUILD IVFADC ----
    print(f"\n[Phase 3] Build IVFADC index on snapshot vectors...")
    t0 = time.time()
    index = build_ivfadc(vectors, nlist=32, m=8, nprobe=8)
    print(f"  Built in {time.time()-t0:.2f}s")

    # ---- PHASE 4: POOL ----
    print(f"\n[Phase 4] Create disk-backed snapshot pool...")
    pool = SnapshotPool(snapshots_dir, max_in_memory=128, device=device)
    print(f"  Pool ready: {len(pool.files)} snapshots indexed")

    # ---- PHASE 5: SYSTEM PROMPT CACHE ----
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    system_prompt = torch.randint(0, vocab, (1, system_len))
    with torch.no_grad():
        _, sys_S, sys_conv, sys_M1, sys_M2 = model(system_prompt, S=S0, conv_state=conv0,
                                                     M1_state=M1_0, M2_state=M2_0, return_states=True)

    # ---- PHASE 6: RUN 100 QUERIES ----
    print(f"\n[Phase 6] Run {n_queries} queries (IVFADC → install → answer)...")
    # generate 100 queries, 10 per topic
    queries = []
    query_topics = []
    for q in range(n_queries):
        topic = q % N_TOPICS
        queries.append(gen_topic_query(topic, query_len, vocab, seed=q))
        query_topics.append(topic)

    # run each query three ways:
    # 1. IVFADC-retrieved top-3 (the real pipeline)
    # 2. GOLD top-3 (3 chunks from the correct topic — the upper bound)
    # 3. RANDOM top-3 (3 random chunks — the lower bound)
    ivfadc_results = []
    gold_results = []
    random_results = []
    no_cache_results = []

    ivfadc_topic_hits = 0
    gold_topic_hits = 0
    random_topic_hits = 0

    t_ivfadc_total = 0
    t_install_total = 0

    for q_idx, (query, q_topic) in enumerate(zip(queries, query_topics)):
        # 1. IVFADC retrieve
        t0 = time.time()
        ivfadc_idxs = ivfadc_retrieve(model, query, index, vectors, top_k=top_k)
        t_ivfadc = time.time() - t0
        t_ivfadc_total += t_ivfadc

        # check retrieval precision (how many of top-3 match the query's topic?)
        ivfadc_topics = [chunk_topics[i] for i in ivfadc_idxs]
        ivfadc_topic_match = sum(1 for t in ivfadc_topics if t == q_topic)
        if ivfadc_topic_match > 0:
            ivfadc_topic_hits += 1

        # install + answer
        t0 = time.time()
        logits_ivfadc = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                           ivfadc_idxs, pool, device)
        t_install = time.time() - t0
        t_install_total += t_install
        ivfadc_results.append(logits_ivfadc)

        # 2. GOLD: 3 random chunks from the correct topic
        correct_topic_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_topic_chunks, min(top_k, len(correct_topic_chunks)))
        gold_topics = [chunk_topics[i] for i in gold_idxs]
        gold_topic_match = sum(1 for t in gold_topics if t == q_topic)
        if gold_topic_match > 0:
            gold_topic_hits += 1
        logits_gold = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                          gold_idxs, pool, device)
        gold_results.append(logits_gold)

        # 3. RANDOM: 3 random chunks from ANY topic
        random_idxs = random.sample(range(n_chunks), top_k)
        random_topics = [chunk_topics[i] for i in random_idxs]
        random_topic_match = sum(1 for t in random_topics if t == q_topic)
        if random_topic_match > 0:
            random_topic_hits += 1
        logits_random = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                            random_idxs, pool, device)
        random_results.append(logits_random)

        # 4. NO CACHE (just system prompt)
        logits_no = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                        [], pool, device)
        no_cache_results.append(logits_no)

        if (q_idx + 1) % 20 == 0:
            print(f"  Query {q_idx+1}/{n_queries}: topic={q_topic}, "
                  f"IVFADC topics={ivfadc_topics}, match={ivfadc_topic_match}/3")

    # ---- ANALYSIS ----
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")

    # 1. Retrieval precision
    print(f"\n1. RETRIEVAL PRECISION (does IVFADC find the right topic?)")
    print(f"   IVFADC:  {ivfadc_topic_hits}/{n_queries} queries had ≥1 correct-topic chunk ({ivfadc_topic_hits/n_queries*100:.1f}%)")
    print(f"   GOLD:    {gold_topic_hits}/{n_queries} ({gold_topic_hits/n_queries*100:.1f}%) — should be 100%")
    print(f"   RANDOM:  {random_topic_hits}/{n_queries} ({random_topic_hits/n_queries*100:.1f}%) — baseline ~{(1-(1-1/N_TOPICS)**top_k)*100:.1f}%")

    # 2. Answer quality: does IVFADC answer match the GOLD answer?
    print(f"\n2. ANSWER QUALITY (does IVFADC-installed match GOLD-installed?)")
    ivfadc_vs_gold_diffs = []
    ivfadc_vs_random_diffs = []
    ivfadc_vs_nocache_diffs = []
    random_vs_gold_diffs = []
    for i in range(n_queries):
        ivf = ivfadc_results[i]
        gld = gold_results[i]
        rnd = random_results[i]
        noc = no_cache_results[i]
        ivfadc_vs_gold_diffs.append((ivf - gld).abs().max().item())
        ivfadc_vs_random_diffs.append((ivf - rnd).abs().max().item())
        ivfadc_vs_nocache_diffs.append((ivf - noc).abs().max().item())
        random_vs_gold_diffs.append((rnd - gld).abs().max().item())
    print(f"   IVFADC vs GOLD:    mean diff = {np.mean(ivfadc_vs_gold_diffs):.4f}  (lower = better)")
    print(f"   RANDOM vs GOLD:    mean diff = {np.mean(random_vs_gold_diffs):.4f}  (baseline)")
    print(f"   IVFADC vs RANDOM:  mean diff = {np.mean(ivfadc_vs_random_diffs):.4f}")
    print(f"   IVFADC vs NO-CACHE: mean diff = {np.mean(ivfadc_vs_nocache_diffs):.4f}")
    ivfadc_better = np.mean(ivfadc_vs_gold_diffs) < np.mean(random_vs_gold_diffs)
    print(f"   IVFADC closer to GOLD than RANDOM: {ivfadc_better}")

    # 3. Latency
    print(f"\n3. LATENCY (per query)")
    print(f"   IVFADC retrieve: {t_ivfadc_total/n_queries*1000:.2f} ms")
    print(f"   Install + answer: {t_install_total/n_queries*1000:.2f} ms")
    print(f"   Total:            {(t_ivfadc_total+t_install_total)/n_queries*1000:.2f} ms")
    print(f"   Pool: hits={pool.hits}, misses={pool.misses}")

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    results = {
        'n_chunks': n_chunks,
        'n_topics': N_TOPICS,
        'n_queries': n_queries,
        'top_k': top_k,
        'ingest_time_s': round(ingest_time, 2),
        'disk_kib': round(snapshot_disk/1024, 1),
        'disk_per_chunk_b': round(snapshot_disk/n_chunks),
        'retrieval_ivfadc_hit_rate': round(ivfadc_topic_hits/n_queries, 3),
        'retrieval_gold_hit_rate': round(gold_topic_hits/n_queries, 3),
        'retrieval_random_hit_rate': round(random_topic_hits/n_queries, 3),
        'retrieval_random_expected': round(1-(1-1/N_TOPICS)**top_k, 3),
        'answer_ivfadc_vs_gold_mean_diff': round(float(np.mean(ivfadc_vs_gold_diffs)), 4),
        'answer_random_vs_gold_mean_diff': round(float(np.mean(random_vs_gold_diffs)), 4),
        'ivfadc_closer_to_gold_than_random': bool(ivfadc_better),
        'latency_ivfadc_ms': round(t_ivfadc_total/n_queries*1000, 2),
        'latency_install_ms': round(t_install_total/n_queries*1000, 2),
        'latency_total_ms': round((t_ivfadc_total+t_install_total)/n_queries*1000, 2),
        'pool_hits': pool.hits,
        'pool_misses': pool.misses,
    }
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-chunks', type=int, default=1000)
    ap.add_argument('--n-queries', type=int, default=100)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/test_1000_results.json')
    ap.add_argument('--work-dir', default='/home/z/my-project/scripts/poc_toy/test_1000')
    args = ap.parse_args()
    results = run_test(n_chunks=args.n_chunks, n_queries=args.n_queries, top_k=args.top_k,
                       work_dir=args.work_dir)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
