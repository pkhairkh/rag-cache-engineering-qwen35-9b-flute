#!/usr/bin/env python3
"""toy_cache_as_vector.py — retrieve by SNAPSHOT, not by hidden state.

The user's exact logic:
1. SNAPSHOT the cache (S, M1, M2) from the user query
2. CONCATENATE the different caches (flatten S + M1 + M2 into one vector)
3. Run IVFADC against the snapshotted chunk caches (the SAME representation)
4. Rerank via cosine sim
5. Install the top-k snapshots into the model
6. Answer from the installed caches

NO hidden states. NO pooled logits. The CACHE ITSELF (S + M1 + M2, flattened)
IS the retrieval vector. The query's cache and the chunk's cache are in the
SAME space (both are linear-attn state matrices) — so cos-sim is meaningful.
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
from toy_kimi_two_caches import LinearAttnWithKimiCaches as GatedDeltaNet, TinyFullAttention
from toy_ivfadc_caches import SimpleIVFADC


# ---------------------------------------------------------------------------
# The model (4 linear + 1 full, with Kimi caches)
# ---------------------------------------------------------------------------

class HybridModel(nn.Module):
    def __init__(self, hidden=128, vocab=512, num_linear=4, num_heads=8,
                 head_k=16, head_v=16, mem_size=32):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.num_linear = num_linear
        self.embed = nn.Embedding(vocab, hidden)
        self.linear_layers = nn.ModuleList([
            GatedDeltaNet(hidden=hidden, num_v_heads=num_heads, num_k_heads=num_heads,
                          head_k_dim=head_k, head_v_dim=head_v, mem_size=mem_size)
            for _ in range(num_linear)
        ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_linear)])
        self.linear_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden))
            for _ in range(num_linear)
        ])
        self.linear_mlp_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_linear)])
        self.full_attn = TinyFullAttention(hidden=hidden, num_heads=8, head_dim=hidden//8)
        self.full_norm = nn.LayerNorm(hidden)
        self.full_mlp = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden))
        self.full_mlp_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, layer_states=None, return_states=False):
        x = self.embed(input_ids)
        new_states = []
        for i, layer in enumerate(self.linear_layers):
            if layer_states is not None and i < len(layer_states):
                S, conv, M1, M2 = layer_states[i]
            else:
                S, conv, M1, M2 = layer.initial_state(x.shape[0])
                S, conv = S.to(x.device), conv.to(x.device)
                M1, M2 = M1.to(x.device), M2.to(x.device)
            lin_out, new_S, new_conv, new_M1, new_M2 = layer.forward_chunk(x, S, conv, M1, M2)
            x = self.linear_norms[i](x + lin_out)
            x = self.linear_mlp_norms[i](x + self.linear_mlps[i](x))
            new_states.append((new_S, new_conv, new_M1, new_M2))
        full_out = self.full_attn(x)
        x = self.full_norm(x + full_out)
        x = self.full_mlp_norm(x + self.full_mlp(x))
        logits = self.lm_head(x)
        if return_states:
            return logits, new_states
        return logits, new_states

    def initial_states(self, batch=1, device='cpu'):
        return [(s.to(device), c.to(device), m1.to(device), m2.to(device))
                for s, c, m1, m2 in
                [self.linear_layers[i].initial_state(batch) for i in range(self.num_linear)]]


# ---------------------------------------------------------------------------
# THE KEY: flatten the cache (S + M1 + M2) into a single vector
# ---------------------------------------------------------------------------

def flatten_cache(states, num_layers):
    """Flatten the cache (S + M1 + M2 from all linear layers) into a single 1D vector.
    This IS the retrieval vector — NO hidden state, NO pooled logits.
    The query's cache and the chunk's cache are in the SAME space."""
    parts = []
    for i in range(num_layers):
        S, conv, M1, M2 = states[i]
        parts.append(S.detach().flatten())
        parts.append(M1.detach().flatten())
        parts.append(M2.detach().flatten())
    return torch.cat(parts).cpu().numpy().astype(np.float32)


def flatten_cache_size(model):
    """Compute the flattened cache vector size."""
    init = model.initial_states(1)
    return flatten_cache(init, model.num_linear).shape[0]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

N_TOPICS = 10

def gen_topic_chunks(n_topics, chunks_per_topic, chunk_len, vocab, seed=42):
    rng = random.Random(seed)
    chunks = []
    topics = []
    for t in range(n_topics):
        topic_tokens = [10 + t * 20 + j for j in range(20)]
        for _ in range(chunks_per_topic):
            tokens = []
            for _ in range(chunk_len):
                if rng.random() < 0.4:
                    tokens.append(rng.choice(topic_tokens))
                else:
                    tokens.append(rng.randint(0, vocab - 1))
            chunks.append(torch.tensor([tokens], dtype=torch.long))
            topics.append(t)
    return chunks, topics


def gen_topic_query(topic, query_len, vocab, seed=0):
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
# Snapshot + save (caches only)
# ---------------------------------------------------------------------------

@dataclass
class ChunkSnapshot:
    delta_S_list: List[np.ndarray]
    delta_M1_list: List[np.ndarray]
    delta_M2_list: List[np.ndarray]
    conv_state_list: List[np.ndarray]
    cache_vector: np.ndarray  # the flattened cache (S+M1+M2) — the retrieval vector


def ingest_and_snapshot(model, chunks, device, snapshots_dir):
    """Ingest: prefill each chunk (conv-reset), snapshot the cache deltas + the
    flattened cache vector. The cache vector IS the retrieval vector."""
    os.makedirs(snapshots_dir, exist_ok=True)
    init_states = model.initial_states(1, device)
    cache_vectors = []
    for idx, chunk in enumerate(chunks):
        with torch.no_grad():
            logits, new_states = model(chunk, layer_states=init_states, return_states=True)
        # the cache vector = flattened (S + M1 + M2) from the chunk's final state
        cache_vec = flatten_cache(new_states, model.num_linear)
        # the deltas (composable by summation)
        delta_S_list, delta_M1_list, delta_M2_list, conv_state_list = [], [], [], []
        for i, (new_S, new_conv, new_M1, new_M2) in enumerate(new_states):
            init_S, init_conv, init_M1, init_M2 = init_states[i]
            delta_S_list.append((new_S - init_S).detach().cpu().numpy())
            delta_M1_list.append((new_M1 - init_M1).detach().cpu().numpy())
            delta_M2_list.append((new_M2 - init_M2).detach().cpu().numpy())
            conv_state_list.append(new_conv.detach().cpu().numpy())
        snap_path = os.path.join(snapshots_dir, f"chunk_{idx:05d}.npz")
        save_dict = {'cache_vector': cache_vec}
        for i in range(len(delta_S_list)):
            save_dict[f'delta_S_{i}'] = delta_S_list[i]
            save_dict[f'delta_M1_{i}'] = delta_M1_list[i]
            save_dict[f'delta_M2_{i}'] = delta_M2_list[i]
            save_dict[f'conv_state_{i}'] = conv_state_list[i]
        np.savez(snap_path, **save_dict)
        cache_vectors.append(cache_vec)
        if (idx + 1) % 100 == 0:
            print(f"  Snapshotted {idx+1}/{len(chunks)} chunks...")
    return np.stack(cache_vectors).astype(np.float32)


class SnapshotPool:
    def __init__(self, snapshots_dir, num_layers, max_in_memory=128, device='cpu'):
        self.snapshots_dir = snapshots_dir
        self.num_layers = num_layers
        self.max_in_memory = max_in_memory
        self.device = device
        self.in_memory: OrderedDict[str, tuple] = OrderedDict()
        self.files = sorted([f for f in os.listdir(snapshots_dir)
                              if f.startswith("chunk_") and f.endswith(".npz")])

    def lookup(self, idx):
        key = str(idx)
        if key in self.in_memory:
            self.in_memory.move_to_end(key)
            return self.in_memory[key]
        if idx >= len(self.files):
            return None
        path = os.path.join(self.snapshots_dir, self.files[idx])
        data = np.load(path, allow_pickle=True)
        snap = (
            [torch.from_numpy(data[f'delta_S_{i}']).to(self.device) for i in range(self.num_layers)],
            [torch.from_numpy(data[f'delta_M1_{i}']).to(self.device) for i in range(self.num_layers)],
            [torch.from_numpy(data[f'delta_M2_{i}']).to(self.device) for i in range(self.num_layers)],
            [torch.from_numpy(data[f'conv_state_{i}']).to(self.device) for i in range(self.num_layers)],
            data['cache_vector'],
        )
        while len(self.in_memory) >= self.max_in_memory:
            self.in_memory.popitem(last=False)
        self.in_memory[key] = snap
        return snap


# ---------------------------------------------------------------------------
# THE QUERY FLOW: snapshot query's cache → IVFADC on cache vectors → install → answer
# ---------------------------------------------------------------------------

def snapshot_query_cache(model, query, device):
    """Snapshot the query's cache (S + M1 + M2, flattened). This is the retrieval vector."""
    init_states = model.initial_states(1, device)
    with torch.no_grad():
        logits, new_states = model(query, layer_states=init_states, return_states=True)
    return flatten_cache(new_states, model.num_linear)


def retrieve_by_cache(query_cache_vec, index, exact_vectors, top_k=3):
    """IVFADC preselect + cos sim rerank on the CACHE vectors."""
    # IVFADC preselect
    candidates = index.search(query_cache_vec, k=min(100, len(exact_vectors)))
    # cos sim rerank
    cand_vecs = exact_vectors[candidates]
    cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = query_cache_vec / (np.linalg.norm(query_cache_vec) + 1e-8)
    scores = cand_norm @ q_norm
    top_k_local = np.argsort(scores)[-top_k:][::-1]
    return candidates[top_k_local].tolist()


def install_and_answer(model, query, sys_states, retrieved_idxs, pool, device):
    """Install the top-k chunks' cache deltas, answer from installed caches."""
    if len(retrieved_idxs) == 0:
        restored_states = sys_states
    else:
        restored_states = []
        for i in range(model.num_linear):
            r_S = sys_states[i][0].clone()
            r_M1 = sys_states[i][2].clone()
            r_M2 = sys_states[i][3].clone()
            last_conv = sys_states[i][1]
            for idx in retrieved_idxs:
                snap = pool.lookup(idx)
                if snap is None:
                    continue
                delta_S_list, delta_M1_list, delta_M2_list, conv_state_list, _ = snap
                r_S = r_S + delta_S_list[i]
                r_M1 = r_M1 + delta_M1_list[i]
                r_M2 = r_M2 + delta_M2_list[i]
                last_conv = conv_state_list[i]
            restored_states.append((r_S, last_conv, r_M1, r_M2))
    with torch.no_grad():
        logits, _ = model(query, layer_states=restored_states, return_states=True)
    return logits


# ---------------------------------------------------------------------------
# Pre-train (so the caches carry info)
# ---------------------------------------------------------------------------

def pretrain(model, chunks, device, n_steps=1500, lr=1e-3):
    print(f"\n[Pre-train] {n_steps} steps, next-token prediction...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(n_steps):
        chunk = random.choice(chunks)
        inp = chunk[:, :-1]
        target = chunk[:, 1:]
        logits, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    model.eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-chunks', type=int, default=500)
    ap.add_argument('--n-queries', type=int, default=50)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--pretrain-steps', type=int, default=1500)
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/cache_as_vector_results.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    model = HybridModel(hidden=128, vocab=512, num_linear=4, num_heads=8,
                         head_k=16, head_v=16, mem_size=32).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    cache_vec_size = flatten_cache_size(model)
    print(f"Model: {n_params:,} params, cache vector size: {cache_vec_size}")

    chunks_per_topic = args.n_chunks // N_TOPICS
    chunks, chunk_topics = gen_topic_chunks(N_TOPICS, chunks_per_topic, 32, model.vocab, seed=42)
    queries = [gen_topic_query(q % N_TOPICS, 16, model.vocab, seed=q) for q in range(args.n_queries)]
    query_topics = [q % N_TOPICS for q in range(args.n_queries)]
    system_prompt = torch.randint(0, model.vocab, (1, 16))

    # PRE-TRAIN
    pretrain(model, chunks, device, n_steps=args.pretrain_steps, lr=1e-3)

    # INGEST + SNAPSHOT
    print(f"\nSnapshotting {len(chunks)} chunks...")
    snapshots_dir = "/tmp/cache_as_vector_snaps"
    if os.path.exists(snapshots_dir): shutil.rmtree(snapshots_dir)
    t0 = time.time()
    cache_vectors = ingest_and_snapshot(model, chunks, device, snapshots_dir)
    print(f"  Done: {time.time()-t0:.1f}s, cache vectors: {cache_vectors.shape}")

    # BUILD IVFADC on the cache vectors
    print(f"\nBuilding IVFADC on cache vectors...")
    index = SimpleIVFADC(nlist=min(32, len(cache_vectors)), m=8, nprobe=8)
    index.build(cache_vectors, list(range(len(cache_vectors))))
    pool = SnapshotPool(snapshots_dir, model.num_linear, max_in_memory=128, device=device)
    sys_states = model.initial_states(1, device)
    with torch.no_grad():
        _, sys_states = model(system_prompt, layer_states=sys_states, return_states=True)

    # RUN QUERIES
    print(f"\n{'='*70}")
    print(f"RUNNING {args.n_queries} QUERIES (snapshot query cache → IVFADC → install → answer)")
    print(f"{'='*70}")
    ivfadc_hits = 0
    correct_higher = 0
    logit_diffs = []
    t_total = 0
    for q_idx in range(args.n_queries):
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        topic_marker = 10 + q_topic * 20

        t0 = time.time()
        # 1. SNAPSHOT the query's cache
        query_cache_vec = snapshot_query_cache(model, query, device)
        # 2. IVFADC preselect + cos sim rerank on CACHE vectors
        retrieved = retrieve_by_cache(query_cache_vec, index, cache_vectors, top_k=args.top_k)
        # 3. INSTALL + answer
        logits = install_and_answer(model, query, sys_states, retrieved, pool, device)
        t_total += time.time() - t0

        # retrieval quality
        retrieved_topics = [chunk_topics[i] for i in retrieved]
        if q_topic in retrieved_topics:
            ivfadc_hits += 1

        # discriminative test: correct cache vs wrong cache
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        logits_correct = install_and_answer(model, query, sys_states, gold_idxs, pool, device)
        wrong_chunks = [i for i, t in enumerate(chunk_topics) if t != q_topic]
        wrong_idxs = random.sample(wrong_chunks, 3)
        logits_wrong = install_and_answer(model, query, sys_states, wrong_idxs, pool, device)
        correct_logit = logits_correct[:, -1, topic_marker].item()
        wrong_logit = logits_wrong[:, -1, topic_marker].item()
        if correct_logit > wrong_logit:
            correct_higher += 1
        logit_diffs.append(correct_logit - wrong_logit)

        if (q_idx + 1) % 10 == 0:
            print(f"  Query {q_idx+1}/{args.n_queries}: topic={q_topic}, retrieved={retrieved_topics}, match={'✓' if q_topic in retrieved_topics else '✗'}")

    # RESULTS
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Retrieval (IVFADC on cache vectors):  {ivfadc_hits}/{args.n_queries} = {ivfadc_hits/args.n_queries*100:.1f}%")
    print(f"  Correct cache > wrong cache:          {correct_higher}/{args.n_queries} = {correct_higher/args.n_queries*100:.1f}%")
    print(f"  Mean logit diff (correct - wrong):    {np.mean(logit_diffs):.4f}")
    print(f"  Avg query latency:                    {t_total/args.n_queries*1000:.1f} ms")
    print(f"  Cache vector size:                     {cache_vec_size}")
    print(f"  Disk per chunk:                        {sum(os.path.getsize(os.path.join(snapshots_dir, f)) for f in os.listdir(snapshots_dir))/len(chunks):.0f} B")

    results = {
        'retrieval_ivfadc': ivfadc_hits / args.n_queries,
        'correct_higher_than_wrong': correct_higher / args.n_queries,
        'mean_logit_diff': float(np.mean(logit_diffs)),
        'avg_query_ms': t_total / args.n_queries * 1000,
        'cache_vector_size': cache_vec_size,
        'n_chunks': len(chunks),
        'n_queries': args.n_queries,
        'pretrain_steps': args.pretrain_steps,
    }
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
