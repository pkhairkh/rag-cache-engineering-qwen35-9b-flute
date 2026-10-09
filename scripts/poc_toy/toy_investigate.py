#!/usr/bin/env python3
"""toy_investigate.py — investigate WHY the cache-installed answer fails.

NO RE-PREFILL. NO GROUND TRUTH FROM FULL PREFILL.
The ground truth is RELATIVE: the correct cache should make the topic-marker
logit HIGHER than the wrong cache. This measures whether the cache carries
discriminative information — without ever running the chunk text again.

Investigations:
1. WRITE STRENGTH: how much does M1/M2 actually change during ingestion?
   (If deltas are tiny, the cache carries no info.)
2. READ STRENGTH: when the query runs with restored M1/M2, is the attention
   over M1 peaked (selective) or uniform (noise)?
3. DISCRIMINATIVE TEST: does the topic-marker logit increase when the
   correct-topic cache is installed vs wrong-topic? (The real test — no re-prefill.)
4. FIX: if the write/read is too weak, strengthen the mechanism
   (bigger write scale, less saturated gates, etc.)
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
from toy_kimi_two_caches import LinearAttnWithKimiCaches, TinyFullAttention


# ---------------------------------------------------------------------------
# The model (same as scaled_up, but we'll instrument the write/read)
# ---------------------------------------------------------------------------

class InstrumentedLinearAttn(LinearAttnWithKimiCaches):
    """Fixed: SELECTIVE write (write to the best-matching slot, not all slots).
    This makes the slots differentiated → the read attention becomes peaked."""
    def __init__(self, *args, write_scale=1.0, read_scale=1.0, selective_write=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_scale = write_scale
        self.read_scale = read_scale
        self.selective_write = selective_write
        self.last_write_mag = 0.0
        self.last_read_attn_entropy = 0.0

    def forward_step(self, x, S, conv_state, M1_state, M2_state):
        B = x.shape[0]
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        new_conv = torch.cat([conv_state[:, :, 1:], mixed], dim=2)
        mixed = F.conv1d(new_conv, self.conv1d.weight, None, groups=self.conv_dim).squeeze(-1)
        mixed = F.silu(mixed)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, self.num_k_heads, self.head_k_dim)
        k = k.view(B, self.num_k_heads, self.head_k_dim)
        v = v.view(B, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads > self.num_k_heads:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        beta = torch.sigmoid(self.in_proj_b(x).view(B, self.num_v_heads, 1))
        a = self.in_proj_a(x).view(B, self.num_v_heads, 1)
        g_per_head = -self.A_log.exp().float() * F.softplus(a.float().squeeze(-1) + self.dt_bias)
        decay = torch.exp(g_per_head).view(B, self.num_v_heads, 1, 1)
        beta_factor = beta.unsqueeze(-1)
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        new_S = decay * S + beta_factor * v.unsqueeze(2) * k_norm.unsqueeze(3)
        o_delta = (new_S * q.view(B, self.num_v_heads, self.head_k_dim, 1)).sum(dim=2)

        # WRITE: SELECTIVE — compute slot attention (which slot to write to)
        mem_write = torch.sigmoid(self.mem_write_gate(x).view(B, self.num_v_heads, 1))
        # compute write attention: which slot's key matches the input k most?
        # M1_state shape: (B, H, mem_size, head_k_dim)
        # k_norm shape: (B, H, head_k_dim) → unsqueeze(2) → (B, H, 1, head_k_dim)
        write_scores = (M1_state * k_norm.unsqueeze(2)).sum(-1)  # (B, H, mem_size)
        write_attn = F.softmax(write_scores / math.sqrt(self.head_k_dim), dim=-1)  # (B, H, mem_size)

        if self.selective_write:
            # selective: write to the best-matching slot (soft assignment)
            # mem_write: (B, H, 1) → unsqueeze(-1): (B, H, 1, 1)
            # k_norm.unsqueeze(2) * write_attn.unsqueeze(-1): (B, H, 1, head_k) * (B, H, mem_size, 1) = (B, H, mem_size, head_k)
            new_M1 = M1_state + self.write_scale * mem_write.unsqueeze(-1) * (
                k_norm.unsqueeze(2) * write_attn.unsqueeze(-1))
            new_M2 = M2_state + self.write_scale * mem_write.unsqueeze(-1) * (
                v.unsqueeze(2) * write_attn.unsqueeze(-1))
        else:
            # old: write to all slots equally (uniform → no differentiation)
            new_M1 = M1_state + self.write_scale * mem_write.unsqueeze(-1) * k_norm.unsqueeze(2)
            new_M2 = M2_state + self.write_scale * mem_write.unsqueeze(-1) * v.unsqueeze(2)
        self.last_write_mag = (new_M1 - M1_state).abs().mean().item()

        # READ: the query attends to M1's keys
        mem_read = torch.sigmoid(self.mem_read_gate(x).view(B, self.num_v_heads, 1)) * self.read_scale
        mem_scores = q.view(B, self.num_v_heads, 1, self.head_k_dim) @ new_M1.transpose(-1, -2)
        mem_attn = F.softmax(mem_scores / math.sqrt(self.head_k_dim), dim=-1)
        attn_probs = mem_attn.squeeze(2)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(-1).mean().item()
        self.last_read_attn_entropy = entropy
        o_mem = (mem_attn @ new_M2).squeeze(2)
        o = o_delta + mem_read * o_mem

        o = self.norm(o.view(B, -1, self.head_v_dim))
        z = self.in_proj_z(x).view(B, 1, self.value_dim)
        o_flat = o.reshape(B, 1, self.value_dim)
        o_flat = o_flat * F.silu(z.float()).to(o.dtype)
        out = self.out_proj(o_flat.view(B, -1))
        return out, new_S, new_conv, new_M1, new_M2


class ScaledHybridModel(nn.Module):
    def __init__(self, hidden=128, vocab=512, num_linear_layers=4, num_v_heads=8,
                 head_k_dim=16, head_v_dim=16, mem_size=32, write_scale=1.0, read_scale=1.0,
                 selective_write=True):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.num_linear_layers = num_linear_layers
        self.embed = nn.Embedding(vocab, hidden)
        self.linear_layers = nn.ModuleList([
            InstrumentedLinearAttn(hidden=hidden, num_v_heads=num_v_heads, num_k_heads=num_v_heads,
                                    head_k_dim=head_k_dim, head_v_dim=head_v_dim, mem_size=mem_size,
                                    write_scale=write_scale, read_scale=read_scale,
                                    selective_write=selective_write)
            for _ in range(num_linear_layers)
        ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_linear_layers)])
        self.linear_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden))
            for _ in range(num_linear_layers)
        ])
        self.linear_mlp_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_linear_layers)])
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
        states = []
        for layer in self.linear_layers:
            S, conv, M1, M2 = layer.initial_state(batch)
            states.append((S.to(device), conv.to(device), M1.to(device), M2.to(device)))
        return states


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
# Snapshot + pool (from scaled_up)
# ---------------------------------------------------------------------------

@dataclass
class LayeredSnapshot:
    delta_S_list: List[torch.Tensor]
    delta_M1_list: List[torch.Tensor]
    delta_M2_list: List[torch.Tensor]
    conv_state_list: List[torch.Tensor]
    hidden: torch.Tensor


def ingest_and_snapshot(model, chunks, device, snapshots_dir):
    os.makedirs(snapshots_dir, exist_ok=True)
    init_states = model.initial_states(1, device)
    vectors = []
    write_mags = []  # track write strength
    for idx, chunk in enumerate(chunks):
        with torch.no_grad():
            logits, new_states = model(chunk, layer_states=init_states, return_states=True)
            hidden = logits.mean(dim=1)
        delta_S_list, delta_M1_list, delta_M2_list, conv_state_list = [], [], [], []
        for i, (new_S, new_conv, new_M1, new_M2) in enumerate(new_states):
            init_S, init_conv, init_M1, init_M2 = init_states[i]
            delta_S_list.append((new_S - init_S).detach().cpu().numpy())
            delta_M1_list.append((new_M1 - init_M1).detach().cpu().numpy())
            delta_M2_list.append((new_M2 - init_M2).detach().cpu().numpy())
            conv_state_list.append(new_conv.detach().cpu().numpy())
        # record write magnitude (layer 0)
        write_mags.append(model.linear_layers[0].last_write_mag)
        snap_path = os.path.join(snapshots_dir, f"chunk_{idx:05d}.npz")
        save_dict = {'hidden': hidden.detach().cpu().numpy()}
        for i in range(len(delta_S_list)):
            save_dict[f'delta_S_{i}'] = delta_S_list[i]
            save_dict[f'delta_M1_{i}'] = delta_M1_list[i]
            save_dict[f'delta_M2_{i}'] = delta_M2_list[i]
            save_dict[f'conv_state_{i}'] = conv_state_list[i]
        np.savez(snap_path, **save_dict)
        vectors.append(hidden.squeeze(0).detach().cpu().numpy())
    vectors = np.stack(vectors).astype(np.float32)
    return vectors, write_mags


class LayeredPool:
    def __init__(self, snapshots_dir, num_layers, max_in_memory=128, device='cpu'):
        self.snapshots_dir = snapshots_dir
        self.num_layers = num_layers
        self.max_in_memory = max_in_memory
        self.device = device
        self.in_memory: OrderedDict[str, LayeredSnapshot] = OrderedDict()
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
        snap = LayeredSnapshot(
            delta_S_list=[torch.from_numpy(data[f'delta_S_{i}']).to(self.device)
                          for i in range(self.num_layers)],
            delta_M1_list=[torch.from_numpy(data[f'delta_M1_{i}']).to(self.device)
                           for i in range(self.num_layers)],
            delta_M2_list=[torch.from_numpy(data[f'delta_M2_{i}']).to(self.device)
                           for i in range(self.num_layers)],
            conv_state_list=[torch.from_numpy(data[f'conv_state_{i}']).to(self.device)
                             for i in range(self.num_layers)],
            hidden=torch.from_numpy(data['hidden']).to(self.device),
        )
        while len(self.in_memory) >= self.max_in_memory:
            self.in_memory.popitem(last=False)
        self.in_memory[key] = snap
        return snap


def install_and_answer(model, query, sys_states, retrieved_idxs, pool, device):
    if len(retrieved_idxs) == 0:
        restored_states = sys_states
    else:
        restored_states = []
        for i in range(model.num_linear_layers):
            r_S = sys_states[i][0].clone()
            r_M1 = sys_states[i][2].clone()
            r_M2 = sys_states[i][3].clone()
            last_conv = sys_states[i][1]
            for idx in retrieved_idxs:
                snap = pool.lookup(idx)
                if snap is None:
                    continue
                r_S = r_S + snap.delta_S_list[i]
                r_M1 = r_M1 + snap.delta_M1_list[i]
                r_M2 = r_M2 + snap.delta_M2_list[i]
                last_conv = snap.conv_state_list[i]
            restored_states.append((r_S, last_conv, r_M1, r_M2))
    with torch.no_grad():
        logits, _ = model(query, layer_states=restored_states, return_states=True)
    return logits


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_ivfadc_caches import SimpleIVFADC

def build_ivfadc(vectors, nlist=32, m=8, nprobe=8):
    index = SimpleIVFADC(nlist=min(nlist, len(vectors)), m=m, nprobe=min(nprobe, nlist))
    index.build(vectors, list(range(len(vectors))))
    return index


# ---------------------------------------------------------------------------
# THE INVESTIGATION (no re-prefill, no full-prefill ground truth)
# ---------------------------------------------------------------------------

def investigate(model, chunks, chunk_topics, queries, query_topics, device,
               system_prompt, pool, sys_states, top_k=5, n_queries=50):
    """Investigate the cache mechanism. NO RE-PREFILL.

    The test: for each query, install (a) correct-topic cache, (b) wrong-topic cache,
    (c) no cache. Measure the topic-marker logit [10 + topic*20] under each.
    The cache carries discriminative info IF the correct cache makes the
    topic-marker logit HIGHER than the wrong cache."""

    # build IVFADC
    init_states = model.initial_states(1, device)
    vectors = []
    read_entropies = []
    for chunk in chunks:
        with torch.no_grad():
            logits, _ = model(chunk, layer_states=init_states, return_states=True)
        vectors.append(logits.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32))
    vectors = np.stack(vectors)
    index = build_ivfadc(vectors, nlist=32, m=8, nprobe=8)

    # for each query, measure the topic-marker logit under 3 conditions
    correct_higher = 0  # correct cache > wrong cache
    correct_vs_no = 0    # correct cache > no cache
    logit_diffs = []     # (correct_logit - wrong_logit) per query
    for q_idx in range(n_queries):
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        topic_marker = 10 + q_topic * 20
        # embed query for IVFADC
        with torch.no_grad():
            q_logits, _ = model(query, layer_states=init_states, return_states=True)
        v_q = q_logits.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        # IVFADC retrieve
        candidates = index.search(v_q, k=min(100, len(vectors)))
        cand_vecs = vectors[candidates]
        cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
        q_norm = v_q / (np.linalg.norm(v_q) + 1e-8)
        scores = cand_norm @ q_norm
        top_k_local = np.argsort(scores)[-top_k:][::-1]
        retrieved = candidates[top_k_local].tolist()

        # (a) correct-topic cache (gold)
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        logits_correct = install_and_answer(model, query, sys_states, gold_idxs, pool, device)
        correct_logit = logits_correct[:, -1, topic_marker].item()

        # (b) wrong-topic cache
        wrong_chunks = [i for i, t in enumerate(chunk_topics) if t != q_topic]
        wrong_idxs = random.sample(wrong_chunks, 3)
        logits_wrong = install_and_answer(model, query, sys_states, wrong_idxs, pool, device)
        wrong_logit = logits_wrong[:, -1, topic_marker].item()

        # (c) no cache
        logits_no = install_and_answer(model, query, sys_states, [], pool, device)
        no_logit = logits_no[:, -1, topic_marker].item()

        if correct_logit > wrong_logit:
            correct_higher += 1
        if correct_logit > no_logit:
            correct_vs_no += 1
        logit_diffs.append(correct_logit - wrong_logit)

    return {
        'correct_higher_than_wrong': correct_higher / n_queries,
        'correct_higher_than_no_cache': correct_vs_no / n_queries,
        'mean_logit_diff_correct_minus_wrong': float(np.mean(logit_diffs)),
        'retrieval_ivfadc': sum(1 for q_idx in range(n_queries)
                                  if query_topics[q_idx] in [chunk_topics[i] for i in
                                  candidates[np.argsort(scores)[-top_k:][::-1]].tolist()]) / n_queries if False else None,
    }


def pretrain_model(model, chunks, device, n_steps=500, lr=1e-3):
    """Pre-train the model on next-token prediction of the chunks.
    This trains the linear-attn layers (including M1/M2/read-write gates)
    to LEARN to use the caches for sequence modeling.
    After pre-training, the snapshots will carry discriminative info.

    This is NOT re-prefill. Re-prefill runs chunk text at QUERY TIME.
    Pre-training trains the MODEL WEIGHTS before snapshotting. Completely different."""
    print(f"\n[Pre-train] Training model on corpus ({n_steps} steps, next-token prediction)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(n_steps):
        # pick a random chunk
        chunk = random.choice(chunks)
        # next-token prediction: input = chunk[:-1], target = chunk[1:]
        inp = chunk[:, :-1]
        target = chunk[:, 1:]
        logits, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    model.eval()
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-scale', type=float, default=1.0)
    ap.add_argument('--read-scale', type=float, default=1.0)
    ap.add_argument('--pretrain-steps', type=int, default=1500)
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/investigate_results.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'

    # build the model with the given write/read scales
    model = ScaledHybridModel(hidden=128, vocab=512, num_linear_layers=4,
                               num_v_heads=8, head_k_dim=16, head_v_dim=16,
                               mem_size=32, write_scale=args.write_scale,
                               read_scale=args.read_scale).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, write_scale={args.write_scale}, read_scale={args.read_scale}")

    chunks_per_topic = 50  # 500 chunks
    chunks, chunk_topics = gen_topic_chunks(N_TOPICS, chunks_per_topic, 32, model.vocab, seed=42)
    queries = [gen_topic_query(q % N_TOPICS, 16, model.vocab, seed=q) for q in range(50)]
    query_topics = [q % N_TOPICS for q in range(50)]
    system_prompt = torch.randint(0, model.vocab, (1, 16))

    # PRE-TRAIN the model on the corpus (so the linear-attn learns to use caches)
    pretrain_losses = pretrain_model(model, chunks, device, n_steps=args.pretrain_steps, lr=1e-3)

    # ingest (AFTER pre-training — the snapshots will carry trained info)
    print(f"\nSnapshotting {len(chunks)} chunks...")
    snapshots_dir = "/tmp/investigate_snaps"
    if os.path.exists(snapshots_dir): shutil.rmtree(snapshots_dir)
    t0 = time.time()
    vectors, write_mags = ingest_and_snapshot(model, chunks, device, snapshots_dir)
    print(f"  Done: {time.time()-t0:.1f}s")
    print(f"  Write magnitude (layer 0, mean): {np.mean(write_mags):.6f}")
    print(f"  Write magnitude (layer 0, max):  {np.max(write_mags):.6f}")

    pool = LayeredPool(snapshots_dir, model.num_linear_layers, max_in_memory=128, device=device)
    sys_states = model.initial_states(1, device)
    with torch.no_grad():
        _, sys_states = model(system_prompt, layer_states=sys_states, return_states=True)

    # INVESTIGATE
    print(f"\n{'='*70}")
    print(f"INVESTIGATION (no re-prefill, no full-prefill ground truth)")
    print(f"{'='*70}")
    results = investigate(model, chunks, chunk_topics, queries, query_topics, device,
                          system_prompt, pool, sys_states, top_k=5, n_queries=50)
    print(f"  Correct cache > wrong cache (topic-marker logit): {results['correct_higher_than_wrong']*100:.1f}%")
    print(f"  Correct cache > no cache:                          {results['correct_higher_than_no_cache']*100:.1f}%")
    print(f"  Mean logit diff (correct - wrong):                 {results['mean_logit_diff_correct_minus_wrong']:.4f}")

    # also measure the read attention entropy during a query
    print(f"\n[Read attention entropy] (low = peaked/selective, high = uniform/noise)")
    query = queries[0]
    gold_idxs = [i for i, t in enumerate(chunk_topics) if t == query_topics[0]][:3]
    install_and_answer(model, query, sys_states, gold_idxs, pool, device)
    for i, layer in enumerate(model.linear_layers):
        print(f"  Layer {i}: read attn entropy = {layer.last_read_attn_entropy:.4f} "
              f"(max possible = {math.log(32):.4f} for mem_size=32)")

    results['write_scale'] = args.write_scale
    results['read_scale'] = args.read_scale
    results['write_mag_mean'] = float(np.mean(write_mags))
    results['write_mag_max'] = float(np.max(write_mags))
    results['read_entropy_layer0'] = model.linear_layers[0].last_read_attn_entropy

    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    disc = results['correct_higher_than_wrong']
    if disc > 0.8:
        print(f"  ✓ Cache carries discriminative info ({disc*100:.0f}% correct > wrong)")
    elif disc > 0.5:
        print(f"  ~ Weak discriminative signal ({disc*100:.0f}%). Write/read needs strengthening.")
    else:
        print(f"  ✗ Cache does NOT carry discriminative info ({disc*100:.0f}%). Write/read too weak.")
    print(f"  Write magnitude: {results['write_mag_mean']:.6f} (mean)")
    print(f"  Read entropy: {results['read_entropy_layer0']:.4f} (lower = more selective)")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
