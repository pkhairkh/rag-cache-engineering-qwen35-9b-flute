#!/usr/bin/env python3
"""toy_bottleneck_analysis.py — find the bottlenecks, propose fixes.

Three diagnostics:
1. RETRIEVAL bottleneck: test 3 snapshot vectors (mean-pool, last-token, max-pool).
   Which gives the best IVFADC precision?
2. ANSWER bottleneck: when retrieval is PERFECT (GOLD), how good is the answer?
   The gap between GOLD and no-cache tells us if the model USES the caches.
3. FINE-TUNE bottleneck: train the model's M1/M2/read-write gates on the
   cache-installation task. Does answer quality improve?

Then propose the fix that stays in the architecture.
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
from toy_1000_chunks import (
    N_TOPICS, CHUNKS_PER_TOPIC, gen_topic_chunks, gen_topic_query,
    CacheSnapshot, ingest_and_snapshot, build_ivfadc, SnapshotPool,
    install_and_answer, embed_query, ivfadc_retrieve,
)
from toy_kimi_two_caches import TinyHybridModelWithKimiCaches


# ---------------------------------------------------------------------------
# Diagnostic 1: snapshot vector variants
# ---------------------------------------------------------------------------

def extract_vector(model, chunk, mode='mean', device='cpu'):
    """Extract a snapshot vector in 3 ways:
    - 'mean': mean-pool the hidden state over the sequence (current)
    - 'last': the last token's hidden state
    - 'max': max-pool over the sequence"""
    with torch.no_grad():
        logits, _, _, _, _ = model(chunk, return_states=True)
        # logits shape: (1, T, vocab). Use the hidden state (pre-lm_head).
        # In the toy, logits IS the hidden (no separate hidden). Use it.
        if mode == 'mean':
            return logits.mean(dim=1).squeeze(0)
        elif mode == 'last':
            return logits[:, -1, :].squeeze(0)
        elif mode == 'max':
            return logits.max(dim=1)[0].squeeze(0)


def test_retrieval_vectors(model, chunks, chunk_topics, queries, query_topics,
                            device, modes=['mean', 'last', 'max'], top_k=3):
    """Test IVFADC retrieval precision with different snapshot vectors."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 1: Retrieval precision by snapshot vector")
    print(f"{'='*70}")
    results = {}
    for mode in modes:
        # build vectors with this mode
        vectors = []
        for chunk in chunks:
            v = extract_vector(model, chunk, mode=mode, device=device)
            vectors.append(v.cpu().numpy().astype(np.float32))
        vectors = np.stack(vectors)
        index = build_ivfadc(vectors, nlist=32, m=8, nprobe=8)
        # run queries
        hits = 0
        for q_idx, query in enumerate(queries):
            q_topic = query_topics[q_idx]
            v_q = extract_vector(model, query, mode=mode, device=device).cpu().numpy().astype(np.float32)
            retrieved = ivfadc_retrieve(model, query, index, vectors, top_k=top_k)
            retrieved_topics = [chunk_topics[i] for i in retrieved]
            if q_topic in retrieved_topics:
                hits += 1
        rate = hits / len(queries)
        results[mode] = rate
        print(f"  {mode:5s}-pool: {hits}/{len(queries)} = {rate*100:.1f}%")
    return results


# ---------------------------------------------------------------------------
# Diagnostic 2: answer quality with PERFECT retrieval (GOLD)
# ---------------------------------------------------------------------------

def test_gold_answer_quality(model, chunks, chunk_topics, queries, query_topics,
                              device, system_prompt, n_queries=100, top_k=3):
    """When retrieval is perfect (GOLD), how good is the answer?
    The gap between GOLD and no-cache tells us if the model USES the caches."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 2: Answer quality with PERFECT retrieval (GOLD)")
    print(f"{'='*70}")
    # ingest
    snapshots_dir = "/tmp/diag2_snaps"
    if os.path.exists(snapshots_dir):
        shutil.rmtree(snapshots_dir)
    vectors = ingest_and_snapshot(model, chunks, device, snapshots_dir)
    pool = SnapshotPool(snapshots_dir, max_in_memory=256, device=device)
    # system prompt caches
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    with torch.no_grad():
        _, sys_S, sys_conv, sys_M1, sys_M2 = model(system_prompt, S=S0, conv_state=conv0,
                                                     M1_state=M1_0, M2_state=M2_0, return_states=True)
    # for each query: install GOLD (3 chunks from correct topic) vs no-cache
    gold_diffs = []  # GOLD vs no-cache
    for q_idx in range(n_queries):
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        # GOLD: 3 random chunks from the correct topic
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, min(top_k, len(correct_chunks)))
        logits_gold = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                          gold_idxs, pool, device)
        logits_no = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                         [], pool, device)
        diff = (logits_gold - logits_no).abs().max().item()
        gold_diffs.append(diff)
    mean_diff = np.mean(gold_diffs)
    print(f"  GOLD vs NO-CACHE mean diff: {mean_diff:.4f}")
    print(f"  (If small: the model barely uses the installed caches — the bottleneck is the model.)")
    print(f"  (If large: the model DOES use the caches — the bottleneck is retrieval.)")
    return mean_diff


# ---------------------------------------------------------------------------
# Diagnostic 3: fine-tune the model to use caches
# ---------------------------------------------------------------------------

def install_and_answer_grad(model, query_tokens, sys_S, sys_conv, sys_M1, sys_M2,
                             retrieved_idxs, pool, device):
    """Same as install_and_answer but WITHOUT torch.no_grad (for fine-tuning).
    Loads snapshots (detached — they're frozen), sums deltas, runs the model
    forward WITH gradient flow through M1/M2/read-write gates."""
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
            # detach the snapshot (frozen), but the model's forward is grad-enabled
            restored_S = restored_S + snap.delta_S.detach()
            restored_M1 = restored_M1 + snap.delta_M1.detach()
            restored_M2 = restored_M2 + snap.delta_M2.detach()
            last_snap = snap
        restored_conv = last_snap.conv_state.detach().clone() if last_snap else sys_conv
    # grad-enabled forward
    logits, _, _, _, _ = model(query_tokens,
                                S=restored_S, conv_state=restored_conv,
                                M1_state=restored_M1, M2_state=restored_M2)
    return logits


def fine_tune_cache_usage(model, chunks, chunk_topics, queries, query_topics,
                          device, system_prompt, n_steps=300, lr=1e-3):
    """Fine-tune the model's M1/M2/read-write gates on the cache-installation task.
    Uses install_and_answer_grad (gradient-enabled forward)."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 3: Fine-tune the model to use caches ({n_steps} steps)")
    print(f"{'='*70}")
    snapshots_dir = "/tmp/diag3_snaps"
    if os.path.exists(snapshots_dir):
        shutil.rmtree(snapshots_dir)
    vectors = ingest_and_snapshot(model, chunks, device, snapshots_dir)
    pool = SnapshotPool(snapshots_dir, max_in_memory=256, device=device)
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    with torch.no_grad():
        _, sys_S, sys_conv, sys_M1, sys_M2 = model(system_prompt, S=S0, conv_state=conv0,
                                                     M1_state=M1_0, M2_state=M2_0, return_states=True)

    optimizer = torch.optim.AdamW([
        {'params': model.linear_attn.M1, 'lr': lr},
        {'params': model.linear_attn.M2, 'lr': lr},
        {'params': model.linear_attn.mem_write_gate.parameters(), 'lr': lr},
        {'params': model.linear_attn.mem_read_gate.parameters(), 'lr': lr},
    ], lr=lr)
    model.train()
    losses = []
    for step in range(n_steps):
        q_idx = step % len(queries)
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        correct_idxs = random.sample(correct_chunks, 3)
        logits_correct = install_and_answer_grad(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                                  correct_idxs, pool, device)
        wrong_chunks = [i for i, t in enumerate(chunk_topics) if t != q_topic]
        wrong_idxs = random.sample(wrong_chunks, 3)
        logits_wrong = install_and_answer_grad(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                                wrong_idxs, pool, device)
        # contrastive: maximize divergence (negative MSE)
        loss = -F.mse_loss(logits_correct, logits_wrong)
        # confidence: reduce entropy of the correct answer
        probs = F.softmax(logits_correct[:, -1, :], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        loss = loss + 0.1 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    model.eval()
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/bottleneck_results.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    n_chunks = 1000
    n_queries = 100
    top_k = 3
    chunk_len, query_len, system_len = 32, 16, 16
    vocab = 256

    model = TinyHybridModelWithKimiCaches(hidden=32, vocab=vocab, mem_size=16).to(device)
    model.eval()

    # generate chunks and queries
    chunks, chunk_topics = gen_topic_chunks(N_TOPICS, CHUNKS_PER_TOPIC, chunk_len, vocab, seed=42)
    queries = [gen_topic_query(q % N_TOPICS, query_len, vocab, seed=q) for q in range(n_queries)]
    query_topics = [q % N_TOPICS for q in range(n_queries)]
    system_prompt = torch.randint(0, vocab, (1, system_len))

    # DIAGNOSTIC 1: retrieval vectors
    retrieval_results = test_retrieval_vectors(model, chunks, chunk_topics, queries, query_topics,
                                                device, modes=['mean', 'last', 'max'], top_k=top_k)

    # DIAGNOSTIC 2: gold answer quality
    gold_diff = test_gold_answer_quality(model, chunks, chunk_topics, queries, query_topics,
                                          device, system_prompt, n_queries=n_queries, top_k=top_k)

    # DIAGNOSTIC 3: fine-tune
    print(f"\n[Before fine-tune] measuring baseline answer quality...")
    # re-measure the IVFADC vs GOLD diff before fine-tune
    snapshots_dir = "/tmp/before_ft"
    if os.path.exists(snapshots_dir):
        shutil.rmtree(snapshots_dir)
    vectors = ingest_and_snapshot(model, chunks, device, snapshots_dir)
    index = build_ivfadc(vectors, nlist=32, m=8, nprobe=8)
    pool = SnapshotPool(snapshots_dir, max_in_memory=256, device=device)
    S0, conv0, M1_0, M2_0 = model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    with torch.no_grad():
        _, sys_S, sys_conv, sys_M1, sys_M2 = model(system_prompt, S=S0, conv_state=conv0,
                                                     M1_state=M1_0, M2_state=M2_0, return_states=True)
    before_diffs = []
    for q_idx in range(n_queries):
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        retrieved = ivfadc_retrieve(model, query, index, vectors, top_k=top_k)
        logits_ivf = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                          retrieved, pool, device)
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        logits_gold = install_and_answer(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                          gold_idxs, pool, device)
        before_diffs.append((logits_ivf - logits_gold).abs().max().item())
    before_mean = np.mean(before_diffs)
    print(f"  Before FT: IVFADC vs GOLD mean diff = {before_mean:.4f}")

    # fine-tune
    losses = fine_tune_cache_usage(model, chunks, chunk_topics, queries, query_topics,
                                    device, system_prompt, n_steps=300, lr=1e-3)

    # re-measure after fine-tune
    print(f"\n[After fine-tune] measuring answer quality...")
    snapshots_dir2 = "/tmp/after_ft"
    if os.path.exists(snapshots_dir2):
        shutil.rmtree(snapshots_dir2)
    vectors2 = ingest_and_snapshot(model, chunks, device, snapshots_dir2)
    index2 = build_ivfadc(vectors2, nlist=32, m=8, nprobe=8)
    pool2 = SnapshotPool(snapshots_dir2, max_in_memory=256, device=device)
    with torch.no_grad():
        _, sys_S2, sys_conv2, sys_M12, sys_M22 = model(system_prompt, S=S0, conv_state=conv0,
                                                         M1_state=M1_0, M2_state=M2_0, return_states=True)
    after_diffs = []
    after_hits = 0
    for q_idx in range(n_queries):
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        retrieved = ivfadc_retrieve(model, query, index2, vectors2, top_k=top_k)
        retrieved_topics = [chunk_topics[i] for i in retrieved]
        if q_topic in retrieved_topics:
            after_hits += 1
        logits_ivf = install_and_answer(model, query, sys_S2, sys_conv2, sys_M12, sys_M22,
                                         retrieved, pool2, device)
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        logits_gold = install_and_answer(model, query, sys_S2, sys_conv2, sys_M12, sys_M22,
                                          gold_idxs, pool2, device)
        after_diffs.append((logits_ivf - logits_gold).abs().max().item())
    after_mean = np.mean(after_diffs)
    after_rate = after_hits / n_queries
    print(f"  After FT:  IVFADC vs GOLD mean diff = {after_mean:.4f}")
    print(f"  After FT:  retrieval hit rate = {after_rate*100:.1f}%")

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print(f"BOTTLENECK ANALYSIS SUMMARY")
    print(f"{'='*70}")
    results = {
        'diagnostic_1_retrieval_vectors': retrieval_results,
        'diagnostic_2_gold_vs_no_cache': float(gold_diff),
        'diagnostic_3_before_ft_ivfadc_vs_gold': float(before_mean),
        'diagnostic_3_after_ft_ivfadc_vs_gold': float(after_mean),
        'diagnostic_3_after_ft_retrieval_rate': float(after_rate),
        'diagnostic_3_ft_improved_answer': bool(after_mean < before_mean),
        'diagnostic_3_ft_loss_first': losses[0] if losses else None,
        'diagnostic_3_ft_loss_last': losses[-1] if losses else None,
    }
    for k, v in results.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*70}")
    print(f"WHERE IS THE BOTTLENECK?")
    print(f"{'='*70}")
    print(f"  1. Retrieval vector: best mode = {max(retrieval_results, key=retrieval_results.get)}")
    print(f"     ({retrieval_results[max(retrieval_results, key=retrieval_results.get)]*100:.1f}%)")
    print(f"  2. Model uses caches: GOLD vs NO-CACHE diff = {gold_diff:.4f}")
    if gold_diff < 0.3:
        print(f"     → SMALL: the model barely uses the installed caches. BOTTLENECK = model training.")
    else:
        print(f"     → LARGE: the model DOES use caches. BOTTLENECK = retrieval precision.")
    print(f"  3. Fine-tune effect: {before_mean:.4f} → {after_mean:.4f}")
    if after_mean < before_mean:
        print(f"     → IMPROVED. Fine-tuning the read/write gates helps the model use caches.")
    else:
        print(f"     → NO IMPROVEMENT. The contrastive loss may not be the right objective.")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
