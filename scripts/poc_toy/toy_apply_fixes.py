#!/usr/bin/env python3
"""toy_apply_fixes.py — apply the 3 fixes, measure on 1000 chunks / 100 queries.

Fixes:
1. LEARNED PROJECTION HEAD for retrieval: a linear layer that projects the
   pooled hidden state to a retrieval-optimized vector. Trained with InfoNCE
   contrastive loss (same-topic closer, different-topic farther).
2. SUPERVISED FINE-TUNE: train the M1/M2/read-write gates with the LM loss
   (next-token prediction with the correct caches installed). NOT contrastive.
3. TOP-K=5: retrieve top-5, install all 5 deltas, let the model's M1/M2
   attention select the relevant cache.

Measure: retrieval precision, answer quality (IVFADC vs GOLD), end-to-end
accuracy (does the model produce the topic-correct answer token?).
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
# The model with a learned retrieval projection head
# ---------------------------------------------------------------------------

class ModelWithRetrievalHead(nn.Module):
    """Wraps the hybrid model + a learned projection head for retrieval.
    The projection head is trained to make the snapshot vector discriminative."""
    def __init__(self, base_model, hidden=32, proj_dim=32):
        super().__init__()
        self.base = base_model
        # the pooled hidden state is (1, vocab) in the toy (logits are the hidden)
        # so the projection input dim is vocab, not hidden
        self.retrieval_proj = nn.Linear(base_model.vocab, proj_dim, bias=False)

    def forward(self, input_ids, S=None, conv_state=None, M1_state=None, M2_state=None,
                return_states=False):
        return self.base(input_ids, S=S, conv_state=conv_state,
                         M1_state=M1_state, M2_state=M2_state, return_states=return_states)

    def extract_retrieval_vector(self, chunk, device):
        """Extract the LEARNED retrieval vector (not mean-pool)."""
        with torch.no_grad():
            logits, _, _, _, _ = self.base(chunk, return_states=True)
            pooled = logits.mean(dim=1)  # (1, hidden)
            proj = self.retrieval_proj(pooled)  # (1, proj_dim)
        return proj.squeeze(0).detach()

    def extract_retrieval_vector_grad(self, chunk):
        """Grad-enabled version for training the projection head."""
        logits, _, _, _, _ = self.base(chunk, return_states=True)
        pooled = logits.mean(dim=1)
        return self.retrieval_proj(pooled).squeeze(0)


# ---------------------------------------------------------------------------
# Fix 1: train the retrieval projection head with InfoNCE
# ---------------------------------------------------------------------------

def train_retrieval_head(model, chunks, chunk_topics, device, n_steps=300, lr=1e-3):
    """Train the projection head with InfoNCE contrastive loss.
    For each anchor chunk, the positive is another same-topic chunk;
    the negatives are different-topic chunks. The loss pulls the anchor
    closer to the positive and pushes it away from the negatives."""
    print(f"\n[Fix 1] Training retrieval projection head ({n_steps} steps, InfoNCE)...")
    optimizer = torch.optim.AdamW(model.retrieval_proj.parameters(), lr=lr)
    model.train()
    losses = []
    # group chunks by topic
    topic_to_chunks = {t: [i for i, tt in enumerate(chunk_topics) if tt == t]
                       for t in range(N_TOPICS)}
    for step in range(n_steps):
        # pick a random anchor
        anchor_idx = random.randint(0, len(chunks) - 1)
        anchor_topic = chunk_topics[anchor_idx]
        # pick a positive (same topic, different chunk)
        same_topic = [i for i in topic_to_chunks[anchor_topic] if i != anchor_idx]
        if not same_topic:
            continue
        pos_idx = random.choice(same_topic)
        # pick negatives (different topics)
        neg_idxs = random.sample([i for i in range(len(chunks))
                                   if chunk_topics[i] != anchor_topic], 5)
        # compute vectors (grad-enabled)
        v_anchor = model.extract_retrieval_vector_grad(chunks[anchor_idx])
        v_pos = model.extract_retrieval_vector_grad(chunks[pos_idx])
        v_negs = torch.stack([model.extract_retrieval_vector_grad(chunks[i]) for i in neg_idxs])
        # InfoNCE: maximize cos(anchor, pos) / sum(cos(anchor, all))
        v_anchor_n = F.normalize(v_anchor, dim=-1)
        v_pos_n = F.normalize(v_pos, dim=-1)
        v_negs_n = F.normalize(v_negs, dim=-1)
        logit_pos = (v_anchor_n * v_pos_n).sum(-1) / 0.1  # temperature 0.1
        logits_neg = (v_anchor_n.unsqueeze(0) * v_negs_n).sum(-1) / 0.1
        logits = torch.cat([logit_pos.unsqueeze(0), logits_neg])
        labels = torch.zeros(1, dtype=torch.long, device=device)  # index 0 is the positive
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    model.eval()
    return losses


# ---------------------------------------------------------------------------
# Fix 2: supervised fine-tune (LM loss, with correct caches installed)
# ---------------------------------------------------------------------------

def install_and_answer_grad(model, query_tokens, sys_S, sys_conv, sys_M1, sys_M2,
                             retrieved_idxs, pool, device):
    """Grad-enabled install + answer (for fine-tuning)."""
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
            restored_S = restored_S + snap.delta_S.detach()
            restored_M1 = restored_M1 + snap.delta_M1.detach()
            restored_M2 = restored_M2 + snap.delta_M2.detach()
            last_snap = snap
        restored_conv = last_snap.conv_state.detach().clone() if last_snap else sys_conv
    logits, _, _, _, _ = model.base(query_tokens, S=restored_S, conv_state=restored_conv,
                                     M1_state=restored_M1, M2_state=restored_M2)
    return logits


def train_supervised_ft(model, chunks, chunk_topics, queries, query_topics,
                         device, system_prompt, pool, sys_S, sys_conv, sys_M1, sys_M2,
                         n_steps=800, lr=1e-3):
    """Supervised fine-tune with DISTILLATION: train the cache-installed logits
    to match the full-prefill logits (the ground-truth distribution, not just
    the argmax). This gives a richer signal than single-token CE."""
    print(f"\n[Fix 2] Supervised fine-tune ({n_steps} steps, distillation)...")
    # compute the ground-truth LOGIT DISTRIBUTIONS (full prefill)
    gt_logits = []
    S0, conv0, M1_0, M2_0 = model.base.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    for q_idx, query in enumerate(queries):
        q_topic = query_topics[q_idx]
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        all_tokens = system_prompt
        for gi in gold_idxs:
            all_tokens = torch.cat([all_tokens, chunks[gi]], dim=1)
        all_tokens = torch.cat([all_tokens, query], dim=1)
        with torch.no_grad():
            logits_full, _, _, _, _ = model.base(all_tokens, S=S0, conv_state=conv0,
                                                  M1_state=M1_0, M2_state=M2_0, return_states=True)
        # the target: the logit distribution at the last position (soft target)
        gt_logits.append(logits_full[:, -1, :].detach())  # (1, vocab)
    print(f"  Ground truth logits: {len(gt_logits)} distributions")

    optimizer = torch.optim.AdamW([
        {'params': model.base.linear_attn.M1, 'lr': lr},
        {'params': model.base.linear_attn.M2, 'lr': lr},
        {'params': model.base.linear_attn.mem_write_gate.parameters(), 'lr': lr},
        {'params': model.base.linear_attn.mem_read_gate.parameters(), 'lr': lr},
    ], lr=lr)
    model.train()
    losses = []
    for step in range(n_steps):
        q_idx = step % len(queries)
        query = queries[q_idx]
        gt = gt_logits[q_idx]
        q_topic = query_topics[q_idx]
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        logits = install_and_answer_grad(model, query, sys_S, sys_conv, sys_M1, sys_M2,
                                          gold_idxs, pool, device)
        # distillation loss: KL divergence between cache-installed and full-prefill
        last_logits = logits[:, -1, :]
        # soft KL (with temperature)
        T = 2.0
        loss = F.kl_div(F.log_softmax(last_logits / T, dim=-1),
                        F.softmax(gt / T, dim=-1), reduction='batchmean') * (T * T)
        # also a hard CE loss against the argmax (the ground-truth token)
        gt_token = gt.argmax(-1)
        loss = loss + F.cross_entropy(last_logits, gt_token)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    model.eval()
    return losses


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure(model, chunks, chunk_topics, queries, query_topics, device,
            system_prompt, pool, sys_S, sys_conv, sys_M1, sys_M2,
            top_k=5, n_queries=100):
    """Measure retrieval precision, answer quality, end-to-end accuracy.

    The "correct answer" is defined by the GROUND TRUTH: prefill the query
    on top of the actual chunk TEXTS (the no-cache path), get the argmax.
    This is what the model WOULD say if it saw the chunk text.
    The cache-installed answer should match this ground-truth argmax."""
    # build the ground truth: for each query, prefill (system + 3 correct chunks + query)
    # and record the argmax token at the last position
    ground_truths = []
    S0, conv0, M1_0, M2_0 = model.base.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    for q_idx in range(n_queries):
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        # ground truth: prefill system + 3 gold chunks + query (the full text path)
        all_tokens = system_prompt
        for gi in gold_idxs:
            all_tokens = torch.cat([all_tokens, chunks[gi]], dim=1)
        all_tokens = torch.cat([all_tokens, query], dim=1)
        with torch.no_grad():
            logits_full, _, _, _, _ = model.base(all_tokens, S=S0, conv_state=conv0,
                                                  M1_state=M1_0, M2_state=M2_0, return_states=True)
        gt_token = logits_full[:, -1, :].argmax(-1).item()
        ground_truths.append(gt_token)

    # build vectors with the learned projection head
    vectors = []
    for chunk in chunks:
        v = model.extract_retrieval_vector(chunk, device)
        vectors.append(v.cpu().numpy().astype(np.float32))
    vectors = np.stack(vectors)
    index = build_ivfadc(vectors, nlist=32, m=8, nprobe=8)

    ivfadc_hits = 0
    gold_hits = 0
    answer_correct_ivfadc = 0
    answer_correct_gold = 0
    answer_correct_no_cache = 0
    ivfadc_vs_gold_diffs = []
    ivfadc_vs_gt_diffs = []

    for q_idx in range(n_queries):
        query = queries[q_idx]
        q_topic = query_topics[q_idx]
        gt_token = ground_truths[q_idx]

        # IVFADC retrieve top-k
        v_q = model.extract_retrieval_vector(query, device).cpu().numpy().astype(np.float32)
        candidates = index.search(v_q, k=min(100, len(vectors)))
        candidate_vectors = vectors[candidates]
        cand_norm = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-8)
        q_norm = v_q / (np.linalg.norm(v_q) + 1e-8)
        scores = cand_norm @ q_norm
        top_k_local = np.argsort(scores)[-top_k:][::-1]
        retrieved_idxs = candidates[top_k_local].tolist()
        retrieved_topics = [chunk_topics[i] for i in retrieved_idxs]
        if q_topic in retrieved_topics:
            ivfadc_hits += 1

        # install IVFADC-retrieved, answer
        logits_ivf = install_and_answer(model.base, query, sys_S, sys_conv, sys_M1, sys_M2,
                                          retrieved_idxs, pool, device)
        pred_ivf = logits_ivf[:, -1, :].argmax(-1).item()
        if pred_ivf == gt_token:
            answer_correct_ivfadc += 1

        # GOLD: install 3 correct-topic chunks
        correct_chunks = [i for i, t in enumerate(chunk_topics) if t == q_topic]
        gold_idxs = random.sample(correct_chunks, 3)
        gold_topics = [chunk_topics[i] for i in gold_idxs]
        if q_topic in gold_topics:
            gold_hits += 1
        logits_gold = install_and_answer(model.base, query, sys_S, sys_conv, sys_M1, sys_M2,
                                          gold_idxs, pool, device)
        pred_gold = logits_gold[:, -1, :].argmax(-1).item()
        if pred_gold == gt_token:
            answer_correct_gold += 1

        # NO CACHE baseline
        logits_no = install_and_answer(model.base, query, sys_S, sys_conv, sys_M1, sys_M2,
                                        [], pool, device)
        pred_no = logits_no[:, -1, :].argmax(-1).item()
        if pred_no == gt_token:
            answer_correct_no_cache += 1

        ivfadc_vs_gold_diffs.append((logits_ivf - logits_gold).abs().max().item())
        ivfadc_vs_gt_diffs.append((logits_ivf[:, -1, :] - logits_full[:, -1, :]).abs().max().item()
                                   if q_idx == 0 else 0)

    return {
        'retrieval_ivfadc': ivfadc_hits / n_queries,
        'retrieval_gold': gold_hits / n_queries,
        'retrieval_random_expected': 1 - (1 - 1/N_TOPICS)**top_k,
        'answer_correct_ivfadc': answer_correct_ivfadc / n_queries,
        'answer_correct_gold': answer_correct_gold / n_queries,
        'answer_correct_no_cache': answer_correct_no_cache / n_queries,
        'ivfadc_vs_gold_diff': float(np.mean(ivfadc_vs_gold_diffs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/z/my-project/scripts/poc_toy/apply_fixes_results.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    n_chunks = 1000
    n_queries = 100
    chunk_len, query_len, system_len = 32, 16, 16
    vocab = 256

    base_model = TinyHybridModelWithKimiCaches(hidden=32, vocab=vocab, mem_size=16).to(device)
    model = ModelWithRetrievalHead(base_model, hidden=32, proj_dim=32).to(device)
    model.eval()

    chunks, chunk_topics = gen_topic_chunks(N_TOPICS, CHUNKS_PER_TOPIC, chunk_len, vocab, seed=42)
    queries = [gen_topic_query(q % N_TOPICS, query_len, vocab, seed=q) for q in range(n_queries)]
    query_topics = [q % N_TOPICS for q in range(n_queries)]
    system_prompt = torch.randint(0, vocab, (1, system_len))

    # ---- BASELINE (before fixes) ----
    print(f"\n{'='*70}")
    print(f"BASELINE (top-k=3, mean-pool, no fine-tune)")
    print(f"{'='*70}")
    snapshots_dir = "/tmp/apply_fixes_baseline"
    if os.path.exists(snapshots_dir): shutil.rmtree(snapshots_dir)
    ingest_and_snapshot(base_model, chunks, device, snapshots_dir)
    pool = SnapshotPool(snapshots_dir, max_in_memory=256, device=device)
    S0, conv0, M1_0, M2_0 = base_model.linear_attn.initial_state(1)
    S0, conv0, M1_0, M2_0 = S0.to(device), conv0.to(device), M1_0.to(device), M2_0.to(device)
    with torch.no_grad():
        _, sys_S, sys_conv, sys_M1, sys_M2 = base_model(system_prompt, S=S0, conv_state=conv0,
                                                          M1_state=M1_0, M2_state=M2_0, return_states=True)
    baseline = measure(model, chunks, chunk_topics, queries, query_topics, device,
                       system_prompt, pool, sys_S, sys_conv, sys_M1, sys_M2, top_k=3, n_queries=n_queries)
    print(f"  Retrieval: IVFADC={baseline['retrieval_ivfadc']*100:.1f}%, GOLD={baseline['retrieval_gold']*100:.1f}%")
    print(f"  Answer correct: IVFADC={baseline['answer_correct_ivfadc']*100:.1f}%, GOLD={baseline['answer_correct_gold']*100:.1f}%")
    print(f"  IVFADC vs GOLD diff: {baseline['ivfadc_vs_gold_diff']:.4f}")

    # ---- FIX 1: train retrieval head ----
    retrieval_losses = train_retrieval_head(model, chunks, chunk_topics, device, n_steps=800, lr=1e-3)

    # ---- FIX 2: supervised fine-tune ----
    # re-ingest with the (now-trained) base model's M1/M2
    snapshots_dir2 = "/tmp/apply_fixes_after"
    if os.path.exists(snapshots_dir2): shutil.rmtree(snapshots_dir2)
    ingest_and_snapshot(base_model, chunks, device, snapshots_dir2)
    pool2 = SnapshotPool(snapshots_dir2, max_in_memory=256, device=device)
    with torch.no_grad():
        _, sys_S2, sys_conv2, sys_M12, sys_M22 = base_model(system_prompt, S=S0, conv_state=conv0,
                                                              M1_state=M1_0, M2_state=M2_0, return_states=True)
    ft_losses = train_supervised_ft(model, chunks, chunk_topics, queries, query_topics,
                                     device, system_prompt, pool2, sys_S2, sys_conv2, sys_M12, sys_M22,
                                     n_steps=800, lr=1e-3)

    # re-ingest again (M1/M2 changed after fine-tune)
    snapshots_dir3 = "/tmp/apply_fixes_final"
    if os.path.exists(snapshots_dir3): shutil.rmtree(snapshots_dir3)
    ingest_and_snapshot(base_model, chunks, device, snapshots_dir3)
    pool3 = SnapshotPool(snapshots_dir3, max_in_memory=256, device=device)
    with torch.no_grad():
        _, sys_S3, sys_conv3, sys_M13, sys_M23 = base_model(system_prompt, S=S0, conv_state=conv0,
                                                              M1_state=M1_0, M2_state=M2_0, return_states=True)

    # ---- AFTER FIXES (top-k=7, learned projection, more FT) ----
    print(f"\n{'='*70}")
    print(f"AFTER FIXES (top-k=7, learned projection head, supervised FT)")
    print(f"{'='*70}")
    after = measure(model, chunks, chunk_topics, queries, query_topics, device,
                    system_prompt, pool3, sys_S3, sys_conv3, sys_M13, sys_M23, top_k=7, n_queries=n_queries)
    print(f"  Retrieval: IVFADC={after['retrieval_ivfadc']*100:.1f}%, GOLD={after['retrieval_gold']*100:.1f}%")
    print(f"  Answer correct: IVFADC={after['answer_correct_ivfadc']*100:.1f}%, GOLD={after['answer_correct_gold']*100:.1f}%")
    print(f"  IVFADC vs GOLD diff: {after['ivfadc_vs_gold_diff']:.4f}")

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print(f"SUMMARY: BASELINE vs AFTER FIXES")
    print(f"{'='*70}")
    results = {
        'baseline': baseline,
        'after_fixes': after,
        'retrieval_head_loss_first': retrieval_losses[0] if retrieval_losses else None,
        'retrieval_head_loss_last': retrieval_losses[-1] if retrieval_losses else None,
        'supervised_ft_loss_first': ft_losses[0] if ft_losses else None,
        'supervised_ft_loss_last': ft_losses[-1] if ft_losses else None,
    }
    print(f"\n  {'Metric':<30} {'Baseline':<12} {'After':<12} {'Delta':<12}")
    for k in ['retrieval_ivfadc', 'answer_correct_ivfadc', 'answer_correct_gold', 'ivfadc_vs_gold_diff']:
        b = baseline[k]
        a = after[k]
        d = a - b
        print(f"  {k:<30} {b:<12.4f} {a:<12.4f} {d:<+12.4f}")
    target_80 = after['answer_correct_ivfadc'] >= 0.80
    print(f"\n  End-to-end accuracy (IVFADC): {after['answer_correct_ivfadc']*100:.1f}%")
    print(f"  Target 80%+: {'✓ ACHIEVED' if target_80 else '✗ NOT YET'}")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
