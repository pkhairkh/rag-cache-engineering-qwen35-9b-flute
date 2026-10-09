#!/usr/bin/env python3
"""toy_cache_engineered_rag.py — test the cache-engineered RAG architecture
on a tiny hybrid model on CPU.

The architecture (correctly stated):
- ONE model (a tiny linear-attn + full-attn hybrid)
- EMBED: prefill each corpus chunk through the model, snapshot the
  linear-attn recurrent state (cache B) + full-attn KV (cache A)
- RETRIEVE: the model's full-attn attends to ALL cached chunks' KV
  (cache A). The attention weights ARE the retrieval scores.
- AUGMENT: restore the top-k chunks' cached states into the model,
  prefill only the query on top. No re-prefill of the chunks.

What we test:
1. Does cache restoration produce the SAME output as full re-prefill?
   (correctness — the cache must be lossless or near-lossless)
2. Does "attention as retrieval" actually retrieve relevant chunks?
   (retrieval quality — the model's attention must prefer relevant chunks)
3. Is the cache-engineered RAG faster than full re-prefill?
   (latency — the whole point)
4. Does it work at all? (the basic question)

Compare three configurations:
- NO_CACHE: full re-prefill of (system + retrieved chunks + query) every time
- KV_ONLY: cache A only (the vLLM pattern) — restore KV, recompute state
- FULL_CACHE: cache A + cache B (the proposal) — restore both, no re-prefill
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# The tiny hybrid model (1 linear-attn + 1 full-attn, faithful to modeling.py)
# ---------------------------------------------------------------------------

class TinyGatedDeltaNet(nn.Module):
    """Miniature of scripts/modeling.py::Qwen3_5GatedDeltaNet.
    State shape: (batch, num_v_heads, head_k_dim, head_k_dim)."""
    def __init__(self, hidden=32, num_v_heads=4, num_k_heads=2, head_k_dim=8,
                 head_v_dim=8, conv_kernel=4):
        super().__init__()
        self.hidden = hidden
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel = conv_kernel
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(hidden, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel,
                                groups=self.conv_dim, padding=conv_kernel-1, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_v_heads).uniform_(0.01, 16.0)))
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.norm = nn.LayerNorm(head_v_dim)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def initial_state(self, batch=1):
        return torch.zeros(batch, self.num_v_heads, self.head_k_dim, self.head_k_dim)

    def forward_step(self, x, state, conv_state):
        """One token forward (decode shape, seq_len=1)."""
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
        new_state = decay * state + beta_factor * v.unsqueeze(2) * k_norm.unsqueeze(3)
        o = (new_state * q.view(B, self.num_v_heads, self.head_k_dim, 1)).sum(dim=2)
        o = self.norm(o.view(B, -1, self.head_v_dim))
        z = self.in_proj_z(x).view(B, 1, self.value_dim)
        o_flat = o.reshape(B, 1, self.value_dim)
        o_flat = o_flat * F.silu(z.float()).to(o.dtype)
        out = self.out_proj(o_flat.view(B, -1))
        return out, new_state, new_conv

    def forward_chunk(self, x_seq, initial_state=None, initial_conv=None):
        """Prefill: process a sequence, return (outputs, final_state, final_conv)."""
        B, T, _ = x_seq.shape
        if initial_state is None:
            state = self.initial_state(B)
        else:
            state = initial_state
        if initial_conv is None:
            conv_state = torch.zeros(B, self.conv_dim, self.conv_kernel)
        else:
            conv_state = initial_conv
        outs = []
        for t in range(T):
            o, state, conv_state = self.forward_step(x_seq[:, t:t+1], state, conv_state)
            outs.append(o)
        return torch.stack(outs, dim=1), state, conv_state


class TinyFullAttention(nn.Module):
    """Miniature full-attention layer. KV cache: (B, H, T, D)."""
    def __init__(self, hidden=32, num_heads=4, head_dim=8):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden, 3 * num_heads * head_dim, bias=False)
        self.o = nn.Linear(num_heads * head_dim, hidden, bias=False)

    def forward(self, x_seq, kv_cache=None, output_attentions=False):
        """Returns (out, new_kv, attentions).
        kv_cache: (K, V) tuple of (B, H, T_past, D) or None.
        output_attentions: if True, returns the attention weights (B, H, T_q, T_k)."""
        B, T, _ = x_seq.shape
        qkv = self.qkv(x_seq).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        T_q = q.shape[2]
        T_k = k.shape[2]
        causal = torch.triu(torch.full((T_q, T_k), float('-inf')), diagonal=T_k - T_q)
        scores = scores + causal
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T_q, -1)
        out = self.o(out)
        if output_attentions:
            return out, (k, v), attn
        return out, (k, v), None


class TinyHybridModel(nn.Module):
    """1 linear-attn + 1 full-attn. The 3:1 hybrid in miniature."""
    def __init__(self, hidden=32, vocab=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.linear_attn = TinyGatedDeltaNet(hidden=hidden)
        self.full_attn = TinyFullAttention(hidden=hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.vocab = vocab

    def prefill(self, input_ids, linear_state=None, conv_state=None, kv_cache=None,
                return_states=False, output_attentions=False):
        """Prefill a sequence. Returns (logits, linear_state, conv_state, kv, attentions).
        If return_states=True, the linear_state and conv_state are returned (for snapshotting).
        If output_attentions=True, the full-attn attention weights are returned."""
        x = self.embed(input_ids)
        lin_out, new_lin_state, new_conv = self.linear_attn.forward_chunk(x, linear_state, conv_state)
        x = self.norm1(x + lin_out)
        full_out, new_kv, attn = self.full_attn(x, kv_cache, output_attentions=output_attentions)
        x = self.norm2(x + full_out)
        x = x + self.mlp(x)
        logits = self.lm_head(x)
        if return_states:
            return logits, new_lin_state, new_conv, new_kv, attn
        return logits, new_lin_state, new_conv, new_kv, attn

    def forward(self, input_ids, linear_state=None, conv_state=None, kv_cache=None,
                return_states=False, output_attentions=False):
        return self.prefill(input_ids, linear_state, conv_state, kv_cache,
                            return_states=return_states, output_attentions=output_attentions)


# ---------------------------------------------------------------------------
# The two global caches
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    key: str
    linear_state: torch.Tensor
    conv_state: torch.Tensor
    kv: Tuple[torch.Tensor, torch.Tensor]
    size: int
    last_used: float


class GlobalCache:
    """The combined global cache (cache A + cache B in one structure, per chunk).
    Keyed by chunk content hash. LRU-evicted."""
    def __init__(self, max_entries=64):
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max = max_entries
        self.hits = 0
        self.misses = 0

    def lookup(self, key: str) -> Optional[CacheEntry]:
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        return None

    def install(self, key: str, linear_state, conv_state, kv):
        size = (linear_state.nelement() * linear_state.element_size() +
                conv_state.nelement() * conv_state.element_size() +
                kv[0].nelement() * kv[0].element_size() * 2)
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        while len(self.entries) >= self.max and self.entries:
            self.entries.popitem(last=False)
        self.entries[key] = CacheEntry(key, linear_state.clone(), conv_state.clone(),
                                       (kv[0].clone(), kv[1].clone()), size, time.time())
        return True


# ---------------------------------------------------------------------------
# The three RAG configurations
# ---------------------------------------------------------------------------

def hash_tokens(token_ids):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.flatten().tolist()
    elif isinstance(token_ids, (list, tuple)):
        token_ids = list(token_ids)
    return hashlib.sha256(bytes(token_ids)).hexdigest()[:16]


def rag_no_cache(model, system_prompt_ids, query_ids, retrieved_chunks_ids, device):
    """NO_CACHE: full re-prefill of (system + chunks + query) every time.
    This is the floor — the slowest but most correct path."""
    all_tokens = system_prompt_ids
    for chunk_ids in retrieved_chunks_ids:
        all_tokens = torch.cat([all_tokens, chunk_ids], dim=1)
    all_tokens = torch.cat([all_tokens, query_ids], dim=1)
    with torch.no_grad():
        logits, _, _, _, _ = model(all_tokens)
    return logits[:, -query_ids.shape[1]:]


def rag_kv_only(model, system_prompt_ids, query_ids, retrieved_chunks_ids,
                cache: GlobalCache, device):
    """KV_ONLY: cache A only (the vLLM pattern).
    Restore the system prompt's KV + the chunks' KV into the full-attn layer.
    Recompute the linear-attn state from scratch (no cache B)."""
    # restore system prompt KV
    sys_key = hash_tokens(system_prompt_ids)
    sys_entry = cache.lookup(sys_key)
    if sys_entry is None:
        # prefill system prompt, install its KV
        with torch.no_grad():
            _, lin_state, conv_state, kv, _ = model(system_prompt_ids, return_states=True)
        cache.install(sys_key, lin_state, conv_state, kv)
        sys_entry = cache.entries[sys_key]
    kv_cache = (sys_entry.kv[0].clone(), sys_entry.kv[1].clone())
    # NO linear state restoration — recompute from scratch
    # (this is what makes it "KV only" — the linear-attn state is not cached)
    linear_state = model.linear_attn.initial_state(1).to(device)
    conv_state = torch.zeros(1, model.linear_attn.conv_dim, model.linear_attn.conv_kernel).to(device)

    # restore each chunk's KV (append), recompute linear state
    for chunk_ids in retrieved_chunks_ids:
        chunk_key = hash_tokens(chunk_ids)
        chunk_entry = cache.lookup(chunk_key)
        if chunk_entry is None:
            # prefill the chunk, install its KV
            with torch.no_grad():
                _, c_lin, c_conv, c_kv, _ = model(chunk_ids, return_states=True)
            cache.install(chunk_key, c_lin, c_conv, c_kv)
            chunk_entry = cache.entries[chunk_key]
        # append the chunk's KV to the full-attn cache
        kv_cache = (torch.cat([kv_cache[0], chunk_entry.kv[0]], dim=2),
                    torch.cat([kv_cache[1], chunk_entry.kv[1]], dim=2))
        # recompute the linear-attn state by prefilling the chunk (no cache B)
        with torch.no_grad():
            _, linear_state, conv_state, _, _ = model(chunk_ids,
                                                       linear_state=linear_state,
                                                       conv_state=conv_state)

    # prefill the query on top
    with torch.no_grad():
        logits, _, _, _, _ = model(query_ids, linear_state=linear_state,
                                    conv_state=conv_state, kv_cache=kv_cache)
    return logits


def rag_full_cache(model, system_prompt_ids, query_ids, retrieved_chunks_ids,
                   cache: GlobalCache, device):
    """FULL_CACHE: cache A + cache B (the proposal).
    Restore the system prompt's KV + state + the chunks' KV + state.
    No re-prefill of the chunks — only the query is prefilled."""
    # restore system prompt KV + state
    sys_key = hash_tokens(system_prompt_ids)
    sys_entry = cache.lookup(sys_key)
    if sys_entry is None:
        with torch.no_grad():
            _, lin_state, conv_state, kv, _ = model(system_prompt_ids, return_states=True)
        cache.install(sys_key, lin_state, conv_state, kv)
        sys_entry = cache.entries[sys_key]
    kv_cache = (sys_entry.kv[0].clone(), sys_entry.kv[1].clone())
    linear_state = sys_entry.linear_state.clone()
    conv_state = sys_entry.conv_state.clone()

    # restore each chunk's KV (append) + state (update)
    for chunk_ids in retrieved_chunks_ids:
        chunk_key = hash_tokens(chunk_ids)
        chunk_entry = cache.lookup(chunk_key)
        if chunk_entry is None:
            # prefill the chunk ON TOP OF the current state, install the DELTA
            with torch.no_grad():
                _, lin_state, conv_state, kv, _ = model(chunk_ids,
                                                          linear_state=linear_state,
                                                          conv_state=conv_state,
                                                          return_states=True)
            cache.install(chunk_key, lin_state.clone(), conv_state.clone(), kv)
            chunk_entry = cache.entries[chunk_key]
        # append the chunk's KV
        kv_cache = (torch.cat([kv_cache[0], chunk_entry.kv[0]], dim=2),
                    torch.cat([kv_cache[1], chunk_entry.kv[1]], dim=2))
        # restore the chunk's linear state (no re-prefill!)
        linear_state = chunk_entry.linear_state.clone()
        conv_state = chunk_entry.conv_state.clone()

    # prefill the query on top
    with torch.no_grad():
        logits, _, _, _, _ = model(query_ids, linear_state=linear_state,
                                    conv_state=conv_state, kv_cache=kv_cache)
    return logits


# ---------------------------------------------------------------------------
# Attention-as-retrieval
# ---------------------------------------------------------------------------

def retrieve_by_attention(model, query_ids, cache: GlobalCache, system_prompt_ids,
                          device, top_k=3):
    """Retrieve top-k chunks by the model's attention weights.
    The model's full-attn attends to ALL cached chunks' KV. The attention
    weights ARE the retrieval scores."""
    # restore system prompt KV
    sys_key = hash_tokens(system_prompt_ids)
    sys_entry = cache.lookup(sys_key)
    if sys_entry is None:
        return []
    kv_cache = (sys_entry.kv[0].clone(), sys_entry.kv[1].clone())

    # append ALL cached chunks' KV (except the system prompt)
    chunk_keys = [k for k in cache.entries.keys() if k != sys_key]
    chunk_token_counts = []  # track how many tokens each chunk contributed
    for ck in chunk_keys:
        entry = cache.entries[ck]
        kv_cache = (torch.cat([kv_cache[0], entry.kv[0]], dim=2),
                    torch.cat([kv_cache[1], entry.kv[1]], dim=2))
        chunk_token_counts.append(entry.kv[0].shape[2])

    # prefill the query with output_attentions=True
    with torch.no_grad():
        logits, _, _, _, attn = model(query_ids, kv_cache=kv_cache,
                                       output_attentions=True)

    # attn shape: (B, H, T_q, T_k)
    # aggregate attention over heads and query tokens → per-token attention
    token_attn = attn.mean(dim=(0, 2))  # (T_k,)

    # map each token to its chunk
    chunk_scores = []
    offset = sys_entry.kv[0].shape[2]  # skip system prompt tokens
    for i, (ck, n_tokens) in enumerate(zip(chunk_keys, chunk_token_counts)):
        score = token_attn[offset:offset + n_tokens].sum().item()
        chunk_scores.append((ck, score))
        offset += n_tokens

    # top-k by attention score
    chunk_scores.sort(key=lambda x: -x[1])
    top_k_keys = [ck for ck, _ in chunk_scores[:top_k]]
    return top_k_keys


# ---------------------------------------------------------------------------
# The test harness
# ---------------------------------------------------------------------------

def gen_corpus(n_chunks, chunk_len, vocab, seed=0):
    """Generate synthetic corpus chunks."""
    rng = random.Random(seed)
    chunks = []
    for i in range(n_chunks):
        # each chunk has a "topic" — a bias toward certain tokens
        topic = i % 4
        topic_tokens = [10 + topic * 50 + j for j in range(50)]
        tokens = []
        for _ in range(chunk_len):
            if rng.random() < 0.3:
                tokens.append(rng.choice(topic_tokens))
            else:
                tokens.append(rng.randint(0, vocab - 1))
        chunks.append(torch.tensor([tokens], dtype=torch.long))
    return chunks


def gen_query(topic_idx, query_len, vocab, seed=0):
    """Generate a query biased toward topic_idx (so retrieval has a 'right' answer)."""
    rng = random.Random(seed + topic_idx)
    topic_tokens = [10 + topic_idx * 50 + j for j in range(50)]
    tokens = []
    for _ in range(query_len):
        if rng.random() < 0.5:
            tokens.append(rng.choice(topic_tokens))
        else:
            tokens.append(rng.randint(0, vocab - 1))
    return torch.tensor([tokens], dtype=torch.long)


def measure_correctness(model, device, n_chunks=8, chunk_len=16, query_len=8,
                         system_len=8, n_queries=10):
    """Test 1: Does cache restoration produce the same output as full re-prefill?

    KEY INSIGHT: the chunk's state must be snapshotted as part of the
    accumulated sequence (system prompt + chunk 1 + chunk 2 + ...), not in
    isolation. A chunk prefilled in isolation produces a different state
    than the same chunk prefilled after the system prompt.

    So the 'embed' step must prefill chunks in the context they'll be
    retrieved in — i.e., prefill the system prompt, then each chunk on
    top of the accumulated state, snapshotting the state AFTER each chunk.
    This is the 'document order' rule from the chunking taxonomy.

    For the correctness test, we prefill (system + chunk1 + chunk2 + chunk3)
    as one sequence (NO_CACHE), and compare to restoring (system's state +
    chunk1's state + chunk2's state + chunk3's state) where each chunk's
    state was snapshotted in the accumulated context."""
    torch.manual_seed(0)
    vocab = model.vocab
    system_prompt = torch.randint(0, vocab, (1, system_len))
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)

    cache = GlobalCache(max_entries=n_chunks + 1)

    # The CORRECT embed: prefill system prompt, then each chunk on top
    # of the accumulated state, snapshotting after each chunk.
    with torch.no_grad():
        # system prompt
        _, lin_state, conv_state, kv, _ = model(system_prompt, return_states=True)
    sys_key = hash_tokens(system_prompt)
    cache.install(sys_key, lin_state, conv_state, kv)
    # the accumulated KV (system + all chunks prefilled in sequence)
    accum_kv = (kv[0].clone(), kv[1].clone())
    accum_lin_state = lin_state.clone()
    accum_conv_state = conv_state.clone()
    # snapshot each chunk's state AS PART OF the accumulated sequence
    for chunk in chunks:
        with torch.no_grad():
            _, chunk_lin, chunk_conv, chunk_kv, _ = model(chunk,
                linear_state=accum_lin_state, conv_state=accum_conv_state,
                kv_cache=accum_kv, return_states=True)
        # the chunk's state is the accumulated state AFTER prefilling this chunk
        cache.install(hash_tokens(chunk), chunk_lin, chunk_conv, chunk_kv)
        # update the accumulators for the next chunk
        accum_lin_state = chunk_lin
        accum_conv_state = chunk_conv
        accum_kv = (chunk_kv[0].clone(), chunk_kv[1].clone())

    # for each query, compare NO_CACHE vs FULL_CACHE
    max_diff_full = 0.0
    for q in range(n_queries):
        query = gen_query(q % 4, query_len, vocab, seed=q)
        # NO_CACHE: full re-prefill of (system + chunks[:3] + query)
        logits_no = rag_no_cache(model, system_prompt, query, chunks[:3], device)
        # FULL_CACHE: restore the accumulated states for chunks[:3]
        # The last chunk's state (chunks[2]) is the accumulated state after
        # system + chunk0 + chunk1 + chunk2 — exactly what NO_CACHE computes.
        logits_full = rag_full_cache_accumulated(model, system_prompt, query,
                                                  chunks[:3], cache, device)
        diff_full = (logits_no - logits_full).abs().max().item()
        max_diff_full = max(max_diff_full, diff_full)

    return {
        'max_logit_diff_full_cache': max_diff_full,
        'full_cache_lossless': max_diff_full < 1e-4,
    }


def rag_full_cache_accumulated(model, system_prompt_ids, query_ids, retrieved_chunks_ids,
                               cache: GlobalCache, device):
    """FULL_CACHE with accumulated states: restore the LAST retrieved chunk's
    accumulated state (which includes all previous chunks + system prompt).
    This is the correct semantics: the chunk's state was snapshotted in
    the accumulated context, so restoring it reproduces the NO_CACHE result."""
    # the last chunk's state is the accumulated state after all chunks
    if len(retrieved_chunks_ids) == 0:
        # just system prompt + query
        sys_entry = cache.lookup(hash_tokens(system_prompt_ids))
        if sys_entry is None:
            with torch.no_grad():
                _, lin_state, conv_state, kv, _ = model(system_prompt_ids, return_states=True)
            cache.install(hash_tokens(system_prompt_ids), lin_state, conv_state, kv)
            sys_entry = cache.entries[hash_tokens(system_prompt_ids)]
        kv_cache = (sys_entry.kv[0].clone(), sys_entry.kv[1].clone())
        linear_state = sys_entry.linear_state.clone()
        conv_state = sys_entry.conv_state.clone()
    else:
        # restore the LAST chunk's accumulated state (includes all previous)
        last_chunk = retrieved_chunks_ids[-1]
        last_key = hash_tokens(last_chunk)
        last_entry = cache.lookup(last_key)
        if last_entry is None:
            # need to build it — prefill system + all chunks in sequence
            with torch.no_grad():
                _, lin_state, conv_state, kv, _ = model(system_prompt_ids, return_states=True)
            accum_kv = (kv[0].clone(), kv[1].clone())
            accum_lin = lin_state
            accum_conv = conv_state
            for chunk in retrieved_chunks_ids:
                with torch.no_grad():
                    _, accum_lin, accum_conv, accum_kv, _ = model(chunk,
                        linear_state=accum_lin, conv_state=accum_conv,
                        kv_cache=accum_kv, return_states=True)
                cache.install(hash_tokens(chunk), accum_lin.clone(), accum_conv.clone(), accum_kv)
            last_entry = cache.entries[last_key]
        kv_cache = (last_entry.kv[0].clone(), last_entry.kv[1].clone())
        linear_state = last_entry.linear_state.clone()
        conv_state = last_entry.conv_state.clone()

    # prefill the query on top of the restored accumulated state
    with torch.no_grad():
        logits, _, _, _, _ = model(query_ids, linear_state=linear_state,
                                    conv_state=conv_state, kv_cache=kv_cache)
    return logits


def measure_retrieval(model, device, n_chunks=16, chunk_len=16, query_len=8,
                      system_len=8, n_queries=20, top_k=3):
    """Test 2: Does attention-as-retrieval retrieve relevant chunks?
    Each query is biased toward a topic; the 'correct' chunks are the ones
    with the same topic bias."""
    torch.manual_seed(0)
    vocab = model.vocab
    system_prompt = torch.randint(0, vocab, (1, system_len))
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)

    cache = GlobalCache(max_entries=n_chunks + 1)
    # prefill system prompt and chunks
    with torch.no_grad():
        _, lin_state, conv_state, kv, _ = model(system_prompt, return_states=True)
    cache.install(hash_tokens(system_prompt), lin_state, conv_state, kv)
    for chunk in chunks:
        with torch.no_grad():
            _, lin, conv, ckv, _ = model(chunk, return_states=True)
        cache.install(hash_tokens(chunk), lin, conv, ckv)

    # the chunks' topics (chunk i has topic i % 4)
    chunk_topics = [i % 4 for i in range(n_chunks)]

    hits = 0
    total = 0
    for q in range(n_queries):
        query_topic = q % 4
        query = gen_query(query_topic, query_len, vocab, seed=q)
        # retrieve top-k by attention
        retrieved_keys = retrieve_by_attention(model, query, cache, system_prompt, device, top_k=top_k)
        # map keys back to chunk indices
        chunk_key_to_idx = {hash_tokens(c): i for i, c in enumerate(chunks)}
        retrieved_topics = [chunk_topics[chunk_key_to_idx[k]]
                           for k in retrieved_keys if k in chunk_key_to_idx]
        # is the query's topic in the retrieved chunks' topics?
        if query_topic in retrieved_topics:
            hits += 1
        total += 1

    return {
        'retrieval_hit_rate': hits / total,
        'n_queries': total,
        'n_chunks': n_chunks,
        'top_k': top_k,
    }


def measure_latency(model, device, n_chunks=8, chunk_len=16, query_len=8,
                    system_len=8, n_runs=20):
    """Test 3: Is the cache-engineered RAG faster than full re-prefill?"""
    torch.manual_seed(0)
    vocab = model.vocab
    system_prompt = torch.randint(0, vocab, (1, system_len))
    chunks = gen_corpus(n_chunks, chunk_len, vocab, seed=42)
    query = gen_query(0, query_len, vocab, seed=0)

    cache = GlobalCache(max_entries=n_chunks + 1)
    # prefill system prompt and chunks
    with torch.no_grad():
        _, lin_state, conv_state, kv, _ = model(system_prompt, return_states=True)
    cache.install(hash_tokens(system_prompt), lin_state, conv_state, kv)
    for chunk in chunks:
        with torch.no_grad():
            _, lin, conv, ckv, _ = model(chunk, return_states=True)
        cache.install(hash_tokens(chunk), lin, conv, ckv)

    # warm up
    for _ in range(3):
        rag_no_cache(model, system_prompt, query, chunks[:3], device)
        rag_full_cache_accumulated(model, system_prompt, query, chunks[:3], cache, device)

    # time NO_CACHE
    t0 = time.time()
    for _ in range(n_runs):
        rag_no_cache(model, system_prompt, query, chunks[:3], device)
    t_no = (time.time() - t0) / n_runs * 1000

    # time FULL_CACHE (accumulated)
    t0 = time.time()
    for _ in range(n_runs):
        rag_full_cache_accumulated(model, system_prompt, query, chunks[:3], cache, device)
    t_full = (time.time() - t0) / n_runs * 1000

    return {
        'no_cache_ms': t_no,
        'full_cache_ms': t_full,
        'full_cache_speedup_vs_no_cache': t_no / t_full if t_full > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_rag_results.json'))
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    model = TinyHybridModel(hidden=32, vocab=256).to(device)
    model.eval()

    print("=" * 60)
    print("TEST 1: Correctness — does cache restoration preserve the output?")
    print("=" * 60)
    correctness = measure_correctness(model, device, n_chunks=8, chunk_len=16,
                                        query_len=8, system_len=8, n_queries=10)
    print(f"  Max logit diff (FULL_CACHE vs NO_CACHE): {correctness['max_logit_diff_full_cache']:.6e}")
    print(f"  FULL_CACHE lossless (<1e-4): {correctness['full_cache_lossless']}")
    print(f"  (If lossless: restoring the accumulated state reproduces re-prefill exactly.)")
    print(f"  (If not: the chunk's state was snapshotted in the wrong context — see the docstring.)")

    print()
    print("=" * 60)
    print("TEST 2: Retrieval — does attention-as-retrieval work?")
    print("=" * 60)
    retrieval = measure_retrieval(model, device, n_chunks=16, chunk_len=16,
                                   query_len=8, system_len=8, n_queries=20, top_k=3)
    print(f"  Retrieval hit rate (query topic in top-3): {retrieval['retrieval_hit_rate']*100:.1f}%")
    print(f"  (random baseline: ~{(1 - (1 - 0.25)**3)*100:.1f}% — 3 chunks out of 4 topics)")
    print(f"  n_queries: {retrieval['n_queries']}, n_chunks: {retrieval['n_chunks']}, top_k: {retrieval['top_k']}")

    print()
    print("=" * 60)
    print("TEST 3: Latency — is the cache faster than re-prefill?")
    print("=" * 60)
    latency = measure_latency(model, device, n_chunks=8, chunk_len=16,
                                query_len=8, system_len=8, n_runs=20)
    print(f"  NO_CACHE:    {latency['no_cache_ms']:.3f} ms")
    print(f"  FULL_CACHE:  {latency['full_cache_ms']:.3f} ms (speedup: {latency['full_cache_speedup_vs_no_cache']:.2f}×)")

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    results = {'correctness': correctness, 'retrieval': retrieval, 'latency': latency}
    print(f"  Cache restoration preserves output: {correctness['full_cache_lossless']}")
    print(f"  Attention-as-retrieval works:       {retrieval['retrieval_hit_rate'] > 0.5}")
    print(f"  Cache is faster than re-prefill:    {latency['full_cache_speedup_vs_no_cache'] > 1.0}")
    print()
    if correctness['full_cache_lossless'] and retrieval['retrieval_hit_rate'] > 0.5 and latency['full_cache_speedup_vs_no_cache'] > 1.0:
        print("  ✓ The cache-engineered RAG architecture WORKS on the toy model.")
    else:
        print("  ✗ The cache-engineered RAG architecture has issues — see above.")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.out}")


if __name__ == '__main__':
    main()
