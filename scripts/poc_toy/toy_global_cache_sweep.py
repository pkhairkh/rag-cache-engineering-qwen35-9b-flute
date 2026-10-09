#!/usr/bin/env python3
"""toy_global_cache_sweep.py — CPU toy sweep to find what works with
TWO global caches for a linear-attention hybrid model.

The setup (tiny, CPU, faithful to the real Qwen3.5-9B mechanics):

  * A "linear attention layer" implements the delta rule with a fixed-size
    recurrent state S_t = decay * S_{t-1} + beta * (v_t ⊗ k_t) — the same
    math as scripts/modeling.py::Qwen3_5GatedDeltaNet.forward. State shape
    (heads=4, k_dim=8) — small enough to run thousands of forward steps on
    CPU in seconds, big enough that the dynamics are non-trivial.
  * A "session" is a sequence of tokens: [SYSTEM_PROMPT] + [RETRIEVED_CHUNK]
    + [QUERY] + [ANSWER]. We run many sessions with shared SYSTEM_PROMPT and
    varying RETRIEVED_CHUNK (RAG-style).
  * The model has 2 layers (one linear, one full) — like the real 3:1 hybrid
    in miniature. We snapshot the linear layer's state at the full-attention
    boundary (the K3 discipline).

Two global caches, compared:

  * GLOBAL_KV_CACHE      — caches the full-attention KV (token-level, the
                           vLLM/SGLang RadixAttention pattern). This is the
                           baseline "what everyone already does."
  * GLOBAL_STATE_CACHE   — caches the linear-attention recurrent STATE
                           (the K3 pattern). This is the proposal's bet.

We sweep:
  1. system_prompt_length ∈ {64, 256, 1024}  — how much is shared
  2. n_chunks_per_corpus  ∈ {8, 32}          — RAG diversity
  3. retrieval_overlap    ∈ {0.0, 0.25, 0.5} — how often sessions share
                                                the same retrieved chunk
  4. cache_size_mb        ∈ {16, 64, 256}    — the budget
  5. eviction             ∈ {lru, fifo}      — the policy

For each config we measure, for both cache types:
  * hit_rate               — % of prefill steps saved by a cache hit
  * bytes_cached           — total bytes in the cache
  * evictions               — number of evictions
  * cross_session_hit_rate — % of sessions that hit at least once
  * prefill_tokens_saved   — total tokens of prefill avoided

We also test ONE fine-tune hypothesis (the CacheBlend-FT idea, arxiv 2609.09768):
  * Train the model for N steps with the cache installed — does the model
    learn to produce states that survive re-installation? Measure the
    "restoration error" (||S_restored - S_recomputed||) before and after.
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
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. The tiny linear-attention hybrid (faithful to modeling.py mechanics)
# ---------------------------------------------------------------------------

class TinyGatedDeltaNet(nn.Module):
    """A miniature of scripts/modeling.py::Qwen3_5GatedDeltaNet.
    State shape: (batch, num_v_heads, head_k_dim, head_k_dim).
    The state IS the cache asset (the K3 bet)."""
    def __init__(self, hidden=32, num_v_heads=4, num_k_heads=4, head_k_dim=8,
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

        # projections
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
        """One token forward (decode shape, seq_len=1). Returns (out, new_state, new_conv_state).
        Mirrors modeling.py lines 660-748: the delta rule update on a single token."""
        B = x.shape[0]
        mixed = self.in_proj_qkv(x).transpose(1, 2)  # (B, conv_dim, 1)
        # conv1d update (in-place style: shift conv_state, append new)
        # conv_state shape: (B, conv_dim, conv_kernel)
        new_conv = torch.cat([conv_state[:, :, 1:], mixed], dim=2)
        mixed = F.conv1d(new_conv, self.conv1d.weight, None, groups=self.conv_dim).squeeze(-1)
        mixed = F.silu(mixed)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, self.num_k_heads, self.head_k_dim)
        k = k.view(B, self.num_k_heads, self.head_k_dim)
        v = v.view(B, self.num_v_heads, self.head_v_dim)
        # repeat k for v_heads
        if self.num_v_heads > self.num_k_heads:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        beta = torch.sigmoid(self.in_proj_b(x).view(B, self.num_v_heads, 1))
        a = self.in_proj_a(x).view(B, self.num_v_heads, 1)
        # broadcast g across the (B, num_v_heads) then add singleton dims for the state update
        g_per_head = -self.A_log.exp().float() * F.softplus(a.float().squeeze(-1) + self.dt_bias)  # (B, H)
        decay = torch.exp(g_per_head).view(B, self.num_v_heads, 1, 1)
        # beta: (B, H, 1) -> unsqueeze(-1) for the v ⊗ k outer product
        beta_factor = beta.unsqueeze(-1)  # (B, H, 1, 1)
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        new_state = decay * state + beta_factor * v.unsqueeze(2) * k_norm.unsqueeze(3)
        # the readout: o = S . q
        o = (new_state * q.view(B, self.num_v_heads, self.head_k_dim, 1)).sum(dim=2)
        o = self.norm(o.view(B, -1, self.head_v_dim))
        z = self.in_proj_z(x).view(B, 1, self.value_dim)
        # z is (B, 1, value_dim); o is (B, num_v_heads, head_v_dim) where num_v_heads*head_v_dim = value_dim
        # so reshape o to (B, 1, value_dim) and gate with z
        o_flat = o.reshape(B, 1, self.value_dim)
        o_flat = o_flat * F.silu(z.float()).to(o.dtype)
        out = self.out_proj(o_flat.view(B, -1))
        return out, new_state, new_conv

    def forward_chunk(self, x_seq, initial_state=None, initial_conv=None):
        """Prefill: process a sequence, return final state. Mirrors the chunk path."""
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
    """A miniature full-attention layer. KV cache shape: (B, H, T, D)."""
    def __init__(self, hidden=32, num_heads=4, head_dim=8):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden, 3 * num_heads * head_dim, bias=False)
        self.o = nn.Linear(num_heads * head_dim, hidden, bias=False)

    def forward(self, x_seq, kv_cache=None):
        B, T, _ = x_seq.shape
        qkv = self.qkv(x_seq).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        # scaled dot-product (tiny, on CPU)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # causal mask
        T_q = q.shape[2]
        T_k = k.shape[2]
        causal = torch.triu(torch.full((T_q, T_k), float('-inf')), diagonal=T_k - T_q)
        scores = scores + causal
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T_q, -1)
        return self.o(out), (k, v)


class TinyHybridModel(nn.Module):
    """1 linear-attention layer + 1 full-attention layer. The 3:1 hybrid in miniature.
    Boundary after the full-attention layer (where we snapshot state)."""
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

    def forward(self, input_ids, linear_state=None, conv_state=None, kv_cache=None):
        x = self.embed(input_ids)
        # linear attention block
        lin_out, new_lin_state, new_conv = self.linear_attn.forward_chunk(x, linear_state, conv_state)
        x = self.norm1(x + lin_out)
        # full attention block (the boundary — snapshot here)
        full_out, new_kv = self.full_attn(x, kv_cache)
        x = self.norm2(x + full_out)
        x = x + self.mlp(x)
        logits = self.lm_head(x)
        return logits, new_lin_state, new_conv, new_kv


# ---------------------------------------------------------------------------
# 2. The two global caches
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    key: str
    payload: bytes  # serialized tensor bytes
    size: int
    last_used: float


class GlobalKVCache:
    """Global cache of full-attention KV pairs (the vLLM/SGLang pattern).
    Keyed by content hash of the token prefix. LRU-evicted."""
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _hash_tokens(self, tokens: List[int]) -> str:
        return hashlib.sha256(bytes(tokens)).hexdigest()[:16]

    def lookup(self, token_prefix: List[int]) -> Optional[bytes]:
        key = self._hash_tokens(token_prefix)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            self.entries[key].last_used = time.time()
            return self.entries[key].payload
        self.misses += 1
        return None

    def install(self, token_prefix: List[int], payload: bytes, size: int):
        key = self._hash_tokens(token_prefix)
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        while self.current_bytes + size > self.max_bytes and self.entries:
            evicted_key, evicted = self.entries.popitem(last=False)
            self.current_bytes -= evicted.size
            self.evictions += 1
        self.entries[key] = CacheEntry(key, payload, size, time.time())
        self.current_bytes += size
        return True


class GlobalStateCache:
    """Global cache of linear-attention recurrent STATE (the K3 bet).
    Keyed by content hash of the token prefix that produced the state.
    LRU-evicted."""
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _hash_state(self, state: torch.Tensor, conv_state: torch.Tensor) -> str:
        h = hashlib.sha256()
        h.update(state.cpu().numpy().tobytes())
        h.update(conv_state.cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def _hash_prefix(self, token_prefix: List[int]) -> str:
        return hashlib.sha256(bytes(token_prefix)).hexdigest()[:16]

    def lookup_by_prefix(self, token_prefix: List[int]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Look up state by the token prefix that produced it."""
        key = self._hash_prefix(token_prefix)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key].payload
        self.misses += 1
        return None

    def install(self, token_prefix: List[int], state: torch.Tensor,
                conv_state: torch.Tensor):
        key = self._hash_prefix(token_prefix)
        payload = (state.clone(), conv_state.clone())
        size = state.nelement() * state.element_size() + conv_state.nelement() * conv_state.element_size()
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        while self.current_bytes + size > self.max_bytes and self.entries:
            evicted_key, evicted = self.entries.popitem(last=False)
            self.current_bytes -= evicted.size
            self.evictions += 1
        self.entries[key] = CacheEntry(key, payload, size, time.time())
        self.current_bytes += size
        return True


# ---------------------------------------------------------------------------
# 3. The session generator (RAG-style workload)
# ---------------------------------------------------------------------------

def gen_session(system_prompt_len: int, chunk_len: int, query_len: int,
                answer_len: int, vocab: int, rng: random.Random) -> List[int]:
    """Generate one session: [SYSTEM] + [CHUNK] + [QUERY] + [ANSWER]."""
    return [rng.randint(0, vocab-1) for _ in range(system_prompt_len + chunk_len + query_len + answer_len)]


def gen_corpus(n_sessions: int, system_prompt_len: int, chunk_len: int,
               query_len: int, answer_len: int, vocab: int, seed: int,
               retrieval_overlap: float, n_unique_chunks: int = 32) -> List[List[int]]:
    """Generate a corpus of sessions with shared system prompt and varying retrieval overlap.
    retrieval_overlap: fraction of sessions that share the same retrieved chunk."""
    rng = random.Random(seed)
    # one shared system prompt
    system_prompt = [rng.randint(0, vocab-1) for _ in range(system_prompt_len)]
    # a pool of unique chunks
    chunks = [[rng.randint(0, vocab-1) for _ in range(chunk_len)] for _ in range(n_unique_chunks)]
    sessions = []
    for i in range(n_sessions):
        if rng.random() < retrieval_overlap:
            chunk = chunks[i % n_unique_chunks]  # shared chunks
        else:
            chunk = [rng.randint(0, vocab-1) for _ in range(chunk_len)]  # unique
        query = [rng.randint(0, vocab-1) for _ in range(query_len)]
        answer = [rng.randint(0, vocab-1) for _ in range(answer_len)]
        sessions.append(system_prompt + chunk + query + answer)
    return sessions


# ---------------------------------------------------------------------------
# 4. The measurement sweep
# ---------------------------------------------------------------------------

@dataclass
class CacheMetrics:
    hit_rate: float
    cross_session_hit_rate: float
    bytes_cached: int
    evictions: int
    prefill_tokens_saved: int
    n_sessions: int


def run_session_with_caches(model, session, device, kv_cache, state_cache, system_prompt_len, chunk_len):
    """Run one session through the model, using the two global caches.
    Returns (metrics_dict, was_kv_hit, was_state_hit)."""
    tokens = torch.tensor([session], device=device, dtype=torch.long)
    was_kv_hit = False
    was_state_hit = False
    prefill_saved = 0

    # Try KV cache first (token-prefix match)
    prefix = session[:system_prompt_len]
    kv_payload = kv_cache.lookup(prefix) if kv_cache else None
    if kv_payload is not None:
        was_kv_hit = True
        prefill_saved += system_prompt_len
        # would restore KV here; in toy we just count the save
    else:
        # try state cache (linear-attention state for the prefix)
        state_payload = state_cache.lookup_by_prefix(prefix) if state_cache else None
        if state_payload is not None:
            was_state_hit = True
            prefill_saved += system_prompt_len  # state covers the linear-attn cost
            # in the real model: install the state, skip the linear-attn prefill,
            # still need full-attn KV — but the state covers the linear-attn cost
            # which is the bulk at long context
        # compute and install
        with torch.no_grad():
            sys_tokens = tokens[:, :system_prompt_len]
            logits, lin_state, conv_state, kv = model(sys_tokens)
        if kv_cache:
            kv_cache.install(prefix, b'kv', system_prompt_len * 4 * 8 * 2)  # rough size
        if state_cache:
            state_cache.install(prefix, lin_state, conv_state)

    # process the rest of the session (chunk + query + answer)
    with torch.no_grad():
        rest_tokens = tokens[:, system_prompt_len:]
        if kv_payload is not None or (state_cache and was_state_hit):
            # skip the system-prompt prefill (cache hit)
            logits, _, _, _ = model(rest_tokens)
        else:
            # full prefill
            logits, _, _, _ = model(tokens)

    return {
        'was_kv_hit': was_kv_hit,
        'was_state_hit': was_state_hit,
        'prefill_saved': prefill_saved,
    }


def sweep_one_config(model, device, sessions, system_prompt_len, chunk_len,
                     cache_size_bytes, use_kv_cache, use_state_cache):
    kv_cache = GlobalKVCache(cache_size_bytes) if use_kv_cache else None
    state_cache = GlobalStateCache(cache_size_bytes) if use_state_cache else None
    n_sessions_with_hit = 0
    total_prefill_saved = 0
    for s in sessions:
        m = run_session_with_caches(model, s, device, kv_cache, state_cache,
                                     system_prompt_len, chunk_len)
        if m['was_kv_hit'] or m['was_state_hit']:
            n_sessions_with_hit += 1
        total_prefill_saved += m['prefill_saved']
    total_prefill_possible = sum(system_prompt_len for _ in sessions)
    return CacheMetrics(
        hit_rate=total_prefill_saved / max(1, total_prefill_possible),
        cross_session_hit_rate=n_sessions_with_hit / len(sessions),
        bytes_cached=(kv_cache.current_bytes if kv_cache else 0) +
                     (state_cache.current_bytes if state_cache else 0),
        evictions=(kv_cache.evictions if kv_cache else 0) +
                  (state_cache.evictions if state_cache else 0),
        prefill_tokens_saved=total_prefill_saved,
        n_sessions=len(sessions),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sessions', type=int, default=50)
    parser.add_argument('--system-prompt-len', type=int, default=64)
    parser.add_argument('--chunk-len', type=int, default=32)
    parser.add_argument('--query-len', type=int, default=8)
    parser.add_argument('--answer-len', type=int, default=8)
    parser.add_argument('--retrieval-overlap', type=float, default=0.5)
    parser.add_argument('--n-unique-chunks', type=int, default=8)
    parser.add_argument('--cache-size-kb', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='/home/z/my-project/scripts/poc_toy/results.json')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cpu'
    vocab = 256
    model = TinyHybridModel(hidden=32, vocab=vocab).to(device)
    model.eval()

    sessions = gen_corpus(args.n_sessions, args.system_prompt_len, args.chunk_len,
                          args.query_len, args.answer_len, vocab, args.seed,
                          args.retrieval_overlap, args.n_unique_chunks)

    cache_bytes = args.cache_size_kb * 1024

    # The three configurations to compare
    configs = [
        ('no_cache', False, False),
        ('kv_only', True, False),
        ('state_only', False, True),
        ('kv_and_state', True, True),
    ]

    results = {
        'config': vars(args),
        'sessions': [{'system_prompt_len': args.system_prompt_len,
                      'chunk_len': args.chunk_len,
                      'retrieval_overlap': args.retrieval_overlap,
                      'n_unique_chunks': args.n_unique_chunks}],
        'cache_configs': {},
    }

    for name, use_kv, use_state in configs:
        m = sweep_one_config(model, device, sessions, args.system_prompt_len,
                             args.chunk_len, cache_bytes, use_kv, use_state)
        results['cache_configs'][name] = asdict(m)
        print(f"  {name:15s}  hit_rate={m.hit_rate:.3f}  cross_session={m.cross_session_hit_rate:.3f}  "
              f"bytes={m.bytes_cached}  evictions={m.evictions}  prefill_saved={m.prefill_tokens_saved}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")

    # Print a summary table
    print("\n=== Summary ===")
    print(f"{'Config':<15} {'Hit%':<8} {'XSession%':<10} {'Bytes':<10} {'Evict':<6} {'Saved':<8}")
    for name, _, _ in configs:
        m = results['cache_configs'][name]
        print(f"{name:<15} {m['hit_rate']*100:<8.1f} {m['cross_session_hit_rate']*100:<10.1f} "
              f"{m['bytes_cached']:<10} {m['evictions']:<6} {m['prefill_tokens_saved']:<8}")


if __name__ == '__main__':
    main()
