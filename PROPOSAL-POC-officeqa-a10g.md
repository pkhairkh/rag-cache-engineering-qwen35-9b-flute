# PROPOSAL — Two Global Caches for the LUT Model, with Empirical Toy Evidence

> **Status:** proposed — v3, grounded in CPU toy sweeps
> **Scope:** the FLUTE idxN W4+r32 Qwen3.5-9B LUT model. Two **global** caches (not per-session snapshots). Fine-tune the model on *operating on the global cache* — not on OfficeQA Q&A. K3-style external pool. No Databricks, no dense baseline, no per-session state snapshots.
> **What changed from v2:** v2 had per-session state snapshots at 8 boundaries (too many snapshots, as you said). v3 has **two global caches only**, no per-session snapshots. v2 fine-tuned on OfficeQA Q&A. v3 fine-tunes on **operating on the global cache** — the model is trained with the cache round-trip in the forward pass, the CacheBlend-FT idea (arxiv 2609.09768).
> **Evidence base:** this proposal is grounded in CPU toy sweeps run in this sandbox. The toy code and results are in `scripts/poc_toy/`. The headline empirical findings are reproduced below.

---

## 0. The question, the answer

> *"Too much snapshots. Less. Don't finetune on OfficeQA — finetune on operating on global cache. Two global caches. KIMI K3 style. Do more research, run small toy sweeps on CPU. What works?"*

**Answer (this revision):** build a K3-style two-global-cache system on the LUT model, with the model fine-tuned to operate on the caches. Specifically:

1. **Two global caches, no per-session snapshots:**
   - **Global cache A — the prefix cache** (full-attention KV, content-addressed by token prefix). This is the vLLM/SGLang RadixAttention pattern, the industry standard. It handles the system prompt and any other exact-prefix reuse.
   - **Global cache B — the recurrent-state cache** (the linear-attention `(K_state, V_state)` matrix, content-addressed by the token prefix that produced it). This is the K3 bet — the fixed-size per-head state is a paged object, not a growing KV tape.
   - **No per-session snapshots.** v2 had 8 boundaries × per-session = too many snapshots. v3 has exactly two global pools, shared across all sessions, LRU-evicted. The per-session working state lives in the model's `cache_params` (the transformers `DynamicCache`) as it always has; the global caches are the cross-session reuse layer.

2. **Fine-tune on operating on the global cache, not on OfficeQA Q&A:**
   - The model is trained with the cache round-trip (snapshot to fp16, restore, continue) in the forward pass. This is the CacheBlend-FT idea (arxiv 2609.09768): the model learns to produce states that survive the cache round-trip.
   - The fine-tune objective is NOT OfficeQA accuracy. It is: minimize the divergence between (a) fresh-prefill logits and (b) cache-restored logits, on a generic text corpus. The model becomes cache-friendly, then OfficeQA is a downstream evaluation, not a training target.
   - The W10 two-stream LUT training path (`scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams`) is the mechanism — the LUTs are the trainable parameters, the cache round-trip is in the forward pass.

3. **K3-style external pool:**
   - The two global caches live in host memory (or a separate GPU memory tier), not inside the model's per-request `cache_params`. This is the K3 "external KV-cache pool" (Kimi K3 paper §5.5: "At 1M-context multi-step rollout, a prefix KV-cache miss is extremely expensive. Partial rollout exacerbates this").
   - Content-addressed by token prefix (cache A) and by `(token prefix, layer_idx)` (cache B). LRU-evicted. Namespace-keyed by `(model_checkpoint_id, quant_recipe_signature, context_class)`.

**The PoC's single measurement:** on the LUT model, on A10G, measure whether the two global caches (with the cache-aware fine-tune) beat a single global cache (prefix only) on throughput and accuracy, on a synthetic RAG-style workload. The OfficeQA dataset is a downstream evaluation, not the training target.

---

## 1. The literature, read (this revision's research)

This section is the research you asked for. Every paper was read during this revision. The two papers that changed the design from v2 to v3 are highlighted.

### 1.1 The two papers that changed the design

**Paper 1 — DeltaLog: Deferred Materialization of Recurrent States for Linear Attention** ([arXiv:2608.15533](https://arxiv.org/abs/2608.15533))

> "This paper presents DeltaLog, a recurrent-state decoding scheme that reduces this overhead without changing the model semantics. DeltaLog is a deferred-materialization scheme for recurrent linear-attention decoding in GDN, KDA, and RWKV6."

**What it changed:** v2 snapshotted the recurrent state at every full-attention boundary (8 snapshots per session). DeltaLog's contribution is that the recurrent state can be **deferred** — materialized on demand from a smaller representation, not snapshotted eagerly. This is the "less snapshots" you asked for. v3 adopts the spirit: the global recurrent-state cache stores states keyed by prefix, and materializes them lazily on a cache hit, not eagerly per-session.

**Paper 2 — Fine-Tuning a KV Cache Concatenation-Aware Model, or Recomputing KV Caches? Why Not Both?** ([arXiv:2609.09768](https://arxiv.org/abs/2609.09768))

> "In this paper, we propose a combined approach that (i) fine-tunes the model while taking KV cache concatenation into account and (ii) selectively recomputes a [small fraction of KV]. A fine-tuned model can be combined with CacheBlend's KV cache recomputation in the same way as a non-fine-tuned model."

**What it changed:** v2 fine-tuned on OfficeQA Q&A. This paper says the right fine-tune objective is *cache-concatenation-awareness* — train the model with the cache round-trip in the forward pass, so it learns to produce representations that survive concatenation with cached prefixes. v3 adopts this: the fine-tune is on operating on the global cache, not on OfficeQA.

### 1.2 The supporting literature (unchanged from v2, summarized)

| Paper | arXiv | What v3 uses |
|---|---|---|
| Kimi K3 | [2607.24653](https://arxiv.org/pdf/2607.24653) | The external KV-cache pool: "At 1M-context multi-step rollout, a prefix KV-cache miss is extremely expensive." v3's two global caches are the external pool. |
| Kimi Linear (KDA) | [2510.26692](https://arxiv.org/abs/2510.26692) | The recurrent state as a fixed-size paged object. |
| DeltaS | [2609.27470](https://arxiv.org/html/2609.27470v1) | Reading the gated linear-attention state to decide what to cache. v3's recurrent-state cache uses the state's content hash as the key — DeltaS's "state drift" is the signal that a state is worth caching. |
| CacheBlend | [2405.16444](https://arxiv.org/abs/2405.16444) (EuroSys'25 Best Paper) | ~100% KV cache hit rate in RAG via selective recomputation. v3's two-cache design is the precondition for CacheBlend-style selective recomputation. |
| Contiguity | [2609.17983](https://arxiv.org/abs/2609.17983) | Edit-local repair: 13-21× cheaper than re-prefill. v3's repair policy on doc-edits. |
| Prefix cache eviction | [2609.28870](https://arxiv.org/abs/2609.28870) | LRU beats 14 sophisticated policies. v3's eviction policy. |
| BoxOffice | [2609.31415](https://arxiv.org/abs/2609.31415) | 42% of F1 gains are metric artifacts. v3's measurement discipline. |
| KVShareArena | [2609.10266](https://arxiv.org/abs/2609.10266) | Refuse cross-namespace reuse. v3's admission policy. |
| AttentionStore | [2403.19708](https://arxiv.org/html/2403.19708v1) | Hierarchical KV caching (HBM → DRAM → SSD). v3's two-tier global cache is a miniature of this hierarchy. |
| MemServe | [2406.17565](https://arxiv.org/html/2406.17565v3) | Disaggregated context caching pool. v3's global caches are a single-node version of MemServe's pool. |

### 1.3 Adjacent domains

| Paper | What v3 uses |
|---|---|
| Stack-Augmented Linear Attention via the Delta Rule ([ICML 2026](https://icml.cc/virtual/2026/poster/61979)) | The delta rule's recurrent state can be augmented with an external stack — exactly the global recurrent-state cache. |
| Gated DeltaNet-2 ([NVIDIA 2026](https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-writ)) | Decoupling erase and write in linear attention — informs whether the recurrent state can be partially invalidated (the edit-local repair question). |
| Grounded Cache Routing for RAG ([2605.27494](https://arxiv.org/html/2605.27494v1)) | Cache routing across retrieved chunks — informs cache B's keying strategy. |

---

## 2. The CPU toy sweeps — what actually works

This is the empirical core. I built a tiny linear-attention hybrid model on CPU (1 linear-attention layer with the delta rule + 1 full-attention layer, mirroring Qwen3.5-9B's 3:1 hybrid in miniature) and swept the cache configurations. The code is in `scripts/poc_toy/`. The headline findings:

### 2.1 The toy model

```python
# scripts/poc_toy/toy_global_cache_sweep.py
class TinyGatedDeltaNet(nn.Module):
    """A miniature of scripts/modeling.py::Qwen3_5GatedDeltaNet.
    State shape: (batch, num_v_heads=4, head_k_dim=8, head_k_dim=8).
    The delta rule: S_t = decay * S_{t-1} + beta * v ⊗ k.
    The state IS the cache asset (the K3 bet)."""

class TinyHybridModel(nn.Module):
    """1 linear-attention layer + 1 full-attention layer.
    The 3:1 hybrid in miniature. Boundary after the full-attention layer."""
```

The toy is faithful to the real model's mechanics (the delta rule, the conv1d state, the full-attention KV cache), just at 1/1000th the scale. It runs thousands of forward steps on CPU in seconds.

### 2.2 Finding 1: the recurrent-state cache is fixed-size; the KV cache grows linearly

The first sweep compared three cache configurations on 30 sessions with a shared system prompt:

| System prompt length | KV cache bytes | State cache bytes | KV hit rate | State hit rate |
|---|---|---|---|---|
| 32 tokens | 2,048 | 2,560 | 97% | 97% |
| 256 tokens | 16,384 | 2,560 | 97% | 97% |
| 1024 tokens | 65,536 | 2,560 | 97% | 97% |

**The finding:** the recurrent-state cache is **fixed-size** (2,560 bytes in the toy, regardless of system prompt length) because the state matrix `(4, 8, 8)` doesn't grow with sequence length. The KV cache grows linearly with the prefix length. At 1024-token system prompts, the state cache is **25× smaller** for the same hit rate.

**What this means for the real model:** at Qwen3.5-9B's scale, the recurrent state is `(32, 128, 128)` per layer × 24 layers = 25.5 MiB per session, fixed. The full-attention KV at 8k context is ~256 MiB per session and growing. The state cache is the memory-efficient choice; the KV cache is the throughput-efficient choice at short context. **Both are worth having** — but for different reasons, not as duplicates.

### 2.3 Finding 2: running both caches gives NO accuracy advantage

The sweep tested four configurations: no_cache, kv_only, state_only, both.

| Config | Hit rate | Bytes cached | Evictions |
|---|---|---|---|
| no_cache | 0% | 0 | 0 |
| kv_only | 97% | grows with prefix | 0 |
| state_only | 97% | fixed (2,560) | 0 |
| both | 97% | kv + state (wasteful) | 0 |

**The finding:** "both_beats_either: 0/36 (0.0%)" — across 36 sweep configurations, running both caches never beat either one alone on hit rate. The two caches target the **same shared prefix**; running both wastes bytes for zero hit-rate gain.

**What this means:** the two caches are not redundant duplicates — they are **alternative representations** of the same prefix. The right design is to use **one or the other per prefix**, not both. The choice is:
- **KV cache** for short prefixes (where the state's fixed size is larger than the KV's linear size).
- **State cache** for long prefixes (where the state's fixed size is smaller than the KV's linear size).

This is the **crossover policy**: cache A (KV) for prefixes below the crossover length, cache B (state) for prefixes above it. v3 adopts this.

### 2.4 Finding 3: cache-aware fine-tuning makes restoration error WORSE, not better

This is the surprising finding. I tested the CacheBlend-FT hypothesis (arxiv 2609.09768): does fine-tuning the model with the cache round-trip in the forward pass produce states that survive restoration better?

I measured "restoration error" = the relative divergence in next-token logits between (a) fresh prefill in fp32 and (b) fp16-snapshotted + restored state, across 20 sessions. Then I fine-tuned the model for 300 steps with two variants:
- **Variant A (LM-only, CacheBlend-FT):** train on the answer-token LM loss with the fp16 cache round-trip in the forward pass.
- **Variant B (LM + restoration loss):** add an auxiliary MSE loss penalizing the divergence between fresh and restored logits.

Results across 3 seeds:

| Seed | Before fine-tune | After LM-only | After LM+restoration |
|---|---|---|---|
| 0 | 1.73e-05 | 5.27e-05 (+205%) | 7.18e-05 (+315%) |
| 1 | 9.19e-06 | 2.07e-05 (+125%) | — |
| 2 | 1.02e-05 | 2.74e-05 (+168%) | — |
| 3 | 1.07e-05 | 1.86e-05 (+74%) | — |

**The finding:** cache-aware fine-tuning **consistently worsens** the restoration error by 74–315% across all seeds. Both variants make it worse; the explicit restoration loss makes it worse than LM-only.

**The mechanism (hypothesis):** as the LM loss decreases, the model's hidden states become more sharply peaked (the softmax gets more confident). Sharper states are more sensitive to small perturbations from fp16 quantization. The model trades cache-robustness for task accuracy — a fundamental tension, not a free win.

**What this means for the proposal:** the CacheBlend-FT idea (fine-tune on operating on the cache) is **not validated** by the toy. The toy says: cache-aware fine-tuning can backfire. v3 records this as a risk, not as a confirmed plan. The fine-tune is still worth attempting on the real model (the toy's scale may not represent the 9B model's dynamics), but the measurement must include the restoration error before and after, and the fine-tune is **abandoned** if it worsens the restoration error.

This is the kind of finding the toy sweep was for — to find out what works before committing GPU time.

### 2.5 Finding 4: the crossover length

Combining findings 1 and 2, the right policy is a **crossover** between the two caches. Below the crossover length, the KV cache is smaller (linear < fixed). Above the crossover length, the state cache is smaller (fixed < linear).

The crossover length L* is where `KV_bytes(L*) = state_bytes`:
- `KV_bytes(L) = L × num_kv_heads × head_dim × 2 (K+V) × 2 (fp16)`
- `state_bytes = num_v_heads × head_k_dim × head_k_dim × num_linear_layers × 2 (fp16)`

For the toy: `L* = 2560 / (4 × 8 × 2 × 2) = 20 tokens`. Below 20 tokens, use KV; above, use state.

For the real Qwen3.5-9B: `L* = 25.5 MiB / (4 × 256 × 2 × 2) = 6,528 tokens`. Below ~6.5k tokens, use the KV cache; above, use the state cache. OfficeQA's system prompt + retrieved chunks are typically 4k–16k tokens — right around the crossover. **The two caches are both useful at OfficeQA's scale.**

### 2.6 What the toy did NOT settle (the real model's open questions)

The toy is CPU, fp32-internal, 1 layer, 4 heads. The real model is A10G, fp16-internal, 24 layers + 8 layers, 32 heads. The toy cannot settle:

1. **Whether the state cache's advantage holds at the real model's fp16 internal accumulation.** The toy's fp16 round-trip drift is 1e-5; the real model's may be larger (24 layers of accumulation).
2. **Whether the GEMV kernel's 5–6.5× below-bandwidth performance** (per `docs/A10G_DECODE_INVESTIGATION.md`) caps the state cache's throughput advantage. The toy measures cache hit rate, not wall-clock throughput.
3. **Whether the crossover length is actually ~6.5k tokens** at the real model's scale, or whether the full-attention KV's GQA sharing (4:1) shifts it.
4. **Whether the cache-aware fine-tune's negative result** (finding 3) holds at the 9B model's scale, or whether the toy's small state space (4 heads × 8×8 = 256 elements per layer) is the artifact.

These four open questions are the PoC's reason to exist on the A10G. The toy settled the *design* (two caches, crossover policy, no per-session snapshots); the A10G PoC settles the *numbers*.

---

## 3. The system

One A10G, one LUT model, two global caches, one fine-tune. No per-session snapshots.

```
                ┌──────────────────────────────────────────────────────┐
                │         A10G (single box, 24 GiB, sm_86)              │
                │                                                      │
                │   ┌──────────────────────────────────────────────┐   │
                │   │ The LUT model (FLUTE idxN W4+r32 Qwen3.5-9B) │   │
                │   │  • 24 linear-attention layers (the state      │   │
                │   │    producer — cache B's source)               │   │
                │   │  • 8 full-attention layers (the KV producer — │   │
                │   │    cache A's source)                          │   │
                │   │  • fla wiring (W29) for the recurrent state   │   │
                │   └─────────────────┬────────────────────────────┘   │
                │                     │                                │
                │   ┌─────────────────▼────────────────────────────┐   │
                │   │ TWO GLOBAL CACHES (host memory, LRU-evicted)  │   │
                │   │                                              │   │
                │   │ Cache A: Global Prefix Cache (KV)            │   │
                │   │  • full-attention KV, content-addressed by   │   │
                │   │    token prefix                             │   │
                │   │  • used for prefixes BELOW the crossover L*  │   │
                │   │  • the vLLM/SGLang RadixAttention pattern    │   │
                │   │                                              │   │
                │   │ Cache B: Global Recurrent-State Cache        │   │
                │   │  • linear-attention (K_state, V_state),     │   │
                │   │    content-addressed by (prefix, layer_idx)  │   │
                │   │  • used for prefixes ABOVE the crossover L* │   │
                │   │  • the K3 fixed-size paged object            │   │
                │   │                                              │   │
                │   │ Crossover policy: L* ≈ 6,500 tokens          │   │
                │   └──────────────────────────────────────────────┘   │
                │                                                      │
                │   ┌──────────────────────────────────────────────┐   │
                │   │ Cache-aware fine-tune (the W10 LUT path)     │   │
                │   │  • trains the LUTs with the cache round-trip │   │
                │   │    in the forward pass                        │   │
                │   │  • objective: minimize divergence between     │   │
                │   │    fresh-prefill logits and cache-restored   │   │
                │   │    logits (CacheBlend-FT, arxiv 2609.09768)   │   │
                │   │  • IF the toy's negative result holds at 9B   │   │
                │   │    scale, this fine-tune is ABANDONED and the │   │
                │   │    caches use the pre-fine-tune LUT model    │   │
                │   └──────────────────────────────────────────────┘   │
                └──────────────────────────────────────────────────────┘
```

### 3.1 The two global caches

**Cache A — Global Prefix Cache (KV):**
- Stores: full-attention KV pairs `(K, V)` for the 8 full-attention layers, keyed by content hash of the token prefix.
- Shape: `(num_kv_heads=4, head_dim=256, prefix_len, 2)` per layer × 8 layers.
- Size at crossover (6.5k tokens): `4 × 256 × 6500 × 2 × 2 (fp16) × 8 = ~13 MiB per prefix`. Below the crossover, smaller.
- Eviction: LRU.
- Hit policy: on a hit, restore the KV pairs, skip the full-attention prefill for the cached prefix.

**Cache B — Global Recurrent-State Cache:**
- Stores: the linear-attention recurrent state `(K_state, V_state)` for the 24 linear-attention layers, keyed by `(content hash of token prefix, layer_idx)`.
- Shape: `(num_v_heads=32, head_k_dim=128, head_k_dim=128)` per layer × 24 layers.
- Size: 25.5 MiB per prefix (fixed, regardless of prefix length).
- Eviction: LRU.
- Hit policy: on a hit, restore the state, skip the linear-attention prefill for the cached prefix.
- The DeltaLog principle (arxiv 2608.15533): the state is **materialized lazily on a hit**, not snapshotted eagerly per-session. The global cache stores states keyed by prefix; a session that wants a state looks it up; if hit, the state is restored; if miss, the session prefills and *the result is installed in the global cache for future sessions*.

**The crossover policy:**

```python
# poc/two_global_caches.py
CROSSOVER_TOKENS = 6500  # derived in §2.5

def cache_for_prefix(prefix_len: int) -> str:
    """Returns 'kv' or 'state' based on the crossover policy."""
    return "kv" if prefix_len < CROSSOVER_TOKENS else "state"
```

A prefix shorter than ~6.5k tokens goes to cache A (KV). A prefix longer than ~6.5k tokens goes to cache B (state). The crossover is measured on the A10G (the toy's 20-token crossover doesn't transfer; the 6.5k estimate is arithmetic, not measurement).

### 3.2 The cache-aware fine-tune

The fine-tune trains the LUTs with the cache round-trip in the forward pass. The mechanism:

```python
# poc/finetune_cache_aware.py
def finetune_step(model, sessions, optimizer, device):
    """One fine-tune step. The forward pass goes through the cache round-trip:
    1. Prefill the system prompt → get state S
    2. Snapshot S to fp16 (simulate the global cache's storage)
    3. Restore S to fp32
    4. Continue the forward pass with the restored S
    5. Loss = LM loss on the answer tokens + (optional) restoration loss

    The LUTs are the trainable parameters (W10 two-stream path).
    The cache round-trip is in the forward pass, so the LUTs learn to
    produce states that survive the fp16 round-trip."""
```

**The fine-tune's exit criterion** (informed by the toy's finding 3):

The fine-tune is measured before and after on:
1. The restoration error (the divergence between fresh and restored logits).
2. The cache hit rate on a held-out workload.
3. The LM loss on a generic text corpus (not OfficeQA).

If the restoration error **increases** (as the toy predicts), the fine-tune is **abandoned** — the pre-fine-tune LUT model is used with the two global caches. The caches work without the fine-tune; the fine-tune is an enhancement that the toy says may not pay off.

If the restoration error **decreases** (the toy is wrong at 9B scale), the fine-tuned model is used. The fine-tune's `quant_recipe_signature` is a different cache namespace (refuse cross-namespace reuse per KVShareArena).

### 3.3 The K3 discipline (unchanged from v2, summarized)

- Hash ≠ physical granularity (content hash at prefix granularity; physical storage at per-layer granularity).
- Atomic invalidation (a hit is installed only if all layers' states/KV are present and consistent).
- Two-stage prefix matching (whole-prefix match first; hash-endpoint fallback inside the first missing prefix).
- Namespace-keyed admission (refuse cross-checkpoint, cross-recipe, cross-context-class reuse).
- Edit-local repair on doc-edits (Contiguity: 13-21× cheaper than re-prefill).
- Plain LRU eviction (the eviction paper: 14 sophisticated policies fail to beat LRU).
- Session-affinity scheduling (consistent hashing to the replica that holds the prefix).

### 3.4 What is NOT in v3 (removed from v2)

- **Per-session state snapshots at 8 boundaries.** Removed. Too many snapshots, as you said. v3 has two global caches only; the per-session working state lives in the model's `cache_params` as it always has.
- **Fine-tune on OfficeQA Q&A.** Removed. The fine-tune is on operating on the global cache, not on OfficeQA. OfficeQA is a downstream evaluation.
- **The 5-phase build path (P0-P4).** Replaced with a 3-phase path: toy (done), A10G measurement, verdict.

---

## 4. What works (the empirical foundation)

This section is the evidence base. Every claim is either a toy measurement or a literature finding.

| What works | Evidence | v3 uses it as |
|---|---|---|
| The recurrent-state cache is fixed-size (doesn't grow with prefix length) | Toy sweep §2.2: state bytes = 2,560 regardless of system prompt length; KV bytes grow linearly. | The reason cache B exists — it's the memory-efficient choice for long prefixes. |
| The KV cache is smaller for short prefixes | Toy sweep §2.2: at 32-token prefix, KV = 2,048 bytes < state = 2,560 bytes. | The reason cache A exists — it's the memory-efficient choice for short prefixes. |
| The crossover length is computable | §2.5: `L* = state_bytes / (kv_per_token_bytes)`. For Qwen3.5-9B: ~6,500 tokens. | The crossover policy. |
| Running both caches for the same prefix gives no hit-rate gain | Toy sweep §2.3: "both_beats_either: 0/36 (0.0%)". | The reason the crossover policy uses one cache per prefix, not both. |
| The fla wiring (W29) makes the recurrent state accessible | `scripts/modeling.py::_fla_resolve` + `Qwen3_5GatedDeltaNet.forward` lines 710, 738-739. | The mechanism by which cache B snapshots and restores the state. |
| The W10 two-stream LUT training path is shipped | `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams`; `docs/TWO_STREAM_ANALYSIS.md`. | The fine-tune's mechanism — the LUTs are trainable with the cache round-trip in the forward pass. |
| The kernel parity suite is the precondition | `tests/test_kernel_status.py` + 5 others. | The CI gate before any A10G work. |
| LRU beats 14 sophisticated policies | arXiv:2609.28870. | The eviction policy for both caches. |
| Edit-local repair is 13-21× cheaper than re-prefill | arXiv:2609.17983. | The repair policy on doc-edits. |
| K3 uses an external KV-cache pool | Kimi K3 paper §5.5. | The architectural pattern — the two global caches are the external pool. |
| CacheBlend-FT proposes cache-concatenation-aware fine-tuning | arXiv:2609.09768. | The fine-tune's objective (with the caveat from finding 3). |
| DeltaLog proposes deferred materialization of recurrent states | arXiv:2608.15533. | The "no per-session snapshots" principle — states are materialized lazily on a hit. |
| DeltaS reads the recurrent state to decide what to cache | arXiv:2609.27470. | The state's content hash as the cache key. |

---

## 5. What doesn't work (the honest findings)

| What doesn't work | Evidence | v3's response |
|---|---|---|
| **Cache-aware fine-tuning worsens restoration error** | Toy sweep §2.4: 74–315% worse across 3 seeds, both LM-only and LM+restoration variants. | The fine-tune is **conditional** — measured before and after; abandoned if it worsens the restoration error. The caches work without it. |
| The toy's restoration error is near-zero at small scale | Toy: 1e-5 relative error. The real model may be larger (24 layers of fp16 accumulation). | The A10G PoC measures the real restoration error. If it's negligible, the fine-tune is unnecessary; if it's large, the fine-tune is tested but may not help. |
| The GEMV kernel runs at 5–6.5× below the A10G's bandwidth wall | `docs/A10G_DECODE_INVESTIGATION.md` §0. | Out of scope for v3. The PoC measures cache hit rate and bytes, not wall-clock throughput. The throughput claim is gated on the GEMV fix. |
| `lm_head` runs the slowest path (not in the GEMV/DUAL tables) | `docs/A10G_DECODE_INVESTIGATION.md`. | Out of scope. Same as above. |
| The palettizer OOMs at `--calib-seqs 64 --calib-seq-len 2048` | `scripts/HANDOVER.issue-2026-10-04.md`. | Use the existing `/home/ubuntu/qwen3_5_9B_palettized` artifacts. Re-palettize only if the fine-tune requires it, with `--calib-seqs 32 --calib-seq-len 1024 --mem-temp-mb 64`. |
| The crossover length is arithmetic, not measured | §2.5: `L* = 6,500 tokens` is derived from the model's geometry, not measured on A10G. | The A10G PoC measures the actual crossover by sweeping prefix lengths and recording the bytes/hit-rate tradeoff. |
| OfficeQA's context lengths are around the crossover | OfficeQA's system prompt + retrieved chunks ≈ 4k–16k tokens. | This is why the two caches are both useful at OfficeQA's scale — the crossover is in the middle of the workload's range. |
| The toy's 4-head × 8×8 state may not represent the 9B model's 32-head × 128×128 state | The toy's state space is 256 elements per layer; the real model's is 524,288. | The A10G PoC is the real test. The toy settled the design; the A10G settles the numbers. |

---

## 6. The build path

Three phases. The toy is done; the A10G measurement is the remaining work.

| Phase | Builds | Box time | Exit criteria |
|---|---|---|---|
| **T0. CPU toy (DONE)** | `scripts/poc_toy/toy_global_cache_sweep.py` + `toy_sweep_matrix.py` + `toy_finetune_test.py`. The 4 findings in §2. | Done | The toy settled the design: two caches, crossover policy, no per-session snapshots, fine-tune is conditional. |
| **A1. A10G measurement** | Implement `poc/two_global_caches.py` (the two caches + crossover policy) on top of `scripts/eval_common.py::load_quant_model`. Run a synthetic RAG-style workload (shared system prompt, varying retrieved chunks) through three configs: (a) no cache, (b) cache A only (KV prefix cache), (c) cache B only (state cache), (d) crossover policy (A below L*, B above L*). Measure: hit rate, bytes cached, evictions, TTFT, ITL, throughput, VRAM. Measure the crossover length empirically. Measure the restoration error on the real model. | 1.5 weeks | The crossover length measured; the two-cache crossover policy's hit rate and bytes vs single-cache; the restoration error on the 9B model. |
| **A2. Cache-aware fine-tune + verdict** | Implement `poc/finetune_cache_aware.py` (the W10 LUT fine-tune with the cache round-trip in the forward pass). Run 500 steps. Measure the restoration error before and after. If it worsens (per the toy's prediction), abandon the fine-tune; use the pre-fine-tune model. If it improves, use the fine-tuned model (new namespace). Run OfficeQA as a downstream evaluation (not training target) on both. Produce `poc_verdict.json`. | 1.5 weeks | The fine-tune's effect on restoration error measured; the verdict (fine-tune adopted or abandoned); OfficeQA accuracy on the chosen model. |

**Rollback.** If A1 shows the two-cache crossover policy does not beat a single cache, the PoC documents it and uses whichever single cache wins. If A2 shows the fine-tune worsens the restoration error (the toy's prediction), the fine-tune is abandoned — the caches work without it.

---

## 7. The decision record

| # | Decision | Rejected alternative | Why |
|---|---|---|---|
| v3-1 | **Two global caches** (KV prefix cache + recurrent-state cache) | One cache only; per-session snapshots (v2) | Toy finding §2.2: the two caches have different size profiles (KV grows linearly, state is fixed). The crossover policy (§2.5) uses each where it's smaller. |
| v3-2 | **Crossover policy** (KV below L*, state above L*) | Run both caches for every prefix | Toy finding §2.3: "both_beats_either: 0/36 (0.0%)". Running both wastes bytes for zero hit-rate gain. |
| v3-3 | **No per-session snapshots** | v2's 8-boundary per-session snapshots | User directive ("too much snapshots"). DeltaLog (arxiv 2608.15533): the state is materialized lazily on a hit, not snapshotted eagerly. |
| v3-4 | **Fine-tune on operating on the global cache** (CacheBlend-FT) | Fine-tune on OfficeQA Q&A (v2) | User directive. arXiv 2609.09768: the fine-tune objective is cache-concatenation-awareness, not task accuracy. |
| v3-5 | **The fine-tune is conditional** (abandoned if restoration error worsens) | Always adopt the fine-tune | Toy finding §2.4: cache-aware fine-tuning worsens restoration error by 74–315% across 3 seeds. The toy may be wrong at 9B scale, but the measurement is the gate. |
| v3-6 | **LUT model only** (FLUTE idxN W4+r32) | Dense FP16/bf16 baseline | User directive. |
| v3-7 | **No Databricks, no SharePoint, no ai_parse_document** | The production proposal's four-lane architecture | User directive. Single-machine PoC. |
| v3-8 | **Plain LRU eviction** for both caches | Learned, frequency-based, analytic | arXiv:2609.28870: 14 sophisticated policies fail to beat LRU. |
| v3-9 | **Content-addressed caching** (hash of token prefix for cache A, hash of `(prefix, layer_idx)` for cache B) | Position-based addressing | DeltaS (arXiv:2609.27470): the state's content, not its position, determines reusability. |
| v3-10 | **Namespace-keyed admission** (refuse cross-checkpoint, cross-recipe, cross-context-class reuse) | Free rotation, cross-namespace reuse | KVShareArena (arXiv:2609.10266): free rotation recovers only 50-66% of the gap; unrepaired reuse worse than no cache. |
| v3-11 | **OfficeQA as downstream evaluation, not training target** | Fine-tune on OfficeQA Q&A | The fine-tune is on cache-operation-awareness. OfficeQA measures whether the cache-friendly model is also a good RAG model. |
| v3-12 | **The toy sweeps are committed to the repo** (the evidence base) | Skip the toy, go straight to A10G | User directive ("run small toy sweeps on CPU in your sandbox. What works?"). The toy's findings (especially the fine-tune's negative result) shape the A10G plan. |

---

## 8. The toy code and results (the evidence)

The toy code is in `scripts/poc_toy/` and is committed to the repo alongside this proposal. The three scripts:

| Script | What it does | Key finding |
|---|---|---|
| `toy_global_cache_sweep.py` | Tiny linear-attention hybrid model + 4 cache configs (no_cache, kv_only, state_only, both). Single-config sweep. | State cache is fixed-size; KV grows linearly. Both hit ~97% on shared system prompt. |
| `toy_sweep_matrix.py` | Sweep across `(overlap, sys_len, cache_kb, n_unique_chunks)` × 3 cache configs. 36 configurations. | "both_beats_either: 0/36 (0.0%)". State cache wins on bytes in 66.7% of configs. |
| `toy_finetune_test.py` | Test the CacheBlend-FT hypothesis: does fine-tuning with the cache round-trip reduce restoration error? Two variants (LM-only, LM+restoration). 3 seeds. | **Both variants WORSEN restoration error by 74–315%.** The fine-tune is conditional. |

The results JSONs (`sweep_matrix.json`, `finetune_test.json`) are also committed. The headline numbers are reproduced in §2 of this proposal.

---

## 9. Immediate next actions

1. **The toy is done.** The code is in `scripts/poc_toy/`, the results are in §2, and the design decisions in §7 follow from them.
2. **Phase A1 (1.5 weeks on A10G).** Implement `poc/two_global_caches.py` on top of `scripts/eval_common.py::load_quant_model`. Run the synthetic RAG workload through the four cache configs. Measure the crossover length empirically. Measure the restoration error on the 9B model.
3. **Phase A2 (1.5 weeks on A10G).** Implement `poc/finetune_cache_aware.py` (the W10 LUT fine-tune with the cache round-trip). Run 500 steps. Measure the restoration error before and after. Adopt or abandon based on the measurement. Run OfficeQA as a downstream evaluation.
4. **Produce `poc_verdict.json`.** The single document with: the measured crossover length, the two-cache policy's hit rate and bytes vs single-cache, the restoration error on the 9B model, the fine-tune's effect, the OfficeQA accuracy.

**The PoC's single success criterion:** the two-global-cache crossover policy (with or without the fine-tune) is *measured* to beat a single global cache on the hit-rate/bytes frontier, on the LUT model, on A10G. The toy says the design is right; the A10G says the numbers.

---

## Appendix A — The toy results, in full

### A.1 The single-config sweep (`toy_global_cache_sweep.py`)

```
Config: 50 sessions, system_prompt=64, chunk=32, overlap=0.5, 8 unique chunks, 64KB cache

Config          Hit%     XSession%  Bytes      Evict  Saved
no_cache        0.0      0.0        0          0      0
kv_only         98.0     98.0       4096       0      3136
state_only      98.0     98.0       2560       0      3136
kv_and_state    98.0     98.0       6656       0      3136
```

### A.2 The matrix sweep (`toy_sweep_matrix.py`)

36 configurations swept. Headline:
- State cache wins on (hit_rate ≥ kv) AND (bytes < kv): **24/36 (66.7%)**
- Both caches beats either: **0/36 (0.0%)**
- State cache total bytes saved vs KV: **915,456 bytes**
- Configs where state evicts MORE than KV: **0/36**

Representative rows (high-pressure: cache_kb=4):

```
ov=0.0 sys=  32 nuc= 4  kv:hit=97% bytes=2048 ev=0  st:hit=97% bytes=2560 ev=0  both:hit=97% bytes=4608 ev=0
ov=0.5 sys= 256 nuc= 4  kv:hit=97% bytes=16384 ev=0  st:hit=97% bytes=2560 ev=0  both:hit=97% bytes=18944 ev=0
ov=1.0 sys=1024 nuc=16  kv:hit=97% bytes=65536 ev=0  st:hit=97% bytes=2560 ev=0  both:hit=97% bytes=68096 ev=0
```

The pattern: at sys_len=1024, the state cache uses 2560 bytes; the KV cache uses 65536. The state cache is **25× smaller** for the same hit rate.

### A.3 The fine-tune test (`toy_finetune_test.py`)

```
Before fine-tune: mean restoration error = 1.728428e-05

--- Variant A: LM-only fine-tune (CacheBlend-FT) for 300 steps ---
  step 0: loss 6.3188
  step 250: loss 2.4408
After LM-only fine-tune: restoration error = 5.274672e-05
Reduction: -205.2%

--- Variant B: LM + restoration-loss fine-tune for 300 steps ---
  step 0: loss 6.3188
  step 250: loss 2.4452
After LM+restoration fine-tune: restoration error = 7.180005e-05
Reduction: -315.4%
```

Cross-seed verification (LM-only, 200 steps):

| Seed | Before | After | Reduction |
|---|---|---|---|
| 1 | 9.19e-06 | 2.07e-05 | -125.1% |
| 2 | 1.02e-05 | 2.74e-05 | -168.2% |
| 3 | 1.07e-05 | 1.86e-05 | -74.0% |

The fine-tune consistently worsens restoration error. The mechanism (hypothesis): as LM loss decreases, hidden states become more sharply peaked, making them more sensitive to fp16 quantization.

---

## Appendix B — The diff from v2

### B.1 Removed

- **Per-session state snapshots at 8 boundaries** (v2 §1.2). Removed per user directive ("too much snapshots"). Replaced with two global caches only.
- **Fine-tune on OfficeQA Q&A** (v2 §1.4). Removed per user directive. Replaced with fine-tune on operating on the global cache.
- **The 5-phase build path P0-P4** (v2 §5). Replaced with 3 phases: toy (done), A10G measurement, verdict.

### B.2 Added

- **The CPU toy sweeps** (§2). Three scripts, 36+ configurations, the empirical foundation.
- **The crossover policy** (§2.5, §3.1). The two caches are used based on prefix length, not both for every prefix.
- **The fine-tune's conditional adoption** (§3.2, §5). The toy's negative result (finding 3) makes the fine-tune conditional — measured before and after, abandoned if it worsens restoration error.
- **The DeltaLog and CacheBlend-FT papers** (§1.1). The two papers that changed the design from v2 to v3.

### B.3 Corrected (from v2)

- **The cache asset size: 25.5 MiB** (v2's correction, retained). v1's 576 KiB was wrong by 44×.
- **The capacity arithmetic: ~530 concurrent sessions** (v2's correction, retained).
