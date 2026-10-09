# PIPELINE v2 — Cache-Engineered RAG on the LUT Model (Single Model, No External Embedder)

> **Supersedes:** `PIPELINE-50k-subsample.md` (which drifted — it introduced an external bge-m3 embedder, a FAISS index, a separate retriever. That's the wrong architecture.)
> **The correction:** everything is the LUT model. The cache IS the RAG. There is no embedder, no index, no retriever, no augmenter — these are all the same model's hidden states and the same model's forward pass.
> **The architecture:** the model's hidden states are the cache assets. Retrieval is a cache lookup. Augmentation is the assembly of cached hidden states into the model's context. The corpus is stored as the model's per-chunk hidden states (snapshots), not as embeddings from a different model.

---

## 0. The single-model architecture (correctly stated)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ THE LUT MODEL (Qwen3.5-9B, FLUTE idxN W4+r32) — the ONLY model          │
│                                                                         │
│  24 linear-attention layers → produce the recurrent state S_L           │
│         (the cache asset for "embed")                                   │
│  8 full-attention layers → produce the KV cache                          │
│         (the cache asset for "augment")                                  │
│                                                                         │
│  Everything below happens INSIDE this model's forward pass.              │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ THE CACHE (the RAG) — two global caches, no per-session snapshots       │
│                                                                         │
│  Cache A: Global Prefix Cache (KV)                                      │
│    • stores: the full-attention KV for token prefixes                   │
│    • used for: the system prompt + retrieved chunk prefixes             │
│    • the "augmentation" cache — assembles retrieved chunks' KV into     │
│      the model's context                                                │
│                                                                         │
│  Cache B: Global Recurrent-State Cache                                  │
│    • stores: the linear-attention recurrent state (K_state, V_state)   │
│      for token prefixes                                                  │
│    • used for: the "embed" of each corpus chunk — the chunk's hidden    │
│      state is its representation in the cache                            │
│    • the "retrieval" cache — looking up a chunk = restoring its state   │
│                                                                         │
│  No embedder. No FAISS. No retriever. No augmenter.                     │
│  The cache IS the embedder (cache B), the retriever (cache B lookup),   │
│  and the augmenter (cache A assembly).                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. What "embed" means (the correction)

**Wrong (v1):** use bge-m3 to embed each chunk into a 1024-dim vector, store in FAISS, retrieve by cos sim.

**Right (v2):** "embed" a chunk = **prefill the chunk through the LUT model, snapshot the linear-attention recurrent state S_L and the full-attention KV for each layer**. The chunk's "embedding" is its hidden state in the model — not a vector from a different model.

```python
# poc/cache_engineer/embed_chunk.py
def embed_chunk(model, chunk_token_ids, cache_b, cache_a):
    """Embed a chunk = prefill it through the model, snapshot the states.

    The chunk's 'embedding' is:
    - Cache B: the linear-attention recurrent state S_L for each of the
      24 linear-attention layers. Shape (1, 32, 128, 128) per layer.
      This is the chunk's representation in the model's linear-attention
      memory — what the model 'remembers' about the chunk.
    - Cache A: the full-attention KV for the 8 full-attention layers.
      Shape (1, 4, chunk_len, 256) per layer. This is the chunk's
      token-level representation — what the model attends to.

    Both are stored in the global caches, keyed by the chunk's content hash.
    A later query that retrieves this chunk = restores these states."""

    # prefill the chunk through the model (no generation, just forward)
    with torch.no_grad():
        logits, recurrent_states, conv_states, kv_states = model.prefill(
            chunk_token_ids, return_states=True)

    # snapshot to the global caches
    chunk_hash = hash_tokens(chunk_token_ids)
    for layer_idx, state in enumerate(recurrent_states):
        cache_b.install(chunk_hash, layer_idx, state, conv_states[layer_idx])
    for layer_idx, (k, v) in enumerate(kv_states):
        cache_a.install(chunk_hash, layer_idx, k, v)

    return chunk_hash  # the chunk's 'embedding ID' — its content hash
```

**The key insight:** the model's hidden states ARE the embeddings. There is no separate embedder. The LUT model prefills the chunk once, snapshots the states to the global caches, and the chunk is "embedded" — its representation lives in the cache, keyed by its content hash.

This is the K3 discipline: the model's own internal state is the cache asset. K3 was trained with this in mind; we apply it to the LUT model.

---

## 2. What "retrieval" means (the correction)

**Wrong (v1):** query the FAISS index with the query embedding, get top-3 chunks by cos sim.

**Right (v2):** "retrieve" = **prefill the query through the model, compute the model's attention scores over all cached chunks' KV (cache A), return the top-3 chunks by attention weight.** The model itself does the retrieval — its full-attention layers attend to the cached chunks' KV.

```python
# poc/cache_engineer/retrieve.py
def retrieve_chunks(model, query_token_ids, cache_a, top_k=3):
    """Retrieve the top-k chunks for a query.

    The model's full-attention layers attend to ALL cached chunks' KV
    (cache A). The chunks with the highest attention scores are 'retrieved'.

    Mechanism:
    1. Prefill the query through the model's linear-attention layers
       (producing the query's recurrent state — cache B lookup).
    2. At each full-attention layer, attend the query's Q against ALL
       cached chunks' K (from cache A). The attention weights tell us
       which chunks the model 'thinks' are relevant.
    3. Aggregate attention weights across the 8 full-attention layers
       and the query's tokens. The top-k chunks by aggregated attention
       are the retrieved chunks.

    This is 'attention as retrieval' — the model's own attention mechanism
    IS the retriever. No external embedder, no FAISS, no cos sim."""

    # prefill the query, capturing attention weights at each full-attn layer
    with torch.no_grad():
        outputs = model.prefill(query_token_ids,
                                  attend_to_cache=cache_a,  # the model attends to all cached KV
                                  output_attentions=True)

    # aggregate attention weights: (layer, head, query_token, cached_chunk)
    # → (cached_chunk,) — sum over layers, heads, query tokens
    chunk_attention = aggregate_attention_to_chunks(outputs.attentions)

    # top-k chunks by attention weight
    top_k_chunks = chunk_attention.topk(top_k)
    return top_k_chunks  # list of chunk_hashes with their attention scores
```

**The key insight:** the model's full-attention layers, when attending to all cached chunks' KV, perform the retrieval. The attention weights ARE the relevance scores. This is "attention as retrieval" — the model is the retriever.

This is more expensive than FAISS (the model must attend to all cached chunks, not a small ANN subset), but it's the model's native mechanism — the retrieved chunks are exactly what the model would attend to anyway. No embedding mismatch, no recall gap between the retriever and the model.

---

## 3. What "augmentation" means (the correction)

**Wrong (v1):** concatenate the retrieved chunks' text into the context contract, re-tokenize, re-prefill.

**Right (v2):** "augment" = **restore the retrieved chunks' KV (from cache A) into the model's full-attention layers' KV cache, in document order.** The model's context is augmented by installing the cached KV — no re-tokenization, no re-prefill. The chunks' KV IS the augmented context.

```python
# poc/cache_engineer/augment.py
def augment_context(model, query_token_ids, retrieved_chunk_hashes,
                    cache_a, cache_b, system_prompt_hash):
    """Augment the model's context with the retrieved chunks.

    Mechanism:
    1. Restore the system prompt's KV (from cache A) into the model's
       full-attention layers. (The system prompt is cached; restoring
       it skips its prefill.)
    2. Restore the system prompt's recurrent state (from cache B) into
       the model's linear-attention layers. (Skips the linear-attn prefill.)
    3. For each retrieved chunk, in DOCUMENT ORDER (cache-friendly —
       preserves longest common prefixes):
       a. Restore the chunk's KV (from cache A) — appends to the model's
          full-attention KV cache. No re-prefill of the chunk.
       b. Restore the chunk's recurrent state (from cache B) — updates
          the model's linear-attention state. No re-prefill.
    4. Prefill the query (the only thing actually computed) on top of
       the restored context.

    The augmented context is the model's KV cache + recurrent state
    after the restores. No text concatenation, no re-tokenization."""

    # 1. restore the system prompt
    system_kv = cache_a.lookup(system_prompt_hash)
    system_state = cache_b.lookup(system_prompt_hash)
    model.restore_kv(system_kv)
    model.restore_recurrent_state(system_state)

    # 2. restore the retrieved chunks in document order
    for chunk_hash in retrieved_chunk_hashes:  # already sorted by doc, then page
        chunk_kv = cache_a.lookup(chunk_hash)
        chunk_state = cache_b.lookup(chunk_hash)
        model.append_kv(chunk_kv)  # append to the full-attn KV cache
        model.update_recurrent_state(chunk_state)  # update the linear-attn state

    # 3. prefill the query on top of the restored context
    with torch.no_grad():
        logits = model.prefill(query_token_ids)

    return logits
```

**The key insight:** augmentation is the assembly of cached KV and recurrent states into the model's context — not text concatenation. The model's KV cache and recurrent state ARE the augmented context. The chunks' KV is appended; the recurrent state is updated. No re-tokenization, no re-prefill of the retrieved chunks.

This is the K3 cache-engineering discipline: the cache IS the context. Retrieval installs the cached chunks' states; the model continues from the augmented state. The prefill cost is paid once per chunk (at embed time, stored in the cache), not per query.

---

## 4. The end-to-end flow (single model, cache IS the RAG)

```
                    ┌─────────────────────────────────┐
                    │ THE LUT MODEL (Qwen3.5-9B W4)   │
                    │  24 linear-attn + 8 full-attn   │
                    └─────────────────────────────────┘
                              │       │
              ┌───────────────┘       └───────────────┐
              ▼                                       ▼
    ┌───────────────────┐                   ┌───────────────────┐
    │ Cache B            │                   │ Cache A            │
    │ (recurrent state)  │                   │ (KV prefix)        │
    │                    │                   │                    │
    │ EMBED step:        │                   │ AUGMENT step:      │
    │ prefill each chunk │                   │ restore chunks' KV │
    │ → snapshot S_L     │                   │ into the model's   │
    │ → store in cache B │                   │ full-attn layers   │
    │                    │                   │                    │
    │ RETRIEVE step:     │                   │                    │
    │ (the model's       │                   │                    │
    │  full-attn attends │                   │                    │
    │  to cache A's KV)  │                   │                    │
    └───────────────────┘                   └───────────────────┘
              │                                       │
              └───────────────┬───────────────────────┘
                              ▼
                    ┌─────────────────────────────────┐
                    │ THE QUERY FLOW                  │
                    │                                 │
                    │ 1. Restore system prompt's      │
                    │    KV (cache A) + state (cache B)│
                    │ 2. The model's full-attn layers │
                    │    attend to ALL cached chunks' │
                    │    KV (cache A) → attention      │
                    │    weights = retrieval scores   │
                    │ 3. Top-k=3 chunks by attention  │
                    │ 4. Restore the 3 chunks' KV     │
                    │    (cache A) in document order  │
                    │ 5. Restore the 3 chunks' state  │
                    │    (cache B)                   │
                    │ 6. Prefill the query on top of  │
                    │    the augmented context        │
                    │ 7. Decode the answer            │
                    └─────────────────────────────────┘
```

### 4.1 The three operations, restated

| Operation | v1 (wrong) | v2 (correct) |
|---|---|---|
| **Embed** | bge-m3 → 1024-dim vector → FAISS | LUT model prefill → snapshot recurrent state (cache B) + KV (cache A) |
| **Retrieve** | FAISS HNSW → top-3 by cos sim | LUT model's full-attn attends to all cached KV (cache A) → top-3 by attention weight |
| **Augment** | concatenate text → re-tokenize → re-prefill | restore cached KV (cache A) + recurrent state (cache B) into the model → no re-prefill |

### 4.2 The single model does everything

- The LUT model prefills the corpus chunks → embeds them (stores states in the caches)
- The LUT model's attention mechanism retrieves (attends to cached KV)
- The LUT model's KV cache + recurrent state IS the augmented context

No bge-m3. No FAISS. No separate embedder. No separate retriever. The cache IS the RAG, and the LUT model IS the engine.

---

## 5. Dataset preparation (the corrected version)

### 5.1 What we prepare

The corpus is **OfficeQA's Treasury Bulletins**, but we don't embed them with bge-m3. We **prefill them through the LUT model** and snapshot the hidden states to the global caches.

### 5.2 The preparation pipeline

```
Stage 1: Ingest          Stage 2: Parse         Stage 3: Chunk
  PDFs (bronze)    →     docling layout-   →    structure-based
  raw bytes              aware markdown          1024-token chunks
                         (silver)               (gold candidate)

Stage 4: Tokenize       Stage 5: Embed (the LUT model)
  int32 token IDs  →     prefill each chunk through the model
  (the model's           → snapshot recurrent state (cache B) + KV (cache A)
  own tokenizer)         → the chunk is now 'embedded' — its states are
                         in the global caches, keyed by content hash
```

**Stages 1-3** are unchanged from v1 (ingest PDFs, parse with docling, chunk structure-based). The chunker is still needed — we need to split the 89,000-page corpus into 1024-token chunks before the model can prefill them (the model's prefill is O(L²) in sequence length; 1024 tokens is the sweet spot).

**Stage 4** is the model's own tokenizer (not a separate embedder's tokenizer). The LUT model's `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")` tokenizes each chunk into int32 token IDs.

**Stage 5** is the embed step, corrected:

```python
# poc/prep/05_embed_lut.py
def embed_chunks_with_lut_model(model, chunks, cache_a, cache_b, device):
    """Prefill each chunk through the LUT model, snapshot the hidden states.

    For each chunk:
    1. Tokenize with the model's own tokenizer.
    2. Prefill through the model (forward pass, no generation).
    3. Snapshot the 24 linear-attention recurrent states → cache B.
    4. Snapshot the 8 full-attention KV pairs → cache A.
    5. Key by the chunk's content hash.

    This is the 'embed' step. The chunk's embedding is its hidden state
    in the LUT model — not a vector from a different model."""

    tokenizer = model.tokenizer
    for chunk in chunks:
        token_ids = tokenizer(chunk.text, return_tensors="pt").input_ids.to(device)
        chunk_hash = hash_tokens(token_ids)

        with torch.no_grad():
            logits, recurrent_states, conv_states, kv_states = model.prefill(
                token_ids, return_states=True)

        # snapshot to the global caches
        for layer_idx in range(24):  # 24 linear-attention layers
            cache_b.install(chunk_hash, layer_idx,
                            recurrent_states[layer_idx], conv_states[layer_idx])
        for layer_idx in range(8):  # 8 full-attention layers
            cache_a.install(chunk_hash, layer_idx,
                            kv_states[layer_idx][0], kv_states[layer_idx][1])

    # the chunks are now 'embedded' — their states are in the caches
    return len(chunks)
```

### 5.3 The disk footprint (corrected)

The chunks' **text and token IDs** are stored on disk (for provenance and re-embedding on model upgrades). The **hidden states** are stored in the caches (HBM or host DRAM, depending on capacity — see `SIZING-officeqa-cache.md`).

| Component | Disk | HBM / Host DRAM (the caches) |
|---|---|---|
| Raw PDFs (bronze) | ~8.7 GiB | — |
| Parsed markdown (silver) | ~1.5 GiB | — |
| Chunk text + token IDs (gold) | ~0.5 GiB | — |
| Hidden states (the "embeddings") | — | Cache A: 8 layers × KV per chunk; Cache B: 24 layers × 25.5 MiB per session-equivalent |

**The disk footprint is the same as v1** (~11 GiB, dominated by raw PDFs). The difference is that the "embeddings" are not on disk — they're in the caches, as the model's hidden states.

### 5.4 The cache sizing (corrected)

This is where it gets real. The 50k subsample means **50,000 chunks' hidden states in the caches**.

**Cache B (recurrent state) per chunk:**
- 24 layers × (1, 32, 128, 128) fp16 = 24 × 1 MiB = **24 MiB per chunk**
- 50,000 chunks × 24 MiB = **1,171,584 MiB ≈ 1.14 TiB**

**Cache A (KV) per chunk (1024-token chunks):**
- 8 layers × (1, 4, 1024, 256) × 2 (K+V) × 2 (fp16) = 8 × 4 MiB = **32 MiB per chunk**
- 50,000 chunks × 32 MiB = **1,600,000 MiB ≈ 1.53 TiB**

**Total cache for 50k chunks: ~2.67 TiB**

This is the same magnitude as `SIZING-officeqa-cache.md` computed (1.3 TiB for cache B alone) — but now cache A is ALSO needed (the KV for augmentation), doubling the requirement.

**On A10G (13 GiB cache budget):**
- 13 GiB / (24 + 32) MiB per chunk = ~233 hot chunks (0.47% of 50k)
- The LRU eviction paper's finding (96 GiB → <0.2% cached → high hit rate) suggests 13 GiB → moderate hit rate, dominated by the system prompt + the most-retrieved chunks

**For production (host DRAM, the K3 external pool):**
- ~2.67 TiB of host DRAM (~$270 at $0.10/GiB)
- Or: a 96 GiB HBM replica per the LRU paper, with LRU eviction caching the hot ~1,700 chunks (3.4% of 50k)

**The honest reality:** cache-engineered RAG on the full 50k corpus requires either (a) a large host-DRAM pool (~2.67 TiB, the K3 pattern) or (b) aggressive LRU eviction on HBM (~96 GiB, caching ~3% of chunks). The PoC on a single A10G can only cache ~233 chunks — enough to validate the concept on the PoC's 50 questions, but not enough for the full 50k subsample without LRU.

---

## 6. Retrieval (the corrected version)

### 6.1 The retrieval mechanism (attention as retrieval)

```python
# poc/cache_engineer/retrieve.py
def retrieve_chunks(model, query_token_ids, cache_a, top_k=3):
    """Retrieve the top-k chunks for a query.

    The model's full-attention layers attend to ALL cached chunks' KV
    (cache A). The attention weights ARE the retrieval scores.

    Mechanism:
    1. The query is prefilled through the model's linear-attention layers
       (producing the query's recurrent state — cache B lookup, or fresh
       prefill if the query prefix isn't cached).
    2. At each full-attention layer, the query's Q attends to ALL cached
       chunks' K (from cache A). This is a single batched matmul:
         attention_scores = Q @ K_all^T  (shape: query_tokens × all_cached_tokens)
    3. The attention weights are aggregated across:
       - the 8 full-attention layers (sum or max)
       - the query's tokens (sum, weighted by the query's attention to
         each token)
       - the heads (sum, or per-head top-k then merge)
    4. The top-k chunks by aggregated attention weight are returned.

    This is 'attention as retrieval'. The model itself is the retriever."""

    # prefill the query, capturing attention weights
    with torch.no_grad():
        outputs = model.prefill(query_token_ids,
                                  attend_to_all_cached_kv=cache_a,  # the model attends to cache A's KV
                                  output_attentions=True)

    # aggregate: (layer, head, query_token, cached_chunk) → (cached_chunk,)
    chunk_scores = torch.zeros(len(cache_a))
    for layer_idx in range(8):  # 8 full-attn layers
        attn = outputs.attentions[layer_idx]  # (1, heads, q_len, all_kv_len)
        # map each KV position to its chunk
        chunk_attn = scatter_attention_to_chunks(attn, cache_a.chunk_offsets)
        chunk_scores += chunk_attn.sum(dim=(0, 2))  # sum over heads and query tokens

    top_k_chunks = chunk_scores.topk(top_k).indices
    return top_k_chunks
```

### 6.2 The retrieval latency

The model attends to ALL cached chunks' KV. This is O(query_len × total_cached_tokens), not O(query_len × k) like FAISS.

| Cached chunks | Total cached tokens | Attention matmul size | Latency (A10G, fp16) |
|---|---|---|---|
| 233 (A10G LRU) | 233 × 1024 = 239k | 64 × 239k | ~3 ms |
| 1,700 (96 GiB) | 1,700 × 1024 = 1.74M | 64 × 1.74M | ~22 ms |
| 50,000 (full) | 50,000 × 1024 = 51.2M | 64 × 51.2M | ~650 ms |

**On A10G with LRU (233 chunks): retrieval is ~3 ms.** Comparable to FAISS (~10 ms in v1). The model's attention IS the retriever — no separate index, no embedding mismatch.

**For the full 50k (no LRU): ~650 ms per retrieval.** This is the cost of "attention as retrieval" at scale — the model attends to everything. The mitigation is the LRU eviction (only hot chunks are cached) or the K3 external pool (the attention is computed over the hot subset).

### 6.3 The retrieval quality gate

The retrieval quality is measured the same way as v1 (recall@3 document-level ≥ 80%), but the retrieval mechanism is the model's attention, not cos sim. If the recall is below 80%, the issue is the model's attention (not the embedder/index) — tuning options are limited to the model itself (which is why the cache-aware fine-tune matters: it trains the model to attend well to the cached chunks).

---

## 7. Augmentation (the corrected version)

### 7.1 The augmentation mechanism (cache restoration)

```python
# poc/cache_engineer/augment.py
def augment_context(model, query_token_ids, retrieved_chunk_hashes,
                    cache_a, cache_b, system_prompt_hash):
    """Augment the model's context with the retrieved chunks by restoring
    their cached states. No text concatenation, no re-prefill.

    The model's KV cache and recurrent state ARE the augmented context."""

    # 1. restore the system prompt's states (the shared prefix)
    sys_kv = cache_a.lookup(system_prompt_hash)
    sys_state = cache_b.lookup(system_prompt_hash)
    model.restore_full_attn_kv(sys_kv)
    model.restore_linear_attn_state(sys_state)

    # 2. restore the retrieved chunks' states in DOCUMENT ORDER
    for chunk_hash in retrieved_chunk_hashes:  # sorted by doc, then page
        chunk_kv = cache_a.lookup(chunk_hash)
        chunk_state = cache_b.lookup(chunk_hash)
        model.append_full_attn_kv(chunk_kv)       # append to the KV cache
        model.update_linear_attn_state(chunk_state)  # update the recurrent state

    # 3. prefill the query on top of the restored context
    with torch.no_grad():
        logits = model.prefill(query_token_ids)

    return logits
```

### 7.2 The augmentation cost

| Step | Cost |
|---|---|
| Restore system prompt KV (8 layers × ~500 tokens) | ~16 MiB transfer, ~0.25 ms |
| Restore system prompt state (24 layers) | ~25.5 MiB transfer, ~0.4 ms |
| Restore 3 chunks' KV (8 layers × 3 × 1024 tokens) | ~96 MiB transfer, ~1.5 ms |
| Restore 3 chunks' state (24 layers × 3) | ~76.5 MiB transfer, ~1.2 ms |
| Prefill the query (~32 tokens on top of the restored context) | ~2 ms |
| **Total augmentation** | **~5.4 ms** |

vs. v1's re-prefill of (system + 3 chunks + query) = ~500 + 3072 + 32 = 3,604 tokens → ~50-100 ms. **The cache restoration is ~10-20× faster than re-prefill.** This is the cache-engineering win — the Contiguity paper's 13-21× ratio, applied to the full augmentation, not just edit repair.

---

## 8. The benchmark protocol (corrected)

The benchmark is the same structure as v1 (5 configs, meaningful-query filter, staleness envelope, two-oracle), but the measurements reflect the single-model cache-engineered RAG:

### 8.1 The benchmark configurations

| Config | Cache A (KV) | Cache B (state) | Fine-tune | What it tests |
|---|---|---|---|---|
| **A: no cache** | off | off | off | The floor — every query re-prefills the full context |
| **B: KV only** | on | off | off | Augmentation cache only (the vLLM pattern) |
| **C: state only** | off | on | off | Embed/retrieve cache only (no KV augmentation) |
| **D: both (the proposal)** | on | on | off | The full cache-engineered RAG |
| **E: both + fine-tune** | on | on | on (cache-aware) | The full proposal + the fine-tune |

### 8.2 The measurements

| Metric | How measured | Target |
|---|---|---|
| Accuracy | OfficeQA gold exact-match + numeric-tolerance | Config D ≥ Config A - 2% |
| Retrieval recall@3 | Document-level, via the model's attention weights | ≥ 80% |
| TTFT | Time from query to first decoded token | Config D ≤ Config A / 5 |
| ITL | Inter-token latency during decode | Config D ≈ Config A |
| Throughput | tok/s (decode) | Config D ≥ Config A × 1.5 |
| VRAM peak | `torch.cuda.max_memory_allocated` | Config D ≤ 22 GiB |
| Cache hit rate | % of prefill tokens saved by cache restoration | ≥ 90% (the cache pays for itself) |
| Restoration error | ‖logits_fresh - logits_restored‖ / ‖logits_fresh‖ | ≤ 1e-3 (the fp16 round-trip drift) |
| Staleness envelope | Accuracy vs doc-edit distance | Edit-local repair within 13-21× of re-prefill |
| Two-oracle gap | Belady vs BeladyCompute on a trace sample | <5% → ship LRU |

### 8.3 The meaningful-query filter (unchanged)

The BoxOffice gate still applies: drop model-capability failures, world-knowledge questions, low-information yes/no. The cut list is a finding.

### 8.4 The fine-tune (the cache-aware training)

The fine-tune trains the LUT model to operate on the global caches — the CacheBlend-FT idea (arXiv 2609.09768). The model is trained with the cache round-trip (snapshot to fp16, restore, continue) in the forward pass, so it learns to produce states that survive restoration.

**The fine-tune's data is the corpus chunks themselves** (not OfficeQA Q&A). The model prefills chunks, snapshots their states, restores them, and continues — learning to be cache-friendly. OfficeQA is a downstream evaluation, not a training target.

**The fine-tune's gate:** if the restoration error worsens (the toy's prediction), abandon the fine-tune; use Config D without it.

---

## 9. The corrected summary

### 9.1 The architecture in one sentence

The LUT model prefills the corpus chunks and snapshots their hidden states (recurrent state + KV) to two global caches; retrieval is the model's attention over the cached KV; augmentation is the restoration of the cached states into the model's context.

### 9.2 The three operations, restated

| Operation | v1 (wrong — separate components) | v2 (correct — single LUT model) |
|---|---|---|
| **Embed** | bge-m3 → FAISS vector | LUT model prefill → cache B (recurrent state) + cache A (KV) |
| **Retrieve** | FAISS HNSW cos sim → top-3 | LUT model's full-attn attends to cache A's KV → top-3 by attention |
| **Augment** | concatenate text → re-tokenize → re-prefill | restore cache A's KV + cache B's state into the model → no re-prefill |

### 9.3 The cache sizing (the real constraint)

| Corpus | Cache B (state) | Cache A (KV) | Total cache | A10G fit? |
|---|---|---|---|---|
| PoC (50 Q, top-3) | ~7.6 GiB | ~3.0 GiB | ~10.6 GiB | yes (of 13 GiB budget) |
| 50k subsample (all) | 1.14 TiB | 1.53 TiB | ~2.67 TiB | no (needs host DRAM) |
| 50k subsample (LRU, 233 hot) | 5.6 GiB | 7.5 GiB | ~13.1 GiB | yes (tight, fills the budget) |

### 9.4 The latency budget (per query, A10G with LRU)

| Step | Latency |
|---|---|
| Restore system prompt (cache A + B) | ~0.65 ms |
| Retrieve (attention over 233 cached chunks' KV) | ~3 ms |
| Restore top-3 chunks (cache A + B) | ~2.7 ms |
| Prefill the query (32 tokens on restored context) | ~2 ms |
| Decode the answer (~50 tokens) | ~2 s (at ~25 tok/s) |
| **Total** | **~2.01 s per query** |

vs. v1's ~2.1 s (with FAISS retrieval at ~10 ms). The cache-engineered RAG is comparable in latency, but with NO external embedder, NO FAISS index, and NO re-prefill of the retrieved chunks.

---

## 10. What this changes about the PoC

### 10.1 What's removed

- **bge-m3** (the external embedder) — removed. The LUT model does the embedding.
- **FAISS HNSW** (the external index) — removed. The LUT model's attention does the retrieval.
- **The separate retriever** — removed. The model's forward pass IS the retrieval.
- **The text concatenation augmentation** — removed. The cache restoration IS the augmentation.
- **The 1024-dim fp16 embedding vectors** — removed. The model's hidden states ARE the embeddings.

### 10.2 What's added

- **The corpus prefill** (the "embed" step): prefill all 50k chunks through the LUT model, snapshot their states to the caches. This is the one-time cost of building the cache-engineered RAG.
- **Attention as retrieval**: the model's full-attn layers attend to the cached KV. The attention weights are the retrieval scores.
- **Cache restoration as augmentation**: restore the retrieved chunks' states into the model. No re-prefill.

### 10.3 The one-time corpus prefill cost

To embed the 50k subsample (prefill each chunk through the LUT model):

| Step | Per-chunk cost | 50k chunks | Time (A10G) |
|---|---|---|---|
| Tokenize | ~1 ms | 50k | ~50 s |
| Prefill (1024 tokens, 32 layers) | ~50 ms | 50k | ~42 min |
| Snapshot to caches (24 + 8 states) | ~5 ms | 50k | ~4 min |
| **Total corpus prefill** | | | **~46 min** |

The one-time cost of building the cache-engineered RAG on 50k chunks is **~46 minutes on A10G**. After that, every query is cache lookup + restore + small prefill, not full re-prefill.

---

## 11. The honest constraints

### 11.1 The cache capacity is the binding constraint

The 50k subsample's hidden states (cache A + B) total ~2.67 TiB. On A10G (13 GiB cache budget), we can only cache ~233 chunks (0.47% of 50k). The LRU eviction paper says this still yields a meaningful hit rate (RAG retrieval is power-law — the system prompt + top chunks dominate), but the full 50k cannot fit in HBM.

**The path to the full 50k:** the K3 external cache pool — host DRAM (~2.67 TiB, ~$270) with GPU↔host transfers (~2 ms per chunk restore at PCIe Gen4). This is a production concern, not a PoC concern.

### 11.2 The attention-as-retrieval cost scales linearly with cached chunks

Retrieval = the model's full-attn attends to all cached KV. On A10G with 233 cached chunks, this is ~3 ms. With 50,000 chunks (if they all fit), it's ~650 ms — too slow for interactive use.

**The mitigation:** LRU eviction (only hot chunks are cached, keeping the attention matmul small). The LRU paper's finding (96 GiB → high hit rate on <0.2% of unique blocks) suggests that on A10G (13 GiB), the 233 hot chunks cover the bulk of retrievals for power-law-distributed queries.

### 11.3 The fine-tune's uncertainty (from the toy)

The toy found that cache-aware fine-tuning worsens restoration error by 74-315%. The PoC measures this on the real model — if it holds, the fine-tune is abandoned and the caches use the pre-fine-tune LUT model. The cache-engineered RAG works without the fine-tune; the fine-tune is an enhancement that may not pay off.

---

## 12. The corrected PoC timeline

| Phase | What | Time | Output |
|---|---|---|---|
| **P0. Prep** | Ingest + parse + chunk the corpus (docling, structure-based) | 4-6 hours | `data/gold/chunks.jsonl` (50k chunks) |
| **P1. Corpus prefill** | Prefill all 50k chunks through the LUT model, snapshot to caches | 46 min | The two global caches populated |
| **P2. Benchmark sweep** | Run configs A-E on the 50 OfficeQA questions | 2 hours | 5 benchmark reports |
| **P3. Staleness envelope** | The doc-edit envelope (edit-local repair vs re-prefill) | 2 hours | `staleness_envelope.json` |
| **P4. Two-oracle diagnostic** | Belady vs BeladyCompute on a trace sample | 1 hour | `two_oracle.json` |
| **P5. Verdict** | Assemble `poc_verdict.json` | 30 min | The go/no-go |
| **Total** | | **~1 day** | The full PoC |

**The PoC is faster than v1** (~1 day vs ~2 days) because there's no FAISS index to build, no bge-m3 to run, and no separate retrieval step. The corpus prefill (46 min) replaces the embed step (30-60 min) and is comparable in time.

---

## Appendix — the diff from v1 (PIPELINE-50k-subsample.md)

### Removed
- bge-m3 (the external embedder)
- FAISS HNSW (the external index)
- The SQLite metadata sidecar
- The separate retriever component
- The text concatenation augmentation
- The cos sim reranking step

### Changed
- "Embed" = LUT model prefill + state snapshot (was: bge-m3 → vector)
- "Retrieve" = model's attention over cached KV (was: FAISS HNSW cos sim)
- "Augment" = restore cached states into the model (was: concatenate text + re-prefill)
- The cache sizing doubles (cache A is now needed for augmentation, not just the system prompt)
- The retrieval latency grows with cached chunks (O(N) attention, not O(log N) HNSW)

### Added
- The corpus prefill step (the one-time "embed" of all 50k chunks)
- Attention as retrieval (the model's own attention mechanism)
- Cache restoration as augmentation (no re-prefill)
- The honest constraint that 50k chunks need ~2.67 TiB of cache (host DRAM for production, LRU for PoC)
