# PROCESS — The Exact End-to-End (IVFADC + Two Global Caches)

> **The architecture, in one line:** IVFADC on disk preselects the top-100 candidate chunks cheaply; cos sim reranks to top-3; our two global caches (KV prefix + recurrent state) restore the top-3 chunks' states into the model; only the query is prefilled.
> **What's on disk:** the IVFADC index (chunk vectors, PQ-coded) + the chunk text/token IDs. **What's in HBM (A10G):** the model weights + the two global caches (hot prefixes only, LRU-evicted). **What's in host DRAM (optional, for overflow):** the cold cache entries.
> **The key tension, stated upfront:** the chunk's cached states are snapshotted from an *independent* prefill (one per chunk, for IVFADC compatibility). The toy showed this has a ~4.1e-3 logit diff vs. re-prefill-in-document-order. This is the **restoration error** — the cost of caching. It is small, measurable, and (if too high) mitigated by re-prefilling only the top-3 (cheap).

---

## 1. The three stages, exactly

### 1.1 SNAPSHOTTING (one-time, per chunk — the "embed" step)

**Goal:** for each of the 50,000 corpus chunks, produce (a) an IVFADC vector for the on-disk index, and (b) the cached states (KV + recurrent) for augmentation.

```
For each chunk c (50,000 total):
  1. Tokenize c with the LUT model's tokenizer → token_ids (1024 tokens)
  2. Prefill token_ids through the LUT model (INDEPENDENTLY — no prefix, no system prompt)
     → produces, per layer:
       - linear-attn recurrent state S_L(c)   [shape (1, 32, 128, 128) fp16, 1 MiB]
       - linear-attn conv state C_L(c)         [shape (1, 8192, 4) fp16, 64 KiB]
       - full-attn KV pair (K_c, V_c)          [shape (1, 4, 1024, 256) fp16, 2 MiB per layer × 8]
       - the final hidden state h(c)           [shape (1, 1024, 4096) fp16]
  3. Extract the IVFADC vector v(c):
     - mean-pool h(c) over the sequence dimension → v(c) [shape (4096,) fp16, 8 KiB]
     - (this is the chunk's "embedding" for IVFADC — the LUT model's own representation)
  4. Store on disk (IVFADC index):
     - PQ-encode v(c) → 64 bytes (the IVFADC code)
     - append to the IVFADC inverted list
  5. Store in the global caches (HBM or host DRAM):
     - cache_b.install(chunk_hash(c), S_L(c), C_L(c))    [1.06 MiB per chunk]
     - cache_a.install(chunk_hash(c), K_c, V_c)         [16 MiB per chunk (8 layers × 2 MiB)]
  6. Store on disk (chunk payload):
     - chunk text + token_ids + metadata (doc_id, page, section) → SQLite/mmap
```

**Per-chunk cost:** ~50 ms prefill + ~5 ms snapshot = ~55 ms. **50k chunks: ~46 minutes on A10G.**

**Disk footprint:** IVFADC codes (50k × 64 B = 3 MiB) + exact vectors for rerank (50k × 8 KiB = 390 MiB) + chunk payloads (50k × ~10 KiB = 490 MiB) = **~880 MiB on disk.**

**Cache footprint (the binding constraint):** 50k × (1.06 + 16) MiB = **~833 GiB** if caching ALL chunks. On A10G's 13 GiB cache budget: ~156 hot chunks (LRU). For production: host DRAM (~833 GiB, ~$83) or 96 GiB HBM replica (~1,150 hot chunks).

---

### 1.2 RETRIEVAL (per query — the IVFADC preselect)

**Goal:** given a query, find the top-3 most relevant chunks. Cheap (on disk), fast (~5 ms).

```
For a query q:
  1. Tokenize q → query_token_ids (32 tokens)
  2. Prefill q through the LUT model (independently) → query hidden state h(q)
  3. Mean-pool h(q) → query vector v(q) [shape (4096,) fp16]
     - ~5 ms on GPU (the prefill), ~negligible (the pool)
  4. IVFADC preselect on disk (in CPU RAM, the index is mmap'd):
     - coarse quantizer: find the ~√N = 224 nearest clusters to v(q)
     - PQ scan: approximate distance from v(q) to all vectors in those clusters
     - returns top-100 candidate chunk IDs + approximate distances
     - ~2-3 ms (the index is ~880 MiB, mmap'd; only the probed clusters are read)
  5. Cos sim rerank (in CPU RAM):
     - load the 100 candidates' EXACT fp16 vectors v(c) from disk (mmap, ~800 KiB)
     - compute cos sim (dot product, since vectors are normalized) for each
     - sort → top-3 chunk IDs
     - ~1 ms
  6. Return top-3 chunk IDs + their scores
```

**Total retrieval: ~8 ms per query** (5 ms query prefill + 3 ms IVFADC + 1 ms rerank).

**Retrieval quality gate:** recall@3 (document-level) ≥ 80%, measured against the OfficeQA gold document references. If below, tune IVFADC's `nprobe` (number of clusters probed) or check the embedder (the LUT model's pooled hidden state may not be discriminative enough — the cache-aware fine-tune addresses this).

---

### 1.3 AUGMENTATION (per query — the cache restoration)

**Goal:** install the top-3 chunks' cached states into the model, then prefill only the query. No re-prefill of the chunks (unless the restoration error is too high).

```
For a query q with retrieved top-3 chunks [c1, c2, c3] (in document order):

  1. Restore the system prompt's cached states (always a cache hit):
     - sys_hash = hash(system_prompt_token_ids)
     - sys_kv = cache_a.lookup(sys_hash)     → restore into model's full-attn KV cache
     - sys_state = cache_b.lookup(sys_hash)  → restore into model's linear-attn state
     - ~0.65 ms (16 MiB KV + 25.5 MiB state transfer from HBM/DRAM)

  2. For each chunk ci (i = 1, 2, 3) in document order:
     a. Look up the chunk's INDEPENDENT cached states:
        - chunk_hash = hash(ci.token_ids)
        - chunk_kv = cache_a.lookup(chunk_hash)     → (K_ci, V_ci)
        - chunk_state = cache_b.lookup(chunk_hash)   → (S_L(ci), C_L(ci))
     b. If cache MISS (chunk not in the hot LRU set):
        - read ci.token_ids from disk (the SQLite/mmap payload, ~4 KiB)
        - re-prefill ci through the model ON TOP OF the current accumulated state
          (this is the lossless path — the chunk is prefilled in context)
        - snapshot the resulting states → install in the caches (LRU may evict)
        - skip to step 3 (the state is already correct, no restore needed)
     c. If cache HIT:
        - append chunk_kv to the model's full-attn KV cache:
          model.kv_cache = cat([model.kv_cache, chunk_kv], dim=2)
        - update the model's linear-attn state:
          model.linear_state = chunk_state.S_L
          model.conv_state = chunk_state.C_L
        - NOTE: this is the INDEPENDENT state (snapshotted without prefix).
          It has a ~4.1e-3 logit diff vs. the lossless re-prefill-in-context.
          This is the restoration error — measured per query.

  3. Prefill the query (32 tokens) on top of the restored context:
     - model.prefill(query_token_ids, linear_state, conv_state, kv_cache)
     - ~2 ms (only 32 tokens; the context is already in the caches)

  4. Decode the answer (~50-200 tokens):
     - model.decode(max_new_tokens=200)
     - ~2-8 s (at ~25 tok/s on A10G)

  5. (Optional, if restoration error matters) Measure the error:
     - re-prefill the top-3 chunks in document order (lossless)
     - compare logits → if diff > threshold (e.g., 1e-2), log a "restoration miss"
     - the re-prefill is the fallback; the cache is the fast path
```

**Total augmentation (cache hit path): ~5 ms** (0.65 ms system restore + 3 × 1.45 ms chunk restore + 2 ms query prefill).

**Total augmentation (cache miss path): ~155 ms** (0.65 ms system + 3 × 50 ms chunk re-prefill + 2 ms query). Still faster than re-prefilling the full 50k corpus.

---

## 2. The exact data flow, end to end

```
                        ┌─────────────────────────────────────┐
                        │ DISK (persistent)                   │
                        │                                     │
                        │ • IVFADC index (PQ codes)    3 MiB │
                        │ • Exact vectors (for rerank) 390 MiB│
                        │ • Chunk payloads (text+tokens) 490 MiB│
                        │ • Raw PDFs (bronze)          8.7 GiB│
                        │                                     │
                        │ Total disk: ~9.6 GiB                │
                        └──────────┬──────────────────────────┘
                                   │
                      IVFADC preselect (top-100)
                      cos sim rerank (top-3)
                                   │
                                   ▼ top-3 chunk IDs
                        ┌─────────────────────────────────────┐
                        │ HBM (A10G, 24 GiB)                  │
                        │                                     │
                        │ • Model weights (W4+r32)    5.85 GiB│
                        │ • Cache A (KV, hot chunks)  ~6 GiB  │
                        │ • Cache B (state, hot chunks) ~0.4 GiB│
                        │ • Working state (KV+state)   ~0.3 GiB│
                        │ • Framework/safety           ~5 GiB │
                        │                                     │
                        │ Total HBM: ~18 GiB (fits)            │
                        └──────────┬──────────────────────────┘
                                   │
                                   ▼ top-3 chunks' states restored
                        ┌─────────────────────────────────────┐
                        │ THE QUERY FLOW                      │
                        │                                     │
                        │ 1. Embed query → v(q)        ~5 ms │
                        │ 2. IVFADC + rerank → top-3   ~4 ms │
                        │ 3. Restore system prompt    ~0.65 ms│
                        │ 4. Restore top-3 chunks     ~4.4 ms │
                        │ 5. Prefill query (32 tok)   ~2 ms  │
                        │ 6. Decode answer (~100 tok) ~4 s   │
                        │                                     │
                        │ Total: ~4.02 s per query             │
                        └─────────────────────────────────────┘
```

---

## 3. The two global caches, exactly

### 3.1 Cache A — Global KV Prefix Cache

| Property | Value |
|---|---|
| Stores | Full-attention KV pairs (K, V) for the 8 full-attn layers |
| Key | `hash(token_prefix)` — content-addressed |
| Per-entry size | 8 layers × (1, 4, prefix_len, 256) × 2 (K+V) × 2 (fp16) = `prefix_len × 32 KiB` |
| For a 1024-token chunk | 32 MiB per chunk |
| For the system prompt (500 tokens) | 15.6 MiB |
| Eviction | LRU |
| Location | HBM (hot) → host DRAM (cold, optional overflow) |
| Hit policy | On hit: append (K, V) to the model's full-attn KV cache. No re-prefill. |

### 3.2 Cache B — Global Recurrent-State Cache

| Property | Value |
|---|---|
| Stores | Linear-attn recurrent state S_L + conv state C_L for the 24 linear-attn layers |
| Key | `hash(token_prefix)` — content-addressed |
| Per-entry size | 24 × (1, 32, 128, 128) × 2 (fp16) + 24 × (1, 8192, 4) × 2 = 25.5 MiB per chunk (fixed, regardless of chunk length) |
| Eviction | LRU |
| Location | HBM (hot) → host DRAM (cold, optional overflow) |
| Hit policy | On hit: restore S_L and C_L into the model's linear-attn state. No re-prefill. |

### 3.3 The crossover (when to use which cache)

From `SIZING-officeqa-cache.md` (corrected): the crossover is ~816 tokens. Below 816 tokens, cache A (KV) is smaller. Above 816 tokens, cache B (state) is smaller.

**Practical implication:** the system prompt (~500 tokens) uses cache A. All retrieved chunks (1024 tokens each) use cache B. The crossover is naturally respected by the chunk size choice (1024 > 816).

---

## 4. The restoration error (the honest constraint)

The toy test revealed: snapshotting a chunk's state from an *independent* prefill (no prefix) and restoring it into a model that has the system prompt's state produces a ~4.1e-3 logit diff vs. the lossless re-prefill-in-context.

**Why:** the chunk's recurrent state S_L captures what the model "remembers" about the chunk. If the chunk was prefilled in isolation, S_L only reflects the chunk's own tokens. If prefilled after the system prompt, S_L reflects system+chunk. Restoring the isolated S_L into a model that has the system prompt's state creates a mismatch — the model's linear-attn state is "missing" the system prompt's contribution to the chunk's processing.

**The magnitude:** ~4.1e-3 in logits (max abs diff). This translates to a small probability shift in the softmax. For most queries, the argmax is unchanged (the answer is still correct). For borderline queries (where two tokens have close logits), the answer may flip.

**Three responses (the PoC measures all three):**

1. **Accept it:** the 4.1e-3 diff is within the model's natural noise (the W4 quantization already introduces ~3.5% PPL degradation). Measure the end-to-end OfficeQA accuracy with the cache vs. without — if the accuracy delta is < 2%, the cache is worth it.

2. **Re-prefill the top-3 on a miss:** when the cache produces a high restoration error (measured per query), fall back to re-prefilling the top-3 chunks in document order (lossless). This costs ~150 ms (3 × 50 ms) but only on the queries that need it. The cache handles the rest.

3. **Cache-aware fine-tune:** train the LUT model with the independent-snapshot + restore in the forward pass, so it learns to produce states that survive the prefix mismatch. (The toy found this *worsens* the error by 74-315% — so this is conditional, measured on the real model.)

---

## 5. The cache keying (the prefix question)

The toy revealed that the state must be snapshotted in the accumulated context to be lossless. But IVFADC requires one vector per chunk (independent of prefix). The resolution:

**Two cache entries per chunk:**
- **The independent entry** (keyed by `hash(chunk_token_ids)`): snapshotted from the chunk's independent prefill. Used by IVFADC (the pooled vector) and by the cache on a "fast path" hit (with the ~4.1e-3 restoration error).
- **The prefix-aware entry** (keyed by `hash(prefix_token_ids + chunk_token_ids)`): snapshotted from the chunk's prefill on top of a specific prefix. Used on a "lossless path" hit (zero restoration error, but requires the exact prefix to be cached).

**In practice:**
- The independent entry is always cached (one per chunk, 25.5 MiB + 32 MiB = 57.5 MiB per chunk).
- The prefix-aware entry is cached only for hot (prefix, chunk) pairs — e.g., (system_prompt, chunk_1) for the most-retrieved chunks. These are LRU-evicted.
- On a query: try the prefix-aware entry first (lossless); if miss, try the independent entry (fast, ~4.1e-3 error); if miss, re-prefill (lossless, slow).

**The cache sizing (corrected for two entry types):**
- 50k independent entries: 50k × 57.5 MiB = **2.81 TiB** (the dominant cost; needs host DRAM or LRU)
- Prefix-aware entries: only for hot pairs, ~156 on A10G (LRU) → ~9 GiB

---

## 6. The exact process, as a sequence

### 6.1 One-time setup (the "embed" step)

```python
# poc/setup/embed_corpus.py
def embed_corpus(model, chunks, cache_a, cache_b, ivfadc_index, disk_store):
    for chunk in chunks:  # 50,000 chunks
        token_ids = model.tokenizer(chunk.text, return_tensors="pt").input_ids.to(device)
        # 1. independent prefill
        with torch.no_grad():
            logits, lin_state, conv_state, kv, _ = model.prefill(token_ids, return_states=True)
            h = logits  # the final hidden state (before lm_head, actually — use the layer-before-last)
        # 2. extract the IVFADC vector
        v = h.mean(dim=1).squeeze()  # (4096,) fp16 — the pooled hidden state
        # 3. install in IVFADC (on disk)
        ivfadc_index.add(v.cpu().numpy(), chunk_id=chunk.id)
        # 4. install in the caches (independent entry)
        chunk_hash = hash_tokens(token_ids)
        cache_a.install(chunk_hash, kv)           # 32 MiB
        cache_b.install(chunk_hash, lin_state, conv_state)  # 25.5 MiB
        # 5. store the chunk payload on disk
        disk_store.write(chunk.id, chunk.text, token_ids, chunk.metadata)
    # 6. build the IVFADC structure (coarse quantizer + PQ training)
    ivfadc_index.build(nlist=224, m=64)
```

**Time: ~46 minutes on A10G. Disk: ~880 MiB (index) + ~490 MiB (payloads). Cache: 2.81 TiB (all chunks) or LRU subset.**

### 6.2 Per-query flow (retrieval + augmentation)

```python
# poc/serve/query.py
def handle_query(model, query_text, cache_a, cache_b, ivfadc_index, disk_store,
                system_prompt_tokens, top_k=3):
    # 1. tokenize + embed the query
    query_tokens = model.tokenizer(query_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits, _, _, _, _ = model.prefill(query_tokens, return_states=True)
    v_q = logits.mean(dim=1).squeeze().cpu().numpy()  # the query vector

    # 2. IVFADC preselect on disk → top-100
    candidate_ids = ivfadc_index.search(v_q, k=100)  # ~3 ms

    # 3. cos sim rerank → top-3
    candidate_vectors = disk_store.load_vectors(candidate_ids)  # ~800 KiB from disk
    scores = candidate_vectors @ v_q  # normalized → cos sim
    top3_ids = candidate_ids[np.argsort(scores)[-top_k:][::-1]]

    # 4. load the top-3 chunks' token IDs (for document ordering and re-prefill fallback)
    top3_chunks = [disk_store.load_chunk(cid) for cid in top3_ids]
    top3_chunks.sort(key=lambda c: (c.doc_id, c.page))  # document order

    # 5. restore the system prompt's states (always hit)
    sys_hash = hash_tokens(system_prompt_tokens)
    sys_kv = cache_a.lookup(sys_hash)
    sys_state = cache_b.lookup(sys_hash)
    model.restore_kv(sys_kv)
    model.restore_state(sys_state)

    # 6. restore the top-3 chunks' states (in document order)
    for chunk in top3_chunks:
        chunk_hash = hash_tokens(chunk.token_ids)
        # try prefix-aware entry first (lossless)
        prefix_hash = hash_tokens(system_prompt_tokens + chunk.token_ids)  # simplified
        prefix_entry = cache_b.lookup(prefix_hash)
        if prefix_entry is not None:
            # lossless restore
            model.append_kv(cache_a.lookup(prefix_hash))
            model.update_state(prefix_entry)
        else:
            # try independent entry (fast, ~4.1e-3 error)
            indep_kv = cache_a.lookup(chunk_hash)
            indep_state = cache_b.lookup(chunk_hash)
            if indep_kv is not None and indep_state is not None:
                model.append_kv(indep_kv)
                model.update_state(indep_state)
            else:
                # cache miss — re-prefill the chunk (lossless, slow)
                with torch.no_grad():
                    _, lin, conv, kv, _ = model.prefill(chunk.token_ids,
                        linear_state=model.linear_state,
                        conv_state=model.conv_state,
                        kv_cache=model.kv_cache, return_states=True)
                # install the prefix-aware entry for future queries
                cache_a.install(prefix_hash, kv)
                cache_b.install(prefix_hash, lin, conv)

    # 7. prefill the query on top of the restored context
    with torch.no_grad():
        logits = model.prefill(query_tokens,
                              linear_state=model.linear_state,
                              conv_state=model.conv_state,
                              kv_cache=model.kv_cache)

    # 8. decode the answer
    answer = model.decode(max_new_tokens=200)
    return answer
```

---

## 7. The latency budget (per query, A10G)

| Step | Cache hit (all 3 chunks) | Cache miss (re-prefill 3 chunks) |
|---|---|---|
| 1. Embed query | 5 ms | 5 ms |
| 2. IVFADC preselect | 3 ms | 3 ms |
| 3. Cos sim rerank | 1 ms | 1 ms |
| 4. Restore system prompt | 0.65 ms | 0.65 ms |
| 5a. Restore 3 chunks (cache hit) | 4.4 ms | — |
| 5b. Re-prefill 3 chunks (cache miss) | — | 150 ms |
| 6. Prefill query (32 tokens) | 2 ms | 2 ms |
| 7. Decode answer (~100 tokens) | 4,000 ms | 4,000 ms |
| **Total** | **~4,016 ms** | **~4,162 ms** |

**The cache saves ~146 ms per query on a hit (3.6% of total).** The decode dominates (99% of latency); the prefill savings are real but proportionally small at this context length. At longer contexts (8k+ tokens retrieved), the cache savings grow proportionally.

---

## 8. The benchmark (what to measure)

| Metric | How | Target |
|---|---|---|
| Retrieval recall@3 | IVFADC + rerank vs. gold documents | ≥ 80% |
| Restoration error | ‖logits_cache - logits_reprefill‖ / ‖logits_reprefill‖ | ≤ 1e-2 (accept); > 1e-2 (re-prefill fallback) |
| Cache hit rate | % of top-3 chunks found in the hot LRU set | ≥ 60% (power-law retrieval) |
| Accuracy (OfficeQA gold) | Exact-match + numeric-tolerance | Cache ≥ no-cache - 2% |
| TTFT | Time to first decoded token | ≤ 20 ms |
| Throughput | queries/s (decode-bound) | ~0.25 (at 4 s/query) |
| VRAM peak | `torch.cuda.max_memory_allocated` | ≤ 22 GiB |
| Disk | IVFADC index + payloads | ~880 MiB (50k chunks) |

---

## 9. What's on disk, what's in HBM, what's in host DRAM

| Tier | What | Size (50k chunks) | Notes |
|---|---|---|---|
| **Disk** | IVFADC index (PQ codes) | 3 MiB | The preselect structure |
| **Disk** | Exact vectors (for rerank) | 390 MiB | mmap'd, read on demand |
| **Disk** | Chunk payloads (text + token IDs + metadata) | 490 MiB | SQLite or mmap |
| **Disk** | Raw PDFs (bronze, for re-parsing) | 8.7 GiB | Kept for parser upgrades |
| **Disk total** | | **~9.6 GiB** | |
| **HBM (A10G)** | Model weights (W4+r32) | 5.85 GiB | Always resident |
| **HBM** | Cache A (KV, hot ~156 chunks) | 5.0 GiB | LRU |
| **HBM** | Cache B (state, hot ~156 chunks) | 4.0 GiB | LRU |
| **HBM** | Working state (current query's KV + state) | 0.3 GiB | Per query |
| **HBM** | Framework/safety | 5.0 GiB | |
| **HBM total** | | **~20 GiB** (fits in 24 GiB) | |
| **Host DRAM (optional)** | Cold cache entries (overflow) | up to 2.81 TiB | The K3 external pool; ~$280 |

**The A10G holds ~156 hot chunks in HBM.** For the PoC (50 questions, top-3 retrieval), the hot set is small enough to fit. For the full 50k subsample with many queries, host DRAM overflow is the production path.

---

## 10. Summary — the exact answers

### Snapshotting (one-time, per chunk)
1. Tokenize the chunk with the LUT model's tokenizer.
2. Prefill the chunk **independently** (no prefix) through the LUT model.
3. Extract: the pooled hidden state (→ IVFADC vector), the recurrent state + conv state (→ cache B), the KV pairs (→ cache A).
4. PQ-encode the vector → IVFADC index on disk (64 bytes per chunk).
5. Store the chunk's text + token IDs on disk (for re-prefill fallback).
6. Store the states in the two global caches (HBM hot, host DRAM cold).

### Retrieval (per query)
1. Tokenize + prefill the query → query vector (pooled hidden state).
2. IVFADC preselect on disk → top-100 candidates (~3 ms).
3. Cos sim rerank on the 100 exact vectors → top-3 chunks (~1 ms).
4. Return top-3 chunk IDs.

### Augmentation (per query)
1. Restore the system prompt's cached states (cache A + B) into the model.
2. For each of the top-3 chunks (in document order):
   - Try the prefix-aware cache entry (lossless) → if hit, restore.
   - Else try the independent cache entry (fast, ~4.1e-3 error) → if hit, restore.
   - Else re-prefill the chunk in context (lossless, slow) → install the prefix-aware entry.
3. Prefill the query (32 tokens) on top of the restored context.
4. Decode the answer.

### Benchmark
- Retrieval recall@3 ≥ 80% (the IVFADC + rerank quality gate).
- Restoration error ≤ 1e-2 (the cache quality gate; re-prefill on exceed).
- Cache hit rate ≥ 60% (the LRU + power-law assumption).
- OfficeQA accuracy: cache ≥ no-cache - 2% (the end-to-end gate).
