# SIZING — How Much Space to Embed OfficeQA (or a 50k Subsample)

> **Companion to:** `PROPOSAL-POC-officeqa-a10g.md` (v3)
> **Question:** How much space do we need to embed whole OfficeQA or a 50k subsample within our plan?
> **Answer:** Two separate budgets. The **embedding index** (persistent, on disk) is small — 0.5–1 GiB. The **cache pool** (volatile, in HBM) is the binding constraint — and v3's crossover estimate was wrong by 8× (forgot to multiply by the 8 full-attention layers). Corrected crossover: **~816 tokens**, not 6,500. This changes the cache sizing substantially.

---

## 0. The two budgets, upfront

| Budget | What it stores | Full OfficeQA | 50k subsample | PoC (50 questions) |
|---|---|---|---|---|
| **Embedding index** (disk) | chunk text + embeddings + metadata + index graph | ~1.0 GiB | ~0.5 GiB | ~5 MiB |
| **Cache pool** (HBM, A10G) | KV prefix cache + recurrent-state cache | 1.3 TiB if caching ALL prefixes (infeasible); with LRU on 13 GiB pool: ~520 hot prefixes (high hit rate per the eviction paper) | same: ~520 hot prefixes on 13 GiB | ~7.6 GiB (fits) |

**The headline:** the embedding index is trivial (sub-GiB, fits on any disk). The cache pool is the real constraint — and the corrected crossover (~816 tokens) means the state cache (25.5 MiB fixed) is used for almost all RAG prefixes (which are typically 1k–8k tokens), making the cache pool fill up fast. The LRU eviction paper's finding (cache ≈ arrival rate × session duration, ~96 GiB/replica yields >99% hit rate caching <0.2% of unique blocks) is what makes this feasible — we cache the HOT prefixes, not all of them.

---

## 1. The embedding index (persistent, on disk)

This is the Vector Search index for retrieval — the corpus that the RAG system searches over. It's separate from the cache pool.

### 1.1 The corpus

From the OfficeQA Pro paper (arXiv:2603.08655):
- **89,000 pages** of U.S. Treasury Bulletins (1939–2025)
- **26 million+ numerical values** (dense financial tables)
- Mix of unstructured text and tabular data

**Token count estimate:** Treasury Bulletins are table-dense. A typical page of financial tables tokenizes to ~800–1,200 tokens (tables are token-dense — each cell boundary, header, and number is a token). Conservative average: **~1,000 tokens/page**.

- Full OfficeQA: 89,000 pages × 1,000 tokens/page = **~89M tokens**
- 50k subsample: depends on what "50k" means (see below)

### 1.2 What "50k subsample" means

Three interpretations, sized separately:

| Interpretation | Chunk count | Token count | Embedding index size |
|---|---|---|---|
| 50k **chunks** (at 1024 tokens/chunk) | 50,000 | 51M | ~0.5 GiB |
| 50k **pages** (at ~1000 tokens/page) | ~49,000 chunks (at 1024 tok/chunk) | 50M | ~0.5 GiB |
| 50k **documents** (OfficeQA has ~100 docs, so this doesn't apply) | N/A | N/A | N/A |

The most sensible interpretation: **50k chunks at 1024 tokens/chunk ≈ 51M tokens** — roughly half the full corpus.

### 1.3 Per-chunk storage breakdown

| Component | Bytes | Notes |
|---|---|---|
| Embedding vector (BAAI/bge-m3, 1024-dim, fp16) | 2,048 | The dense vector for retrieval |
| Chunk text (UTF-8, ~1024 tokens × ~4 bytes/token) | ~4,096 | The raw text, stored for re-reading |
| Token IDs (int32, 1024 tokens) | 4,096 | For feeding to the model |
| Metadata (doc_id, page, chunk_id, section, table_flag, provenance SHA) | ~256 | For filtering and provenance |
| **Per-chunk subtotal** | **~10,496** | **~10.3 KiB** |
| Index graph overhead (HNSW, ~1.5× embedding) | ~3,072 | The ANN graph structure |
| **Per-chunk total** | **~13,568** | **~13.3 KiB** |

### 1.4 Total embedding index size

| Corpus | Chunks | Per-chunk | Total | Rounded |
|---|---|---|---|---|
| Full OfficeQA (89k pages, 1024 tok/chunk) | ~87,000 | 13.3 KiB | 1,159 MiB | **~1.1 GiB** |
| Full OfficeQA (89k pages, 512 tok/chunk) | ~174,000 | 13.3 KiB | 2,318 MiB | ~2.3 GiB |
| 50k subsample (1024 tok/chunk) | 50,000 | 13.3 KiB | 667 MiB | **~0.65 GiB** |
| 50k subsample (512 tok/chunk) | 50,000 | 13.3 KiB | 667 MiB | ~0.65 GiB |
| PoC (50 questions, top-5 retrieval) | ~250 | 13.3 KiB | 3.3 MiB | **~3 MiB** |

**The embedding index is not the binding constraint.** Even the full OfficeQA corpus at 512 tokens/chunk is ~2.3 GiB — trivial on any modern disk. The 50k subsample is ~0.65 GiB. The PoC's 250 chunks are ~3 MiB.

---

## 2. The cache pool (volatile, in HBM)

This is the binding constraint. The cache pool stores the prefilled states/KV for the two global caches, in GPU HBM (or host memory if spilled).

### 2.1 Correction: the crossover is ~816 tokens, not 6,500

v3's formula was: `L* = state_bytes / (num_kv_heads × head_dim × 2 × 2)`. This is **per layer**. The correct formula multiplies by the 8 full-attention layers:

```
KV per token (all 8 full-attn layers):
  = num_kv_heads × head_dim × 2 (K+V) × 2 (fp16) × num_full_attn_layers
  = 4 × 256 × 2 × 2 × 8
  = 32,768 bytes
  = 32 KiB per token

State (24 linear-attn layers, fixed):
  = 25.5 MiB = 26,112 KiB

Crossover L*:
  = 26,112 KiB / 32 KiB per token
  = 816 tokens
```

**The corrected crossover is ~816 tokens.** v3's 6,500 was wrong by 8× (the 8 full-attention layers were omitted).

**What this means:** almost every RAG prefix is ABOVE the crossover (system prompt + one retrieved chunk is typically 1,500+ tokens). The state cache (25.5 MiB fixed) is used for almost all prefixes, not the KV cache. The KV cache is only smaller for very short prefixes (< 816 tokens, e.g., a bare system prompt with no retrieved context).

### 2.2 Per-prefix cache sizes

| Prefix type | Token length | Below/above crossover | Cache used | Size per prefix |
|---|---|---|---|---|
| System prompt only | ~500 | below (816) | KV cache | 500 × 32 KiB = **15.6 MiB** |
| System + 1 chunk | ~1,500 | above | State cache | **25.5 MiB** (fixed) |
| System + 2 chunks | ~2,500 | above | State cache | **25.5 MiB** (fixed) |
| System + 5 chunks (top-5 retrieval) | ~5,500 | above | State cache | **25.5 MiB** (fixed) |
| Full 8k context | ~8,000 | above | State cache | **25.5 MiB** (fixed) |

The state cache's fixed size is its advantage: a 5,500-token prefix and an 8,000-token prefix both cost 25.5 MiB. The KV cache would cost 172 MiB and 250 MiB respectively.

### 2.3 The PoC (50 questions, top-5 retrieval)

**Unique prefixes to cache:**
1. System prompt (~500 tokens): 1 unique → KV cache, 15.6 MiB
2. System + 1 chunk (~1,500 tokens): up to 50 × 5 = 250 unique (with retrieval overlap, ~200 unique) → State cache, 200 × 25.5 MiB = **5.1 GiB**
3. System + 2 chunks (~2,500 tokens): ~100 unique (sessions that retrieve the same 2 chunks) → State cache, 100 × 25.5 MiB = **2.5 GiB**
4. Deeper prefixes (session-specific turns): not cached (LRU evicts — these are one-hit)

**Total cache for PoC: ~7.6 GiB** (15.6 MiB + 5.1 GiB + 2.5 GiB)

**Fits on A10G?** A10G has 24 GiB. W4+r32 weights ~5.85 GiB. Framework/activations ~5 GiB. Safety margin ~2 GiB. **Cache pool budget: ~13 GiB.** The PoC's 7.6 GiB fits with room to spare.

### 2.4 The 50k subsample (the binding case)

**Unique prefixes to cache (if we tried to cache ALL):**
- 1 system prompt prefix: 15.6 MiB (KV cache)
- 50,000 unique (system + 1 chunk) prefixes: 50,000 × 25.5 MiB = **1,275,000 MiB = 1,275 GiB = 1.275 TiB**
- This is **infeasible** on A10G (24 GiB) or any single GPU.

**With LRU eviction on A10G's 13 GiB cache pool:**
- 13 GiB / 25.5 MiB per prefix = **~520 hot prefixes** cached simultaneously
- Out of 50,000 unique chunks, we cache the hottest 520 (1.04%)
- The LRU eviction paper (arXiv:2609.28870) measured: "cache size ≈ arrival rate × session duration; ~96 GiB/replica already yields very high hit ratios while caching <0.2% of unique blocks"
- At 13 GiB (vs the paper's 96 GiB), we cache ~520 prefixes (0.13× the paper's capacity). The hit rate will be lower but still meaningful for the HOT prefixes (the system prompt + the most-retrieved chunks)

**Hit rate estimate for 50k subsample on A10G:**
- The system prompt: always cached (1 prefix, 100% hit rate for this prefix)
- The top-520 most-retrieved chunks: cached. If retrieval follows a power-law (typical in RAG — a few chunks are retrieved by many questions), the top-520 might cover ~40-60% of all retrievals.
- The long-tail chunks (49,480 of them): cache misses, full re-prefill.

**To cache ALL 50k prefixes (production, not A10G):**
- 50,000 × 25.5 MiB = **1,275 GiB ≈ 1.3 TiB** of cache pool
- This requires a multi-GPU or CPU-host-memory cache pool (like K3's external KV-cache pool, or MemServe's disaggregated pool)
- At ~$10/GiB for HBM: ~$13,000 in GPU memory (infeasible for a PoC)
- At ~$0.10/GiB for DRAM: ~$130 in host memory (feasible for production, with the latency cost of GPU↔host transfers)

### 2.5 Full OfficeQA (89k pages, ~87k chunks)

Same arithmetic, scaled up:
- 87,000 unique (system + chunk) prefixes × 25.5 MiB = **2,218 GiB ≈ 2.2 TiB** to cache ALL
- With LRU on 13 GiB: ~520 hot prefixes (0.6% of unique)
- For production (96 GiB/replica per the LRU paper): ~3,760 hot prefixes (4.3% of unique) — "very high hit ratios" per the paper

---

## 3. The A10G reality check

| Component | Bytes (W4+r32) | A10G fit? |
|---|---|---|
| Weights (248 palettized modules + lm_head + embed + LUTs + r32) | 5.85 GiB | yes |
| Framework + CUDA context + Triton cache | 1.5 GiB | yes |
| Activations + intermediates (at 8k context, batch 1) | 1.5 GiB | yes |
| Safety margin | 2.0 GiB | yes |
| **Cache pool budget (A10G)** | **~13 GiB** | — |
| PoC (50 questions) cache need | ~7.6 GiB | **yes, fits** |
| 50k subsample (cache ALL prefixes) | ~1,275 GiB | **no, infeasible** |
| 50k subsample (LRU, 520 hot prefixes) | ~13 GiB | **yes, fits exactly** |
| Full OfficeQA (cache ALL) | ~2,200 GiB | **no** |
| Full OfficeQA (LRU, 520 hot prefixes) | ~13 GiB | **yes, fits exactly** |

**The conclusion:** on a single A10G, the cache pool fits the PoC (50 questions) comfortably, and fits the 50k/full OfficeQA **with LRU eviction** — caching ~520 hot prefixes (1% of unique) out of 50,000. The hit rate depends on the retrieval distribution's power-law exponent; the LRU paper's measurement (96 GiB → <0.2% cached → very high hit rate) suggests 13 GiB → ~0.13× that capacity → moderate hit rate, dominated by the system prompt and the top chunks.

**For production (not A10G):** the K3 external KV-cache pool pattern (host memory or a separate GPU tier) is the path to caching ALL prefixes. At 1.3 TiB for the 50k subsample, host DRAM (~$130) is the economical choice; the cost is the GPU↔host transfer latency (~25.5 MiB at PCIe Gen4 ~64 GB/s = ~0.4 ms per restore, acceptable relative to the prefill it saves).

---

## 4. The corrected crossover (and what it changes)

| | v3 (wrong) | corrected |
|---|---|---|
| Crossover L* | ~6,500 tokens | **~816 tokens** |
| Formula | `state_bytes / (kv_heads × head_dim × 2 × 2)` (per layer) | `state_bytes / (kv_heads × head_dim × 2 × 2 × 8 layers)` |
| Effect on crossover policy | KV cache used for prefixes up to 6,500 tokens | KV cache used for prefixes up to **816 tokens only** |
| Effect on OfficeQA | Most prefixes (1k–8k tokens) would use KV cache | Most prefixes (1k–8k tokens) use **state cache** |
| Effect on cache sizing | KV cache (grows linearly) would dominate | State cache (fixed 25.5 MiB) dominates — simpler, more predictable |

**The correction makes the state cache MORE important, not less.** At the corrected crossover, almost every RAG prefix (system + retrieved chunk = 1,500+ tokens) uses the state cache. The KV cache is only for the bare system prompt (~500 tokens). The cache pool's sizing becomes simpler: it's `N_hot_prefixes × 25.5 MiB`, regardless of prefix length.

---

## 5. Summary: what you need

### For the PoC (50 OfficeQA questions, top-5 retrieval, 8k context, A10G)

| Item | Size | Storage | Fits A10G? |
|---|---|---|---|
| Embedding index (250 chunks) | ~3 MiB | disk | yes (trivial) |
| Cache pool (per-session + shared prefixes) | ~7.6 GiB | HBM | yes (fits in 13 GiB budget) |
| Model weights (W4+r32) | 5.85 GiB | HBM | yes |
| **Total HBM** | **~15 GiB** | — | **yes (of 24 GiB)** |

### For a 50k subsample (50,000 chunks, many questions, A10G)

| Item | Size | Storage | Fits A10G? |
|---|---|---|---|
| Embedding index (50k chunks) | ~0.65 GiB | disk | yes (trivial) |
| Cache pool (cache ALL 50k prefixes) | 1,275 GiB | HBM | **no** |
| Cache pool (LRU, ~520 hot prefixes) | ~13 GiB | HBM | **yes (fills the budget)** |
| Model weights | 5.85 GiB | HBM | yes |
| **Total HBM (with LRU)** | **~21 GiB** | — | **yes (tight, of 24 GiB)** |

### For full OfficeQA (89k pages, ~87k chunks, A10G)

| Item | Size | Storage | Fits A10G? |
|---|---|---|---|
| Embedding index (87k chunks) | ~1.1 GiB | disk | yes (trivial) |
| Cache pool (cache ALL 87k prefixes) | 2,200 GiB | HBM | **no** |
| Cache pool (LRU, ~520 hot prefixes) | ~13 GiB | HBM | **yes** |
| Model weights | 5.85 GiB | HBM | yes |
| **Total HBM (with LRU)** | **~21 GiB** | — | **yes (tight)** |

### For production (not A10G, external cache pool)

| Item | Size | Storage | Cost |
|---|---|---|---|
| Embedding index (50k chunks) | ~0.65 GiB | disk | trivial |
| Cache pool (cache ALL 50k prefixes) | 1,275 GiB | host DRAM (K3 external pool) | ~$130 |
| Cache pool (96 GiB/replica, LRU) | 96 GiB | HBM (per the eviction paper) | ~$960 per replica |
| Model weights | 5.85 GiB | HBM | per replica |

---

## 6. The honest answer to your question

**"How much space do we need to embed whole OfficeQA or a 50k subsample?"**

1. **The embedding index** (the retrieval corpus): **~0.65 GiB for 50k chunks, ~1.1 GiB for full OfficeQA.** Trivial — fits on any disk. This is not the constraint.

2. **The cache pool on A10G**: 
   - **PoC (50 questions): ~7.6 GiB.** Fits comfortably in the 13 GiB cache budget.
   - **50k subsample (cache ALL prefixes): ~1,275 GiB = 1.3 TiB.** Infeasible on A10G. With LRU eviction, ~520 hot prefixes fit in 13 GiB — but that's 1% of the unique prefixes, and the hit rate depends on the retrieval distribution's skew.
   - **Full OfficeQA (cache ALL): ~2.2 TiB.** Same story — LRU on 13 GiB caches ~520 hot prefixes.

3. **The crossover correction**: v3's ~6,500 tokens was wrong (forgot 8 full-attention layers). Corrected: **~816 tokens.** This means the state cache (25.5 MiB fixed) is used for almost all RAG prefixes, making the cache pool sizing simpler and more predictable: `N_hot_prefixes × 25.5 MiB`.

4. **The path to caching ALL prefixes** (if that's the goal): the K3 external cache pool pattern — host DRAM at ~$0.10/GiB. For 50k prefixes: ~1.3 TiB of DRAM (~$130). The cost is the GPU↔host transfer latency (~0.4 ms per 25.5 MiB restore at PCIe Gen4), which is acceptable relative to the prefill it saves (the Contiguity paper's 13-21× repair ratio is the floor; a full re-prefill of a 5k-token prefix is ~100-500 ms).

**The recommendation:** for the PoC on A10G, cache the hot prefixes with LRU (fits in 13 GiB). For production, move the cache pool to host DRAM (the K3 external pool) once the hit rate on A10G proves the concept. The embedding index is never the constraint.
