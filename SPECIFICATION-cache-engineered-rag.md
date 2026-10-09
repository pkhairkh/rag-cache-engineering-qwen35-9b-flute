# SPECIFICATION — Cache-Engineered RAG (Cache-as-Vector)

> **Status:** specification of record
> **Validated by:** `scripts/poc_toy/toy_cache_as_vector.py` (100% retrieval, 90% correct > wrong, logit diff +1.45)
> **Scope:** the FLUTE idxN W4+r32 Qwen3.5-9B LUT model. The model's own linear-attention cache (S + M1 + M2) IS the retrieval vector. NO separate embedder. NO hidden states. NO pooled logits. NO chunk text on disk. NO re-prefill.

---

## 1. The principle

**The cache IS the retrieval vector.** The linear-attention recurrent state (S) and the two Kimi-style memory matrices (M1, M2) — flattened into a single vector — are the representation used for retrieval. The query's cache and the chunk's cache are in the SAME space (both are linear-attn state matrices produced by the same model). Cosine similarity between them measures content overlap, which IS relevance.

**Why this works:** the delta-rule state accumulates token patterns. A query about topic X and a chunk about topic X produce similar state matrices, because the state captures what was processed. Comparing caches directly is the right relevance function — no separate embedder needed, no hidden-state projection needed.

---

## 2. The model

### 2.1 The base model

**Qwen3.5-9B** (the FLUTE idxN W4+r32 palettization), loaded via `scripts/eval_common.py::load_quant_model`. The model is a 3:1 hybrid:

| Component | Count | Role |
|---|---|---|
| Linear-attention layers (`Qwen3_5GatedDeltaNet`) | 24 | Produce the recurrent state S + the Kimi caches M1, M2 |
| Full-attention layers (standard SDPA / `scripts/attn_sm86.py`) | 8 | Fresh per query (NO KV cache snapshot) |
| `lm_head` | 1 | Next-token prediction (for pretraining + answering) |

### 2.2 The cache components (per linear-attention layer)

Each of the 24 linear-attention layers maintains THREE cache objects during the forward pass:

| Cache | Shape (per layer) | Description |
|---|---|---|
| **S** (recurrent state) | `(batch, num_v_heads=32, head_k_dim=128, head_k_dim=128)` | The delta-rule state matrix. Accumulates `decay * S_prev + beta * v ⊗ k` per token. Fixed-size per head — does NOT grow with sequence length. |
| **M1** (Kimi key-memory) | `(batch, num_v_heads=32, mem_size, head_k_dim=128)` | The external key-memory. Accumulates the chunk's keys via a gated, selective write. Extends the model's geometrical expressivity beyond S. |
| **M2** (Kimi value-memory) | `(batch, num_v_heads=32, mem_size, head_v_dim=128)` | The external value-memory. Paired with M1 — the model reads via `softmax(q @ M1^T) @ M2`. |
| **conv_state** | `(batch, conv_dim=8192, conv_kernel=4)` | The conv1d sliding-window state. Carries the last 4 tokens' worth of conv1d activations into the next forward. |

### 2.3 The cache vector (the retrieval representation)

At ingestion and at query time, the cache is **flattened** into a single 1D vector:

```
cache_vector = concat([
    flatten(S_layer_0), flatten(M1_layer_0), flatten(M2_layer_0),
    flatten(S_layer_1), flatten(M1_layer_1), flatten(M2_layer_1),
    ...
    flatten(S_layer_23), flatten(M1_layer_23), flatten(M2_layer_23),
])
```

**Size calculation (Qwen3.5-9B):**
- S per layer: 32 × 128 × 128 = 524,288 fp16 = 1,048,576 bytes = 1.0 MiB
- M1 per layer: 32 × mem_size × 128 fp16 (at mem_size=128: same as S = 1.0 MiB)
- M2 per layer: 32 × mem_size × 128 fp16 (at mem_size=128: same = 1.0 MiB)
- Per layer total: 3.0 MiB
- 24 layers: **72 MiB per cache vector** (at mem_size=128)
- Flattened dimension: 24 × (524,288 + 32×128×128 + 32×128×128) = 24 × 1,572,864 = **37,748,736 dims** (fp16)

The cache vector is large (72 MiB) but fixed-size — it does NOT grow with the chunk length or the corpus size.

### 2.4 The conv-reset discipline (composability)

**At ingestion:** each chunk is prefilled with conv_state reset to zero at the chunk boundary. This makes each chunk's cache deltas (delta_S, delta_M1, delta_M2) **path-independent** — the chunk's contribution to the cache does not depend on which chunks came before it.

**Consequence:** the deltas are **composable by summation**. Installing any subset of chunks' deltas (in any order) produces the correct accumulated cache for that subset. The toy validated this: diff = 0.0 (lossless) across all subsets and orderings.

---

## 3. Ingestion (one-time, per chunk)

### 3.1 The ingestion pipeline

```
For each chunk c in the corpus (50,000 chunks):
  1. Tokenize c with the LUT model's tokenizer → token_ids (1024 tokens)
  2. Prefill token_ids through the LUT model (conv-reset at boundary):
     - layer_states = model.initial_states()  # zero init
     - logits, new_states = model(token_ids, layer_states=layer_states, return_states=True)
  3. Compute the cache deltas (per layer):
     - For each linear layer i (0..23):
       delta_S[i] = new_states[i].S - layer_states[i].S
       delta_M1[i] = new_states[i].M1 - layer_states[i].M1
       delta_M2[i] = new_states[i].M2 - layer_states[i].M2
       conv_state[i] = new_states[i].conv_state  # the final conv state
  4. Compute the cache vector (the retrieval representation):
     - cache_vector = flatten_cache(new_states)  # flatten S + M1 + M2 from all 24 layers
  5. Save to disk:
     - snapshots/chunk_XXXXX.npz:
         delta_S_0, delta_M1_0, delta_M2_0, conv_state_0,  # layer 0
         delta_S_1, delta_M1_1, delta_M2_1, conv_state_1,  # layer 1
         ...
         cache_vector  # the flattened cache (for IVFADC)
  6. Append cache_vector to the IVFADC vectors array
```

### 3.2 What's saved per chunk (the snapshot format)

| Component | Per layer | × 24 layers | Total (fp16) |
|---|---|---|---|
| delta_S | 524,288 × 2 B = 1.0 MiB | 24 MiB | |
| delta_M1 (mem_size=128) | 524,288 × 2 B = 1.0 MiB | 24 MiB | |
| delta_M2 (mem_size=128) | 524,288 × 2 B = 1.0 MiB | 24 MiB | |
| conv_state | 32,768 × 2 B = 64 KiB | 1.5 MiB | |
| cache_vector | — | — | 72 MiB |
| **Total per chunk** | | | **~145.5 MiB** |

**For 50,000 chunks:** ~7.0 TiB on disk (the snapshots). The IVFADC index (PQ-coded cache vectors): ~3.5 TiB (the cache vectors are 72 MiB each; PQ reduces to ~64 bytes each, but the exact vectors for rerank are 72 MiB each).

**Sizing note:** the cache vector is large (72 MiB). For IVFADC, the PQ codes are small (~64 bytes per chunk), but the exact vectors for rerank are 72 MiB each. For 50k chunks, the exact vectors total 3.5 TiB — this must be on disk, not in RAM. The rerank step loads only the top-100 candidates' exact vectors (~7 GiB) from disk per query.

### 3.3 The IVFADC index

Built on the **cache vectors** (the flattened S + M1 + M2), NOT on hidden states or pooled logits.

```python
# poc/build_index.py
import faiss

# the cache vectors: (50000, 37748736) fp16 → fp32 for FAISS
cache_vectors = load_all_cache_vectors()  # from the snapshots
cache_vectors_fp32 = cache_vectors.astype(np.float32)

# build IVFADC
nlist = 224  # ~sqrt(50000)
m = 64       # PQ sub-quantizers (each 16384-dim → 256 centroids)
quantizer = faiss.IndexFlatIP(cache_vectors_fp32.shape[1])  # inner product (normalized = cos)
index = faiss.IndexIVFPQ(quantizer, cache_vectors_fp32.shape[1], nlist, m, nbits=8)
index.train(cache_vectors_fp32)
index.add(cache_vectors_fp32)
faiss.write_index(index, "disk/ivfadc_cache.index")
```

### 3.4 What's NOT saved

- ❌ No chunk text
- ❌ No token IDs
- ❌ No re-prefill fallback
- ❌ No hidden states
- ❌ No pooled logits

**The disk holds ONLY the cache snapshots (deltas + cache vector) + the IVFADC index.**

---

## 4. The query flow (per user query)

### 4.1 The 8-step query pipeline

```
User question: "What was the revenue in Table 3?"
    │
    ▼
[Step 1: Tokenize the query]
    query_token_ids = model.tokenizer(question)  →  (1, 32) int32
    │
    ▼
[Step 2: Prefill the query through the LUT model]
    layer_states = model.initial_states()  # zero init
    logits, query_states = model(query_token_ids, layer_states=layer_states, return_states=True)
    → produces the query's cache (S + M1 + M2 per layer)
    → ~5 ms on A10G (32 tokens)
    │
    ▼
[Step 3: Snapshot the query's cache → the retrieval vector]
    query_cache_vector = flatten_cache(query_states)  # flatten S + M1 + M2 from 24 layers
    → (37,748,736,) fp16 → fp32 for FAISS
    → <1 ms (just a flatten + cast)
    │
    ▼
[Step 4: IVFADC preselect on the cache vectors]
    index.nprobe = 8
    distances, candidate_ids = index.search(query_cache_vector, k=100)
    → top-100 candidate chunk indices
    → ~10 ms (the index is large; IVFADC probes 8 clusters)
    │
    ▼
[Step 5: Cosine similarity rerank]
    Load the 100 candidates' exact cache vectors from disk (mmap)
    candidate_vectors = exact_vectors[candidate_ids]  # (100, 37748736) fp16
    cand_norm = candidate_vectors / norm(candidate_vectors)
    q_norm = query_cache_vector / norm(query_cache_vector)
    scores = cand_norm @ q_norm
    top_k_ids = argsort(scores)[-3:]  # top-3
    → ~50 ms (100 × 72 MiB dot products — the dominant cost)
    │
    ▼
[Step 6: Load the top-3 chunks' cache snapshots from disk]
    For each chunk_id in top_k_ids:
        snapshot = load_npz(f"snapshots/chunk_{chunk_id:05d}.npz")
        → delta_S, delta_M1, delta_M2, conv_state per layer
    → ~5 ms (3 × 145 MiB from disk, or from the in-memory LRU pool)
    │
    ▼
[Step 7: Install the caches into the running model]
    For each linear layer i (0..23):
        restored_S[i] = system_S[i] + sum(delta_S[i] for each retrieved chunk)
        restored_M1[i] = system_M1[i] + sum(delta_M1[i] for each retrieved chunk)
        restored_M2[i] = system_M2[i] + sum(delta_M2[i] for each retrieved chunk)
        restored_conv[i] = last_retrieved_chunk.conv_state[i]
    → ~10 ms (24 × 3 delta sums + 24 state restores)
    │
    ▼
[Step 8: Answer from the installed caches]
    logits = model(query_token_ids, layer_states=restored_states)
    answer = model.decode(max_new_tokens=200)
    → ~2 ms prefill (32 tokens, caches installed) + ~8 s decode (200 tokens)
    → NO re-prefill of chunk text
```

### 4.2 The timing budget (per query, A10G)

| Step | Time | Notes |
|---|---|---|
| 1. Tokenize | <1 ms | |
| 2. Prefill query | ~5 ms | 32 tokens through the LUT model |
| 3. Snapshot query cache | <1 ms | Flatten S + M1 + M2 |
| 4. IVFADC preselect | ~10 ms | Probe 8 clusters on the 37M-dim vectors |
| 5. Cos sim rerank | ~50 ms | 100 × 72 MiB dot products (the dominant cost) |
| 6. Load snapshots | ~5 ms | 3 chunks from disk/LRU |
| 7. Install caches | ~10 ms | Sum deltas, restore into model |
| 8. Answer + decode | ~8 s | 32-token prefill + 200-token decode |
| **Total** | **~8.08 s** | Dominated by decode |

**The rerank step (50 ms) is the bottleneck** — it computes 100 dot products on 72 MiB vectors. This can be optimized with batched BLAS or by reducing the cache vector dimension (PCA on the cache vectors, or using only a subset of layers).

### 4.3 The system prompt cache (precomputed once)

The system prompt (e.g., "You are a financial analyst...") is prefilled ONCE at startup, and its cache (S + M1 + M2 + conv_state) is kept in memory. Every query starts from the system prompt's cache as the base state, then adds the retrieved chunks' deltas on top.

```python
# at startup
system_token_ids = model.tokenizer(SYSTEM_PROMPT)
_, system_states = model(system_token_ids, layer_states=model.initial_states(), return_states=True)
# system_states is kept in memory for the lifetime of the server
```

---

## 5. Pretraining (one-time, before ingestion)

### 5.1 Why pretrain

The toy showed that a randomly-initialized model's cache carries NO discriminative info (read attention uniform, 40% correct-vs-wrong). After 1500 pretrain steps on next-token prediction, the cache carries strong discriminative info (90% correct-vs-wrong, logit diff +1.45).

**Pretraining trains the linear-attn layers (including M1/M2) to produce states that are discriminative** — states where similar content produces similar caches, and different content produces different caches.

### 5.2 The pretrain target

**Next-token prediction on the corpus chunks.** The model learns to predict token `t+1` from tokens `[0..t]` within each chunk. This is standard language modeling — the model learns the corpus's patterns, and the cache (S + M1 + M2) becomes a meaningful representation of the content.

### 5.3 The pretrain mechanism (using the repo's existing files)

| File | Role |
|---|---|
| `scripts/eval_common.py::load_quant_model` | Loads the W4+r32 LUT model |
| `scripts/qlora.py::attach_qlora` + `QLoRAConfig` | Wraps with trainable LUTs (the W10 path) |
| `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` | The W10 autograd Function (trains LUTs with the cache mechanism in the forward pass) |
| `scripts/trainer.py` | The layerwise distillation trainer (two-layer residency, fits on A10G) |
| `scripts/muon_optimizer.py` | The Muon optimizer for the LoRA branch |
| `scripts/data.py::load_sft_dataset` | Loads the corpus as SFT examples (add OfficeQA chunks as a dataset) |
| `scripts/calibrate_real_text.py` | Calibration data (FineWeb-Edu, real text) |

### 5.4 The pretrain on the real model

**The real Qwen3.5-9B is ALREADY trained** (by Qwen). Its linear-attn layers already produce meaningful states. The pretrain on OfficeQA is a **fine-tune** — a short adaptation (~500 steps via the W10 LUT path) to make the model's cache work well on the OfficeQA corpus specifically.

**The pretrain does NOT train on Q&A pairs.** The pretrain is on the corpus CHUNKS (next-token prediction). The Q&A is the downstream evaluation.

### 5.5 The pretrain's output

The pretrain produces **fine-tuned LUTs** (the W10 two-stream path's output). These LUTs, when loaded into the model, make the linear-attn layers produce states that carry discriminative info. The pretrain is a ONE-TIME cost (~4-8 hours on A10G).

---

## 6. The augmentation mechanism (cache installation)

### 6.1 The installation

**Augmentation = installing the retrieved chunks' cache deltas into the running model.** The model's linear-attn state + Kimi M1/M2 caches ARE the augmented context. No text is concatenated, no tokens are re-prefilled.

```python
# poc/augment.py
def install_and_answer(model, query_token_ids, retrieved_chunk_idxs,
                      snapshot_pool, system_states, device):
    """Install the retrieved chunks' caches, answer from installed caches."""

    # 1. Sum the deltas from all retrieved chunks (composable, lossless)
    restored_states = []
    for layer_idx in range(model.num_linear):  # 24 layers
        r_S = system_states[layer_idx][0].clone()       # start from system prompt's S
        r_M1 = system_states[layer_idx][2].clone()      # start from system prompt's M1
        r_M2 = system_states[layer_idx][3].clone()      # start from system prompt's M2
        last_conv = system_states[layer_idx][1]

        for chunk_idx in retrieved_chunk_idxs:  # top-3 chunks
            snap = snapshot_pool.lookup(chunk_idx)
            r_S = r_S + snap.delta_S_list[layer_idx]       # sum the S delta
            r_M1 = r_M1 + snap.delta_M1_list[layer_idx]    # sum the M1 delta
            r_M2 = r_M2 + snap.delta_M2_list[layer_idx]    # sum the M2 delta
            last_conv = snap.conv_state_list[layer_idx]    # last chunk's conv

        restored_states.append((r_S, last_conv, r_M1, r_M2))

    # 2. The model answers from the installed caches (NO re-prefill)
    with torch.no_grad():
        logits, _ = model(query_token_ids, layer_states=restored_states, return_states=True)

    return logits
```

### 6.2 Why this works (the toy's validated findings)

1. **Composable (lossless):** the conv-reset at chunk boundaries makes each chunk's deltas path-independent. Summing any subset of chunks' deltas gives the correct state for that subset. The toy confirmed: diff = 0.0 (lossless) across all subsets and orderings.

2. **Discriminative (after pretrain):** the pretrain trains the linear-attn layers to produce states that carry discriminative info. The toy showed: after 1500 pretrain steps, the correct-topic cache makes the topic-marker logit 1.45 higher than the wrong-topic cache (90% correct-vs-wrong).

3. **No re-prefill:** the model answers from the installed caches in ~2 ms (only the query's 32 tokens are prefilled). The ~3,000 retrieved tokens' prefill is saved.

### 6.3 The augmentation's cost

| Component | Per-query cost |
|---|---|
| Load 3 snapshots from disk | ~5 ms (or 0 ms if in LRU) |
| Sum 24 × 3 deltas | ~10 ms |
| Restore into the model | ~1 ms (in-place tensor copies) |
| **Total augmentation** | **~16 ms** |

---

## 7. The retrieval mechanism (cache-as-vector)

### 7.1 The retrieval vector

**The cache vector = flattened (S + M1 + M2) from all 24 linear-attention layers.** This is the SAME representation for both the query and the chunks — both are the model's linear-attn state after processing tokens.

```
cache_vector = concat([
    S_layer_0.flatten(),      # (32, 128, 128) → 524,288
    M1_layer_0.flatten(),      # (32, mem_size, 128) → 524,288 (at mem_size=128)
    M2_layer_0.flatten(),      # (32, mem_size, 128) → 524,288
    S_layer_1.flatten(),       # ...
    ...
    S_layer_23.flatten(),
    M1_layer_23.flatten(),
    M2_layer_23.flatten(),
])
# total dimension: 24 × 3 × 524,288 = 37,748,736
```

### 7.2 The IVFADC index (on cache vectors)

The IVFADC index is built on the cache vectors using FAISS:

```python
# poc/build_index.py
quantizer = faiss.IndexFlatIP(37748736)  # inner product (normalized = cos)
index = faiss.IndexIVFPQ(quantizer, 37748736, nlist=224, m=64, nbits=8)
index.train(cache_vectors)
index.add(cache_vectors)
```

- **nlist=224** coarse clusters (~sqrt(50000))
- **m=64** PQ sub-quantizers (each ~589,824-dim → 256 centroids)
- **nprobe=8** clusters probed per query

### 7.3 The retrieval flow

```python
# poc/retrieve.py
def retrieve(model, query_token_ids, ivfadc_index, exact_vectors, top_k=3):
    """Retrieve by snapshotting the query's cache, then IVFADC + cos sim rerank."""

    # 1. Snapshot the query's cache
    init_states = model.initial_states()
    with torch.no_grad():
        logits, query_states = model(query_token_ids, layer_states=init_states, return_states=True)
    query_cache_vec = flatten_cache(query_states)  # (37748736,) fp32

    # 2. IVFADC preselect (on disk, mmap'd)
    candidates = ivfadc_index.search(query_cache_vec, k=100)  # top-100

    # 3. Cos sim rerank (load exact cache vectors, compute dot products)
    candidate_vecs = exact_vectors[candidates]  # (100, 37748736) fp16
    cand_norm = candidate_vecs / (norm(candidate_vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = query_cache_vec / (norm(query_cache_vec) + 1e-8)
    scores = cand_norm @ q_norm
    top_k_local = argsort(scores)[-top_k:][::-1]

    return candidates[top_k_local].tolist()  # top-k chunk indices
```

### 7.4 Why cache-to-cache retrieval works

The query's cache (S + M1 + M2) and the chunk's cache (S + M1 + M2) are the SAME type of object — both are the linear-attn state after processing tokens. The delta-rule state accumulates token patterns; a query about topic X and a chunk about topic X produce similar state matrices because the state captures what was processed.

**Cos-sim in the cache space measures content overlap, which IS relevance.** No separate embedder needed, no hidden-state projection needed, no retrieval head needed. The cache IS the retrieval vector.

### 7.5 The retrieval quality (toy-validated)

| Metric | Toy result |
|---|---|
| IVFADC retrieval hit rate | 100% (50/50 queries found the correct topic) |
| Correct cache > wrong cache | 90% (45/50 queries) |
| Mean logit diff (correct - wrong) | +1.45 |
| Random baseline | ~27% (3 out of 10 topics) |

---

## 8. The disk layout

```
disk/
├── ivfadc_cache.index              # FAISS IVFADC index on cache vectors
│                                   # (PQ codes + coarse centroids)
│                                   # ~3.5 TiB for 50k chunks (the PQ codes are small,
│                                   # but the exact vectors for rerank are 72 MiB each)
│
├── exact_cache_vectors.bin         # the exact fp16 cache vectors (for rerank)
│                                   # 50,000 × 72 MiB = 3.5 TiB
│                                   # mmap'd; only the top-100 are read per query
│
├── snapshots/                      # per-chunk cache snapshots
│   ├── chunk_00000.npz             # delta_S, delta_M1, delta_M2, conv_state per layer
│   ├── chunk_00001.npz             # + the cache_vector (redundant with exact_cache_vectors)
│   └── ...
│                                   # 50,000 × 145 MiB = 7.0 TiB
│
└── pretrained_luts/                # the fine-tuned LUTs (from pretraining)
    └── ...                         # ~5.85 GiB (the W4+r32 model weights)
```

**Total disk: ~10.5 TiB** for 50k chunks (dominated by the snapshots + exact vectors).

**Sizing note:** the cache vector is large (72 MiB). This is the trade-off of using the full cache (S + M1 + M2 from 24 layers) as the retrieval vector. Options to reduce:
- **Subset of layers:** use only the top 4 linear layers' caches → 12 MiB per vector (9× smaller)
- **PCA on the cache vectors:** reduce 37M dims to ~4096 dims → 8 KiB per vector (9000× smaller)
- **PQ-only (no exact rerank):** use the PQ-approximate scores directly → no exact vectors needed

These are optimizations for the PoC to measure; the specification uses the full cache for correctness.

---

## 9. The VRAM layout (A10G, 24 GiB)

| Component | Size | Notes |
|---|---|---|
| Model weights (W4+r32) | 5.85 GiB | Always resident |
| System prompt cache (S + M1 + M2 + conv) | ~72 MiB | Computed once at startup |
| Query cache (during query) | ~72 MiB | Per query |
| Installed retrieved caches (3 chunks' deltas) | ~435 MiB | Per query (3 × 145 MiB) |
| IVFADC index (in CPU RAM, not VRAM) | — | mmap'd |
| Exact cache vectors (on disk) | — | mmap'd; top-100 read per query |
| Framework + activations | ~5 GiB | |
| Safety margin | ~2 GiB | |
| **Total VRAM** | **~13.5 GiB** | Fits in 24 GiB |

---

## 10. The evaluation protocol

### 10.1 The metrics

| Metric | How measured | Target |
|---|---|---|
| Retrieval hit rate | IVFADC + rerank vs gold document | ≥ 95% |
| Correct > wrong (logit diff) | Correct cache vs wrong cache on the topic marker | ≥ 80% |
| Answer accuracy | OfficeQA gold exact-match + numeric-tolerance | ≥ 80% |
| TTFT | Time to first decoded token | ≤ 100 ms |
| Throughput | queries/s (decode-bound) | ~0.12 (at 8 s/query) |
| VRAM peak | `torch.cuda.max_memory_allocated` | ≤ 22 GiB |

### 10.2 The meaningful-query filter (BoxOffice gate)

Before any accuracy claim, the meaningful-query filter removes three classes:
1. Questions the cache-free reference run fails anyway (model capability)
2. Questions answerable without context (world knowledge)
3. Low-information yes/no questions

### 10.3 The baseline comparison

| Config | What it tests |
|---|---|
| No cache (query only) | The floor — the model answers from the query alone |
| Gold cache (correct-topic chunks installed) | The ceiling — the best the cache can do |
| IVFADC cache (retrieved chunks installed) | The real pipeline |

---

## 11. The repo files used at each step

### 11.1 Pretrain

| File | Role |
|---|---|
| `scripts/eval_common.py::load_quant_model` | Loads the W4+r32 LUT model |
| `scripts/qlora.py::attach_qlora` + `QLoRAConfig` | Wraps with trainable LUTs (W10) |
| `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` | The W10 autograd Function |
| `scripts/trainer.py` | The layerwise distillation trainer |
| `scripts/muon_optimizer.py` | The optimizer |
| `scripts/data.py::load_sft_dataset` | Loads the corpus as SFT |

### 11.2 Ingestion + snapshot

| File | Role |
|---|---|
| `scripts/modeling.py::Qwen3_5ForCausalLM` | The model's forward (prefill) |
| `scripts/modeling.py::Qwen3_5GatedDeltaNet` | The linear-attn layer (produces S) |
| `scripts/modeling.py::_fla_resolve` (W29) | The fla wiring for the recurrent state |
| `scripts/palettized_modules.py::PalettizedLinear` | The palettized forward (W4 kernel) |
| New: `poc/ingest.py` | The snapshot loop |
| New: `poc/flatten_cache.py` | The cache flattening |

### 11.3 IVFADC index

| File | Role |
|---|---|
| New: `poc/build_index.py` | Builds the FAISS IVFADC on cache vectors |

### 11.4 Query

| File | Role |
|---|---|
| `scripts/modeling.py::Qwen3_5ForCausalLM` | The model's forward (query prefill + answer) |
| New: `poc/retrieve.py` | IVFADC preselect + cos sim rerank on cache vectors |
| New: `poc/augment.py` | Cache installation (sum deltas, restore into model) |
| New: `poc/snapshot_pool.py` | The disk-backed LRU pool |

### 11.5 Evaluation

| File | Role |
|---|---|
| `scripts/eval_greedy_match.py::greedy_decode` | Decode the answer |
| `scripts/eval_common.py::atomic_json_dump` | Write the report |
| `scripts/measure_energy.py::EnergyMeasurement` | Measure latency/throughput/VRAM |

---

## 12. The toy validation

The toy (`scripts/poc_toy/toy_cache_as_vector.py`) validated the full pipeline on a CPU model:

| Finding | Toy result | What it means |
|---|---|---|
| Composability (conv-reset) | diff = 0.0 (lossless) | Deltas sum correctly — any subset, any order |
| IVFADC retrieval (cache vectors) | 100% (50/50) | The cache vector IS discriminative |
| Correct cache > wrong cache | 90% (45/50) | The cache carries discriminative info |
| Logit diff (correct - wrong) | +1.45 | Strong signal |
| Latency | 163 ms/query | Includes IVFADC + rerank + install + answer |

**The toy confirms the architecture works.** The A10G PoC tests it on the real 9B model.

---

## 13. Summary

| Question | Answer |
|---|---|
| Against what will we pretrain? | The OfficeQA corpus chunks (next-token prediction, W10 LUT path). Trains the linear-attn layers to produce discriminative caches. ~500 steps on A10G. |
| What's in the vectorDB? | IVFADC index on the **cache vectors** (flattened S + M1 + M2 from 24 layers, 37M-dim fp16) + per-chunk cache snapshots (deltas + conv_state). NO chunk text. NO hidden states. |
| How does the user query work? | Tokenize → prefill query → snapshot query's cache → IVFADC on cache vectors → cos sim rerank → load top-3 snapshots → install deltas → answer from installed caches → decode. ~8 s/query. |
| How does retrieval work? | Snapshot the query's cache (S + M1 + M2 flattened) → IVFADC preselect on the chunk cache vectors → cos sim rerank → top-3 chunk indices. ~60 ms. |
| How does augmentation work? | Sum the top-3 chunks' deltas (delta_S + delta_M1 + delta_M2 per layer) — composable, lossless. Install into the model. Answer from installed caches. ~16 ms. NO re-prefill. |
