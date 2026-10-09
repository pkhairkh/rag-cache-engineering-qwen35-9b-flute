# EXACT WORKFLOW — Cache-Engineered RAG on the Qwen3.5-9B LUT Model

> **Grounded in:** the repo's actual files (`scripts/`, `docs/`) + the toy's validated findings (`scripts/poc_toy/toy_investigate.py`).
> **Scope:** the FLUTE idxN W4+r32 Qwen3.5-9B LUT model. NO separate embedder. NO FAISS text index. NO re-prefill. The model's own linear-attention state + the two Kimi-style M1/M2 caches ARE the retrieval + augmentation.
> **The validated mechanism (from the toy):** pre-train the model → snapshot the caches per chunk → IVFADC on the snapshot vectors → install the top-k snapshots' deltas → the model answers from the installed caches. NO text on disk. NO re-prefill.

---

## 0. The architecture, in one diagram

```
                    ┌─────────────────────────────────────────────┐
                    │ THE LUT MODEL (Qwen3.5-9B, FLUTE idxN W4)  │
                    │                                             │
                    │  24 linear-attention layers (the state      │
                    │  producer — each with S + M1 + M2)          │
                    │  8 full-attention layers (fresh per query)  │
                    │  fla wiring (W29) for the recurrent state    │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │ INGESTION (one-time, per chunk)              │
                    │                                             │
                    │  For each corpus chunk:                     │
                    │  1. Prefill the chunk through the model      │
                    │     (conv-reset at boundaries → composable) │
                    │  2. Snapshot per linear layer:              │
                    │     - delta_S (the recurrent state delta)    │
                    │     - delta_M1 (Kimi key-memory delta)      │
                    │     - delta_M2 (Kimi value-memory delta)    │
                    │     - conv_state (the conv1d final state)   │
                    │  3. Extract the pooled hidden vector         │
                    │     (for IVFADC retrieval)                   │
                    │  4. Save ALL of this to disk (caches only,   │
                    │     NO chunk text)                          │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │ DISK (persistent, all chunks)                │
                    │                                             │
                    │  Per chunk:                                 │
                    │  - delta_S, delta_M1, delta_M2, conv_state   │
                    │    for each of 24 linear layers             │
                    │  - the pooled hidden vector                  │
                    │                                             │
                    │  NO chunk text. NO token IDs. NO re-prefill. │
                    │                                             │
                    │  IVFADC index on the pooled hidden vectors   │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │ QUERY (per user query)                      │
                    │                                             │
                    │  1. Prefill the query through the model     │
                    │     → get the query's pooled hidden vector  │
                    │  2. IVFADC preselect → top-100 candidates    │
                    │  3. Cos sim rerank → top-k=3 chunk indices   │
                    │  4. Load those 3 chunks' CACHE SNAPSHOTS     │
                    │     from disk                               │
                    │  5. SUM the deltas (delta_S + delta_M1 +    │
                    │     delta_M2) — composable, lossless        │
                    │  6. INSTALL into the running model:         │
                    │     S = system_S + Σ delta_S, etc.           │
                    │  7. The model answers FROM THE INSTALLED    │
                    │     CACHES — NO re-prefill of chunk text    │
                    └─────────────────────────────────────────────┘
```

---

## 1. Against what will we pretrain?

### 1.1 The pretrain target: the corpus itself (next-token prediction)

**The pretrain objective:** next-token prediction on the corpus chunks. The model learns to predict token `t+1` from tokens `[0..t]` within each chunk.

**The pretrain data:** the OfficeQA corpus — the 89,000 pages of U.S. Treasury Bulletins, chunked into ~87,000 chunks at 1024 tokens each (or a 50k subset). This is the SAME corpus that will be snapshotted.

**The pretrain mechanism (using the repo's existing files):**
- `scripts/data.py::load_sft_dataset` — loads the corpus as SFT examples (already supports FineWeb-Edu; we add OfficeQA's chunks as a dataset)
- `scripts/trainer.py` — the layerwise distillation trainer (two-layer residency, fits on A10G)
- `scripts/qlora.py::attach_qlora` — wraps the palettized model with trainable LUT masters (the W10 path)
- `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` — the W10 autograd Function (trains LUTs with the cache mechanism in the forward pass)
- `scripts/muon_optimizer.py` — the Muon optimizer for the LoRA branch

**Why pretrain (the toy's finding):** the toy showed that a randomly-initialized model's cache carries NO discriminative info (read attention uniform, 40% correct-vs-wrong). After 1500 pretrain steps, the cache carries discriminative info (76% correct-vs-wrong, logit diff +1.11). **Pretraining trains the linear-attn layers (including M1/M2) to learn meaningful key/value representations.**

### 1.2 The pretrain does NOT train on Q&A pairs

The pretrain is on the corpus CHUNKS (next-token prediction), NOT on OfficeQA Q&A. The Q&A is the downstream evaluation. The pretrain makes the model's cache mechanism work; the Q&A measures if the cache-installed answer is correct.

### 1.3 The pretrain on the real model (Qwen3.5-9B LUT)

**The real Qwen3.5-9B is ALREADY trained** (by Qwen, on a massive corpus). Its linear-attn layers already produce meaningful states. The pretrain on OfficeQA is a FINE-TUNE — a short adaptation (the W10 LUT path, ~500 steps) to make the model's cache work well on the OfficeQA corpus specifically.

**The pretrain uses the existing repo infrastructure:**
```python
# the pretrain script (new, but built on existing files)
# poc/pretrain.py
from scripts.eval_common import load_quant_model  # loads the W4+r32 LUT model
from scripts.qlora import attach_qlora, QLoRAConfig  # W10 trainable LUTs
from scripts.trainer import Trainer  # the layerwise distillation trainer
from scripts.data import load_sft_dataset  # loads OfficeQA chunks as SFT

model, metadata = load_quant_model("/path/to/palettized", "Qwen/Qwen3.5-9B", "cuda:0",
                                    residual=True, forward="kernel", heads_dir="/path/to/heads")
config = QLoRAConfig(r=64, alpha=16, scope="all", base_model="Qwen/Qwen3.5-9B",
                     artifacts_dir="/path/to/palettized")
attach_qlora(model, metadata, **config.__dict__)
# load OfficeQA chunks as next-token-prediction SFT examples
dataset = load_officeqa_chunks_as_ntp(seq_len=1024, tokenizer=model.tokenizer)
trainer = Trainer(model, metadata, dataset, ...)
trainer.train(n_steps=500)  # the pretrain
trainer.export("/path/to/pretrained_luts")  # the fine-tuned LUTs
```

### 1.4 The pretrain's output

The pretrain produces **fine-tuned LUTs** (the W10 two-stream path's output). These LUTs, when loaded into the model, make the linear-attn layers produce states that carry discriminative info. The pretrain is a ONE-TIME cost (~4-8 hours on A10G).

---

## 2. How will the vectorDB work and what will be stored there?

### 2.1 What's stored (caches ONLY, NO text)

The "vectorDB" is an IVFADC index on the **pooled hidden vectors** (one per chunk), PLUS the per-chunk cache snapshots. Stored on disk:

```
disk/
├── ivfadc_index.npz          # the IVFADC index (PQ codes + coarse centroids)
│                              # built on the pooled hidden vectors
├── exact_vectors.bin         # the exact fp16 vectors (for cos sim rerank)
└── snapshots/
    ├── chunk_00000.npz        # per-chunk: deltas + conv_state + hidden
    ├── chunk_00001.npz
    └── ...                    # 50,000 files for the 50k subsample
```

**Per chunk (the `chunk_XXXXX.npz` file):**
| Component | Shape (per layer) | Layers | Size (fp16) |
|---|---|---|---|
| delta_S | (32, 128, 128) | 24 | 24 MiB |
| delta_M1 | (32, mem_size, 128) | 24 | 24 × mem_size/128 MiB |
| delta_M2 | (32, mem_size, 128) | 24 | 24 × mem_size/128 MiB |
| conv_state | (8192, 4) | 24 | 1.5 MiB |
| hidden (pooled) | (4096,) | 1 | 8 KiB |
| **Total per chunk** | | | **~50 MiB** (at mem_size=128) |

**For 50k chunks: ~2.4 TiB on disk** (the snapshots). The IVFADC index is ~113 MiB (PQ codes + exact vectors).

### 2.2 The IVFADC index

The IVFADC index is built on the **pooled hidden vectors** (one per chunk, 4096-dim fp16). This is the model's own representation of the chunk — NOT a separate embedder's vector.

**The IVFADC structure (using FAISS):**
```python
import faiss
# build the index
quantizer = faiss.IndexFlatIP(4096)  # inner product (vectors are normalized)
index = faiss.IndexIVFPQ(quantizer, 4096, nlist=224, m=64, nbits=8)
index.train(vectors)  # train the coarse quantizer + PQ
index.add(vectors)    # add the vectors
# search
index.nprobe = 8
distances, ids = index.search(query_vector, k=100)  # top-100 candidates
```

**The IVFADC is on disk**, mmap'd into CPU RAM at startup. The search is ~3 ms per query.

### 2.3 What's NOT stored

- ❌ No chunk text
- ❌ No token IDs
- ❌ No re-prefill fallback
- ❌ No "augmentation by concatenating text"

The disk holds ONLY the cache snapshots + the IVFADC index. The model answers from the installed caches.

---

## 3. How does the user query work?

### 3.1 The query flow (end to end)

```
User question ("What was the revenue in Table 3?")
    │
    ▼
[1. Tokenize the query]
    Use the LUT model's tokenizer (scripts/modeling.py's AutoTokenizer)
    → query_token_ids (32 tokens)
    │
    ▼
[2. Prefill the query through the LUT model]
    model.prefill(query_token_ids)
    → produces the query's pooled hidden vector (4096-dim fp16)
    → ~5 ms on A10G (only 32 tokens)
    │
    ▼
[3. IVFADC preselect on disk]
    index.search(query_vector, k=100)
    → top-100 candidate chunk indices
    → ~3 ms (the index is mmap'd in CPU RAM)
    │
    ▼
[4. Cos sim rerank]
    Load the 100 candidates' exact vectors from disk (mmap)
    Compute dot products (vectors are normalized)
    → top-k=3 chunk indices
    → ~1 ms
    │
    ▼
[5. Load the top-3 chunks' cache snapshots from disk]
    For each of the 3 chunk indices:
      Load the .npz file (delta_S, delta_M1, delta_M2, conv_state per layer)
    → ~1 ms (3 × ~50 MiB from disk, or from the in-memory LRU pool)
    │
    ▼
[6. Install the caches into the running model]
    For each of the 24 linear-attn layers:
      S = system_S + Σ(delta_S)        # sum the deltas (composable)
      M1 = system_M1 + Σ(delta_M1)    # sum the Kimi key-memory deltas
      M2 = system_M2 + Σ(delta_M2)    # sum the Kimi value-memory deltas
      conv_state = last_chunk's conv_state
    → ~5 ms (summing 24 × 3 tensors, then restoring into the model)
    │
    ▼
[7. The model answers FROM THE INSTALLED CACHES]
    model.prefill(query_token_ids, layer_states=restored_states)
    → logits (the model's answer, computed from the installed caches)
    → NO re-prefill of chunk text
    → ~2 ms (only 32 query tokens)
    │
    ▼
[8. Decode the answer]
    model.decode(max_new_tokens=200)
    → the answer string
    → ~8 s (at ~25 tok/s on A10G)
```

### 3.2 The query timing (per query, A10G)

| Step | Time | What |
|---|---|---|
| 1. Tokenize | <1 ms | The model's tokenizer |
| 2. Prefill query | ~5 ms | 32 tokens through the LUT model |
| 3. IVFADC preselect | ~3 ms | On disk, mmap'd |
| 4. Cos sim rerank | ~1 ms | 100 candidates |
| 5. Load snapshots | ~1 ms | 3 chunks from disk/LRU |
| 6. Install caches | ~5 ms | Sum deltas, restore into model |
| 7. Answer from caches | ~2 ms | 32 tokens, caches installed |
| 8. Decode | ~8 s | 200 tokens at ~25 tok/s |
| **Total** | **~8.02 s** | Dominated by decode |

**The cache engineering saves the prefill of ~3,000 retrieved tokens** (3 chunks × 1024 tokens). At the model's prefill rate, that's ~50-100 ms saved per query. The decode dominates.

---

## 4. How does retrieval work?

### 4.1 The retrieval mechanism (IVFADC + cos sim rerank)

**The retrieval vector:** the query's pooled hidden state — the LUT model's own representation of the query, NOT a separate embedder's vector.

```python
# poc/retrieve.py
def retrieve(model, query_token_ids, ivfadc_index, exact_vectors, top_k=3):
    """Retrieve the top-k chunks for a query.
    The model's pooled hidden state IS the retrieval vector."""
    # 1. prefill the query through the LUT model
    with torch.no_grad():
        logits, _ = model(query_token_ids, layer_states=model.initial_states(), return_states=True)
    query_vector = logits.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)  # (4096,)

    # 2. IVFADC preselect (on disk, mmap'd)
    candidates = ivfadc_index.search(query_vector, k=100)  # top-100

    # 3. cos sim rerank (load exact vectors, compute dot products)
    candidate_vectors = exact_vectors[candidates]  # mmap, ~800 KiB
    cand_norm = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-8)
    q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
    scores = cand_norm @ q_norm
    top_k_local = np.argsort(scores)[-top_k:][::-1]

    return candidates[top_k_local].tolist()  # top-k chunk indices
```

**The retrieval quality gate (from the toy):** the toy showed that the pooled hidden vector is discriminative (87% retrieval hit rate with mean-pool, before any fine-tuning). On the real trained model, retrieval should be 95%+.

### 4.2 Why IVFADC (not HNSW, not brute-force)

- **IVFADC** is the right choice for 50k+ vectors: ~3 ms per query, ~92% recall, ~113 MiB on disk.
- **HNSW** is faster (~2 ms) but uses more memory (~20 MiB in RAM) — fine for 50k, but IVFADC scales better to 1M+.
- **Brute-force** is ~50 ms for 50k vectors — too slow.

The IVFADC is built with FAISS (`faiss.IndexIVFPQ`), the industry standard.

### 4.3 What the retrieval returns

The retrieval returns **chunk indices** (not text, not embeddings). The indices are used to load the cache snapshots from disk. The model never sees the chunk text.

---

## 5. How does augmentation work?

### 5.1 The augmentation mechanism (cache installation, NO re-prefill)

**Augmentation = installing the retrieved chunks' cache snapshots into the running model.** The model's linear-attn state + Kimi M1/M2 caches ARE the augmented context. No text is concatenated, no tokens are re-prefilled.

```python
# poc/augment.py
def augment_and_answer(model, query_token_ids, retrieved_chunk_idxs,
                      snapshot_pool, system_states, device):
    """Install the retrieved chunks' caches into the model, answer the query.
    NO re-prefill. The model answers from the installed caches."""

    # 1. Sum the deltas from all retrieved chunks (composable, lossless)
    restored_states = []
    for layer_idx in range(model.num_linear_layers):  # 24 linear-attn layers
        r_S = system_states[layer_idx][0].clone()      # system prompt's S
        r_M1 = system_states[layer_idx][2].clone()     # system prompt's M1
        r_M2 = system_states[layer_idx][3].clone()     # system prompt's M2
        last_conv = system_states[layer_idx][1]

        for chunk_idx in retrieved_chunk_idxs:  # top-3 chunks
            snap = snapshot_pool.lookup(chunk_idx)  # load from disk/LRU
            r_S = r_S + snap.delta_S_list[layer_idx]      # sum the S delta
            r_M1 = r_M1 + snap.delta_M1_list[layer_idx]   # sum the M1 delta
            r_M2 = r_M2 + snap.delta_M2_list[layer_idx]   # sum the M2 delta
            last_conv = snap.conv_state_list[layer_idx]   # last chunk's conv

        restored_states.append((r_S, last_conv, r_M1, r_M2))

    # 2. The model answers from the installed caches (NO re-prefill)
    with torch.no_grad():
        logits, _ = model(query_token_ids, layer_states=restored_states, return_states=True)

    return logits  # the answer, computed from the installed caches
```

### 5.2 Why this works (the toy's validated findings)

1. **Composable (lossless):** the conv-reset at chunk boundaries makes each chunk's deltas path-independent. Summing any subset of chunks' deltas gives the correct state for that subset. The toy confirmed: diff = 0.0 (lossless) across all subsets and orderings.

2. **Discriminative (after pretrain):** the pretrain trains the linear-attn layers (including M1/M2) to produce states that carry discriminative info. The toy showed: after 1500 pretrain steps, the correct-topic cache makes the topic-marker logit 1.11 higher than the wrong-topic cache (76% correct-vs-wrong).

3. **No re-prefill:** the model answers from the installed caches in ~2 ms (only the query's 32 tokens are prefilled). The ~3,000 retrieved tokens' prefill is saved.

### 5.3 The augmentation's cost

| Component | Per-query cost |
|---|---|
| Load 3 snapshots from disk | ~1 ms (or 0 ms if in LRU) |
| Sum 24 × 3 deltas | ~5 ms |
| Restore into the model | ~0.5 ms (in-place tensor copies) |
| **Total augmentation** | **~6.5 ms** |

vs. re-prefilling 3 × 1024 = 3,072 tokens: ~50-100 ms. **The cache installation is ~10× faster.**

---

## 6. The exact files (from the repo) used at each step

### 6.1 Pretrain (one-time)

| File | Role |
|---|---|
| `scripts/eval_common.py::load_quant_model` | Loads the W4+r32 LUT model |
| `scripts/qlora.py::attach_qlora` + `QLoRAConfig` | Wraps with trainable LUTs (W10) |
| `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` | The W10 autograd Function |
| `scripts/trainer.py` | The layerwise distillation trainer |
| `scripts/muon_optimizer.py` | The optimizer |
| `scripts/data.py::load_sft_dataset` | Loads the corpus as SFT (add OfficeQA chunks) |
| `scripts/calibrate_real_text.py` | Calibration data (FineWeb-Edu, real text) |

### 6.2 Ingestion + snapshot (one-time, per chunk)

| File | Role |
|---|---|
| `scripts/modeling.py::Qwen3_5ForCausalLM` | The model's forward (prefill) |
| `scripts/modeling.py::Qwen3_5GatedDeltaNet` | The linear-attn layer (produces S) |
| `scripts/modeling.py::_fla_resolve` (W29) | The fla wiring for the recurrent state |
| `scripts/palettized_modules.py::PalettizedLinear` | The palettized forward (W4 kernel) |
| New: `poc/ingest.py` | The snapshot loop (prefill each chunk, save deltas) |

### 6.3 IVFADC index (one-time)

| File | Role |
|---|---|
| New: `poc/build_index.py` | Builds the FAISS IVFADC on the pooled vectors |

### 6.4 Query (per user query)

| File | Role |
|---|---|
| `scripts/modeling.py::Qwen3_5ForCausalLM` | The model's forward (query prefill + answer) |
| New: `poc/retrieve.py` | IVFADC preselect + cos sim rerank |
| New: `poc/augment.py` | Cache installation (sum deltas, restore into model) |
| New: `poc/snapshot_pool.py` | The disk-backed LRU pool |

### 6.5 Evaluation

| File | Role |
|---|---|
| `scripts/eval_greedy_match.py::greedy_decode` | Decode the answer |
| `scripts/eval_common.py::atomic_json_dump` | Write the report |
| `scripts/measure_energy.py::EnergyMeasurement` | Measure latency/throughput/VRAM |

---

## 7. The exact workflow, step by step

### Step 0: Prepare the environment (one-time, ~3 days)
1. Clone the repo, build the FLUTE idxN wheels (`pip install -e flute_extended/ flute_train_kernels/`)
2. Run the kernel parity suite (`pytest tests/test_kernel_status.py tests/test_lut_gradients.py tests/test_two_stream_training.py tests/test_attn_kernel.py`)
3. Palettize Qwen3.5-9B: `python scripts/palettize_qwen3_5_9b.py --recipe auto --auto-cos 0.9995 --calib-seqs 32 --calib-seq-len 1024 --mem-temp-mb 64`
4. Verify the W4+r32 model loads: `python scripts/generate.py --artifacts-dir /path/to/palettized --heads-dir /path/to/heads --residual --prompt "Hello"`

### Step 1: Pretrain (one-time, ~4-8 hours on A10G)
1. Load the W4+r32 LUT model (`eval_common.load_quant_model`)
2. Attach QLoRA (`qlora.attach_qlora`, W10 trainable LUTs)
3. Load the OfficeQA corpus as next-token-prediction examples (`data.load_sft_dataset`)
4. Run the layerwise trainer (`trainer.Trainer`) for 500 steps
5. Export the fine-tuned LUTs to `/path/to/pretrained_luts`

### Step 2: Ingest + snapshot (one-time, ~46 min for 50k chunks on A10G)
1. Load the pretrained LUT model (with the fine-tuned LUTs from Step 1)
2. For each of the 50,000 chunks:
   a. Prefill the chunk through the model (conv-reset at boundaries)
   b. Snapshot per linear layer: delta_S, delta_M1, delta_M2, conv_state
   c. Extract the pooled hidden vector
   d. Save to disk: `snapshots/chunk_XXXXX.npz`
3. Build the IVFADC index on the pooled vectors
4. Save the index: `ivfadc_index.npz` + `exact_vectors.bin`

### Step 3: Query (per user query, ~8 s on A10G)
1. Tokenize the query
2. Prefill the query through the model → query vector
3. IVFADC preselect → top-100 candidates (~3 ms)
4. Cos sim rerank → top-3 chunk indices (~1 ms)
5. Load the 3 chunks' snapshots from disk (~1 ms)
6. Sum the deltas (delta_S + delta_M1 + delta_M2) per layer (~5 ms)
7. Install into the model (S = system_S + Σ delta_S, etc.)
8. The model answers from the installed caches (~2 ms)
9. Decode the answer (~8 s for 200 tokens)

### Step 4: Evaluate (the PoC)
- Measure: retrieval hit rate (IVFADC vs gold), answer quality (cache-installed vs gold-installed), latency, VRAM
- The meaningful-query filter (BoxOffice gate): drop model-capability failures, world-knowledge, low-information questions
- The verdict: does the cache-installed answer match the gold answer?

---

## 8. What the toy validated (and what the A10G PoC tests)

### 8.1 The toy validated (on the CPU toy model)

| Finding | Toy result | What it means for the real model |
|---|---|---|
| Composability (conv-reset) | diff = 0.0 (lossless) | The deltas sum correctly — any subset, any order |
| IVFADC retrieval | 87% (mean-pool) | The pooled hidden vector is discriminative |
| Cache carries info (after pretrain) | 76% correct > wrong | The pretrain makes the cache discriminative |
| Latency | 3.48× speedup | The cache installation is faster than re-prefill |
| No re-prefill needed | ✓ | The model answers from the installed caches |

### 8.2 The A10G PoC tests (on the real 9B model)

| Question | How the PoC answers it |
|---|---|
| Does the real model's cache carry discriminative info? | The real model is already trained — measure the correct-vs-wrong logit diff directly (no pretrain needed) |
| Does IVFADC retrieve the right chunks at 50k scale? | Build the IVFADC on 50k pooled vectors, measure recall@3 |
| Is the cache-installed answer correct? | Run 50 OfficeQA questions, measure exact-match + numeric-tolerance |
| Is the latency acceptable? | Measure TTFT, ITL, throughput |
| Does the cache fit in A10G's HBM? | Measure VRAM peak (target: <22 GiB) |

---

## 9. The honest constraints

### 9.1 The cache pool size (the binding constraint)

At ~50 MiB per chunk (24 layers × deltas):
- 50,000 chunks × 50 MiB = **2.4 TiB** on disk (fits on any disk)
- On A10G (13 GiB cache budget): ~260 hot chunks in HBM (LRU)
- For the full 50k: host DRAM (~2.4 TiB, ~$240) — the K3 external pool

**The PoC (50 questions, top-3 retrieval):** ~150 chunks' snapshots needed, ~7.5 GiB — fits in A10G's HBM.

### 9.2 The pretrain's necessity

- **On the toy (random init):** pretrain is ESSENTIAL — without it, the cache carries no info (40%).
- **On the real model (already trained):** pretrain may be a smaller lever — the model's linear-attn already produces meaningful states. The PoC measures this directly: snapshot the trained model's caches, install them, measure the answer quality WITHOUT any pretrain.

### 9.3 The IVFADC retrieval precision

The toy showed 87% with mean-pool. On the real model (4096-dim hidden, trained), retrieval should be 95%+. If below 80%, the fix is a learned projection head (a small linear layer trained with InfoNCE) — but this is a refinement, not a blocker.

---

## 10. Summary — the exact answers

### Against what will we pretrain?
**The OfficeQA corpus chunks (next-token prediction), using the repo's `scripts/trainer.py` + `scripts/qlora.py` (W10 LUT path).** The pretrain trains the linear-attn layers (including M1/M2) to produce states that carry discriminative info. ~500 steps on A10G. On the real model (already trained by Qwen), this is a short fine-tune.

### How will the vectorDB work and what will be stored there?
**An IVFADC index (FAISS `IndexIVFPQ`) on the pooled hidden vectors (4096-dim fp16, one per chunk), PLUS the per-chunk cache snapshots (delta_S, delta_M1, delta_M2, conv_state for 24 layers).** NO chunk text. NO token IDs. NO re-prefill fallback. ~2.4 TiB on disk for 50k chunks.

### How does the user query work?
**Tokenize → prefill the query (32 tokens) → IVFADC preselect → cos sim rerank → top-3 → load snapshots → sum deltas → install into model → answer from installed caches → decode.** ~8 s per query (dominated by decode). No chunk text loaded, no re-prefill.

### How does retrieval work?
**The query's pooled hidden vector (the LUT model's own representation) is searched against the IVFADC index → top-100 candidates → cos sim rerank on exact vectors → top-3 chunk indices.** ~4 ms. The retrieval returns indices (not text), used to load cache snapshots.

### How does augmentation work?
**Sum the top-3 chunks' deltas (delta_S + delta_M1 + delta_M2 per layer) — composable, lossless (conv-reset makes them path-independent). Install into the model: S = system_S + Σ delta_S, M1 = system_M1 + Σ delta_M1, M2 = system_M2 + Σ delta_M2. The model answers from the installed caches.** ~6.5 ms. No text concatenated, no tokens re-prefilled.
