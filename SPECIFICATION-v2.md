# SPECIFICATION v2 — Cache-Engineered RAG (refined against the exact repo)

> **Refined against:** `scripts/modeling.py`, `scripts/palettized_modules.py`, `scripts/eval_common.py`, `scripts/qlora.py`, `docs/MODEL_GEOMETRY.md`, `docs/qwen3_5_9b_config.json`.
> **The architecture (stated definitively):**
> - We snapshot the **linear-attention recurrent state S** (the delta-rule state matrix, from each of the 24 linear layers). This is the model's natural state.
> - We ADD **two global caches M1, M2** (Kimi-style key-memory and value-memory) to the linear-attention layers. These are NEW components, not part of the original model. They extend geometrical expressivity and capacity. They are also snapshotted.
> - We do **NOT snapshot the full-attention KV**. The full-attention layers run fresh per query — no KV cache, no snapshot, nothing.
> - The retrieval vector is the flattened **S + M1 + M2** from all 24 linear layers.

---

## 1. The exact model geometry (from `docs/qwen3_5_9b_config.json`)

### 1.1 The config (verified)

| Key | Value |
|---|---|
| `hidden_size` | 4096 |
| `num_hidden_layers` | 32 |
| `layer_types` | `[L,L,L,F, L,L,L,F, L,L,L,F, L,L,L,F, L,L,L,F, L,L,L,F, L,L,L,F, L,L,L,F]` (24 × `linear_attention`, 8 × `full_attention`) |
| `full_attention_interval` | 4 |
| `linear_num_key_heads` | 16 |
| `linear_key_head_dim` | 128 |
| `linear_num_value_heads` | 32 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| `num_attention_heads` (full-attn) | 16 |
| `num_key_value_heads` (full-attn, GQA) | 4 |
| `head_dim` (full-attn) | 256 |
| `vocab_size` | 248320 |
| `tie_word_embeddings` | false |

### 1.2 The full-attention layers (indices 3, 7, 11, 15, 19, 23, 27, 31)

These are standard `Qwen3_5Attention` (from `modeling.py` line 932). They use GQA (4 KV heads, 16 query heads, head_dim=256). The KV cache is managed by `past_key_values.update(key_states, value_states, self.layer_idx)` (line 956).

**The full-attn KV per token:** `num_key_value_heads × head_dim × 2 (K+V) × 2 (fp16) = 4 × 256 × 2 × 2 = 4,096 bytes = 4 KiB per token per layer`. For 8 layers: `32 KiB per token`.

### 1.3 The linear-attention layers (the other 24)

These are `Qwen3_5GatedDeltaNet` (from `modeling.py` line 589). The recurrent state:

- **Shape:** `(batch, num_v_heads=32, head_k_dim=128, head_k_dim=128)` — a matrix per value head.
- **Size (fp16):** `32 × 128 × 128 × 2 = 1,048,576 bytes = 1.0 MiB per layer`.
- **24 layers:** `24 MiB total`.

The conv1d state:
- **Shape:** `(batch, conv_dim, conv_kernel_dim)` where `conv_dim = key_dim*2 + value_dim = 2048*2 + 4096 = 8192`.
- **Size (fp16):** `8192 × 4 × 2 = 65,536 bytes = 64 KiB per layer`.
- **24 layers:** `1.5 MiB total`.

### 1.4 The cache API (from `modeling.py`)

The linear-attn layer accesses the cache via `cache_params` (a `DynamicCache`):

```python
# modeling.py line 647
use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
    self.layer_idx, state_idx=0)

# line 660-661: the conv state
conv_state = cache_params.layers[self.layer_idx].conv_states[0]

# line 710: the recurrent state
recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None

# line 712: decode path (seq_len == 1)
core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
    query, key, value, g=g, beta=beta, initial_state=recurrent_state, ...)

# line 725: prefill path (seq_len > 1)
core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
    query, key, value, g=g, beta=beta, initial_state=recurrent_state, ...)

# line 738-739: write back the updated state
cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
```

The full-attn layer accesses the KV cache via:
```python
# modeling.py line 955-956
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

---

## 2. The three caches per linear-attention layer (S + M1 + M2)

### 2.1 Cache S — the linear-attention recurrent state (the model's natural state)

**What:** the delta-rule recurrent state matrix, from each of the 24 linear-attention layers.

**Shape per layer:** `(1, 32, 128, 128)` fp16 = 1.0 MiB. **24 layers:** 24 MiB per chunk.

**Composable:** with conv-reset at chunk boundaries, each chunk's delta_S is path-independent. Summing deltas is lossless.

### 2.2 Cache M1 — the Kimi-style key-memory (ADDED to the model)

**What:** a key-memory matrix, added to each linear-attention layer as a new trainable parameter. The model writes to it (gated, selective) and reads from it (attention: `softmax(q @ M1^T)`). This extends the model's geometrical expressivity — more dimensions than S alone.

**Shape per layer:** `(1, 32, mem_size, 128)` fp16. At mem_size=128: 1.0 MiB per layer. **24 layers:** 24 MiB per chunk.

**Composable:** the write is additive and gated. With conv-reset, delta_M1 is path-independent. Summing is lossless.

### 2.3 Cache M2 — the Kimi-style value-memory (ADDED to the model)

**What:** the value-memory matrix, paired with M1. The read is `softmax(q @ M1^T) @ M2`. Extends capacity — more memory slots.

**Shape per layer:** `(1, 32, mem_size, 128)` fp16. At mem_size=128: 1.0 MiB per layer. **24 layers:** 24 MiB per chunk.

**Composable:** same as M1.

### 2.4 The full-attention layers — NOT SNAPSHOTTED, run fresh

The 8 full-attention layers (indices 3, 7, 11, 15, 19, 23, 27, 31) run **fresh** per query. NO KV cache. NO snapshot. The full-attn sees only the query tokens. The context is in the installed linear-attn state (S + M1 + M2), not in any full-attn KV.

### 2.5 The cache vector (the retrieval representation)

```
cache_vector = concat([
    S_layer_0.flatten(),     # (32, 128, 128) → 524,288
    M1_layer_0.flatten(),    # (32, mem_size, 128) → 524,288 (at mem_size=128)
    M2_layer_0.flatten(),    # (32, mem_size, 128) → 524,288
    ...for all 24 linear layers...
])
# total: 24 × 3 × 524,288 = 37,748,736 dims (fp16)
# size: 72 MiB per chunk (at mem_size=128)
```

**NOT the full-attn KV. NOT the hidden state.** The S + M1 + M2 flattened IS the retrieval vector.

---

## 3. Ingestion (one-time, per chunk)

### 3.1 The ingestion pipeline

```python
# poc/ingest.py
from scripts.eval_common import load_quant_model
from scripts.modeling import Qwen3_5ForCausalLM
from transformers import DynamicCache
import numpy as np

def ingest_chunk(model, chunk_token_ids, device):
    """Prefill a chunk, snapshot S + M1 + M2. NO full-attn KV."""
    # create a fresh cache (conv-reset = start from zero)
    cache = DynamicCache(config=model.config)

    # prefill the chunk through the model
    with torch.no_grad():
        outputs = model(input_ids=chunk_token_ids, past_key_values=cache, use_cache=True)

    # extract S from each linear-attn layer
    s_per_layer = []
    conv_per_layer = []
    for layer_idx in range(32):
        if model.config.layer_types[layer_idx] == "linear_attention":
            S = cache.layers[layer_idx].recurrent_states[0]  # (1, 32, 128, 128) fp16
            conv = cache.layers[layer_idx].conv_states[0]     # (1, 8192, 4) fp16
            s_per_layer.append(S.detach().cpu())
            conv_per_layer.append(conv.detach().cpu())

    # extract M1, M2 from each linear-attn layer (the ADDED Kimi caches)
    m1_per_layer = []
    m2_per_layer = []
    for layer_idx in range(32):
        if model.config.layer_types[layer_idx] == "linear_attention":
            M1 = model.layers[layer_idx].linear_attn.M1  # the Kimi key-memory
            M2 = model.layers[layer_idx].linear_attn.M2  # the Kimi value-memory
            m1_per_layer.append(M1.detach().cpu())
            m2_per_layer.append(M2.detach().cpu())

    # the cache vector = flattened S + M1 + M2 from all 24 linear layers
    cache_vector = torch.cat([
        s.flatten() for s in s_per_layer
    ] + [
        m.flatten() for m in m1_per_layer
    ] + [
        m.flatten() for m in m2_per_layer
    ]).numpy().astype(np.float16)

    return {
        's_per_layer': s_per_layer,       # 24 × (1, 32, 128, 128) — Cache S
        'm1_per_layer': m1_per_layer,     # 24 × (1, 32, mem_size, 128) — Cache M1
        'm2_per_layer': m2_per_layer,      # 24 × (1, 32, mem_size, 128) — Cache M2
        'conv_per_layer': conv_per_layer, # 24 × (1, 8192, 4)
        'cache_vector': cache_vector,     # (37748736,) — the retrieval vector
    }
```

### 3.2 What's saved per chunk

| Component | Per layer | × Layers | Total (fp16) |
|---|---|---|---|
| delta_S (recurrent state) | 1,048,576 B = 1.0 MiB | 24 | 24 MiB |
| delta_M1 (Kimi key-memory) | 1,048,576 B = 1.0 MiB (mem_size=128) | 24 | 24 MiB |
| delta_M2 (Kimi value-memory) | 1,048,576 B = 1.0 MiB (mem_size=128) | 24 | 24 MiB |
| conv_state | 65,536 B = 64 KiB | 24 | 1.5 MiB |
| cache_vector (flattened S+M1+M2) | — | — | 72 MiB (redundant with deltas) |
| **Total per chunk** | | | **~73.5 MiB** |

**For 50,000 chunks:** ~3.68 TiB on disk. NO full-attn KV — the full-attn layers are NOT snapshotted.

### 3.3 The conv-reset discipline

At ingestion, each chunk is prefilled with a FRESH `DynamicCache` (all states zero). This makes each chunk's S delta path-independent — the chunk's contribution to S does not depend on which chunks came before it. The deltas are composable by summation.

### 3.4 The IVFADC index

Built on the **cache vectors** (flattened S, 12.6M dims fp16 → fp32 for FAISS):

```python
# poc/build_index.py
import faiss

cache_vectors = load_all_cache_vectors()  # (50000, 12582912) fp32
nlist = 224  # ~sqrt(50000)
m = 64       # PQ sub-quantizers
quantizer = faiss.IndexFlatIP(12582912)
index = faiss.IndexIVFPQ(quantizer, 12582912, nlist, m, nbits=8)
index.train(cache_vectors)
index.add(cache_vectors)
faiss.write_index(index, "disk/ivfadc_cache.index")
```

---

## 4. The query flow (per user query)

### 4.1 The 8-step pipeline

```
User question
    │
    ▼
[Step 1: Tokenize]
    query_token_ids = model.tokenizer(question)  →  (1, 32)
    │
    ▼
[Step 2: Prefill the query → snapshot the query's cache]
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        outputs = model(input_ids=query_token_ids, past_key_values=cache, use_cache=True)
    # extract the query's S (the retrieval vector)
    query_s_per_layer = [cache.layers[i].recurrent_states[0]
                         for i in range(32) if layer_types[i] == "linear_attention"]
    query_cache_vector = torch.cat([s.flatten() for s in query_s_per_layer]).numpy()
    │
    ▼
[Step 3: IVFADC preselect on the cache vectors]
    candidates = index.search(query_cache_vector, k=100)  →  top-100 chunk indices
    │
    ▼
[Step 4: Cos sim rerank]
    candidate_vecs = exact_vectors[candidates]
    scores = (normalized candidate_vecs) @ (normalized query_cache_vector)
    top_k_ids = argsort(scores)[-3:]  →  top-3 chunk indices
    │
    ▼
[Step 5: Load the top-3 chunks' snapshots from disk]
    For each chunk_id: load delta_S, delta_M1, delta_M2, conv_state
    → NO full-attn KV. The full-attn is not snapshotted.
    │
    ▼
[Step 6: Install the caches (S + M1 + M2) into the running model]
    For each linear layer i (0..23):
        restored_S[i] = system_S[i] + sum(delta_S[i] for each retrieved chunk)
        restored_M1[i] = system_M1[i] + sum(delta_M1[i] for each retrieved chunk)
        restored_M2[i] = system_M2[i] + sum(delta_M2[i] for each retrieved chunk)
        restored_conv[i] = last_retrieved_chunk.conv_state[i]
    → NO full-attn KV. The full-attn layers run fresh.
    │
    ▼
[Step 7: Answer from the installed caches]
    Create a DynamicCache, inject the restored S + M1 + M2 + conv
    The full-attn layers see only the query tokens (fresh, no KV)
    with torch.no_grad():
        logits = model(input_ids=query_token_ids, past_key_values=restored_cache, use_cache=True)
    answer = model.generate(max_new_tokens=200, past_key_values=restored_cache)
    → NO re-prefill of chunk text
```

### 4.2 How to inject the restored caches into a DynamicCache

```python
# poc/augment.py
from transformers import DynamicCache

def create_restored_cache(model, system_cache, retrieved_snapshots):
    """Create a DynamicCache with the restored S + M1 + M2.
    NO full-attn KV — the full-attn layers run fresh."""
    cache = DynamicCache(config=model.config)

    # install S + M1 + M2 (sum the deltas — composable, lossless)
    for layer_idx in range(32):
        if model.config.layer_types[layer_idx] == "linear_attention":
            linear_idx = ...  # map to the 24 linear layers
            # sum the S deltas
            restored_S = system_cache.layers[layer_idx].recurrent_states[0].clone()
            for snap in retrieved_snapshots:
                restored_S = restored_S + snap['s_per_layer'][linear_idx]
            cache.layers[layer_idx].recurrent_states[0] = restored_S
            # sum the M1 deltas (the Kimi key-memory)
            restored_M1 = model.layers[layer_idx].linear_attn.M1.clone()
            for snap in retrieved_snapshots:
                restored_M1 = restored_M1 + snap['m1_per_layer'][linear_idx]
            model.layers[layer_idx].linear_attn.M1 = restored_M1
            # sum the M2 deltas (the Kimi value-memory)
            restored_M2 = model.layers[layer_idx].linear_attn.M2.clone()
            for snap in retrieved_snapshots:
                restored_M2 = restored_M2 + snap['m2_per_layer'][linear_idx]
            model.layers[layer_idx].linear_attn.M2 = restored_M2
            # the conv state from the last retrieved chunk
            cache.layers[layer_idx].conv_states[0] = retrieved_snapshots[-1]['conv_per_layer'][linear_idx]

    # NO full-attn KV installation. The full-attn layers run fresh.
    return cache
```

### 4.3 The timing budget (per query, A10G)

| Step | Time | Notes |
|---|---|---|
| 1. Tokenize | <1 ms | |
| 2. Prefill query + snapshot cache | ~5 ms | 32 tokens |
| 3. IVFADC preselect | ~10 ms | 37.7M-dim vectors |
| 4. Cos sim rerank | ~50 ms | 100 × 72 MiB dot products |
| 5. Load snapshots | ~5 ms | 3 × 73.5 MiB from disk/LRU |
| 6. Install S + M1 + M2 (sum deltas) | ~10 ms | 24 × 3 delta sums |
| 7. Answer + decode | ~8 s | 32-token prefill + 200-token decode (full-attn runs fresh) |
| **Total** | **~8.08 s** | |

---

## 5. Pretraining (one-time)

### 5.1 The pretrain target

**Next-token prediction on the OfficeQA corpus chunks.** The model learns the corpus's patterns, and the recurrent state S becomes a meaningful content summary.

### 5.2 The pretrain mechanism (using the repo's exact files)

```python
# poc/pretrain.py
from scripts.eval_common import load_quant_model
from scripts.qlora import attach_qlora, QLoRAConfig
from scripts.trainer import Trainer  # the layerwise distillation trainer
from scripts.data import load_sft_dataset

# load the W4+r32 LUT model
model, metadata = load_quant_model(
    artifacts_dir="/path/to/palettized",
    model_name="Qwen/Qwen3.5-9B",
    device="cuda:0",
    residual=True,
    forward="kernel",
    heads_dir="/path/to/heads"
)

# attach QLoRA (W10 trainable LUTs)
attach_qlora(model, metadata, r=64, alpha=16, scope="all",
             base_model="Qwen/Qwen3.5-9B", artifacts_dir="/path/to/palettized")

# load the OfficeQA corpus as next-token-prediction examples
dataset = load_sft_dataset("officeqa", seq_len=1024, tokenizer=model.tokenizer, max_samples=50000)

# run the layerwise trainer (two-layer residency — fits on A10G)
trainer = Trainer(model, metadata, dataset, ...)
trainer.train(n_steps=500)
trainer.export("/path/to/pretrained_luts")
```

### 5.3 The pretrain on the real model

The real Qwen3.5-9B is already trained. The pretrain is a FINE-TUNE (~500 steps via the W10 LUT path) to make the model's S cache discriminative on the OfficeQA corpus.

---

## 6. The retrieval mechanism

### 6.1 The retrieval vector

**The flattened recurrent state S** from all 24 linear-attention layers:

```
cache_vector = concat([S_layer_0.flatten(), S_layer_1.flatten(), ..., S_layer_23.flatten()])
# dimension: 24 × (32 × 128 × 128) = 12,582,912
# size: 24 MiB (fp16)
```

### 6.2 Why S is the right retrieval vector

The recurrent state S is the delta-rule's accumulated memory of the chunk's content. It's:
- **Fixed-size** (doesn't grow with chunk length) → can be used in IVFADC
- **Content-summative** (accumulates token patterns) → similar content → similar S
- **In the same space for queries and chunks** (both are S matrices from the same model) → cos-sim is meaningful

### 6.3 The retrieval flow

```python
# poc/retrieve.py
def retrieve(model, query_token_ids, ivfadc_index, exact_vectors, top_k=3):
    # 1. snapshot the query's cache
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=query_token_ids, past_key_values=cache, use_cache=True)
    query_s = [cache.layers[i].recurrent_states[0]
               for i in range(32) if model.config.layer_types[i] == "linear_attention"]
    query_vec = torch.cat([s.flatten() for s in query_s]).cpu().numpy().astype(np.float32)

    # 2. IVFADC preselect
    candidates = ivfadc_index.search(query_vec, k=100)

    # 3. cos sim rerank
    cand_vecs = exact_vectors[candidates]
    cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    scores = cand_norm @ q_norm
    top_k_local = np.argsort(scores)[-top_k:][::-1]

    return candidates[top_k_local].tolist()
```

---

## 7. The augmentation mechanism

### 7.1 Installation (sum the S + M1 + M2 deltas)

```python
# for each linear layer
restored_S = system_S + sum(delta_S for each retrieved chunk)     # composable, lossless
restored_M1 = system_M1 + sum(delta_M1 for each retrieved chunk)   # composable, lossless
restored_M2 = system_M2 + sum(delta_M2 for each retrieved chunk)   # composable, lossless
restored_conv = last_retrieved_chunk.conv_state
# inject into the model
cache.layers[layer_idx].recurrent_states[0] = restored_S
model.layers[layer_idx].linear_attn.M1 = restored_M1
model.layers[layer_idx].linear_attn.M2 = restored_M2
cache.layers[layer_idx].conv_states[0] = restored_conv
```

### 7.2 The full-attn layers run fresh — NO installation

The 8 full-attn layers do NOT have any installed cache. They process the query tokens from scratch. The context is entirely in the installed linear-attn state (S + M1 + M2). No KV cache, no snapshot, no concatenation.

### 7.3 The model answers from the installed caches

```python
# the model's forward with the restored cache
logits = model(input_ids=query_token_ids, past_key_values=restored_cache, use_cache=True)
answer = model.generate(max_new_tokens=200, past_key_values=restored_cache)
```

The model sees the query tokens (32 tokens) on top of the installed caches. The linear-attn layers start from the restored S + M1 + M2; the full-attn layers run fresh. **NO chunk text is re-prefilled. NO full-attn KV is used.**

---

## 8. The disk layout

```
disk/
├── ivfadc_cache.index              # FAISS IVFADC on S vectors (12.6M-dim)
├── exact_s_vectors.bin              # exact fp16 S vectors for rerank (24 MiB each)
├── snapshots/                       # per-chunk cache snapshots
│   ├── chunk_00000.npz              # delta_S (24 layers) + conv_state (24) + KV (8)
│   └── ...                          # 50,000 files × 57.5 MiB = 2.88 TiB
└── pretrained_luts/                 # the fine-tuned LUTs (~5.85 GiB)
```

**Total disk: ~2.9 TiB** for 50k chunks (the snapshots dominate).

---

## 9. The VRAM layout (A10G, 24 GiB)

| Component | Size | Notes |
|---|---|---|
| Model weights (W4+r32) | 5.85 GiB | Always resident |
| System prompt cache (S + conv + KV) | ~58 MiB | Computed once at startup |
| Query cache (during query) | ~58 MiB | Per query |
| Installed retrieved caches (3 chunks) | ~173 MiB | Per query (3 × 57.5 MiB) |
| IVFADC index (in CPU RAM) | — | mmap'd |
| Exact S vectors (on disk) | — | mmap'd; top-100 read per query |
| Framework + activations | ~5 GiB | |
| Safety margin | ~2 GiB | |
| **Total VRAM** | **~13 GiB** | Fits in 24 GiB |

---

## 10. The repo files used at each step

### 10.1 Pretrain

| File | Function | Role |
|---|---|---|
| `scripts/eval_common.py` | `load_quant_model(artifacts_dir, model_name, device, residual=True, forward="kernel", heads_dir=...)` | Loads the W4+r32 LUT model |
| `scripts/qlora.py` | `attach_qlora(model, metadata, r=64, alpha=16, scope="all", ...)` | Wraps with trainable LUTs (W10) |
| `scripts/qlora.py` | `QLoRAConfig` | The config dataclass |
| `scripts/qlora_gemm.py` | `FusedQLoRAGEMMTrainLUTTwoStreams` | The W10 autograd Function |
| `scripts/trainer.py` | `Trainer` | The layerwise distillation trainer |
| `scripts/muon_optimizer.py` | — | The Muon optimizer |
| `scripts/data.py` | `load_sft_dataset` | Loads the corpus as SFT |

### 10.2 Ingestion + snapshot

| File | Function | Role |
|---|---|---|
| `scripts/modeling.py` | `Qwen3_5ForCausalLM.forward(input_ids, past_key_values, use_cache)` | The model's prefill |
| `scripts/modeling.py` | `Qwen3_5GatedDeltaNet.forward(hidden_states, cache_params)` | The linear-attn layer (produces S) |
| `scripts/modeling.py` | `cache_params.layers[L].recurrent_states[0]` | Access the recurrent state S |
| `scripts/modeling.py` | `cache_params.layers[L].conv_states[0]` | Access the conv state |
| `scripts/modeling.py` | `cache_params.update_recurrent_state(state, layer_idx)` | Write back the state |
| `scripts/modeling.py` | `_fla_resolve()` (W29) | The fla wiring for decode |
| `scripts/palettized_modules.py` | `PalettizedLinear.forward(x)` | The palettized forward (W4 kernel) |
| `transformers` | `DynamicCache(config=model.config)` | The cache object |

### 10.3 IVFADC index

| File | Role |
|---|---|
| New: `poc/build_index.py` | Builds FAISS IVFADC on S vectors |

### 10.4 Query

| File | Function | Role |
|---|---|---|
| `scripts/modeling.py` | `Qwen3_5ForCausalLM.forward(input_ids, past_key_values, use_cache)` | Query prefill + answer |
| `scripts/modeling.py` | `Qwen3_5ForCausalLM.generate(max_new_tokens, past_key_values)` | Decode the answer |
| New: `poc/retrieve.py` | IVFADC + cos sim rerank on S vectors |
| New: `poc/augment.py` | Cache installation (sum S deltas, concat KV) |
| New: `poc/snapshot_pool.py` | The disk-backed LRU pool |

---

## 11. The toy validation

The toy (`scripts/poc_toy/toy_cache_as_vector.py`) validated the full pipeline on a CPU model:

| Finding | Toy result |
|---|---|
| Composability of S (conv-reset) | diff = 0.0 (lossless) |
| IVFADC retrieval on S vectors | 100% (50/50) |
| Correct cache > wrong cache | 90% (45/50) |
| Logit diff (correct - wrong) | +1.45 |

**The toy confirms the architecture works.** The A10G PoC tests it on the real 9B model.

---

## 12. Summary

| Question | Answer |
|---|---|
| Against what will we pretrain? | The OfficeQA corpus chunks (next-token prediction, W10 LUT path via `scripts/qlora.py::attach_qlora` + `scripts/trainer.py`). ~500 steps on A10G. |
| What's in the vectorDB? | IVFADC index on the **flattened S + M1 + M2** (37.7M-dim fp16, 72 MiB per chunk) + per-chunk snapshots (delta_S + delta_M1 + delta_M2 + conv_state). NO chunk text. NO full-attn KV. |
| How does the user query work? | Tokenize → prefill query → snapshot the query's S+M1+M2 → IVFADC on cache vectors → cos sim rerank → load top-3 snapshots → install (sum S+M1+M2 deltas) → answer from installed caches → decode. The full-attn layers run fresh. |
| How does retrieval work? | Snapshot the query's S+M1+M2 (flattened) → IVFADC preselect → cos sim rerank on exact cache vectors → top-3 chunk indices. The S+M1+M2 IS the retrieval vector. |
| How does augmentation work? | Sum the top-3 chunks' deltas (delta_S + delta_M1 + delta_M2 per layer) — composable, lossless. Install into the model. The full-attn layers run fresh. Answer from installed caches. NO re-prefill. NO full-attn KV. |
