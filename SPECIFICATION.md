# Cache-Engineered RAG — Specification

> **The LUT model's cache IS the retrieval vector. NO separate embedder. NO chunk text on disk. NO re-prefill. TurboQuant online (3.5-bit, quality-neutral).**

---

## 1. The model

Qwen3.5-9B (FLUTE idxN W4+r32), loaded via `scripts/loader.py::load_quant_model`.

32 layers: 24 linear-attention (`Qwen3_5GatedDeltaNet`) + 8 full-attention (`Qwen3_5Attention`).

```
Layer  0: linear   ← HOOK 0 (capture S_0)
Layer  1: linear
Layer  2: linear   ← HOOK 1 (after layer 3 full-attn, capture S_1, S_2)
Layer  3: full
Layer  4: linear
Layer  5: linear
Layer  6: linear   ← HOOK 2 (after layer 7, capture S_4, S_5, S_6)
Layer  7: full
...
Layer 28: linear
Layer 29: linear
Layer 30: linear   ← HOOK 8 (after layer 31, capture S_28, S_29, S_30)
Layer 31: full
```

**9 hooks capture all 24 per-layer S tensors.** Hook 0 captures S_0 (1 layer). Hooks 1-8 capture 3 layers each. Total: 1 + 8×3 = 24. ✓

Config (from `docs/qwen3_5_9b_config.json`):

| Key | Value |
|---|---|
| `hidden_size` | 4096 |
| `num_hidden_layers` | 32 |
| `layer_types` | 24 × `linear_attention` + 8 × `full_attention`, pattern `[L,L,L,F,...]` |
| `linear_num_value_heads` | 32 |
| `linear_key_head_dim` | 128 |
| `linear_num_value_heads` | 32 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| `vocab_size` | 248320 |

---

## 2. The caches

### 2.1 S — the per-layer recurrent state (the model's natural state)

Each of the 24 `Qwen3_5GatedDeltaNet` layers has its own S, stored in `cache_params.layers[layer_idx].recurrent_states[0]`.

- **Shape per layer:** `(1, 32, 128, 128)` — the delta-rule state matrix.
- **Size per layer (fp16):** 1.0 MiB.
- **24 layers:** 24 MiB.
- **Composable:** with conv-reset at chunk boundaries, each chunk's delta_S is path-independent. Summing deltas is lossless (toy-validated: diff = 0.0).

### 2.2 M1, M2 — the two global caches (ADDED, not in the original model)

We ADD two global memory matrices, shared across ALL 24 linear layers. These extend geometrical expressivity and capacity.

- **M1 (global key-memory):** `(1, 32, mem_size, 128)`. At mem_size=128: 1.0 MiB.
- **M2 (global value-memory):** `(1, 32, mem_size, 128)`. At mem_size=128: 1.0 MiB.
- **TWO for the whole model.** NOT per-layer.
- The model writes to them (gated, additive) and reads from them (`softmax(q @ M1^T) @ M2`).
- **Composable:** the additive write makes delta_M1/M2 path-independent. Summing is lossless.

### 2.3 conv_state — per-layer

Each linear layer has its own conv_state: `cache_params.layers[layer_idx].conv_states[0]`.

- **Shape per layer:** `(1, 8192, 4)`.
- **Size per layer (fp16):** 64 KiB.
- **24 layers:** 1.5 MiB.

### 2.4 Full-attention layers — NOT snapshotted

The 8 full-attn layers run fresh. They keep their state after the user query is prefilled, then converge by the next forward passes. No KV snapshot. No special handling.

---

## 3. Online TurboQuant (3.5-bit, quality-neutral)

### 3.1 The method (arXiv:2504.19874)

TurboQuant is a **data-oblivious, online** vector quantizer with provably near-optimal distortion (within 2.7× of the information-theoretic lower bound).

1. **Random rotation:** multiply the vector by a random rotation matrix (Hadamard/FHT — the repo has `flute_extended/src/kernel_fht.cu`). This makes every coordinate follow the same Beta distribution.
2. **Optimal scalar quantization:** per-coordinate Lloyd-Max quantizer on the Beta distribution. Codebooks precomputed once per bit-width. No calibration.
3. **At 3.5 bits:** the paper proved **absolute quality neutrality** for KV cache — identical to fp16 on LongBench-E and Needle-In-A-Haystack.

### 3.2 Online integration (NOT a sidechain)

The cache states (S, M1, M2, conv_state) are **ALWAYS** TurboQuant codes — during the model's forward pass, on disk, at installation:

```
WRITE (state update in the forward pass):
  the delta rule produces fp16 S
  → TurboQuant.quant(S) → codes stored in cache
  → the cache NEVER holds fp16

READ (state consumption in the next forward step):
  → TurboQuant.dequant(cache.codes) → fp16 for the delta rule
  → the result is re-quantized on write
```

The monkey-patched cache API intercepts both reads (`cache.layers[L].recurrent_states[0]` → dequantize) and writes (`cache.update_recurrent_state(state, L)` → quantize). The model's forward doesn't know it's using quantized caches.

### 3.3 Compression

| Object | fp16 | TurboQuant 3.5-bit | Ratio |
|---|---|---|---|
| S (24 layers) | 24 MiB | 5.4 MiB | 4.6× |
| conv_state (24 layers) | 1.5 MiB | 0.34 MiB | 4.6× |
| M1 (global) | 1.0 MiB | 0.22 MiB | 4.6× |
| M2 (global) | 1.0 MiB | 0.22 MiB | 4.6× |
| **Per chunk** | **27.5 MiB** | **~6.0 MiB** | **4.6×** |
| **50k chunks** | **1.375 TiB** | **~300 GiB** | **4.6×** |

---

## 4. The retrieval vector

```
cache_vector = concat([
    S_0.flatten(),  S_1.flatten(),  S_2.flatten(),      # layers 0, 1, 2
    S_4.flatten(),  S_5.flatten(),  S_6.flatten(),      # layers 4, 5, 6
    ...all 24 linear layers...
    S_28.flatten(), S_29.flatten(), S_30.flatten(),     # layers 28, 29, 30
    M1.flatten(),                                        # the global key-memory
    M2.flatten(),                                        # the global value-memory
])
# at mem_size=128: 24 × 524,288 + 2 × 524,288 = 26 × 524,288 = 13,631,488 dims
# size: 26.0 MiB (fp16) — dequantized from TurboQuant codes for IVFADC
```

The query's cache vector and the chunk's cache vector are in the **SAME space** (both are linear-attn state matrices from the same model). Cos-sim measures content overlap = relevance. NO hidden states. NO separate embedder.

---

## 5. Ingestion (one-time, per chunk)

```python
def ingest_chunk(model, chunk_token_ids, device, tq_cache):
    """Prefill a chunk. The cache is ALWAYS TurboQuant codes (online).
    Capture S at 9 hooks + M1/M2. Save TurboQuant codes to disk."""

    # the cache is already TurboQuant-wrapped (online)
    # prefill → the forward writes TurboQuant codes on every state update
    with torch.no_grad():
        model(input_ids=chunk_token_ids, past_key_values=tq_cache, use_cache=True)

    # snapshot: just read the TurboQuant codes from the cache (already compressed)
    s_codes = {}
    conv_codes = {}
    for layer_idx in range(32):
        if layer_types[layer_idx] == "linear_attention":
            s_codes[layer_idx] = tq_cache.s_codes[layer_idx]       # TurboQuant codes
            conv_codes[layer_idx] = tq_cache.conv_codes[layer_idx]  # TurboQuant codes

    m1_codes = tq_cache.m1_codes  # global M1 TurboQuant codes
    m2_codes = tq_cache.m2_codes  # global M2 TurboQuant codes

    # the retrieval vector = dequantized flattened S + M1 + M2 (for IVFADC)
    dq_s = [tq_s.dequantize(*s_codes[i]) for i in sorted(s_codes.keys())]
    dq_m1 = tq_m1.dequantize(*m1_codes)
    dq_m2 = tq_m2.dequantize(*m2_codes)
    cache_vector = np.concatenate(dq_s + [dq_m1, dq_m2]).astype(np.float32)

    # save to disk (TurboQuant codes only — ~6 MiB per chunk)
    np.savez(f"snapshots/chunk_{idx:05d}.npz",
             s_codes=s_codes, conv_codes=conv_codes,
             m1_codes=m1_codes, m2_codes=m2_codes,
             cache_vector=cache_vector)

    return cache_vector  # for IVFADC index building
```

### Per-chunk disk (TurboQuant codes only)

| Component | Size |
|---|---|
| 24 × S codes (3.5-bit, 524,288 dims each) | 5.4 MiB |
| 24 × conv_state codes (3.5-bit, 32,768 dims each) | 0.34 MiB |
| M1 codes (3.5-bit, 524,288 dims) | 0.22 MiB |
| M2 codes (3.5-bit, 524,288 dims) | 0.22 MiB |
| **Total per chunk** | **~6.0 MiB** |

**50k chunks: ~300 GiB.** NO chunk text. NO token IDs. NO fp16.

---

## 6. The query flow

```
[1. Tokenize the query]
[2. Prefill the query through the model (with online TurboQuant cache)
    → the query's S (24) + M1 + M2 are captured as TurboQuant codes
    → dequantize for IVFADC → the query's cache_vector]
[3. IVFADC preselect on cache vectors → top-100 candidates]
[4. Cos sim rerank → top-3 chunk indices]
[5. Load the top-3 chunks' TurboQuant codes from disk (~18 MiB total)]
[6. Install: sum the S deltas (24 per-layer) + sum M1/M2 deltas (2 global)
    → the sums are on the TurboQuant CODES (not dequantized)
    → install the summed codes into the cache]
[7. Answer from the installed caches
    → the model's forward dequantizes on read (online TurboQuant)
    → the full-attn layers run fresh]
[8. Decode the answer]
```

### The installation

```python
def install_and_answer(model, query_token_ids, retrieved_code_snapshots, system_codes, tq_cache):
    """Install TurboQuant codes (summed) into the cache. Model dequantizes on read."""

    # 1. sum the S code deltas (24 per-layer)
    for layer_idx in range(32):
        if layer_types[layer_idx] == "linear_attention":
            # sum the TurboQuant codes from the retrieved chunks
            summed_s_codes = sum_turboquant_codes(
                system_codes.s_codes[layer_idx],
                [snap['s_codes'][layer_idx] for snap in retrieved_code_snapshots])
            tq_cache.s_codes[layer_idx] = summed_s_codes

            # conv_state: use the last retrieved chunk's
            tq_cache.conv_codes[layer_idx] = retrieved_code_snapshots[-1]['conv_codes'][layer_idx]

    # 2. sum the M1/M2 code deltas (2 global)
    tq_cache.m1_codes = sum_turboquant_codes(
        system_codes.m1_codes,
        [snap['m1_codes'] for snap in retrieved_code_snapshots])
    tq_cache.m2_codes = sum_turboquant_codes(
        system_codes.m2_codes,
        [snap['m2_codes'] for snap in retrieved_code_snapshots])

    # 3. the full-attn layers run fresh — NO installation
    # 4. answer from the installed caches (model dequantizes on read)
    with torch.no_grad():
        logits = model(input_ids=query_token_ids, past_key_values=tq_cache, use_cache=True)
    return logits
```

**Note on summing TurboQuant codes:** TurboQuant codes are b-bit indices + outlier masks + norms. Summing codes requires dequantizing, summing the fp16 vectors, and re-quantizing. This is a fast operation (dequant + sum + requant for 26 tensors of ~500K dims). The alternative is to sum in the rotated space (before dequantization) — possible because the rotation is linear and the scalar quantizer is per-coordinate. This is an optimization for the PoC to test.

---

## 7. Pretraining

Next-token prediction on the OfficeQA corpus via the W10 LUT path:
- `scripts/loader.py::load_quant_model` — loads the W4+r32 model
- A lean training loop written as part of the RAG build on the GPU box
  (the parent project's QLoRA/distillation trainer is NOT part of this
  repo); straight-through LUTs via `PalettizedLinear.make_trainable()`
  on the reference path (`forward="reference"`)
- Trains: the 24 per-layer linear-attn params (so S is discriminative) + the M1/M2 read/write gates (so the global caches carry info) + the LUTs (W10)
- ~500 steps on A10G. The real model is already trained by Qwen — this is a fine-tune.
- The fine-tuned LUTs are served as full LUT artifacts (`pretrained_luts/`, §11) — not as QLoRA adapters.

---

## 8. IVFADC

Built on the **dequantized cache vectors** (fp32, 13.6M dims) using FAISS:

```python
quantizer = faiss.IndexFlatIP(13631488)
index = faiss.IndexIVFPQ(quantizer, 13631488, nlist=224, m=64, nbits=8)
index.train(cache_vectors_fp32)
index.add(cache_vectors_fp32)
```

- **nprobe=8** clusters probed per query
- The exact vectors (for rerank) are the dequantized cache vectors (loaded on demand from the TurboQuant codes)
- **Indexing time: ~0** (TurboQuant is data-oblivious — no k-means training, unlike PQ)

---

## 9. Timing (per query, A10G)

| Step | Time |
|---|---|
| 1. Tokenize | <1 ms |
| 2. Prefill query + snapshot cache (online TQ) | ~8 ms |
| 3. IVFADC preselect | ~10 ms |
| 4. Cos sim rerank | ~20 ms |
| 5. Load TurboQuant codes (3 chunks × 6 MiB) | ~3 ms |
| 6. Install (sum codes, dequant+sum+requant) | ~15 ms |
| 7. Answer + decode | ~8 s |
| **Total** | **~8.06 s** |

---

## 10. VRAM (A10G, 24 GiB)

| Component | Size |
|---|---|
| Model weights (W4+r32) | 5.85 GiB |
| System prompt cache (TQ codes) | ~6 MiB |
| Query cache (TQ codes, during query) | ~6 MiB |
| Installed retrieved codes (3 chunks) | ~18 MiB |
| IVFADC index (CPU RAM, mmap'd) | — |
| Framework + activations + safety | ~7 GiB |
| **Total VRAM** | **~13 GiB** (fits in 24 GiB) |

---

## 11. Disk layout

```
disk/
├── ivfadc_cache.index              # FAISS IVFADC on dequantized cache vectors
├── snapshots/                       # TurboQuant-compressed per-chunk caches
│   ├── chunk_00000.npz              # ~6 MiB: 24 S codes + 24 conv codes + M1 + M2 codes
│   └── ...                          # 50,000 × 6 MiB = ~300 GiB
└── pretrained_luts/                # the fine-tuned LUTs (~5.85 GiB)
```

**Total disk: ~306 GiB** for 50k chunks.

---

## 12. The toy validation

`scripts/poc_toy/toy_cache_as_vector.py` validated the cache-as-vector retrieval on a CPU model:
- **100% retrieval** (IVFADC on cache vectors found the correct topic)
- **90% correct > wrong** (installing the correct cache made the topic-marker logit 1.45 higher)
- **Lossless composability** (conv-reset, diff = 0.0)

---

## 13. The repo files used

| File | Function | Role |
|---|---|---|
| `scripts/loader.py` | `load_quant_model(...)` | Loads the W4+r32 LUT model |
| `scripts/modeling.py` | `Qwen3_5GatedDeltaNet` | The linear-attn layer (produces S) |
| `scripts/modeling.py` | `cache_params.layers[L].recurrent_states[0]` | Access S (monkey-patched for TQ) |
| `scripts/modeling.py` | `cache_params.layers[L].conv_states[0]` | Access conv_state (monkey-patched) |
| `scripts/modeling.py` | `_fla_resolve()` | The fla wiring for decode |
| `scripts/palettized_modules.py` | `PalettizedLinear.forward(x)` | The W4 forward kernel |
| `flute_extended/src/kernel_fht.cu` | `fht_forward_kernel` | The FHT (TurboQuant's rotation) |
| `transformers` | `DynamicCache(config=model.config)` | The cache object (monkey-patched) |
