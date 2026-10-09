# SPECIFICATION v3 — Cache-Engineered RAG (corrected)

> **The architecture, stated simply:**
> - The model is Qwen3.5-9B (FLUTE idxN W4+r32). 32 layers: 24 linear-attn + 8 full-attn, pattern `[L,L,L,F,L,L,L,F,...]`.
> - We ADD **TWO global caches** (M1 key-memory + M2 value-memory) to the model. These are SHARED across ALL 24 linear layers — there are only TWO, for the whole model. Not per-layer.
> - We snapshot the **linear-attn recurrent state S** at **9 boundaries** only: after layer 0 (the first linear layer), and after each of the 8 full-attn layers (layers 3, 7, 11, 15, 19, 23, 27, 31). That's 9 snapshots of S per chunk — NOT 24.
> - The full-attn layers run fresh. NO KV snapshot. Ever.
> - The retrieval vector = the flattened S (from the 9 boundaries) + M1 + M2 (the two global caches). Flattened into one vector for IVFADC.
> - Augmentation = sum the S deltas (at the 9 boundaries) + restore M1, M2 (the two global caches). Install into the model. Answer from installed caches.

---

## 1. The model (from `docs/qwen3_5_9b_config.json`)

### 1.1 The layer structure

```
Layer  0: linear_attn  ← BOUNDARY 0 (first layer, snapshot S here)
Layer  1: linear_attn
Layer  2: linear_attn
Layer  3: full_attn    ← BOUNDARY 1 (after full-attn, snapshot S here)
Layer  4: linear_attn
Layer  5: linear_attn
Layer  6: linear_attn
Layer  7: full_attn    ← BOUNDARY 2
Layer  8: linear_attn
Layer  9: linear_attn
Layer 10: linear_attn
Layer 11: full_attn    ← BOUNDARY 3
Layer 12: linear_attn
Layer 13: linear_attn
Layer 14: linear_attn
Layer 15: full_attn    ← BOUNDARY 4
Layer 16: linear_attn
Layer 17: linear_attn
Layer 18: linear_attn
Layer 19: full_attn    ← BOUNDARY 5
Layer 20: linear_attn
Layer 21: linear_attn
Layer 22: linear_attn
Layer 23: full_attn    ← BOUNDARY 6
Layer 24: linear_attn
Layer 25: linear_attn
Layer 26: linear_attn
Layer 27: full_attn    ← BOUNDARY 7
Layer 28: linear_attn
Layer 29: linear_attn
Layer 30: linear_attn
Layer 31: full_attn    ← BOUNDARY 8
```

**9 boundaries:** after layer 0, and after each full-attn layer (3, 7, 11, 15, 19, 23, 27, 31).

**Why these boundaries:** the full-attn layers "reset" the attention context. The linear-attn state S at these points captures the accumulated memory of the 3 preceding linear layers (or just 1 for boundary 0). These are the natural snapshot points — the state is "complete" for that block.

### 1.2 The cache objects

**Per boundary (9 total):** the recurrent state S from the linear-attn layer at that boundary.

| Object | Shape | Size (fp16) |
|---|---|---|
| S at boundary b | `(1, 32, 128, 128)` | 1.0 MiB |
| conv_state at boundary b | `(1, 8192, 4)` | 64 KiB |
| **Per boundary total** | | **~1.06 MiB** |
| **× 9 boundaries** | | **~9.5 MiB** |

**The two global caches (2 total, for the WHOLE model):**

| Object | Shape | Size (fp16) |
|---|---|---|
| M1 (global key-memory) | `(1, 32, mem_size, 128)` | depends on mem_size |
| M2 (global value-memory) | `(1, 32, mem_size, 128)` | depends on mem_size |

At mem_size=128: each is 1.0 MiB. **Two global caches: 2.0 MiB total.**

**The full-attn layers:** run fresh. No snapshot. No KV. Nothing stored.

---

## 2. The retrieval vector

```
cache_vector = concat([
    S_boundary_0.flatten(),    # (32, 128, 128) → 524,288
    S_boundary_1.flatten(),    # after layer 3
    S_boundary_2.flatten(),    # after layer 7
    S_boundary_3.flatten(),    # after layer 11
    S_boundary_4.flatten(),    # after layer 15
    S_boundary_5.flatten(),    # after layer 19
    S_boundary_6.flatten(),    # after layer 23
    S_boundary_7.flatten(),    # after layer 27
    S_boundary_8.flatten(),    # after layer 31
    M1.flatten(),              # the global key-memory
    M2.flatten(),              # the global value-memory
])
# at mem_size=128:
# 9 × 524,288 + 2 × 524,288 = 11 × 524,288 = 5,767,168 dims
# size: 11.0 MiB (fp16)
```

**NOT the full-attn KV. NOT the hidden state. NOT per-layer M1/M2.** The 9 boundary S snapshots + the 2 global caches (M1, M2), flattened into one ~11 MiB vector.

---

## 3. Ingestion (one-time, per chunk)

```python
# poc/ingest.py
def ingest_chunk(model, chunk_token_ids, device):
    """Prefill a chunk. Snapshot S at 9 boundaries + M1, M2 (global).
    NO full-attn KV."""
    # the 9 boundary layer indices
    BOUNDARIES = [0, 3, 7, 11, 15, 19, 23, 27, 31]

    # create a fresh cache (conv-reset)
    cache = DynamicCache(config=model.config)

    # prefill
    with torch.no_grad():
        outputs = model(input_ids=chunk_token_ids, past_key_values=cache, use_cache=True)

    # snapshot S at the 9 boundaries
    s_snapshots = []
    conv_snapshots = []
    for b in BOUNDARIES:
        S = cache.layers[b].recurrent_states[0]  # (1, 32, 128, 128)
        conv = cache.layers[b].conv_states[0]     # (1, 8192, 4)
        s_snapshots.append(S.detach().cpu())
        conv_snapshots.append(conv.detach().cpu())

    # snapshot M1, M2 (the two global caches — shared across all layers)
    M1 = model.global_M1  # (1, 32, mem_size, 128) — the global key-memory
    M2 = model.global_M2  # (1, 32, mem_size, 128) — the global value-memory

    # the cache vector = flattened S (9 boundaries) + M1 + M2
    cache_vector = torch.cat(
        [s.flatten() for s in s_snapshots] + [M1.flatten(), M2.flatten()]
    ).numpy().astype(np.float16)

    return {
        's_snapshots': s_snapshots,       # 9 × (1, 32, 128, 128) — Cache S
        'conv_snapshots': conv_snapshots, # 9 × (1, 8192, 4)
        'M1': M1.detach().cpu(),          # (1, 32, mem_size, 128) — Global Cache M1
        'M2': M2.detach().cpu(),          # (1, 32, mem_size, 128) — Global Cache M2
        'cache_vector': cache_vector,     # (5767168,) — the retrieval vector
    }
```

### What's saved per chunk

| Component | Count | Size each | Total (fp16) |
|---|---|---|---|
| S snapshots (at 9 boundaries) | 9 | 1.0 MiB | 9.0 MiB |
| conv_state snapshots | 9 | 64 KiB | 0.56 MiB |
| M1 (global) | 1 | 1.0 MiB (mem_size=128) | 1.0 MiB |
| M2 (global) | 1 | 1.0 MiB (mem_size=128) | 1.0 MiB |
| cache_vector (redundant) | 1 | 11.0 MiB | 11.0 MiB |
| **Total per chunk** | | | **~12.5 MiB** (without redundant vector: ~11.5 MiB) |

**For 50,000 chunks:** ~625 GiB on disk. Manageable.

---

## 4. The query flow

```
[1. Tokenize the query]
[2. Prefill the query → snapshot the query's S (9 boundaries) + M1 + M2]
[3. IVFADC preselect on the cache vector → top-100 candidates]
[4. Cos sim rerank → top-3 chunk indices]
[5. Load the top-3 chunks' snapshots from disk]
[6. Install: sum the S deltas (9 boundaries) + restore M1, M2 (2 global caches)]
[7. Answer from the installed caches — full-attn runs fresh]
[8. Decode the answer]
```

### The installation

```python
# for each of the 9 boundaries
for b_idx, boundary in enumerate(BOUNDARIES):
    restored_S = system_S[b_idx] + sum(snap['s_snapshots'][b_idx] for snap in retrieved)
    restored_conv = retrieved[-1]['conv_snapshots'][b_idx]
    cache.layers[boundary].recurrent_states[0] = restored_S
    cache.layers[boundary].conv_states[0] = restored_conv

# restore the 2 global caches (sum the deltas)
model.global_M1 = system_M1 + sum(snap['M1'] - init_M1 for snap in retrieved)
model.global_M2 = system_M2 + sum(snap['M2'] - init_M2 for snap in retrieved)

# the full-attn layers run fresh — NO installation
```

---

## 5. Pretraining

Same as before: next-token prediction on the corpus, W10 LUT path. Trains the linear-attn layers + M1/M2 to be discriminative.

---

## 6. Summary

| Question | Answer |
|---|---|
| Against what will we pretrain? | The OfficeQA corpus (NTP, W10 LUT path). ~500 steps. |
| What's in the vectorDB? | IVFADC on the flattened **S (9 boundaries) + M1 + M2** (~5.8M dims, 11 MiB per chunk). Per-chunk snapshots: 9 S states + conv states + M1 + M2. NO full-attn KV. |
| How does retrieval work? | Snapshot the query's S (9 boundaries) + M1 + M2 → IVFADC → cos sim rerank → top-3. |
| How does augmentation work? | Sum the S deltas (9 boundaries) + restore M1, M2 (2 global caches). Install into the model. Full-attn runs fresh. Answer from installed caches. NO re-prefill. |
| How many global caches? | **TWO.** M1 and M2. For the whole model. Not per-layer. |
| Where do we snapshot S? | At **9 boundaries**: layer 0 (first), and after each full-attn layer (3, 7, 11, 15, 19, 23, 27, 31). |
| Do we snapshot full-attn KV? | **NO.** The full-attn layers run fresh. |
