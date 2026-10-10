# Cache-Engineered RAG — Specification

Purpose: cache-engineered RAG on Qwen3.5-9B — the LUT model's cache IS the retrieval vector; NO separate embedder, NO chunk text on disk, NO re-prefill; TurboQuant online at 3.5 bits (quality-neutral).
Authority: top of the doc chain SPECIFICATION.md > PROPOSAL.md > TASKS.md; design decisions D1–D5 live in PROPOSAL.md, execution waves in TASKS.md.
Status: v1 semantics preserved (format rewritten Wv2-6.1); implemented under `src/rag/` — 169 tests green; §9–§10 are A10G targets to measure on the GPU box.

## 1. The model
- **N1.** Load Qwen3.5-9B (FLUTE idxN hybrid palettization) via `src/scripts/loader.py::load_quant_model`.
- **N2.** Run 32 layers: 24 linear-attention (`Qwen3_5GatedDeltaNet`) + 8 full-attention (`Qwen3_5Attention`) (`src/scripts/modeling.py`); config source `src/docs/qwen3_5_9b_config.json`.
- **N3.** Capture S at 9 hooks (`src/rag/hooks.py::hook_map`) — hook 0 after layer 0, hooks 1–8 after the full-attention layers 3, 7, …, 31; the 9 hooks capture all 24 per-layer S tensors: 1 + 2 + 7×3 = 24.

| hook | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| after layer | 0 | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
| captures | S_0 | S_1, S_2 | S_4, S_5, S_6 | S_8, S_9, S_10 | S_12, S_13, S_14 | S_16, S_17, S_18 | S_20, S_21, S_22 | S_24, S_25, S_26 | S_28, S_29, S_30 |

| key | `hidden_size` | `num_hidden_layers` | `layer_types` | `linear_num_value_heads` | `linear_num_key_heads` | `linear_key_head_dim` | `linear_value_head_dim` | `linear_conv_kernel_dim` | `vocab_size` |
|---|---|---|---|---|---|---|---|---|---|
| value | 4096 | 32 | 24×`linear_attention` + 8×`full_attention`, `[L,L,L,F]`×8 | 32 | 16 | 128 | 128 | 4 | 248320 |

## 2. The caches
| object | location | shape | size | total |
|---|---|---|---|---|
| S (×24, one per linear layer) | `cache_params.layers[L].recurrent_states[0]` | (1, 32, 128, 128) | 1.0 MiB fp16 | 24 MiB |
| conv_state (×24, one per linear layer) | `cache_params.layers[L].conv_states[0]` | (1, 8192, 4) | 64 KiB fp16 | 1.5 MiB |
| M1 (global key-memory, ×1) | model-global, `src/rag/m1m2.py::M1M2` | (1, 32, mem_size, 128) | 1.0 MiB @ mem_size=128 | 1.0 MiB |
| M2 (global value-memory, ×1) | model-global, `src/rag/m1m2.py::M1M2` | (1, 32, mem_size, 128) | 1.0 MiB @ mem_size=128 | 1.0 MiB |

### 2.1 S — the per-layer recurrent state (the model's natural state)
- **N4.** S is the per-layer delta-rule state matrix of `Qwen3_5GatedDeltaNet`, stored at `cache_params.layers[layer_idx].recurrent_states[0]`.
- **N5.** S is composable: with conv-reset at chunk boundaries, each chunk's delta_S is path-independent and summing deltas is lossless (toy-validated, diff = 0.0).

### 2.2 M1, M2 — the two global caches (ADDED, not in the original model)
- **N6.** M1/M2 are global memory matrices shared across ALL 24 linear layers — TWO for the whole model, NOT per-layer; they extend geometrical expressivity and capacity.
- **N7.** The model writes to them (gated, additive) and reads from them (`softmax(q @ M1^T) @ M2`). The additive write makes delta_M1/M2 path-independent; summing is lossless.

### 2.3 conv_state — per-layer
- **N8.** Each linear layer has its own conv_state at `cache_params.layers[layer_idx].conv_states[0]`: (1, 8192, 4), 64 KiB fp16, 1.5 MiB over 24 layers.

### 2.4 Full-attention layers — NOT snapshotted
- **N9.** The 8 full-attention layers run fresh — no KV snapshot, no special handling; they keep their state after the user query is prefilled, then converge by the next forward passes.

## 3. Online TurboQuant (3.5-bit, quality-neutral)
### 3.1 The method (arXiv:2504.19874)
- **N10.** TurboQuant is a data-oblivious, online vector quantizer with provably near-optimal distortion — within 2.7× of the information-theoretic lower bound.
- **N11.** Random rotation: multiply the vector by a random rotation matrix (Hadamard/FHT — `src/flute_extended/src/kernel_fht.cu`); every coordinate then follows the same Beta distribution.
- **N12.** Optimal scalar quantization: per-coordinate Lloyd-Max quantizer on the Beta distribution; codebooks precomputed once per bit-width (`src/rag/codebooks.py`), no calibration.
- **N13.** At 3.5 bits the paper proved absolute quality neutrality for KV cache — identical to fp16 on LongBench-E and Needle-In-A-Haystack.

### 3.2 Online integration (NOT a sidechain)
- **N14.** The cache states (S, M1, M2, conv_state) are ALWAYS TurboQuant codes — during the model's forward pass, on disk, at installation.
- **N15.** WRITE (state update in the forward pass): the delta rule produces fp16 S → `TurboQuant.quant` → codes stored in the cache; the cache NEVER holds fp16. READ (state consumption in the next forward step): `TurboQuant.dequant` → fp16 for the delta rule → re-quantized on write.
- **N16.** The cache API intercepts reads (`cache.layers[L].recurrent_states[0]` → dequantize) and writes (`cache.update_recurrent_state(state, L)` → quantize); the model's forward doesn't know it uses quantized caches. Implementation: `src/rag/tq_cache.py::TQCache` (a `DynamicCache` subclass) + `src/rag/turboquant.py::TurboQuant`.
- **N16.1.** Conv windows with a non-power-of-two flattened size (the Qwen3.5 in_proj Q+V geometry: 1×6144×4 = 24,576) are zero-padded to the next power of two before the FHT — 24,576 → 32,768 = the canonical conv d, so the D3 rotation (seed 202) stays shared across geometries; dequant strips the pad. Padding adds no energy: the stored norm is unchanged and the rel-MSE budget on the real 24,576 coordinates holds (measured 0.016 vs 0.022 for the unpadded random unit). Power-of-two geometries keep the exact pre-padding behavior (pad = 0).

### 3.3 Compression
| object | fp16 | TurboQuant 3.5-bit | ratio |
|---|---|---|---|
| S (24 layers) | 24 MiB | 5.4 MiB | 4.6× |
| conv_state (24 layers, 24,576 elts → padded per N16.1) | 1.125 MiB | 0.34 MiB | 3.3× |
| M1 (global) | 1.0 MiB | 0.22 MiB | 4.6× |
| M2 (global) | 1.0 MiB | 0.22 MiB | 4.6× |
| per chunk | 27.1 MiB | ~6.0 MiB | 4.4× |
| 50k chunks | 1.29 TiB | ~300 GiB | 4.4× |

## 4. The retrieval vector
```
cache_vector = concat([S_0.flatten(), S_1.flatten(), S_2.flatten(), S_4.flatten(), ..., S_30.flatten(),
                       M1.flatten(), M2.flatten()])
# 24 S tensors (all linear layers, §1 order) + M1 + M2; dims 24 × 524,288 + 2 × 524,288 = 26 × 524,288 = 13,631,488
# size 26.0 MiB (fp16) — dequantized from TurboQuant codes for IVFADC
```
- **N17.** The query's cache vector and the chunk's cache vector are in the SAME space — both are linear-attention state matrices from the same model; cos-sim measures content overlap = relevance. NO hidden states, NO separate embedder.

## 5. Ingestion (one-time, per chunk)
- **N18.** Ingest a chunk once via `src/rag/ingest.py::ingest_chunk(model, chunk_token_ids, cache, system)`: prefill under `torch.no_grad()` with `past_key_values=tq_cache, use_cache=True` — every state update writes TurboQuant codes (online, §3.2).
- **N19.** Snapshot the codes from the cache — 24 S codes + 24 conv codes + M1 + M2 — as one npz per chunk at `snapshots/chunk_XXXXX.npz` (`src/rag/snapshot.py::save_chunk`); driver: `src/rag/ingest.py::IngestDriver` (system-prompt prefill → per-chunk deltas → save; resumable).
- **N20.** Store DELTA codes for S/M1/M2 relative to the system-prompt state and ABSOLUTE codes for conv_state (§6: conv is never summed, last retrieved chunk wins).
- **N21.** Store ONLY TurboQuant codes on disk — NO chunk text, NO token IDs, NO fp16; the retrieval vector (dequantized codes, §4 order, fp32) is computed on demand for IVFADC index building and never stored.

Per-chunk disk (TurboQuant codes only):

| component | size |
|---|---|
| 24 × S codes (3.5-bit, 524,288 dims each) | 5.4 MiB |
| 24 × conv_state codes (3.5-bit, 32,768 dims each) | 0.34 MiB |
| M1 codes (3.5-bit, 524,288 dims) | 0.22 MiB |
| M2 codes (3.5-bit, 524,288 dims) | 0.22 MiB |
| total per chunk | ~6.0 MiB |
| 50k chunks | ~300 GiB |

## 6. The query flow
| step | action |
|---|---|
| 1 | Tokenize the query. |
| 2 | Prefill the query through the model (online TurboQuant cache) → the query's S (24) + M1 + M2 are captured as TurboQuant codes → dequantize for IVFADC → the query's cache_vector. |
| 3 | IVFADC preselect on cache vectors → top-100 candidates. |
| 4 | Cos-sim rerank → top-3 chunk indices. |
| 5 | Load the top-3 chunks' TurboQuant codes from disk (~18 MiB total). |
| 6 | Install: sum the S deltas (24 per-layer) + sum the M1/M2 deltas (2 global) — the sums are on the TurboQuant CODES (not dequantized) — and install the summed codes into the cache. |
| 7 | Answer from the installed caches: the model's forward dequantizes on read (online TurboQuant); the full-attn layers run fresh. |
| 8 | Decode the answer. |
- **N22.** Installation, per linear layer: sum the retrieved chunks' S code deltas onto the system codes (`sum_turboquant_codes`) and install into the cache; conv_state takes the LAST retrieved chunk's codes (never summed). Sum the M1/M2 code deltas the same way (2 global); the full-attention layers get NO installation.
- **N23.** Install formula (the contract — dequant, sum, requant):

```
summed_codes = TurboQuant.quant( Σ_i TurboQuant.dequant(codes_i) )
```
- **N24.** Summing codes is dequant + sum + re-quant over 26 tensors of ~500K dims — a fast operation. Summing in the rotated space (rotation is linear, quantizer is per-coordinate) is a possible optimization, not the contract.
- **N25.** Implementations: `src/rag/install.py::sum_turboquant_codes / install_snapshot / install_from_disk`; `src/rag/query.py::answer_query` (the full §6 flow, preselect_k=100, rerank_k=3).

## 7. Pretraining (fine-tune)
- **N26.** Train by next-token prediction on the OfficeQA corpus via the LUT path, on the idxN hybrid model loaded by `src/scripts/loader.py::load_quant_model`.
- **N27.** Use a lean training loop written as part of the RAG build on the GPU box — the parent project's QLoRA/distillation trainer is NOT part of this repo. Straight-through LUTs via `PalettizedLinear.make_trainable()` on the reference path (`forward="reference"`).
- **N28.** Train the 24 per-layer linear-attention params (so S is discriminative) + the M1/M2 read/write gates (so the global caches carry info) + the LUTs (W10); ~500 steps on A10G. The real model is already trained by Qwen — this is a fine-tune.
- **N29.** Serve the fine-tuned LUTs as full LUT artifacts (`pretrained_luts/`, §11) — not as QLoRA adapters. Implementations: `src/rag/finetune.py::train / build_trainables / freeze_all_luts`; `src/rag/lut_export.py::export_luts / import_luts`.

## 8. IVFADC
- **N30.** Build the index on the dequantized cache vectors (fp32, 13.6M dims) with FAISS:

```
quantizer = faiss.IndexFlatIP(13631488)
index = faiss.IndexIVFPQ(quantizer, 13631488, nlist=224, m=64, nbits=8)
index.train(cache_vectors_fp32); index.add(cache_vectors_fp32)
```
- **N31.** Probe nprobe=8 clusters per query; the exact vectors for the rerank are the dequantized cache vectors, loaded on demand from the TurboQuant codes. Indexing time ~0 — TurboQuant is data-oblivious (no k-means training, unlike PQ).
- **N32.** Implementation: `src/rag/index.py::build_index / load_index / preselect / rerank` (`IndexConfig`: d=13,631,488, nlist=224, m=64, nbits=8, nprobe=8, preselect_k=100, rerank_k=3).

## 9. Timing (per query, A10G)
| step | 1. Tokenize | 2. Prefill query + snapshot cache (online TQ) | 3. IVFADC preselect | 4. Cos sim rerank | 5. Load TurboQuant codes (3 chunks × 6 MiB) | 6. Install (sum codes, dequant+sum+requant) | 7. Answer + decode | total |
|---|---|---|---|---|---|---|---|---|
| time | <1 ms | ~8 ms | ~10 ms | ~20 ms | ~3 ms | ~15 ms | ~8 s | ~8.06 s |

## 10. VRAM (A10G, 24 GiB)
| component | size |
|---|---|
| Model weights (idxN hybrid) | 6.35 GiB |
| System prompt cache (TQ codes) | ~6 MiB |
| Query cache (TQ codes, during query) | ~6 MiB |
| Installed retrieved codes (3 chunks) | ~18 MiB |
| IVFADC index (CPU RAM, mmap'd) | — |
| Framework + activations + safety | ~7 GiB |
| total VRAM | ~13 GiB (fits in 24 GiB) |

## 11. Disk layout
```
disk/
├── ivfadc_cache.index              # FAISS IVFADC on dequantized cache vectors
├── snapshots/                       # TurboQuant-compressed per-chunk caches
│   └── chunk_00000.npz ...          # ~6 MiB each: 24 S + 24 conv + M1 + M2 codes; 50,000 × 6 MiB = ~300 GiB
└── pretrained_luts/                 # the fine-tuned LUTs (~5.85 GiB)
```
- **N33.** Total disk: ~306 GiB for 50k chunks.

## 12. The toy validation
- **N34.** `scripts/poc_toy/toy_cache_as_vector.py` validated the cache-as-vector retrieval on a CPU model:

| result | value |
|---|---|
| retrieval — IVFADC on cache vectors found the correct topic | 100% |
| correct > wrong — installing the correct cache made the topic-marker logit 1.45 higher | 90% |
| lossless composability (conv-reset) | diff = 0.0 |

## 13. The repo files used
| path | symbol | role |
|---|---|---|
| `src/scripts/loader.py` | `load_quant_model(...)` | loads the idxN hybrid LUT model |
| `src/scripts/modeling.py` | `Qwen3_5GatedDeltaNet` | the linear-attn layer (produces S) |
| `src/scripts/modeling.py` | `cache_params.layers[L].recurrent_states[0]` / `.conv_states[0]` | access S / conv_state (intercepted for TQ) |
| `src/scripts/modeling.py` | `_fla_resolve()` | the fla wiring for decode |
| `src/scripts/palettized_modules.py` | `PalettizedLinear.forward(x)` | the idxN LUT forward |
| `src/flute_extended/src/kernel_fht.cu` | `fht_forward_kernel` | the FHT (TurboQuant's rotation) |
| `transformers` | `DynamicCache(config=model.config)` | the cache object (TQ-intercepted) |
| `src/rag/` | `turboquant`, `codebooks`, `tq_cache`, `m1m2`, `hooks`, `ingest`, `snapshot`, `install`, `query`, `finetune`, `lut_export`, `index`, `evals` (+ `codebooks/`, `tests/`) | the §1–§8 implementation; per-section clauses above name each entry point; 169 tests under `src/rag/tests/` |
| `src/flute_extended/src/` | `kernel_gemv*.cu`, `kernel_streaming.cu`, `kernel_fht.cu`, `kernel_debug_simple.cu`, `kernel_cutlass_dense.cu` + `include/flute/` headers | the idxN inference CUDA kernels |
