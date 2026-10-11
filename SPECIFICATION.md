# Cache-Engineered RAG — Specification

Purpose: cache-engineered RAG on Qwen3.5-9B — the LUT model's cache IS the retrieval vector; NO separate embedder, NO chunk text on disk, NO re-prefill; TurboQuant online at 3.5 bits (quality-neutral).
Authority: top of the doc chain SPECIFICATION.md > PROPOSAL.md > TASKS.md; design decisions D1–D5 live in PROPOSAL.md, execution waves in TASKS.md.
Status: v1 semantics preserved (format rewritten Wv2-6.1); implemented under `src/rag/` — 179 tests green; §9–§10 are A10G targets to measure on the GPU box.

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
- **N7.1 (W21).** The write's slot ADDRESS is the key's CONTENT, not its position: slot = simhash of the per-(batch, head) mean-CENTERED key over a fixed Sylvester-Hadamard mixing (`addr_blocks` independent sign-blocks). The v1 rule (slot = p mod mem_size) is retained verbatim behind `write_addr="positional"` as the A/B. Rationale: a position-addressed state compares by position-aligned key correlation (surface overlap); a content-addressed state compares by content overlap — the §4/N17 retrieval contract. The per-(b,h) centering removes the keys' common direction (the representation-anisotropy pedestal; the box measured ~+0.36 cosine between random cross pairs) inside the write, identically at train and serve. A T=1 (decode) call centers to zero and writes nothing — the memory freezes at prefill end.
- **N7.2 (W21).** The write gates are per-(linear layer, head) — `(num_linear_layers, num_heads)` — zero-init (P3 unchanged); read_gate stays per-layer. Gates .npz artifacts carry the shapes and the addressing regime; the loader refuses pre-W21 (L,)-shaped and cross-regime artifacts loudly.
- **N7.3 (W21.6).** The content write is DETERMINISTIC and META-SAFE by construction: the per-slot token sums are formed by a fixed-order stable-sort segment sum (no atomics on the value path — `torch.index_add`'s CUDA atomic accumulation order is run-to-run nondeterministic once >= 3 colliding rows land in one slot, which the kappa-block collisions of the content mode guarantee; the v1 positional mode's <= 2 colliders commute, which is why v1 could ride it), and the fixed Sylvester mixing master — a plain attribute, never a buffer — is rebuilt from the recursion when born under `from_pretrained`'s meta-device context (a plain attribute is invisible to the missing-key re-init). Contracts restored: write == write_deltas bit-exact, repeated replays of the same calls bit-identical, the capture/replay pair shares the W18 exact-gradient scatter.
- **N7.4 (W22).** The read is OPTIONALLY SPARSE: `read_topk` (None = the §2.2 dense softmax over all mem slots, bit-identical to every pre-W22 read — the DEFAULT; an int k in [1, mem_size] softmaxes the TOP-k score row only, unselected slots at exactly zero probability). Rationale: every reference key-value memory retrieves sparsely (Product-Key Memory top-32, Lample et al. 2019; Memorizing Transformers exact kNN, Wu et al. 2022; RETRO top-k chunks, Borgeaud et al. 2022) — a dense softmax over an accumulating memory averages toward the state mean. The §7 fine-tune never exercises the read (it scores WRITTEN states), so gates artifacts are regime-valid at any k; the value of k is a deployment-time retrieval-quality A/B (drivers expose `--m1m2-read-topk`, the same k at ingest and query), measured by the retrieval evals. P3 holds at any k (reads of zero memories are exact zeros).
- **N7.5 (W22).** The §7 InfoNCE may carry CURATED hard-negative columns: `m1_n`/`m2_n` (label-0 non-pair side-states) append N columns to the logits that are never targets — each anchor must rank its own pair above the curated non-pairs too (DPR, Karpukhin et al. 2020; MoCo's shared queue, He et al. 2020). `neg_pairs=0` (the library default) is the exact W21 loss bit-identically; the finetune driver's default run passes 8, drawn from the label-0 rows EXCLUDING the held-out control slice. The gradient reaches the negatives' own gate rows (push-away through the same replay path).

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
- **N16.1.** Conv windows with a non-power-of-two flattened size (the Qwen3.5 in_proj Q+V geometry: 1×6144×4 = 24,576) are quantized as the FULL window in ONE code unit at d = the flat size rounded up to a multiple of 32 — 24,576 stays 24,576 (round-up pad = 0). The segmented FHT (descending power-of-two blocks: 24,576 = 16,384 + 8,192) transforms the unit block-diagonally; every segment fits the CUDA kernel's 64 KiB shared-memory tile, and the unit rides the conv kind's D3 rotation (seed 202, custom-d instance). NO truncation (the W10-GPU session's 16,384 cap zeroed the last 8,192 coordinates of every read — the ingestion deltas and installed windows were built on a one-third-blinded conv state) and NO zero-pad to 32,768 (the W10 policy's single 32,768-segment FHT needs 128 KiB shared memory — beyond the ~99 KiB consumer-device opt-in; K ≥ 32,768 runs the torch reference on-device instead, `fht._kernel_eligible`). Power-of-two/multiple-of-32 geometries are bit-identical no-ops (pad = 0).

### 3.3 Compression
| object | fp16 | TurboQuant 3.5-bit | ratio |
|---|---|---|---|
| S (24 layers) | 24 MiB | 5.4 MiB | 4.6× |
| conv_state (24 layers, 24,576 elts → full-window unit per N16.1) | 1.125 MiB | 0.25 MiB | 4.6× |
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
| 24 × conv_state codes (3.5-bit, 24,576 dims each) | 0.25 MiB |
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
- **N25.1 (W23).** The ANSWER CONTEXT has TWO modes (drivers' `--answer-mode`): **install** (the §6 steps 5-6 contract above, unchanged and bit-identical) and **prefill** (the driver DEFAULT: steps 5-6 are replaced by prefilling the retrieved chunks' TOKENS through a fresh reseeded cache, ascending chunk id, then the step-7 full-attn reset). Basis, measured not argued: a gated-delta-rule recurrent state is a SEQUENCE state (S_t = (I − β k kᵀ)·S_{t−1} + β k vᵀ, per-head decay) — the N22/N23 sum is exact arithmetic on the codes and bit-clean at n = 1 (the W11/W13 CPU adjudications: install-vs-true 0.048, contractive — the n = 1 verbatim install in N16's absolute protocol is BIT-EXACT reuse), but at n ≥ 2 it SUPERPOSES unrelated chunks' associations (an O(1) state change the delta rule's contractivity bound does not cover — that bound is local), and no published method composes delta-rule states without replaying tokens (the CacheBlend/Block-Attention results are KV-cache results; their "recompute a fraction" translates to "re-prefill the text" here). The prefill mode is the proven quality floor (the W14/W15 true-doc control, conf 0.895); the retrieval surface is UNCHANGED in both modes (steps 2-4 run on the captured codes — the codes are the index, no embedding model). The answer-time memory read (N7/N7.4) is orthogonal and A/B-able per W23 (`enabled`, `--no-answer-m1m2`).

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
| `src/rag/` | `turboquant`, `codebooks`, `tq_cache`, `m1m2`, `hooks`, `ingest`, `snapshot`, `install`, `query`, `finetune`, `m1m2_finetune`, `lut_export`, `index`, `evals` (+ `codebooks/`, `tests/`) | the §1–§8 implementation; per-section clauses above name each entry point; 237 tests under `src/rag/tests/` |
| `src/flute_extended/src/` | `kernel_gemv*.cu`, `kernel_streaming.cu`, `kernel_fht.cu`, `kernel_debug_simple.cu`, `kernel_cutlass_dense.cu` + `include/flute/` headers | the idxN inference CUDA kernels |

## 14. W20 implementation status (addendum — doc-only, authority-preserving)

Appended 2026-10-11 at the W20 close (the TQ kernel family wave,
PROPOSAL.md rev 2). Doc-only: every clause above (N1–N34) stands as
written; nothing here amends a contract — this section records what
the W20 waves placed under them. TASKS.md remains the live execution
ledger (§0 the wave ledger, §4 the gate registry, §5 the attestation
ledger); PROPOSAL.md remains the kernel design layer.

- **The kernel-first runtime is implemented** (§3.2's online
  semantics, N14–N16, unchanged): `src/rag/turboquant.py`'s
  dispatcher routes CUDA + kernel-eligible geometries to the CUDA
  kernels, ALWAYS; CPU, absent-extension, or ineligible geometries
  take the torch twin — the differential reference AND the CPU path
  (the performance fallback, never a correctness path; mixed-device
  inputs raise by name). The W15 outlier split and the 24,576 conv
  window (N16.1) ride the twin.
- **The TQ kernel family is authored**
  (`src/flute_extended/src/kernel_tq.cu` +
  `include/flute/fht_large.cuh`, built with FLUTE_CUDA_ARCHES=86):
  `fht_forward_large` / `fht_backward_large` (the multi-block
  butterfly for K = 2^19…2^22, zero atomics), `tq_quant_batched` (the
  fused encode — norm, butterfly, boundary ladder, pack: one host
  call, zero device→host round trips), `tq_dequant_batched` (the
  fused decode), `tq_qjl_quant` / `tq_qjl_compensate` (the Alg.-2
  chain, flag-off) and `tq_l2norm_rows` (the serving-side row
  normalize). Deterministic by construction: fixed tile/warp/lane
  ownership, fixed-order folds.
- **The two foreign edges hold, restated**: numpy in the runtime
  exists at exactly the npz disk edge (`TQCodes.to_arrays` /
  `from_arrays`, the snapshot save/load) and the faiss boundary (the
  final `.numpy()` at index add/search); the codebooks solve island
  (`src/rag/codebooks.py`) is offline, cached to disk, never runtime.
  Evidence: `env/w20_cpu_elimination_audit.md` — 112 audited
  touchpoints, 0 outside that allowlist.
- **The gates are AUTHORED, all PENDING the box**: 9/9 acceptance
  gates (G20.P1–P3, G20.1–G20.6) live in `scripts/gpu/w20_gates.py`
  with thresholds mirrored 1:1 from TASKS.md §4; a gate is GREEN or
  RED ONLY from `env/w20_gate_reports/<id>.json` produced on the A10G
  box (the five-step execution sequence: `env/W20_GPU_PROTOCOL.md`).
  Parity + determinism are PROPOSAL §5's three levels: run-to-run
  `torch.equal` (hard, by construction); cross-backend codes equal
  except ties ≤ 0.01%, norm ≤ 1 ulp, reconstruction allclose ≤ 1e-5;
  npz snapshots decode cross-backend allclose ≤ 1e-5, the W16
  verbatim install bit-exact WITHIN a backend.
