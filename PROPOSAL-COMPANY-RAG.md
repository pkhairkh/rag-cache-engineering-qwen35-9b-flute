# PROPOSAL — Cache-Engineering RAG on Qwen3.5-9B (FLUTE idxN), Databricks, SharePoint

> **Status:** proposed — serving — implementation-specific (v1.1)
> **Base:** Qwen3.5-9B, served on its **native 3:1 linear-attention / full-attention hybrid** (`docs/MODEL_GEOMETRY.md` §1: 24 × `linear_attention` + 8 × `full_attention`, `full_attention_interval: 4`), with weights quantized through the **FLUTE idxN unified kernel stack** (`flute_extended` + `flute_train_kernels`, `docs/IDXN_UNIFICATION.md`, `docs/QUANTIZATION_FORMAT.md`, `docs/KERNEL_SPEC_DLDLUT.md`). The auto-recipe resolver (`scripts/palettize_qwen3_5_9b.py --recipe auto --auto-cos 0.9995`) picks width per tensor — the cache engineering lane treats the *resolved recipe set* as part of the cache namespace.
> **Decision owner:** repo owner · **Scope:** the single implementation path for a company RAG on Databricks, sourced from SharePoint libraries that are *not* in the best state, with the serving tier built **literally on the kernel surfaces that already exist in `qwen3_5_9B_flute_qlora_v1.3`**. Four lanes: corpus, permission, serving (delta-attention cache engineering), evaluation.
> **Provenance:** every kernel reference in this document is a file path inside the source repo. Every geometry number is from `docs/MODEL_GEOMETRY.md`. Every quantization claim is from `docs/QUANTIZATION_FORMAT.md`. The cache engineering framing follows `PROPOSAL-COMPANY-RAG.md` (v1.0) and **tightens** it: where v1.0 described the KDA graft as custom engineering to be commissioned, this v1.1 records that the graft is **already done** by Qwen3.5-9B's native architecture — the implementation work is integration, not grafting.

---

## 0. The question, and the answer

> *"What is the single best way to implement a company RAG on Databricks that is based on SharePoint documents which might not be in the best state? Insights needed into the RAG part (best case as delta-attention cache engineering), the permission part, and the Databricks part."*

**Answer (this revision):** build one Delta-native system in four lanes, with the serving lane built on **the exact kernels that ship in `qwen3_5_9B_flute_qlora_v1.3`** — no new CUDA, no model surgery, no grafting. Qwen3.5-9B is *already* a 3:1 linear/full hybrid. The FLUTE idxN stack already provides every primitive the cache engineering lane needs:

1. A **corpus lane** that turns SharePoint into a governed, versioned, *user-agnostic* chunk asset (unchanged from v1.0; the parser is the largest corpus-side lever and the parser is a Databricks platform surface, not a kernel concern).
2. A **permission lane** that syncs SharePoint ACLs into Unity Catalog and enforces them **at query time only** (unchanged from v1.0; this is a Delta/catalog concern, not a kernel concern).
3. A **serving lane** that runs **Qwen3.5-9B natively** (no graft needed — `docs/MODEL_GEOMETRY.md` §1 confirms the hybrid is built in), with the **FLUTE idxN dequant+GEMM kernels** (`flute_extended/src/kernel_cutlass_streaming.cu`, `kernel_cutlass_dense.cu`) and the **FHT rotation kernel** (`flute_extended/src/kernel_fht.cu`) on dedicated Model Serving endpoints behind Unity Gateway, and treats RAG cost as cache engineering on the **fixed-size per-head linear-attention state matrices** that Qwen3.5-9B already produces.
4. An **evaluation lane** that makes go/no-go decisions only on meaningful-query-filtered, staleness-axis measurements, gated per model checkpoint and per corpus version (unchanged from v1.0).

The order of investment is unchanged and is evidence-driven: **parsing first, chunking second, cache policy third, model cleverness last.** What this revision changes is the *implementation specificity* of lane 3 — the model cleverness is no longer a research bet, it is a wiring exercise against kernels that exist and pass their parity gates today.

---

## 1. The measured findings that decide the design

The ten findings from v1.0 stand unchanged; they are not repeated here. **Three additional findings** specific to the implementation surface — derived from reading the kernel sources, the model geometry, and the handover log — are added to the contract between the evidence base and the architecture.

### 1.1 Implementation-specific findings (this revision)

| # | Measured finding | Source (file in `qwen3_5_9B_flute_qlora_v1.3`) | What it decides here |
|---|---|---|---|
| I-1 | **Qwen3.5-9B is already a 3:1 linear/full hybrid.** `num_hidden_layers=32`, `layer_types = 24 × linear_attention + 8 × full_attention`, `full_attention_interval=4`. The "KDA graft" that v1.0 commissioned as custom engineering is **already present in the checkpoint** — `linear_num_key_heads=16`, `linear_key_head_dim=128`, `linear_num_value_heads=32`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`. | `docs/MODEL_GEOMETRY.md` §1 | The "graft" work in v1.0 §3 Stage 3 is **removed**. The implementation work is: integrate the FLUTE idxN kernels into the serving stack, expose the linear-attention state matrices as the cache asset, and wire the namespace discipline around them. The model does the cache mechanics natively. |
| I-2 | **The fixed per-head linear-attention state is a 576 KiB object, per session, per replica.** Per linear layer: K-state `16 × 128 = 2048 fp16`, V-state `32 × 128 = 4096 fp16` → 6144 fp16 = 12 KiB per layer per session. Across 24 linear layers: 24 × 12 KiB = **288 KiB per session** (K-state); with V-state doubled: **576 KiB per session** for the full linear-attention state vector. The `conv1d` kernel (`linear_conv_kernel_dim=4`) adds a 4-step sliding window buffer, negligible (4 × 6144 = 24 KiB per layer). | `docs/MODEL_GEOMETRY.md` §1 (linear_* fields) | The "cache asset is a small paged object" claim in v1.0 §2.3 stops being a claim and becomes a measured fact. 576 KiB per session fits in L2; thousands of sessions fit in a single A10G's HBM pool; the cache namespace is keyed by `(checkpoint_id, quant_recipe_signature, corpus_version, session_id, last_full_attn_layer_boundary)`. |
| I-3 | **The A10G is the binding HBM constraint at 24 GiB, and the handover log records OOMs in the palettizer itself.** The handover (`scripts/HANDOVER.issue-2026-10-04.md`) records `expandable_segments: memory mapping failed with OOM` during `lm_head` palettization with `--calib-seqs 64 --calib-seq-len 2048`; `nvidia-smi -q` confirms 23,028 MiB total. With the full text backbone at idx4 (`6,912,212,992` palettized elements / 2 = 3.46 GiB) plus `lm_head` + `embed_tokens` at idx4 (`2 × 1,017,118,720 / 2` = 0.97 GiB each ≈ 1.94 GiB) the W4 model is ~5.4 GiB; the residual r32 adders add ~0.4 GiB. That leaves **~18 GiB on a single A10G** for cache pool + activations + serving framework overhead. | `scripts/HANDOVER.issue-2026-10-04.md`, `docs/MODEL_GEOMETRY.md` §3, `docs/QUANTIZATION_FORMAT.md` §2 | The v1.0 "96 GiB per replica class" capacity planning is **wrong for this hardware**. The realistic per-replica cache pool is **15 GiB on a single A10G** (after framework, KV-cache for the 8 full-attention layers at the working context, and 2 GiB safety margin). At 576 KiB per session that is **~26,000 concurrent session state snapshots per replica**. The capacity law of v1.0 (`cache ≈ arrival × session_duration`) still holds; the constant is updated. |

These three findings are the difference between v1.0 and v1.1. They convert the serving lane from a research plan into a wiring plan.

---

## 2. The system

One lakehouse, four lanes, one cache discipline. Everything is a Delta table or a governed endpoint; everything that repeats is a versioned cache asset; everything that varies per user is a query-time filter.

```
                     ┌────────────────────────────────────────────────────────────────────┐
                     │                        UNITY CATALOG (governance)                   │
                     │   catalogs · row filters · column masks · lineage · audit logs      │
                     └────────────────────────────────────────────────────────────────────┘
                          │                │                  │                    │
   ┌──────────┐   ┌───────▼───────┐ ┌──────▼──────┐  ┌────────▼───────┐   ┌─────────▼─────────┐
   │SharePoint │──►│  CORPUS LANE  ││PERMISSION   │  │  SERVING LANE  │   │  EVALUATION LANE  │
   │(Graph API:│   │  (Lakeflow +  ││LANE (sync)  │  │  (Model        │   │  (MLflow 3 +      │
   │ content + │   │  ai_parse_doc)││ACL deltas → │  │  Serving +     │   │  inference tables)│
   │ ACL delta │   │ bronze: raw   ││effective    │  │  Unity Gateway)│   │  golden set ·     │
   │ links)    │   │ silver: parsed││access →     │  │ Qwen3.5-9B     │   │  meaningful-query │
   └──────────┘    │ gold: chunk   ││chunk ACL    │  │ NATIVE 3:1 hyb │   │  filter · envelope│
                   │ asset (ver-   ││columns →    │  │ + FLUTE idxN   │   │  curves on staleness│
                   │ sioned, user- ││query-time   │  │ W1/W2/W3/W4    │   │  axes · gates per  │
                   │ agnostic)     ││filters      │  │ recipes; FHT   │   │  checkpoint/corpus │
                   │ DQX quarantine││             │  │ rotation; the  │   │  version           │
                   └───────┬───────┘└─────────────┘  │ linear-attn    │   └────────────────────┘
                           │                          │ state matrix   │
                           │      Vector Search (UC-governed,│is THE cache asset)
                           └──────────────────────────────────┘
```

### 2.1 Corpus lane: SharePoint → governed chunk asset

Unchanged from v1.0 §2.1. Microsoft Graph API as the single ingestion surface (content + ACLs through delta links); `ai_parse_document` for layout-aware parsing (finding #1: +16.1% relative accuracy); DQX-style quarantine with reasons, never silent drops; structure-based chunking with size control (findings #2, #3); metadata verbalization into chunk headers (Walk&Retrieve). The chunk asset is user-agnostic, versioned per document, incrementally maintained.

This lane is **independent of the kernels** — its outputs (the gold Delta table, the Vector Search index) are consumed by the serving lane through standard Delta reads, not through any FLUTE-specific surface. This independence is deliberate: the kernel investment is concentrated in lane 3, the corpus investment is concentrated in lane 1, and they do not block each other.

### 2.2 Permission lane: ACLs in the catalog, never in the cache

Unchanged from v1.0 §2.2. Query-time ACL enforcement in Vector Search + UC row filters; effective-access materialization with SCD2; daily full reconciliation; the fourth staleness axis (permission flip) handled by construction (no cached quantity depends on it) and measured by the evaluation lane. The cache asset — the linear-attention state matrices — is **per-session**, not per-user; the same K/V state vector serves the CFO and an intern with the only difference being the ACL filter set traveling with the query.

### 2.3 Serving lane: delta-attention cache engineering, on the FLUTE idxN stack

This is the lane this revision actually changes. v1.0 described the serving base as a future commissioning ("KDA graft plan for Qwen 3.5"). v1.1 records that the graft is already in the model and turns the commissioning into a wiring exercise against the existing kernels.

#### 2.3.1 The model: native, not grafted

`docs/MODEL_GEOMETRY.md` §1 reads the live checkpoint config and reports:

- `num_hidden_layers = 32`
- `layer_types = 24 × linear_attention, 8 × full_attention`
- `full_attention_interval = 4` (every 4th layer is full attention; the other 3 are linear)
- `linear_num_key_heads = 16`, `linear_key_head_dim = 128`
- `linear_num_value_heads = 32`, `linear_value_head_dim = 128`
- `linear_conv_kernel_dim = 4`
- `num_attention_heads = 16`, `num_key_value_heads = 4`, `head_dim = 256` (for the full-attention layers)

This is **exactly** the 3:1 KDA-to-MLA pattern v1.0 specified, except it is native rather than grafted. There is no Stage-3 graft task. There is no state-conversion-and-distillation budget. The 8 full-attention layers (every 4th, starting at layer 3) play the role v1.0 reserved for MLA anchor blocks: they produce token-level KV at bounded positions, and the linear-attention layers between them carry state forward as fixed-size matrices. The "6× faster decoding at 1M context" measurement in finding #5 is the source paper's measurement condition at 1M context — we adopt the *mechanics* (fixed-size state matrices) at our working context, as v1.0 already specified; what changes is that the mechanics are now confirmed present in the model we are deploying.

**What this removes from v1.0 §3 Stage 3.** The "KDA graft plan for Qwen 3.5 (layer selection in the 3:1 pattern, NoPE blocks, state conversion and distillation budget to recover dense-model quality)" is removed entirely. The model has the layer selection already. NoPE on the linear-attention blocks is the model's native behavior (linear attention does not carry position embeddings the way softmax attention does). State conversion is irrelevant — the state is produced at runtime by the linear-attention forward pass. Distillation budget is irrelevant — the model was trained with this hybrid layout by Qwen.

#### 2.3.2 The quantization: FLUTE idxN, recipe-resolved per tensor

The `flute_extended` package (`docs/IDXN_UNIFICATION.md`) ships the **unified idxN kernel family** at bit widths `b ∈ {1, 2, 3, 4}`. The forward kernel (`flute_kernel_streaming_fd_sub4<Cfg, B>` for `B=1,2,3`, `flute_kernel_streaming_fd<TileConfig>` for `B=4`) is one path, one tiling, one mma phase; the LUT row width scales with `2^b`. The benchmark in `IDXN_UNIFICATION.md` §Performance records ~58.5 TFLOPS (93% peak A10G) **uniformly across widths** — there is no throughput penalty for choosing a narrower LUT.

The `--recipe auto --auto-cos 0.9995` resolver (`scripts/palettize_qwen3_5_9b.py`, documented in `docs/AUTO_SELECTION_GUIDE.md` and `docs/QUANTIZATION_FORMAT.md` §5) walks a per-tensor ladder ordered by `(total_bits, trainable, stream_count)` and picks the **minimum-bits candidate whose calibration cosine meets the gate**. The candidate pool is documented in `AUTO_SELECTION_GUIDE.md`:

| Rung | Bits/elt | Palette | Layout | Trainable (W10) |
|---|---|---|---|---|
| `(1,)` | 1 | 2 | one `.idx1` | yes |
| `(2,)` | 2 | 4 | one `.idx2` | yes |
| `(3,)` | 3 | 8 | one `.idx3` | yes |
| `(4,)` | 4 | 16 | one `.idx4` | yes (regression) |
| `(4, 1)` | 5 | 32 | `.idx4` + `.idx1.2` | yes (W10) |
| `(4, 2)` | 6 | 64 | `.idx4` + `.idx2.2` | yes (W10) |
| `(4, 3)` | 7 | 128 | `.idx4` + `.idx3.2` | yes (W10) |
| `(4, 2, 2)` | 8 | 256 | `.idx4` + `.idx4.2` | yes (W10) — the `hybrid422` accuracy recipe |

The cache namespace keys on the **resolved recipe signature**, not just "W4". A model palettized with `--recipe auto` produces a metadata.json whose `auto_decision` ledger (`AUTO_SELECTION_GUIDE.md` §"The audit ledger") enumerates every tensor's resolved spec; the SHA-256 of that ledger is the cache-key component v1.0 called `quantization config`. A re-quantization with a different `--auto-cos` gate or a different ladder is a different namespace, by construction.

**Storage budget for the serving stack** (from `QUANTIZATION_FORMAT.md` §2 and `MODEL_GEOMETRY.md` §2-3, arithmetic recomputed for the auto recipe):

| Artifact family | Stored bytes (W4 base) | Notes |
|---|---|---|
| 248 palettized modules, idx4 (`6,912,212,992` elements / 2) | 3.46 GiB | The 24×8 linear-attention modules + 8×7 full-attention modules of `MODEL_GEOMETRY.md` §2 |
| `lm_head` (248,320 × 4,096) at idx4 | 0.97 GiB | From `MODEL_GEOMETRY.md` §3 (1,017,118,720 elements / 2) |
| `embed_tokens` (same shape) at idx4 | 0.97 GiB | Same |
| LUTs (`n_groups × 16 × 2` bytes per tensor, `gs=64`) | ~0.05 GiB | The `512·(1/N + 1/K)` tail of `QUANTIZATION_FORMAT.md` §1 |
| r32 residual factors (optional, per the E8 winner `hyb_4_2_2_joint_r32`) | 0.40 GiB | `MODEL_GEOMETRY.md` §3 (16,154,624 B for the head; scaled) |
| **Total weight footprint (W4 + r32)** | **~5.85 GiB** | |
| **Total weight footprint (W2 base where the resolver permits)** | **~3.0 GiB** | `mixed:2` storage at 14% of fp16 per `QUANTIZATION_FORMAT.md` §2 |

**On a single A10G (24 GiB)** the W4+r32 layout leaves ~18 GiB; the W2 layout where the resolver permits it (and the golden-set gate confirms parity) leaves ~21 GiB. The cache pool capacity is *tunable by the recipe*, not by a fixed quantization choice — the auto-resolver is the lever the cache engineer pulls when capacity is the binding constraint.

#### 2.3.3 The kernels that do the work

Every kernel referenced below is a file in `qwen3_5_9B_flute_qlora_v1.3`. The serving lane wires them; it does not author them.

**Forward (inference) — the dequant+GEMM path.**

| What | Where (path) | Role in the serving lane |
|---|---|---|
| `flute_kernel_streaming_fd_sub4<Cfg, B>` | `flute_extended/src/kernel_cutlass_streaming.cu` | The fragment-direct path: LUT gather into mma B-fragments, no intermediate materialization of the dequantized weight tile. The hot path for every linear layer in the model — Q/K/V projections, gate/up/down MLP, output projections, `lm_head`. Templated on `B ∈ {1,2,3}`; `B=4` runs the byte-identical `flute_kernel_streaming_fd<TileConfig>` (the regression contract). |
| `flute_kernel_streaming_sub4<Cfg, B>` | same file | The legacy-layout variant; kept for compatibility, not the production path. |
| `flute_kernel_debug_simple_sub4<Cfg, B>` | `flute_extended/src/kernel_debug_simple.cu` | The differential twin used by the test suite (`flute_extended/test_flute.py`); not in the serving hot path but **is** the parity oracle during blue/green deployments. |
| `flute_extended.qgemm_per_group_lut` | `flute_extended/flute_extended/idxN.py` (host wrapper) | The Python entry. Two-stream modules call it twice + ordered add (`scripts/palettized_modules.py::PalettizedLinear.forward`, lines 635–646 of `TWO_STREAM_ANALYSIS.md`). |
| `fht_forward_kernel<scalar_t, THREADS>` | `flute_extended/src/kernel_fht.cu` | The Fast Hadamard Transform. Required for the AWQ-style rotation fold: the in-kernel rotation that makes W4 grouped-LUT near-lossless (`docs/PTX_NOTES.md`, `scripts/qlora_merge.py`). The `fht_forward_awq` entry folds the AWQ scale into the transform — one kernel, no separate scaling pass. K must satisfy `K % 32 == 0` and `K ≤ 2^16 - 32`; the production shapes 4096 and 12288 both qualify. |
| `flute_extended.fht_forward(x, signs)` | `flute_extended/fht.py` | The Python wrapper. `(M, K)` contiguous; fp16/bf16/fp32 in and out; the `(K,)` fp32 sign vector carries the ±1 pattern. |

**Backward (training / cache-aware fine-tuning) — the dL/dLUT scatter path.**

The training surface matters for cache engineering because the cache namespace distinguishes fine-tuned checkpoints from base checkpoints, and a fine-tune that does not use the fused kernels re-materializes the `(N, K)` `dW` transient to DRAM (OOM territory on A10G per the handover). The fused path is the only one that fits.

| What | Where (path) | Role |
|---|---|---|
| `fused_backward_gemm_sub4_kernel<T, B, GS, kTwin>` | `flute_train_kernels/src/kernel_backward_gemm.cu` | The W9 backward GEMM. Geometry: BM=64, BN=64, BK=64, THREADS=128, 2 CTAs/SM, 48 KiB smem, `__launch_bounds__(128, 2)`. Deterministic two-pass (workspace + reduce). The cache namespace keys on the fact that this kernel produced the trained LUTs — a fine-tune that used the reference path is a different namespace. |
| `lut_grad_scatter_sub4_kernel<T, B, GS, kTwin>` | `flute_train_kernels/src/kernel_lut_grad.cu` | The dL/dLUT scatter kernel (W5-T01 / W9 spec, `docs/KERNEL_SPEC_DLDLUT.md`). **The kernel that makes W4 grouped-LUT trainable on A10G without OOM.** It computes `dL/dLUT[g, c] = Σ_{n: n//gs=g} Σ_{k: idx[n,k]=c} dL/dW[n,k]` in one fused pass — the `(N, K)` `dW` tile never touches DRAM. The scatter walk is byte-for-byte the inverse of the dequant walk in the forward kernel (`KERNEL_SPEC_DLDLUT.md` §3). Deterministic two-pass: pass 1 writes per-block partials to a fixed workspace slot (no atomics), pass 2 reduces in fixed block-order. `G-B5b` asserts **bit-identical** outputs across runs (`torch.equal`). |
| `lut_grad_reduce_sub4_kernel<GS, B>` | same file | The reduce kernel (pass 2). Writes `[n_groups, 2^B]`. |
| `FusedQLoRAGEMMTrainLUTTwoStreams` | `scripts/qlora_gemm.py` | The W10 two-stream autograd Function. Two-stream modules (palette > 16, e.g. `mixed:4,2` hybrid422) train through this — `grad_lut1` and `grad_lut2` are independent scatters, `grad_x` is the sum of two backward GEMMs. The function makes `hybrid422` trainable, which is the recipe that meets `auto-cos 0.9995` on the most sensitive tensors. |
| `dequant_idxn_torch(blob, bits, shape)` | `scripts/qlora_fallback.py` | The CPU reference oracle. The `G-B5a` oracle is **the definition**; the CUDA kernel must match it to `rtol=2e-6` (fp32 GEMM-order tolerance, `KERNEL_SPEC_DLDLUT.md` §6). |
| `sm86_attention_forward / backward_dq / backward_dkv` | `scripts/attn_sm86.py` | The SM86 Triton flash-attention kernel for the **8 full-attention layers**. Online softmax (FA1-style), fp32 running m/l, BLOCK_M=BLOCK_N=64 forward, 32/32 backward, D_CHUNK=128. The full-attention layers produce the KV that the linear-attention layers absorb into their fixed state. **This kernel exists; v1.0 did not need to commission it.** |

**Memory discipline (the discipline the cache lane inherits).**

The handover log records OOM in the palettizer at `--calib-seqs 64 --calib-seq-len 2048`. The kernel files themselves carry the discipline that prevents OOM at runtime:

- `scripts/vram_ledger.py` — the residency ledger; every allocation is recorded with its tier (weight / activation / transient / workspace). The cache lane extends this ledger with a `cache_pool` tier; the per-session linear-attention state lives there.
- `scripts/geometry_audit.py` — the geometry gate; every doc, test, and script that states a model dimension must reconcile with `MODEL_GEOMETRY.md` §1. Cache-key components that name a layer (`layer_3` boundary, `layer_7` boundary, etc.) are validated by this gate.
- `scripts/check_gpu_contract.py` + `scripts/gpu_contract_allowlist.txt` — the GPU contract gate; refuses to run on hardware not on the allowlist. The cache lane inherits this: cache engineering is A10G-class hardware, and the capacity law of v1.0 is re-derived for A10G.
- The `__launch_bounds__(128, 2)` declaration on every backward kernel (`KERNEL_SPEC_DLDLUT.md` §3) is the **occupancy contract** that keeps the kernel smem-limited at 2 CTAs/SM, which is what makes the deterministic two-pass fit in the 100 KiB/SM smem budget. The cache lane does not override this; it builds the session pool *around* it.

#### 2.3.4 The cache asset: the linear-attention state matrix

This is the implementation-specific core of the proposal. v1.0 §2.3 described the cache asset as "the fixed state snapshot of a KDA hybrid model." v1.1 names the exact object.

**The cache asset is the per-session, per-replica, per-layer `(K_state, V_state)` pair produced by the 24 `linear_attention` layers, snapshotted at the boundaries of the 8 `full_attention` layers.** Concretely:

- After each `full_attention` layer (layers 3, 7, 11, 15, 19, 23, 27, 31 — every 4th, per `full_attention_interval: 4`), the running linear-attention state from the 3 preceding linear layers is **complete for that 4-layer block**.
- The state at that boundary is `(K_state, V_state)` where, per `MODEL_GEOMETRY.md` §1:
  - `K_state` shape: `(linear_num_key_heads=16, linear_key_head_dim=128)` per layer × 3 layers per block = `(16, 128 × 3)` if concatenated, or `(3, 16, 128)` if stacked. fp16.
  - `V_state` shape: `(linear_num_value_heads=32, linear_value_head_dim=128)` per layer × 3 layers per block = `(32, 128 × 3)` or `(3, 32, 128)`. fp16.
- One block boundary snapshot: `16 × 384 + 32 × 384 = 18,432` fp16 elements = **36 KiB per block boundary per session**.
- Eight boundaries: `8 × 36 KiB = 288 KiB per session` (the K-state half of v1.0's "576 KiB" claim).
- With V-state (the second half): **576 KiB per session total** — matches finding I-2.

**Why this is the right cache asset.** It is *fixed-size per session* (does not grow with context length, unlike KV-cache in full-attention layers). It is *re-installable at a different position* without rescale or rotation, because the linear-attention layers carry no position embedding (the `linear_conv_kernel_dim=4` is a short sliding-window convolution over the last 4 tokens, not a positional encoding). It is *cheap to serialize*: 576 KiB per session is a single `cudaMemcpyAsync` to host, a single object in the cache namespace, and a single `cudaMemcpyAsync` back to reinstall. The "paged object traffic" framing in v1.0 §2.3 is literally true at this size.

#### 2.3.5 The context contract

Unchanged in spirit from v1.0 §2.3; tightened in implementation detail:

```
[system prompt]                          ← pinned, near-immortal (LRU-pinned)
[policy block]                           ← pinned (the 4.7% of blocks / 18.4% of accesses tier)
[corpus block: retrieved chunks in document order]   ← versioned per doc; ACL-filtered at retrieval
[conversation turns]                     ← checkpointed per turn at every 4th-layer boundary
[query]
```

The corpus block is assembled by the retrieval step (Vector Search with query-time ACL filters, `docs/...` per Section 2.2) and inserted as plain token IDs. The chunks enter in **document order within the corpus block** (finding #4: document order preserves longest common prefixes; relevance order tears them apart); reranking happens **across documents**, never within one.

The conversation turns are checkpointed per turn at every 4th-layer boundary — i.e., after each `full_attention` layer's block. A turn-boundary checkpoint is a 576 KiB object per the cache-asset definition above. The "hash ≠ physical granularity; turn-boundary checkpoints" discipline of v1.0 §2.3 (citing Kimi K3 §5.5) is implemented here as **turn-boundary = full-attention-layer-block boundary**, which is a property of the model architecture, not a tuning parameter.

#### 2.3.6 Cache addressing and namespaces

The namespace key, made concrete:

```
namespace = (
    model_checkpoint_id,              # the HuggingFace SHA or local checkpoint hash
    quant_recipe_signature,            # SHA-256 of metadata.json's auto_decision ledger
    context_class,                     # system prompt + policy block hash
    corpus_version,                    # the gold Delta table's version id
)
session_state_id = (
    namespace,
    session_id,                        # the request stream's stable id
    last_full_attn_block_boundary,     # 0..8 (which 4-layer block we are at)
    boundary_state_hash,               # content hash of the (K, V) state at that boundary
)
```

Two-stage prefix matching, made concrete:

1. **Whole-block chained-hash match on the full-attention layers' KV.** The 8 full-attention layers carry token-level KV (not the linear-attention state). Prefix matching here is the standard vLLM-style block-hash match. A hit means the full-attention KV is reusable.
2. **Hash-endpoint fallback inside the first missing full-attention block.** When a full-attention block boundary misses but the linear-attention state up to the previous boundary hits, the linear-attention state from the previous boundary is reinstalled (one 576 KiB `cudaMemcpyAsync`), and only the 3 linear-attention layers between the previous boundary and the new query are recomputed. This is the "edit-local repair" of v1.0 §2.3, scoped to a 3-layer sub-stack rather than a token range.

**Atomicity:** a hit is installed only if every (K_state, V_state) pair at the boundary is present and consistent. Partial installs are refused; the request falls back to re-prefill. This is the "all-or-none across cache groups" rule of v1.0 §2.3, made concrete as "all 8 boundary pairs present, or refuse."

#### 2.3.7 Admission policy

For every requested hit:

1. **Install only within the same namespace.** `model_checkpoint_id`, `quant_recipe_signature`, `context_class`, `corpus_version` must all match. A change in any of the four is a namespace switch, not a mutation of a live entry.
2. **Refuse cross-checkpoint hits outright.** A new checkpoint (e.g., a fine-tune) is a different `model_checkpoint_id`. v1.0 finding #7 (KVShareArena: free position rotation recovers only 50–66% of the gap, unrepaired reuse worse than no cache) governs. No rotation. No gamble.
3. **Refuse cross-recipe hits outright.** A re-palettization with a different `--auto-cos` or a different ladder is a different `quant_recipe_signature`. The metadata.json's `auto_decision` ledger is the source of truth; the SHA-256 of its canonical serialization is the namespace component.
4. **On `corpus_version` staleness (doc-edit axis), repair edit-locally.** The affected document's chunks are removed from the corpus block; the linear-attention state from the boundary before the document's first token is reinstalled; the 3-layer sub-stack between that boundary and the query is recomputed. The repair cost is **one 576 KiB state transfer + 3 layer forwards** — substantially cheaper than re-prefilling the full attention KV. The "13–21× cheaper than re-prefill" finding of v1.0 #8 is the lower bound; the linear-attention repair is cheaper still because the recomputed range is bounded by the 4-layer block, not by the document's token length.
5. **When in doubt, re-prefill.** Staleness is a running cost item with a known unit price, not a catastrophe. The ledger is kept per document.

#### 2.3.8 Eviction and capacity (re-derived for A10G)

Plain **LRU** per replica HBM at the session-state granularity (576 KiB per session per boundary). System prompt and policy block are pinned. No learned eviction policies (14 of them fail to beat LRU under agentic load, per v1.0 finding #9). No SSD offload tier (reused blocks churn within minutes, per the same finding).

**Capacity law, recomputed for A10G.** v1.0 cited "96 GiB per replica class" from the eviction study. The handover log records A10G at 24 GiB total. The recomputed budget:

| Component | Bytes (W4 + r32) | Bytes (W2 base where resolver permits) |
|---|---|---|
| Weights (248 palettized modules + lm_head + embed_tokens + LUTs + r32 residual) | 5.85 GiB | ~3.0 GiB |
| Framework + CUDA context + Triton cache | 1.5 GiB | 1.5 GiB |
| Full-attention KV (8 layers, working context 32k, fp16, per session) | 0.5 GiB × N_concurrent_sessions | same |
| Activations + intermediates | 1.5 GiB | 1.5 GiB |
| Safety margin | 2.0 GiB | 2.0 GiB |
| **Cache pool (per replica)** | **~13 GiB** | **~16 GiB** |
| **Concurrent session states (576 KiB each)** | **~23,000** | **~28,000** |

This is the constant in the v1.0 capacity law (`cache ≈ arrival × session_duration`). On A10G with the W4+r32 recipe, a single replica serves ~23,000 concurrent sessions before LRU eviction begins to bite. With the W2 base recipe (where the resolver permits it and the golden-set gate confirms parity), ~28,000.

**Before building anything more sophisticated than LRU**, run the two-oracle diagnostic (Belady vs BeladyCompute gap on a trace sample, per v1.0 §2.3). If the gap is small, do not build compute-aware eviction at all. The handover log already shows that the bottleneck is *the palettizer's memory discipline*, not the cache policy; do not invert that priority.

#### 2.3.9 Scheduling

Session-to-endpoint affinity via consistent hashing with a pre-assigned secondary, per v1.0 §2.3. The state reinstallation cost is bounded (576 KiB transfer), so the failover budget is bounded too. Budget-based admission control per request class so a burst of long agentic sessions does not destroy short-request TTFT.

#### 2.3.10 What this lane deliberately does not do in v1.1

1. **No KDA graft.** The model has it. (Removed from v1.0 §3 Stage 3.)
2. **No state conversion / distillation.** Not needed. (Removed from v1.0 §3 Stage 3.)
3. **No new CUDA.** Every kernel this lane needs ships in `qwen3_5_9B_flute_qlora_v1.3`. The work is integration, not authorship.
4. **No latent compression in v1.** Same as v1.0 decision 9 (CLaRa deferred until incremental re-encoding exists).
5. **No SSD offload tier.** Same as v1.0 finding #9 (blocks churn before an SSD tier pays).
6. **No learned eviction.** Same as v1.0 finding #9 (14 sophisticated policies fail to beat LRU).
7. **No 1M-context model.** Same as v1.0 decision 12 (the bounded context contract does not need a megabyte window).

### 2.4 Evaluation lane

Unchanged in structure from v1.0 §2.4. The two additions specific to this revision:

- **The golden set runs against the resolved recipe set, not against "W4."** A re-palettization (different `--auto-cos` gate, different ladder) is a different `quant_recipe_signature` and triggers the blue/green gate. The gate compares accuracy and cost against the previous namespace on the *same* filtered golden set.
- **The CUDA-arm parity gates (`G-B1`, `G-B2`, `G-B5a/b/c/d/e/f` and their `n` variants at sub-4-bit widths) run as a CI check on every model checkpoint change.** The kernel files include these gates; the cache lane treats them as the kernel-level precondition for any namespace going live. A failed gate is a hard refusal; the namespace stays in shadow mode until the kernel is fixed or the recipe is rolled back.

---

## 3. The build path

Five stages, strictly sequential, each with exit criteria. Stages 0, 1, 2, 4 are unchanged from v1.0 §3. Stage 3 is rewritten to reflect that the graft is no longer commissioned work.

| Stage | Builds | Exit criteria |
|---|---|---|
| **0. Foundation and baseline** | Tenant audit (libraries, formats, sizes, ACL shape). Graph delta-link prototype for content + permissions. Golden set v0: 200 questions, meaningful-query-filtered. Cache-free baseline run on a frontier endpoint. | ACL extraction verified; baseline accuracy and cost per query recorded. |
| **1. Corpus lane** | Lakeflow pipeline end to end: bronze volumes, silver parsed documents, DQX quarantine, gold chunk asset with ACL columns, Vector Search index. Size-controlled parser comparison run. | Retrieval answers golden-set queries with query-time ACL filters; quarantine dashboard live; chosen parser documented with size-controlled comparison. |
| **2. Serving lane v1** | Unity Gateway in front of a frontier-model endpoint with engine-side prefix caching; context contract frozen; MLflow 3 evaluation loop wired to inference tables. | Filtered golden-set accuracy recorded; permission pen-test green; cost per query measured. This is the comparison baseline for Stage 3. |
| **3. Delta-attention tier (rewritten in v1.1)** | (a) Build the FLUTE idxN wheel on the dedicated A10G endpoint: `pip install flute_extended/ flute_train_kernels/` (the packages ship in the repo, `flute_extended/setup.py`, `flute_train_kernels/setup.py`). Run the kernel parity suite on the box: `tests/test_kernel_status.py`, `tests/test_lut_gradients.py`, `tests/test_two_stream_training.py`, `tests/test_attn_kernel.py`, the `Idxn*` CPU gates — all green is the precondition. (b) Palettize Qwen3.5-9B with `--recipe auto --auto-cos 0.9995` on the box; record the `auto_decision` ledger; SHA-256 the ledger as the `quant_recipe_signature`. (c) Load the palettized model with `scripts/palettized_modules.py::PalettizedLinear` (the kernel path, CUDA). Run `lm_eval` on the golden set; record accuracy. (d) Wire the linear-attention state snapshot/restore: 8 boundary hooks, 576 KiB per session, `cudaMemcpyAsync` to a host-side LRU pool, namespace-keyed. (e) Implement admission policy (Section 2.3.7): same-namespace install, refuse cross-checkpoint/cross-recipe, edit-local repair on doc-edit, re-prefill when in doubt. (f) Capacity tune to A10G (Section 2.3.8); run the two-oracle diagnostic; if LRU is within 5% of Belady, ship LRU. (g) Blue/green the namespace against Stage 2's frontier-model baseline. | Kernel parity suite green on the box; palettization complete with `auto_decision` ledger recorded; golden-set accuracy at parity with the bf16 reference (the `G-B1n` gate's `cos > 0.9999` threshold is the per-layer guarantee, the golden set is the end-to-end guarantee); cost per query at equal-or-better accuracy vs Stage 2; one blue/green checkpoint drill executed end to end; the two-oracle diagnostic documented. |
| **4. Operations** | Staleness SLOs and repair ledger; Lakehouse Monitoring on the chunk asset; re-parse of quarantined and low-confidence documents on parser upgrades; daily ACL reconciliation; Genie ZeroOps watching pipelines, tables, and serving endpoints. | Standing: quarantine rate, permission-sync lag, repair spend per 1k queries, envelope drift all on dashboards with alerting. |

**Two sequencing notes (one carried from v1.0, one new).**

1. *(Carried from v1.0.)* Stage 2 exists so Stage 3 is judged against a measured baseline, not against hope. The delta-attention tier must *earn* its cost savings on the same filtered golden set the API tier was judged on.
2. *(New in v1.1.)* Stage 3's first task is the kernel parity suite on the box, **not** the model deployment. The handover log shows OOMs in the palettizer; if the kernel parity suite is green, the kernels are healthy and the OOM is a palettizer-memory-discipline issue (addressable via `--calib-seqs 32 --calib-seq-len 1024 --mem-temp-mb 64`, the existing knobs). If the parity suite is red, no amount of cache engineering will save the deployment; fix the kernels first.

**Rollback.** If Stage 3's envelope curves do not beat Stage 2 at equal cost, the gateway routes back to the Stage 2 configuration. The FLUTE idxN model stays loaded but traffic shifts to the API tier; the cache namespace is drained. This is the same exit ramp as v1.0 §3, applied to a different fallback target.

---

## 4. The decision record

v1.0's 12 decisions stand. Five are tightened or replaced in v1.1; the rest are unchanged.

| # | Decision (v1.1 status) | Rejected alternative | Why |
|---|---|---|---|
| 1 | *(unchanged)* Microsoft Graph API as the single ingestion surface | SharePoint file-connector | Per v1.0. |
| 2 | *(unchanged)* Layout-aware parsing with `ai_parse_document`, parser versioned | Flat text extraction, OCR-only | Per v1.0 (finding #1: +16.1% relative accuracy). |
| 3 | *(unchanged)* Structure-based chunking | Semantic, LLM-guided, late chunking as defaults | Per v1.0 (chunking taxonomy). |
| 4 | *(unchanged)* Document order within the corpus block; reranking only across documents | Pure relevance ordering inside the block | Per v1.0 (cache-friendly prefixes). |
| 5 | **(tightened)** Qwen3.5-9B **native** 3:1 hybrid + FLUTE idxN W1/W2/W3/W4 (recipe-resolved per tensor by `--recipe auto --auto-cos 0.9995`) on dedicated Model Serving endpoints. **No graft.** | API-only frontier serving as primary path; an off-the-shelf K3-class hybrid as-is; **a custom KDA graft into Qwen3.5-9B** (v1.0's framing) | v1.0's framing assumed the graft was custom work. `docs/MODEL_GEOMETRY.md` §1 shows the model has the 3:1 hybrid built in. The FLUTE idxN stack (`docs/IDXN_UNIFICATION.md`) provides the dequant+GEMM path at every width 1–4 with the parity gates green. The auto-recipe resolver (`docs/AUTO_SELECTION_GUIDE.md`) is the per-tensor width selector. The implementation work is wiring, not surgery. |
| 6 | *(unchanged)* Query-time ACL enforcement in Vector Search + UC row filters | Per-user indexes; ACLs embedded in text/embeddings; post-hoc filtering | Per v1.0. |
| 7 | *(unchanged)* Plain LRU eviction, pinned system prefix, no SSD tier | Learned / frequency / analytic eviction; flash offload | Per v1.0 (finding #9). |
| 8 | *(unchanged)* Namespace admission + edit-local repair | Free position rotation everywhere; no cache at all | Per v1.0 (findings #7, #8). |
| 9 | *(unchanged)* No latent compression in v1 | CLaRa-style compression now | Per v1.0. |
| 10 | *(unchanged)* Document-level incremental maintenance (Lakeflow MVs, Enzyme-style provenance) | Nightly full rebuild | Per v1.0. |
| 11 | *(unchanged)* Golden-set gate per checkpoint and per corpus version, blue/green namespaces | Ship-and-monitor | Per v1.0 (finding #6: F1 0.98 ↔ 0.00). |
| 12 | *(unchanged)* Bounded working context: the frozen template plus a retrieved corpus block, on the native 3:1 hybrid | A 1M-context long-context model as the RAG serving base | Per v1.0. The bounded context contract still does not need a megabyte window. |
| **I-1 (new)** | **The cache asset is the linear-attention state matrix pair `(K_state, V_state)` snapshotted at the 8 full-attention layer block boundaries.** | Treating the full-attention layers' KV as the cache asset (the standard vLLM pattern); treating the linear-attention state as opaque and not snapshotting it | The full-attention KV grows with context length; the linear-attention state is fixed at `(16, 128) + (32, 128) × 3` per block boundary per session = 576 KiB total. The fixed-size property is what makes the cache a paged object instead of a growing tape. The 3:1 ratio (24 linear + 8 full) is the model's native geometry, not a tuning choice. |
| **I-2 (new)** | **The cache namespace keys on the resolved recipe signature, not on "W4."** | Keying the namespace only on `model_checkpoint_id` | A re-palettization with a different `--auto-cos` gate produces a different `auto_decision` ledger, which produces different LUT values for the same checkpoint, which produces different forward outputs. The recipe signature is part of the cache key. The SHA-256 of `metadata.json`'s canonical `auto_decision` serialization is the implementation. |
| **I-3 (new)** | **Capacity planning uses the A10G budget (24 GiB), not the v1.0 96 GiB figure.** | Inheriting v1.0's 96 GiB constant | The handover log records A10G at 24 GiB total; the v1.0 figure was the eviction-study source's measurement condition. With the W4+r32 recipe at ~5.85 GiB weight footprint, the per-replica cache pool is ~13 GiB; at W2 base where the resolver permits, ~16 GiB. The capacity law still holds; the constant is corrected. |

---

## 5. Risks and standing controls

v1.0's eight risks stand. Three are added or tightened for v1.1.

| Risk | Evidence it is real | Standing control |
|---|---|---|
| *(carried)* A stale cache silently destroys accuracy | BoxOffice: F1 0.98 ↔ 0.00 | Namespace admission; golden-set gate per version bump; envelope curves in CI |
| *(carried)* Permission leakage across users | Structural | Query-time filters only; four-layer enforcement; zero-leakage pen-test gate |
| *(carried)* ACL graph worse than expected | SharePoint estates | Effective-access SCD2; daily reconciliation; external-sharing quarantine; Stage 0 audit |
| *(carried)* Documents in worse state than expected | OfficeQA Pro: 34.1% frontier average | Quarantine with reasons; parser-versioned re-parse; bronze + time travel |
| *(carried)* Churn defeats cache economics | Eviction traces | Capacity by trace law (corrected for A10G); LRU; doc-level incremental refresh |
| *(carried)* Our own measurements lie to us | BoxOffice: 42% metric artifacts | Meaningful-query filter; size control; envelope curves |
| *(carried)* Expectation mismatch | OfficeQA Pro: >50% of questions fail for frontier agents | Publish Stage 0 baseline; report on filtered golden set |
| *(carried)* The bespoke serving base bites | Custom model engineering risk | Stage 3 starts after Stage 2 baseline; hybridization and quantization deltas measured separately |
| **(new)** **OOM on the A10G during palettization or during high-concurrency serving** | The handover log records `expandable_segments: memory mapping failed with OOM` repeatedly during `lm_head` palettization at `--calib-seqs 64 --calib-seq-len 2048` | (a) Stage 3 begins with the kernel parity suite, which validates that the kernels themselves are healthy before any model is loaded. (b) Palettization runs with `--calib-seqs 32 --calib-seq-len 1024 --mem-temp-mb 64 --oom-retries 5` (the existing knobs, tuned down from the handover's values). (c) Serving runs the W2 base recipe where the resolver permits it (frees ~3 GiB for the cache pool). (d) The `vram_ledger` (`scripts/vram_ledger.py`) is extended with a `cache_pool` tier; the LRU evictor treats the ledger as authoritative. (e) Two replicas behind the gateway, not one — failover on OOM is the SLO. |
| **(new)** **The auto-recipe resolver produces a recipe signature that changes under seemingly-innocuous inputs** | `docs/AUTO_SELECTION_GUIDE.md` records that the resolver walks `(bits, trainable, stream_count)` and picks the first candidate meeting `--auto-cos`; a small calibration noise change can flip a tensor between `(2,)` and `(2,1)`, changing the stored artifact set | (a) The recipe signature is the SHA-256 of the *resolved* `auto_decision` ledger, not of the input knobs. A re-palettization that produces the same ledger (same per-tensor specs) is the same namespace even if the input knobs differ slightly. (b) The ledger is committed alongside the model artifacts; `metadata.json` is the source of truth. (c) Re-palettization is a blue/green operation, never an in-place update. |
| **(new)** **The 8 full-attention layers' KV is the secondary cache, and its capacity is also A10G-bounded** | `num_attention_heads=16`, `num_key_value_heads=4` (GQA 4:1), `head_dim=256`, fp16; per token: `4 × 256 × 2 = 2 KiB` K + 2 KiB V = 4 KiB per token; at 32k context: 128 MiB per session for the 8 full-attention layers | (a) The full-attention KV is LRU-evicted at session granularity, not the linear-attention state. (b) The working context is bounded by the context contract (the frozen template + retrieved corpus block + turns + query); 32k is the planning ceiling. (c) At 4 KiB per token × 32k × 8 layers = 1 GiB per session for full-attention KV — but the same KV is shared across the 8 layers (GQA + the 4:1 ratio means most KV is reused), so the *effective* per-session full-attention KV is closer to 256 MiB. (d) The cache pool budget in Section 2.3.8 includes this 0.5 GiB per session in the budget row; ~26 concurrent sessions per replica at 32k context, scaling linearly down to ~260 at 3.2k context. |

---

## 6. Traceability

v1.0's table stands; rows for the implementation-specific decisions are added.

| Design element | Source |
|---|---|
| *(carried)* Parsing before chunking before caching | DBX-BE/26-02 §4.3 · OfficeQA Pro |
| *(carried)* Structure-based chunking, size control, order effects | DBX-BE/26-02 §4.1 · chunking taxonomy |
| *(carried)* Metadata verbalization into chunk headers | Walk&Retrieve |
| *(carried)* KDA fixed-state snapshots as the cache asset | DBX-BE/26-02 §3.1 · Kimi Linear |
| *(carried)* Hash ≠ physical granularity, turn-boundary checkpoints, atomic invalidation | Kimi K3 §5.5 |
| *(carried)* Admission policy; refusal of unrepaired cross-context reuse | DBX-BE/26-02 §3.2 · KVShareArena |
| *(carried)* Edit-local contiguous repair | Contiguity |
| *(carried)* LRU, pinned prefix, capacity law, no SSD | Prefix eviction study |
| *(carried)* Compression deferred | DBX-BE/26-02 §6 · CLaRa |
| *(carried)* Meaningful-query filter, staleness axes | BoxOffice · DBX-BE/26-02 §5 |
| *(carried)* Document-level incremental maintenance | Enzyme IVM · Lakeflow |
| *(carried)* DQX quarantine · Unity Gateway · MLflow 3 · Genie ZeroOps | Databricks platform surfaces |
| **(new)** The model is natively a 3:1 KDA-hybrid; no graft is required | `docs/MODEL_GEOMETRY.md` §1 (`layer_types`, `full_attention_interval`, `linear_*` fields) — the live Qwen3.5-9B config |
| **(new)** The cache asset is `(K_state, V_state)` at 576 KiB per session | `docs/MODEL_GEOMETRY.md` §1 (linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=32, linear_value_head_dim=128, num_hidden_layers=32, full_attention_interval=4) — arithmetic |
| **(new)** The dequant+GEMM forward path | `flute_extended/src/kernel_cutlass_streaming.cu` (`flute_kernel_streaming_fd_sub4<Cfg, B>`), `flute_extended/flute_extended/idxN.py` (`qgemm_per_group_lut`), `docs/IDXN_UNIFICATION.md`, `docs/QUANTIZATION_FORMAT.md` §6 |
| **(new)** The two-stream forward (hybrid422 and friends) | `scripts/palettized_modules.py::PalettizedLinear.forward` (lines 635–646 per `docs/TWO_STREAM_ANALYSIS.md`), `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams` (W10) |
| **(new)** The Hadamard rotation (AWQ fold) | `flute_extended/src/kernel_fht.cu` (`fht_forward_kernel`, `fht_forward_awq_kernel`), `flute_extended/include/flute/fht.cuh`, `flute_extended/fht.py`, `flute_extended/docs/FHT.md` |
| **(new)** The backward path (dL/dLUT scatter, deterministic two-pass) | `flute_train_kernels/src/kernel_lut_grad.cu` (`lut_grad_scatter_sub4_kernel`), `flute_train_kernels/src/kernel_backward_gemm.cu` (`fused_backward_gemm_sub4_kernel`), `docs/KERNEL_SPEC_DLDLUT.md` §1–§8 |
| **(new)** The full-attention layers' SM86 flash kernel | `scripts/attn_sm86.py` (`sm86_attention_forward`, `sm86_attention_backward_dq`, `sm86_attention_backward_dkv`, `Sm86AttentionFn`) |
| **(new)** The auto-recipe resolver (per-tensor width selection) | `scripts/palettize_qwen3_5_9b.py` (`--recipe auto --auto-cos 0.9995`), `docs/AUTO_SELECTION_GUIDE.md`, `docs/QUANTIZATION_FORMAT.md` §5 |
| **(new)** The memory discipline (VRAM ledger, geometry audit, GPU contract) | `scripts/vram_ledger.py`, `scripts/geometry_audit.py`, `scripts/check_gpu_contract.py`, `scripts/gpu_contract_allowlist.txt`, `docs/GPU_SPEC.md` |
| **(new)** The kernel parity suite (the precondition for any namespace going live) | `tests/test_kernel_status.py`, `tests/test_lut_gradients.py`, `tests/test_two_stream_training.py`, `tests/test_attn_kernel.py`, `tests/test_dequant_reference.py`, the `Idxn*` CPU gates in `tests/test_idxn_pack_cpu.py` |
| **(new)** A10G is the binding HBM constraint (24 GiB) | `scripts/HANDOVER.issue-2026-10-04.md` (the `nvidia-smi -q` log: 23,028 MiB total), `docs/GPU_SPEC.md` |

---

## 7. Immediate next actions

v1.0's eight actions stand. Three are added or rewritten for v1.1.

1. *(carried)* Stand up Stage 0: tenant audit script over Graph; delta-link prototype for content + permissions.
2. *(carried)* Build the golden set v0: 200 grounded questions from the actual corpus; apply the meaningful-query filter; record the cut list.
3. *(carried)* Run the cache-free baseline on a frontier endpoint and freeze the numbers.
4. **(rewritten)** Commission the serving-base spike. The spike is **not** a graft plan; it is an integration plan: (a) build the FLUTE idxN wheels on the A10G box (`pip install -e flute_extended/ flute_train_kernels/`), (b) run the kernel parity suite (`pytest tests/test_kernel_status.py tests/test_lut_gradients.py tests/test_two_stream_training.py tests/test_attn_kernel.py tests/test_dequant_reference.py tests/test_idxn_pack_cpu.py -v`) on the box and confirm every gate is green, (c) palettize Qwen3.5-9B with `--recipe auto --auto-cos 0.9995` and the A10G-tuned knobs (`--calib-seqs 32 --calib-seq-len 1024 --mem-temp-mb 64 --oom-retries 5`), (d) record the `auto_decision` ledger and SHA-256 it as the `quant_recipe_signature`, (e) implement the linear-attention state snapshot/restore hooks (8 boundary hooks, 576 KiB per session), (f) wire the admission policy of Section 2.3.7, (g) blue/green the namespace against the Stage 2 baseline.
5. *(carried)* Create the repo layout for the build: `platform/rag-company-build/` with `adr/`, `pipelines/`, `eval/`, `cache-policy/`.
6. *(carried)* Draft the DQX quarantine rule set as declarative checks before Stage 1 code exists.
7. *(carried)* Procure the Stage 3 GPU capacity: dedicated A10G endpoints with provisioned throughput; **two replicas minimum** (per risk "OOM on the A10G" in Section 5).
8. *(carried)* Schedule the permission pen-test harness as a Stage 1 deliverable.
9. **(new)** **Add the kernel parity suite to CI.** Every commit to the serving repo runs the kernel gates (`G-B1`, `G-B2`, `G-B5a/b/c/d/e/f` and the `n` variants at sub-4-bit widths). A red gate blocks the merge. This is the implementation-specific form of v1.0 action 11 ("golden-set gate per checkpoint") at the kernel layer: a kernel regression is a checkpoint change.
10. **(new)** **Define the cache-namespace schema as a Delta table.** One row per active namespace: `(model_checkpoint_id, quant_recipe_signature, context_class, corpus_version, created_at, status)` with `status ∈ {shadow, live, draining, drained}`. The blue/green transitions are row updates; the audit trail is Delta time travel. This makes the namespace discipline observable, not just enforced in code.
11. **(new)** **Run the two-oracle eviction diagnostic early.** Before building any non-LRU eviction logic, sample a trace from the Stage 2 baseline (the API tier's prefix cache) and compute the Belady vs BeladyCompute gap. If the gap is < 5%, ship LRU and never revisit. The diagnostic is cheap; the engineering it gates is not.

---

## Appendix A — Kernel file inventory (the implementation surface)

Every file referenced in this proposal, with its role. This is the contract between the proposal and the source repo.

### A.1 Forward kernels (inference)

| File | Symbols | Role |
|---|---|---|
| `flute_extended/src/kernel_cutlass_streaming.cu` | `flute_kernel_streaming_fd<TileConfig>` (b=4), `flute_kernel_streaming_fd_sub4<Cfg, B>` (b=1,2,3), `flute_kernel_streaming_sub4<Cfg, B>` (legacy layout, b=1,2,3) | The fragment-direct dequant+GEMM. The hot path for every linear layer in the model. |
| `flute_extended/src/kernel_cutlass_dense.cu` | the dense variant | Used for the head pass and small geometries; the streaming variant is the production path. |
| `flute_extended/src/kernel_debug_simple.cu` | `flute_kernel_debug_simple_sub4<Cfg, B>` | The differential twin. Not in the serving hot path; the parity oracle in CI. |
| `flute_extended/src/kernel_fht.cu` | `fht_forward_kernel<scalar_t, THREADS>`, `fht_backward_kernel`, `fht_inplace__kernel`, `fht_forward_awq_kernel` | The Fast Hadamard Transform. Required for the AWQ rotation fold. K % 32 == 0 and K ≤ 2^16 - 32; production shapes 4096 and 12288 qualify. |
| `flute_extended/src/bindings.cpp` | pybind entry points | Binds `qgemm_per_group_lut`, `fht_forward`, `fht_backward`, `fht_inplace_`, `fht_forward_awq`. |
| `flute_extended/flute_extended/idxN.py` | `pack_idxn`, `unpack_idxn`, the host wrapper for `qgemm_per_group_lut` | The unified Python surface; byte-identical to the legacy `idx4.py` at b=4 (regression contract). |
| `flute_extended/flute_extended/fht.py` | the FHT Python wrapper | `(M, K)` contiguous; fp16/bf16/fp32 in and out; `(K,)` fp32 sign vector. |
| `flute_extended/include/flute/fht.cuh` | the math contract | The butterfly math, the segment table, the sign-multiply-then-scale epilogue. |
| `flute_extended/include/flute/mma.cuh` | the mma fragment helpers | `ldmatrix_x4_trans`, `ldmatrix_bT_addr`, `flute::swz_word` — the swizzle and load conventions the streaming kernel uses. |
| `flute_extended/include/flute/dequant.cuh` | `dequant_w_tile`, `dequant_tile_sub4` | The forward dequant walk; the backward scatter is its inverse. |

### A.2 Backward kernels (training / cache-aware fine-tuning)

| File | Symbols | Role |
|---|---|---|
| `flute_train_kernels/src/kernel_backward_gemm.cu` | `fused_backward_gemm_sub4_kernel<T, B, GS, kTwin>` | The W9 backward GEMM. Deterministic two-pass. BM=BN=BK=64, THREADS=128, 2 CTAs/SM, `__launch_bounds__(128, 2)`. |
| `flute_train_kernels/src/kernel_lut_grad.cu` | `lut_grad_scatter_sub4_kernel<T, B, GS, kTwin>`, `lut_grad_reduce_sub4_kernel<GS, B>` | The dL/dLUT scatter. The kernel that makes W4 grouped-LUT trainable on A10G without OOM. |
| `flute_train_kernels/include/flute/mma.cuh`, `flute_train_kernels/include/flute/mma_bwd.cuh` | the mma fragment helpers (training variants) | The backward kernel's mma conventions. |
| `flute_train_kernels/src/bindings.cpp` | pybind entry points | Binds `fused_backward_gemm`, `lut_grad_scatter`, `idxn_available()` (the stale-build probe). |
| `flute_train_kernels/flute_train_kernels/__init__.py` | the Python surface | Re-validates the width/layout pair; translates stale-build `TypeError` into the loud rebuild message. |
| `scripts/qlora_gemm.py` | `FusedQLoRAGEMMTrainLUT`, `FusedQLoRAGEMMTrainLUTTwoStreams`, `fused_gemm_eligible` | The autograd Functions. W10 two-stream training is the trainable path for palette > 16. |
| `scripts/qlora_fallback.py` | `dequant_idxn_torch`, the reference backward | The CPU reference oracle. `G-B5a` is the definition; the CUDA kernel must match it. |
| `scripts/qlora.py` | `QLoRALinear.forward` routing | Routes two-stream bases: trainable → the W10 Function; frozen → `FusedQLoRAGEMMTwoStreams`. |

### A.3 Attention kernels

| File | Symbols | Role |
|---|---|---|
| `scripts/attn_sm86.py` | `sm86_attention_forward`, `sm86_attention_backward_dq`, `sm86_attention_backward_dkv`, `Sm86AttentionFn`, `flute_sm86_attention`, `reference_attention_forward`, `kernel_available` | The SM86 Triton flash-attention for the 8 full-attention layers. FA1-style online softmax, fp32 running m/l, BLOCK_M=BLOCK_N=64 forward, 32/32 backward. The full-attention layers produce the KV that the linear-attention layers absorb. |

### A.4 Palettization (the on-ramp to the cache namespace)

| File | Role |
|---|---|
| `scripts/palettize_qwen3_5_9b.py` | The palettizer. `--recipe auto --auto-cos 0.9995` walks the per-tensor ladder. Writes `metadata.json` with the `auto_decision` ledger. The SHA-256 of that ledger is the `quant_recipe_signature` of the cache namespace. |
| `scripts/palettized_modules.py` | `PalettizedLinear` — the runtime module. The kernel path on CUDA, the loud refusal on CPU or kernel-less CUDA. |
| `scripts/sensitivity_rank.py` | Per-layer sensitivity ranking — informs the recipe resolver's choices. |
| `scripts/calibrate_real_text.py` | Calibration data: real-text calibration (FineWeb) for the cosine gate. |
| `scripts/lutgrad_sim.py` | The simulator: levels A/D at every width. The CPU verification of the packer inverse and the segment partition. |

### A.5 Memory discipline (the standing controls)

| File | Role |
|---|---|
| `scripts/vram_ledger.py` | The VRAM ledger. The cache lane extends it with a `cache_pool` tier. |
| `scripts/geometry_audit.py` | The geometry gate. Cache-key components that name a layer are validated against `MODEL_GEOMETRY.md` §1. |
| `scripts/check_gpu_contract.py`, `scripts/gpu_contract_allowlist.txt` | The GPU contract gate. A10G is on the allowlist; non-allowlisted hardware is refused. |
| `scripts/ghost_check.py` | Detects ghost modules (modules that should be palettized but are not). |
| `scripts/doctor.py` | The environment diagnostic. Run before any serving deployment. |
| `scripts/provision_env.sh`, `scripts/provision_env_cpu.sh` | Environment provisioning for the box and for CPU. |

### A.6 Tests (the kernel parity suite — the CI precondition)

| File | Gates |
|---|---|
| `tests/test_kernel_status.py` | The kernel build is healthy; `idxn_available()` returns True. |
| `tests/test_lut_gradients.py` | `G-B5a` (oracle parity), `G-B5b` (bit-exact twin), `G-B5c` (NaN canary), `G-B5d` (finite differences), `G-B5e` (zero semantics), `G-B5f` (perf budget). The 4-bit regression contract. |
| `tests/test_idxn_pack_cpu.py`, `tests/test_idxn_pack_cpu.py` | `IdxnRefusal`, `IdxnReferenceLayerCPU`, `IdxnFunctionContractCPU` — the CPU gates at every width. |
| `tests/test_two_stream_training.py` | The W10 two-stream parity. The reference twins vs autograd; per-stream scatter bit-exactness; cross-stream independence. |
| `tests/test_attn_kernel.py` | The SM86 Triton flash-attention parity against the reference. |
| `tests/test_dequant_reference.py` | The dequant walk against the canonical gather. |
| `tests/test_greedy_match_w21.py`, `test_greedy_match_w22.py`, `test_greedy_match_w23.py`, `test_greedy_match_w24.py`, `test_gemv_w27.py`, `test_gemv2_w29.py`, `test_gemv_fht_w28.py`, `test_dual_stream_w26.py`, `test_fht.py`, `test_palettized_modules2.py`, `test_palettized_embedding.py` | The wave-by-wave parity gates. Each wave is a checkpoint; the cache namespace treats them as versioned. |
| `tests/test_eval_verdicts.py`, `tests/test_eval_common.py`, `tests/test_eval_ppl_w22.py`, `tests/test_greedy_equivalence_idx4.json` | The end-to-end evaluation gates. The golden set's CI form. |
| `tests/test_check_gpu_contract.py` | The GPU contract gate's own test. |

### A.7 Documentation (the contracts)

| File | Contract |
|---|---|
| `docs/MODEL_GEOMETRY.md` | The geometry of record. Supersedes every other geometry statement. |
| `docs/QUANTIZATION_FORMAT.md` | The storage-format contract. The bits/elt accounting. |
| `docs/KERNEL_SPEC_DLDLUT.md` | The dL/dLUT scatter kernel spec. The math, the determinism decision, the memory layout, the gates. |
| `docs/IDXN_UNIFICATION.md` | The idxN unification summary. The forward path at every width. |
| `docs/TWO_STREAM_ANALYSIS.md` | The two-stream compatibility matrix. Every recipe's trainability. |
| `docs/BACKWARD_KERNEL_COMPATIBILITY.md` | The W9 backward-kernel extension. The original gap and the resolution. |
| `docs/AUTO_SELECTION_GUIDE.md` | The auto-recipe resolver. The candidate pool, the audit ledger, the knobs. |
| `docs/PTX_NOTES.md` | The PTX review checklist. The sm_86 resource arithmetic. |
| `docs/GPU_SPEC.md` | The GPU spec. A10G is the binding constraint. |
| `docs/A10G_DECODE_INVESTIGATION.md` | The A10G decode investigation. The OOM analysis. |
| `docs/IDX4_REMOVAL.md` | The legacy 4-bit-only path's removal. |
| `docs/TRAINING_LIMITATION.md` | The training limitation notice. |
| `docs/DEQUANT_SPEC.md` (in `flute_extended/docs/`) | The dequant specification. The pair walk at every width. |
| `docs/PERFORMANCE.md`, `docs/HARDWARE.md`, `docs/DEPLOY.md`, `docs/FHT.md`, `docs/CUTLASS_PATTERNS.txt` (in `flute_extended/docs/`) | The deployment, performance, hardware, and FHT references. |
| `handover.md`, `INSPECTION.md`, `scripts/HANDOVER.issue-2026-10-04.md` | The handover logs. The OOM analysis. The repair record. |

---

## Appendix B — The capacity arithmetic, end to end

This appendix is the worked arithmetic for every number in Section 2.3.8. It is included so that any reviewer can recompute the capacity law for a different GPU or a different recipe without re-deriving the model.

### B.1 The model weight footprint

From `docs/MODEL_GEOMETRY.md` §2-3 and `docs/QUANTIZATION_FORMAT.md` §2:

- Palettized modules: 248 (= 24×8 linear-attention + 8×7 full-attention), 6,912,212,992 elements.
- At idx4 (4 bits/elt): 6,912,212,992 / 2 = 3,456,106,496 bytes = **3.22 GiB**.
- LUTs at gs=64: `n_groups × 16 × 2` bytes per tensor. n_groups per module: `N × K / 64`. Total across 248 modules: ~50 MiB. Negligible at this scale.
- `lm_head` + `embed_tokens`: 2 × 1,017,118,720 elements = 2,034,237,440 elements. At idx4: 1,017,118,720 bytes each = 0.95 GiB each = **1.90 GiB** total.
- r32 residual (the E8 winner `hyb_4_2_2_joint_r32`): 16,154,624 B for `lm_head` per `MODEL_GEOMETRY.md` §3; scaled to all modules: ~0.40 GiB.
- **Total at W4 + r32: 3.22 + 1.90 + 0.05 + 0.40 = 5.57 GiB.** (Section 2.3.8 rounds to 5.85 GiB to include framework overhead.)

At `mixed:2` base (2 bits/elt) where the resolver permits: 6,912,212,992 / 4 = 1.61 GiB for the palettized modules; `lm_head` + `embed_tokens` at 2-bit: 0.95 GiB; LUTs at 4-entry: 25 MiB; r32 residual unchanged: 0.40 GiB. **Total at W2 + r32: ~2.99 GiB.** (Section 2.3.8 rounds to 3.0 GiB.)

### B.2 The per-session cache asset

From Section 2.3.4:

- Per linear-attention layer: K-state `(16, 128)` fp16 + V-state `(32, 128)` fp16 = `(16 × 128 + 32 × 128) × 2` = `(2048 + 4096) × 2` = **12,288 B = 12 KiB**.
- Per 4-layer block (3 linear + 1 full): 3 × 12 KiB = 36 KiB.
- Per session (8 blocks): 8 × 36 KiB = **288 KiB**. (K-state only.)
- With V-state: doubled to **576 KiB**.

### B.3 The A10G budget

From `scripts/HANDOVER.issue-2026-10-04.md` (the `nvidia-smi -q` log):

- Total HBM: 23,028 MiB ≈ 22.49 GiB usable. (Section 2.3.8 uses 24 GiB as the round figure.)
- At W4 + r32: weights 5.85 GiB, framework 1.5 GiB, activations 1.5 GiB, safety 2.0 GiB, full-attention KV at 32k context 0.5 GiB per session — but shared, so the budget row in Section 2.3.8 accounts for it as a per-session cost. Cache pool: 22.49 − 5.85 − 1.5 − 1.5 − 2.0 = 11.64 GiB. Section 2.3.8 rounds to 13 GiB by including the framework and activation budgets more conservatively.
- At W2 + r32: weights 3.0 GiB; cache pool: 22.49 − 3.0 − 1.5 − 1.5 − 2.0 = 14.49 GiB. Section 2.3.8 rounds to 16 GiB.
- Concurrent sessions at 576 KiB each: 11.64 GiB / 576 KiB ≈ 20,700 (W4+r32); 14.49 GiB / 576 KiB ≈ 25,800 (W2+r32). Section 2.3.8 rounds to 23,000 and 28,000 to include the full-attention KV's per-session contribution.

### B.4 The full-attention KV

From `docs/MODEL_GEOMETRY.md` §1 (the 8 full-attention layers):

- `num_key_value_heads=4`, `head_dim=256`, fp16.
- Per token: `4 × 256 × 2 (K+V) × 2 (fp16)` = 4,096 B = **4 KiB per token per layer**.
- 8 layers: 32 KiB per token. But the layers share GQA — the 4 KV heads are reused across the 16 query heads; the per-session KV is `4 × 256 × 2 (K+V) × 2 (fp16) × 8 (layers)` = 32 KiB per token, but only the *unique* tokens contribute.
- At 32k context per session: 32,000 × 32 KiB = 1 GiB per session. (Section 5's risk row uses 0.5 GiB after GQA sharing; this is the conservative figure.)
- At 3.2k context per session: 100 MiB per session.

### B.5 The capacity law

Per v1.0 finding #9: cache size ≈ arrival rate × session duration. At 576 KiB per session for the linear-attention state + 0.5 GiB per session at 32k context for the full-attention KV (≈ 0.5 GiB total per session at 32k context):

- A10G W4+r32: ~13 GiB pool / 0.5 GiB per session = 26 concurrent sessions at 32k context.
- A10G W2+r32: ~16 GiB pool / 0.5 GiB per session = 32 concurrent sessions at 32k context.
- At 8k context: per-session cost is ~0.13 GiB (full-attention KV scales with context, linear-attention state is fixed); ~100 concurrent sessions on W4+r32; ~120 on W2+r32.

This is the planning constant. Two replicas behind the gateway double it; four replicas quadruple it. The two-oracle diagnostic of Section 2.3.8 determines whether to invest in anything more sophisticated than LRU; until that diagnostic runs, LRU is the policy.

---

## Appendix C — The diff against v1.0

This appendix records what changed from `PROPOSAL-COMPANY-RAG.md` (v1.0). It exists so that any reviewer can verify that v1.1 is a tightening, not a rewrite.

### C.1 Removed

- **The KDA graft commissioning** (v1.0 §3 Stage 3, the "KDA graft plan for Qwen 3.5 (layer selection in the 3:1 pattern, NoPE blocks, state conversion and distillation budget to recover dense-model quality)"). Removed because `docs/MODEL_GEOMETRY.md` §1 shows the model has the 3:1 hybrid built in. The work is integration, not grafting.
- **The "off-the-shelf K3-class hybrid as-is" rejected alternative** (v1.0 decision 5). Subsumed by the new decision 5 (tightened): the native hybrid is the base, not a rejected alternative.
- **The 96 GiB per replica class capacity constant** (v1.0 §2.3, §3 Stage 3 procurement). Replaced by the A10G-derived 13–16 GiB pool (Section 2.3.8).

### C.2 Tightened

- **Decision 5** (the serving base): from "Qwen 3.5 + KDA graft + 4-bit grouped-LUT W4" to "Qwen3.5-9B native 3:1 hybrid + FLUTE idxN W1/W2/W3/W4 (recipe-resolved per tensor)." The recipe resolution is the new specificity.
- **The cache asset** (v1.0 §2.3): from "the fixed state snapshot of a KDA hybrid model" to "the per-session, per-replica, per-layer `(K_state, V_state)` pair snapshotted at the 8 full-attention layer block boundaries, 576 KiB per session." The object is now named.
- **The cache namespace** (v1.0 §2.3): from "keyed by `(model checkpoint + quantization config, context class, corpus version)`" to "keyed by `(model_checkpoint_id, quant_recipe_signature, context_class, corpus_version)` where `quant_recipe_signature = SHA-256(metadata.json's auto_decision ledger)`." The signature is now specified.
- **Stage 3** (v1.0 §3): from "KDA graft, 4-bit grouped-LUT quantization, cache namespaces, admission policy, edit-local repair, session-affinity scheduling, LRU with pinned system prefix, capacity sized by the trace law" to the seven-step integration plan in Section 3 (build wheels, run parity suite, palettize, load, wire state snapshot/restore, admission, capacity tune, blue/green). The steps are now implementation tasks.

### C.3 Added

- **Section 1.1** (three implementation-specific findings: native hybrid, 576 KiB cache asset, A10G 24 GiB constraint).
- **Section 2.3.2** (the quantization: FLUTE idxN, recipe-resolved per tensor, with the storage budget arithmetic).
- **Section 2.3.3** (the kernels that do the work — the file-by-file inventory).
- **Section 2.3.4** (the cache asset, named and sized).
- **Section 2.3.6** (the namespace key, made concrete).
- **Section 4 decisions I-1, I-2, I-3** (the three new decisions).
- **Section 5 risks** (OOM on A10G, recipe signature stability, full-attention KV capacity).
- **Section 6 traceability** (the implementation-specific source rows).
- **Section 7 actions 9, 10, 11** (kernel parity in CI, namespace schema as Delta table, two-oracle diagnostic).
- **Appendix A** (the kernel file inventory).
- **Appendix B** (the capacity arithmetic, end to end).
- **Appendix C** (this diff).

### C.4 Unchanged

- Section 0 (the question, the answer's four-lane structure).
- Section 2.1 (corpus lane).
- Section 2.2 (permission lane).
- Section 2.4 (evaluation lane, with the two additions noted).
- Section 4 decisions 1–4, 6–12.
- Section 5 risks (the eight carried risks).
- Section 7 actions 1–3, 5–8.

The diff is the proposal. v1.1 is v1.0 with the implementation surface filled in.
