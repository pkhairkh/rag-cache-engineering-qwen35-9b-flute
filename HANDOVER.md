# RAGGA Handover Document

## Executive Summary

RAGGA is a RAG system using TurboQuant-compressed KV-cache snapshots for retrieval. After 15 waves of debugging and fixes, the technical pipeline is stable but **answer quality remains poor**. This document summarizes what works, what doesn't, and the remaining open issues.

---

## Current State (W15-post)

| item | status |
|---|---|
| CPU test suite | **191 tests green** (`python3 -m pytest src/rag/tests -q`) |
| GPU kernel compilation | **OK** — FHT, GEMV, split-K kernels built |
| Ingestion | **OK** — 100 chunks in ~360s |
| Frame check (FHT) | **OK** — kernel matches reference (rel-MSE ~1e-14) |
| S-read / conv paths | **OK** — all variants pass quant/dequant roundtrip |
| true-doc control | **OK** — conf 0.895, generates correctly from raw cache |
| Verification ladder | **6/6 stages PASS** |
| QJL path | **BROKEN** — device mismatch bug |
| Answer quality | **POOR** — garbage output on test queries |
| Retrieval relevance | **POOR** — wrong documents retrieved |

---

## What Works

### 1. Core Pipeline

```
ingestion → index → retrieval → install_snapshot → generation
```

- **Ingestion**: `run_ingestion.py` processes documents, captures TQ-compressed cache snapshots
- **Index**: `run_index.py` builds IVFADC index (or flat at high dimension)
- **Retrieve**: `rerank()` computes similarity scores between query vector and chunk vectors
- **Install**: `install_snapshot()` sums TQ codes and installs to cache
- **Generate**: `_greedy_decode()` produces tokens from installed cache state

### 2. TQ Quantization (W15 fixes)

The W15 coding wave implemented paper-correct quantization:

- **Outlier split** (§LongBench): top-k channels at 4 bits, rest at 3 bits, each with own pow2 full rotation
- **Frame check passes**: CUDA FHT kernel (adjoint-roundtrip, kernel-vs-reference) at 1e-14 level
- **TRUE-dist measurably improved**: conv window at 0.068 rel-MSE (vs 0.327 with old fixed split)
- **All bisection variants pass read checks**

### 3. Cache State Installation

- **S codes**: sum via dequant-add-requant chain (1.3x single-shot noise)
- **conv codes**: installed verbatim (never summed, §6 requirement)
- **M1/M2 flags**: plumbed for future feature work

### 4. GPU Stack

- NVIDIA A10G 24GB VRAM
- CUDA 13.2, PyTorch 2.7.0a0
- FLA 0.5.2 (gated delta rule kernels)
- Triton 3.2 (attention kernels)

---

## What Doesn't Work

### 1. Answer Quality is Poor

**Symptom**: Running the official questions against the corpus produces garbage text.

```
Query: What are the default size limits for file uploads...?
Top 3: [70, 80, 71] scores [0.7232, 0.7072, 0.7071]
Answer: -M", ( your  in J by,,, \, "...
```

The gold answer is:
```
The default limits are 10 MiB per file (max_file_size) and 50 MiB total 
per request (max_total_request_size) for multipart uploads on the 
OpenAI-compatible endpoints.
```

**Root cause unknown**. The bisection shows:
- `true-doc` generates correctly (conf 0.895) from RAW cache
- `full` (TQ-installed cache) generates garbage (conf 0.30)
- Frame check passes — no kernel/reference split
- S-read/conv checks pass — quant/dequant roundtrip OK

**Hypothesis**: The TQ layer reads codes correctly in isolation but the **composed forward pass** corrupts state in a way the read checks don't catch.

### 2. Retrieval Relevance is Poor

The official question `qst_0001` expects document `dsid_ae068ee4aa9640159427cd941bef0238`. The system retrieved chunks `[70, 80, 71]` instead.

This suggests either:
- The query vector (from TQCache state) doesn't match the document's stored vector
- The rerank similarity metric is misaligned
- The ingestion captured incorrect cache state

### 3. QJL Path Has Device Bug

```
RuntimeError: Expected all tensors to be on the same device, but found at least 
two devices, cuda:0 and cpu!
```

In `turboquant.py::_attach_qjl`:
```python
resid = x32 - x_mse  # x32 is CUDA, x_mse is CPU
```

The MSE reconstruction wasn't moved to device before residual computation.

---

## The W15 Diagnosis

W14 proved every READ-level check passes while the full install generates garbage. W15 read the paper line-by-line, audited every TQ code path, and measured the WRITE-path distortion. Key findings:

### 1. The 3.5-bit recipe was NOT the paper's recipe (FIXED)

The paper (§LongBench): non-integer bits come from "splitting channels into outlier and non-outlier sets, and applying two independent instances of TurboQuant to each, allocating higher bit precision to outliers."

The old repo: FIXED 50/50 coordinate split — data-oblivious to channel-energy structure.

**The fix**: Implemented outlier split with per-subset pow2 full rotation.

**Measured improvement**:
| recipe | eff bits | write-path rel-MSE |
|---|---|---|
| old fixed half split | 3.5 | 0.017-0.025 |
| paper outlier split | 3.25 | **0.007** |

### 2. Exonerated: the math

- **codebooks.py**: faithful to the paper — Lloyd-Max solver, 1/sqrt(d) law
- **S single-shot**: 0.022 rel-MSE = exactly the 3.5-bit blend floor
- **install sum**: 1.3x single-shot, not 4x
- **online decode loop**: idempotent (0.3%/step)

### 3. The residual noise budget

The installed S state is ~0.08-0.11 rel-MSE from the true doc state — dominated by ONLINE INGESTION drift (the chunk prefill evolves from a 2.2%-noisy reseeded state through 300 tokens of recurrence).

true-doc (0% noise) generates at conf 0.895; every TQ variant sat at 0.28-0.43 — the model is sensitive to state noise.

---

## Key Files

| file | purpose |
|------|---------|
| `src/rag/tq_cache.py` | TQLinearAttentionLayer, _StateView, s_codes/conv_codes setters |
| `src/rag/turboquant.py` | TurboQuant.quant/dequant, outlier split, QJL path |
| `src/rag/install.py` | install_snapshot, sum_turboquant_codes |
| `src/rag/snapshot.py` | Snapshot, save_snapshot, load_snapshot, load_chunk |
| `src/rag/query.py` | answer_query, query_cache_vector, _greedy_decode |
| `src/rag/index.py` | build_index, preselect, rerank, ChunkVectorLoader |
| `scripts/gpu/run_ingestion.py` | Ingestion driver |
| `scripts/gpu/run_query.py` | Query driver |
| `scripts/gpu/bisect_install.py` | Diagnostic bisection script |
| `scripts/gpu/verify_pipeline.py` | Verification ladder (G1-G6) |

---

## Model Details

- **Model**: Qwen3.5-9B palettized
  - Path: `/home/ubuntu/qwen3_5_9B_palettized` + `_heads`
  - vocab_size: 248,320
  - EOS: `<|im_end|>` (id 248,046)
  - 32 layers: [L,L,L,F]×8 (24 linear attention + 8 full attention)

- **GDN Geometry**:
  - k_heads: 8, v_heads: 32
  - head_dim: 128
  - conv window: 24,576 = 6,144×4 (FHT segments 16,384 + 8,192)
  - S: 524,288 per layer (single 2^19 full rotation)

- **Compression**:
  - 6.01x achieved (4.63 GiB model)
  - 3.5 bits per element effective
  - Per-group LUT tables

---

## Dataset Details

- **Corpus**: EnterpriseRAG-Bench (subset)
  - Path: `/home/ubuntu/enterprise_rag_bench/documents/documents.jsonl`
  - Documents ingested: 100 (out of 512K corpus)
  - Questions: `/home/ubuntu/enterprise_rag_bench/questions/questions.jsonl`
  - Each question has `expected_doc_ids` and `gold_answer`

---

## Bisection Matrix (W15)

```
variant      S-read   conv     TRUE   gen        conf   rep
true-doc     -        -        -      OK        0.895  0.00
reseed       OK       OK       OK     OK        0.449  0.00
s-only       OK       OK       HIGH   OK        0.274  0.36
conv-only    OK       OK       OK     OK        0.419  0.27
full         OK       OK       HIGH   OK        0.300  0.36
```

**Key observation**: All variants "pass" gen gate (conf ≥ 0.20) but have low confidence. true-doc at 0.895 shows the model CAN generate correctly from document state.

---

## Open Investigations

### 1. Why does full install produce garbage when true-doc produces correct output?

**Hypothesis A: _StateView device/shape mismatch**

The installed codes are readable by `TurboQuant.dequant` but `TQLinearAttentionLayer._StateView` may:
- Read from stale `_s_codes` / `_conv_codes` attributes
- Have device mismatch between codes and layer weights
- Have shape mismatch between dequant output and expected geometry

**Test**: Trace the exact code path when model generates:
1. Does `_StateView.s` read `_s_codes` via dequant?
2. Is the dequant output on CUDA?
3. Does the shape match what the layer expects?

**Hypothesis B: TQ layer forward corrupts state**

The quant/dequant roundtrip passes standalone checks, but the composed layer forward may introduce corruption not visible in isolated dequant.

**Test**: Run one forward pass through a TQLinearAttentionLayer with installed codes and compare output against the same input with raw cache.

### 2. Why does retrieval return wrong documents?

**Hypothesis: Query vector doesn't match stored vectors**

The ingestion creates vectors from TQCache state at end-of-chunk. The query creates a vector from fresh TQCache after query prefill. If these states differ in quantization path, the vectors won't match.

**Test**: 
1. Load a chunk snapshot
2. Run its document text through query prefill
3. Compare the resulting vector against the stored vector

**Hypothesis: Vector dimension mismatch**

The manifest shows `vector_dims: 12582912` (12.5M). This is S (524288) × 24 layers. But the actual cache state may have different size.

**Test**: Verify that `query_cache_vector` and `ChunkVectorLoader` use identical vector construction.

### 3. Why is generation low-confidence even when passes?

All variants show conf 0.27-0.45, far below true-doc's 0.895. This suggests:
- The TQ noise floor is high enough to degrade generation
- The model is sensitive to cache state perturbation
- Or the generation is actually broken but passes the weak gate

---

## Commands

```bash
# Compile kernels
cd /home/ubuntu/RAGGA/src/flute_extended
python3 setup.py build_ext --inplace

# Run tests (CPU-only tests pass, GPU tests need CUDA)
cd /home/ubuntu/RAGGA
python3 -m pytest src/rag/tests -q --tb=short

# Ingest documents
python3 scripts/gpu/run_ingestion.py

# Build index
python3 scripts/gpu/run_index.py

# Run verification ladder
python3 scripts/gpu/verify_pipeline.py

# Run bisection diagnostics
python3 scripts/gpu/bisect_install.py

# Run query
python3 scripts/gpu/run_query.py --question "Your question here" --max-new-tokens 64

# Run against benchmark questions
python3 scripts/gpu/run_query.py --questions-file /home/ubuntu/enterprise_rag_bench/questions/questions.jsonl --n-questions 10 --max-new-tokens 64
```

---

## Session History Summary

- **W1-W11**: Fixed 179 tests, built kernels, ingested documents
- **W12**: Investigated "S-read DRIFT" — turned out to be measurement bug in bisection script
- **W13**: Added test gates for S+conv combination
- **W14**: Proved frame check OK, true-doc OK, but full install still garbage
- **W15**: Implemented paper-correct outlier split, improved conv TRUE-dist
- **W15-post**: Verification ladder passes, but answer quality remains poor

---

## Remaining Work

1. **Debug retrieval**: Why are wrong documents being retrieved?
2. **Debug generation**: Why does TQ-installed cache produce garbage when true-doc produces correct output?
3. **Fix QJL**: Device mismatch in `_attach_qjl`
4. **Evaluate against full benchmark**: Run all 500 questions, compute accuracy

---

## Contact Points in Code

- Query vector: `src/rag/query.py::query_cache_vector`
- Retrieval: `src/rag/index.py::rerank`
- Install: `src/rag/install.py::install_snapshot`
- Generation: `src/rag/query.py::_greedy_decode`
- TQ layer: `src/rag/tq_cache.py::TQLinearAttentionLayer`
- Quant: `src/rag/turboquant.py::TurboQuant.quant`
- Dequant: `src/rag/turboquant.py::TurboQuant.dequant`

---

## End State

**Technical infrastructure works. Answer quality does not.**

The pipeline executes without crashes. All test gates pass. But the end-to-end RAG experience produces garbage answers.

The next investigator should focus on:
1. Why `true-doc` (raw cache) works but `full` (TQ cache) doesn't
2. Why retrieval returns wrong documents
3. Whether the query vector construction matches the stored vectors

Good luck.
