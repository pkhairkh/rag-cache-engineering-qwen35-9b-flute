# DISK SIZING — Embedding Index on Disk, with HNSW / IVFADC Preselect

> **Companion to:** `SIZING-officeqa-cache.md` (which conflated the cache pool with the index — corrected here)
> **Question:** How much **disk space** do we need, given that we'll use HNSW or IVFADC to preselect on disk, then load only top-k=3 into VRAM for the final cos sim?
> **Answer:** The disk footprint is small and well-understood. **Full OfficeQA: ~3.5 GiB with HNSW, ~1.4 GiB with IVFADC.** **50k subsample: ~2.0 GiB with HNSW, ~0.8 GiB with IVFADC.** The choice between HNSW and IVFADC is a latency-vs-disk tradeoff, not a capacity constraint.

---

## 0. The architecture, stated correctly

```
┌─────────────────────────────────────────────────────────────────────┐
│  DISK (persistent, all chunks)                                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ The embedding index                                           │   │
│  │  • chunk text (UTF-8)                                         │   │
│  │  • embedding vectors (fp16, 1024-dim for bge-m3)              │   │
│  │  • metadata (doc_id, page, chunk_id, ACL, provenance)         │   │
│  │  • ANN structure (HNSW graph OR IVFADC coarse+PQ codes)       │   │
│  │  • token IDs (int32, for re-feeding to the model)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Retrieval: query embedding → ANN preselect (top-100) → cos sim    │
│  on the 100 → return top-k=3 to VRAM                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ top-k=3 chunks (≈ 3 × 1024 tokens × 4 B = 12 KiB)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VRAM (A10G, 24 GiB) — only the hot working set                    │
│                                                                     │
│  • Model weights (W4+r32): 5.85 GiB                                │
│  • The 3 retrieved chunks (token IDs + embeddings): ~40 KiB         │
│  • The model's per-session cache_params (KV + recurrent state):    │
│    ~25.5 MiB + ~256 MiB at 8k context                              │
│  • The two global caches (hot prefixes only, LRU): ~13 GiB budget  │
└─────────────────────────────────────────────────────────────────────┘
```

The disk holds everything. VRAM holds: (a) the model, (b) the current session's working state, (c) the LRU-hot prefixes in the two global caches. Retrieval is a 3-stage pipeline:

1. **ANN preselect on disk** (HNSW graph traversal, or IVFADC coarse-quantizer + PQ code scan): returns top-100 candidates in ~1–5 ms.
2. **Cos sim on the 100 candidates** (load their fp16 vectors from disk, compute dot products): ~1 ms.
3. **Load top-k=3 chunks into VRAM** (token IDs + text for the model): ~12 KiB, microseconds.

The disk is the index. VRAM never sees the full corpus.

---

## 1. Per-chunk disk footprint

### 1.1 The chunk payload (the data, before the ANN structure)

| Component | Size | Notes |
|---|---|---|
| Embedding vector (BAAI/bge-m3, 1024-dim, fp16) | 2,048 B | The dense vector — the retrieval key |
| Chunk text (UTF-8, ~1024 tokens × ~4 B/token) | 4,096 B | The raw text, for re-reading and for the model |
| Token IDs (int32, 1024 tokens) | 4,096 B | Pre-tokenized, for direct feeding to the model |
| Metadata (doc_id, page, chunk_id, section, table_flag, ACL set, provenance SHA) | 256 B | For filtering and provenance |
| **Per-chunk payload subtotal** | **10,496 B** | **~10.3 KiB** |

### 1.2 The ANN structure overhead (the index, on top of the payload)

Two options, sized separately:

**Option A: HNSW (Hierarchical Navigable Small World graph)**
- A graph where each node is a chunk, edges connect to ~16–32 nearest neighbors at each layer.
- Storage: the graph edges. For 1024-dim fp16 vectors with M=16 (the typical HNSW parameter):
  - Per-node edge list: ~16 edges × 4 B (int32 node ID) × log(N) layers ≈ 16 × 4 × 5 = 320 B per node at N=50k
  - Plus the entry-point pointers and layer assignments: ~64 B per node
  - **HNSW overhead: ~384 B per chunk** (independent of vector dimensionality — it's graph structure, not vectors)
- The vectors themselves are stored separately (the 2,048 B per chunk from §1.1) because HNSW traverses them during search.
- **Total per chunk with HNSW: 10,496 + 384 = 10,880 B ≈ 10.6 KiB**

**Option B: IVFADC (Inverted File with Additive Quantization / Product Quantization)**
- A coarse quantizer (k-means with ~√N centroids) + per-cluster inverted lists + PQ codes for fast distance approximation.
- Storage:
  - Coarse quantizer: √N × 1024 × 2 B (the centroids). For N=50k: 224 × 2 KiB = 448 KiB (negligible per chunk)
  - PQ codes per vector: 1024-dim → 64 sub-vectors of 16-dim, each encoded as 1 byte (256 centroids per sub-vector) = 64 B per chunk
  - Inverted list overhead: ~16 B per chunk (the cluster assignment + position)
- **IVFADC code overhead: ~80 B per chunk** (the PQ codes replace the full vectors for distance computation)
- BUT: for the final cos sim on the top-100, we need the full fp16 vectors. So we still store the 2,048 B per chunk (from §1.1) for the reranking step.
- **Total per chunk with IVFADC: 10,496 (payload) + 80 (PQ codes) = 10,576 B ≈ 10.3 KiB**
- The PQ codes are ~80 B vs HNSW's ~384 B — IVFADC is smaller, but the payload dominates either way.

### 1.3 The realistic per-chunk total

| Component | HNSW | IVFADC |
|---|---|---|
| Payload (text + tokens + embedding + metadata) | 10,496 B | 10,496 B |
| ANN structure overhead | 384 B | 80 B |
| **Per-chunk total** | **10,880 B** | **10,576 B** |
| **Rounded** | **~10.6 KiB** | **~10.3 KiB** |

The ANN structure is a small fraction of the total (~3.5% for HNSW, ~0.8% for IVFADC). The payload (text + tokens + embedding) dominates. The choice between HNSW and IVFADC is a latency-vs-recall tradeoff, not a disk-capacity tradeoff.

---

## 2. Total disk space — full OfficeQA

OfficeQA Pro: 89,000 pages, ~1,000 tokens/page (table-dense Treasury Bulletins) = ~89M tokens.

At 1024 tokens/chunk: ~87,000 chunks.

| Component | HNSW | IVFADC |
|---|---|---|
| Chunk payloads (87k × 10,496 B) | 913 MiB | 913 MiB |
| ANN structure (87k × 384 B or 80 B) | 33 MiB | 7 MiB |
| Index metadata (catalog, centroids, manifest) | ~10 MiB | ~10 MiB |
| **Total disk** | **~956 MiB ≈ 0.93 GiB** | **~930 MiB ≈ 0.91 GiB** |

**Full OfficeQA on disk: ~0.9 GiB.** Trivial on any modern disk. Even a 10-year-old laptop SSD holds this.

At 512 tokens/chunk (smaller chunks, more of them): ~174,000 chunks → ~1.86 GiB. Still trivial.

### 2.1 If we keep the raw PDFs too (for re-parsing on parser upgrades)

| Component | Size |
|---|---|
| 89,000 pages of Treasury Bulletins as PDFs (avg ~100 KiB/page for scanned+OCR'd government docs) | ~8.7 GiB |
| Parsed markdown (the silver layer, before chunking) | ~1.5 GiB |
| **Total with raw + parsed + index** | **~11 GiB** |

Still fits on a single disk. The raw PDFs are the largest component; the index is small.

---

## 3. Total disk space — 50k subsample

50,000 chunks at 1024 tokens/chunk ≈ 51M tokens (roughly half the full corpus).

| Component | HNSW | IVFADC |
|---|---|---|
| Chunk payloads (50k × 10,496 B) | 525 MiB | 525 MiB |
| ANN structure (50k × 384 B or 80 B) | 19 MiB | 4 MiB |
| Index metadata | ~5 MiB | ~5 MiB |
| **Total disk** | **~549 MiB ≈ 0.54 GiB** | **~534 MiB ≈ 0.52 GiB** |

**50k subsample on disk: ~0.5 GiB.** Trivial.

---

## 4. Total disk space — PoC (50 questions, top-3 retrieval)

For the PoC, we don't index the whole corpus — we index only the chunks that the 50 questions' gold answers reference, plus a small distractor set.

| Component | HNSW | IVFADC |
|---|---|---|
| Chunk payloads (250 chunks × 10,496 B) | 2.6 MiB | 2.6 MiB |
| ANN structure (250 × 384 B or 80 B) | 0.1 MiB | 0.02 MiB |
| **Total disk** | **~2.7 MiB** | **~2.6 MiB** |

**PoC on disk: ~3 MiB.** Fits in /tmp.

---

## 5. The HNSW vs IVFADC tradeoff (since disk isn't the constraint)

Since both fit on disk easily, the choice is about query latency and recall:

| Metric | HNSW (M=16) | IVFADC (nlist=224, PQ m=64) |
|---|---|---|
| Disk space (50k chunks) | 0.54 GiB | 0.52 GiB |
| Query latency (top-100 preselect, 50k chunks) | ~1-2 ms | ~3-5 ms |
| Recall@100 (vs brute-force) | ~98% | ~92% |
| Build time (50k chunks) | ~30 s | ~10 s |
| Memory at query time (the ANN structure loaded) | ~20 MiB | ~5 MiB |
| Best for | High-recall, low-latency (the PoC's choice) | Massive scale (10M+ chunks), lower recall acceptable |

**Recommendation for the PoC: HNSW.** The 50k-subsample is small enough that HNSW's ~2 ms latency and ~98% recall are the right tradeoff. IVFADC becomes interesting at 10M+ chunks where HNSW's graph traversal gets slow.

Libraries: **FAISS** (both HNSW and IVFADC), **hnswlib** (HNSW only, faster build), **Milvus** (production, both). For the PoC on A10G: `faiss.IndexHNSWFlat(1024, 16)` with fp16 vectors — ~2 ms per query, ~98% recall@100.

---

## 6. The VRAM footprint during retrieval (the part that matters for A10G)

Retrieval flow: query embedding → ANN preselect (on disk or in CPU RAM) → top-100 → cos sim (CPU RAM) → top-3 → load to VRAM.

**What enters VRAM during retrieval:**

| Component | Size | When |
|---|---|---|
| Query embedding (1024-dim fp16) | 2 KiB | Once per query (computed on GPU if the embedder runs on GPU, or on CPU) |
| Top-3 chunk token IDs (3 × 1024 × int32) | 12 KiB | Once per query (loaded from disk) |
| Top-3 chunk text (3 × 4 KiB UTF-8) | 12 KiB | Once per query (for the model's input) |
| Top-3 chunk embeddings (3 × 2 KiB fp16) | 6 KiB | Once per query (for the cos sim reranking, if done on GPU) |
| **Total VRAM for retrieval** | **~32 KiB** | **Per query** |

The retrieval footprint in VRAM is **~32 KiB per query** — negligible. The ANN preselect and the cos sim on the top-100 happen on CPU (the index lives in CPU RAM, ~525 MiB for the 50k subsample, loaded once at startup). Only the top-3 results cross into VRAM.

**The A10G's VRAM budget is dominated by:**
- Model weights (W4+r32): 5.85 GiB
- The two global caches (LRU-hot prefixes): up to 13 GiB
- The current session's cache_params (KV + recurrent state at 8k context): ~280 MiB
- Framework/activations/safety: ~5 GiB

Retrieval adds ~32 KiB. It's not even a rounding error.

---

## 7. Summary — disk space needed

| Corpus | Chunks | HNSW disk | IVFADC disk | Notes |
|---|---|---|---|---|
| PoC (50 questions, top-3 retrieval) | 250 | ~3 MiB | ~3 MiB | Fits in /tmp |
| 50k subsample | 50,000 | ~0.54 GiB | ~0.52 GiB | Trivial on any disk |
| Full OfficeQA (1024 tok/chunk) | 87,000 | ~0.93 GiB | ~0.91 GiB | Trivial |
| Full OfficeQA (512 tok/chunk) | 174,000 | ~1.86 GiB | ~1.82 GiB | Trivial |
| Full OfficeQA + raw PDFs + parsed markdown | — | ~11 GiB | ~11 GiB | Still trivial |

**The answer to your question:** for the 50k subsample, **~0.5 GiB of disk** with either HNSW or IVFADC. For full OfficeQA, **~0.9 GiB**. For the PoC, **~3 MiB**. The disk is not the constraint — it never was. The constraint is (a) the cache pool in VRAM (covered in `SIZING-officeqa-cache.md`) and (b) the query latency (HNSW ~2 ms vs IVFADC ~5 ms, both fast enough).

**Recommendation:** use HNSW (FAISS `IndexHNSWFlat(1024, 16)`) for the PoC and the 50k subsample. Switch to IVFADC only if the corpus grows past ~1M chunks (where HNSW's graph traversal slows). Disk space is not a factor in this decision.
