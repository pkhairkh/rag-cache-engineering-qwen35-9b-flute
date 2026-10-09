# PIPELINE — Dataset Prep, Retrieval, Augmentation, Benchmark (50k Subsample)

> **Companion to:** `PROPOSAL-POC-officeqa-a10g.md` (v3), `DISK-SIZING-officeqa-index.md`, `SIZING-officeqa-cache.md`
> **Question:** How exactly will dataset preparation work? How exactly will retrieval work? How exactly will augmentation work? For the 50k subsample. And how do we benchmark?
> **Scope:** the operational spec for the 50k-subsample PoC on A10G. Every step is concrete: the script, the input, the output, the gates. The benchmark protocol is grounded in the 2026 audit-wave papers (BoxOffice, KVShareArena, Contiguity, the LRU paper) — the same protocol the v3 PoC prescribes, made operational.

---

## 0. The 50k subsample, defined

"50k" = **50,000 chunks** at 1024 tokens/chunk, drawn from OfficeQA's 89,000-page Treasury Bulletin corpus. This is roughly half the full corpus (~87k chunks at 1024 tok/chunk).

The subsample is **stratified by document**, not random — we keep whole documents (so multi-document reasoning questions are answerable) and sample documents until we hit 50k chunks. This preserves the question-answerability of the OfficeQA gold set: a question that references Treasury Bulletin 2018-Q3 must have that document in the subsample, or it's unanswerable.

**The subsample selection algorithm:**

```python
# poc/subsample_corpus.py
def select_50k_subsample(corpus_chunks, target_chunks=50000, seed=0):
    """Stratified-by-document subsample.
    1. Group chunks by doc_id.
    2. Sort documents by chunk count (descending) — keep the densest docs
       (most chunk-rich, most likely to be referenced by gold questions).
    3. Greedily add whole documents until chunk count >= target.
    4. Record which gold questions are answerable (their referenced docs are
       in the subsample). Discard unanswerable questions from the eval set.
    Returns: (subsample_chunks, answerable_questions, subsample_manifest)."""
```

**Expected outcome:** ~50-60 documents (Treasury Bulletins are ~1,000 pages each, ~1,000 chunks each), ~50,000 chunks, ~80-90% of the gold questions answerable (the rest reference documents not in the subsample).

---

## 1. Dataset preparation (exactly how it works)

### 1.1 The inputs

| Input | Source | Size |
|---|---|---|
| OfficeQA dataset (questions + gold answers + document references) | `databricks/officeqa` on HuggingFace (CC-BY-SA-4.0, CSV) | ~5 MiB (the Q&A pairs) |
| Treasury Bulletin documents (the corpus) | The dataset's metadata references the documents; the documents are U.S. government works (public domain). Fetch via the dataset's document URLs or the OfficeQA GitHub repo. | ~89,000 pages, ~8.7 GiB as PDFs |
| The parser | **docling** (CPU-side, Apache 2.0) — the `ai_parse_document` proxy. Handles tables, reading order, layout. | pip install |
| The chunker | structure-based (heading hierarchy + paragraphs + atomic tables) — per the chunking taxonomy (arXiv:2602.16974) | custom, ~100 lines |
| The embedder | **BAAI/bge-m3** (multilingual, 8K context, Apache 2.0, 1024-dim fp16) | pip install + 2.3 GiB model download |

### 1.2 The pipeline (5 stages, sequential)

```
Stage 1: Ingest          Stage 2: Parse         Stage 3: Chunk
  PDFs (bronze)    →     layout-aware     →    structure-based
  raw bytes              markdown + tables      1024-token chunks
                         (silver)               (gold candidate)

Stage 4: Embed          Stage 5: Index
  bge-m3 fp16      →     FAISS HNSW
  1024-dim vectors        + metadata SQLite
  (gold)                  (the index on disk)
```

**Stage 1 — Ingest (bronze layer):**

```python
# poc/prep/01_ingest.py
def ingest_documents(dataset_name="databricks/officeqa", output_dir="data/bronze"):
    """Download the OfficeQA dataset, fetch the referenced Treasury Bulletin
    PDFs, store as raw bytes in data/bronze/<doc_id>.pdf.
    Records: doc_id, source_url, sha256, page_count, fetch_timestamp.
    Output: data/bronze/manifest.json + <doc_id>.pdf files."""
```

- Input: `databricks/officeqa` CSV (the Q&A pairs with document references)
- Output: `data/bronze/<doc_id>.pdf` (raw PDFs) + `data/bronze/manifest.json`
- Time: ~30 min for the 50k subsample's ~60 documents (network-bound)
- Gate: every PDF's SHA-256 matches the manifest; no zero-byte files

**Stage 2 — Parse (silver layer, the `ai_parse_document` proxy):**

```python
# poc/prep/02_parse.py
from docling.document_converter import DocumentConverter

def parse_documents(bronze_dir="data/bronze", silver_dir="data/silver"):
    """Parse each PDF with docling (layout-aware: tables, reading order,
    cell merges, footnotes). Output structured markdown + a tables JSON.
    Records: doc_id, parser_version, parse_confidence, table_count, page_count.
    Quarantine: low-confidence parses (< 0.7) go to data/silver/quarantine/
    with the reason — never a silent drop (DQX discipline)."""
```

- Input: `data/bronze/*.pdf`
- Output: `data/silver/<doc_id>.md` (parsed markdown) + `data/silver/<doc_id>.tables.json` (extracted tables) + `data/silver/quarantine/` (failed parses with reasons)
- Time: ~2-4 hours for 60 documents × ~1,000 pages (docling is ~1-2 s/page on CPU)
- Gate: parse confidence ≥ 0.7 for non-quarantined docs; table integrity check (every table has ≥2 rows and ≥2 columns); no empty extractions
- The parser version is recorded per chunk (a parser upgrade is a corpus-version bump)

**Stage 3 — Chunk (structure-based, the chunking taxonomy's rule):**

```python
# poc/prep/03_chunk.py
def chunk_documents(silver_dir="data/silver", gold_dir="data/gold",
                    target_tokens=1024, overlap_tokens=64):
    """Structure-based chunking:
    1. Split on heading hierarchy (H1, H2, H3) — each section is a chunk boundary.
    2. Within a section, split on paragraphs.
    3. Tables are ATOMIC — never split a table across chunks (chunking taxonomy
       finding: tables must stay together for retrieval to work).
    4. Captions bind to their tables.
    5. If a chunk exceeds target_tokens (1024 + 64 overlap), split on the
       nearest paragraph boundary, not mid-sentence.
    6. Verbalize metadata into a chunk header: '[doc: tb-2018-q3, page: 47,
       section: Table 3, type: table]' (the Walk&Retrieve pattern).

    Output per chunk: chunk_id, doc_id, page, section, text, token_ids,
    token_count, chunk_type (text|table|caption), provenance_sha.
    """
```

- Input: `data/silver/*.md` + `*.tables.json`
- Output: `data/gold/chunks.jsonl` (one JSON per chunk, ~50,000 lines)
- Time: ~10 min (CPU-bound, fast)
- Gate: every table is in exactly one chunk; no chunk exceeds 1024 + 64 tokens; every chunk has a verbalized metadata header; size distribution reported (chunking taxonomy finding: chunker comparisons are often size comparisons in disguise)

**Stage 4 — Embed (bge-m3, the dense vectors):**

```python
# poc/prep/04_embed.py
from FlagEmbedding import BGEM3FlagModel

def embed_chunks(gold_dir="data/gold", embeddings_dir="data/embeddings"):
    """Embed each chunk's text with bge-m3 (1024-dim, fp16).
    Batch the chunks (bge-m3 handles batches of ~32 efficiently on GPU).
    Output: data/embeddings/chunk_<id>.npy (1024-dim fp16 vector) +
    data/embeddings/vectors.bin (the concatenated vectors for FAISS).
    Records: embedder_version, embedding_dim, dtype, n_vectors."""
```

- Input: `data/gold/chunks.jsonl`
- Output: `data/embeddings/vectors.bin` (50,000 × 1024 × 2 B = 97 MiB) + `data/embeddings/metadata.json`
- Time: ~30-60 min on A10G (bge-m3 is ~100 chunks/s on GPU) or ~3-5 hours on CPU
- Gate: every chunk has an embedding; the embedding norm is in [0.9, 1.1] (bge-m3 normalizes); no NaN/Inf

**Stage 5 — Index (FAISS HNSW, the disk-resident ANN):**

```python
# poc/prep/05_index.py
import faiss

def build_index(embeddings_dir="data/embeddings", index_dir="data/index"):
    """Build a FAISS HNSW index (M=16, efConstruction=200) over the fp16 vectors.
    Output: data/index/hnsw.index (the graph + the vectors) +
    data/index/metadata.db (SQLite: chunk_id → doc_id, page, text, tokens,
    ACL, provenance — for the reranking and retrieval step).
    """
    vectors = np.memmap("data/embeddings/vectors.bin", dtype=np.float16,
                        mode="r", shape=(50000, 1024))
    index = faiss.IndexHNSWFlat(1024, 16)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64  # tuned for ~98% recall@100
    index.add(vectors.astype(np.float32))  # FAISS HNSW wants fp32 internally
    faiss.write_index(index, "data/index/hnsw.index")
    # build the SQLite metadata sidecar
    build_metadata_db("data/gold/chunks.jsonl", "data/index/metadata.db")
```

- Input: `data/embeddings/vectors.bin`
- Output: `data/index/hnsw.index` (~525 MiB — the HNSW graph + the fp32 copies of the vectors FAISS keeps internally) + `data/index/metadata.db` (~50 MiB SQLite — the chunk text, token IDs, metadata)
- Time: ~30 s build (HNSW build is fast for 50k vectors)
- Gate: index contains 50,000 vectors; a brute-force kNN on a sample of 100 queries matches HNSW's top-100 with ≥98% recall; the metadata.db joins correctly (every chunk_id in the index has a row in metadata)

### 1.3 The total dataset prep time and disk

| Stage | Time (A10G) | Disk output |
|---|---|---|
| 1. Ingest | 30 min | 8.7 GiB (PDFs) |
| 2. Parse | 2-4 hours | 1.5 GiB (markdown + tables) |
| 3. Chunk | 10 min | 0.5 GiB (chunks.jsonl) |
| 4. Embed | 30-60 min | 97 MiB (vectors) |
| 5. Index | 30 s | 575 MiB (HNSW + metadata.db) |
| **Total** | **~4-6 hours** | **~11 GiB** (dominated by raw PDFs) |

The index itself (what retrieval queries against) is ~575 MiB. The raw PDFs are kept for re-parsing on parser upgrades (the bronze layer + time travel discipline).

---

## 2. Retrieval (exactly how it works)

### 2.1 The retrieval flow (per query)

```
User question
     │
     ▼
[Embed the query with bge-m3]  →  query vector (1024-dim fp16)
     │                              (~5 ms on GPU, ~20 ms on CPU)
     ▼
[HNSW preselect on disk]       →  top-100 candidate chunk IDs
     │                              (~2 ms, in CPU RAM — the index is loaded
     │                               once at startup, ~525 MiB)
     ▼
[Load the 100 candidates'      →  100 fp16 vectors + metadata
 embeddings + metadata from     →  (~200 KiB read from disk or
 disk]                              mmap'd into CPU RAM)
     │
     ▼
[Cos sim rerank on the 100]    →  top-k (k=3) chunk IDs
     │                              (the final reranking — ~1 ms on CPU)
     ▼
[Load top-3 chunks' token IDs  →  3 × 1024 tokens = 3,072 tokens
 + text from metadata.db]          (~12 KiB, microseconds)
     │
     ▼
[Return top-3 chunks to the augmentation step]
```

### 2.2 The retrieval code

```python
# poc/retrieve.py
import faiss, numpy as np, sqlite3, torch
from FlagEmbedding import BGEM3FlagModel

class Retriever:
    def __init__(self, index_path="data/index/hnsw.index",
                 metadata_db_path="data/index/metadata.db",
                 embedder_model="BAAI/bge-m3",
                 device="cuda:0",  # GPU for the query embedding
                 ef_search=64, top_k_preselect=100, top_k_final=3):
        # Load the HNSW index into CPU RAM (once, at startup)
        self.index = faiss.read_index(index_path)
        self.index.hnsw.efSearch = ef_search
        # mmap the metadata DB
        self.meta_db = sqlite3.connect(metadata_db_path, check_same_thread=False)
        # the embedder (on GPU for the query)
        self.embedder = BGEM3FlagModel(embedder_model, use_fp16=True, device=device)
        self.top_k_preselect = top_k_preselect
        self.top_k_final = top_k_final

    def retrieve(self, query: str) -> list[dict]:
        """Returns the top-3 chunks for the query.
        Each chunk: {chunk_id, doc_id, page, section, text, token_ids,
                    score, chunk_type}."""
        # 1. Embed the query (GPU, ~5 ms)
        query_vec = self.embedder.encode([query], max_length=512,
                                          return_dense=True)["dense_vecs"]
        query_vec = query_vec.astype(np.float32)  # FAISS wants fp32

        # 2. HNSW preselect (CPU, ~2 ms) → top-100 chunk IDs
        _, preselect_ids = self.index.search(query_vec, self.top_k_preselect)

        # 3. Load the 100 candidates' vectors + metadata (disk/CPU, ~1 ms)
        #    (FAISS returns the vectors via reconstruct_batch, or we read
        #    from vectors.bin by offset)
        candidate_vectors = self.index.reconstruct_batch(preselect_ids[0])
        candidate_metadata = self._load_metadata(preselect_ids[0])

        # 4. Cos sim rerank on the 100 (CPU, ~1 ms) → top-3
        #    (bge-m3 vectors are normalized, so dot product = cos sim)
        scores = candidate_vectors @ query_vec[0]
        top3_local_idx = np.argsort(scores)[-self.top_k_final:][::-1]

        # 5. Return the top-3 chunks with their token IDs + text
        return [self._load_chunk_tokens(candidate_metadata[i]) 
                for i in top3_local_idx]

    def _load_metadata(self, chunk_ids):
        """Batch-load metadata for the candidate chunk IDs from SQLite."""
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.meta_db.execute(
            f"SELECT chunk_id, doc_id, page, section, chunk_type FROM chunks "
            f"WHERE chunk_id IN ({placeholders})", chunk_ids).fetchall()
        return rows

    def _load_chunk_tokens(self, metadata_row):
        """Load the chunk's text + pre-tokenized token_ids from the DB."""
        chunk_id = metadata_row[0]
        row = self.meta_db.execute(
            "SELECT text, token_ids FROM chunks WHERE chunk_id = ?",
            (chunk_id,)).fetchone()
        return {
            "chunk_id": chunk_id,
            "doc_id": metadata_row[1],
            "page": metadata_row[2],
            "section": metadata_row[3],
            "chunk_type": metadata_row[4],
            "text": row[0],
            "token_ids": np.frombuffer(row[1], dtype=np.int32).tolist(),
        }
```

### 2.3 The retrieval latency budget (per query)

| Step | Where | Time | Notes |
|---|---|---|---|
| Query embedding | A10G GPU | ~5 ms | bge-m3 on GPU |
| HNSW preselect (top-100) | CPU RAM | ~2 ms | The index is loaded once at startup |
| Load 100 candidate vectors | CPU RAM (mmap'd) | ~1 ms | ~200 KiB |
| Cos sim rerank (top-3) | CPU | ~1 ms | 100 × 1024-dim dot products |
| Load top-3 chunk tokens | CPU/disk | ~1 ms | ~12 KiB from SQLite |
| **Total retrieval latency** | | **~10 ms** | |

Retrieval is **~10 ms per query** — dominated by the query embedding (5 ms). The HNSW preselect and reranking are negligible. This is fast enough that retrieval is NOT the bottleneck; the model's prefill + decode is.

### 2.4 The retrieval quality gate (recall@3)

Before the PoC measures end-to-end accuracy, retrieval quality is validated independently:

```python
# poc/tests/test_retrieval_recall.py
def test_recall_at_3(retriever, gold_questions):
    """For each gold question, the ground-truth document is known.
    Measure: of the top-3 retrieved chunks, how many come from the
    correct document? (recall@3 at the document level).
    Target: ≥ 80% (the OfficeQA Pro paper's frontier agents achieved
    ~34% end-to-end accuracy; retrieval at 80%+ leaves room for the
    model to reason)."""
```

- Target: **recall@3 (document-level) ≥ 80%** on the answerable gold questions
- If below 80%, the retrieval is the bottleneck — tune `ef_search` (try 128), check the embedder, or check the chunker (tables might be split)
- This is measured BEFORE any model inference — retrieval quality is separable

---

## 3. Augmentation (exactly how it works)

### 3.1 The context contract (the augmented prompt)

Augmentation is the assembly of the retrieved chunks into the model's context. The contract is frozen — it's the cache key structure:

```
[system prompt]                                    ← pinned, near-immortal
   "You are a financial analyst. Answer the question
    based ONLY on the provided context. If the answer
    is not in the context, say 'I don't know.'"

[retrieved chunks in DOCUMENT order]               ← the augmented part
   [chunk 1: doc=tb-2018-q3, page=47, section=Table 3, type=table]
   <chunk 1 text>
   [chunk 2: doc=tb-2018-q3, page=48, section=Table 3 (cont.), type=text]
   <chunk 2 text>
   [chunk 3: doc=tb-2018-q3, page=50, section=Notes, type=text]
   <chunk 3 text>

[conversation turns]                               ← empty for single-turn PoC

[the question]
   "What was the total revenue reported in Table 3
    for fiscal year 2018?"
```

**The two rules from the chunking taxonomy (arXiv:2602.16974):**
1. **Chunks enter the corpus block in document order** (preserves longest common prefixes — cache-friendly). Relevance ordering tears prefixes apart and pays re-prefill.
2. **Reranking happens across documents, never within one.** If the top-3 are all from the same document, they stay in page order. If they're from different documents, the documents are ordered by score, but within each document, page order is preserved.

### 3.2 The augmentation code

```python
# poc/augment.py
def build_context(system_prompt: str, retrieved_chunks: list[dict],
                  question: str) -> tuple[list[int], dict]:
    """Assemble the context contract. Returns (token_ids, provenance).

    The token_ids are what get fed to the model. The provenance records
    which chunks went where (for the cache namespace key)."""

    # 1. Sort the retrieved chunks: by document (score order), then by
    #    page within each document (document order — cache-friendly)
    chunks_by_doc = group_by_doc_and_sort(retrieved_chunks)

    # 2. Build the token sequence
    token_ids = []
    chunk_boundaries = []  # for the cache namespace key
    for doc_id, doc_chunks in chunks_by_doc.items():
        for chunk in doc_chunks:
            # the verbalized metadata header
            header = f"[doc={chunk['doc_id']}, page={chunk['page']}, " \
                     f"section={chunk['section']}, type={chunk['chunk_type']}]\n"
            header_ids = tokenize(header)
            # the chunk's pre-tokenized text
            chunk_ids = chunk['token_ids']
            token_ids.extend(header_ids)
            chunk_boundaries.append({
                "chunk_id": chunk["chunk_id"],
                "start": len(token_ids),
                "end": len(token_ids) + len(chunk_ids),
            })
            token_ids.extend(chunk_ids)
            token_ids.extend(tokenize("\n\n"))  # chunk separator

    # 3. Prepend the system prompt
    sys_ids = tokenize(system_prompt + "\n\n")
    token_ids = sys_ids + token_ids

    # 4. Append the question
    q_ids = tokenize("\n\nQuestion: " + question + "\nAnswer:")
    token_ids = token_ids + q_ids

    provenance = {
        "system_prompt_hash": hash(system_prompt),
        "chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        "chunk_boundaries": chunk_boundaries,
        "corpus_version": CORPUS_VERSION,  # SHA of (parser, chunker, embedder) versions
        "context_class": hash(system_prompt),
    }
    return token_ids, provenance
```

### 3.3 The augmentation gates

| Gate | What it checks | Target |
|---|---|---|
| Document order | Chunks from the same document are in page order | 100% (hard gate) |
| No mid-table splits | A table chunk is atomic (no table spans two chunks) | 100% (hard gate, enforced at chunk time) |
| Metadata verbalized | Every chunk has a `[doc=..., page=..., section=..., type=...]` header | 100% |
| Context length | Total tokens (sys + chunks + question) ≤ 8,192 | 100% (truncate to top-2 if exceeded) |
| Provenance recorded | Every chunk's position in the context is in the provenance dict | 100% (for the cache namespace key) |

### 3.4 The augmentation → cache namespace

The augmentation's output (the provenance dict) feeds directly into the cache namespace:

```python
# the cache namespace for this augmented context
namespace = CacheNamespace(
    model_checkpoint_id=MODEL_SHA,
    quant_recipe_signature=RECIPE_SHA,  # SHA of metadata.json's auto_decision ledger
    context_class=provenance["context_class"],  # hash of system prompt
    corpus_version=provenance["corpus_version"],  # SHA of (parser, chunker, embedder)
)
```

A different system prompt, a different corpus version, or a different model checkpoint → different namespace → no cross-namespace cache reuse (KVShareArena's rule).

---

## 4. The benchmark protocol (exactly how we measure)

This is the operational form of the v3 PoC's measurement plan, grounded in the 2026 audit-wave papers. The protocol is what makes the numbers falsifiable.

### 4.1 The benchmark dimensions

| Dimension | What it measures | How |
|---|---|---|
| **Accuracy** | OfficeQA gold-answer correctness | Exact-match + numeric-tolerance (±1%) on the answerable questions |
| **Retrieval quality** | recall@3 (document-level) | Compared to the gold document references |
| **Throughput** | tok/s (decode), queries/s (end-to-end) | Wall-clock, n=50 questions, warm caches |
| **Latency** | TTFT (time to first token), ITL (inter-token latency) | Per-query, with percentiles (p50, p95, p99) |
| **VRAM** | Peak GiB during inference | `torch.cuda.max_memory_allocated` |
| **Cache hit rate** | % of prefill tokens saved by the two global caches | Per-cache (A: KV prefix, B: recurrent-state) + crossover |
| **Restoration error** | The fp16 round-trip drift (the toy's finding 3, on the real model) | ‖logits_fresh - logits_restored‖ / ‖logits_fresh‖ |
| **Staleness envelope** | Accuracy as a function of doc-edit distance | The Contiguity paper's protocol |

### 4.2 The meaningful-query filter (mandatory, the BoxOffice gate)

Before any accuracy claim, the meaningful-query filter removes three classes of questions (BoxOffice, arXiv:2609.31415: 42% of F1 gains are metric artifacts):

```python
# poc/benchmark/meaningful_query_filter.py
def apply_filter(questions, model, retriever, n_questions):
    """Remove three classes:
    1. Questions the cache-free reference run fails anyway (model capability)
    2. Questions answerable without context (world knowledge — test by asking
       the model with NO retrieved chunks; if it answers correctly, drop)
    3. Low-information yes/no or either/or questions (carry almost no signal)

    Returns (filtered_questions, cut_list). The cut list is itself a finding."""

    cache_free_answers = run_without_context(model, questions)
    world_knowledge_correct = []
    for q, a in zip(questions, cache_free_answers):
        if is_correct(a, q.gold_answer):
            world_knowledge_correct.append(q)  # class 2: drop

    low_information = [q for q in questions if is_yes_no_or_either_or(q)]
    # class 3: drop

    cache_free_failures = [q for q, a in zip(questions, cache_free_answers)
                           if not is_correct(a, q.gold_answer)]
    # class 1: drop (the cache can't fix a model-capability failure)

    filtered = [q for q in questions
                if q not in world_knowledge_correct
                and q not in low_information
                and q not in cache_free_failures]
    return filtered, {"world_knowledge": len(world_knowledge_correct),
                      "low_information": len(low_information),
                      "cache_free_failure": len(cache_free_failures)}
```

**The cut list is reported as a finding.** If >50% of questions are cut, the PoC expands the question set to retain statistical power.

### 4.3 The benchmark configurations (the A/B/C/D sweep)

Four configurations, measured on the same filtered question set:

| Config | Cache A (KV prefix) | Cache B (recurrent state) | Fine-tune | What it tests |
|---|---|---|---|---|
| **A: no cache** | off | off | off | The floor — every query re-prefills |
| **B: KV only** | on | off | off | The vLLM/SGLang standard |
| **C: state only** | off | on | off | The K3 bet |
| **D: crossover** | on (prefixes < 816 tok) | on (prefixes ≥ 816 tok) | off | The v3 proposal |
| **E: crossover + fine-tune** | on | on | on (cache-aware) | The full proposal; conditional on the fine-tune not worsening restoration error |

```python
# poc/benchmark/run_benchmark.py
def run_benchmark(model, retriever, questions, configs, device):
    """For each config, run all filtered questions and record:
    - accuracy (exact-match + numeric-tolerance)
    - retrieval recall@3
    - TTFT, ITL (p50, p95, p99)
    - throughput (tok/s, queries/s)
    - VRAM peak
    - cache hit rate (per cache, per boundary)
    - restoration error (if cache B is on)

    Each config runs in a FRESH process (no cache state leaks between configs).
    The order of questions is randomized per config (detects any
    order-dependent cache warming)."""
    results = {}
    for config in configs:
        # fresh process: load model, load caches, run questions
        proc_result = subprocess_run("poc/benchmark/run_one_config.py",
                                     args=[config, questions_file])
        results[config] = parse_report(proc_result.report_path)
    return results
```

### 4.4 The staleness envelope (the Contiguity protocol)

For the doc-edit axis (the most important for RAG):

```python
# poc/benchmark/staleness_envelope.py
def measure_doc_edit_envelope(model, retriever, questions, cache_backend):
    """For each question, simulate a document edit by replacing one chunk
    in the retrieved set. Measure:
    - accuracy before edit (baseline)
    - accuracy after edit (with edit-local repair)
    - accuracy after edit (with full re-prefill — the upper bound)
    - repair time (ms) vs re-prefill time (ms) — the 13-21× ratio

    Sweep the edit distance (tokens between the edit and the answer):
    - edit at the start of the chunk (far from answer)
    - edit in the middle
    - edit at the end (adjacent to answer — the Contiguity paper's
      'adjacency' condition where edit-local works best)

    Produce an envelope curve: accuracy vs edit distance, for both
    edit-local repair and full re-prefill."""
```

- The envelope curve is the unit of measurement, not a point estimate (BoxOffice: F1 flips 0.98 ↔ 0.00 on staleness)
- Target: edit-local repair within 13-21× of re-prefill (the Contiguity paper's measurement; the PoC validates it)

### 4.5 The two-oracle diagnostic (the LRU paper's protocol)

Before building anything beyond LRU:

```python
# poc/benchmark/two_oracle.py
def run_two_oracle_diagnostic(cache_backend, trace_sample):
    """Sample 1000 requests from the benchmark trace. Compute:
    - Belady (the offline optimal: evict the block whose next use is farthest)
    - BeladyCompute (Belady weighted by recompute cost)

    If the gap between LRU and Belady is <5%, do not build compute-aware
    eviction — plain LRU is within 5% of optimal. (The LRU paper's finding:
    14 sophisticated policies fail to beat LRU under agentic load.)"""
```

- If the gap is <5%, ship LRU. This is the v3 decision record's rule, made operational.

### 4.6 The benchmark report schema

```
poc/reports/
├── retrieval_recall.json              # recall@3 (document-level) on the gold set
├── meaningful_query_filter.json      # the cut list (itself a finding)
├── benchmark_config_A_no_cache.json   # accuracy, TTFT, ITL, throughput, VRAM
├── benchmark_config_B_kv_only.json
├── benchmark_config_C_state_only.json
├── benchmark_config_D_crossover.json
├── benchmark_config_E_crossover_finetuned.json
├── staleness_envelope_doc_edit.json   # accuracy vs edit distance, edit-local vs re-prefill
├── two_oracle_diagnostic.json         # Belady vs BeladyCompute gap
├── restoration_error_pre_finetune.json
├── restoration_error_post_finetune.json
└── poc_verdict.json                   # the final go/no-go with all numbers
```

### 4.7 The verdict criteria (the go/no-go gates)

| Gate | Criterion | If failed |
|---|---|---|
| Retrieval quality | recall@3 ≥ 80% | Tune `ef_search` or check the chunker; do not proceed to model benchmarks |
| Accuracy floor | Config A (no cache) accuracy ≥ 10% on the filtered set | If below, the model is too weak for the workload — report and stop |
| Cache benefit | Config D (crossover) throughput ≥ 1.5× Config A (no cache) | If below, the cache doesn't pay off on this workload — report |
| Cache accuracy | Config D accuracy ≥ Config A accuracy - 2% | If the cache destroys accuracy, the restoration error is too high — report |
| Fine-tune | Restoration error post-fine-tune ≤ pre-fine-tune | If worsened (the toy's prediction), abandon the fine-tune; use Config D without it |
| Edit-local repair | Repair time ≤ 1/10 of re-prefill time | If worse, the edit-local policy is wrong — report |
| LRU sufficiency | Belady vs LRU gap < 5% | If >5%, consider compute-aware eviction (rare) |

---

## 5. The end-to-end PoC timeline

| Phase | What | Time | Output |
|---|---|---|---|
| **P0. Prep** | Stages 1-5 of dataset prep (ingest, parse, chunk, embed, index) | 4-6 hours | `data/index/hnsw.index` + `data/index/metadata.db` (575 MiB) |
| **P1. Retrieval validation** | recall@3 on the gold set; tune `ef_search` if needed | 1 hour | `retrieval_recall.json` |
| **P2. Meaningful-query filter** | Run the filter; record the cut list | 1 hour | `meaningful_query_filter.json` |
| **P3. Benchmark sweep** | Run configs A, B, C, D on the filtered set | 2 hours | 4 benchmark reports |
| **P4. Fine-tune (conditional)** | The cache-aware W10 LUT fine-tune (500 steps) | 4-8 hours | `restoration_error_post_finetune.json` |
| **P5. Config E benchmark** | If the fine-tune passed the restoration-error gate | 1 hour | `benchmark_config_E.json` |
| **P6. Staleness envelope** | The doc-edit envelope (edit-local vs re-prefill) | 2 hours | `staleness_envelope_doc_edit.json` |
| **P7. Two-oracle diagnostic** | Belady vs BeladyCompute on a trace sample | 1 hour | `two_oracle_diagnostic.json` |
| **P8. Verdict** | Assemble `poc_verdict.json` | 30 min | The go/no-go |
| **Total** | | **~2 days** | The full PoC |

---

## 6. Summary — the exact answers

### 6.1 Dataset preparation (exactly)

1. **Ingest:** download `databricks/officeqa` + the referenced Treasury Bulletin PDFs → `data/bronze/` (~8.7 GiB PDFs).
2. **Parse:** docling (layout-aware, the `ai_parse_document` proxy) → `data/silver/*.md` + `*.tables.json` (~1.5 GiB, ~2-4 hours).
3. **Chunk:** structure-based (headings + paragraphs + atomic tables, 1024 tok/chunk + 64 overlap) → `data/gold/chunks.jsonl` (~50,000 chunks).
4. **Embed:** bge-m3 (1024-dim fp16, GPU) → `data/embeddings/vectors.bin` (~97 MiB, ~30-60 min).
5. **Index:** FAISS HNSW (M=16, efSearch=64) + SQLite metadata → `data/index/hnsw.index` + `metadata.db` (~575 MiB, ~30 s build).

**Total prep: ~4-6 hours, ~11 GiB disk (dominated by raw PDFs). The index itself is ~575 MiB.**

### 6.2 Retrieval (exactly)

1. Embed the query with bge-m3 (GPU, ~5 ms) → 1024-dim fp16 vector.
2. HNSW preselect on the disk-resident index (CPU RAM, ~2 ms) → top-100 candidate chunk IDs.
3. Load the 100 candidates' vectors + metadata (mmap, ~1 ms) → 100 fp16 vectors.
4. Cos sim rerank on the 100 (CPU, ~1 ms) → top-3 chunk IDs.
5. Load the top-3 chunks' token IDs + text from SQLite (~1 ms) → 3 × 1024 tokens.
6. Return to augmentation.

**Total retrieval: ~10 ms per query. Retrieval quality gate: recall@3 (document-level) ≥ 80%.**

### 6.3 Augmentation (exactly)

1. Group the top-3 chunks by document (score order across docs).
2. Within each document, sort by page (document order — cache-friendly, preserves longest common prefixes).
3. Verbalize metadata into a chunk header: `[doc=..., page=..., section=..., type=...]`.
4. Concatenate: system prompt + chunks (with headers, in document order) + question.
5. Record provenance (chunk IDs, positions, corpus version) → the cache namespace key.

**The context contract is frozen: system prompt + corpus block (document order) + question. The cache key is the provenance hash.**

### 6.4 Benchmark (exactly)

1. **Apply the meaningful-query filter** (BoxOffice gate) — drop model-capability failures, world-knowledge questions, low-information yes/no. Record the cut list.
2. **Run 5 configurations** on the filtered set: no cache, KV only, state only, crossover, crossover+fine-tune.
3. **Measure per config:** accuracy (exact-match + numeric-tolerance), retrieval recall@3, TTFT/ITL (p50/p95/p99), throughput (tok/s), VRAM peak, cache hit rate per cache.
4. **Measure the staleness envelope** (doc-edit axis): accuracy vs edit distance, edit-local repair vs re-prefill (the Contiguity protocol).
5. **Run the two-oracle diagnostic** (Belady vs BeladyCompute) — if gap <5%, ship LRU.
6. **Measure restoration error** before and after the fine-tune — if worsened, abandon the fine-tune.
7. **Produce `poc_verdict.json`** with all numbers and the go/no-go.

**The benchmark is falsifiable: every gate has a target, and failure has a defined response. Total PoC time: ~2 days on A10G.**
