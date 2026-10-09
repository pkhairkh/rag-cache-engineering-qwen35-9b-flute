# PROPOSAL — PoC: Cache-Engineering RAG on Qwen3.5-9B (FLUTE idxN), OfficeQA on A10G

> **Status:** proposed — proof-of-concept — feasibility + accuracy benchmarks
> **Companion to:** `PROPOSAL-COMPANY-RAG.md` (v1.1) — the production proposal. This document is the **smaller, faster, measurable** version: a PoC scoped to one box (A10G), one dataset (`databricks/officeqa`), and the question *"does the cache-engineering thesis hold when measured, not assumed?"*
> **Decision owner:** repo owner · **Scope:** one PoC, three measurement rounds, six weeks of box time. The production proposal's four-lane architecture is *compressed* into a single-machine script that exercises the same primitives: corpus (subset of OfficeQA's Treasury Bulletins), permission (omitted — OfficeQA's corpus is single-tenant), serving (Qwen3.5-9B native hybrid + FLUTE idxN, with the linear-attention state matrix as the cache asset), evaluation (the OfficeQA gold answers + the meaningful-query filter).
> **Provenance:** every claim about the repo's existing infrastructure references a file path in `qwen3_5_9B_flute_qlora_v1.3`. Every claim about the literature references an arXiv paper with its abstract read.

---

## 0. The question, and the answer

> *"We need a PoC with benchmarks for accuracy and feasibility. What works? How can we get a small PoC on OfficeQA on the A10G machine? Read the literature — what does the latest arxiv say?"*

**Answer:** build a **single-machine, three-round PoC** that measures, in this order:

1. **Round 1 — Feasibility floor.** Can Qwen3.5-9B (dense FP16 vs FLUTE idxN W4+r32) even run on A10G under the OfficeQA workload, and at what accuracy/throughput/VRAM cost? This is the *floor* — if the dense model OOMs at OfficeQA's context length, or if the W4 quantization drops accuracy below the OfficeQA frontier-agent average of 34.1%, the cache-engineering thesis is moot for this hardware.
2. **Round 2 — Cache-engineering delta.** With the W4 model as the base, measure the delta from (a) **no cache** (every request re-prefills), (b) **vLLM-style prefix cache** (the standard, full-attention KV reuse), and (c) **linear-attention state snapshot** (the proposal's cache asset — 576 KiB per session, reinstalled at the 8 full-attention block boundaries). The delta is measured on accuracy (OfficeQA gold answers), throughput (tok/s), latency (TTFT, ITL), and VRAM (GiB resident).
3. **Round 3 — Staleness envelope.** With cache-engineering active, measure the **staleness envelope** on the four axes of the production proposal: doc-edit (the OfficeQA corpus is versioned — simulate edits), role-flip (system prompt change), checkpoint-change (re-quantization), permission-flip (synthetic ACL — out of scope for OfficeQA's single-tenant corpus, but we measure the **structural invariant**: that a permission flip is a query-time filter, not a cache mutation). This is the round that produces the envelope curves the production proposal gates every release on.

The PoC exists to **fail fast** on the production proposal's riskiest assumption: that the cache asset (576 KiB linear-attention state per session) is the right object to cache, on this hardware, at this scale, with this model. If the measured delta between vLLM prefix cache and linear-attention state cache is *not* meaningfully in favor of the state cache, the production proposal's serving lane is reconsidered before any Databricks/SharePoint/Unity Catalog work is commissioned.

---

## 1. The literature, read

This section is the contract between the PoC and the field. Every paper cited here was read (abstract + key claims) during the writing of this proposal. The PoC's measurement protocol is derived from these papers' findings, not assumed alongside them.

### 1.1 OfficeQA Pro — the benchmark, and the measured frontier

**Paper:** [arXiv:2603.08655](https://arxiv.org/abs/2603.08655) — *OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning* (Opsahl-Ong et al., Databricks AI Research, submitted 9 Mar 2026).

**Dataset on HuggingFace:** [`databricks/officeqa`](https://huggingface.co/datasets/databricks/officeqa) (the v1 corpus; `databricks/officeqa-pro-v2` is the v2 release with 90 questions over 120,000 pages of U.S. Treasury Accounts of Receipts and Expenditures). License: CC-BY-SA-4.0. Format: CSV; pandas-loadable.

**Measured findings the PoC inherits:**

| # | Finding | What it decides in the PoC |
|---|---|---|
| L-1 | The corpus is 89,000 pages of U.S. Treasury Bulletins spanning nearly 100 years, with over 26 million numerical values — dense financial tables, charts, and narrative text. | The PoC's "documents not in the best state" condition is met *for free*: OfficeQA's corpus is the same shape as the production proposal's SharePoint target. No synthetic-poor-state document generation is needed. |
| L-2 | Frontier LLMs (Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro Preview) score **<5% accuracy on parametric knowledge** and **<12% with web access** on OfficeQA Pro's 133 questions. | The PoC's "cache-free baseline" is *expected* to be low (Qwen3.5-9B is not a frontier model). The bar is not "match GPT-5.4"; the bar is "measure the delta from cache engineering on a fixed, smaller model." |
| L-3 | Frontier agents with direct corpus access score **34.1% average** on OfficeQA Pro; over half of questions fail. | This is the production proposal's expectation-setting number. The PoC must report its measured accuracy against this baseline — if Qwen3.5-9B W4 lands at, say, 12–18%, that is *expected* for a 9B model, not a failure of cache engineering. |
| L-4 | Providing agents with a structured document representation produced by Databricks' `ai_parse_document` yields a **+16.1% average relative performance gain** across agents. | The PoC's corpus lane must use `ai_parse_document` (or a CPU-side proxy: marker, docling, or unstructured) for the layout-aware parse. Skipping this is the largest single accuracy lever the PoC can pull. The 16.1% is the *measured* ceiling of that lever; the PoC targets at least half of it (8% relative) on the smaller model. |
| L-5 | OfficeQA Pro's questions require "precise document parsing, retrieval, and analytical reasoning across both unstructured text and tabular data." | The PoC's retrieval must handle tables (not just text chunks). The chunker must keep tables atomic — the chunking taxonomy's rule, see §1.3. |

**The PoC uses `databricks/officeqa` (the v1 dataset), not `officeqa-pro-v2`.** v1 is larger and CC-BY-SA-4.0; v2 has 90 questions and is the more recent release. The PoC's choice is driven by one factor: v1 is the dataset the existing repo's eval infrastructure (`scripts/eval_*.py`) is closest to consuming — CSV-shaped, pandas-loadable, no agent loop required. v2's agent harness is out of scope for a 6-week PoC on one box.

### 1.2 KV-cache reuse — the 2026 audit wave

Three papers from September 2026 form the audit wave the production proposal cites as its evaluation lane's foundation. The PoC's Round 3 (staleness envelope) is *literally* the measurement protocol these papers prescribe.

**Paper A:** [arXiv:2609.31415](https://arxiv.org/abs/2609.31415) — *Evaluating the accuracy of KV cache reuse techniques* (the "BoxOffice" paper, found via search).

| Finding | What it decides |
|---|---|
| The same KV-reuse method can score an **F1 of 0.98 or 0.00** depending only on the context in which the reuse occurs. | Round 3's staleness measurement reports *envelope curves* (accuracy as a function of staleness axis), not point estimates. A single F1 number is forbidden in the PoC's output. |
| **42% of reported F1 gains** from KV reuse are **metric artifacts**, not real accuracy improvements. | Round 2's cache-vs-no-cache delta must apply the **meaningful-query filter** before any accuracy claim: drop questions the cache-free run fails anyway (model capability, not cache), questions answerable without context (world knowledge), and low-information yes/no questions. The filter's cut list is itself a PoC finding. |

**Paper B:** [arXiv:2609.10266](https://arxiv.org/abs/2609.10266) — *KV-Cache Reuse Across Contexts and Model Checkpoints* (KVShareArena).

| Finding | What it decides |
|---|---|
| Free position rotation recovers only **50–66% of the gap** to the cache-free run when a checkpoint changes. | Round 3's checkpoint-change axis: the PoC measures a re-quantization (e.g., `--auto-cos 0.9995` → `0.9990`) as a namespace switch, not a rotation. Refuse cross-checkpoint cache reuse outright. |
| Unrepaired cross-context reuse can score **worse than no cache at all**. | Round 2's admission policy: when in doubt, re-prefill. The PoC's baseline (no cache) is the lower bound; if cache-with-staleness scores below baseline, that is the finding, not a bug. |

**Paper C:** [arXiv:2609.17983](https://arxiv.org/abs/2609.17983) — *Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits* (Mao, Mackey, Lin, submitted 16 Sep 2026).

| Finding | What it decides |
|---|---|
| A contiguous edit-local window recovers **at least 0.94 of the post-edit answer margin** and substantially outperforms attention-based, KV-deviation, and structural selectors. | Round 3's doc-edit axis: the repair policy is *edit-local contiguous*, not importance-based. The PoC simulates a document edit by replacing one chunk in the corpus block and measures the repair cost (re-prefill of the contiguous neighborhood) vs. the full re-prefill. |
| Repair is **13–21× faster** than full re-prefill. | Round 3's repair ledger: per-document repair cost is a line item. The PoC records `repair_ms / reprefill_ms` per edit; the 13–21× figure is the source's measurement, the PoC's measurement is the validation. |
| The edit-local advantage **disappears when the answer-bearing text moves downstream** of the edit. | Round 3's known limitation: the PoC reports the *distribution* of repair effectiveness, not just the mean. A long tail (answer moves downstream) is expected and reported honestly. |

### 1.3 Chunking — the taxonomy paper

**Paper:** [arXiv:2602.16974](https://arxiv.org/abs/2602.16974) — *Beyond Chunk-Then-Embed: A Comprehensive Taxonomy and Evaluation of Document Chunking* (found via search, February 2026).

| Finding | What it decides |
|---|---|
| Simple **structure-based methods outperform LLM-guided alternatives** for corpus-wide retrieval; LLM-guided chunking pays only for intra-document tasks. | The PoC's chunker is structure-based only: heading hierarchy, paragraphs, tables atomic. No LLM in the chunker. |
| **Late chunking hurts needle queries** (questions requiring a specific fact from a specific page). | The PoC uses pre-retrieval chunking (chunk → embed → index → retrieve at query time), not late chunking (embed full doc → retrieve → chunk at the model). |
| **Document order preserves longest common prefixes**; relevance order tears prefixes apart. | Round 2's corpus block assembly: chunks enter the context in *document order* within a document; reranking happens *across documents*, never within one. This is the cache-friendly order. |
| Chunker comparisons are often **size comparisons in disguise**. | Round 1's chunker comparison controls for resulting chunk size and reports the size distribution. A naive "structure-based beat semantic" claim without size control is forbidden. |

### 1.4 Prefix cache eviction — the LRU verdict

**Paper:** [arXiv:2609.28870](https://arxiv.org/abs/2609.28870) — *When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse* (Liu, Yu, Yang, submitted 24 Sep 2026, revised 1 Oct 2026).

| Finding | What it decides |
|---|---|
| 14 sophisticated eviction policies **fail to beat LRU** under agentic load across two production traces. | Round 2's eviction policy is plain LRU. No learned eviction, no frequency-based policy, no analytic policy. |
| Prefix reuse is dominated by **the regular pacing of active sessions**, making recency unusually predictive. | Round 2's session-affinity scheduling: a conversation returns to the replica that already holds its prefix. The PoC uses consistent hashing with a pre-assigned secondary. |
| Effective prefix-cache management should retain recency as its foundation while selectively adding **quick demotion for one-hit prefixes**, **compute-aware partial eviction for expensive misses**, and **capacity-dependent eviction granularity**. | Round 2's *optional* enhancements, measured against plain LRU as the baseline. The PoC reports the delta; if it is <5%, plain LRU ships. |
| The paper introduces the **compute-savings ratio** and two offline oracles (Belady, BeladyCompute). | Round 2's two-oracle diagnostic: sample a trace, compute the Belady vs BeladyCompute gap. If the gap is small, do not build compute-aware eviction. This is the same diagnostic the production proposal prescribes; the PoC validates it on the smaller workload. |

### 1.5 Adjacent domains — what the adjacent literature says

The PoC does not live in isolation. Three adjacent-domain findings shape the design:

**Adjacent A — Speculative KV cache reuse for RAG serving** ([ACL 2026](https://aclanthology.org/2026.acl-long.859), found via search). Reduces TTFT by 2.17–3.95× and increases throughput by 2.7–5.2× over full KV recomputation, with negligible accuracy loss. **What it decides:** the PoC's Round 2 cache hit rate target is *not* 100% — even speculative reuse (which is approximate) buys 2.7–5.2× throughput. The PoC's hit-rate target is "the rate at which the throughput delta matches the source's measurement" — a calibration, not a maximum.

**Adjacent B — RelayCaching** ([ICML 2026](https://icml.cc/virtual/2026/poster/66638), found via search). Training-free inference method that directly reuses decoding-phase KV caches from previous agents in subsequent agents. **What it decides:** the PoC's multi-turn (agentic) measurement reuses the *decoding-phase* KV, not just the prefill KV. The production proposal's "session-paced reuse" claim is validated by measuring multi-turn OfficeQA sessions, not single-turn queries.

**Adjacent C — PatchKV** ([arXiv:2609.26219](https://arxiv.org/html/2609.26219v1), found via search). Efficient KV cache recovery for dynamically edited documents. **What it decides:** the PoC's Round 3 doc-edit repair is benchmarked against PatchKV-style transport-based recovery, not just against full re-prefill. If transport-based recovery (move KV from old position to new position) beats edit-local contiguous repair on the OfficeQA corpus, the production proposal's edit-local rule is reconsidered.

**Adjacent D — Hybrid attention on NPUs** ([arXiv:2609.32114](https://arxiv.org/html/2609.32114v1), found via search). Hybrid attention models (Qwen3.5, Kimi series) on NPUs. **What it decides:** the PoC's measurement of the linear-attention state mechanics is *not* GPU-specific in principle — the same cache asset would apply on NPU. The PoC reports the A10G measurement; the NPU transferability is a noted future-work item, not a PoC deliverable.

### 1.6 What the literature does NOT settle — the PoC's open questions

The literature establishes the *protocol* (envelope curves, meaningful-query filter, edit-local repair, LRU baseline). It does **not** establish:

1. **Whether the linear-attention state matrix (576 KiB per session) is a *better* cache asset than the full-attention KV** at the OfficeQA workload's context lengths (likely 4k–16k tokens, not 1M). The Kimi Linear paper measures the 6× decode speedup at 1M context; at OfficeQA's context lengths, the linear-attention state's advantage may be smaller. **The PoC measures this directly.**
2. **Whether the FLUTE idxN W4 quantization preserves enough accuracy** on Qwen3.5-9B for the OfficeQA workload to make the cache engineering meaningful. The repo's existing measurement (`reports/greedy_equivalence_idx4.json`) shows +3.52% PPL on WikiText-2 — but that is perplexity, not RAG accuracy. **The PoC measures this directly.**
3. **Whether the A10G's 24 GiB is enough** to run the W4 model + a meaningful cache pool + the OfficeQA workload's context. The handover log records OOMs in the palettizer; the serving-time budget is unmeasured. **The PoC measures this directly.**

These three open questions are the PoC's reason to exist. The production proposal *assumes* favorable answers; the PoC *measures* them.

---

## 2. The PoC system

One machine, one dataset, one model, three rounds. The architecture is the production proposal's four lanes *compressed* into a single Python process with on-disk intermediates.

```
                ┌──────────────────────────────────────────────────────┐
                │         A10G (single box, 24 GiB, sm_86)              │
                │                                                      │
   ┌────────┐   │   ┌──────────────────────────────────────────────┐   │
   │ OfficeQA│──►│   │ Round 1: Feasibility floor                    │   │
   │ HF data │   │   │  • dense FP16 baseline                       │   │
   │ (CSV)   │   │   │  • FLUTE idxN W4+r32 baseline                 │   │
   └────────┘   │   │  • measure: VRAM, throughput, OfficeQA acc     │   │
                │   └──────────────────────────────────────────────┘   │
                │   ┌──────────────────────────────────────────────┐   │
                │   │ Round 2: Cache-engineering delta              │   │
                │   │  • no-cache (re-prefill every request)       │   │
                │   │  • vLLM prefix cache (full-attn KV reuse)     │   │
                │   │  • linear-attn state snapshot (proposal)    │   │
                │   │  • measure: TTFT, ITL, tok/s, acc, VRAM       │   │
                │   └──────────────────────────────────────────────┘   │
                │   ┌──────────────────────────────────────────────┐   │
                │   │ Round 3: Staleness envelope                   │   │
                │   │  • doc-edit (chunk replacement)               │   │
                │   │  • role-flip (system prompt change)           │   │
                │   │  • checkpoint-change (re-quantization)        │   │
                │   │  • permission-flip (synthetic, structural)    │   │
                │   │  • measure: envelope curves on 4 axes         │   │
                │   └──────────────────────────────────────────────┘   │
                └──────────────────────────────────────────────────────┘
```

### 2.1 What the PoC reuses (everything in the existing repo)

The existing `qwen3_5_9B_flute_qlora_v1.3` repo ships the **entire inference and measurement stack**. The PoC is a *thin orchestrator* on top of it; it does not reinvent any of these:

| Existing component | File in repo | PoC role |
|---|---|---|
| Dense model loader | `scripts/eval_common.py::load_dense_fp16` | Round 1 baseline arm |
| Palettized model loader | `scripts/eval_common.py::load_quant_model(artifacts_dir, model, device, residual=True, forward="kernel", heads_dir=...)` | Round 1 quantized arm; Round 2 cache arms |
| Memory release | `scripts/eval_common.py::release_model_memory` | Between arms (sequential residency) |
| Greedy decode with KV cache | `scripts/eval_greedy_match.py::greedy_decode` | Round 1 generation; Round 2 no-cache arm |
| CUDA-graph decode | `scripts/eval_greedy_match.py::greedy_decode_dispatch(backend="auto")` | Round 1/2 hot path (the W23+ graphs path) |
| WikiText-2 PPL | `scripts/eval_ppl.py::evaluate_nll` | Round 1 quality sanity check (the repo's existing +3.52% PPL figure) |
| Energy & throughput harness | `scripts/measure_energy.py::EnergyMeasurement` | Round 1/2 throughput, energy, VRAM measurement |
| Per-document paired probe | `scripts/o1_baseline_check.py::score_docs` + `paired_diff` | Round 3 doc-edit repair paired measurement |
| Calibration capture | `scripts/calibrate_real_text.py::CalibrationCapture` | Pre-PoC: re-palettize if needed with A10G-tuned knobs |
| Palettizer | `scripts/palettize_qwen3_5_9b.py` | Pre-PoC: produce the W4+r32 artifacts |
| Modeling (vendored Qwen3.5) | `scripts/modeling.py` | The forward pass; both `linear_attention` and `full_attention` layer types |
| Linear-attention fla wiring | `scripts/modeling.py::_fla_resolve` (W29) | The linear-attention state mechanics — the cache asset's producer |
| SM86 flash attention | `scripts/attn_sm86.py` | The 8 full-attention layers' KV — the secondary cache |
| VRAM ledger | `scripts/vram_ledger.py` | Extend with a `cache_pool` tier |
| GPU contract gate | `scripts/check_gpu_contract.py` | Pre-PoC: confirm A10G is on the allowlist |
| Atomic JSON dump | `scripts/eval_common.py::atomic_json_dump` | Every PoC report |
| The 32 deterministic prompts | `scripts/eval_greedy_match.py::PROMPTS` | Round 1 sanity check (bit-exact reproduction of `reports/greedy_equivalence_idx4.json`) |

**What the PoC does NOT reinvent:** the loader, the QLoRA wrapper, the kernel dispatch, the AWQ compensation, the chunked-CE PPL, the OOM ladder, the greedy compare, the paired-diff stats, the atomic JSON writer. All of these are gated by permanent CPU tests (`tests/test_eval_common.py`'s one-definition census); any new code that re-implements them will fail the census.

### 2.2 What the PoC adds (three small scripts)

The PoC adds **three new scripts** under a new `poc/` directory in the new repo. Each is small (<400 lines), tested on CPU first (`poc/tests/`), and built on top of the existing eval plane.

#### 2.2.1 `poc/officeqa_loader.py` — the OfficeQA corpus adapter

**Role:** load `databricks/officeqa` from HuggingFace, parse the Treasury Bulletin documents (with a CPU-side proxy for `ai_parse_document`), chunk them structure-based, embed them, and produce the gold question set.

**Why a CPU-side proxy for `ai_parse_document`:** `ai_parse_document` is a Databricks SQL function; it requires a Databricks workspace. The PoC runs on a single A10G box, not on Databricks. The proxy is one of: **marker** (MIT, fast, layout-aware), **docling** (IBM, Apache 2.0, table-aware), or **unstructured** (Apache 2.0, the industry default). The PoC picks **docling** for its table-extraction quality — OfficeQA's corpus is table-heavy, and the L-5 finding (tables must be atomic) makes table extraction the chunker's binding constraint.

**Public API:**
```python
# poc/officeqa_loader.py
def load_officeqa_corpus(
    hf_dataset: str = "databricks/officeqa",
    split: str = "test",          # the v1 dataset's evaluation split
    cache_dir: str = "~/.cache/officeqa",
    parser: str = "docling",       # or "marker", "unstructured"
    chunker: str = "structure",    # structure-based only (L-3 chunking taxonomy)
    target_chunk_tokens: int = 512,
    embedder: str = "BAAI/bge-m3",  # multilingual, 8K context, good table handling
) -> OfficeQACorpus:
    """Returns the parsed, chunked, embedded corpus + the gold QA pairs."""

@dataclass
class OfficeQACorpus:
    documents: List[OfficeQADocument]   # parsed, with provenance
    chunks: List[OfficeQAChunk]          # structure-based chunks with embeddings
    questions: List[OfficeQAQuestion]   # the gold QA pairs
    parser_version: str
    chunker_version: str
    embedder_version: str
    corpus_version: str                 # SHA-256 of (parser, chunker, embedder) versions
```

**The meaningful-query filter** (mandatory per finding L-2 of the BoxOffice paper):

```python
def apply_meaningful_query_filter(
    questions: List[OfficeQAQuestion],
    cache_free_reference_answers: Dict[str, str],  # from Round 1's dense baseline
) -> Tuple[List[OfficeQAQuestion], FilterCutList]:
    """Removes three classes:
    1. Questions the cache-free reference run fails anyway (model capability)
    2. Questions answerable without context (world knowledge — test by asking
       the model with NO corpus block; if it answers correctly, drop)
    3. Low-information yes/no or either/or questions (carry almost no signal)

    Returns (filtered_questions, cut_list). The cut list is itself a PoC finding.
    """
```

**Test gate:** `poc/tests/test_officeqa_loader.py` — CPU-only; loads a 5-document subset, asserts the chunker keeps tables atomic, asserts the meaningful-query filter removes at least one question of each of the three classes.

#### 2.2.2 `poc/cache_engine.py` — the cache-engineering core

**Role:** implement the three cache policies (no-cache, vLLM prefix cache, linear-attention state snapshot) as pluggable backends behind a common interface. The PoC measures the delta between them on the same workload.

**Public API:**
```python
# poc/cache_engine.py
class CacheBackend(ABC):
    @abstractmethod
    def install(self, request: CacheableRequest) -> CacheInstallResult: ...
    @abstractmethod
    def lookup(self, request: CacheableRequest) -> CacheLookupResult: ...
    @abstractmethod
    def evict(self, policy: EvictionPolicy = EvictionPolicy.LRU) -> int: ...
    @abstractmethod
    def repair(self, edit: DocumentEdit) -> RepairResult: ...
    @abstractmethod
    def stats(self) -> CacheStats: ...

class NoCache(CacheBackend):
    """Every request re-prefills. The baseline."""

class VLLMPrefixCache(CacheBackend):
    """The standard: full-attention KV reuse on prefix match.
    Implemented on top of transformers' StaticCache + the repo's
    greedy_decode_dispatch. Hash granularity 512 tokens; physical
    blocks 1024-6144 (the K3 discipline). LRU eviction."""

class LinearAttentionStateCache(CacheBackend):
    """The proposal's cache asset: the (K_state, V_state) pair at the
    8 full-attention block boundaries, 576 KiB per session. Reinstalled
    via cudaMemcpyAsync. Namespace-keyed by (checkpoint, recipe, corpus_version,
    context_class). Edit-local repair on doc-edit."""

@dataclass
class CacheableRequest:
    namespace: CacheNamespace
    session_id: str
    context_contract: ContextContract  # system, policy, corpus block, turns, query
    last_full_attn_block_boundary: int  # 0..8

@dataclass
class CacheNamespace:
    model_checkpoint_id: str            # SHA-256 of the model weights
    quant_recipe_signature: str         # SHA-256 of metadata.json's auto_decision ledger
    context_class: str                  # hash of (system_prompt, policy_block)
    corpus_version: str                 # from officeqa_loader
```

**The namespace key** is the production proposal's namespace, made concrete. The `quant_recipe_signature` is the SHA-256 of `metadata.json`'s `auto_decision` ledger — a re-palettization with a different `--auto-cos` gate produces a different namespace by construction.

**The linear-attention state snapshot/restore hooks:**

```python
# poc/cache_engine.py
class LinearAttentionStateHooks:
    """Hooks into the 8 full_attention layer boundaries (layers 3, 7, 11,
    15, 19, 23, 27, 31 per MODEL_GEOMETRY.md §1's full_attention_interval: 4).

    At each boundary, after the full_attention layer's forward pass:
      1. Snapshot the running (K_state, V_state) from the 3 preceding
         linear_attention layers.
      2. Hash the boundary state (content-addressable).
      3. Store in the LRU pool, keyed by (namespace, session_id, boundary_idx).

    On a cache hit at a boundary:
      1. Look up the stored (K_state, V_state) by content hash.
      2. cudaMemcpyAsync the 576 KiB back to the GPU.
      3. Skip the 3 linear_attention layers between the boundary and the
         new query — recompute only the full_attention block.
    """

    BOUNDARY_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]  # every 4th, 0-indexed
    STATE_BYTES_PER_BOUNDARY = 36 * 1024   # 36 KiB per boundary per session
    TOTAL_STATE_BYTES_PER_SESSION = 576 * 1024  # 8 boundaries × 36 KiB × 2 (K+V)
```

**Test gate:** `poc/tests/test_cache_engine.py` — CPU-only; uses the `tests/test_greedy_match_w23.py::_tiny_hybrid_model()` fixture (a 2-linear + 2-full layer Qwen3.5) to validate that the snapshot/restore produces bit-identical outputs to a no-snapshot run on the same input.

#### 2.2.3 `poc/run_poc.py` — the round orchestrator

**Role:** run the three rounds in sequence, write reports to `poc/reports/`, produce the final PoC verdict.

**CLI:**
```bash
python poc/run_poc.py \
  --artifacts-dir /home/ubuntu/qwen3_5_9B_palettized \
  --heads-dir /home/ubuntu/qwen3_5_9B_palettized_heads \
  --residual \
  --hf-dataset databricks/officeqa \
  --split test \
  --n-questions 50 \
  --max-context 8192 \
  --max-new-tokens 256 \
  --device cuda:0 \
  --rounds 1,2,3 \
  --output-dir poc/reports
```

**The three rounds, concretely:**

**Round 1 — Feasibility floor** (1 week of box time):
1. Run `scripts/check_gpu_contract.py` — confirm A10G is on the allowlist.
2. Run `scripts/doctor.py` — confirm the environment is healthy.
3. Run the kernel parity suite: `pytest tests/test_kernel_status.py tests/test_lut_gradients.py tests/test_two_stream_training.py tests/test_attn_kernel.py tests/test_dequant_reference.py tests/test_idxn_pack_cpu.py -v`. **Green is the precondition for Round 1.**
4. Load dense FP16 Qwen3.5-9B (`eval_common.load_dense_fp16`). Measure VRAM at idle, at 8k context, at 16k context.
5. Run the 32 deterministic prompts (`eval_greedy_match.PROMPTS`) at `max_new_tokens=256`. Record throughput, TTFT, ITL.
6. Run WikiText-2 PPL (`eval_ppl.evaluate_nll`). Confirm parity with the repo's existing 9.2495 figure (within 0.01).
7. Release dense model. Load W4+r32 palettized model (`eval_common.load_quant_model`).
8. Repeat steps 5–6 on the W4 model. Confirm the repo's existing 9.5749 PPL figure (within 0.01).
9. Load the OfficeQA corpus (via `poc/officeqa_loader`). Run 50 questions with the dense model, no cache. Record accuracy.
10. Run 50 questions with the W4 model, no cache. Record accuracy.
11. Apply the meaningful-query filter. Record the cut list.

**Round 1 exit criteria:**
- Dense model fits in VRAM at 8k context (target: <22 GiB).
- W4 model fits in VRAM at 8k context (target: <7 GiB per the production proposal's arithmetic).
- W4 PPL within 0.01 of 9.5749 (sanity check against the repo's existing measurement).
- W4 OfficeQA accuracy measured (no target — this is the floor).
- The meaningful-query filter's cut list recorded.

**Round 2 — Cache-engineering delta** (2 weeks of box time):
1. With the W4 model from Round 1, run the 50 (filtered) OfficeQA questions through three cache backends:
   - `NoCache` (the Round 1 baseline, re-measured for paired comparison).
   - `VLLMPrefixCache` (full-attention KV reuse, 512-token hash granularity, LRU eviction).
   - `LinearAttentionStateCache` (the proposal's cache asset, 576 KiB per session, boundary-keyed).
2. For each backend, measure: TTFT (ms), ITL (ms), throughput (tok/s), accuracy (OfficeQA gold), VRAM peak (GiB), cache hit rate (%), cache eviction count.
3. Run the two-oracle diagnostic on a sampled trace (Belady vs BeladyCompute gap).
4. Run the energy harness (`measure_energy.EnergyMeasurement`) on each backend.

**Round 2 exit criteria:**
- `LinearAttentionStateCache` throughput ≥ `VLLMPrefixCache` throughput at equal accuracy.
- `LinearAttentionStateCache` VRAM peak ≤ `VLLMPrefixCache` VRAM peak (the 576 KiB per session should make this true by construction; the measurement validates it).
- The two-oracle gap documented. If <5%, plain LRU is the production policy.
- The accuracy delta between the three backends is within the meaningful-query filter's noise floor (i.e., cache engineering does not destroy accuracy on the filtered set).

**Round 3 — Staleness envelope** (2 weeks of box time):
1. With the W4 model + `LinearAttentionStateCache` from Round 2, run the staleness envelope on four axes:
   - **Doc-edit axis:** for each of the 50 questions, simulate a document edit by replacing one chunk in the corpus block. Measure accuracy and repair cost (ms) as a function of edit distance (tokens between the edit and the answer).
   - **Role-flip axis:** change the system prompt (e.g., from "you are a helpful assistant" to "you are a financial analyst"). Measure accuracy and re-prefill cost as a function of prompt change magnitude (token-level diff).
   - **Checkpoint-change axis:** re-palettize with `--auto-cos 0.9990` (a different recipe signature). Measure accuracy on the new namespace; refuse cross-checkpoint cache reuse.
   - **Permission-flip axis:** synthetic. Add a query-time ACL filter that excludes one document from the retrieval set. Measure that the answer does not change for users who already lacked access to that document (the structural invariant: permission flip is a query variable, not a cache mutation).
2. For each axis, produce an **envelope curve**: accuracy (or repair cost) as a function of staleness magnitude. The curve is the unit of measurement, not a point estimate (BoxOffice finding L-2).
3. Run the edit-local repair vs. PatchKV-style transport-based recovery comparison on the doc-edit axis (Adjacent C finding).

**Round 3 exit criteria:**
- The doc-edit envelope shows edit-local repair within 13–21× of full re-prefill (the Contiguity paper's measurement; the PoC validates it).
- The role-flip envelope shows the namespace switch is a hard boundary (no cross-namespace reuse).
- The checkpoint-change envelope shows the recipe signature is a hard boundary.
- The permission-flip envelope shows zero leakage (the structural invariant holds).
- The PatchKV comparison is documented; if PatchKV beats edit-local on OfficeQA, the production proposal's edit-local rule is reconsidered.

**Round 3's PoC verdict:**
- If all four exit criteria pass → the production proposal's serving lane thesis is *measured-valid* on this hardware, at this scale, with this model.
- If any criterion fails → the failure is documented with the measured numbers; the production proposal is revised before any Databricks work is commissioned.

### 2.3 What the PoC measures (the report schema)

Every PoC report uses the repo's existing schema (`eval_common.atomic_json_dump`): `timestamp_utc`, `git_head`, `args`, `results`/`aggregate`, `environment`, plus per-record details. The three rounds produce:

```
poc/reports/
├── round1_feasibility_<ts>.json       # VRAM, throughput, PPL, OfficeQA acc (dense + W4)
├── round1_meaningful_query_filter_<ts>.json  # the cut list (itself a finding)
├── round2_cache_delta_<ts>.json       # 3 backends × 6 metrics × 50 questions
├── round2_two_oracle_<ts>.json        # Belady vs BeladyCompute gap
├── round2_energy_<ts>.json            # measure_energy output per backend
├── round3_doc_edit_envelope_<ts>.json # accuracy & repair cost vs edit distance
├── round3_role_flip_envelope_<ts>.json
├── round3_checkpoint_change_envelope_<ts>.json
├── round3_permission_flip_envelope_<ts>.json
├── round3_patchkv_comparison_<ts>.json
└── poc_verdict_<ts>.json              # the final go/no-go with measured numbers
```

The `poc_verdict_<ts>.json` is the single document the production proposal's Stage 3 reads before going live. It contains:

```json
{
  "verdict": "go" | "no_go" | "conditional_go",
  "rounds": {
    "round1": {"status": "pass" | "fail", "exit_criteria": [...]},
    "round2": {"status": "pass" | "fail", "exit_criteria": [...]},
    "round3": {"status": "pass" | "fail", "exit_criteria": [...]}
  },
  "measured_numbers": {
    "dense_fp16_officeqa_accuracy": 0.XX,
    "w4_officeqa_accuracy": 0.XX,
    "w4_ppl_wikitext2": 9.XX,
    "vllm_prefix_cache_throughput_tok_s": XX.X,
    "linear_attn_state_cache_throughput_tok_s": XX.X,
    "vllm_prefix_cache_vram_peak_gib": XX.X,
    "linear_attn_state_cache_vram_peak_gib": XX.X,
    "two_oracle_gap_pct": X.X,
    "doc_edit_repair_vs_reprefill_ratio": XX.X,
    "meaningful_query_filter_cut_list": {"model_capability": N, "world_knowledge": N, "low_information": N}
  },
  "literature_targets": {
    "officeqa_pro_frontier_average": 0.341,
    "officeqa_pro_layout_aware_gain": 0.161,
    "contiguity_repair_ratio_range": [13, 21],
    "boxoffice_metric_artifact_rate": 0.42,
    "kvsharearena_rotation_recovery_range": [0.50, 0.66],
    "lru_beats_sophisticated_count": 14
  },
  "open_questions_resolved": {
    "linear_attn_state_better_than_full_attn_kv_at_officeqa_context": true | false,
    "w4_preserves_enough_accuracy": true | false,
    "a10g_24gib_is_enough": true | false
  }
}
```

---

## 3. What works (the existing infrastructure's wins)

The PoC is built on top of infrastructure that already works. These are the measured wins from the repo's existing reports and the literature:

| What works | Evidence | PoC reuse |
|---|---|---|
| **FLUTE idxN forward kernel at every width 1–4** | `docs/IDXN_UNIFICATION.md` §Performance: ~58.5 TFLOPS (93% peak A10G), uniform across widths. | Round 1's W4 model load + inference. |
| **Two-stream training (W10)** | `docs/TWO_STREAM_ANALYSIS.md`: every recipe (palette ≤ 16 composite, palette > 16 two-stream) is trainable. | Round 3's checkpoint-change arm (re-palettization is a different recipe signature). |
| **Greedy decode with CUDA graphs** | `reports/greedy_equivalence_idx4.json`: W23+ graphs path verified, 32 prompts × 512 tokens completed on both arms. | Round 1's 32-prompt sanity check; Round 2's no-cache arm. |
| **OOM ladder on PPL** | `tests/test_eval_ppl_w22.py`: batch 4 → 2 → 1 halving on OOM, self-heals. | Round 1's PPL measurement on the W4 model at 8k context. |
| **Strict energy measurement protocol** | `scripts/measure_energy.py`: NVML cumulative energy counter, idle baseline subtraction, per-run CIs (n=10), clock locking. | Round 2's energy measurement per backend. |
| **Paired per-document probe** | `scripts/o1_baseline_check.py::paired_diff`: t-stat, p5/p95, n_improved/n_worsened. | Round 3's doc-edit repair paired measurement. |
| **The 3:1 hybrid is native to Qwen3.5-9B** | `docs/MODEL_GEOMETRY.md` §1: 24 linear + 8 full, interval 4. | The cache asset (linear-attn state at the 8 boundaries) is a property of the model, not a tuning choice. |
| **The fla wiring for linear-attention decode** | `scripts/modeling.py::_fla_resolve` (W29): `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` + `causal_conv1d.causal_conv1d_update`. | The cache asset's producer — the linear-attention state is what this wiring computes. |
| **The sm_86 Triton flash attention** | `scripts/attn_sm86.py`: FA1-style online softmax, fp32 running m/l, GQA 4:1. | The 8 full-attention layers' KV — the secondary cache. |
| **The VRAM ledger** | `scripts/vram_ledger.py`: every allocation recorded with its tier. | Extend with a `cache_pool` tier; the LRU evictor treats the ledger as authoritative. |
| **The geometry audit gate** | `scripts/geometry_audit.py`: every doc/test/script that states a model dimension must reconcile with `MODEL_GEOMETRY.md` §1. | Cache-key components that name a layer are validated by this gate. |
| **OfficeQA dataset on HuggingFace** | `databricks/officeqa`: CC-BY-SA-4.0, CSV, pandas-loadable. | Round 1's corpus load. |
| **OfficeQA Pro's layout-aware parse finding** | arXiv:2603.08655: +16.1% relative accuracy from `ai_parse_document`. | Round 1's parser choice (docling as the CPU-side proxy). |
| **LRU beats 14 sophisticated policies** | arXiv:2609.28870. | Round 2's eviction policy: plain LRU. |
| **Edit-local repair is 13–21× cheaper** | arXiv:2609.17983. | Round 3's doc-edit repair policy. |
| **The meaningful-query filter** | arXiv:2609.31415 (BoxOffice): 42% of F1 gains are metric artifacts. | Round 1's filter mandatory before any accuracy claim. |

---

## 4. What doesn't work (the known risks and gaps)

The PoC is honest about what doesn't work yet. These are the measured gaps and the open risks:

| What doesn't work | Evidence | PoC mitigation |
|---|---|---|
| **Quant decode throughput is 0.571× dense** (quant is SLOWER) | `reports/greedy_equivalence_idx4.json` (2026-10-08): 13.514 tok/s quant vs 23.653 tok/s dense. Acceptance bar was ≥2× FASTER. | The PoC's Round 2 measures whether the linear-attention state cache *changes this*. The hypothesis: at OfficeQA's context lengths (4k–16k), the linear-attention state's reinstall cost (576 KiB transfer) may be cheaper than the full-attention KV's re-prefill, flipping the throughput ratio. **This is the PoC's central measurement.** |
| **The GEMV kernel runs at 5–6.5× below the A10G's bandwidth wall** | `docs/A10G_DECODE_INVESTIGATION.md` §0: ~90–110 GB/s effective vs 600 GB/s streaming. The grid is `N/128` CTAs with no K-split; only `gate/up_proj` (96 CTAs) and `lm_head` (1940 CTAs) fill the machine. | Out of scope for the PoC. The PoC measures the cache engineering delta *on top of* the existing kernel. If the GEMV kernel is the binding constraint, the PoC documents it; the kernel fix is a separate workstream (in `flute_extended/src/kernel_cutlass_streaming.cu`). |
| **`lm_head` runs the slowest path** (not in the GEMV nor DUAL pair tables) | `docs/A10G_DECODE_INVESTIGATION.md`: the (4,4) pair + rank-32 residual is in NEITHER table; the single biggest module (~25% of all traffic) runs the slow path. | Out of scope for the PoC. Same as above. |
| **The palettizer OOMs at `--calib-seqs 64 --calib-seq-len 2048`** | `scripts/HANDOVER.issue-2026-10-04.md`: `expandable_segments: memory mapping failed with OOM` repeatedly. | Pre-PoC: re-palettize with `--calib-seqs 32 --calib-seq-len 1024 --mem-temp-mb 64 --oom-retries 5` (the existing knobs, tuned down). |
| **The repo's existing greedy match shows exact_match 0.0** at ~2.31 bits/weight | `reports/greedy_equivalence_idx4.json`: exact_match 0.0, first_divergence mean 13.78. | This is EXPECTED at ~2.3 bits/weight. The PoC's W4 model uses `--recipe auto --auto-cos 0.9995` which lands at ~4 bits/weight average (not 2.31) — the greedy match should be substantially better. Round 1 measures this. |
| **OfficeQA Pro's frontier average is 34.1%** (over half of questions fail) | arXiv:2603.08655. | The PoC's Qwen3.5-9B W4 is expected to score *below* 34.1% — it is a 9B model, not a frontier model. The PoC reports the measured number against this baseline; it does not target 34.1%. |
| **The BoxOffice paper shows 42% of F1 gains are metric artifacts** | arXiv:2609.31415. | The PoC's meaningful-query filter (Round 1) is the mitigation. Without the filter, the cache delta measurement is unfalsifiable. |
| **The Contiguity paper shows edit-local repair fails when the answer moves downstream** | arXiv:2609.17983. | Round 3's doc-edit envelope reports the *distribution* of repair effectiveness, not just the mean. The long tail is expected and documented. |
| **`ai_parse_document` requires a Databricks workspace** | Databricks docs: it is a SQL function. | The PoC uses docling (CPU-side) as the proxy. The +16.1% finding is the *measured ceiling* on `ai_parse_document`; the PoC targets at least half of it (8% relative) on the proxy parser. |
| **OfficeQA's corpus is single-tenant** (no ACL complexity) | The dataset is a public benchmark. | The PoC's permission-flip axis (Round 3) is *synthetic* — it adds a query-time filter that excludes one document. This tests the structural invariant (permission flip is a query variable, not a cache mutation), not the production system's ACL enforcement chain. |
| **The linear-attention state's advantage at sub-1M context is unmeasured** | The Kimi Linear paper measures the 6× decode speedup at 1M context. | **The PoC's central open question.** Round 2 measures the delta at OfficeQA's context lengths (4k–16k). If the advantage is small, the production proposal's serving lane is reconsidered for this workload. |

---

## 5. The build path

Five phases, strictly sequential, each with exit criteria. Total: 6 weeks of box time on one A10G.

| Phase | Builds | Box time | Exit criteria |
|---|---|---|---|
| **P0. Environment + kernel parity** | Confirm A10G is on the GPU contract allowlist. Run `scripts/doctor.py`. Build the FLUTE idxN wheels (`pip install -e flute_extended/ flute_train_kernels/`). Run the kernel parity suite (the 6 test files). Re-palettize Qwen3.5-9B with A10G-tuned knobs. Record the `auto_decision` ledger; SHA-256 it. | 3 days | Kernel parity suite green on the box; palettization complete; recipe signature recorded. |
| **P1. Round 1 — Feasibility floor** | `poc/officeqa_loader.py`. Dense FP16 baseline. W4+r32 baseline. OfficeQA accuracy on both. Meaningful-query filter. | 1 week | Round 1 exit criteria (§2.2.3). |
| **P2. Round 2 — Cache-engineering delta** | `poc/cache_engine.py`. Three cache backends. Two-oracle diagnostic. Energy harness. | 2 weeks | Round 2 exit criteria (§2.2.3). |
| **P3. Round 3 — Staleness envelope** | Four envelope axes. PatchKV comparison. | 2 weeks | Round 3 exit criteria (§2.2.3). |
| **P4. PoC verdict + report** | `poc_verdict_<ts>.json`. The single document the production proposal reads. | 3 days | The verdict is `go`, `no_go`, or `conditional_go` with measured numbers for every claim. |

**Rollback.** If Phase P1 fails (the W4 model does not fit, or the OfficeQA accuracy is catastrophically low), the PoC stops. The production proposal is revised before any Databricks work. If Phase P2 fails (the linear-attention state cache does not beat vLLM prefix cache), the PoC continues to Phase P3 to document the failure mode, but the production proposal's serving lane is reconsidered.

---

## 6. The decision record

| # | Decision | Rejected alternative | Why |
|---|---|---|---|
| PoC-1 | Use `databricks/officeqa` (v1) as the corpus | `officeqa-pro-v2` (90 questions, 120k pages) | v1 is larger, CC-BY-SA-4.0, CSV-shaped, and closest to the repo's existing eval infrastructure. v2's agent harness is out of scope for a 6-week PoC. |
| PoC-2 | Use **docling** as the `ai_parse_document` proxy | marker, unstructured | OfficeQA's corpus is table-heavy (L-5); docling's table-extraction quality is the binding constraint. The +16.1% finding is the ceiling; the PoC targets at least half (8% relative). |
| PoC-3 | Use **structure-based chunking** only | Semantic, LLM-guided, late chunking | The chunking taxonomy (arXiv:2602.16974) shows structure-based wins corpus-wide; LLM-guided pays only for intra-document tasks; late chunking hurts needle queries. |
| PoC-4 | Use **BAAI/bge-m3** as the embedder | OpenAI text-embedding-3-large, Cohere embed-v3 | Multilingual (OfficeQA is English but the Treasury Bulletins have non-ASCII artifacts), 8K context (fits the chunker's output), Apache 2.0 (no API key needed on the A10G box). |
| PoC-5 | Measure **three** cache backends, not two | Measure only no-cache vs linear-attn state cache | The vLLM prefix cache is the *industry standard*; measuring only the proposal's cache against no-cache is unfalsifiable. The delta against vLLM prefix cache is the PoC's central measurement. |
| PoC-6 | Use **plain LRU** eviction | Learned, frequency-based, analytic | arXiv:2609.28870: 14 sophisticated policies fail to beat LRU under agentic load. The PoC validates this on the OfficeQA workload. |
| PoC-7 | Apply the **meaningful-query filter** before any accuracy claim | Report raw accuracy | arXiv:2609.31415 (BoxOffice): 42% of F1 gains are metric artifacts. Without the filter, the cache delta measurement is unfalsifiable. |
| PoC-8 | Report **envelope curves**, not point estimates | Report mean accuracy | arXiv:2609.31415: F1 flips between 0.98 and 0.00 depending on staleness. A point estimate hides this; the envelope curve is the unit of measurement. |
| PoC-9 | Use the **edit-local contiguous repair** policy | Importance-based, attention-based, KV-deviation-based, structural selectors | arXiv:2609.17983 (Contiguity): edit-local recovers ≥0.94 of the post-edit answer margin at 13–21× below re-prefill. The PoC validates this on OfficeQA. |
| PoC-10 | Measure the **PatchKV comparison** on the doc-edit axis | Measure only edit-local vs re-prefill | arXiv:2609.26219 (PatchKV): transport-based recovery may beat edit-local. If it does, the production proposal's edit-local rule is reconsidered. |
| PoC-11 | Run the **two-oracle diagnostic** before building anything beyond LRU | Build compute-aware eviction by default | arXiv:2609.28870: if the Belady vs BeladyCompute gap is small, compute-aware eviction does not pay. The PoC measures the gap; if <5%, plain LRU ships. |
| PoC-12 | Use the **W4+r32 recipe** as the baseline quantization | W2 base, hybrid422 | The repo's existing artifacts are W4+r32 (`reports/greedy_equivalence_idx4.json`); the +3.52% PPL is the measured baseline. W2 and hybrid422 are Round 3's checkpoint-change axis (re-palettization with a different recipe). |
| PoC-13 | Measure on **one A10G**, not multiple GPUs | Multi-GPU sharding | The production proposal's serving lane targets A10G-class hardware (per the handover log). Multi-GPU is a production concern, not a PoC concern. |
| PoC-14 | The PoC's permission-flip axis is **synthetic** | Real ACL enforcement | OfficeQA's corpus is single-tenant. The PoC tests the structural invariant (permission flip is a query variable, not a cache mutation), not the production system's ACL enforcement chain. The four-layer enforcement chain is a production concern. |

---

## 7. Risks and standing controls

| Risk | Evidence it is real | Standing control |
|---|---|---|
| The W4 model OOMs at OfficeQA's context length | `docs/A10G_DECODE_INVESTIGATION.md`: quant VRAM peak 20.19 GiB at the repo's existing workload; OfficeQA's context may push higher. | Round 1 measures VRAM at 4k, 8k, 16k context before any cache work. If 8k OOMs, the PoC drops to 4k and documents the constraint. |
| The linear-attention state cache does not beat vLLM prefix cache | The Kimi Linear paper's 6× figure is measured at 1M context, not at OfficeQA's 4k–16k. | Round 2 measures the delta directly. If the linear-attn state cache is *not* meaningfully better, the PoC documents it; the production proposal's serving lane is reconsidered for this workload. |
| The GEMV kernel's 5–6.5× below-bandwidth performance caps the throughput | `docs/A10G_DECODE_INVESTIGATION.md` §0. | Out of scope for the PoC. The PoC measures the cache delta *on top of* the existing kernel. The kernel fix is a separate workstream. |
| The meaningful-query filter removes too many questions | arXiv:2609.31415: the filter is mandatory but its cut rate is workload-dependent. | Round 1 records the cut list. If >50% of questions are cut, the PoC expands to 100 questions to retain statistical power. |
| OfficeQA's accuracy is too low to measure cache delta | arXiv:2603.08655: frontier agents average 34.1%; Qwen3.5-9B W4 is expected to be lower. | The PoC's accuracy measurement is on the *filtered* set (the meaningful-query filter removes questions the cache-free run fails anyway). The cache delta is measured on the questions where the cache can plausibly matter. |
| The palettizer OOMs again | `scripts/HANDOVER.issue-2026-10-04.md`. | Pre-PoC re-palettization with `--calib-seqs 32 --calib-seq-len 1024 --mem-temp-mb 64 --oom-retries 5`. If this OOMs, the PoC uses the existing `/home/ubuntu/qwen3_5_9B_palettized` artifacts (the handover's产物). |
| The fla wiring is not active at decode time | `scripts/modeling.py::_fla_resolve` (W29): the wiring settles during the pre-capture warmup. | Round 1's sanity check (the 32 deterministic prompts) validates the wiring is active. If the W4 model's throughput is below 10 tok/s, the fla wiring is suspected; `FLUTE_NO_FLA=1` restores the pure-torch fallback for differential diagnosis. |
| The two-oracle diagnostic is too expensive to run | arXiv:2609.28870: the oracles are offline, computed on a trace sample. | Round 2 samples 1000 requests from the Round 2 trace; the diagnostic runs on the sample, not the full trace. |
| The PoC's 6-week timeline is too short | The repo's existing palettization took 6.7 hours for the heads pass alone. | The PoC uses the existing artifacts (re-palettization only if Round 3's checkpoint-change axis requires it). The 6 weeks are: P0 (3 days), P1 (1 week), P2 (2 weeks), P3 (2 weeks), P4 (3 days). |

---

## 8. Traceability

| PoC element | Source |
|---|---|
| OfficeQA dataset | [databricks/officeqa on HuggingFace](https://huggingface.co/datasets/databricks/officeqa) (CC-BY-SA-4.0, CSV) |
| OfficeQA Pro's +16.1% layout-aware parse finding | [arXiv:2603.08655](https://arxiv.org/abs/2603.08655) (Opsahl-Ong et al., Databricks, 9 Mar 2026) |
| OfficeQA Pro's 34.1% frontier average | same paper |
| BoxOffice's 42% metric artifact rate | [arXiv:2609.31415](https://arxiv.org/abs/2609.31415) (the KV-cache reuse accuracy evaluation paper) |
| BoxOffice's F1 0.98 ↔ 0.00 flip | same paper |
| KVShareArena's 50–66% rotation recovery | [arXiv:2609.10266](https://arxiv.org/abs/2609.10266) |
| Contiguity's 13–21× repair ratio | [arXiv:2609.17983](https://arxiv.org/abs/2609.17983) (Mao, Mackey, Lin, 16 Sep 2026) |
| Contiguity's 0.94 answer margin recovery | same paper |
| LRU beats 14 sophisticated policies | [arXiv:2609.28870](https://arxiv.org/abs/2609.28870) (Liu, Yu, Yang, 24 Sep 2026, revised 1 Oct 2026) |
| The two-oracle diagnostic (Belady, BeladyCompute) | same paper |
| Chunking taxonomy (structure-based wins corpus-wide) | [arXiv:2602.16974](https://arxiv.org/abs/2602.16974) (February 2026) |
| Chunking taxonomy (document order preserves prefixes) | same paper |
| Speculative KV cache reuse (2.17–3.95× TTFT) | [ACL 2026](https://aclanthology.org/2026.acl-long.859) |
| RelayCaching (decoding-phase KV reuse) | [ICML 2026](https://icml.cc/virtual/2026/poster/66638) |
| PatchKV (transport-based recovery) | [arXiv:2609.26219](https://arxiv.org/html/2609.26219v1) |
| Hybrid attention on NPUs (Qwen3.5, Kimi) | [arXiv:2609.32114](https://arxiv.org/html/2609.32114v1) |
| Kimi Linear (KDA, 6× decode at 1M context) | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) |
| Kimi K3 (3:1 KDA-to-MLA, NoPE blocks) | Kimi K3 platform docs + [Semianalysis](https://inferencex.semianalysis.com/model/kimi-k3) |
| Qwen3.5's 3:1 hybrid is native | `docs/MODEL_GEOMETRY.md` §1 (`layer_types`, `full_attention_interval`, `linear_*` fields) |
| The 576 KiB linear-attention state per session | `docs/MODEL_GEOMETRY.md` §1 arithmetic (16×128 + 32×128 × 3 × 8 × 2 fp16) |
| The FLUTE idxN forward kernel | `flute_extended/src/kernel_cutlass_streaming.cu` |
| The FLUTE idxN backward kernel | `flute_train_kernels/src/kernel_lut_grad.cu` |
| The FHT rotation kernel | `flute_extended/src/kernel_fht.cu` |
| The sm_86 Triton flash attention | `scripts/attn_sm86.py` |
| The fla wiring for linear-attention decode | `scripts/modeling.py::_fla_resolve` (W29) |
| The dense model loader | `scripts/eval_common.py::load_dense_fp16` |
| The palettized model loader | `scripts/eval_common.py::load_quant_model` |
| The greedy decode with KV cache | `scripts/eval_greedy_match.py::greedy_decode` |
| The CUDA-graph decode | `scripts/eval_greedy_match.py::greedy_decode_dispatch` |
| The WikiText-2 PPL | `scripts/eval_ppl.py::evaluate_nll` |
| The energy harness | `scripts/measure_energy.py::EnergyMeasurement` |
| The paired per-document probe | `scripts/o1_baseline_check.py::score_docs` + `paired_diff` |
| The calibration capture | `scripts/calibrate_real_text.py::CalibrationCapture` |
| The palettizer | `scripts/palettize_qwen3_5_9b.py` |
| The modeling (vendored Qwen3.5) | `scripts/modeling.py` |
| The VRAM ledger | `scripts/vram_ledger.py` |
| The geometry audit gate | `scripts/geometry_audit.py` |
| The GPU contract gate | `scripts/check_gpu_contract.py` |
| The atomic JSON dump | `scripts/eval_common.py::atomic_json_dump` |
| The 32 deterministic prompts | `scripts/eval_greedy_match.py::PROMPTS` |
| The kernel parity suite | `tests/test_kernel_status.py`, `tests/test_lut_gradients.py`, `tests/test_two_stream_training.py`, `tests/test_attn_kernel.py`, `tests/test_dequant_reference.py`, `tests/test_idxn_pack_cpu.py` |
| The existing greedy match report (the sanity-check target) | `reports/greedy_equivalence_idx4.json` (2026-10-08: exact_match 0.0, +3.52% PPL, 0.571× dense throughput) |
| The handover OOM log | `scripts/HANDOVER.issue-2026-10-04.md` |
| The A10G decode investigation | `docs/A10G_DECODE_INVESTIGATION.md` |
| The GPU spec | `docs/GPU_SPEC.md` |
| `ai_parse_document` (the production parser) | [Databricks docs](https://docs.databricks.com/aws/sql/language-manual/functions/ai_parse_document) |

---

## 9. Immediate next actions

1. **Stand up Phase P0.** Clone the new repo (`rag-cache-engineering-qwen35-9b-flute`). Confirm A10G is on the GPU contract allowlist. Build the FLUTE idxN wheels. Run the kernel parity suite. Re-palettize with A10G-tuned knobs (or reuse the existing `/home/ubuntu/qwen3_5_9B_palettized` artifacts).
2. **Implement `poc/officeqa_loader.py`.** Load `databricks/officeqa` from HuggingFace. Wire docling as the parser. Structure-based chunker. BAAI/bge-m3 embedder. Test on CPU with a 5-document subset.
3. **Implement `poc/cache_engine.py`.** The three cache backends behind a common interface. The linear-attention state snapshot/restore hooks at the 8 boundary layers. Test on CPU with the `_tiny_hybrid_model()` fixture.
4. **Implement `poc/run_poc.py`.** The round orchestrator. Test the round transitions on CPU.
5. **Run Phase P1 (Round 1).** Dense FP16 baseline. W4+r32 baseline. OfficeQA accuracy on both. Meaningful-query filter. Record the cut list.
6. **Run Phase P2 (Round 2).** Three cache backends. Two-oracle diagnostic. Energy harness. The central measurement: does the linear-attention state cache beat vLLM prefix cache at OfficeQA's context lengths?
7. **Run Phase P3 (Round 3).** Four envelope axes. PatchKV comparison. The staleness envelope curves.
8. **Produce `poc_verdict_<ts>.json`.** The single document the production proposal reads. The verdict is `go`, `no_go`, or `conditional_go` with measured numbers for every claim.

**The PoC's single success criterion:** the production proposal's riskiest assumption (the linear-attention state matrix is the right cache asset, on this hardware, at this scale, with this model) is *measured*, not assumed. Whatever the measurement says, the PoC has done its job.

---

## Appendix A — The OfficeQA dataset, concretely

From the HuggingFace dataset page ([databricks/officeqa](https://huggingface.co/datasets/databricks/officeqa)):

- **License:** CC-BY-SA-4.0
- **Format:** CSV (pandas-loadable)
- **Size:** <1K questions (the v1 dataset; v2 is 90 questions over 120k pages)
- **Corpus:** U.S. Treasury Bulletins, 1939–2025 (89,000 pages, 26M+ numerical values)
- **Question types:** question–answer pairs requiring reasoning over dense financial tables, charts, and narrative text
- **License obligation:** "By accessing this dataset, you agree not to use the answer keys to train models evaluated on OfficeQA or to artificially inflate benchmark scores."

**Loading pattern (the PoC's `officeqa_loader.py`):**

```python
from datasets import load_dataset
import pandas as pd

# The v1 dataset
ds = load_dataset("databricks/officeqa", split="test")
df = ds.to_pandas()  # CSV-shaped

# The corpus documents are referenced by the questions; the documents
# themselves are downloaded separately (the Treasury Bulletins are public
# domain US government works). The PoC's loader fetches the referenced
# documents from the dataset's metadata, parses them with docling, chunks
# them structure-based, and embeds them with BAAI/bge-m3.
```

**The OfficeQA Pro paper's corpus description (arXiv:2603.08655):** "89,000 pages and over 26 million numerical values. OfficeQA Pro consists of 133 questions that require precise document parsing, retrieval, and analytical reasoning across both unstructured text and tabular data."

The PoC uses the v1 dataset's larger question set (not the Pro v2's 133/90 questions) because the v1 dataset is the one closest to the repo's existing eval infrastructure. The Pro paper's findings (the +16.1% layout-aware parse gain, the 34.1% frontier average) are the PoC's literature targets regardless of which dataset version is used.

---

## Appendix B — The literature, by citation

| Paper | arXiv | Key finding the PoC uses |
|---|---|---|
| OfficeQA Pro | [2603.08655](https://arxiv.org/abs/2603.08655) | +16.1% relative accuracy from layout-aware parsing; 34.1% frontier agent average |
| BoxOffice (KV reuse accuracy) | [2609.31415](https://arxiv.org/abs/2609.31415) | 42% of F1 gains are metric artifacts; F1 flips 0.98 ↔ 0.00 on staleness |
| KVShareArena | [2609.10266](https://arxiv.org/abs/2609.10266) | Free position rotation recovers only 50–66% of the gap; unrepaired reuse worse than no cache |
| Contiguity (stale KV repair) | [2609.17983](https://arxiv.org/abs/2609.17983) | Edit-local repair recovers ≥0.94 of post-edit answer margin at 13–21× below re-prefill |
| Prefix cache eviction (LRU) | [2609.28870](https://arxiv.org/abs/2609.28870) | 14 sophisticated policies fail to beat LRU; recency is unusually predictive under agentic load |
| Chunking taxonomy | [2602.16974](https://arxiv.org/abs/2602.16974) | Structure-based wins corpus-wide; document order preserves prefixes; chunker comparisons are often size comparisons |
| Kimi Linear (KDA) | [2510.26692](https://arxiv.org/abs/2510.26692) | KDA: −75% cache memory, 6× faster decoding at 1M context (the source's measurement condition) |
| PatchKV | [2609.26219](https://arxiv.org/html/2609.26219v1) | Transport-based KV recovery for edited documents (the comparison baseline for Round 3's doc-edit axis) |
| Speculative KV reuse | [ACL 2026](https://aclanthology.org/2026.acl-long.859) | 2.17–3.95× TTFT reduction with negligible accuracy loss (the throughput target) |
| RelayCaching | [ICML 2026](https://icml.cc/virtual/2026/poster/66638) | Decoding-phase KV reuse across agents (the multi-turn reuse validation) |
| Hybrid attention on NPUs | [2609.32114](https://arxiv.org/html/2609.32114v1) | Hybrid attention models (Qwen3.5, Kimi) on NPUs (the transferability note) |

---

## Appendix C — The PoC's diff against the production proposal (v1.1)

The PoC is the production proposal's serving lane, *measured*:

| Production proposal (v1.1) | PoC (this document) |
|---|---|
| Four lanes (corpus, permission, serving, evaluation) on Databricks | Three rounds (feasibility, cache delta, staleness envelope) on one A10G |
| SharePoint corpus via Microsoft Graph | OfficeQA corpus via HuggingFace |
| `ai_parse_document` for layout-aware parsing | docling (CPU-side proxy) |
| Unity Catalog for ACL enforcement | Synthetic permission-flip (structural invariant test) |
| Qwen3.5-9B native 3:1 hybrid + FLUTE idxN | Same model, same kernels |
| Linear-attention state matrix as the cache asset (576 KiB per session) | Same cache asset, *measured* against vLLM prefix cache |
| Namespace keyed by (checkpoint, recipe, context_class, corpus_version) | Same namespace, *measured* on the checkpoint-change axis |
| Plain LRU eviction, pinned system prefix | Same, *measured* against the two-oracle diagnostic |
| Edit-local repair on doc-edit | Same, *measured* against PatchKV |
| Golden-set gate per checkpoint and per corpus version | The PoC's verdict JSON is the gate's measurement |
| 96 GiB capacity (corrected to A10G's 13–16 GiB) | *Measured* on the A10G |
| The KDA graft (removed in v1.1 — the model has it natively) | Confirmed: the model has it natively |
| Stage 3 commissioning (the integration plan) | The PoC *is* Stage 3, measured |

**The PoC is the production proposal's riskiest assumption, measured.** Whatever the measurement says, the production proposal is either validated or revised before any Databricks work is commissioned.
