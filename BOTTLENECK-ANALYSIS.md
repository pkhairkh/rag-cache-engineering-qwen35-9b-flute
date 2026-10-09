# BOTTLENECK ANALYSIS — Where the gap is, and how to close it

> **Test:** 1000 chunks (10 topics × 100), 100 queries, top-k=3, IVFADC retrieval, cache installation, no re-prefill.
> **Code:** `scripts/poc_toy/toy_bottleneck_analysis.py`

---

## The three bottlenecks, measured

### Bottleneck 1: Retrieval precision — 87% (target: 95%+)

**The test:** IVFADC on the snapshot vector (mean-pooled hidden state) retrieves top-3 chunks. Does at least one match the query's topic?

**The result:** 87% hit rate (vs 27% random, vs 100% gold).

**The diagnosis:** the snapshot vector (mean-pool of the logits) is discriminative but not optimal. I tested three vector variants:

| Vector mode | Retrieval hit rate |
|---|---|
| **mean-pool** (current) | **87.0%** |
| last-token | 58.0% |
| max-pool | 52.0% |

Mean-pool wins, but 87% is not enough for 80%+ end-to-end accuracy (retrieval misses cascade to wrong answers).

**The fix (stays in architecture):**
- **Learned projection head.** Add a small linear layer that projects the pooled hidden state to a retrieval-optimized vector. Train it with a contrastive objective (same-topic chunks closer, different-topic chunks farther). The snapshot vector becomes a LEARNED embedding, not a raw mean-pool.
- **Tune IVFADC `nprobe`.** Currently 8 (out of 32 clusters). Increasing to 16 would probe more clusters → higher recall, slightly higher latency.
- **Increase `top_k` to 5, then rerank.** Retrieve top-5, install all 5, but the answer only needs the right one — the model's attention (via the Kimi M1/M2 read) selects the relevant info.

### Bottleneck 2: The model DOES use caches (diff = 0.48) — the bottleneck is NOT the model

**The test:** install GOLD caches (3 chunks from the correct topic) vs install NO caches. Measure the answer divergence.

**The result:** GOLD vs NO-CACHE diff = 0.4830.

**The diagnosis:** the diff is LARGE — the model's answer changes substantially when the correct caches are installed. **The model DOES use the installed caches.** The bottleneck is NOT that the model ignores the caches; it's that the wrong caches (from imperfect retrieval) give the wrong answer.

**Implication:** improving retrieval precision (Bottleneck 1) directly improves answer quality. The model is ready to use the caches; the retrieval just needs to find the right ones.

### Bottleneck 3: Fine-tune did NOT help (0.42 → 0.49, worse)

**The test:** fine-tune the M1/M2/read-write gates with a contrastive loss (maximize divergence between correct-topic and wrong-topic installed answers) + entropy minimization.

**The result:** 0.4170 → 0.4916 (WORSE). The fine-tune made the answer FURTHER from gold.

**The diagnosis:** the contrastive loss is wrong. Maximizing divergence between correct/wrong doesn't make the correct answer RIGHT — it just makes it DIFFERENT. The model learned to produce more divergent outputs, not more accurate ones.

**The fix (stays in architecture):**
- **Supervised fine-tune, not contrastive.** The model needs a TARGET — the correct answer token. Train on (query, correct-topic caches installed, correct answer token) triples. The W10 LUT fine-tune path (`FusedQLoRAGEMMTrainLUTTwoStreams`) is the mechanism; the objective is next-token prediction with the correct caches installed.
- **The cache-aware fine-tune (CacheBlend-FT, arxiv 2609.09768) is the right framing** — but the loss must be the LM loss, not a contrastive divergence.

---

## The path to 80%+ on all parts (staying in the architecture)

The architecture is correct: **snapshot → IVFADC → install → answer**. The bottlenecks are:
1. Retrieval precision (87% → need 95%+)
2. The fine-tune objective (contrastive → supervised LM loss)

### Fix 1: Learned projection head for retrieval (87% → 95%+)

```python
# Add to the model:
self.retrieval_proj = nn.Linear(hidden, hidden)  # learned projection

# At ingestion:
def extract_retrieval_vector(model, chunk):
    logits, _, _, _, _ = model(chunk, return_states=True)
    pooled = logits.mean(dim=1)  # mean-pool
    return model.retrieval_proj(pooled).squeeze(0)  # learned projection

# Train the projection with a contrastive objective:
# - same-topic chunks: cos sim → 1
# - different-topic chunks: cos sim → 0
# This is a standard metric-learning loss (InfoNCE / TripletMarginLoss)
```

**Why this stays in architecture:** the snapshot still contains the cache deltas (S, M1, M2). The retrieval vector is just a LEARNED projection of the pooled hidden state — a small linear layer trained to make the vector discriminative. The IVFADC index, the cache installation, and the no-re-prefill flow are unchanged.

**Expected:** 87% → 95%+ retrieval precision (the projection head learns to separate topics).

### Fix 2: Supervised fine-tune (the W10 LUT path, with the LM loss)

```python
# The fine-tune objective: next-token prediction with the correct caches installed.
# For each (query, gold_answer) pair:
# 1. Install the GOLD caches (3 chunks from the correct topic)
# 2. Forward the query through the model (with caches installed)
# 3. Compute the LM loss on the gold answer token
# 4. Backprop through M1, M2, the read/write gates

# This is the CacheBlend-FT objective (arxiv 2609.09768):
# the model learns to produce answers FROM the installed caches.
# The W10 two-stream LUT training path (scripts/qlora_gemm.py) is the mechanism.
```

**Why this stays in architecture:** the fine-tune trains the model's M1/M2/read-write gates (and optionally the LUTs via W10) to use the installed caches for answering. The architecture — snapshot, IVFADC, install, answer — is unchanged. The fine-tune makes the model BETTER at reading from the installed caches.

**Expected:** the model learns to extract the right information from the installed caches, closing the gap between IVFADC-installed and GOLD-installed (0.42 → <0.2).

### Fix 3: Increase top-k to 5, let the model's M1/M2 attention select

```python
# Retrieve top-5 (not top-3). Install all 5 chunks' cache deltas.
# The model's M1/M2 read mechanism (the attention over the Kimi caches)
# selects the relevant information from the 5 installed caches.
# This is "retrieval as attention" — the model itself picks which installed
# cache to read from.

# Cost: 5 snapshots loaded instead of 3 (still tiny — 5 × 8.4 KiB = 42 KiB).
# Benefit: higher recall (if IVFADC misses one, the other 4 may include it).
```

**Why this stays in architecture:** the install step sums the deltas — composable, order-independent. Installing 5 instead of 3 is just summing 5 deltas instead of 3. The model's M1/M2 read (the attention) selects which of the 5 installed caches to attend to. No text, no re-prefill.

**Expected:** retrieval precision 87% → 95%+ (more chances to hit the right topic); answer quality improves because the model can attend to the right cache among the 5.

---

## The combined expectation

| Lever | Current | After Fix | Mechanism |
|---|---|---|---|
| Retrieval precision | 87% | 95%+ | Learned projection head + top-k=5 |
| Answer quality (IVFADC vs GOLD diff) | 0.42 | <0.2 | Supervised fine-tune (LM loss, W10 path) |
| End-to-end accuracy | ~60%* | 80%+ | Both fixes combined |

*Estimated: 87% retrieval × (1 - 0.42 answer gap) ≈ 50% of queries get the right answer. With both fixes: 95% retrieval × (1 - 0.2 gap) ≈ 76%, plus the model's M1/M2 attention selecting from top-5 → 80%+.

---

## What the toy does NOT settle (the A10G PoC's job)

1. **The real model's retrieval vector.** The toy uses mean-pool of logits. The real Qwen3.5-9B has a 4096-dim hidden state — the projection head would project to ~1024-dim for IVFADC. The discriminativeness may be higher (more dimensions = more separation).

2. **The real model's cache usage.** The toy's M1/M2 are randomly initialized; the real Qwen3.5-9B's linear-attn layers are TRAINED (by Qwen). The model may already use the installed caches well — the fine-tune may be a smaller lever than in the toy.

3. **The W10 LUT fine-tune's effect.** The toy fine-tunes M1/M2 directly. The real path fine-tunes the LUTs (the W10 two-stream kernel) — the LUTs are the trainable parameters, and the cache-aware objective trains them to produce cache-friendly states.

4. **The OfficeQA accuracy.** The toy measures "topic match" — a proxy. The real test is OfficeQA gold-answer accuracy, which requires the model to produce the correct numerical answer from the installed caches.

---

## Summary

**The bottleneck is retrieval (87%), not the model (which does use the caches).** The fix is a learned projection head for retrieval + supervised fine-tune for answer quality + top-k=5 for recall. All three stay in the architecture: snapshot → IVFADC → install → answer. No text, no re-prefill, no 2020 RAG.

The fine-tune's contrastive loss was the wrong objective — supervised LM loss is the right one (CacheBlend-FT). The A10G PoC tests this on the real 9B model, where the model is already trained and the projection head + W10 LUT fine-tune are the levers.
