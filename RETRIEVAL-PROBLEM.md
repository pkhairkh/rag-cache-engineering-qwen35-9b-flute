# THE RETRIEVAL PROBLEM — and the real options

> **The user is right.** The workflow as written has a fundamental flaw: IVFADC cos-sim between the query's pooled hidden state and the chunk's pooled hidden state is **semantically meaningless**. They're both 4096-dim, but one is the model's representation of a QUESTION and the other is the model's representation of CONTENT. They live in different regions of the hidden space. Cos-sim between them doesn't measure relevance — it measures token-distribution overlap (which only worked in the toy because the toy's queries were token-biased the same way as the chunks).

---

## 1. Why the toy "worked" (and why it's misleading)

In the toy (`toy_1000_chunks.py`, `toy_investigate.py`):
- Chunks were generated with 40% of tokens from a topic's token range `[10 + t*20 : 10 + t*20 + 20]`
- Queries were generated the SAME way: 50% of tokens from the same topic's token range
- So the query and the chunk shared input tokens → their pooled hidden states were in the same region → IVFADC cos-sim retrieved the right topic (87%)

**This is NOT how real RAG works.** In real RAG:
- The query is a natural-language question ("What was the revenue in Table 3?")
- The chunk is a table of numbers
- They share NO tokens. Their pooled hidden states are in completely different regions.
- IVFADC cos-sim between them is ~0 (or random) — retrieval fails.

**The toy's 87% retrieval was an artifact of the toy's data generation, not a validation of the architecture.**

---

## 2. The actual problem (stated precisely)

The retrieval step needs a **relevance function**: given a query Q and a chunk C, return a score that's high when C contains the answer to Q.

**Option A (what I proposed, WRONG):** `score = cos_sim(pooled_hidden(Q), pooled_hidden(C))`
- This compares the model's representation of the question to the model's representation of the content.
- These representations are in different semantic spaces — the model never trained them to be comparable.
- **Cos-sim is meaningless.** It measures token-distribution overlap, not relevance.

**Why the real Qwen3.5-9B doesn't fix this:** the model's pooled hidden state for a question and for a content chunk are NOT trained to be comparable. The model was trained for next-token prediction, not for question-content matching. Even with pretraining on OfficeQA, the pooled hidden states don't become a retrieval space — they become a language-modeling space.

---

## 3. The real options (what actually works)

### Option B: A separate trained embedder (bge-m3) for retrieval — the standard RAG path

**How:** use a separate, retrieval-trained model (bge-m3, 1024-dim) to embed both the query and the chunks. The embedder is trained specifically for question-content matching (contrastive loss on (question, relevant_doc) pairs). IVFADC on the embedder's vectors retrieves the right chunks.

**Pros:** this is the standard, proven RAG retrieval. The embedder is trained for exactly this.

**Cons:** it requires a SEPARATE model (bge-m3) — which violates the "single LUT model" principle the user wants. The embedder is not the LUT model.

**This is what the user rejected** in the earlier conversation ("WE ARE NOT GOING TO SNAPSHOT THE FUCKING FULL ATTENTION CACHES... NO TEXT ON DISK... NOT RAG 2020"). The user wants the LUT model to do everything.

### Option C: Train a retrieval head on the LUT model (the "learned projection" I mentioned, but correctly)

**How:** add a small projection head to the LUT model that projects the pooled hidden state to a RETRIEVAL SPACE. Train this head with a contrastive loss on (query, relevant_chunk) pairs — the same way bge-m3 is trained, but using the LUT model's hidden state as input.

```
query_hidden (4096) → projection_head → query_retrieval_vec (1024)
chunk_hidden (4096) → projection_head → chunk_retrieval_vec (1024)
loss = InfoNCE(query_retrieval_vec, chunk_retrieval_vec)  # same-topic closer
```

**Pros:** the LUT model does everything. The projection head is a small linear layer, not a separate model.

**Cons:** you need (query, relevant_chunk) pairs to train the head. For OfficeQA, the gold questions reference specific documents — so you have the pairs. But the head must be trained BEFORE the snapshots are built (the snapshot vector is the projected vector, not the raw hidden state).

**This is the "learned projection head" from the bottleneck analysis — but I described it wrong before.** The head projects BOTH the query and the chunk to a shared retrieval space. It's not just "make the chunk vector discriminative" — it's "make the query and chunk vectors COMPARABLE."

### Option D: Use the model's full-attention as the retriever (the "attention as retrieval" I proposed earlier)

**How:** install ALL chunks' caches into the model (or a subset), then run the query. The model's full-attention layers attend to the installed caches. The attention weights tell you which chunks are relevant.

**Pros:** no separate retrieval step. The model itself does the retrieval.

**Cons:**
- You'd have to install ALL 50,000 chunks' caches to let the model attend to all of them — but the M1/M2 memory has a fixed size (mem_size slots), not 50,000 slots.
- Even if you could install all 50k, the attention over 50k slots is O(50k) per token — too slow.
- The model's full-attention is over the KV cache, not over the M1/M2 memory. The M1/M2 read is a separate mechanism (the attention over mem_size slots), and it's not trained for retrieval.

**This doesn't work at scale.** The M1/M2 memory has a fixed size; it can't hold 50k chunks' worth of slots.

### Option E: The hybrid — IVFADC for preselect, the LUT model's M1/M2 for refinement

**How:**
1. Use a separate embedder (bge-m3) OR a trained projection head (Option C) for the IVFADC preselect → top-100 candidates
2. For each of the 100 candidates, install its cache into the LUT model and measure how much the query's answer logit changes (the "cache influence")
3. Pick the top-3 chunks by cache influence → install those → answer

**Pros:** combines cheap preselect (IVFADC) with the model's own relevance signal (the cache influence). The final selection is done by the model, not by cos-sim.

**Cons:** requires running the model 100 times (once per candidate) to measure cache influence — ~100 × 2ms = 200ms. Or batch them.

---

## 4. What I actually propose (the honest answer)

**The user wants the LUT model to do everything. The only way that works is Option C: train a retrieval head on the LUT model.**

```
THE LUT MODEL (Qwen3.5-9B, FLUTE idxN W4)
  │
  ├── 24 linear-attn layers (the cache mechanism — S, M1, M2 per layer)
  ├── 8 full-attn layers (fresh per query)
  ├── lm_head (for next-token prediction)
  └── retrieval_head (NEW — a small linear layer, trained with InfoNCE)
      │
      │ at ingestion:
      │   chunk → prefill → pooled_hidden → retrieval_head → retrieval_vec
      │   (the retrieval_vec is stored in IVFADC, NOT the raw pooled_hidden)
      │
      │ at query:
      │   query → prefill → pooled_hidden → retrieval_head → retrieval_vec
      │   (the query's retrieval_vec is searched against IVFADC)
```

**The retrieval head is trained ONCE, on (query, relevant_chunk) pairs.** For OfficeQA, the gold questions reference specific documents — so the pairs exist. The head projects both queries and chunks to a shared retrieval space where cos-sim measures relevance.

**This stays in the architecture:** the LUT model does everything. The retrieval head is a small linear layer (4096 → 1024), not a separate model. The snapshots still store the cache deltas (S, M1, M2) — the retrieval head is just for the IVFADC vector, not for the cache content.

---

## 5. Why the toy's "learned projection head" was the right idea but the wrong description

In the bottleneck analysis (`toy_apply_fixes.py`), I added a `retrieval_proj` layer and trained it with InfoNCE. Retrieval improved 82% → 89%. But I described it as "make the chunk vector discriminative" — which is incomplete.

**The right description:** the projection head projects BOTH the query and the chunk to a shared retrieval space. The InfoNCE loss pulls (query, relevant_chunk) pairs together and pushes (query, irrelevant_chunk) pairs apart. After training, cos-sim in the retrieval space measures relevance — because the head was TRAINED to make it so.

**The toy's 89% was real** — but only because the toy's queries and chunks shared topic tokens, so the head had an easy job. In real RAG, the head must learn to map a question to the same region as a content chunk — which is harder, but that's what retrieval-trained embedders (bge-m3) do, and what the head must learn.

---

## 6. The corrected workflow

### Pretrain
- Train the LUT model's linear-attn layers on next-token prediction (the W10 path) — this makes the cache mechanism carry discriminative info (the toy's finding)
- Train the retrieval head with InfoNCE on (query, relevant_chunk) pairs — this makes the retrieval space meaningful

### Ingest + snapshot
- For each chunk: prefill → pooled_hidden → retrieval_head → retrieval_vec
- Store the retrieval_vec in IVFADC (NOT the raw pooled_hidden)
- Store the cache deltas (S, M1, M2) per chunk on disk

### Query
1. Tokenize + prefill the query → pooled_hidden → retrieval_head → query_retrieval_vec
2. IVFADC preselect on the retrieval_vec → top-100
3. Cos sim rerank → top-3 chunk indices
4. Load the top-3 chunks' cache snapshots
5. Sum the deltas → install into the model
6. Answer from the installed caches (NO re-prefill)

### What's different from the wrong version
- The IVFADC index is on the **retrieval_head's output** (a trained retrieval space), NOT the raw pooled hidden state
- The retrieval head is trained with (query, relevant_chunk) pairs — so query and chunk vectors are comparable
- The raw pooled hidden state is NEVER used for retrieval — only the projected retrieval vector

---

## 7. The honest constraint

**The retrieval head needs (query, relevant_chunk) pairs to train.** For OfficeQA, these exist (the gold questions reference documents). For a general corpus, you'd need to generate them (e.g., ask the model to generate questions for each chunk — the "synthetic QA" approach).

**Without the retrieval head, the LUT model's pooled hidden state is NOT a retrieval space.** IVFADC on the raw hidden state is meaningless (the toy's success was an artifact). The retrieval head is the bridge between the LUT model's language-modeling space and the retrieval space.

**This is the ONE place where a "separate" component is needed** — but it's a small linear layer ON the LUT model, not a separate model. The LUT model still does everything; the head is just a projection of the model's own hidden state.
