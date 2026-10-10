# RAGGA Handover Document

## Current State

W16 fixes deployed and verified. Retrieval frame works correctly but cache-state vectors don't align questions with documents. **M1/M2 global memories are the intended retrieval mechanism but are currently inactive.**

| Component | Status |
|-----------|--------|
| GPU kernels | ✓ Compiled |
| Ingestion | ✓ 100 chunks with absolute protocol |
| W16 tests | ✓ All 7 pass |
| Retrieval frame | ✓ Works (perfect discrimination when vectors match) |
| TRUE-dist | ✓ Improved (0.10 → 0.040) |
| M1/M2 | ✗ NOT ACTIVATED - loader uses default model class |

---

## The Problem

**Question vectors don't match document vectors:**
- Cosine between question and its target doc: ~0.50-0.58
- Cosine between doc and itself: 1.0
- The S vectors (linear attention state) are not discriminative for retrieval

**Retrieval results (50% hit rate):**
```
Q0: Expected chunk 77 → Got [70, 82, 71] → MISS
Q1: Expected chunk 80 → Got [70, 80, 82] → HIT (rank 2)
Q2: Expected chunk 86 → Got [70, 71, 80] → MISS  
Q3: Expected chunk 71 → Got [70, 71, 80] → HIT (rank 2)
```

---

## The Solution: M1/M2 Global Memories

M1/M2 are two global memory matrices shared across ALL 24 linear attention layers. They are designed for retrieval:

**From SPECIFICATION.md §2.2:**
- M1 (global key-memory): shape (32, mem_size, 128)
- M2 (global value-memory): shape (32, mem_size, 128)
- Read: `softmax(q @ M1ᵀ) @ M2` — retrieves from global memory
- Write: gated additive per-token scatter — path-independent deltas

**Why M1/M2 will work:**
1. Questions and documents that are semantically similar should produce similar M1/M2 states
2. The gated write can be trained to extract discriminative features
3. Global scope across all layers = more expressive than per-layer S

---

## What Needs to Be Done

### 1. ALWAYS Attach M1/M2 (CRITICAL)

The current loader uses `AutoModelForCausalLM.from_pretrained` which instantiates the default Qwen3.5 model class. This class does NOT have M1/M2 wiring.

**The custom `Qwen3_5TextModel` in `src/scripts/modeling.py` has the M1/M2 wiring at lines 1135-1158:**

```python
# modeling.py line 1135
if getattr(config, "use_m1m2", False):
    import m1m2 as _m1m2
    self.m1m2 = _m1m2.M1M2(...)
    for _layer in self.layers:
        if _layer.block_type == "linear_attention":
            object.__setattr__(_layer.linear_attn, "m1m2", self.m1m2)
```

**Problem:** `AutoModelForCausalLM` doesn't use this class. It uses the default transformers `Qwen3_5ForCausalLM`.

**Fix Required:**
1. Modify `src/scripts/loader.py` and `src/scripts/palettized_modules.py` to:
   - Load config with `use_m1m2=True` and `m1m2_mem_size=LARGE` (see below)
   - Instantiate model with custom config so M1/M2 are attached

2. OR: Create a custom model class that inherits from the default and adds M1/M2 post-hoc

**Key Files to Modify:**
- `src/scripts/loader.py` — entry point
- `src/scripts/palettized_modules.py` — `load_palettized_model()` function
- Ensure `config.use_m1m2 = True` is set BEFORE model instantiation

### 2. Expand M1/M2 Size

Current default: `m1m2_mem_size = 128`

**This is too small for retrieval.** The memory needs enough capacity to represent a general semantic space.

**Recommended sizes to try:**
- 1024 (8x larger than current)
- 4096 (32x larger)
- 8192 or larger if memory allows

The shape is `(num_heads=32, mem_size, head_dim=128)`, so:
- mem_size=128 → 0.5M elements per M (M1 + M2 = 1M total)
- mem_size=1024 → 4M elements per M (M1 + M2 = 8M total)
- mem_size=4096 → 16M elements per M (M1 + M2 = 32M total)
- mem_size=8192 → 32M elements per M (M1 + M2 = 64M total)

Set via `config.m1m2_mem_size = 4096` when loading model.

### 3. Fine-tune M1/M2 Gates Against BASELINE (NOT Corpus)

**From m1m2.py:**
> "Zero-init (PROPOSAL P3): the write gates start at 0.0 — M1/M2 start as no-ops and the untrained model's behavior is bit-unchanged"

**What this means:**
- `m1m2.write_gate` parameter is initialized to 0.0
- This means M1/M2 don't accumulate any state during prefill
- The gates must be trained to open and extract useful features

**CRITICAL: Train against BASELINE, NOT specific corpus**

We do NOT want to train M1/M2 against a specific corpus. That would make the system corpus-dependent. Instead:

**Train M1/M2 to produce a general semantic similarity space using:**
- Generic text pairs (paraphrase datasets like MRPC, QQP, PAWS)
- Question-answer pairs from general QA datasets (SQuAD, Natural Questions)
- Text entailment pairs (SNLI, MNLI)
- Any dataset where semantically similar texts should produce similar representations

**The goal:**
- M1/M2 learns to extract "what this text is about" in a general way
- Questions about topic X → M1/M2 state A
- Documents about topic X → M1/M2 state A (similar)
- No corpus-specific training needed
- Works on any unseen corpus at inference time

**Fine-tuning approach:**
1. Freeze ALL model weights (no gradient to main model)
2. Train ONLY the M1/M2 gates: `m1m2.write_gate` and any read parameters
3. Use general-purpose semantic similarity datasets (NOT your target corpus)
4. Objective: similar texts → similar M1/M2 states

**Training objective (contrastive):**
```python
# Use general paraphrase/similarity datasets
# Example: MRPC (Microsoft Research Paraphrase Corpus)
# Sentence 1: "The company announced..."
# Sentence 2: "The firm declared..." (paraphrase → positive pair)
# Sentence 3: "The weather today..." (unrelated → negative pair)

for batch in general_similarity_dataloader:
    text1_ids, text2_ids, label = batch  # label=1 for paraphrase
    
    # Prefill text1
    cache1 = TQCache(...)
    model(text1_ids, past_key_values=cache1)
    m1_1, m2_1 = cache1.read_m1(), cache1.read_m2()
    
    # Prefill text2
    cache2 = TQCache(...)
    model(text2_ids, past_key_values=cache2)
    m1_2, m2_2 = cache2.read_m1(), cache2.read_m2()
    
    # Loss: push similar texts together, dissimilar apart
    sim = cosine(m1_1, m1_2) + cosine(m2_1, m2_2)
    loss = contrastive_loss(sim, label)
    loss.backward()
    optimizer.step()
```

**Datasets to use (all general-purpose, NOT corpus-specific):**
- MRPC (paraphrase)
- QQP (Quora Question Pairs)
- PAWS (Paraphrase Adversaries)
- SNLI / MNLI (entailment)
- STS-B (semantic textual similarity)
- SQuAD (question-context pairs)

**What NOT to do:**
- ❌ Train on your target corpus documents
- ❌ Create corpus-specific embeddings
- ❌ Fine-tune for specific retrieval tasks

**What TO do:**
- ✓ Train on general semantic similarity datasets
- ✓ Learn a universal representation space
- ✓ System works on ANY unseen corpus at inference time

---

## Files Reference

### Core Model Files
- `src/scripts/modeling.py` — Custom Qwen3_5TextModel with M1/M2 wiring (lines 1130-1158)
- `src/scripts/loader.py` — Model loading entry point
- `src/scripts/palettized_modules.py` — `load_palettized_model()` function (line 3264)
- `src/rag/m1m2.py` — M1M2 class implementation

### M1/M2 Integration Points
- `modeling.py:727-749` — M1/M2 read/write in attention forward
- `modeling.py:1135-1158` — M1M2 instantiation and wiring to layers
- `modeling.py:1145-1150` — M1M2 constructor call

### TQCache Integration
- `src/rag/tq_cache.py:594-595` — M1/M2 codes storage
- `src/rag/tq_cache.py:667-680` — `update_m1()` method
- `src/rag/tq_cache.py:698-711` — `update_m2()` method
- `src/rag/tq_cache.py:733-734` — `snapshot_codes()` includes m1/m2

### Spec Documentation
- `SPECIFICATION.md:26-27` — M1/M2 dimensions
- `SPECIFICATION.md:33-35` — M1/M2 description (global, shared across layers)
- `SPECIFICATION.md:N7` — Read/write operations
- `SPECIFICATION.md:N28` — Fine-tune requirement

---

## Verification Steps

After implementing M1/M2 activation:

1. **Verify M1/M2 are attached:**
```python
model, _ = load_model()
print(f"Model has m1m2: {hasattr(model.model, 'm1m2')}")
for i, layer in enumerate(model.model.layers):
    if hasattr(layer, 'linear_attn') and hasattr(layer.linear_attn, 'm1m2'):
        print(f"Layer {i} has m1m2: True")
```

2. **Verify M1/M2 are captured during forward:**
```python
cache = TQCache(...)
model(some_input, past_key_values=cache)
print(f"M1 codes: {cache.m1_codes is not None}")
print(f"M2 codes: {cache.m2_codes is not None}")
```

3. **Run ingestion and check snapshots:**
```python
snap = load_chunk('snapshots/chunk_00000.npz')
print(f"M1: {snap.m1_codes is not None}, M2: {snap.m2_codes is not None}")
```

---

## Expected Results After Fine-tuning

1. **M1/M2 gates open** → M1/M2 accumulate meaningful state during prefill
2. **Similar texts → similar M1/M2** (learned from general datasets)
3. **Works on ANY corpus** — not trained on specific documents
4. **Retrieval using M1/M2 cosine** → high hit rate on unseen corpora

---

## Current Working Directory

```
/home/ubuntu/RAGGA/
├── src/
│   ├── rag/              # RAG pipeline (TQCache, snapshot, index, query)
│   ├── scripts/          # Model code (modeling.py, loader.py, m1m2.py)
│   └── flute_extended/   # CUDA kernels
├── scripts/gpu/          # GPU tools (run_ingestion.py, run_query.py, etc.)
├── disk/ingested_w16/    # Current ingestion (M1/M2 NOT captured)
└── HANDOVER.md
```

---

## Commands for Testing

```bash
# Run W16 tests (all should pass)
python3 -m pytest src/rag/tests/test_w16_install.py -v

# Check if M1/M2 are attached (currently returns False)
python3 -c "
import sys; sys.path.insert(0, 'src/rag'); sys.path.insert(0, 'src/scripts'); sys.path.insert(0, 'scripts/gpu')
from _bootstrap import boot, load_model; boot()
model, _ = load_model()
print(f'M1/M2 attached: {hasattr(model.model, \"m1m2\")}')
"

# Run ingestion (will capture M1/M2 once loader is fixed)
python3 scripts/gpu/run_ingestion.py --out-dir disk/ingested_m1m2 --n-docs 100

# Run bisection
python3 scripts/gpu/bisect_install.py --disk-dir disk/ingested_m1m2
```

---

## Summary for Coding Agent

**GOAL:** Make RAGGA retrieval work by activating and training M1/M2 global memories.

**TASKS:**
1. Fix `loader.py` and `palettized_modules.py` to attach M1/M2 during model load
2. Increase `m1m2_mem_size` parameter (try 1024, 4096, 8192)
3. Write fine-tuning script that:
   - Freezes main model
   - Trains only M1/M2 gates
   - Uses **general semantic similarity datasets** (MRPC, QQP, SNLI, etc.)
   - **NOT trained on target corpus** — works on any unseen corpus
4. Re-run ingestion with M1/M2 enabled
5. Test retrieval using M1/M2 vectors

**KEY INSIGHTS:**
1. M1/M2 write gates start at 0.0 (closed). Training opens them.
2. Train on GENERAL similarity datasets, NOT your specific corpus
3. The model learns a universal semantic space
4. NO corpus-specific training — system generalizes to any corpus

**CORPUS-INDEPENDENT DESIGN:**
- M1/M2 learns "what makes texts semantically similar" from general data
- At inference: any question and its relevant document produce similar M1/M2
- Works on enterprise docs, news, scientific papers, etc. — no retraining needed
