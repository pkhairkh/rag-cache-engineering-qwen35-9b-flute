# RAGGA Handover — W20 (M1/M2 Finetune Investigation Required)

## Summary

All kernel tests pass. All W18/W19/W20 infrastructure works. The M1/M2 gate finetune runs to completion but **fails to learn discrimination** — the core contrastive objective does not converge.

---

## ✅ WHAT WORKS PERFECTLY

### Kernel Tests — ALL PASS

**W20 FHT Large Kernel** (`src/rag/tests/w20/test_w20_fht_large.py`):
- 7 passed, 2 skipped (canary/trap variants not built — expected)
- Geometry ladder parity (K2p19-M2, K2p20-M2, K2p22-M1)
- Run-to-run determinism
- ineligible K refusals (Kmod32, nonpow2, KtooSmall)

**W20 TQ Kernels** (`src/rag/tests/w20/test_w20_tq_kernels.py`):
- 11 passed
- Geometry ladder kernel vs torch twin (d2p19-u2, d2p22-u1)
- Torch-only geometry routes to twin
- Batched grid shapes (M120, M192)
- Run-to-run kernel determinism
- Refusals for bad K values and bad dtype
- Device report

**W20 QJL L2 Norm** (`src/rag/tests/w20/test_w20_qjl_l2norm.py`):
- 7 passed
- QJL kernel vs torch parity
- L2 norm vs torch (M8-N1024-float32, M5-N768-float32, M8-N1024-float16)
- Run-to-run determinism
- Flag off non-invasiveness
- Device report

**W20 Torch Path** (`src/rag/tests/w20/test_w20_torch_path.py`):
- 1 skipped (expected — routes to twin path)

**Full Test Suite** (`src/rag/tests`):
- 260 passed, 3 skipped
- All W18/W19 parity gates green
- Device parametric migration complete (tensors on GPU by default)

### Infrastructure — ALL WORKING

- **Kernel compilation**: `python setup.py build_ext --inplace` — exit 0
- **Model loading**: Qwen3.5-9B palettized loads correctly
- **M1/M2 attachment**: 24 linear layers, state (32, 1024, 128) = 4,194,304 elements per memory
- **Gate initialization**: P3 default (write 0.0, read 1.0) + gate_init escape
- **Capture pipeline**: 8 pairs prefilled per batch, ~110-170 MiB retained k/v
- **Training loop**: Runs to completion (300 steps in 573s)
- **Memory management**: No OOM during finetune (fits in 22GB A10G VRAM)

---

## ❌ THE PROBLEM: M1/M2 Fails to Learn Discrimination

### Training Metrics (300 steps)

| Step | Loss | cos(pos) | cos(neg) | Spread |
|------|------|----------|----------|--------|
| 0 | 2.71 | +0.615 | +0.615 | **0** |
| 25 | 1.91 | +0.561 | +0.561 | **0** |
| 50 | 1.40 | +0.495 | +0.495 | **0** |
| 75 | 1.85 | +0.488 | +0.488 | **0** |
| 100 | 2.11 | +0.422 | +0.422 | **0** |
| 125 | 1.76 | +0.471 | +0.471 | **0** |
| 150 | 1.48 | +0.473 | +0.473 | **0** |
| 175 | 2.10 | +0.429 | +0.429 | **0** |
| 200 | 2.33 | +0.368 | +0.368 | **0** |
| 225 | 0.97 | +0.537 | +0.537 | **0** |
| 250 | 2.19 | +0.425 | +0.425 | **0** |
| 275 | 1.82 | +0.407 | +0.407 | **0** |
| **Final** | **—** | **+0.4801** | **+0.4801** | **0** |

### Key Observations

1. **cos(pos) ≡ cos(neg) at every step** — The cosine similarity between M1/M2 states for positive pairs is *identical* to negative pairs at every logged step.

2. **Loss oscillates but doesn't converge** — Loss bounces between 0.97 and 2.33, not descending smoothly.

3. **Gradients are non-zero** — |g| ranges from 0.07 to 0.5, so backprop is flowing.

4. **Held-out spread = 0.0000** — The final validation shows no discrimination signal.

5. **Gates are being updated** — |gk| final = 0.108, so parameters are moving.

### What This Means

The training loop *mechanically* works:
- Model loads ✓
- Captures run ✓
- Forward pass completes ✓
- Backward pass computes gradients ✓
- Optimizer updates parameters ✓
- No crashes, no OOM ✓

But the *contrastive objective* is not being learned:
- Positive pairs (similar texts) and negative pairs (dissimilar texts) produce M1/M2 states with identical cosine similarity
- The discriminator cannot distinguish between similar and dissimilar pairs
- The gate updates don't translate into discriminative state representations

---

## 🔍 ROOT CAUSE HYPOTHESES

### H1: Write Gates Not Modifying State Differentially

The write gates may be updating M1/M2 states in a way that collapses all inputs to similar vectors. Check:
- Is `m_new = m_init + gate * delta` producing near-identical `m_new` for different inputs?
- Are the deltas themselves similar across all pairs?

### H2: Delta Computation Collapse

The delta computed from the capture may be similar for all inputs. Check:
- Are the captured k/v activations distinct across pairs?
- Is the delta projection (scatter into M1/M2 slots) producing diverse vectors?

### H3: Batch Negative Sampling Bug

The in-batch negatives may not actually be negatives. Check:
- How are batch negatives constructed in the InfoNCE loss?
- Are "negatives" actually sampled from different pairs?
- Is there label leakage causing pos/neg to have identical labels?

### H4: M1/M2 State Dimension Too Small

With mem_size=1024 and 32 slots of 128-dim vectors, the representation capacity may be insufficient. Check:
- Does increasing mem_size help?
- Are the slot assignments collapsing to the same indices?

### H5: InfoNCE Implementation Bug

The contrastive loss may be computed incorrectly. Check:
- Is the similarity matrix computed correctly?
- Are the labels correct (positive = same pair, negative = different pair)?
- Is the temperature (tau=0.07) appropriate?

### H6: Shared Gates Across Layers

Write gates are shared over 24 linear layers. The gradient signal may be averaging out to produce uniform updates. Check:
- Do per-layer gates help?
- Is the gradient accumulating correctly across layers?

---

## 📍 INVESTIGATION STARTING POINTS

### 1. Inspect Delta Diversity

```python
# During capture, log delta statistics
deltas = []  # collect deltas for each pair
for pair in batch:
    delta = compute_delta(pair)
    deltas.append(delta)
# Are deltas nearly identical? Check std/mean ratio.
```

### 2. Inspect M1/M2 State Diversity

```python
# After gate update, check state similarity
states = []
for pair in batch:
    state = get_m1m2_state(pair)  # after write gate
    states.append(state)
# Are all states nearly identical? Check pairwise cosine similarities.
```

### 3. Verify Negative Sampling

```python
# In InfoNCE loss, verify negatives are actually different
for i, anchor in enumerate(batch):
    positives = [j for j in batch if same_pair(i, j)]
    negatives = [j for j in batch if not same_pair(i, j)]
    # Are negatives actually different pairs?
```

### 4. Check Similarity Matrix

```python
# Log the similarity matrix during loss computation
sim_matrix = compute_cosine_sim(all_states)  # B x B
# Is sim_matrix nearly uniform? Are diagonal elements (pos) different from off-diagonal?
```

### 5. Isolate Single Layer

```python
# Disable all but one layer's M1/M2 to see if signal exists
for layer in layers[1:]:
    layer.m1m2.disable()
# Train with single layer — does discrimination emerge?
```

---

## 🛠️ RELEVANT FILES

| File | Purpose |
|------|---------|
| `scripts/gpu/finetune_m1m2.py` | M1/M2 gate finetune script |
| `src/rag/m1m2.py` | M1/M2 memory implementation |
| `src/rag/m1m2_finetune.py` | Finetune logic (replay, loss) |
| `data/pairs.jsonl` | Training pairs (33,595 total, 16,076 positive) |
| `m1m2_gates_trained.npz` | Output gates (trained but non-discriminative) |
| `src/rag/tests/w20/test_w20_fht_large.py` | FHT kernel tests |
| `src/rag/tests/w20/test_w20_tq_kernels.py` | TQ kernel tests |
| `src/rag/tests/w20/test_w20_qjl_l2norm.py` | QJL L2 norm tests |

---

## 🧪 REPRODUCTION

```bash
# Activate environment
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/RAGGA

# Run kernel tests (all pass)
python3 -m pytest src/rag/tests/w20/test_w20_fht_large.py -v
python3 -m pytest src/rag/tests/w20/test_w20_tq_kernels.py -v
python3 -m pytest src/rag/tests/w20/test_w20_qjl_l2norm.py -v

# Run full suite (260 passed, 3 skipped)
python3 -m pytest src/rag/tests -q

# Run M1/M2 finetune (completes but spread=0)
python3 scripts/gpu/finetune_m1m2.py \
    --pairs-file data/pairs.jsonl \
    --gates-out m1m2_gates_trained.npz \
    --m1m2-mem-size 1024 \
    --max-steps 300 \
    --refresh-every 4 \
    --gate-init 0.1
```

---

## 📊 SYSTEM SPECS

- **GPU**: NVIDIA A10G (22GB VRAM)
- **CUDA**: 13.2
- **Compute Capability**: sm_86
- **Python**: 3.12.3
- **Model**: Qwen3.5-9B palettized (6.01x compression, 3.5-bit split quant)
- **Virtual Environment**: `/home/ubuntu/venv`

---

## ✅ SUMMARY

**Everything works perfectly except the M1/M2 contrastive learning.**

- Kernels: ✅ All pass
- Tests: ✅ 260 passed, 3 skipped
- Compilation: ✅ Clean
- Model loading: ✅ Works
- Capture pipeline: ✅ Works
- Training loop: ✅ Runs to completion
- Memory: ✅ No OOM
- **M1/M2 discrimination: ❌ Spread = 0 at all steps**

The system is functionally complete and stable. The only remaining work is debugging why the contrastive objective isn't producing discriminative M1/M2 states. The loss surface may have another saddle point, or the state construction may be collapsing diversity.
