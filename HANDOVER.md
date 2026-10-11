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

## W20.14 — the M1/M2 discrimination metric: the roll bug (CLOSED 2026-10-12)

The post-finetune handover read "M1/M2 fails to learn discrimination —
cos(pos) ≡ cos(neg) at every step, held-out spread 0.0000". The root
cause is a METRIC identity, not a learning failure: `cosine_spread` v1
computed the (B,) true-pair cosine vector and reported the "negatives"
as that SAME vector rolled by one. A roll is a permutation; a
permutation's mean is the same mean; so cos(pos) == cos(neg) EXACTLY,
at every step, for ANY model — the metric was structurally incapable
of reporting a spread. The InfoNCE loss was always correct (the full
B×B matrix, diagonal targets — unchanged this wave), and the box's
below-ln(8) losses (1.40, 0.97, ...) were real ranking signal the
metric hid. The loss oscillation is the capture pool cycling
(refresh_every=4) sampled at log_every=25 — 25 mod 4 = 1 walks the
pool phases; artifact, not divergence.

What landed (W20.14.1–.5, tag `w20-wave-14`):

1. **cosine_spread fixed** (`src/rag/m1m2_finetune.py`): the negatives
   are now the OFF-DIAGONAL mean of the same B×B sim matrix the loss
   trains on (shared `_pair_sim_stats` helper) — the metric scores
   exactly the discrimination the loss optimizes.
2. **Trainer instrumentation**: every history entry carries
   cos_pos/cos_neg/spread/off_max/diag_min + the pool phase; the log
   line prints them (`m1m2_gates_trained_history.json` gets the same
   fields — the box's post-hoc analysis reads them).
3. **The H6 knob**: `--train-layers '5'` / `'0-11,16'` /
   `'all'` — write-gate layer-subset training (masked gates
   bit-anchored: zero grads + weight_decay 0). The
   isolate-single-layer experiment is one flag per run.
4. **Regression gates** (`test_w19_finetune.py`, AUTHORED — PENDING on
   the box): test_w19f_7 (the rigged batch: pos≈1, neg≈0 — v1 reported
   neg == pos == 1.0), test_w19f_8 (history fields, pool values, the
   mask's bit-anchor contract, the out-of-range refusal).

**The data verdict** (the handover's question — `data/pairs.jsonl`,
audited with a pure-stdlib read on the dev box): GOOD. 33,595 rows /
16,076 positive / 17,519 negative; zero schema problems; zero
duplicate rows; zero same-pair conflicting labels; zero cross-order
duplicates; balanced 47.9/52.1; short QA-style members (median 17-22
words — nothing truncates at 256 tokens); unique 29,398/30,292 texts
with only mild hub reuse (max ×15-19); in-batch text-sharing windows
2/2009 (0.1% — negligible false-negative risk); uniform decile profile
(pos-rate 0.46-0.50, stable lengths — the positional val split is
sound); 13 self-pairs. Two honest caveats: 912 train-pair members
(3.2%) share an exact text with the val split (mildly optimistic
held-out numbers — dedup if it matters), and the 17,519 label-0 rows
are UNUSED by the v1 loader (in-batch negatives only) — free held-out
eval capacity, carried as the wave's open item.

**The box re-run sequence** (no rebuild — the wave touches no CUDA):

```
cd /home/ubuntu/RAGGA
git pull --ff-only origin main          # >= W20.14.5
source /home/ubuntu/venv/bin/activate
python3 -m pytest src/rag/tests/w20 -q          # unchanged, all green
python3 -m pytest src/rag/tests -q              # 260 + the 2 new W20.14 gates

# the re-measure (the SAME command as the handover's failing run):
python3 scripts/gpu/finetune_m1m2.py \
    --pairs-file data/pairs.jsonl \
    --gates-out m1m2_gates_trained.npz \
    --m1m2-mem-size 1024 \
    --max-steps 300 \
    --refresh-every 4 \
    --gate-init 0.1
```

What to expect, and what each number now means:

* **cos(neg) is no longer cos(pos) with a label** — it is the mean
  off-diagonal sim (the exact negatives the loss trains against).
  EXPECT: cos(pos) in the 0.4-0.6 band you already saw, cos(neg)
  below it, spread > 0 growing through the run. If the spread is
  positive and rising, the §7 objective is doing what the spec wants
  and the prior "0.0000" was the metric.
* **spread/off_max/diag_min + pool in every log line and in
  `m1m2_gates_trained_history.json`** — worst-neg vs weakest-pos is
  the per-step margin; pool 1/4..4/4 explains the loss sawtooth
  (fresh capture = loss jump; same pool = descent).
* **The H6 experiment**, if wanted: rerun with `--train-layers 5`
  (or any subset) and compare the final spread — per-layer signal
  vs the shared-gate average, one run per subset.

File the numbers verbatim (history json + the [5] held-out line);
if the held-out gap is still ~0 with the fixed metric, THEN it is a
real learning question and the next wave starts from the deltas
(hypotheses H1/H2 — the state construction itself), not the metric.

## W20.15 — the sawtooth decomposed: the regime fixed, the ceiling made measurable (CLOSED 2026-10-12)

The W20.14.5 re-run did what it was asked: cos(neg) finally differs
from cos(pos), the spread is real (+0.1297 held-out, 0.080 at init),
and the loss shows genuine below-chance ranking (53/75 capture blocks
below ln 8). "Doesn't look good" is also correct — here is exactly
why, decomposed from your own history json (every number below is
measured, not argued):

1. **The sawtooth is BATCH IDENTITY, not divergence.** The loss block
   means range 0.40..2.98 and jump ±2.1 nats at every capture refresh
   (every 4 steps); the within-block learning slope is -0.003
   nats/step — **the between-block variance is 7,595x the
   within-block variance**. Each block trains on ONE captured batch
   of 8 pairs; its difficulty is that batch's own pairs' cross-side
   alignment (cos_pos swings 0.35..0.74 across blocks while cos_neg
   sits stable at ~0.35). Nothing is diverging; the loss is measuring
   the batch, not the model.
2. **The regime starved the signal.** 300 steps at B=8/refresh 4
   consumed **600 of 14,469 train pairs (4.1%)** — one epoch never
   completed — and the v1 iterator (`_cycle_batches`) walks ONE fixed
   file order forever, so every pair's in-batch negatives (the
   InfoNCE's entire negative signal) were its file-layout neighbors
   for the whole run. The pre-clip |gv| swung 160x with the batch
   (0.016..2.578) — each batch yanked the 48 gate scalars its own
   way.
3. **The plateau ~0.12 is the gate-reweighting share.** The gates'
   real, generalizable effect was suppressing the common background
   (cos_neg 0.535 -> 0.343); the spread grew 0.08 -> 0.13 and then
   sat. The likely ceiling: the write is **position-addressed**
   (slot = p mod mem_size — and at T < 1024 every slot holds exactly
   ONE token, so a side's state delta IS its key sequence, and the
   state cosine is the position-ALIGNED key correlation). Paraphrase
   pairs with aligned surfaces (MRPC/STS-B style) can separate;
   question~answer-sentence pairs and word-order-adversarial PAWS
   pairs cannot, no matter the gate weights — 48 scalars reweight
   layers, they cannot realign positions.

What landed (W20.15.1–.5, tag `w20-wave-15`):

1. **The regime** (`epoch_batches`): per-epoch re-batching — every
   epoch redraws every pair's in-batch negatives (deterministic per
   seed); the driver's val split is a seeded random sample (was the
   file's head block); the coverage tally prints the number the v1
   run never showed; `--fixed-batches` restores the v1 order for the
   A/B.
2. **The observability**: every history entry carries the gate VALUES
   (gk/gv min/max/absmean — where the gates ARE; your run logged 300
   |g| values and zero locations) + the capture-batch ordinal (the
   block id — group the sawtooth by it).
3. **The instruments** (`lexical_overlap` + `bucketed_spread` + the
   label-0 control line): the [5] report is now a LADDER —
   cos(true pairs) vs **cos(label-0 curated pairs)** (the 17,519
   non-pair rows your loader skipped: the datasets' own curated
   negatives, a cleaner floor than in-batch randoms) vs cos(in-batch
   neg) — and the gap bucketed by the pair's lexical overlap
   (high/med/low).

**The data verdict (your standing question, unchanged from W20.14's
audit, now with its value realized): GOOD — clean, balanced,
deduplicated, 7 sources. The label-0 half was free capacity sitting
unused; it is now the control line of every [5] report, and is the
natural hard-negative pool for a future training wave.**

The box re-run sequence (no rebuild — the wave touches no CUDA):

```
cd /home/ubuntu/RAGGA
git pull --ff-only origin main          # >= W20.15.5
source /home/ubuntu/venv/bin/activate
python3 -m pytest src/rag/tests -q      # 260 + the 5 new W20.15 gates

# THE REGIME RUN (B=16 halves the block-difficulty variance and doubles
# the negatives; 600 steps doubles coverage to ~2,400 pairs; ~35-40 min):
python3 scripts/gpu/finetune_m1m2.py \
    --pairs-file data/pairs.jsonl \
    --gates-out m1m2_gates_trained.npz \
    --m1m2-mem-size 1024 \
    --batch-pairs 16 \
    --max-steps 600 \
    --refresh-every 4 \
    --gate-init 0.1
```

What to expect, and how to read it:

* **The sawtooth shrinks ~30%** (the block-mean variance halves at
  B=16) and every epoch re-draws the negatives — the loss trend, if
  there is one, becomes visible above the block noise. The coverage
  line should read ~2,400 pair-uses (~17%).
* **The [5] ladder is the decision instrument.** File all four
  numbers. Then read them:
  - **`true` >> `label-0`** (gap > ~0.2) and the gap LIVES IN THE
    HIGH-OVERLAP BUCKET ONLY → the position-lock ceiling is
    CONFIRMED: the states separate surface-aligned pairs and cannot
    separate semantics. The next wave is the §2.2 slot-addressing
    conversation (an architecture change — e.g. content-keyed slots
    or sub-T slot wrapping), NOT more gate steps.
  - **Flat buckets** (low-overlap gap ~= high-overlap gap ~= overall)
    → the signal is NOT surface-locked; then it is
    capacity/optimization: `--batch-pairs 32`, more steps, and the
    label-0 hard-negative pool (a masked-InfoNCE loss change, next
    wave).
  - **`true` ~= `label-0`** (both well above in-batch) → the pairs
    barely beat curated negatives — weak but real graded signal;
    same direction as the flat-bucket case.
* A/B rows, if wanted: `--fixed-batches` (the v1 order — the
  sawtooth's cause, isolated); `--tau 0.1` (softens the
  hardest-negative domination that the 0.40..2.98 blocks show);
  `--lr 5e-3` if |gv| still spikes >1 at B=16.

File the numbers verbatim (history json + the [5] ladder); the next
wave starts from the bucket profile, not from the loss curve.
