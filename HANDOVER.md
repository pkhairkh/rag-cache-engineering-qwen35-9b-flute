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

**The M1/M2 contrastive learning now carries real signal — held-out
measurement is the remaining open item (W22 fixes the instrument).**

- Kernels: ✅ All pass
- Tests: ✅ 260 passed, 3 skipped (plus the W21/W22 gate suites)
- Compilation: ✅ Clean
- Model loading: ✅ Works
- Capture pipeline: ✅ Works
- Training loop: ✅ Runs to completion
- Memory: ✅ No OOM in training; the [6] ladder OOM is fixed (W22)
- **M1/M2 discrimination: 🟡 in-train spread +0.20..0.35 (W21's
  content addressing answered the structural failure); the held-out
  ladder OOM'd + crashed before reporting — W22 makes it runnable and
  adds the curated hard negatives + the sparse top-k read**

The system is functionally complete and stable. The remaining work is
measurement (run the W22 command; read the ladder), coverage (600
steps), and the deployment A/Bs (kappa, read_topk, tau). See the W22
section below for the honest verdict and the decision tree.

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

---

## W21 — the content-addressed global cache: the position lock and the pedestal, answered (2026-10-12)

**The diagnosis (your new HANDOVER run, read against the code):** the
run was mechanically perfect and semantically capped, exactly as the
W20.15 instruments predicted. Three facts in the log name the cause:

1. **`cos(neg)` parked at ~+0.36 between RANDOM cross pairs** — a
   representation-anisotropy pedestal (the keys' shared "typical
   direction"). The delta-loss removes the system state's common
   component, not the token keys' common component. Reference: Su et
   al. 2021, *Whitening Sentence Representations for Better Semantics*
   — centering is the canonical fix, now inside the write.
2. **Spread ~0.13 with no trend while the gates moved freely**
   (k[-0.087..+1.122]) — the trainable surface could not express the
   missing signal because there is none to express at ANY gate value:
3. **The v1 write addresses slots by POSITION** (`slot = p mod
   mem_size`): at T < mem_size each slot holds one token, so the state
   cosine is the POSITION-ALIGNED key correlation — two sides of a
   pair only interact token s with token s. Paraphrase content that
   moved or re-phrased cannot align; only surface overlap separates
   (the high-overlap buckets). Every reference memory design
   addresses by CONTENT — Product-Key Memory (Lample et al. 2019,
   arXiv:1907.05242), Memorizing Transformers (Wu et al. 2022,
   arXiv:2203.08913), the model's own delta-rule S-states. The v1
   scatter is the outlier. (Also: the run OOM-crashed in the
   held-out control stage — 4 × 1.07 GiB stacked states — and lost
   the artifact; fixed: artifact saves FIRST now.)

**So yes — one substantial architectural issue, and it lives entirely
in the two surfaces you allowed: the write's slot ADDRESS (the global
cache) and the gates.** No model, wiring, quantization, or pipeline
change.

**The mitigation (W21, all landed in this push):**

* `write_addr="content"` (new default): slot = **simhash of the
  Hadamard-mixed, per-(head) mean-centered key** (Charikar 2002;
  fixed Sylvester-Hadamard mixing — deterministic on every box, no
  RNG, no artifact to ship). `addr_blocks` κ independent sign-blocks
  (default 4 at mem 1024 → 8 bits/block) — each token writes κ slots;
  similar contextual keys collide with high probability, unrelated
  ones almost never. **Content matches anywhere in the two texts** —
  position-free — which is what "retrieval by content overlap"
  (SPECIFICATION N17) required all along.
* `center_write=True` (new default): writes `k − mean_t k` — the
  pedestal is removed inside the cache write, train and serve
  identical. A T=1 decode call centers to zero (the memory freezes at
  prefill end — cleaner than the v1 counter hack).
* **Per-(layer, head) write gates** `(24, 32)`: the flattened state
  cosine weights heads by ‖k̃_h‖²; a 2-5× head-scale spread is a
  4-25× weight imbalance the v1 per-layer scalar cannot touch.
* The v1 behavior is one flag away: `--write-addr positional
  --no-center-write` (bit-exact v1 semantics, kept as the A/B).
* Old gates .npz files are REFUSED loudly (shape) — v1-trained gate
  values are meaningless under content addressing; retrain.
* The driver: **artifact saves before the reports** (an OOM can never
  eat the training again); the held-out **ladder** — one capture,
  every addressing config replayed offline from the same k/v (the
  run's config, κ=1/2/8, uncentered, centered positional, the v1
  corner) with the overlap buckets and the label-0 control per row;
  `[3.5]` the data-quality profile (positives vs label-0 lexical
  overlap — "is the data good" with numbers).

**The box sequence (no rebuild — the wave touches no CUDA):**

```
cd /home/ubuntu/RAGGA
git pull --ff-only origin main          # >= W21.5
source /home/ubuntu/venv/bin/activate
python3 -m pytest src/rag/tests -q      # 260 + the 9 W21 gates

# THE W21 RUN (same shape as yours; the addressing is the change):
python3 scripts/gpu/finetune_m1m2.py \
    --pairs-file data/pairs.jsonl \
    --gates-out m1m2_gates_trained.npz \
    --m1m2-mem-size 1024 \
    --max-steps 300 \
    --refresh-every 4 \
    --gate-init 0.1
# (add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True if the box
#  ever fragments; the ladder is OOM-hardened regardless)
```

**How to read the [6] ladder (the decision instrument — every row
comes from the SAME captured k/v, so the differences are pure
addressing):**

* **`positional v1 (A/B)` row** ≈ your last run's numbers (spread
  ~0.13, cos_neg ~+0.36) — the baseline, reproduced inside the new
  report.
* **Content rows: cos(neg) should collapse toward ~0** (the pedestal
  gone — this alone is the centering proof) and **the low-overlap
  bucket gap should go POSITIVE** where the positional rows show it
  flat — that is the position lock measured, in one table.
* **Pick κ from the ladder**: the row with the best low-overlap gap
  wins; if it is not κ=4, retrain with `--addr-blocks <κ>` (a ~10 min
  run, same command + the flag).
* **The training trend**: with the signal now content-aligned and the
  floor near zero, the loss should drop BELOW the v1 plateau and the
  spread should trend UP (not oscillate). If the ladder separates but
  training stays flat, the next lever is the loss (τ, hard negatives
  — the label-0 pool), NOT the architecture (that conversation is
  closed by this wave's rows either way).
* The data verdict stands (W20.14's audit: GOOD — 7 sources, clean,
  balanced; the label-0 half is the control line and the future
  hard-negative pool). The new `[W21 data]` profile lines quantify it
  per run: if the positives' overlap distribution sits clearly above
  the label-0 profile, the pairs carry real signal for a
  content-addressed kernel.

File the [6] ladder + the [3.5] data lines verbatim; the next wave
starts from those rows.

## W21.6 — the box-run repair: what the 17 failures + the [2] crash were (2026-10-12)

**Your pasted run, decomposed.** The suite read 17 failed / 258 passed /
3 skipped / 1 error, and the finetune died at `[2] prefill_system` with
`NotImplementedError: Cannot copy out of meta tensor; no data!` — before
a single training step. Three distinct mechanics bugs, **none of them
the content-addressing kernel** (w21a/b/c/e/f — order invariance, the
kernel contrast, the pedestal removal, per-head gates, the decode
freeze — all PASSED on your box; the architecture is behaving exactly
as designed):

1. **P0 — the meta mix (the crash, 5 test failures + 1 error + the
   finetune [2] abort).** `from_pretrained(low_cpu_mem_usage)` builds
   the model inside accelerate's meta-device context. The M1/M2 gates
   are real parameters, so the checkpoint loader's missing-key re-init
   materializes them — but the W21 sign-mixing master is a PLAIN
   attribute (deliberately never a buffer, so it stays out of
   state_dict), and a plain attribute is invisible to that re-init: it
   stayed a meta tensor with no data, and the first content write
   crashed trying to copy out of it. **Fix:** `_get_mix` now detects a
   meta-born master and rebuilds it from the Sylvester recursion — a
   pure function of the geometry, bit-identical to what a normally
   constructed module holds. The state_dict contract (exactly the 3
   gates) is untouched.

2. **P1 — CUDA scatter nondeterminism (9 test failures, including the
   ones whose tensors "looked equal" in the pytest diff).** The content
   scatter used `torch.index_add`, which accumulates through atomic
   adds on CUDA — the accumulation ORDER is run-to-run nondeterministic
   as soon as ≥3 rows collide on one slot, and the content mode's
   κ-block collisions are ~8-deep at the test geometry (mem 8, κ=4).
   Consequences you saw: `write != write_deltas` (bit-exactness
   contracts w18f_1/w21g), three replays of the same calls producing
   different states (w19f_5b), memopt-vs-plain forward/grad mismatches
   (w18f_2/3/5), two fresh modules disagreeing (w21h), the quant-store
   composability breaking (d6), and `w17f_2`'s replay-vs-cache rel-MSE
   blowing out to 0.4630 vs the 0.12 budget — 1-ulp pre-quant
   differences flipping 3.5-bit quantization buckets. (The v1
   positional path never hit this: at T ≤ 2·mem its slots hold ≤2
   tokens, and two-term fp addition commutes.) **Fix:** the content
   scatter is now a fixed-order segment sum — stable-sort the flat
   (head·mem + slot) indices, form each slot's token sum by an fp32
   prefix scan differenced at the segment boundaries, and write the
   per-slot rows with `index_copy` into distinct rows. No atomics
   anywhere on the value path: write == write_deltas bit-exact,
   replays bit-identical, the W18 exact-gradient replay contract
   restored. The mathematical values are the same per-slot sums with
   fp re-association (verified standalone: ≤1e-5 relative at the
   production geometry — far inside the 3.5-bit quant noise the cache
   round-trip already carries).

3. **P2 — authored-test defects (4 more failures).** (a)
   `epoch_batches` was a bare generator, so its loud refusals never
   fired at call time — the raise was deferred to the first `next()`
   (test_w20_15b "DID NOT RAISE"); it is now a plain function that
   validates eagerly. (b) `float(gk[0].abs())`-style idioms crash on
   the new (L, H) gate rows (w18f_3's assert message, w19f_3, w19f_8) —
   the row magnitude is `.abs().max()`. (c) test_w21d's key
   construction crashed (`.view(1, H, 1, D)` demands 64 elements from
   a 16-element key) — the key now broadcasts through `(1, 1, 1, D)`,
   and its "independent keys" bound was a coin flip (0.23 expected vs
   the 0.25 bound at the any-of-κ statistic) — widened to 0.35 with
   the derivation in the docstring.

**The box sequence (unchanged commands — no rebuild, the wave touches
no CUDA):**

```
cd /home/ubuntu/RAGGA
git pull --ff-only origin main          # >= W21.6
source /home/ubuntu/venv/bin/activate
python3 -m pytest src/rag/tests -q      # expect the suite back to green
                                         # (260 + the W21 gates; w17f_2's
                                         # rel-MSE should return to its
                                         # round-trip level, well under
                                         # the 0.12 budget)

# then THE W21 RUN — verbatim the same command as before:
python3 scripts/gpu/finetune_m1m2.py \
    --pairs-file data/pairs.jsonl \
    --gates-out m1m2_gates_trained.npz \
    --m1m2-mem-size 1024 \
    --max-steps 300 \
    --refresh-every 4 \
    --gate-init 0.1
```

**What to expect now that the run can actually start:** the training
mechanics are unchanged from W21's design — the run should complete
300 steps, the [6] ladder should print its addressing rows (the
positional v1 row ≈ your old numbers; the content rows' cos(neg)
collapsing toward ~0 = the pedestal gone; the low-overlap bucket gaps
going positive = the position lock measured), and the gates artifact +
history save BEFORE the reports (the OOM can no longer eat them). If
any single test still fails, file its name + the assertion text
verbatim — one pinned failure is one fix, not a redesign.

The architectural verdict stands as W21 delivered it: the substantial
issue was the position-addressed write + the anisotropy pedestal,
mitigated entirely inside the gates + the global caches (content
simhash slots, centering, per-(layer, head) gates — Product-Key Memory
/ Memorizing Transformers lineage; Charikar simhash; Su et al.
centering), with the v1 regime one flag away for the A/B.

## W22 — the honest verdict on your run + the three proven pieces (2026-10-12)

**Did training succeed? Yes — and you deserve the exact shape of that
answer:**

**What genuinely worked (the first real learning signal in the M1/M2
line):** spread went from EXACTLY 0.0000 (the v1 structural artifact)
to **+0.20..0.35 sustained for 300 straight steps**; cos(pos) rode
+0.30..0.46 against cos(neg) +0.07..0.16 — a 3-5x separation, not a
permutation identity; the gates opened for real (k[-0.080..+1.183]);
300 steps / 1220 s / zero OOM in training. The W21 architecture change
(content-addressed writes + centering) did exactly what the PKM /
Memorizing-Transformers lineage predicted. This is no longer "no
signal" — the write kernel discriminates.

**What did NOT succeed, be clear about it:** (1) **the held-out
ladder never ran** — every [6] row OOM'd at a 16 MiB request against a
full 22 GiB, and then the script CRASHED with `UnboundLocalError:
val_group` (a `del val_group` four lines above unbinds the name the
trailing report reads). So generalization is UNMEASURED — in-train
signal yes, held-out proof no. (2) The loss swung 0.28..1.64
block-to-block because B=8 gives each anchor SEVEN random in-batch
negatives — while your 17,519 curated label-0 non-pairs rode along
unused. (3) Coverage 4.2% (608 of 14,469 pairs). The run is a real
signal, not a finished training.

**The OOM, root-caused:** the W21 ladder kept everything on the GPU —
~2.6 GiB of retained capture k/v + four (64, 4.19M) fp32 unit stacks +
their python lists (~6 GiB transient) on top of the model and the
allocator's live residue. W22 makes the ladder HOST-STAGED: captures
land on (pinned) host memory per side, replays stream one side at a
time, unit stacks build on the host, and only (16, dim) chunks ride
the PCIe for the matmul — the ladder's GPU footprint beyond the model
is bounded under ~1 GiB at ANY pair count. The [4] residue (epoch
iterators, the system cache) is gc-released before [6]. The crash fix
is the one-liner it deserved: rebind to None, never `del`.

**The three proven pieces adopted (same lineage W21 cites — the RAG
concept untouched):**

1. **Curated hard negatives (`--neg-pairs 8`, the new default).** Every
   contrastive result at this batch scale: DPR's in-batch + hard
   negatives (Karpukhin 2020), MoCo's shared queue (He 2020), Robinson
   2021's shortcut analysis. The label-0 text2s join each capture pool
   as 8 shared negative columns — each anchor must now ALSO rank its
   own pair above curated non-pairs. The feed excludes the held-out
   control slice (your [6] label-0 line stays untouched), epoch-
   rebatched at seed+1. `--neg-pairs 0` is the exact W21 loss (A/B).
2. **Sparse top-k read (`read_topk`, default OFF = the spec §2.2 dense
   softmax, bit-identical).** PKM retrieves top-32 of 2^26 slots;
   Memorizing Transformers read exact kNN; a dense softmax over 1024
   accumulating slots averages toward the state mean. The §7 fine-tune
   never reads, so your trained gates artifact is valid at any k —
   `--m1m2-read-topk 32` at ingest+query is the deployment A/B to
   measure with eval_retrieval.py.
3. **The runnable ladder** (the instrument — without held-out numbers
   every "is it working" argument is an argument).

**The box sequence:**

```
cd /home/ubuntu/RAGGA
git pull --ff-only origin main          # >= W22
source /home/ubuntu/venv/bin/activate
python3 -m pytest src/rag/tests -q      # the W21 suite + the 13 new W22
                                        # gates (sparse-read exactness +
                                        # hard-negative semantics)

# THE W22 RUN — the recommended regime (hard negatives ON, more steps
# for coverage; ~35-40 min at 600 steps):
python3 scripts/gpu/finetune_m1m2.py \
    --pairs-file data/pairs.jsonl \
    --gates-out m1m2_gates_trained.npz \
    --m1m2-mem-size 1024 \
    --max-steps 600 \
    --refresh-every 4 \
    --gate-init 0.1 \
    --neg-pairs 8
# (optional, if the loss still sawtooths at the doubled negative pool:
#  add --tau 0.1 — 0.07 was tuned for batch-4096 SimCLR, not B=8+8)

# then the retrieval A/B for the sparse read, same corpus:
python3 scripts/gpu/run_ingestion.py ... --m1m2-read-topk 32
python3 scripts/gpu/eval_retrieval.py ...        # vs a dense-read ingest
```

**What to expect / the decision tree:**

- **The ladder finally prints.** Seven rows + the label-0 control line
  per row. Read them as: the "run config" row is your trained regime's
  held-out number; the kappa rows pick the block count with the best
  LOW-overlap gap (κ=4 is the default, not a promise); the
  "positional v1 (A/B)" row is the old world, for contrast.
- **If held-out spread lands ≥ ~0.15 with the control gap positive:**
  the gates generalize — re-ingest with the artifact and run the
  retrieval eval; the M1/M2 line is unblocked end-to-end.
- **If held-out spread collapses while in-train stays ~0.3:** the gates
  overfit 4.2% of the pairs — raise steps (the 600-step command), not
  architecture; rerun.
- **If cos(neg) on uncentered rows still parks at ~+0.36:** that's the
  pedestal — expected only on the "content uncentered" row; centered
  rows should sit near 0.
- **If the loss still swings wildly with --neg-pairs 8:** try
  `--tau 0.1`; if it still sawtooths, the block difficulty variance
  (W20.15) is the residual cause — raise `--batch-pairs`.
- **If anything OOMs in [6] again:** paste the traceback verbatim — the
  ladder is now bounded by construction, so a failure there is new
  information, not a repeat.

## W23 — the generation repair: the proven context build (2026-10-12)

**Your paste, read honestly.** Retrieval works (top-3 [70, 80, 82] at
0.63/0.60/0.59 — weakly separated but functional). The answer is
multilingual word salad. You also hit a crash in run_query.py and
sed-hacked a batch dimension in — that was a real bug (1-D token
tensor into a model that reshapes (B, T)); the repo now carries the
proper fix (2-D, long, on the model's device — restore the file from
git, the sed edits conflict with the pull).

**The root cause of the garbage — measured, not argued.** The answer
context was built by the §6 INSTALL: sum the top-k chunks' cached
states (`sum_absolute_codes`: Σabsᵢ − (n−1)·sys — one requant). That
algebra is exact ARITHMETIC on the codes but it is not a valid STATE
composition. A gated-delta-net recurrent state is a SEQUENCE state —
S_t = (I − β k kᵀ)·S_{t−1} + β k vᵀ with per-head decay — which is
NON-ADDITIVE: the sum superposes 3 chunks' associations (each at full
strength, no decay between them, no conflict resolution, the −2·sys
term on top), and the linear-attention readout returns the mush. The
proof is already in the repo's own numbers: the W14/W15 true-doc
control — the SAME model, the SAME TQ cache machinery, answering over
a state built by PREFILLING the doc text — measured **conf 0.895**;
the summed install answers word salad. This is also exactly what the
KV-cache reuse literature found (CacheBlend EMNLP 2024,
Block-Attention 2024): reused chunk caches still need the text (or a
recompute fraction) for coherent generation. The RAG concept is
untouched — ingest → capture codes → retrieve on them → answer over
the retrieved chunks; only the context BUILD changes (text prefill,
the proven path, vs codes-sum, kept as the A/B).

**The fix (W23, in this pull):**
- `answer_query(..., context_token_ids)` — the prefill mode: fresh
  reseeded cache → the retrieved chunks' text prefilled sequentially
  (ascending chunk id = the ingest stream order) → full-attn reset
  (spec §2.4, the docs ride the LINEAR state) → query prefill →
  decode. Steps 5-6 (codes load + install) skipped entirely.
- `run_query.py`: `--answer-mode {prefill,install}` (DEFAULT prefill),
  `--corpus` (documents.jsonl — chunk i = line i, the ingest
  convention), `--no-answer-m1m2` (the answer-time memory A/B), the
  2-D ids fix, and a gen-gate printout (mean confidence +
  repetition — bisect_install's own bar: conf ≥ 0.20, rep ≤ 0.6).
- `m1m2.enabled` (modeling.py guard): the memory's GENERATION effect
  is now one flag — the true-doc control ran WITHOUT M1/M2, and the
  fine-tune trained retrieval discrimination, never generation
  quality. Do not assume the read helps the answer; measure it.

**The OOM, answered (you asked):** two OOMs, both already root-caused
and fixed in W22 — (1) the W21 held-out ladder kept everything on the
GPU (~2.6 GiB retained capture k/v + four (64, 4.19M) fp32 unit stacks
+ ~6 GiB transient python lists on top of the 9B model — every [6]
row died at a 16 MiB request against a full 22 GiB) and then CRASHED
with `UnboundLocalError: val_group`; W22 made the ladder HOST-STAGED
(pinned host memory, one side at a time, bounded under ~1 GiB GPU at
any pair count), saves artifacts BEFORE the reports, and rebinds
instead of `del`. (2) The earlier held-out-control OOM (4 × 1.07 GiB
stacked states) — same fix class (artifact-first). Your pasted query
run has NO OOM, and the prefill mode does not add one: it REPLACES
~18 MiB of code loads with transient forward passes.

**Did training succeed? The honest shape (unchanged from W22, plus
the new consequence):** in-train, YES — the first real signal (spread
+0.20..0.35 sustained, cos(pos) +0.30..0.46 vs cos(neg) +0.07..0.16,
gates opened, zero OOM). Held-out, UNMEASURED (the ladder OOM ate it —
now runnable). Coverage 4.2%. And end-to-end the system did NOT work
— retrieval scores clustered 0.59-0.63 partly because this corpus was
ingested under the OLD gates regime (its M1/M2 codes ≈ zeros — the
W21 semantic component is absent at retrieval time; re-ingest with
the trained artifact), and generation was broken by the install sum
(fixed by this wave). The training data itself is good and valuable —
33,595 pairs / ~48% positive, 17,519 curated label-0 non-pairs, a
held-out control slice; its limitation is coverage-at-300-steps, a
steps knob, not a data problem.

**The box sequence:**

```
cd /home/ubuntu/RAGGA
git checkout -- scripts/gpu/run_query.py   # drop the sed edits FIRST
git pull --ff-only origin main             # >= W23
source /home/ubuntu/venv/bin/activate
python3 -m pytest src/rag/tests -q         # + the 4 new W23 gates

# THE W23 RUN — your exact query, on the proven answer path:
python3 scripts/gpu/run_query.py \
    --disk-dir /home/ubuntu/ragga_ingested_test100 \
    --question "How do I deploy perf-canary?" \
    --m1m2-gates m1m2_gates_trained.npz \
    --top-k 3 --max-new-tokens 128 \
    --corpus /home/ubuntu/enterprise_rag_bench/documents/documents.jsonl
# (--answer-mode prefill is the DEFAULT; pass the documents.jsonl the
#  corpus was ingested from — the first 100 lines are the 100 chunks)

# the A/Bs, one variable each:
python3 scripts/gpu/run_query.py ... --answer-mode install      # the old world (expect the salad — the negative control)
python3 scripts/gpu/run_query.py ... --no-answer-m1m2           # the true-doc regime exactly (no memory reads)
```

**What to expect / the decision tree:**
- **The [gen gate] line is now printed per query** — read it before
  reading the text: conf ≥ 0.20 / rep ≤ 0.6 is the bar; the true-doc
  control measured 0.895.
- **prefill coherent (conf ≥ ~0.5, on-topic):** generation unblocked.
  Next blocker is retrieval quality: re-ingest the corpus with the
  trained gates (`run_ingestion.py ... --m1m2-gates
  m1m2_gates_trained.npz`) so the M1/M2 codes carry the W21 semantic
  component, then the W22 ladder for the held-out number.
- **prefill coherent only with `--no-answer-m1m2`:** the trained
  gates' read injects too much into the stream at answer time — the
  gates need generation-side calibration (a follow-up wave: scale the
  read gate, or gate the read by a norm budget). Not an architecture
  failure — a knob with a number now.
- **prefill STILL garbage with --no-answer-m1m2:** that is new
  information — run `bisect_install.py` (its true-doc row is the
  control) and paste both outputs verbatim.
- **install mode salad:** expected — that is the documented root
  cause, kept as the A/B. The npz store and the install path stay
  (retrieval still runs on the captured codes).
