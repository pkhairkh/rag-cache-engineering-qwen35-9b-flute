# RAGGA Handover Document

## Current State (post-W15 — the paper-fidelity wave)

| item | status |
|---|---|
| CPU suite | **191 tests green** (179 + 12 new W15 split tests) |
| paper audit | **DONE** — arXiv:2504.19874 read line-by-line; every TQ code path checked against Alg. 1 / Alg. 2 / §5.3 |
| conv write-path | **FIXED (3x)** — the paper's outlier-channel split replaces the fixed coordinate half-split (measured: rel-MSE 0.017-0.025 -> 0.007 at the SAME effective bits) |
| S write-path | **AT the paper's floor** — 0.022 = exactly the Lloyd-Max blend (0.0345 b=3 + 0.0095 b=4); nothing to fix within 3.5-bit MSE quant |
| install math | **EXONERATED** — the dequant-sum-requant chain is only 1.3x single-shot (P2); the sum was never the problem |
| decode loop | **EXONERATED** — the roundtrip is idempotent (0.3%/step; re-quantizing a dequantized vector re-hits the same lattice) |
| QJL (Alg. 2) | **PLUMBED** — `TQCache(qjl=True)` now reaches the whole flow (was unreachable from the cache stack) |
| GPU action | **RE-INGEST + re-run the ladder** (commands below) |

---

## The W15 Diagnosis (why "the TQ layer is fucked up")

W14 proved every READ-level check passes while the full install generates
garbage. W15 read the paper line-by-line, audited every TQ code path, and
measured the WRITE-path distortion (the thing the read checks cannot see —
they compare the cache read against the SAME codes' reference dequant,
never against the true tensor). Three deviations from the paper, one
exonerating finding:

### 1. The 3.5-bit recipe was NOT the paper's recipe (the big one)

The paper (§LongBench, the KV-cache results the repo chases): non-integer
bits come from "splitting channels into outlier and non-outlier sets, and
applying two independent instances of TurboQuant to each, allocating
higher bit precision to outliers."

The repo (D2): a FIXED 50/50 coordinate split — first half 3 bits, second
half 4 — data-oblivious to the channel-energy structure. The conv window
(1x6144x4 mixed_qkv — exactly the paper's KV-cache setting) carries
lognormal channel outliers; the fixed split hands the 4 bits to a fixed
coordinate RANGE that ignores where the energy is.

**Measured (production geometry, channel-structured windows):**

| recipe | eff bits | write-path rel-MSE |
|---|---|---|
| repo fixed half split (segmented 24,576 rotation) | 3.5 | 0.017-0.025 |
| paper outlier split (top-k channels at 4 bits, own frame+norm; rest at 3) | 3.25 (!) | **0.007** |

The paper's recipe is ~3x LOWER distortion at FEWER effective bits.

### 2. The conv rotation was block-diagonal (W11's cost)

The paper's Alg. 1 rotates ONCE with a full random rotation — every
coordinate of a unit vector lands at variance 1/d, which is what the
Lloyd-Max codebook assumes. The W11 24,576 unit rotates as TWO blocks
(16,384 + 8,192) that never mix: block coords carry variance rho_b/b,
matching the codebook only when energy splits proportionally. Measured:
the segmented policy costs +25-35% distortion vs a full rotation at
every outlier level.

**The W15 fix folds both:** the split's two sub-instances each get their
own POWER-OF-TWO full rotation (12,288 -> 16,384 sub-units), so the
paper's invariant holds per sub-set, and magnitude-homogeneous sub-sets
make the block statistics benign by construction.

### 3. QJL (Alg. 2) was unreachable

The repo's W9.2 qjl flag existed on TurboQuant but was never plumbed
through TQCache/resolve_quantizer/install — the A/B contract could not
run end-to-end. W15 plumbs it: `TQCache(..., qjl=True)` (and
`--qjl` on every GPU script). The install's requant now inherits the
cache's setting.

### 4. Exonerated: the math the previous waves suspected

- **codebooks.py**: faithful to the paper — the scaled-density Lloyd-Max
  solver, the 1/sqrt(d) law, symmetric Voronoi layout; measured
  mse_per_variance 0.0345 / 0.0095 = the paper's b=3/4 constants.
- **S single-shot**: 0.022 rel-MSE = EXACTLY the 3.5-bit blend floor.
  The full 2^19 rotation mixes everything; no channel structure to
  exploit; nothing to fix at 3.5-bit MSE.
- **install sum** (dequant-sum-requant): 1.3x single-shot, not 4x.
- **online decode loop**: idempotent (0.3%/step) — re-quantizing a
  dequantized vector re-hits the same lattice. G2's pass was real.
- **the read paths**: W14 was right; they were never the owner.

### The residual noise budget (what G5 actually carries)

The installed S state is ~0.08-0.11 rel-MSE from the true doc state —
dominated by the ONLINE INGESTION drift (the chunk prefill evolves from
a 2.2%-noisy reseeded state through 300 tokens of recurrence) times the
install chain (1.3x). The conv window was 0.02+ and is now ~0.007.
true-doc (0% noise) generates at conf 0.895; every TQ variant sat at
0.28-0.43 — the model is real-model sensitive to state noise, and the
fixed-split conv was the marginal tipper in the W14 matrix (s-only and
conv-only each passed; full failed only with BOTH doc-loaded).

---

## The W15 Implementation (same architecture, no fallbacks)

**`src/rag/turboquant.py`** — `TurboQuant(group=...)`: the paper's
outlier split. group > 1 + non-integer bits => quant() routes to
`_quant_split`: top-k energy channels (k = round(frac*n_ch), stable
argsort) at bits_hi with their OWN pow2 full rotation (seed+555) and own
fp32 norm; the rest at bits_lo with the kind's seed. TQCodes carries
`partition="outlier"`, `group`, `mask` (packed channel bits),
`norm_hi`. **Codes are self-describing** — dequant routes on
codes.partition:
  - outlier codes through the sub-quantizers;
  - legacy "half" codes through the flat twin (pre-W15 snapshots decode
    UNCHANGED);
  - a flat-configured quantizer reading outlier codes routes through the
    split twin (hooks/index/bisect paths stay correct).

**`src/rag/tq_cache.py`** — `_init_conv` resolves the conv quantizer
with `group = kernel`; the conv_codes setter and every generic reader
resolve from the codes' own group; `TQCache(qjl=...)` plumbed to every
layer quantizer.

**`src/rag/install.py`** — the requant inherits the cache's qjl setting;
conv codes still install VERBATIM (§6: conv is never summed); the report
records the installed conv partition.

**`src/rag/snapshot.py`** — the mask/group/norm_hi roundtrip
(all-or-none per unit, refused loudly when partial); the sha256 digest
covers the mask (tampering flips it); pre-W15 files load unchanged
(the fields are optional).

**Committed codebooks**: cb_b{3,4}_d{16384,8192}.npz (the production
split sub-units), all with the exact paper constants.

**GPU scripts**: bisect_install.py gains (a) the PURE frame check —
adjoint-roundtrip + kernel-vs-reference at 1e-4 with NO codebook noise
floor (the old check's ~0.02 floor could hide a subtle kernel drift);
(b) the TRUE-dist row — the installed dequant vs the RAW doc state the
true-doc control materializes (the write-path gate the W14 matrix
lacked); (c) `--qjl` and `--split-half` A/Bs. verify_pipeline.py and
run_query.py gain `--qjl`.

---

## GPU Box: What To Run (W15 verification)

```bash
cd /home/ubuntu/RAGGA && git pull
python3 -m pytest src/rag/tests -q          # 191 must pass

# 0) the pure frame check (no quant noise floor) — run once on the OLD
#    disk to close the kernel question for good:
python3 scripts/gpu/bisect_install.py

# 1) RE-INGEST (the conv codes change geometry: partition=outlier;
#    S/M1/M2 codes are UNCHANGED — same frame, same math):
python3 scripts/gpu/run_ingestion.py    # same corpus, fresh disk dir
python3 scripts/gpu/run_index.py        # rebuild (vectors unchanged in
                                        # content: S codes identical)

# 2) the ladder:
python3 scripts/gpu/verify_pipeline.py
python3 scripts/gpu/bisect_install.py   # now with the TRUE-dist row

# 3) the W15 A/Bs (the paper's levers):
python3 scripts/gpu/bisect_install.py --split-half   # conv recipe A/B
python3 scripts/gpu/verify_pipeline.py --qjl          # Alg.-2 A/B
```

### Reading the new bisect matrix

```
variant      S-read   conv    TRUE     gen      conf   rep
true-doc     -        -       -        OK       0.895  0.00   (control)
reseed       OK       OK      OK       OK       ...
s-only       OK       OK      OK       ...
conv-only    OK       OK      OK       ...
full         OK       OK      OK?      ???      ...    ...
```

- **TRUE HIGH with reads OK** => the write path owns the distortion
  (codes decode fine, encode badly) — the recipe A/Bs are the lever.
- **full gen OK** => the W15 fix closed G5 (expected: the conv window's
  write distortion drops 3x; full's marginal S+conv interaction was the
  W14 cliff).
- **full still GARBAGE with TRUE OK** => the residual is the S-path
  ingestion drift (~0.08-0.11, the online recurrence amplification) —
  then run `--qjl` (the paper's Alg. 2: ~1.6x MSE improvement for +1
  bit/coordinate — the D1 A/B the Phase-5 gate was designed to decide).

### Model Details (unchanged)

- Qwen3.5-9B palettized — /home/ubuntu/qwen3_5_9B_palettized (+ _heads)
- vocab_size 248,320; EOS <|im_end|> (id 248,046)
- 32 layers ([L,L,L,F]x8 — 24 linear + 8 full-attention)
- GDN geometry: k_heads 8 / v_heads 32, head dims 128
- conv window 24,576 = 6,144x4 (W15: split sub-units 12,288 -> 16,384 x2)
- S = 524,288 per layer (single 2^19 full rotation — the paper's invariant)

## Test Command

```bash
cd /home/ubuntu/RAGGA && python3 -m pytest src/rag/tests -q --tb=short
```

191 tests pass on CPU (179 pre-W15 + 12 new:
test_turboquant_split.py — split fidelity vs flat at the same budget,
group=1 bit-parity, serialization/digest/tamper, cross-semantics reads
both directions, the layer path, the install path, construction guards).
