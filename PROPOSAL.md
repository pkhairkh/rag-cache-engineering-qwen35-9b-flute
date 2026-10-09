# PROPOSAL.md — How and What to Build

**Status:** the build plan for the cache-engineered RAG system specified in
`SPECIFICATION.md`, grounded in the TurboQuant paper — arXiv:2504.19874,
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh, Daliri, Hadian & Mirrokni; ICML 2025). The uploaded source tar
(`arXiv-2504.19874v1.tar`) was read in full; every number below is pinned to
it. Note: despite the file nickname, this is the **TurboQuant** paper, not
OmniQuant (the offline learnable-clipping method) — TurboQuant is the
*online, data-oblivious* family member, which is exactly why the spec chose
it: no calibration pass, no codebook training, indexing time ~0.

The SPECIFICATION is the contract (the what and why). This proposal is the
how: the paper's algorithms mapped onto our tensors, the five design
decisions that mapping forces, the build phases with gates, and the risk
register. Everything cut from this repo stays cut — no palettizer, no eval
plane, no QLoRA stack; the model artifacts arrive pre-built via
`src/scripts/loader.py::load_quant_model`.

---

## 1. What the paper actually gives us

TurboQuant quantizes a d-dimensional vector to b bits per coordinate with
near-optimal distortion, **online** (no data-dependent tuning), by
random-rotating the vector so that every coordinate marginally follows the
same concentrated Beta distribution, then applying an optimal *scalar*
quantizer per coordinate. Two variants, two objectives:

### 1.1 TurboQuant_mse — the reconstruction-optimized quantizer (their Alg. 1)

```
setup (once):  Π ← random rotation (paper: QR of an iid N(0,1) matrix)
               c_1..c_{2^b} ← Lloyd-Max centroids for the Beta density
                   f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)
               (continuous 1-D k-means on [-1,1]; solved once per bit-width,
                stored — e.g. b=1: ±√(2/π)/√d,  b=2: ±0.453/√d, ±1.51/√d)
Quant(x):      y ← Πx;  idx_j ← argmin_k |y_j - c_k|      (b-bit ints)
DeQuant(idx):  x̃ ← Πᵀ · c_idx
```

Vectors are quantized on the unit sphere; the L2 norm is stored in
floating point and multiplied back at dequantization. Distortion
guarantees (expected, worst-case input, ‖x‖=1):

| b | 1 | 2 | 3 | 4 | general |
|---|---|---|---|---|--------|
| D_mse | ≈0.36 | ≈0.117 | ≈0.03 | ≈0.009 | ≤ (√3π/2)·4⁻ᵇ ≈ 2.72·4⁻ᵇ |

The paper proves (Shannon lower bound + Yao's minimax) that **no**
randomized b-bit quantizer beats D ≥ 4⁻ᵇ, so TurboQuant_mse is within a
2.7× factor of optimal — and within ~1.45× at b=1. At b=4 the
reconstruction error is ~0.9% of the vector norm squared.

### 1.2 TurboQuant_prod — the unbiased inner-product quantizer (their Alg. 2)

MSE-optimal quantizers are **biased** inner-product estimators (at b=1 the
bias is a hard multiplicative 2/π; it decays with b but does not vanish at
low b). Since retrieval is inner products, the paper's second variant
matters to us:

```
Quant(x):      idx ← Quant_mse(x) at b-1 bits            (the base layer)
               r  ← x − DeQuant_mse(idx)                  (the residual)
               qjl ← sign(S·r)                            (1-bit, S iid N(0,1) d×d)
               γ  ← ‖r‖₂                                  (one fp scalar)
DeQuant:       x̃ ← DeQuant_mse(idx) + (√(π/2)/d)·γ·Sᵀ·qjl
Estimator:     ⟨y, x̃⟩ is UNBIASED:  E[⟨y, x̃⟩] = ⟨y, x⟩
```

| b | 1 | 2 | 3 | 4 | general |
|---|---|---|---|---|--------|
| D_prod | ≈1.57/d | ≈0.56/d | ≈0.18/d | ≈0.047/d | ≤ (√3π²·‖y‖²/d)·4⁻ᵇ |

The residual after a (b-1)-bit MSE pass is small, so the single QJL bit
on it buys unbiasedness at low total distortion — the same trick the QJL
paper (their ref. [qjl]) uses, composed with TurboQuant's base layer.

### 1.3 The "3.5-bit" recipe

Non-integer bit-widths in the paper come from **splitting channels** into
an outlier set (higher bits) and a regular set (lower bits), running two
independent TurboQuant instances. Their worked example (d=128 KV
channels): 32 outlier channels at 3 bits + 96 regular at 2 bits →
(32·3+96·2)/128 = **2.5 bits**. The 3.5-bit configuration is described as
"a different ratio"; the split itself is an engineering choice pinned by
measured outlier statistics, not by theory. Two further recipe facts we
inherit: per-vector **norms are stored in fp** (never quantized), and
entropy-coding the indices would save only ~5% at b=4 (3.8-bit
effective) — the paper deliberately skips it; so do we.

### 1.4 The empirical anchors (why the spec trusts 3.5 bits)

- **Needle-In-A-Haystack** (Llama-3.1-8B, 4k–104k tokens, >4× compression):
  TurboQuant 0.997 = full-precision 0.997; beats SnapKV 0.858, PyramidKV
  0.895, KIVI 0.981, PolarQuant 0.995.
- **LongBench-E** (Llama): TurboQuant **3.5-bit average 50.06 = full
  cache 50.06** (absolute quality neutrality); 2.5-bit 49.44 (−0.6).
  KIVI needs 5 bits to match; PolarQuant 3.9 bits gets 49.78.
- **Streaming**: unlike KIVI/PolarQuant, TurboQuant quantizes *during*
  generation (write-once-per-token, read-many KV entries).
- **Indexing time ≈ 0** (data-oblivious): 100k vectors at 4 bits in
  0.0021 s (d=3072) vs PQ 494 s and RabitQ 3957 s. This is the property
  the spec's §8 "Indexing time: ~0" line leans on.
- **NN recall**: TurboQuant-dequantized vectors beat tuned PQ and RabitQ
  at matched bit budgets on d=200/1536/3072 — supporting our use of
  quantized-then-dequantized cache vectors as retrieval vectors.

### 1.5 What the paper does NOT cover (our real engineering exposure)

1. **Recurrent read-modify-write.** The paper quantizes KV entries that
   are written once and read many times. Our S is *recomputed and
   re-written at every decoding step* through the delta rule:
   dequantize → delta-rule update → quantize. Quantization error can
   compound through the recurrence. Nobody has published results for
   that. The §7 fine-tune is our primary mitigation (the model learns to
   live with quantized-state recurrence); the Phase 2 gate measures it
   before anything else is built on top.
2. **Dimension scale.** The paper's largest d is 3072. Our quantization
   units are d = 2¹⁵ (conv_state) and d = 2¹⁹ (S / M1 / M2 per spec §5's
   arithmetic). Concentration only *improves* with d, but see (3).
3. **The rotation must be fast.** The paper's Π (QR of a Gaussian) is a
   dense d×d matmul — O(d²) per call. At d = 524,288 that is 2.7×10¹¹
   flops per state write: impossible. We substitute the randomized
   Hadamard rotation **FHT ∘ diag(±1)** (O(d log d), ~19 butterfly
   stages at 2¹⁹), which the repo already ships as
   `flute_extended/src/kernel_fht.cu` + pure-torch fallback in
   `flute_extended/fht.py` (`fht_apply`, `fht_adjoint`, autograd-aware,
   `build_rotation_matrix` ground-truth for tests). A sign-flipped
   Hadamard is the standard randomized-rotation substitute and keeps the
   uniform-marginal property; Phase 1 validates the Beta concentration
   *empirically on real cache tensors*.
4. **QJL's dense Gaussian S is also d×d** (memory + flops). If we adopt
   the prod variant for retrieval (decision 2.1, option B), the QJL
   projection is likewise substituted with a structured sketch
   (sign of an FHT-based random projection of the residual). The
   unbiasedness argument survives any JL-valid random projection with
   the right moments; the variance constant must be re-measured.
5. **Our states are not worst-case** — that only helps (the bounds hold
   for arbitrary input), but the *outlier structure* of delta-rule
   states, M1/M2 memories and conv states is simply unknown. Measure,
   then pin the 3.5-bit split (Phase 1).

---

## 2. The mapping: five decisions

### D1 — Which variant, where

Two consumers with different objectives:

| Consumer | Objective | Variant |
|---|---|---|
| Runtime cache (S, conv, M1, M2 — read by the delta rule every step) | reconstruction fidelity (generation quality) | **TurboQuant_mse, 3.5-bit split** |
| Retrieval plane (IVFADC preselect + cos-sim rerank) | inner-product fidelity | default: dequantized MSE codes; **A/B: + QJL residual bit (prod)** |

The generation path uses the MSE variant because the paper's
quality-neutrality result (§1.4) is an MSE-variant result. For retrieval
the spec (§4–§5) indexes the *dequantized* cache vector, so by default
retrieval inherits whatever IP bias the 3.5-bit MSE codes carry — at
b≈3.5 the paper's own bias measurements show it is small but nonzero.
The prod variant exists precisely for this, so the build carries it as a
**flagged A/B, not a rewrite**: per quantization unit, optionally store
`qjl = sign(sketch(r))` + `γ = ‖r‖` (one extra bit per coordinate) and
evaluate retrieval recall both ways (Phase 5 gate). Decision rule: if
recall@100 with MSE-dequantized vectors ≥ target, ship without the extra
bit (halves nothing, saves 1/3.5 of the index-side read volume and all
QJL machinery); otherwise flip it on.

### D2 — Quantization units and norm granularity

The spec's disk arithmetic (§3.3/§5) pins the units:

| Tensor | Unit | dims (d) | FHT | Units per chunk |
|---|---|---|---|---|
| S per linear layer | one layer, flattened `(32,128,128)` | 524,288 = 2¹⁹ | 19 stages | 24 |
| conv_state per layer | one layer, flattened `(8192,4)` | 32,768 = 2¹⁵ | 15 stages | 24 |
| M1 (global) | whole tensor `(32,128,128)` | 524,288 = 2¹⁹ | 19 stages | 1 |
| M2 (global) | whole tensor `(32,128,128)` | 524,288 = 2¹⁹ | 19 stages | 1 |

Power-of-two dims by construction — the FHT segments are single blocks,
no binary decomposition needed. Per unit: one fp16 norm scalar (paper's
recipe), one (or two, if the split is A/B'd per-set) b-bit index arrays.
If measured per-head norm variance within an S unit is extreme (>10×
spread), split the unit into 32 per-head sub-units (d=2¹⁴ each, norms per
head) — a Phase 1 measurement decides; per-head is the fallback, not the
default, because it multiplies norm scalars and IO records by 32.

The **3.5-bit split** defaults to a data-oblivious 50/50 coordinate
partition (half the coordinates at 3 bits, half at 4 bits → exactly 3.5,
no calibration, spec's 6 MiB/chunk arithmetic holds exactly). The
paper-style outlier partition (measured heavy coordinates → the 4-bit
set) is the A/B: it is *sticky* per tensor kind (fixed partition learned
once from a calibration sample of rotated states, then frozen — an index
metadata field, not a per-write computation).

### D3 — One shared rotation instance per tensor kind

Every quantization of a given tensor kind — system prompt, chunk deltas,
query, install sums — uses the **same** FHT sign vector (fixed seed per
kind, generated once, persisted in the index metadata). This is load-
bearing for three reasons: (i) install sums dequantized deltas that must
live in the same rotated frame; (ii) the IVFADC index is built over
dequantized vectors in the *original* frame, so frame drift would
silently corrupt retrieval; (iii) it keeps one codebook set and one
partition per kind. Rotation keys: `S→seed_s`, `conv→seed_c`,
`M1→seed_m1`, `M2→seed_m2`, all persisted in `ivfadc_cache.index`
side-metadata. The FHT never leaves the GPU after the first per-device
sign-vector cache fill (~512 KiB per kind at 2¹⁹ int8 signs — the W13 FHT
doc's convention, now at TurboQuant scale).

### D4 — The delta protocol and install math

Ingestion stores **quantized deltas relative to the system-prompt state**
(the toy-validated path-independent convention):

```
reset point:   S_sys ← prefill(system) codes                (quantized once)
per chunk i:   S⁺ ← dequant(S_sys codes); prefill(chunk_i)   (online TQ:
               delta_i ← S⁺⁺ − dequant(S_sys)                  every write
               store TQ.quant(delta_i)                          re-quantizes)
install:       S_inst ← TQ.quant( dequant(S_sys)              (§6, dequant-
                    + Σ_i dequant(delta_i) )                    sum-requant)
```

The sum runs in the rotated frame where possible (FHT is linear, the
scalar quantizer is per-coordinate, and all deltas of a kind share one
rotation — so dequant → sum → single requant is exactly "sum in the
rotated space" from the spec's §6 note, done once, not per addend).
Lloyd-Max quantization is not additive (Q(a+b) ≠ Q(a)+Q(b)), which is
why install requantizes the *sum*, never sums the *codes*. conv_state is
not summed: the spec pins "use the last retrieved chunk's" (§6). M1/M2
deltas sum identically (additive gated writes → path-independent, toy
§12). Fallback if the recurrent-compounding gate (Phase 2) fails hard:
quantize-on-snapshot only (fp16 during forward, TQ at chunk boundaries)
— disk layout and retrieval plane unchanged, spec §3.2's "ALWAYS codes"
relaxed to "always on disk"; documented, decided by data, not default.

### D5 — The index stack

FAISS IVFADC over the dequantized 13.6M-dim cache vectors, exactly as
spec §8: `IndexIVFPQ(quantizer=IndexFlatIP(13631488), nlist=224, m=64,
nbits=8)`, nprobe=8 → top-100, exact cos-sim rerank on full dequantized
vectors (loaded on demand from TQ codes, ~6 MiB per chunk) → top-3. Two
paper facts license this stack: TurboQuant's dequantized vectors beat
tuned PQ at matched bits in recall (§1.4), and indexing is ~0 because
the *codes* need no training (the IVFPQ coarse/PQ stages do train once
on 50k vectors — a one-time CPU cost, amortized over the index's life;
the spec's "Indexing time: ~0" refers to the TQ side). The 13.6M-dim
vectors are never materialized in RAM en masse: IVFPQ stores 64-byte
compressed entries; exact vectors exist only as TQ codes on disk,
dequantized for the top-100 rerank one candidate at a time (~55 MiB
fp32 transient per vector, or batched to the VRAM budget — §10's
ledger).

---

## 3. The build: phases, gates, files

New code lands under `src/rag/` (one flat package, mirroring the spec's
§2–§9 sections):

```
src/rag/
  turboquant.py     # TQ core: units, FHT rotation binding, Quant/DeQuant,
                    #   3.5-bit split, norm handling, code serialization
  codebooks.py      # Lloyd-Max solver (continuous 1-D k-means on the Beta
                    #   density), on-disk cache per bit-width, validation
  tq_cache.py       # online cache wrapper: monkey-patched DynamicCache
                    #   read/write paths (§3.2), S/conv/M1/M2 code stores
  m1m2.py           # the architectural addition (§2.2): buffers, gated
                    #   additive writes, softmax(q @ M1ᵀ) @ M2 reads
  hooks.py          # the 9-hook capture harness (§1 pattern)
  ingest.py         # delta-protocol ingestion (§5), batch driver
  index.py          # IVFADC build + preselect + rerank (§8)
  query.py          # query flow + install + answer (§6)
  finetune.py       # the lean §7 loop (see Phase 4)
  evals.py          # phase gates: round-trip MSE, streaming neutrality,
                    #   retrieval recall, end-to-end QA, timing/VRAM ledger
```

### Phase 0 — Bring-up (GPU box)

Build `flute_extended` per `src/docs/DEPLOY.md` (ptxas gate, then the
debug_simple differential spot-check with the verified `pack_idxn` API).
Load the model through `src/scripts/loader.py::load_quant_model`
(artifacts + separate heads dir), greedy-generate a fixed prompt set,
record baseline outputs and step times. **Gate:** kernel numerics match
the reference path on the spot-check; baseline generations archived.

### Phase 1 — TurboQuant core (pure torch, CPU-legal)

`codebooks.py`: solve the continuous 1-D k-means (Lloyd-Max iteration on
the Beta density at d=2¹⁵/2¹⁹ — the density is effectively N(0,1/d);
integrate analytically between Voronoi midpoints, iterate to fixed
point). **Gate:** b=1 centroids reproduce the paper's ±√(2/π)/√d; b=2
reproduces ±0.453/√d and ±1.51/√d; distortion matches D_mse ≈
0.117·(‖x‖²) at b=2 on random unit vectors to within 5%.

`turboquant.py`: unit wrappers (norm → FHT `fht_apply` with the kind's
signs → partition → bucketize to centroids → pack codes; dequant the
reverse via `fht_adjoint`). Round-trip gates on **real tensors** — run
the unmodified model on sample chunks, capture S/conv via the §1 hooks,
then: (i) empirical per-coordinate distribution of the rotated states vs
the Beta density (concentration check — the substitution risk of D3);
(ii) round-trip MSE per unit vs the paper's b=3/4 numbers scaled by the
measured norms; (iii) per-head norm spread (decides D2's norm
granularity); (iv) outlier mass (decides the 3.5-bit split A/B). **Gate:**
all four measurements recorded in `evals.py` output with pass/fail lines;
nothing downstream is pinned until this table exists.

### Phase 2 — Online cache wrapper (the compounding gate)

`tq_cache.py`: wrap `transformers`' `DynamicCache` per spec §3.2 —
intercept `layers[L].recurrent_states[0]` reads (dequantize on read) and
state-update writes (quantize on write) for the 24 linear layers, same
for conv_states, plus the M1/M2 slots. The model's forward is untouched.
**Gate (the paper cannot vouch here — §1.5.1):** stream-generate the
Phase 0 prompt set with the online-TQ cache at 3.5 bits; measure greedy
token-match rate vs the fp16-cache baseline and the degradation vs
prompt length (the compounding signature). Target: ≥95% token match on
≤4k prompts. If it fails: outlier-partition A/B, per-head norms, and
only then the quantize-on-snapshot fallback (D4) — each step re-runs
this gate.

### Phase 3 — M1/M2 addition + 9-hook capture

`m1m2.py`: add the two global memories per spec §2.2 — `(1,32,128,128)`
buffers shared across all 24 linear layers, gated additive writes,
`softmax(q @ M1ᵀ) @ M2` reads, wired into `modeling.py`'s
`Qwen3_5GatedDeltaNet` output path behind a config flag (default on for
RAG, off for parity tests). Zero-init the write gates at first (M1/M2
start as no-ops — the untrained model's behavior is bit-unchanged; the
fine-tune opens them). `hooks.py`: the 9 capture points of spec §1
(after layer 0; after each full-attn layer 3,7,...,31 — 3 linear layers
each). **Gate:** with zero gates, generations are bit-identical to
Phase 0 baselines; capture → quantize → dequantize round-trip on the
hooks' S tensors passes Phase 1's MSE table.

### Phase 4 — The fine-tune (spec §7, lean loop)

`finetune.py`: next-token prediction on the OfficeQA corpus with the
online-TQ cache active and M1/M2 attached. Trainables: the 24 per-layer
linear-attn param groups (S discriminative), the M1/M2 read/write gates
(carry info), and the W10 LUTs via `PalettizedLinear.make_trainable()`
on the reference path (`forward="reference"` per the loader) — the
straight-through primitive this repo kept for exactly this. ~500 steps,
AdamW + cosine, batch to fit A10G (fp32 LUT masters, everything else
fp16/bf16). **Gate:** (i) loss curve sane, no divergence; (ii) the
cache-signal check — cosine between cache vectors of same-topic chunks
vs different-topic chunks separates by ≥3× margin vs pre-fine-tune
(this is the "S is discriminative / M1/M2 carry info" acceptance made
measurable); (iii) greedy quality on the Phase 0 prompts does not
regress >1% token-match.

### Phase 5 — Ingestion + index

`ingest.py` + `index.py`: the delta protocol of D4 over the corpus
(system-prompt reset point, per-chunk delta codes + retrieval vector),
then `IndexIVFPQ` training/add, side-metadata = rotation seeds +
partitions + codebook hashes. **Gate:** retrieval@100 / rerank@3 recall
on held-out same-topic queries ≥ target (95% mechanics-level: the query
chunk's own neighbors retrieved; the toy's 100% is the CPU ceiling);
the D1 prod-variant A/B runs here — record recall both ways, pick per
the decision rule. Throughput: 50k chunks at spec's per-chunk cost on
one A10G (ingestion is embarrassingly parallel over chunks; batch it).

### Phase 6 — Install + end-to-end

`query.py`: the full §6 flow (tokenize → online-TQ prefill → preselect →
rerank → load 3 chunks → dequant-sum-requant install → answer → decode).
**Gates:** (i) OfficeQA end-to-end accuracy vs a no-RAG baseline and vs
an oracle-retrieval variant (install the *correct* chunk's codes) — the
gap oracle-vs-actual isolates retrieval quality from generation quality;
(ii) the §9 timing ledger (targets: prefill+snapshot ~8 ms, preselect
~10 ms, rerank ~20 ms, load ~3 ms, install ~15 ms, decode ~8 s);
(iii) the §10 VRAM ledger (~13 GiB on A10G).

### Phase 7 — Hardening (only if gates demand)

CUDA-graph the decode-step write path (quantize-on-write is in the hot
loop: FHT 2¹⁹ + bucketize + pack per S unit per step — µs-scale on GPU,
but graph-capture it once the numerics are frozen); mmap the snapshot
store; optional entropy coding of indices (paper: ~5% at b=4 — skip
unless 300 GiB → 285 GiB matters); the prod-variant structured-QJL if
Phase 5 chose it.

---

## 4. Risk register

| # | Risk | Exposure | Mitigation / decision point |
|---|---|---|---|
| R1 | Recurrent RMW compounding (§1.5.1) | generation quality degrades with length | Phase 2 gate before anything else; outlier split → per-head norms → snapshot-only fallback (D4); fine-tune (P4) is the structural fix |
| R2 | FHT rotation ≠ QR-Gaussian concentration | Beta assumption weaker on real S | Phase 1 empirical concentration table; Hadamard sign-flip is the standard substitute; measured, not assumed |
| R3 | MSE-variant IP bias hurts retrieval | recall below target | D1's prod-variant A/B at Phase 5 (+1 bit/coordinate, structured QJL) |
| R4 | Delta-rule S states are low-rank / spiky | per-unit norm scalar too coarse; distortion concentrated | Phase 1 per-head norm spread measurement → D2 fallback (32 sub-units per layer) |
| R5 | M1/M2 cold start | retrieval vector dominated by untrained noise pre-P4 | P4 ordering (fine-tune before index build); M1/M2 dims excluded from the index until gates open (index rebuild is cheap: TQ codes are already on disk) |
| R6 | IVFPQ at 13.6M dims (m=64, 8-bit) | coarse quantization error at extreme d | nprobe=8 + exact rerank on top-100 is the spec's design; recall gate at P5; raise nprobe/m if short |
| R7 | 300 GiB snapshot store | disk budget, IO latency | 6 MiB × 3 per query is ~3 ms on NVMe per §9; if cold storage, page-cache the top clusters |
| R8 | Fine-tune destabilizes W4 LUTs | LUT masters drift off the fp16 grid | straight-through + fp16 snap on export (freeze_lut); reference-path training; 500-step budget |

---

## 5. Acceptance criteria (summary)

| Spec ref | Criterion | Measured at |
|---|---|---|
| §3.1/§3.3 | 3.5-bit TQ codes, ~6 MiB/chunk, 4.6× vs fp16 | P1/P5 |
| §3.2 | online TQ: quantize-on-write / dequantize-on-read, model unaware | P2 |
| §3.1 | streaming quality: ≥95% greedy token-match vs fp16 cache (≤4k) | P2 |
| §2.2 | M1/M2 added, no-op at init, trained at P4 | P3/P4 |
| §12 (real model) | cache-vector same/different-topic cosine margin ≥3× | P4 |
| §5/§8 | retrieval: recall@100 ≥95%, top-3 correct ≥ toy's 90% trend | P5 |
| §6 | install = dequant-sum-requant; conv = last chunk | P6 |
| §9 | per-query ledger ≤ ~8.1 s total | P6 |
| §10 | VRAM ≤ ~13 GiB on A10G 24 GiB | P6 |
| §7 | ~500-step fine-tune: linear-attn params + M1/M2 gates + W10 LUTs | P4 |

The build order is deliberately evidence-gated: nothing downstream of a
failed gate gets pinned. Phase 1's measurement table is the single most
important artifact of the whole plan — every quantization hyperparameter
(norm granularity, split ratio, variant choice) is decided there, on the
real model, before the 50k-chunk ingestion ever runs.
