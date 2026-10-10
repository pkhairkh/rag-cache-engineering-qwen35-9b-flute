# PROPOSAL.md — The Design Layer

Purpose: HOW the cache-engineered RAG contract in `SPECIFICATION.md` maps onto the TurboQuant paper — arXiv:2504.19874, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh, Daliri, Hadian & Mirrokni; ICML 2025): paper facts, design decisions D1–D5, build phases P0–P7, risk register R1–R8, acceptance.
Authority: subordinate to `SPECIFICATION.md` (the contract; cited by clause N##); execution waves and wave statuses live in `TASKS.md`.
Status: v1 semantics preserved, format rewritten Wv2-7.1; the CPU-side code under `src/rag/` is done (tags `cpu-code-complete` + `base-synced-v1.3`, 161 tests green); phases P0–P7 = GPU box pending; cuts stay cut — no palettizer, no eval plane, no QLoRA stack (model artifacts pre-built per N1).

## 1. Paper facts

| fact | value |
|---|---|
| identity | TurboQuant — the ONLINE, data-oblivious family member, NOT OmniQuant (offline learnable clipping); the arXiv-2504.19874v1 source tar was read in full, every number below pinned to it |
| core method | random-rotate the d-dim vector so every coordinate marginally follows the same concentrated Beta density, then apply the optimal per-coordinate scalar quantizer — b bits per coordinate, near-optimal distortion, online (no data-dependent tuning) |
| why the spec chose it (N10) | no calibration pass, no codebook training, indexing time ~0 (N31) |
| optimality | no randomized b-bit quantizer beats D ≥ 4⁻ᵇ (Shannon lower bound + Yao's minimax, worst-case input, ‖x‖=1); TurboQuant_mse is within a 2.7× factor of optimal, ~1.45× at b=1 |

### 1.1 TurboQuant_mse — the reconstruction quantizer (paper Alg. 1)

| step | operation |
|---|---|
| setup (once per bit-width) | Π ← random rotation (paper: QR of an iid N(0,1) matrix); c_1..c_{2^b} ← Lloyd-Max centroids of the Beta density f_X(x) = Γ(d/2)/(√π·Γ((d−1)/2))·(1−x²)^((d−3)/2) on [−1,1] (continuous 1-D k-means, solved once, stored); b=1: ±√(2/π)/√d; b=2: ±0.453/√d, ±1.51/√d |
| Quant(x) | y ← Πx; idx_j ← argmin_k \|y_j − c_k\| (b-bit ints); x on the unit sphere, ‖x‖ stored in fp, multiplied back at dequantization |
| DeQuant(idx) | x̃ ← Πᵀ · c_idx |

| b | 1 | 2 | 3 | 4 | general |
|---|---|---|---|---|---|
| D_mse (expected, worst-case, ‖x‖=1) | ≈0.36 | ≈0.117 | ≈0.03 | ≈0.009 (error ≈ 0.9% of ‖x‖²) | ≤ (√3π/2)·4⁻ᵇ ≈ 2.72·4⁻ᵇ |

### 1.2 TurboQuant_prod — the unbiased inner-product quantizer (paper Alg. 2)

| step | operation |
|---|---|
| Quant(x) | idx ← Quant_mse(x) at b−1 bits (base layer); r ← x − DeQuant_mse(idx) (residual); qjl ← sign(S·r) (1-bit, S iid N(0,1) d×d); γ ← ‖r‖₂ (one fp scalar) |
| DeQuant | x̃ ← DeQuant_mse(idx) + (√(π/2)/d)·γ·Sᵀ·qjl |
| estimator | ⟨y, x̃⟩ is UNBIASED: E[⟨y, x̃⟩] = ⟨y, x⟩ |

| b | 1 | 2 | 3 | 4 | general |
|---|---|---|---|---|---|
| D_prod | ≈1.57/d | ≈0.56/d | ≈0.18/d | ≈0.047/d | ≤ (√3π²·‖y‖²/d)·4⁻ᵇ |

MSE-optimal quantizers are biased inner-product estimators: at b=1 the bias is a hard multiplicative 2/π; it decays with b but stays nonzero at low b. The (b−1)-bit residual is small, so a single QJL bit on it buys unbiasedness at low total distortion — the QJL trick composed with the TurboQuant base layer.

### 1.3 The 3.5-bit recipe

| element | value |
|---|---|
| mechanism | non-integer widths = split channels into an outlier set (higher bits) + a regular set (lower bits); two independent TurboQuant instances |
| worked example (paper, d=128 KV channels) | 32 outlier channels at 3 bits + 96 regular at 2 bits → (32·3 + 96·2)/128 = 2.5 bits |
| 3.5-bit configuration | "a different ratio" — the split is an engineering choice pinned by measured outlier statistics, not by theory |
| norms | per-vector norms stored in fp, never quantized |
| entropy coding | would save only ~5% at b=4 (3.8-bit effective) — the paper skips it; skipped here too |

### 1.4 Empirical anchors

| benchmark | setup | result |
|---|---|---|
| Needle-In-A-Haystack | Llama-3.1-8B, 4k–104k tokens, >4× compression | TurboQuant 0.997 = full-precision 0.997; SnapKV 0.858, PyramidKV 0.895, KIVI 0.981, PolarQuant 0.995 |
| LongBench-E | Llama | TurboQuant 3.5-bit average 50.06 = full cache 50.06 (absolute quality neutrality); 2.5-bit 49.44 (−0.6); KIVI needs 5 bits to match; PolarQuant 3.9 bits → 49.78 |
| streaming | — | TurboQuant quantizes DURING generation (write-once-per-token, read-many KV entries), unlike KIVI/PolarQuant |
| indexing time | 100k vectors, 4 bits, d=3072 | TurboQuant 0.0021 s vs PQ 494 s vs RabitQ 3957 s — data-oblivious; the fact behind N31's "indexing time ~0" |
| NN recall | d=200/1536/3072, matched bit budgets | TurboQuant-dequantized vectors beat tuned PQ and RabitQ — licenses dequantized cache vectors as retrieval vectors (D5; N17, N21) |

### 1.5 What the paper does NOT cover (the engineering exposure)

1. **Recurrent read-modify-write.** The paper quantizes KV entries written once and read many times; S is recomputed and re-written at every decoding step (dequant → delta-rule update → quant), so quantization error can compound through the recurrence — no published results exist. Mitigation: the fine-tune (N26–N29); the Phase 2 gate measures it before anything is built on top.
2. **Dimension scale.** The paper's largest d is 3072; the quantization units here are d=2¹⁵ (conv_state) and d=2¹⁹ (S / M1 / M2 per the spec §2 geometry). Concentration only improves with d — but see item 3.
3. **Rotation speed.** The paper's Π (QR of a Gaussian) is a dense d×d matmul, O(d²) per call — at d=524,288 that is 2.7×10¹¹ flops per state write. Substitution: the randomized Hadamard rotation FHT ∘ diag(±1), O(d log d), ~19 butterfly stages at 2¹⁹ — the standard substitute, keeps the uniform-marginal property; shipped as `src/flute_extended/src/kernel_fht.cu` + pure-torch `src/flute_extended/fht.py` (`fht_apply`, `fht_adjoint`, autograd-aware, `build_rotation_matrix` ground truth for tests); Phase 1 validates the Beta concentration empirically on real cache tensors.
4. **QJL density.** The prod variant's dense Gaussian S is also d×d (memory + flops). If D1 option B is adopted, the QJL projection is likewise substituted with a structured sketch — sign of an FHT-based random projection of the residual; unbiasedness survives any JL-valid random projection with the right moments, but the variance constant must be re-measured (Phase 5).
5. **Unknown outlier structure.** Delta-rule S, M1/M2 and conv states are not worst-case (the bounds hold for arbitrary input) but their outlier structure is unpublished. Measure, then pin the 3.5-bit split (Phase 1).

## 2. The mapping: design decisions D1–D5

### D1 — which variant, where

| consumer | objective | variant |
|---|---|---|
| runtime cache (S, conv, M1, M2 — read by the delta rule every step) | reconstruction fidelity (generation quality) | TurboQuant_mse, 3.5-bit split |
| retrieval plane (IVFADC preselect + cos-sim rerank, N30–N32) | inner-product fidelity | default: dequantized MSE codes; A/B: + QJL residual bit (prod) |

| field | value |
|---|---|
| decision | MSE for the generation path (the §1.4 quality-neutrality result is an MSE-variant result); retrieval defaults to the dequantized MSE cache vector (N17, N21) |
| options | A: MSE-dequantized retrieval vectors (default) · B: + `qjl = sign(sketch(r))` and `γ = ‖r‖` per quantization unit (one extra bit per coordinate) — a flagged A/B, not a rewrite |
| chosen default | A — at b≈3.5 the paper's own bias measurements show the MSE IP bias small but nonzero |
| A/B trigger | recall@100 with MSE-dequantized vectors < target at Phase 5 |
| gate | Phase 5 (src/rag/evals.py recall): record recall both ways; if A ≥ target, ship without the extra bit (saves 1/3.5 of the index-side read volume + all QJL machinery); otherwise flip B on |

### D2 — quantization units and norm granularity (units pinned by the spec §3.3 disk arithmetic, N21)

| tensor | unit | d | FHT stages | units per chunk |
|---|---|---|---|---|
| S per linear layer | one layer, flattened (32,128,128) | 524,288 = 2¹⁹ | 19 | 24 |
| conv_state per layer | one layer, flattened (8192,4) | 32,768 = 2¹⁵ | 15 | 24 |
| M1 (global) | whole tensor (1,32,128,128) @ mem_size=128 | 524,288 = 2¹⁹ | 19 | 1 |
| M2 (global) | whole tensor (1,32,128,128) | 524,288 = 2¹⁹ | 19 | 1 |

| field | value |
|---|---|
| decision | power-of-two dims by construction → single-block FHT, no binary decomposition; per unit one fp16 norm scalar (the paper's recipe) + one or two b-bit index arrays (two if the split is A/B'd per set) |
| options | whole-unit norm vs 32 per-head sub-units (d=2¹⁴ each, norms per head); data-oblivious 50/50 split vs paper-style measured outlier partition |
| chosen default | whole-unit norms + 50/50 coordinate partition — half the coordinates at 3 bits, half at 4 bits → exactly 3.5; no calibration; the spec's 6 MiB/chunk arithmetic holds exactly |
| A/B trigger | per-head norm spread within an S unit extreme (>10×) → sub-units (the fallback, not the default: multiplies norm scalars + IO records by 32); Phase 1 outlier-mass measurement → outlier partition (sticky per tensor kind — a fixed partition learned once from a calibration sample of rotated states, then frozen; an index-metadata field, not a per-write computation) |
| gate | Phase 1 (src/rag/evals.py roundtrip): measurements (iii) per-head norm spread and (iv) outlier mass decide both |

### D3 — one shared rotation instance per tensor kind

| field | value |
|---|---|
| decision | every quantization of a given tensor kind — system prompt, chunk deltas, query, install sums — uses the SAME FHT sign vector: fixed seed per kind, generated once, persisted in the index metadata |
| options | none — forced by correctness: (i) install sums dequantize deltas that must live in the same rotated frame (N22–N23); (ii) the IVFADC index is built over dequantized vectors in the ORIGINAL frame (N21, N30) — frame drift would silently corrupt retrieval; (iii) one codebook set + one partition per kind |
| chosen default | rotation keys S→seed_s, conv→seed_c, M1→seed_m1, M2→seed_m2, all persisted in ivfadc_cache.index side-metadata |
| A/B trigger | none (a per-write rotation would break the delta protocol) |
| gate | Phase 1 concentration table (the substitution-risk check); seeds pinned by tests |
| residency | the FHT never leaves the GPU after the first per-device sign-vector cache fill (~512 KiB per kind at 2¹⁹ int8 signs) |

### D4 — the delta protocol and install math

| field | value |
|---|---|
| decision | ingestion stores QUANTIZED deltas relative to the system-prompt state (the toy-validated path-independent convention, N5/N7/N34) |
| protocol | reset point: S_sys ← prefill(system) codes (quantized once); per chunk i: S⁺ ← dequant(S_sys codes), prefill(chunk_i) online, delta_i ← S⁺⁺ − dequant(S_sys), store TQ.quant(delta_i); install: S_inst ← TQ.quant(dequant(S_sys) + Σ_i dequant(delta_i)) — the N23 contract |
| options | online TQ during recurrence (every write re-quantizes, N15) vs quantize-on-snapshot only (fp16 during forward, TQ at chunk boundaries) |
| chosen default | online; Lloyd-Max quantization is not additive (Q(a+b) ≠ Q(a)+Q(b)) — install requantizes the SUM, never sums the codes; the sum may run in the rotated frame (FHT linear, per-coordinate quantizer, one rotation per kind → dequant → sum → single requant = the N24 optimization, done once, not per addend) |
| sub-decisions | conv_state is never summed — the last retrieved chunk's codes (N20, N22); M1/M2 deltas sum identically (additive gated writes → path-independent) |
| A/B trigger | the Phase 2 compounding gate fails hard |
| gate | Phase 2 (src/rag/evals.py streaming); the fallback keeps disk layout + retrieval plane unchanged, with N14's "ALWAYS codes" relaxed to "always on disk" — documented, decided by data, not by default |

### D5 — the index stack

| field | value |
|---|---|
| decision | FAISS IVFADC over the dequantized 13.6M-dim cache vectors, per N30: IndexIVFPQ(quantizer=IndexFlatIP(13631488), nlist=224, m=64, nbits=8), nprobe=8 → top-100; exact cos-sim rerank on full dequantized vectors, loaded on demand from TQ codes (~6 MiB per chunk) → top-3 (N31–N32) |
| options | nprobe=8 + exact top-100 rerank (default) vs raise nprobe/m if recall is short (R6) |
| chosen default | the default stack; two paper facts license it: (1) TurboQuant-dequantized vectors beat tuned PQ at matched bits in recall (§1.4 NN-recall row); (2) indexing ~0 because the CODES need no training — the IVFPQ coarse/PQ stages do train once on 50k vectors (one-time CPU cost, amortized over the index's life; N31's "indexing ~0" refers to the TQ side) |
| A/B trigger | recall@100 < target at Phase 5 → raise nprobe/m (R6) or flip D1 option B |
| gate | Phase 5 (src/rag/evals.py recall) |
| memory shape | the 13.6M-dim vectors are never materialized in RAM en masse: IVFPQ stores 64-byte compressed entries; exact vectors exist only as TQ codes on disk, dequantized for the top-100 rerank one candidate at a time (~55 MiB fp32 transient per vector, or batched to the VRAM budget — the spec §10 ledger) |

## 3. The build: phases, gates, files

### 3.1 src/rag/ file map (the SPECIFICATION §1–§8 implementation; plus `src/rag/codebooks/` artifacts + `src/rag/tests/` = 11 test files, 161 tests)

| module | role |
|---|---|
| `src/rag/turboquant.py` | TQ core: units, FHT rotation binding, quant/dequant, 3.5-bit split, norm handling, code serialization |
| `src/rag/codebooks.py` | Lloyd-Max solver (continuous 1-D k-means on the Beta density), on-disk cache per bit-width, validation |
| `src/rag/tq_cache.py` | online cache wrapper: DynamicCache subclass (TQCache) intercepting read/write paths (N14–N16), S/conv/M1/M2 code stores |
| `src/rag/m1m2.py` | the architectural addition (N6–N7): buffers, gated additive writes, softmax(q @ M1ᵀ) @ M2 reads |
| `src/rag/hooks.py` | the 9-hook capture harness (N3) |
| `src/rag/ingest.py` | delta-protocol ingestion (N18–N19) + IngestDriver |
| `src/rag/snapshot.py` | per-chunk TQ-code npz codec (N19) |
| `src/rag/install.py` | sum_turboquant_codes / install_snapshot / install_from_disk (N25) |
| `src/rag/query.py` | query flow + install + answer (N22; the spec §6 steps 1–8) |
| `src/rag/finetune.py` | the lean N26–N29 loop (Phase 4) |
| `src/rag/lut_export.py` | fine-tuned LUT artifact codec (N29; pretrained_luts/, spec §11) |
| `src/rag/index.py` | IVFADC build + preselect + rerank (N30–N32) |
| `src/rag/evals.py` | phase-gate harness: round-trip MSE, streaming neutrality, retrieval recall, end-to-end QA, timing/VRAM ledger |

### 3.2 Phases P0–P7

| phase | work | gate | entry point | status |
|---|---|---|---|---|
| P0 bring-up | build flute_extended (src/docs/BUILD.md: ptxas gate, then the debug_simple differential spot-check with the verified pack_idxn API); load the model via the N1 loader; greedy-generate a fixed prompt set; record baseline outputs + step times | kernel numerics match the reference path on the spot-check; baseline generations archived | src/docs/BUILD.md + src/scripts/loader.py | GPU box pending |
| P1 TurboQuant core | codebooks: solve Lloyd-Max on the Beta density at d=2¹⁵/2¹⁹ (the density is effectively N(0,1/d); integrate analytically between Voronoi midpoints, iterate to fixed point); turboquant: unit wrappers (norm → fht_apply with the kind's signs → partition → bucketize to centroids → pack; dequant the reverse via fht_adjoint); round-trip gates on REAL tensors: (i) per-coordinate distribution of rotated states vs the Beta density (concentration — the D3 substitution risk), (ii) round-trip MSE per unit vs the paper's b=3/4 numbers scaled by measured norms, (iii) per-head norm spread (decides D2 granularity), (iv) outlier mass (decides the split A/B) | b=1 centroids reproduce ±√(2/π)/√d, b=2 reproduces ±0.453/√d + ±1.51/√d; D_mse ≈ 0.117·‖x‖² at b=2 on random unit vectors within 5%; all 4 measurements recorded with pass/fail lines; nothing downstream pinned until this table exists | src/rag/evals.py roundtrip | GPU box pending |
| P2 online cache wrapper | tq_cache per N14–N16: intercept recurrent_states reads (dequantize on read) + state-update writes (quantize on write) for the 24 linear layers, same for conv_states, plus the M1/M2 slots; the model's forward is untouched; stream-generate the P0 prompt set at 3.5 bits; measure greedy token-match vs the fp16-cache baseline + degradation vs prompt length (the compounding signature, §1.5 item 1) | ≥95% token match on ≤4k prompts; if it fails: outlier-partition A/B → per-head norms → quantize-on-snapshot fallback (D4) — each step re-runs this gate | src/rag/evals.py streaming | GPU box pending |
| P3 M1/M2 addition + 9-hook capture | m1m2 (N6–N7): (1,32,128,128) buffers shared across all 24 linear layers, gated additive writes, softmax(q @ M1ᵀ) @ M2 reads, wired into Qwen3_5GatedDeltaNet behind a config flag (default on for RAG, off for parity tests); zero-init the write gates (M1/M2 start as no-ops — untrained behavior bit-unchanged; the fine-tune opens them); hooks: the 9 capture points of N3 | with zero gates, generations bit-identical to P0 baselines; capture → quantize → dequant round-trip on the hooks' S tensors passes P1's MSE table | src/rag/tests/test_m1m2.py + src/rag/tests/test_hooks.py (parity; no evals subcommand) | GPU box pending |
| P4 fine-tune | finetune (N26–N29): next-token prediction on the OfficeQA corpus with the online-TQ cache active + M1/M2 attached; trainables = the 24 per-layer linear-attn param groups (S discriminative), the M1/M2 read/write gates (carry info), the W10 LUTs via PalettizedLinear.make_trainable() on the reference path (forward="reference" — the straight-through primitive this repo kept); ~500 steps, AdamW + cosine, batch to fit A10G (fp32 LUT masters, everything else fp16/bf16) | (i) loss curve sane, no divergence; (ii) the cache-signal check — cosine between cache vectors of same-topic vs different-topic chunks separates by ≥3× vs pre-fine-tune; (iii) greedy quality on the P0 prompts does not regress >1% token-match | src/rag/evals.py margin | GPU box pending |
| P5 ingestion + index | ingest + index: the D4 delta protocol over the corpus (system-prompt reset point, per-chunk delta codes + the N21 retrieval vector), IndexIVFPQ training/add, side-metadata = rotation seeds + partitions + codebook hashes; the D1 prod-variant A/B runs here — record recall both ways, pick per the decision rule; throughput: 50k chunks at ~6 MiB/chunk (spec §5) on one A10G (ingestion is embarrassingly parallel over chunks) | retrieval@100 / rerank@3 recall on held-out same-topic queries ≥ target (95% mechanics-level: the query chunk's own neighbors retrieved; the toy's 100% is the CPU ceiling) | src/rag/evals.py recall | GPU box pending |
| P6 install + end-to-end | query: the full spec §6 flow (tokenize → online-TQ prefill → preselect → rerank → load 3 chunks → dequant-sum-requant install → answer → decode) | (i) OfficeQA e2e accuracy vs a no-RAG baseline AND vs an oracle-retrieval variant (install the CORRECT chunk's codes — the gap oracle-vs-actual isolates retrieval quality from generation quality); (ii) the spec §9 timing ledger (prefill+snapshot ~8 ms, preselect ~10 ms, rerank ~20 ms, load ~3 ms, install ~15 ms, decode ~8 s; total ~8.06 s); (iii) the spec §10 VRAM ledger (~13 GiB on A10G 24 GiB) | src/rag/evals.py e2e + src/rag/evals.py ledger | GPU box pending |
| P7 hardening (only if gates demand) | CUDA-graph the decode-step write path (quantize-on-write is in the hot loop: FHT 2¹⁹ + bucketize + pack per S unit per step — µs-scale on GPU; graph-capture once the numerics are frozen); mmap the snapshot store; optional entropy coding of indices (~5% at b=4 — skip unless 300 GiB → 285 GiB matters); the prod-variant structured QJL if P5 chose it | every hardening flag default OFF with bit-identical flag-off parity (CPU-side verified); the GPU-side benefit is measured once numerics are frozen | flags: src/rag/turboquant.py --qjl, src/rag/snapshot.py use_mmap, src/rag/tq_cache.py graph_safe | GPU box pending |

CPU-side code for P1–P7 is done and green (161 tests; the wave record in TASKS.md); "GPU box pending" marks the real-tensor gates. Build order is evidence-gated — nothing downstream of a failed gate gets pinned; Phase 1's measurement table is the plan's most important artifact: every quantization hyperparameter (norm granularity, split ratio, variant choice) is decided there, on the real model, before the 50k-chunk ingestion runs.

## 4. Risk register

| # | risk | exposure | mitigation | decision point |
|---|---|---|---|---|
| R1 | recurrent RMW compounding (§1.5 item 1) | generation quality degrades with length | Phase 2 gate before anything else; outlier split → per-head norms → snapshot-only fallback (D4); the fine-tune (P4) is the structural fix | P2 |
| R2 | FHT rotation ≠ QR-Gaussian concentration | Beta assumption weaker on real S | Phase 1 empirical concentration table; the Hadamard sign-flip is the standard substitute; measured, not assumed | P1 |
| R3 | MSE-variant IP bias hurts retrieval | recall below target | D1's prod-variant A/B at Phase 5 (+1 bit/coordinate, structured QJL) | P5 |
| R4 | delta-rule S states are low-rank / spiky | per-unit norm scalar too coarse; distortion concentrated | Phase 1 per-head norm spread measurement → D2 fallback (32 sub-units per layer) | P1 |
| R5 | M1/M2 cold start | retrieval vector dominated by untrained noise pre-P4 | P4 ordering (fine-tune before index build); M1/M2 dims excluded from the index until gates open (index rebuild is cheap: TQ codes are already on disk) | P4 |
| R6 | IVFPQ at 13.6M dims (m=64, 8-bit) | coarse quantization error at extreme d | nprobe=8 + exact rerank on top-100 (N31); recall gate at P5; raise nprobe/m if short | P5 |
| R7 | 300 GiB snapshot store | disk budget, IO latency | 6 MiB × 3 per query is ~3 ms on NVMe (spec §9); if cold storage, page-cache the top clusters | P6 |
| R8 | fine-tune destabilizes the W4 LUTs | LUT masters drift off the fp16 grid | straight-through + fp16 snap on export (freeze_all_luts); reference-path training; 500-step budget | P4 |

## 5. Acceptance criteria

| spec ref | criterion | measured at |
|---|---|---|
| N10–N13 + spec §3.3 table | 3.5-bit TQ codes, ~6 MiB/chunk, 4.6× vs fp16 | P1/P5 |
| N14–N16 | online TQ: quantize-on-write / dequantize-on-read, model unaware | P2 |
| N13, N15 | streaming quality: ≥95% greedy token-match vs fp16 cache (≤4k) | P2 |
| N6–N7 | M1/M2 added, no-op at init, trained at P4 | P3/P4 |
| N34 (real-model analogue) | cache-vector same/different-topic cosine margin ≥3× | P4 |
| N18–N21, N30–N32, N34 | retrieval: recall@100 ≥95%, top-3 correct ≥ the toy's 90% trend | P5 |
| N20, N22–N24 | install = dequant-sum-requant; conv = last chunk | P6 |
| spec §9 table | per-query ledger ≤ ~8.1 s total (the spec table's total: ~8.06 s) | P6 |
| spec §10 table | VRAM ≤ ~13 GiB on A10G 24 GiB | P6 |
| N26–N29 | ~500-step fine-tune: linear-attn params + M1/M2 gates + W10 LUTs | P4 |
