# RAGGA Handover Document

## Current State (post-W13 — the W12 resolution)

| item | status |
|---|---|
| CPU suite | **179 tests green** (`python3 -m pytest src/rag/tests -q`) — the 4 new W13 gates pin the S+conv install combination through the real GDN forward |
| W12 "S-read DRIFT" | **RETRACTED — proven a measurement artifact of the bisection script itself** (numbers below) |
| W12 "G6 PASS vs G5 FAIL" contradiction | **the G6 gate was judging on repetition only — low-confidence garbage false-passed; fixed (conf gate now applies)** |
| S / M1/M2 install math | exonerated end-to-end at real GDN math (CPU): install-vs-truth, read paths, and the four-variant bisection all inside the house budgets; **no s-only/full asymmetry exists on the reference path** |
| GPU-only routes (static audit) | CUDA FHT kernel (fwd/adj contracts, segment tiling) and fla-0.5.2 `fused_recurrent_gated_delta_rule` (fp32 state accumulate, fresh final_state, `[N,HV,K,V]` layout, dtype-agnostic h0) — both sound by inspection; the ONE unverifiable-from-CPU risk (kernel-quant vs reference-dequant frame split at the conv's 24,576) is now instrumented |
| **OPEN** | G5 generation quality on the GPU box — two live hypotheses, both now decisively testable with the rewritten `bisect_install.py` (the **true-doc raw control** + the **frame check**), see the procedure below |

---

## The W12 Finding, Adjudicated

### 1. The "S-read DRIFT in every variant" was the script's own bug

The W12 `bisect_install.py::s_read_check` compared the layer-0 S read
against `dequant(sys) + dequant(delta)` **unconditionally**:

* for **reseed / conv-only** the delta was never installed — the printed
  number was just the delta's energy fraction `‖delta‖²/‖sys+delta‖²`:

  | what the box measured | what it actually was |
  |---|---|
  | reseed / conv-only: **7.66e-01 "DRIFT"** | the uninstalled chunk delta's energy (~0.85 uncorrelated, 0.77 at the box's sys/delta correlation) |
  | s-only / full: **1.47e-02 "DRIFT"** | the **expected** single-requant error of `install(sys, delta)` (W11 CPU gate: 1.65e-02; inside the 0.06 house budget) judged against an impossible 1e-4 threshold that was written for the bit-clean reseed case |

  Reproduced exactly on CPU (scripts/repro + the new test gates): the same
  check prints 8.3–8.5e-01 on the reseed rows and ~1.9e-02 on the install
  rows. **The S path was and is clean.** The corrected check (per-variant
  reference, 1e-4 for reseed / 0.06 for installs) is in the rewritten
  bisect.

### 2. The four W12 hypotheses — refuted one by one

1. *Device mismatch (codes on CPU, dequant expects CUDA)* — REFUTED.
   `TurboQuant.dequant` is device-agnostic (numpy codebook → `from_numpy`
   → CPU reference butterfly, then the cache moves the result to the
   tracked device). The reseed read is bit-clean (CPU gate: 4.7e-08; the
   install read matches the direct dequant exactly).
2. *Frame rotation mismatch (D3 seed differs quant vs dequant)* — REFUTED.
   `_check_codes` pins (kind, d, bits, n_lo/hi, seed) on every dequant;
   the seeds are kind constants (S 101 / conv 202 / M1 303 / M2 404);
   `resolve_quantizer` is registry-cached and custom-d keeps the kind's
   seed; `sum_turboquant_codes` raises loudly on drift.
3. *`set_s_codes` doesn't propagate to layer computation* — REFUTED.
   There is no second "computation state": the layer's `_StateView`
   dequantizes `_s_codes` on every read the model makes; the W13 test
   drives the real `Qwen3_5GatedDeltaNet.forward` through it end-to-end.
4. *Why reseed showed DRIFT* — ANSWERED (the reference bug, above).

### 3. The G6 "PASS" was a false pass

`verify_pipeline.py`'s G6 judged `answer_query` e2e output on
**repetition only** — low-confidence mid-word text with no immediate
repeats passed. G5 was judged `conf ≥ 0.20`. The "G6 PASS vs G5 FAIL"
contradiction was partly this: **the same installed state ran through
both gates.** Fixed: `answer_query` now tracks per-step confidences
(`QueryResult.token_confs`) and G6 applies the same conf gate.

### 4. What ACTUALLY remains: G5's generation quality

With the DRIFT ghost dead, the facts are: reseed/s-only/conv-only
generate above the (weak) conf-0.20 gate; the full install — the
**faithful reconstruction of the true end-of-chunk state** (S within the
single-requant budget, conv verbatim) — does not. On CPU (reference
route, real GDN math) **no such asymmetry exists**: full's deviation from
its own ground truth equals s-only's (the W13 gate 3). Two live
hypotheses remain, and the rewritten bisect separates them cleanly:

* **H-SEM (semantic, most likely)**: the model, continuing from a state
  that just processed a 500-token document, faithfully *continues the
  document* rather than answering — mid-word, low-confidence tokens that
  read as "garbage" but are the correct continuation of exactly the
  state the install built. **Test: the new `true-doc` variant** —
  [system + chunk] prefilled on a RAW cache, no TQ anywhere, full-attn
  KV dropped (G5's geometry). If true-doc ALSO fails the gen gate, the
  install math is fully exonerated and the work moves to the answer-flow
  design (how the question dominates a document-loaded state).
* **H-FRAME (GPU-only numeric)**: at quant time the conv window (24,576
  = 16,384 + 8,192, kernel-eligible) rides the CUDA FHT kernel, while
  every dequant runs the torch reference. The kernel source audits clean
  (fwd `((x@H)*s)/√b`, adj `((g*s)@H)/√b`, per-segment disjoint tiling),
  but nothing on the box ever *measured* it at that d.
  **Test: the new `frame` check** — quant a random tensor ON DEVICE,
  dequant through the normal path; ~0.02 = consistent, ~1.0 = frame split.
  If it splits, every conv code on the box is written in an unreadable
  rotation (and conv-only's "pass" would need re-examination under the
  now-printed conf numbers).

---

## Required GPU-box procedure (after pulling W13)

```bash
cd /home/ubuntu/RAGGA && git pull

# 1. The decisive run (all variants + frame check + true-doc control):
python3 scripts/gpu/bisect_install.py

# 2. If gen rows look marginal, read the conf column (now printed);
#    A/B the Triton decode route:
python3 scripts/gpu/bisect_install.py --fla-off

# 3. The fixed gate ladder (G6 now judges conf too — expect G6 to move):
python3 scripts/gpu/verify_pipeline.py
```

Read the matrix per the script's decision table:
`S-read/conv DRIFT` ⇒ real read-path bug · `frame SPLIT` ⇒ the CUDA FHT
kernel disagrees with the reference (conv codes unreadable) ·
`true-doc GARBAGE` with reads/frame OK ⇒ **semantic** (the model
continues the document state — the design question, not corruption) ·
all OK but full GARBAGE ⇒ rerun `--fla-off` (the Triton decode route).

The `true-doc` variant rebuilds the chunk's tokens from the corpus
(`--corpus`, default the box's documents.jsonl) and **requires the same
`--system-prompt` run_ingestion.py used** (default "You are a helpful
AI assistant.") — a mismatched prompt only softens the control, it does
not invalidate the other rows.

---

## The W13 CPU net (what is now pinned on CPU)

`src/rag/tests/test_install_conv_math.py` — the S+conv install
**combination** through the real vendored `Qwen3_5GatedDeltaNet.forward`
(the gap the W11 file explicitly left: "No conv (S-path isolation)"),
4 gates:

1. install-vs-truth: S ≤ 0.15 (the ONLINE ingestion drift over a
   300-token chunk; the install math itself ~2%), conv ≤ 0.06;
2. query-prefill output tracking ≤ 0.10 (residual-stream faithful rig);
3. **no variant asymmetry**: full ≤ 2× the worst half-install (measured
   ~0.94 — the W12 cliff has no reference-path counterpart);
4. 12 teacher-forced decode steps keep the per-layer S drift ≤ 0.8.

Geometry: 4 GDN layers, d_S = 2,048, conv window 768 = 192×4 (multiple
of 32, non-pow2, FHT segments 512 + 256 — structurally the box's
16,384 + 8,192); committed codebooks only (cb_b{3,4}_d{768,2048}.npz).

---

## Model Details

- Qwen3.5-9B palettized — `/home/ubuntu/qwen3_5_9B_palettized` (+ `_heads`)
- vocab_size 248,320; EOS `<|im_end|>` (id 248,046)
- 32 layers ([L,L,L,F]×8 — 24 linear + 8 full-attention)
- the box's GDN geometry: k_heads 8 / v_heads 32, head dims 128 →
  conv window 24,576 = 6,144×4 (FHT segments 16,384 + 8,192), S = 524,288

## Test Command

```bash
cd /home/ubuntu/RAGGA && python3 -m pytest src/rag/tests -q --tb=short
```

179 tests pass on CPU (175 from W11 + the 4 W13 combination gates).
