# HANDOVER.md — the on-box agent handover (W29 · 2026-10-12)

**Who you are**: the A10G GPU-box agent. Your box owns the runtime —
the venv, the palettized model artifacts, the ingested corpus, and
every measurement. The dev box (CPU, no torch) owns the repo
staticly: reading code, editing, committing, pushing. **This file is
the ONE document you need.** Everything it claims about the failure
is classified with evidence in `FINDINGS.md` (repo root) — read that
only if you want the why; this file is the what-to-run.

---

## 0. The situation (60 seconds — updated after the W28 matrix verdict + the W29 repair)

The W28 matrix RAN on this box and the verdict is CLASSIFIED:

- **G2 (TQ + manual loop) = GARBAGE** with EXACT kernel parity
  (max|diff| 0.000e+00 on all three probe distributions) and
  worst-block rel-MSE **7.65 at a 38x head-norm spread** — **F2b
  CONFIRMED** (FINDINGS W28 item 9 / W29 item 13): the flat S unit's
  single fp norm set the per-coordinate noise floor at the GLOBAL
  energy and crushed the small heads at every decode step.
- **G3 (raw + generate) = COHERENT text mislabeled GARBAGE-RANGE** —
  the conf 0.000 was a measurement artifact (generate() returned no
  per-step scores). FIXED in W29 (output_scores=True — the conf is
  real data now). The generate() interplay is ELIMINATED as a cause.
- The W26-era ladder state (E1 soup, E2/E3 crash) is unchanged
  history — W27's fixes (device alignment, forgery refusals, the
  regime diagnosis, the verdict band) are all still in.

**The repair is IN (W29, tag w29-per-head)**: the S state now
quantizes as 32 PER-HEAD units of d=16,384, each with its OWN fp norm
(the paper's per-vector recipe), in the PERMUTED rotation frame (the
2^14 Hadamard's Kronecker alignment with the 128x128 matrix layout
required it — FINDINGS W29 item 15; sim: worst-head 10.69 -> 0.023 at
55x, 297 -> 0.023 at 525x). Python-only — **no kernel rebuild needed**
(the per-head path rides the torch twin; the kernel pair is untouched).

**Your job**: pull, run Gate Zero (pytest, incl. the NEW W29 file),
re-run Gate 0.5 (the matrix — the repair is the DEFAULT now), then
the re-ingest + the W27 ladder (§2). No code changes on your side.

### 0.1 The gates-artifact situation (why the ladder is gateless)

The box's `m1m2_gates_trained.npz` is a pre-W21 per-layer `(24,)`
file. The hermes session "repaired" it by repeating each scalar
across all 32 heads — correct shapes, **zero per-head addressing
signal** (content addressing IS per-head). That file loaded cleanly
under the W26 checks and contaminated E1. `load_gates` now refuses
both the missing-regime-meta case and the broadcast signature, loudly,
with the culprit named. On this corpus the gates are ALSO useless by
construction (the memories are empty — nothing to read); the gateless
ladder measures the real thing.

---

## 1. Bootstrap (once)

The repo is **private**:
`https://github.com/pkhairkh/rag-cache-engineering-qwen35-9b-flute.git`

The token travels out of band (the owner supplies the `ghp_…` value
alongside this handover — never in-tree):

```bash
export GHA_TOKEN=ghp_________________   # <- paste the owner-provided value
```

**The box's `/home/ubuntu/RAGGA` is a plain COPY, not a git clone**
(the W26 bootstrap assumed `git pull`; the hermes session found no
`.git`). Adopt the remote without touching your assets (corpus, model
and gates live OUTSIDE the repo tree):

```bash
cd /home/ubuntu/RAGGA
git init 2>/dev/null || true
git remote add origin \
  https://pkhairkh:${GHA_TOKEN}@github.com/pkhairkh/rag-cache-engineering-qwen35-9b-flute.git \
  2>/dev/null || git remote set-url origin \
  https://pkhairkh:${GHA_TOKEN}@github.com/pkhairkh/rag-cache-engineering-qwen35-9b-flute.git
git fetch origin
git reset --hard origin/main        # the code becomes exactly the repo
git describe --tags                 # MUST print w29-per-head or later
source /home/ubuntu/venv/bin/activate
# W28 — REBUILD THE EXTENSION (the repo is a copy, not a build): a
# stale .so under new wrappers produces silently wrong numbers.
# (W29 is PYTHON-ONLY — the kernel sources did NOT change this wave;
# a box that already rebuilt at W28 may skip this; the rebuild is
# always safe to re-run.)
cd src/flute_extended && pip install -e . --no-build-isolation && cd ../..
python3 -c "import sys; sys.path.insert(0,'src/rag'); sys.path.insert(0,'src/flute_extended'); import turboquant; print('TQ kernels:', 'present' if turboquant._load_tq_kernel() is not None else 'ABSENT (torch twin)')"
```

(If `scripts/gpu/run_query.py` carries the old sed hotfix, the reset
above already replaced it — the repo's 2-D ids fix is the proper one
since W23.3.)

Your assets (adjust if they moved): corpus
`/home/ubuntu/ragga_ingested_test100`, model
`/home/ubuntu/qwen3_5_9B_palettized` (+`_heads`) — these are the
driver defaults already.

**Gate Zero — convert the authored tests into measurements**
(~3 min, once, before the ladder):

```bash
python -m pytest src/rag/tests/test_w27_regime_guards.py -x
python -m pytest "src/rag/tests/test_query.py::test_query_cache_vector_device_alignment" \
                 "src/rag/tests/test_query.py::test_corpus_memory_state_real_loader" \
                 "src/rag/tests/test_query.py::test_install_gdn_best_mode" \
                 "src/rag/tests/test_tq_cache.py::test_w28_protocol_guards_refuse_loudly" -x
# W29 — the per-head repair gates (AUTHORED on the dev box, first run
# here): the vectorization pin, the dynamic-range immunity (incl. the
# production geometry), the codes contract, the wiring, the sums, the
# disk edge. ~2-4 min on CPU-torch; no GPU needed.
python -m pytest src/rag/tests/test_w29_per_head.py -x
```

Expect PASS on all. If anything FAILS, stop and paste the assert —
do not patch. The first file closes the forgery path (regime meta +
broadcast signature + the corpus-regime trichotomy); the second
reproduces and closes the E2/E3 device crash; the third is the W26
install gate; the fourth pins the W28 protocol guards; the fifth pins
the W29 repair (per-head units, the permuted frame, the units codes
contract through every edge).

**Gate 0.5 — THE TQ MALFUNCTION LOCALIZER (W28, run BEFORE the ladder)**

The user's direct measurement: the model + `TQCache(online=True)` +
`model.generate()` = soup; the model without TurboQuant = fine. This
ONE command localizes it (the 2x2 generation matrix raw/TQ x
manual/generate + the kernel parity probe at production geometry on
Gaussian / heavy-tail / per-head-spread states + the build report):

```bash
python3 scripts/gpu/tq_gen_matrix.py --max-new-tokens 48
# the honest A/B — reproduce the W28 defect on demand:
python3 scripts/gpu/tq_gen_matrix.py --max-new-tokens 48 --s-flat
# if the verdict says "the generate() interplay":
python3 scripts/gpu/tq_gen_matrix.py --trace --max-new-tokens 8
```

Paste the WHOLE output. **W29: the per-head repair is the DEFAULT** —
the TQ cells now quantize S as 32 per-head units (own norms, the
permuted frame) and the G3/G4 conf is REAL data (output_scores=True;
the W28 run's conf 0.000 was the missing-data artifact that mislabeled
G3's coherent text GARBAGE-RANGE — fixed). The probe's per-head-spread
row now also prints the **[W29 per-head units]** measurement (the same
state through the repair — expected at the ~0.02 floor vs the flat
row's 7.65). Expected: **G1/G2/G3/G4 all CLEAN**; the --s-flat control
reproduces the W28 soup (G2/G4 garbage) — the controlled proof the
per-head regime is what closed it. If anything is NOT clean, the
printed decision tree names the branch.

---

## 2. THE CORRECTED LADDER — gateless, E1 → E2 → E3 (run in order)

All three run WITHOUT `--m1m2-gates` **on purpose**: the corpus's
memories are empty (the `[W27] corpus M1/M2 memories EMPTY` line in
[1] confirms it on your box), zero-init P3 gates are a bit-identical
no-op, and the only loaded artifact on the box is a refused forgery.
The ladder therefore isolates exactly F1 (the n≥2 S-state sum) vs F2
(the quantized write-path floor). The `[W26] GDN S install` line
confirms the install regime per run.

**W29 regime note**: the codes are self-describing — the OLD corpus
(flat codes) installs FLAT (the codes teach the layer the flat frame;
the session runs the corpus's regime, consistently), a re-ingested
corpus (§2.5) installs per-head. **Run the ladder FIRST on the old
corpus** (the apples-to-apples W27 baseline), then re-ingest and
re-run E1/E3 — the second run's write path rides the per-head repair
end-to-end. A mixed-regime install is REFUSED loudly by design (the
install guards' re-ingest hint).

```bash
cd /home/ubuntu/RAGGA && source /home/ubuntu/venv/bin/activate
Q="--question How do I deploy perf-canary?"
COMMON="--disk-dir /home/ubuntu/ragga_ingested_test100 $Q \
  --max-new-tokens 128"

# E1 — n=1 verbatim install, GATELESS (the read is a no-op on an
#      empty memory; expect E1 == E3 — a difference would be NEW
#      information about the zero-state read: paste both)
python3 scripts/gpu/run_query.py $COMMON --top-k 1

# E2 — n=3 install (the F1 sum) + the M1/M2 answer read OFF
python3 scripts/gpu/run_query.py $COMMON --top-k 3 --no-answer-m1m2

# E3 — the cleanest regime: n=1 verbatim + no memory read
python3 scripts/gpu/run_query.py $COMMON --top-k 1 --no-answer-m1m2
```

**How to read each run** — the `[gen gate]` line now prints the
verdict band:

| verdict | signature |
|---|---|
| **clean** | conf ≥ 0.60 (CLEAN-RANGE), non-ascii ≤ ~0.05, English prose about the retrieved doc (the true-doc control measured **0.895**) |
| **weak** | conf 0.20–0.60 (WEAK — the E1 forgery run "passed" here at 0.382 while being soup): judge the ANSWER TEXT |
| **garbage** | mixed-script soup from token 1, non-ascii ~0.2–0.4, conf ~0.05–0.15 |

**Decision tree** (E1 ≈ E3 expected on this corpus):

| E1 | E2 | E3 | verdict → next step |
|---|---|---|---|
| clean | garbage | clean | **F1 CONFIRMED** (the n≥2 S-sum is the killer) → run §4 now |
| clean | clean | clean | the failure needs the combination → run §4 anyway, then §5 |
| garbage | garbage | clean | **F2 CONFIRMED** (the quantized write-path floor) → §5 E5 bisect, paste the matrix |
| garbage | garbage | garbage | **F2 deep** (even the n=1 verbatim install garbles) → §5 E5, paste the matrix |
| E1 ≠ E3 | any | any | NEW information (the zero-state read does something) → paste both runs verbatim |

---

## 2.5 THE RE-INGEST (W29 — only after the ladder's baseline runs + Gate 0.5 clean)

The per-head repair is the LIVE-CACHE default, but the corpus on disk
still carries the W28 flat codes (the snapshots are self-describing —
they decode fine, they install flat). To activate the repair
END-TO-END (the ingest-side quantization + the install + the decode
write path all per-head), re-ingest the corpus into a NEW directory
(never overwrite the old one — it is the A/B control and the §3
baseline):

```bash
# re-ingest at the per-head regime (the default — the ingest driver's
# TQCache carries it; the chunks + the system snapshot land as
# partition="units" codes)
python3 scripts/gpu/run_ingestion.py \
    --out-dir /home/ubuntu/ragga_ingested_test100_w29 \
    --n-docs 100 --max-tokens 0          # match the ORIGINAL ingest's
                                         # manifest line (n-docs / corpus
                                         # / system prompt) — paste it
# then re-build the index over the NEW corpus (§3's step 1) and re-run
# E1/E3 with --disk-dir /home/ubuntu/ragga_ingested_test100_w29
```

The new snapshots carry `units` keys (partition="units"); the old
readers/old digests are unaffected; a mixed install is REFUSED loudly
with the re-ingest hint. Paste the new manifest line + the E1/E3
verdicts.

---

## 3. Retrieval ground truth (before or in parallel with the ladder)

The retrieval complaint ("rerank fails badly on old snapshots")
decomposes into three measured, non-architectural causes — none of
them needs code: (a) the absolute frame's common-component floor
(W16 measured ~0.70-floor / ~0.016 spread on this box — the centered
frame exists to remove it and **was never built for test100**: every
log prints `centered retrieval frame OFF`); (b) the inert memory
subspace diluting the §4 metric (8,388,608 of 8,486,912 dims); (c)
the query-side regime mismatch (now auto-matched, W27.3). Two
commands:

```bash
# 1 — build the centered frame + the index over the SAME corpus
python3 scripts/gpu/run_index.py --disk-dir /home/ubuntu/ragga_ingested_test100

# 2 — measure retrieval QUALITY against the benchmark's gold docs
#     (hit@1/3/5 in the absolute / sys / sys+mean frames, per question)
python3 scripts/gpu/eval_retrieval.py --disk-dir /home/ubuntu/ragga_ingested_test100
```

Read the eval_retrieval header ("THE READING") for the decision it
encodes: absolute-frame hits at/below chance + a big centered-frame
gap = the frame was the problem (fixed by run_index); centered-frame
misses too = the S signal itself is weak for these weights — the §7
fine-tune is the designed lever and these numbers are its baseline.
Then re-run E1 with the frame picked up automatically (run_query
auto-detects `retrieval_frame.npz`) and paste the new Top-k line.

---

## 4. Fix verification (only after E1/E3 came back clean)

The F1 remedy ships as `--install-gdn best`: **S + conv installed
VERBATIM from the highest-scoring retrieved chunk, M1/M2 still
summed** (their write is additive — the sum is algebra-exact for
them). Everything else in the flow is untouched:

```bash
# E6b — the fix + the honest read mitigation (both causes killed)
python3 scripts/gpu/run_query.py $COMMON --top-k 3 --install-gdn best \
    --no-answer-m1m2
```

Expected: **E6b clean** if E1/E3 were clean. The log line
`[W26] GDN S install (n>=2): best` and the install report's
`s_source_chunk` (== the top-1 retrieved id) confirm the mode ran.
(E6 — `--install-gdn best` WITHOUT `--no-answer-m1m2` — is optional
here: on this corpus the read is a no-op, so E6 ≈ E6b.)

If E6b is clean: the pipeline's answer path is closed at n=3 —
report and STOP. The dev box will then flip the default to `best` in
a one-line follow-up commit (not you).

---

## 5. Follow-ups (only AFTER the ladder — do not jump ahead)

- **E5 (if E1/E3 were garbage — F2 floor)**:
  `python3 scripts/gpu/bisect_install.py --chunk 0 $Q --max-new-tokens 32`
  and paste the whole matrix (the true-doc row re-validates the
  control under the current loader regime).
- **The gates retrain (the real artifact)**: the box's pairs data +
  the repo's finetune driver produce a W21+ content-addressed
  artifact (geometry (24, 32), `meta_write_addr` baked in):
  `python3 scripts/gpu/finetune_m1m2.py --pairs-file data/pairs.jsonl
  --gates-out m1m2_gates_w27.npz --m1m2-mem-size 1024 --max-steps 300
  --refresh-every 4 --gate-init 0.1 --neg-pairs 8`. Then RE-INGEST the
  corpus with `--m1m2-gates m1m2_gates_w27.npz` to activate the
  memory subspace end-to-end (ingest and query must agree — the
  loaders fail loudly on drift).
- **read_gate training (F3's real fix)**: HONEST state — the
  finetune driver has **no `--aux-nll-weight` flag** (FINDINGS Errata
  #1; `--train-read-gate` alone trains nothing). Implementing the
  aux-NLL path is a dev-box task; do not improvise training on the
  box. Until then `--no-answer-m1m2` is the honest answer-quality
  setting — and on THIS corpus the empty memory makes the question
  moot (the read returns zeros).

---

## 6. What to paste back (per run, verbatim)

1. the exact command line
2. the `[W27] corpus M1/M2 memories` line (once — it confirms the
   regime diagnosis on the box)
3. the `[W26] GDN S install` regime line
4. the `Top k:` line (ids + scores)
5. the full `Answer:` line (first 300 chars as printed)
6. the `[gen gate] mean conf … repetition … non-ascii … verdict` line
7. wall-clock time, and the full traceback if anything crashed
   (**never** patch around a crash on the box — paste it)

---

## 7. Rules of engagement (binding)

1. **GPU-ONLY law**: every number you report comes from a command you
   ran on this box. No fabricated results, no "expected" outputs.
2. **No architecture changes**: M1/M2 + GatedDeltaNet + the
   all-concatenated-caches retrieval are settled. The only new
   surfaces are flags (W26 `--install-gdn`, W27's guards are
   validation-only, default bit-identical).
3. **No reprefill**: re-prefilling chunk text at query time was
   removed by order (W24). Do not re-add it.
4. **No forged artifacts**: never reshape/broadcast a gates file to
   make it load — `load_gates` now refuses exactly that; retrain
   instead (§5).
5. Edit nothing outside the flags documented here; if a run needs a
   code change, paste the evidence and let the dev box do it.
6. Run the ladder in order; each next run depends on the previous
   verdict.

---

## 8. Map of the W27/W29 change (where to look in the code)

| file | what |
|---|---|
| `src/rag/query.py` | `query_cache_vector` — the device alignment (W27.1: every §4 piece hops to the S pieces' device; uniform flows are bit-identical no-ops) |
| `src/rag/m1m2_finetune.py` | `save_gates` bakes the write-kernel regime meta; `load_gates` refuses (a) artifacts without `meta_write_addr` on a content-addressing module, (b) write_gate_k rows constant across heads (the broadcast signature) (W27.2) |
| `src/rag/index.py` | `corpus_memory_state(loader)` — the empty / carries-content / absent trichotomy (W27.3); the batched decode stackers are UNITS-SAFE (the kernel eligibility gate refuses non-flat codes before the stack — W29) |
| `scripts/gpu/run_query.py` | the `[W27]` corpus-regime diagnosis + the retrieval-prefill regime match (M1/M2 OFF for the prefill when the corpus memories are empty and gates are loaded) + the gen gate's non-ascii ratio and verdict band (W27.3/4) |
| `src/rag/tests/test_w27_regime_guards.py` | gates A–C (AUTHORED, PENDING this box): the forgery refusals, the honest round-trip with baked regime meta, the corpus-regime trichotomy |
| `src/rag/tests/test_query.py` | `test_query_cache_vector_device_alignment` (the E2/E3 crash reproduced and closed) + `test_corpus_memory_state_real_loader` |
| `src/rag/turboquant.py` | **W29**: `UNITS_PERM_OFFSET` (the permuted frame + the Kronecker note), `quant_units` / `_dequant_units` (the vectorized multi-unit twin + the kernel batch at eligible d), the units routing in `dequant`, the loud refusals (split / qjl / misalignment) |
| `src/rag/tq_codes.py` | **W29**: the `partition="units"` mode — the `units` field, the (M,) per-unit norms vector, the type cops, to_arrays/from_arrays + the units key (old readers/digests unaffected) |
| `src/rag/tq_cache.py` | **W29**: the per-head S wiring (DEFAULT ON; `s_flat=True` the W28 A/B; the qjl x per-head mutual exclusion refused at construction), `_init_s`'s per-unit resolution, the s_codes setter's regime/units guards; the W28 protocol guards (`crop` / `batch_repeat_interleave`) unchanged |
| `src/rag/install.py` | **W29**: `_requant_flat` (the sums' requant routed on the codes' partition — units sums keep per-unit norms) + `_units_guard` (the mixed-regime REFUSAL, re-ingest hint) in both sums |
| `src/rag/snapshot.py` | **W29**: `_read_unit`'s units validation (units present-iff-units, the (M,) norm, the M×L stream lengths) + the digest's units bytes |
| `src/rag/ingest.py` | **W29**: `reference()`'s norms-vector fold (the system-state hash) |
| `FINDINGS.md` | Errata W27 (items 4–7) + Errata W28 (items 8–12) + **Errata W29 (items 13–16: the box verdict classified, the per-head repair, the Kronecker discovery, the surface map)** |
| `scripts/gpu/tq_gen_matrix.py` | W28 the localizer; **W29**: the `[W29 per-head units]` probe row, the honest generate() conf (output_scores), the `--s-flat` A/B flag, the updated verdict tree |
| `scripts/w28_tq_sim.py` | the numpy reproduction — **W29**: the per-head regime rows (the permuted frame; worst-head 10.69→0.023 at 55x, 297→0.023 at 525x) |
| `src/rag/tests/test_w29_per_head.py` | **W29** (AUTHORED, PENDING this box): the vectorization pin, the dynamic-range immunity (d=1,024 + the production geometry), the codes contract + serialization, the seed-addressed frame, the layer wiring, zero-norm units, the install sums, the disk edge, the refusals |
| `src/rag/tests/test_tq_cache.py` + the W16–W19 suites | **W29**: pinned to `s_flat=True` (their authored flat semantics preserved — the per-head coverage is the W29 file's) |
| `m1m2_gates_trained_fixed.npz`, `m1m2_gates_trained_w26.npz` | **DELETED** (the hermes forgeries — refused by W27.2; the full box log that motivated them is preserved at commit `a9bbda2`) |

Default behaviors: W27's device alignment is bit-identical on
every uniform-device flow (the existing suite pins it); W29's units
regime is the NEW DEFAULT for the 4-D S path (the repair) with
`s_flat=True` restoring the pre-W29 bytes exactly — the pinned suites
prove the flat contract, test_w29_per_head.py proves the repair.

---

## Appendix — pinned history (keep these headings; auditors grep them)

## W23 — historical note (superseded)

W23 introduced the "prefill answer mode" (re-prefill retrieved chunk
TEXT at query time) and made it the default; it produced coherent
answers precisely because it rebuilt the S state by real sequential
recurrence, but it bypassed the installed codes — the venture's core
mechanism. **Removed by order in W24**; the install (steps 5–6) is the
only answer-context build since. Full history: `git log`, FINDINGS.md.

## ❌ THE PROBLEM: M1/M2 Fails to Learn Discrimination

Historical training-state note (W21-era, superseded by the W21
content-addressing fix and the W22 hard-negative regime): the first
finetune showed cos(pos) ≡ cos(neg) and spread 0 for 300 steps. The
current known-good training state is in the uploaded box logs; the
**read-side** gap (read_gate never trained) is FINDINGS.md F3 and is
open.
