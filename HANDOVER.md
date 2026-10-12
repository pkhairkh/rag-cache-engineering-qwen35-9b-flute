# HANDOVER.md — the on-box agent handover (W27 · 2026-10-12)

**Who you are**: the A10G GPU-box agent. Your box owns the runtime —
the venv, the palettized model artifacts, the ingested corpus, and
every measurement. The dev box (CPU, no torch) owns the repo
staticly: reading code, editing, committing, pushing. **This file is
the ONE document you need.** Everything it claims about the failure
is classified with evidence in `FINDINGS.md` (repo root) — read that
only if you want the why; this file is the what-to-run.

---

## 0. The situation (60 seconds — updated after the box's E-ladder attempt)

The W26 ladder RAN on this box (the hermes session) and did not
measure what it was designed to measure:

- **E1** (top-k 1, gates): multilingual soup again — but it ran with
  a **hand-broadcast forgery** of a gates artifact (see §0.1), so its
  garbage is classified as contamination, not as evidence.
- **E2 and E3** (the `--no-answer-m1m2` runs): **CRASHED** at
  `query_cache_vector` — `torch.cat` on mixed devices (host M1/M2
  codes vs model-device S codes) — **before the install ever ran**.
  The F1-vs-F3 isolation the ladder exists for NEVER HAPPENED.

The fixes are in the repo, behind tag `w27-ladder-repair`:

- **W27.1** — the device crash is fixed at the crash site (uniform
  flows bit-identical).
- **W27.2** — `load_gates` now REFUSES the box's two "fixed" gates
  files (no `meta_write_addr`; broadcast write_gate_k rows). They are
  deleted from the repo. **Do not rebuild them** — the honest artifact
  comes from a W21+ retrain (§5).
- **W27.3** — `run_query.py` now introspects the corpus's M1/M2
  ingest regime and prints it (`[W27] corpus M1/M2 memories EMPTY
  ...`): the test100 corpus was ingested with ZERO write gates, its
  memory codes are exact zeros, the memory subspace is INERT — F3 is
  untestable on this corpus, and retrieval rides the S subspace alone.
- **W27.4** — the gen gate now prints a non-ascii ratio and a verdict
  band: E1's soup "passed" the 0.20 conf bar at 0.382 — a conf number
  alone cannot see mixed-script soup. Judge the TEXT.

**Your job**: pull, run Gate Zero (pytest), then the corrected
GATELESS ladder (§2), then retrieval ground truth (§3). No code
changes on your side.

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
git describe --tags                 # MUST print w27-ladder-repair or later
source /home/ubuntu/venv/bin/activate
```

(If `scripts/gpu/run_query.py` carries the old sed hotfix, the reset
above already replaced it — the repo's 2-D ids fix is the proper one
since W23.3.)

Your assets (adjust if they moved): corpus
`/home/ubuntu/ragga_ingested_test100`, model
`/home/ubuntu/qwen3_5_9B_palettized` (+`_heads`) — these are the
driver defaults already.

**Gate Zero — convert the authored tests into measurements**
(~2 min, once, before the ladder):

```bash
python -m pytest src/rag/tests/test_w27_regime_guards.py -x
python -m pytest "src/rag/tests/test_query.py::test_query_cache_vector_device_alignment" \
                 "src/rag/tests/test_query.py::test_corpus_memory_state_real_loader" \
                 "src/rag/tests/test_query.py::test_install_gdn_best_mode" -x
```

Expect PASS on all. If anything FAILS, stop and paste the assert —
do not patch. The first file closes the forgery path (regime meta +
broadcast signature + the corpus-regime trichotomy); the second
reproduces and closes the E2/E3 device crash; the third is the W26
install gate.

---

## 2. THE CORRECTED LADDER — gateless, E1 → E2 → E3 (run in order)

All three run WITHOUT `--m1m2-gates` **on purpose**: the corpus's
memories are empty (the `[W27] corpus M1/M2 memories EMPTY` line in
[1] confirms it on your box), zero-init P3 gates are a bit-identical
no-op, and the only loaded artifact on the box is a refused forgery.
The ladder therefore isolates exactly F1 (the n≥2 S-state sum) vs F2
(the quantized write-path floor). The `[W26] GDN S install` line
confirms the install regime per run.

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

## 8. Map of the W27 change (where to look in the code)

| file | what |
|---|---|
| `src/rag/query.py` | `query_cache_vector` — the device alignment (W27.1: every §4 piece hops to the S pieces' device; uniform flows are bit-identical no-ops) |
| `src/rag/m1m2_finetune.py` | `save_gates` bakes the write-kernel regime meta; `load_gates` refuses (a) artifacts without `meta_write_addr` on a content-addressing module, (b) write_gate_k rows constant across heads (the broadcast signature) (W27.2) |
| `src/rag/index.py` | `corpus_memory_state(loader)` — the empty / carries-content / absent trichotomy (W27.3) |
| `scripts/gpu/run_query.py` | the `[W27]` corpus-regime diagnosis + the retrieval-prefill regime match (M1/M2 OFF for the prefill when the corpus memories are empty and gates are loaded) + the gen gate's non-ascii ratio and verdict band (W27.3/4) |
| `src/rag/tests/test_w27_regime_guards.py` | gates A–C (AUTHORED, PENDING this box): the forgery refusals, the honest round-trip with baked regime meta, the corpus-regime trichotomy |
| `src/rag/tests/test_query.py` | `test_query_cache_vector_device_alignment` (the E2/E3 crash reproduced and closed) + `test_corpus_memory_state_real_loader` |
| `FINDINGS.md` | Errata W27 (items 4–7): the forged-artifact classification, the corrected gateless ladder, the retrieval decomposition |
| `m1m2_gates_trained_fixed.npz`, `m1m2_gates_trained_w26.npz` | **DELETED** (the hermes forgeries — refused by W27.2; the full box log that motivated them is preserved at commit `a9bbda2`) |

Default behaviors are bit-identical to pre-W27 on every uniform-device
flow — pinned by the existing suite plus the two new gates above.

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
