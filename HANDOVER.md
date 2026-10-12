# HANDOVER.md — the on-box agent handover (W26 · 2026-10-12)

**Who you are**: the A10G GPU-box agent. Your box owns the runtime —
the venv, the palettized model artifacts, the ingested corpus, and
every measurement. The dev box (CPU, no torch) owns the repo
statically: reading code, editing, committing, pushing. **This file is
the ONE document you need.** Everything it claims about the failure is
classified with evidence in `FINDINGS.md` (repo root) — read that only
if you want the why; this file is the what-to-run.

---

## 0. The situation (60 seconds)

Retrieval is sane (top-3 scores 0.63/0.60/0.59 over 100 chunks) but
the answer is multilingual word-soup from the very first token. The
root cause is classified (FINDINGS.md, math-certain):

- **F1 (root cause)**: the n≥2 install **sums GatedDeltaNet recurrent
  states** — the delta-rule recurrence `S ← λ(I−βkkᵀ)S + βkvᵀ` has
  erase+decay, so chunk states are **non-additive**. The synthetic demo
  in the repo (`scripts/demo_install_algebra.py`) measures n=1
  rel-MSE **0.0000** (exact) vs n=3 rel-MSE **2.63** — ~20× over the
  0.15 corruption budget this project measured for itself. Your
  `--top-k 3` run sits exactly on the broken path.
- **F3 (second)**: the answer-phase M1/M2 read is untrained
  (`read_gate` frozen at 1.0) and the 0.895 true-doc control ran with
  zero-init gates, so that path was never validated either.

**The fix is already in the repo, behind a flag** — your job is to
(1) run the three-command ladder that turns the classification into
measurements, then (2) verify the fix. No code changes on your side.

---

## 1. Bootstrap (once)

The repo is **private**:

```
https://github.com/pkhairkh/rag-cache-engineering-qwen35-9b-flute.git
```

The access token is **NOT embedded in this file** — GitHub's secret
scanner rejects any push that carries a token, so it travels out of
band: the owner supplies the `ghp_…` value alongside this handover
(the chat message that delivered it). Store it once in the shell you
run the bootstrap from:

```bash
export GHA_TOKEN=ghp_________________   # <- paste the owner-provided value
```

**If the box already has a checkout** (e.g. `/home/ubuntu/RAGGA`):

```bash
cd /home/ubuntu/RAGGA
git checkout -- scripts/gpu/run_query.py   # drop the old sed hotfix —
                                           # the repo has carried the
                                           # proper 2-D ids fix since
                                           # W23.3; a dirty file blocks
                                           # the pull
git pull https://pkhairkh:${GHA_TOKEN}@github.com/pkhairkh/rag-cache-engineering-qwen35-9b-flute.git main
git describe --tags                        # MUST print: w26-gdn-best-install
source /home/ubuntu/venv/bin/activate
```

**Fresh clone**:

```bash
git clone https://pkhairkh:${GHA_TOKEN}@github.com/pkhairkh/rag-cache-engineering-qwen35-9b-flute.git /home/ubuntu/RAGGA
cd /home/ubuntu/RAGGA
source /home/ubuntu/venv/bin/activate
```

(If the checkout predates the token: `git remote set-url origin
https://pkhairkh:${GHA_TOKEN}@github.com/pkhairkh/rag-cache-engineering-qwen35-9b-flute.git`
then plain `git pull origin main` works from then on.)

Your assets (adjust if they moved): corpus
`/home/ubuntu/ragga_ingested_test100`, gates `m1m2_gates_trained.npz`
(repo root), model `/home/ubuntu/qwen3_5_9B_palettized` (+
`_heads`) — these are the driver defaults already.

**Gate zero — convert the authored W26 test into a measurement**
(~1 min, once, before the ladder):

```bash
python -m pytest src/rag/tests/test_query.py::test_install_gdn_best_mode -x
```

Expect PASS. If it FAILS, stop and paste the assert — do not patch.

---

## 2. THE DECISIVE LADDER — E1 → E2 → E3 (run in order)

All three run on the **default `sum` install on purpose** — the ladder
measures the broken path. The `[W26] GDN S install: sum` line in the
log confirms the regime. One question, one command each (~minutes):

```bash
cd /home/ubuntu/RAGGA && source /home/ubuntu/venv/bin/activate
Q="--question How do I deploy perf-canary?"
COMMON="--disk-dir /home/ubuntu/ragga_ingested_test100 $Q \
  --m1m2-gates m1m2_gates_trained.npz --max-new-tokens 128"

# E1 — n=1 verbatim install (the algebra-exact path; M1/M2 read ON)
python3 scripts/gpu/run_query.py $COMMON --top-k 1

# E2 — n=3 install (the F1 sum) + the M1/M2 answer read OFF
python3 scripts/gpu/run_query.py $COMMON --top-k 3 --no-answer-m1m2

# E3 — the cleanest regime: n=1 verbatim + no memory read
python3 scripts/gpu/run_query.py $COMMON --top-k 1 --no-answer-m1m2
```

**How to read each run** — the `[gen gate]` line the driver prints:

| verdict | signature |
|---|---|
| **clean** | mean conf **≥ 0.20**, repetition **≤ 0.6**, answer is English prose about the retrieved doc (the true-doc control measured **0.895**) |
| **garbage** | mixed-script soup from token 1 (`"by byoims车 车cse..."`), conf ~0.05–0.15 |

**Decision tree** (this is the whole point of the three runs):

| E1 | E2 | E3 | verdict → next step |
|---|---|---|---|
| clean | garbage | — | **F1 CONFIRMED** (the n≥2 S-sum is the killer) → run §3 now |
| clean | clean | — | failure needs the combination → run §3 anyway, then §5 E4 |
| garbage | clean | clean | **F3 CONFIRMED** (the M1/M2 read is the killer) → §5 follow-up |
| garbage | garbage | clean | F1+F3 together → §3 + §5 |
| garbage | garbage | garbage | **F2 is the floor** (quantized write-path tolerance) → §5 E5 bisect, paste the matrix |

---

## 3. Fix verification (only after E1 came back clean)

The F1 remedy ships in the repo as `--install-gdn best`: **S + conv
installed VERBATIM from the highest-scoring retrieved chunk, M1/M2
still summed** (their write is additive — the sum is algebra-exact for
them). Everything else in the flow is untouched. Two runs:

```bash
# E6 — the fix alone at n=3 (M1/M2 read still ON)
python3 scripts/gpu/run_query.py $COMMON --top-k 3 --install-gdn best

# E6b — the fix + the honest read mitigation (both causes killed)
python3 scripts/gpu/run_query.py $COMMON --top-k 3 --install-gdn best \
    --no-answer-m1m2
```

Expected: E6 at least partially coherent (the untrained read may
still blur it — that is F3, documented), **E6b clean** if E1 was
clean. The log line `[W26] GDN S install (n>=2): best` and the
install report's `s_source_chunk` (== the top-1 retrieved id) confirm
the mode actually ran. Paste both runs.

If E6b is clean: the pipeline's answer path is closed at n=3 — report
and STOP. The dev box will then flip the default to `best` in a
one-line follow-up commit (not you).

---

## 4. What to paste back (per run, verbatim)

1. the exact command line
2. the `[W26] GDN S install` regime line
3. the `Top k:` line (ids + scores)
4. the full `Answer:` line (first 300 chars as printed)
5. the `[gen gate] mean conf … repetition …` line
6. wall-clock time, and the full traceback if anything crashed
   (**never** patch around a crash on the box — paste it)

---

## 5. Follow-ups (only AFTER the ladder — do not jump ahead)

- **E4 (if E2 was clean — F3 confirmed)**: the sparse-read candidate
  `python3 scripts/gpu/run_query.py $COMMON --top-k 3 --m1m2-read-topk 32`
- **E5 (if E3 was garbage — F2 floor)**:
  `python3 scripts/gpu/bisect_install.py --chunk 0 $Q --max-new-tokens 32`
  and paste the whole matrix (the true-doc row re-validates the
  control under the current loader+gates regime).
- **read_gate training (F3's real fix)**: HONEST state — the finetune
  driver has **no `--aux-nll-weight` flag** (FINDINGS.md Errata #1;
  `--train-read-gate` alone trains nothing because the contrastive
  loss never touches the read). Implementing the aux-NLL path is a
  dev-box task; do not improvise training on the box. Until then
  `--no-answer-m1m2` is the honest answer-quality setting.

---

## 6. Rules of engagement (binding)

1. **GPU-ONLY law**: every number you report comes from a command you
   ran on this box. No fabricated results, no "expected" outputs.
2. **No architecture changes**: M1/M2 + GatedDeltaNet + the
   all-concatenated-caches retrieval are settled. The only new surface
   is the install mode (flag-controlled, default = the old behavior,
   bit-identical).
3. **No reprefill**: re-prefilling chunk text at query time was
   removed by order (W24). Do not re-add it.
4. Edit nothing outside the flags documented here; if a run needs a
   code change, paste the evidence and let the dev box do it.
5. Run the ladder in order; each next run depends on the previous
   verdict.

---

## 7. Map of the W26 change (where to look in the code)

| file | what |
|---|---|
| `src/rag/install.py` | `_best_conv` (~L236) + `install_snapshot(..., gdn_install="sum"\|"best")` (~L344) — the F1 remedy: S-addends sliced to the best chunk (absolute: verbatim n==1 hand-off; delta-v1: one-delta install), conv from the same chunk, M1/M2 summed unchanged |
| `src/rag/query.py` | `answer_query(..., gdn_install=)` (~L133) → the install call (~L227) |
| `scripts/gpu/run_query.py` | `--install-gdn` (~L75), the `[W26]` regime print (~L166), the pass-through (~L218) |
| `src/rag/tests/test_query.py` | `test_install_gdn_best_mode` (~L422) — the gate you run in §1 |
| `FINDINGS.md` | the full failure classification F1–F5 + the E-ladder + the Errata |
| `scripts/demo_install_algebra.py` | the synthetic delta-rule demo behind F1's numbers |

Default (`sum`) behavior is **bit-identical** to the pre-W26 install —
that is pinned by the existing test suite and by the W26 gate's (b)
leg.

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
