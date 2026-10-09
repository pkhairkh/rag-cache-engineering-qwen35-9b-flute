# TASKS.md v2 — Base Sync + Docs Overhaul

| field | value |
|---|---|
| mission | (A) replace inherited base files with the evolved main project `qwen3_5_9B_flute_qlora_v1.3` @ `ab78893`; (B) rewrite all docs: terse, self-contained, zero prose |
| executor | sub-agents (1 task = 1 module or 1 doc group); ORCH = verifier + committer |
| vcs | commit per task (`Wv2-<wave>.<task>`), push per wave after DoD |
| gate | every wave DoD run by ORCH in a fresh shell; red blocks push |
| plan v1 | executed 100%, tag `cpu-code-complete` (git history) |

## Sync decisions (fixed by recon, commit-time facts)

| item | action |
|---|---|
| `src/scripts/modeling.py` | REPLACE from main; RE-APPLY M1/M2 wiring (W3.2 patch) |
| `src/scripts/palettized_modules.py` | REPLACE from main; RE-APPLY flat-path idxN candidates |
| `src/scripts/attn_sm86.py` | REPLACE from main; RE-APPLY qlora de-referencing (6 refs) |
| `src/scripts/loader.py` | KEEP (ours; signature-compatible with new `load_palettized_model`) |
| `src/flute_extended/*` | REPLACE wholesale (new kernels, `include/`, nested package, tools, tests, SHA256SUMS) |
| `src/docs/*` | REPLACE with main's 12-doc set; scope to this repo in Wave 8 |
| `requirements*.txt` | MERGE: main base + corrections (transformers>=5.0, faiss-cpu, no datasets) |
| `src/rag/*`, tests, SPECIFICATION/PROPOSAL/TASKS/README roots | KEEP; re-verify; rewrite format in Waves 6–8 |
| main's trainer/palettizer/eval plane, `flute_train_kernels/` | SKIP (cut in a700100, stays cut) |

## Doc style contract (enforced in Waves 6–8)

1. Header: 3 lines — purpose, authority, status.
2. Normative content = numbered clauses (N1, N2, …), imperative, ≤2 sentences each.
3. All data in tables. No narrative, no "we/our", no motivation prose.
4. Self-contained: each doc restates every constant it uses; no doc requires another to decode a number.
5. No filler words. No paragraph >2 sentences.

## Waves

| wave | tasks (T1–T4 per wave) | DoD (fresh shell) |
|---|---|---|
| W0 sync recon | T1 diff report (done, in worklog) · T2 API-contract check (done) · T3 decision table (above) · T4 this plan committed | plan pushed |
| W1 scripts | T1 modeling.py replace+M1/M2 patch · T2 palettized_modules.py replace+idxN patch · T3 attn_sm86.py replace+de-ref · T4 loader.py verify | py_compile all; import closure; `test_finetune` green |
| W2 kernels | T1 flute_extended wholesale replace · T2 fht API gate · T3 docs base replace · T4 reference fixes | `test_turboquant` + `test_tq_cache*` + `test_m1m2` + `test_hooks` green; pushed |
| W3 requirements | T1 merge requirements.txt · T2 flute requirements · T3 install/refresh CPU lock · T4 suite run #1 (triage, no fix) | lock committed; triage list in worklog |
| W4 drift repair | T1–T2 fix triage batches · T3 loader/deploy refs · T4 re-gate | full suite green |
| W5 re-verification | T1 full ladder L1–L5 · T2 fix-forward · T3 coverage matrix re-check · T4 tag `base-synced-v1.3` | 161+ tests green; tag pushed |
| W6 root docs I | T1 SPECIFICATION rewrite (semantics preserved, diff-checked) · T2 README rewrite · T3 clause→code cross-check · T4 gate | spec normative-diff = 0 losses; pushed |
| W7 root docs II | T1 PROPOSAL rewrite (D1–D5/risk/acceptance tables) · T2 TASKS final-state record · T3 doc lint (no dangling refs) · T4 gate | every doc standalone; pushed |
| W8 src/docs | T1 scope main's doc set to this repo · T2 BUILD/RUNBOOK adopt + RAG-box page · T3 in-repo reference fixes · T4 gate | zero stale doc names repo-wide; pushed |
| W9 release | T1 full ladder · T2 docs self-containment audit · T3 worklog closure · T4 tag `v2-base-synced` | tag pushed; report |

## Verifier checklist (ORCH, per task)

1. Files changed = task's declared set only (`git status`).
2. Gate commands green in a fresh shell.
3. Worklog entry appended (Task ID `Wv2-<w>.<t>`).
4. Commit message matches `Wv2-<wave>.<task>: <imperative>`.
