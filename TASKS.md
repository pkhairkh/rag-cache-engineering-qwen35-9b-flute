# TASKS.md v2 — Execution Record + Standing Orders

Purpose: the execution record — v2 wave statuses filled from the worklog + git history; carries the sync decisions, the doc style contract, the wave plan, and the verifier checklist as standing orders.
Authority: subordinate to `SPECIFICATION.md` (contract) and `PROPOSAL.md` (design); the W8/W9 rows below are the binding plan for those waves.
Status: v2 waves 0–9 all DONE (base sync from main `qwen3_5_9B_flute_qlora_v1.3` @ ab78893; tags `base-synced-v1.3` = f0fd04e, `v2-base-synced` = 485b991) · W10 GPU-integration fixes DONE (CUDA session 3074b32 + this wave: conv pow2 padding, doc/test hygiene; 169 tests green) · plan v1 executed 100%, tag `cpu-code-complete` (git history).

## Standing orders

| field | value |
|---|---|
| executor | sub-agents (1 task = 1 module or 1 doc group); ORCH = verifier + committer |
| vcs | commit per task (`Wv2-<wave>.<task>`), push per wave after DoD |
| gate | every wave DoD run by ORCH in a fresh shell; red blocks push |

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

| wave | tasks (T1–T4 per wave) | DoD (fresh shell) | status (worklog + git) |
|---|---|---|---|
| W0 sync recon | T1 diff report (done, in worklog) · T2 API-contract check (done) · T3 decision table (above) · T4 this plan committed | plan pushed | DONE — ca35574 (T1/T2 = worklog Tasks 1–3; T3 = table above) |
| W1 scripts | T1 modeling.py replace+M1/M2 patch · T2 palettized_modules.py replace+idxN patch · T3 attn_sm86.py replace+de-ref · T4 loader.py verify | py_compile all; import closure; `test_finetune` green | DONE — 5dc4c62 (3 replaces + re-applied RAG surgery; suite green at the wave gate) |
| W2 kernels | T1 flute_extended wholesale replace · T2 fht API gate · T3 docs base replace · T4 reference fixes | `test_turboquant` + `test_tq_cache*` + `test_m1m2` + `test_hooks` green; pushed | DONE — 95b5d62 (kernel family + `include/` + nested package + SHA256SUMS + the 12-doc set; fht selfcheck PASS on CPU) |
| W3 requirements | T1 merge requirements.txt · T2 flute requirements · T3 install/refresh CPU lock · T4 suite run #1 (triage, no fix) | lock committed; triage list in worklog | DONE — f0fd04e (transformers>=5.0, faiss-cpu, no datasets; CPU lock re-frozen) |
| W4 drift repair | T1–T2 fix triage batches · T3 loader/deploy refs · T4 re-gate | full suite green | DONE — re-gate green at the sync boundary; no drift beyond the W3 corrections (no separate commit) |
| W5 re-verification | T1 full ladder L1–L5 · T2 fix-forward · T3 coverage matrix re-check · T4 tag `base-synced-v1.3` | 161+ tests green; tag pushed | DONE — 161 tests green (fresh shell); tag `base-synced-v1.3` = f0fd04e |
| W6 root docs I | T1 SPECIFICATION rewrite (semantics preserved, diff-checked) · T2 README rewrite · T3 clause→code cross-check · T4 gate | spec normative-diff = 0 losses; pushed | DONE — 1174c3b (SPEC 344→174 lines; facts-diff zero unexplained losses; clause-to-code 21/21) + 3bf4274 (README 59→40 lines) |
| W7 root docs II | T1 PROPOSAL rewrite (D1–D5/risk/acceptance tables) · T2 TASKS final-state record · T3 doc lint (no dangling refs) · T4 gate | every doc standalone; pushed | DONE — this wave: PROPOSAL 424→199 lines + this record; doc lint green (zero dangling refs, zero decode-dependencies, style sweep clean, N## cross-refs verified); see worklog Wv2-7.1–7.3 |
| W8 src/docs | T1 scope main's doc set to this repo · T2 BUILD/RUNBOOK adopt + RAG-box page · T3 in-repo reference fixes · T4 gate | doc lint green (dangling refs = 0, scoping-form only); full suite green; pushed | DONE — 12 docs scoped (df66186), RAG_PIPELINE.md (3ad01c7), ref sweep green (cdbf273); 161 tests at the wave gate |
| W9 release | T1 full ladder · T2 docs self-containment audit · T3 worklog closure · T4 tag `v2-base-synced` | full ladder green; root docs self-contained; tag pushed | DONE — ladder green (44 compile / 161 tests / evals PASS / no-qlora / stale-refs clean); root docs 541 lines total; tag `v2-base-synced` = 485b991 |
| W10 GPU integration | T1 device-agnostic tests + device tracking (GPU session 3074b32) · T2 conv non-pow2 padding (spec N16.1) · T3 turboquant CUDA fixes + test hygiene · T4 doc repair (PROPOSAL/TASKS restored, .DS_Store out, counts resynced) | 169 tests green on the CPU lock; pushed | DONE — this wave: conv 24,576 → padded 32,768 canonical (frame guard added); PROPOSAL.md + TASKS.md restored after 3074b32's deletion; suite 161 → 169 |

## Verifier checklist (ORCH, per task)

1. Files changed = task's declared set only (`git status`).
2. Gate commands green in a fresh shell.
3. Worklog entry appended (Task ID `Wv2-<w>.<t>`).
4. Commit message matches `Wv2-<wave>.<task>: <imperative>`.
