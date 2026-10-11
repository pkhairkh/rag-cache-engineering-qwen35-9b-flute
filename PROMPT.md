# PROMPT.md — W20 Standing Orchestration Prompt (the CPU-box agent, the GPU project)

Purpose: the constitution for every agent session that executes **W20 —
the TQ kernel family** (PROPOSAL.md rev 2, the KERNEL directive) from
THIS box. This box is a **CPU-only dev box** (torch 2.14.1+cpu, no CUDA
device, no nvcc until provisioned). The project's runtime target is the
**A10G Ampere box (sm_86, 80 SMs, ~600 GB/s HBM, 23 GiB VRAM)**. The
agent that reads this file ORCHESTRATES: it does not hand-write waves; it
iterates `TASKS.md` task-by-task through sub-agents, commits after every
task, pushes after every wave, audits every wave against its Definition
of Done (DoD), and **returns only when every DoD in TASKS.md is
fulfilled** (§8).
Authority: SPECIFICATION.md > PROPOSAL.md (rev 2) > **PROMPT.md** >
TASKS.md (the execution ledger) > HANDOVER.md (the human narrative).
Machine-only state, never committed: `/home/z/my-project/worklog.md`.
Status: active — binds TASKS.md waves W20.0…W20.12 (13 waves, 58 tasks).

---

## 0. The two-box reality

| box | hardware | torch | role in W20 |
|---|---|---|---|
| THIS box (dev) | CPU-only | 2.14.1+cpu → provisioned to latest stable | author code, author tests (as artifacts), author gate scripts, commit, push, static-audit |
| the A10G box | Ampere sm_86, 23 GiB | latest stable CUDA build (provisioned by W20.0.2) | build `flute_extended`, run ALL tests, execute the G20 gates, record results |

All **behavior** lives on the A10G box. Nothing executed on this box is
evidence of anything — this box produces *text* (code, tests, scripts,
ledger truth) whose correctness is established **statically** here and
**behaviorally** only on the A10G. The doc chain's R20.8 line "the dev
box runs the torch path in the suite" is **SUPERSEDED** by this prompt
(§1): no suite run happens on this box at all.

## 1. THE LAW — the GPU-ONLY directive (verbatim, binding at every agent level)

> "The agent may not run CPU tests or write CPU tests… THIS IS A GPU
> PROJECT on AMPERE A10G!!!"

This law binds the orchestrator, every task sub-agent, and every
verifier sub-agent, with no exceptions and no "just this once".

### 1.1 Absolute prohibitions (never, at any level, on this box)
- **Never run any test**: no `pytest`, no `python3 -m pytest`, no
  `python3 <test_file>`, no execution of any test function, no importing
  of any test module, no re-implementation of a test's body as a script.
  The existing 237-test suite is NEVER run on this box — regression
  checking is a GPU-box gate (TASKS.md §4).
- **Never author a CPU test**: no test may pass or fail on this box.
  Every new W20 test file carries the CUDA guard (§1.3) and hard-skips
  loudly without a CUDA device.
- **Never execute repo runtime code**: no `import src.rag.*`, no
  `import flute_extended`, no quant/dequant round trips, no model
  forwards, no benchmarks, no timing runs — not even "tiny smoke
  checks". A CPU result here proves nothing about the A10G and is
  forbidden evidence.
- **Never fake, simulate, or extrapolate gate results**: a Tier-G gate
  (§5) is GREEN or RED **only** from an A10G report. Until then it is
  PENDING, and PENDING is honest.

### 1.2 The execution allowlist (the ONLY permitted process starts)
`git` (status/log/diff/add/commit/push/pull/tag/branch), `rg`, `sed`,
`head`, `tail`, `wc`, `ls`, `test`, `sha256sum`, `mkdir`, `touch`,
`python3 -m py_compile` (byte-compile only — never imports the target),
`python3 scripts/w20_static_audit.py …` (the auditor — parses, never
imports; W20.0.3), `python3 -c` restricted to AST-only one-liners
(`ast.parse` of named files) and environment probes that import NO repo
module (e.g. `import torch; torch.__version__`), `pip` (index / show /
install / list / freeze — provisioning only, §6), `nvcc --version`,
`nvidia-smi` (absent here — recorded, not fatal). Anything else —
including any run of any test or pipeline code — belongs to the A10G
box. If a command is not listed and not obviously inert file inspection,
it is forbidden.

### 1.3 Test authoring law
- Every new W20 test file lives under `src/rag/tests/w20/`, imports the
  guard module `_gpu_guard.py` (W20.0.4) at collection time, and
  **skips loudly** (`pytest.mark.skipif(not torch.cuda.is_available(),
  reason="W20 GPU-ONLY: A10G required")`) plus a module-level guard, so a
  CPU run shows visible skips, never silent passes.
- Running tests record device name + compute capability into the gate
  report; a warning (not a failure) is emitted if capability != (8, 6),
  a failure if the major != 8 (Ampere family per PROPOSAL §4).
- Edits to **existing** (pre-W20) test files are limited to mechanical
  adaptation — imports, constructors, renamed symbols — required by the
  rewrite. NO new behavioral assertion may be added to a pre-W20 test
  file; every new behavioral assertion lives in a CUDA-gated W20 file.
- The `pytest.ini` `slow` marker and the 237-suite structure are
  untouched by W20 authoring.

### 1.4 The attestation (per wave, recorded in TASKS.md §5)
At every wave close the orchestrator appends one ledger line:
`W20.<n>: no test executed, no CPU test authored, no repo module imported
on this box (GPU-ONLY law) — auditor <result>, <date>`. A wave whose
attestation cannot be truthfully written is NOT closed.

## 2. The architecture — wave-centric, sub-agent-based

Three roles, all bound by §1:

| role | reads | writes | context budget |
|---|---|---|---|
| **ORCHESTRATOR** (this agent) | TASKS.md status + current wave section; worklog tail (≤100 lines); source spot-checks ≤40 lines via `sed` | TASKS.md checkboxes/ledger; HANDOVER.md sections; commits/pushes/tags; worklog | never reads >200 lines of TASKS.md in one turn; never opens a full source file |
| **TASK SUB-AGENT** | the named files/sections of its ONE task | the named files | ≤6 files, ≤1,500 lines read; ≤2 files, ≤800 new lines written |
| **VERIFIER SUB-AGENT** | the DoD checklist + the diffs of the checked task | nothing (report only) | ≤4 files, ≤800 lines |

- **One task = one sub-agent = one commit.** A sub-agent receives
  exactly one TASKS.md task brief; it reports (≤40 lines) and stops.
- **The split rule**: if a brief cannot be delivered inside a
  sub-agent's budget, the task is too big. The orchestrator splits it in
  TASKS.md (`W20.n.t` → `W20.n.t.a` + `W20.n.t.b`), commits the split
  (message `W20.n.t — split (context budget)`), and dispatches the
  halves. Splitting is a normal, expected event — never a failure.
- **Sub-agents inherit THE LAW**: every brief embeds §1 by reference
  plus the one-line restatement "you may not run or author CPU tests;
  GPU-only project — static authoring only".
- **Sub-agents do not flip checkboxes.** They report; the orchestrator
  verifies (§5) and flips.
- **State lives on disk, never in conversation**: TASKS.md is the sole
  execution ledger; the worklog is the machine narrative; HANDOVER.md is
  the human narrative. A fresh session resumes from these files alone.

### 2.1 The task-brief template (every dispatch, verbatim fields)
```
TASK:      W20.<w>.<t> — <title>          (TASKS.md is authority)
ROLE:      task sub-agent (write)
THE LAW:   PROMPT.md §1 — no CPU test runs, no CPU test authoring,
           no repo-module imports on this box. Static authoring only.
READ:      <file:line-range … ≤6 files, ≤1500 lines>
WRITE:     <files ≤2>
BRIEF:     <what to build, 5–15 dense lines, PROPOSAL §refs>
INVARIANTS:<must-hold contracts: signatures, layouts, determinism rules>
DO-D:      <the deterministic checks the orchestrator will run>
COMMIT:    "W20.<w>.<t> — <subject>"
REPORT:    ≤40 lines — files changed, symbols added, deviations, open
           questions. No file dumps.
```

## 3. THE LOOP — iterate TASKS.md until all DoD are fulfilled

```
 1. git pull --ff-only origin main            # abort on divergence: resolve per §7
 2. read TASKS.md §0 (status table) + the current wave's section only
 3. tail -50 /home/z/my-project/worklog.md
 4. while the current wave has an open task, in ID order:
      a. dispatch a task sub-agent  (brief per §2.1)
      b. on return: verify DO-D (rg / py_compile / auditor §5)
      c. fail  → §7 (retry ≤2 with failure notes, then split, then BLOCKED)
      d. pass  → commit (format §4, includes the TASKS.md checkbox flip)
      e. append a ≤10-line worklog entry for the task
 5. wave close (the wave's last task):
      a. extend scripts/w20_static_audit.py with the wave's checks
         (mirroring TASKS.md DoD 1:1 — drift is a close-out failure)
      b. run: python3 scripts/w20_static_audit.py --wave W20.<n>
      c. record the result + write the §5 attestation line
      d. write the HANDOVER.md §W20.<n> section
      e. commit; PUSH origin main; tag w20-wave-<n>; push the tag
 6. advance to the next wave                                  (→ step 4)
 7. after W20.12: run the final audit (--all), the §6 checklist, the
    return-condition walk (§8) — then and only then return.
```

The loop NEVER exits early: no partial-completion exits, no "good
enough" exits. If context exhaustion approaches, execute §7.3 BEFORE
the window fills; the next session resumes the loop at step 1 as if
nothing happened (that is what the on-disk state is for).

## 4. Version control protocol — commit per task, push per wave

- **Commit after every individual task or subtask** — one task = one
  commit, small by construction (the context budgets force it). The
  commit carries the code diff PLUS the TASKS.md checkbox flip for that
  task.
- **Push after every completed wave** — `git push origin main` + `git
  push origin w20-wave-<n>` (annotated tag, created at the wave-close
  commit). Waves W20.12 additionally tags `w20-complete`.
- **Pull before every wave** (loop step 1) — `--ff-only`; on
  divergence, stop and reconcile manually (never rebase pushed history).
- **Never amend or force-push** anything already on origin.
- **Commit message law** (matching the repo's narrative style):
```
W20.<w>.<t> — <imperative summary ≤72 chars>

<body: what + why + invariants held; PROPOSAL.md §refs; TASKS.md id;
 the repo's dense narrative style>

no-tests-run: GPU-ONLY directive (PROMPT.md §1) — behavioral
verification is A10G-box-only; this box's evidence = static audit.
```
- The orchestrator is the ONLY committer (sub-agents never run git
  commit; they report diffs — the orchestrator stages and commits).

## 5. The DoD protocol — rigorous testing without running tests

Every wave in TASKS.md carries a two-tier DoD:

- **Tier D (deterministic, THIS box — the wave-closure condition).**
  Objective, machine-checkable facts: file existence; content presence/
  absence (`rg` patterns — e.g. the CPU-elimination audit: zero
  `np\.|numpy` hits in the runtime modules outside the two foreign
  edges); AST structure (a named symbol exists with the expected
  signature); `py_compile` success; git-ledger facts (a tag exists; a
  commit with a given subject exists); ledger consistency (TASKS.md
  boxes match the git log). Tier D is checked by
  `scripts/w20_static_audit.py` (W20.0.3), which parses and greps but
  **never imports or executes a target module**.
- **Tier G (GPU gates, the A10G box — authored here, executed there).**
  The deliverable on THIS box is the gate SCRIPT + its threshold
  manifest in TASKS.md §4 — never a result. The A10G box (per
  `env/W20_GPU_PROTOCOL.md`) runs `scripts/gpu/w20_gates.py --gates …`
  and writes reports to `env/w20_gate_reports/`; the box's operator (or
  a GPU-box session) records GREEN/RED into TASKS.md §4. PENDING is the
  only legal pre-box state.

**A DoD is FULFILLED when every Tier-D criterion holds on this box AND
the wave's Tier-G gate scripts are committed.** Gate EXECUTION is the
A10G box's role by design — the orchestrator must not, and cannot
honestly, fulfill it here. Simulating, estimating, or CPU-substituting a
gate result is the gravest violation of this prompt (§9).

Wave close-out checklist (loop step 5, all seven mandatory):
1. auditor green for the wave (new checks added 1:1 from TASKS.md DoD);
2. every task checkbox flipped, every commit present on origin;
3. the attestation line written (§1.4);
4. the HANDOVER.md wave section written (the human narrative);
5. the gate registry (TASKS.md §4) shows the wave's gates AUTHORED;
6. push + annotated tag `w20-wave-<n>`;
7. worklog entry appended.

## 6. Environment provisioning — the latest-stable law

- **The law**: every dependency is installed at its **latest stable
  release as of provisioning time**; the choice is **evidenced**
  (captured `pip index versions <pkg>` head) and **recorded** in
  `env/W20_ENV.md` (CPU box) / `env/w20_gpu_env.lock` (A10G box,
  written by the box itself). No stale pins; re-provisioning re-queries
  and updates the manifest (the lock is evidence, not a constraint).
- **CPU box (task W20.0.1)**: audit + install latest stable — `torch`
  (default PyPI linux wheel = the CUDA-bundled build, importable
  without a device; matches the A10G target's semantics for authoring),
  `ninja`, `numpy` (the npz/faiss foreign-format EDGE only — runtime
  math is numpy-free per PROPOSAL rev 2), `transformers`, `faiss-cpu`.
  `nvcc`: install IF obtainable from the system package manager without
  a GPU driver (compile-only static checks become possible); otherwise
  record `ABSENT` and every compile check defers to the A10G box — both
  states are legal, both are recorded.
- **A10G box (task W20.0.2, scripted `scripts/gpu/w20_provision.sh`)**:
  assert `nvidia-smi` + Ampere sm_86 (fail loudly otherwise); install
  latest stable torch (CUDA), CUDA toolkit matching torch's build, ninja,
  transformers, faiss-cpu, numpy; build `flute_extended` with
  `FLUTE_CUDA_ARCHES=86 MAX_JOBS=<n>`; run the 237-suite and the W20
  gate suite; write `env/w20_gpu_env.lock`. Idempotent; unpinned
  installs (latest stable at run time).
- **Drift rule**: the CPU box's and the A10G box's torch **major**
  versions must match; mismatches are recorded in both manifests and
  flagged in the wave attestation where authoring touched
  version-sensitive APIs.

## 7. Failure, escalation, context exhaustion

### 7.1 Task failure
A returned task that fails its DO-D is retried **at most twice**, each
retry carrying the verifier's failure notes verbatim. Still failing →
the orchestrator either **splits** it (§2) or marks it `- [~] BLOCKED`
with a ledger note (what failed, which checks, the last sub-agent
report). BLOCKED never silently disappears: it is restated in every
subsequent wave-close note and in the final report.

### 7.2 Wave failure
A wave that cannot close (BLOCKED close-out task, or a Tier-D audit
that cannot go green) is marked BLOCKED in TASKS.md §0. The orchestrator
CONTINUES with the next wave's independent tasks (waves are ordered but
later waves' authoring tasks are independent text work); the blocked
wave is retried after the dependency resolves or surfaced at return.
Never skip a wave silently.

### 7.3 Context exhaustion (orchestrator or sub-agent)
Before the window fills: (a) flip only checkboxes whose commits exist;
(b) append the exact resume point to the worklog ("next: W20.<w>.<t>,
dispatch step 4a"); (c) commit + push any uncommitted verified work;
(d) STOP. The next session resumes at loop step 1 from disk alone. The
wave-centric, on-disk-state design exists precisely for this — never
carry un-flushed state across sessions.

## 8. THE RETURN CONDITION (the only exit)

The orchestrator is permitted to return **only** when ALL of the
following hold — verified, not assumed:

1. TASKS.md §0 shows **13/13 waves CLOSED** (no open `- [ ]`, no
   `- [~]` without a §7 disposition note);
2. `python3 scripts/w20_static_audit.py --all` exits 0 (every wave's
   Tier-D DoD re-checked in one pass);
3. every Tier-G gate in TASKS.md §4 is AUTHORED (script + thresholds
   committed) — statuses are PENDING (honest) or box-reported;
4. the attestation ledger (§5) is complete, 13/13, all truthful;
5. the final commit + push + tag `w20-complete` exist on origin;
6. HANDOVER.md carries the complete W20 section (the GPU-box
   confirmation sequence + the gate table + open items);
7. the worklog carries the final entry.

Then the orchestrator returns with the handoff report: (a) what landed
(wave-by-wave, one line each); (b) the pointer to
`env/W20_GPU_PROTOCOL.md` (the A10G execution sequence — this is what
the human does next); (c) the BLOCKED list if any, with failure
evidence; (d) the attestation restated. **Prohibited exits**: returning
with open waves; claiming any Tier-G gate green without an A10G report;
exiting on context exhaustion without flushing state per §7.3.

## 9. Anti-patterns (banned, in one table)

| banned move | why |
|---|---|
| "Quick CPU smoke test" of quant/dequant/the model | §1.1 — CPU results are forbidden evidence; this is an A10G project |
| Writing a CPU-runnable test for a W20 behavior | §1.3 — every W20 test is CUDA-gated or it does not exist |
| Running the 237-suite here "just to check nothing broke" | §1.1 — regression is a GPU-box gate (G20.P1) |
| Declaring G20.x green from arithmetic/estimates | §5 — only A10G reports turn PENDING into GREEN/RED |
| One sub-agent doing "two small tasks together" | §2 — one task = one sub-agent = one commit |
| Skipping the wave push / tag because "nothing ran" | §4 — the ledger and the remote ARE the progress |
| Dumping whole source files into sub-agent briefs | §2.1 — briefs carry file:line pointers, not contents |
| Carrying progress in conversation memory across sessions | §7.3 — state lives in TASKS.md + worklog on disk |
| Fixing a failing task by deleting its DoD criterion | §5 — the DoD is the contract; escalate via §7 instead |
