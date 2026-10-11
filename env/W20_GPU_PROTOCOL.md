# W20 GPU PROTOCOL — the A10G execution sequence (fresh clone → final gate run)

Who this is for: the **A10G box operator** — the human at the Ampere
box (sm_86, 23 GiB VRAM), or a GPU-box agent session — working from a
fresh clone of this repo. The five steps below are the whole job, in
order, each block copy-pasteable. The dev box (CPU-only) authored
every script referenced here and executed nothing: THE LAW (PROMPT.md
§1) transfers to this box. Results ONLY from the A10G count — a
Tier-G gate is GREEN or RED solely from a report produced here, and
PENDING is the honest state until then (PROMPT.md §5). You are the
project's only source of behavioral evidence.

## Step 1 — Provision the box

```bash
bash scripts/gpu/w20_provision.sh
```

Idempotent and re-runnable. What it does: asserts `nvidia-smi` and an
Ampere device (fails loudly if the capability major is not 8, warns if
it is not the A10G's 8.6); records the driver/CUDA driver versions;
pip-installs the latest stable, UNPINNED torch, ninja, transformers,
faiss-cpu, numpy; installs the CUDA toolkit matching torch's build;
builds `flute_extended` in place (`FLUTE_CUDA_ARCHES=86`, MAX_JOBS
capped); runs the 237-test baseline suite — the PASS/FAIL banner is
the pre-W20 regression record; writes `env/w20_gpu_env.lock`
(versions + device + capability + build hash — evidence, not a
constraint). A failing baseline is never masked: record the banner and
stop; everything downstream compares against that baseline.

## Step 2 — Capture the frozen fixtures

```bash
git fetch --tags
git checkout --detach w20-baseline     # the numpy-era tree — BY DESIGN
python3 scripts/gpu/w20_capture_fixtures.py
```

The checkout is the pre-rewrite **numpy-era** tree by design — the
capture pins the legacy TurboQuant semantics that the rewrite must
reproduce. The script asserts the exact commit
`e9ea357fee64478dc9738e1c6955793c8b3a108d` (tag `w20-baseline`, short
`e9ea357`) and refuses any other checkout; it also refuses to run
without CUDA. It runs the 14 seeded probes (G1 d=2^19 ×4 units;
G2 d=24,576 ×4, the conv full-window; G3 d=2^22 ×2; kinds mse, plain
b∈{3,4}, the 3.5-bit lo/hi split, qjl on/off) through the legacy path
and writes `src/rag/tests/w20/fixtures/*.npz` plus
`src/rag/tests/w20/fixtures/fixtures_manifest.json` (per-file sha256,
probe seeds, turboquant config, codebook digests). Any codebook the
capture solves and caches lands in `src/rag/codebooks/`
(`cb_b{bits}_d{d}.npz`) — those files ride the Step 3 commit.

## Step 3 — Commit the fixtures FROM the box

```bash
git checkout main
git add src/rag/tests/w20/         # fixtures/ + fixtures_manifest.json
git add src/rag/codebooks/         # only if the capture cached new cb_*.npz
git commit -m "W20.G: fixtures captured from w20-baseline (GPU box)"
git push origin main
```

Fixtures are committed **from the box** — the dev box never fabricates
them (a dev-box "fixture" would be forbidden evidence, PROMPT.md
§1.1). The manifest's sha256 values are the frozen reference that
every later parity gate (G20.P1/P3/P4) compares against, so push
exactly what the capture wrote: no hand edits, no re-formatting, no
partial selection. If the box cannot push, keep the commit local and
say so in the handback — the dev box must not author the files
instead.

## Step 4 — Run the per-wave gates as waves land

After each authoring wave lands on origin: pull, then run the gates
that wave authored (`scripts/gpu/w20_gates.py` itself lands with
W20.4's push — until then there is nothing to run):

```bash
git pull --ff-only origin main
python3 scripts/gpu/w20_gates.py --gates G20.P1                    # after W20.4
python3 scripts/gpu/w20_gates.py --gates G20.P2,G20.6              # after W20.6
python3 scripts/gpu/w20_gates.py --gates G20.P3,G20.1,G20.2,G20.4  # after W20.8 (qjl: W20.9)
python3 scripts/gpu/w20_gates.py --gates G20.5                     # after W20.10
python3 scripts/gpu/w20_gates.py --gates G20.2,G20.3               # the W20.11 full runs
```

Each run writes `env/w20_gate_reports/<gate_id>.json` — gitignored BY
DESIGN (box-written, never committed; this repo's .gitignore entry
exists for exactly that). Each report carries the status, the
headline numbers, and the device identity. The box operator (or the
next dev-box session) transcribes the statuses + headline numbers
into TASKS.md §4 — the transcription, not the json files, is the
ledger.

### The gate registry (TASKS.md §4, verbatim — 9 gates)

| gate | contract | thresholds | authored in | status |
|---|---|---|---|---|
| G20.P1 | K1: the torch single path vs the frozen fixtures + the 237-suite regression | codes equal except ties ≤ 0.01%; norm ≤ 1 ulp; recon allclose ≤ 1e-5 rel; run-to-run `torch.equal` (CUDA); 237-suite 0 failures | W20.4 | NOT-AUTHORED |
| G20.P2 | K2: fht_large differential twins + determinism | kernel vs debug twin vs torch allclose ≤ 1e-5; run-to-run `torch.equal`; K%32≠0 and K≤16,384 refusals raise | W20.6 | NOT-AUTHORED |
| G20.P3 | K3+K4: TQ kernel parity (quant+dequant+qjl) | same as G20.P1 thresholds on kernel outputs; batched-grid shapes 120–192 units | W20.8 (qjl folds in W20.9) | NOT-AUTHORED |
| G20.1 | capture cycle + TQ share | capture ≤ 10 s/cycle after K3; TQ ≤ 5 ms/prefill (phase timers) | W20.8 (phase-1) | NOT-AUTHORED |
| G20.2 | step time + the 300-step run | step ≤ 2 s after K3; ≤ 1 s at refresh 8; 300 steps ≤ 10 min | W20.8 (phase-1) / W20.11 (full) | NOT-AUTHORED |
| G20.3 | W19 convergence (re-run on the kernel path) | loss leaves ln(8) at step 0; gk/gv_absmax > 0; held-out spread positive | W20.11 | NOT-AUTHORED |
| G20.4 | the parity contract (levels 1–2) | level 1: run-to-run `torch.equal` (codes + recon, kernel); level 2: ties ≤ 0.01%, norm ≤ 1 ulp, allclose ≤ 1e-5 | W20.8 | NOT-AUTHORED |
| G20.5 | protocol integrity | TRAIN → RE-INGEST → INDEX → EVAL → QUERY on the kernel path; npz digests stable run-to-run; cross-backend decode allclose ≤ 1e-5; W16 verbatim install bit-exact within backend | W20.10 | NOT-AUTHORED |
| G20.6 | canary/trap | canary build: zero poison survives; trap build: FAILS the determinism gate (the negative control must fail) | W20.6 | NOT-AUTHORED |

Statuses above are the authoring-time snapshot (this doc is written in
W20.1, before any gate exists); TASKS.md §4 is the live ledger you
transcribe into. Thresholds are each gate's source of truth, mirrored
1:1 into `w20_gates.py`'s THRESHOLDS dict — the registry table and the
script are authoritative over any transcription.

## Step 5 — The final full registry run

After W20.12 lands (the final push + the `w20-complete` tag), re-run
every gate in one pass:

```bash
git pull --ff-only origin main
python3 scripts/gpu/w20_gates.py --gates all
sha256sum env/w20_gate_reports/*.json
```

Every gate re-executes against the final tree; the full report set
stays under `env/w20_gate_reports/` (gitignored — archive a copy
elsewhere if you need durability; the reports are never committed).
Transcribe the final statuses into TASKS.md §4 and copy the verdict
table (gate, status, headline numbers) back into the dev-box
TASKS.md §4 + HANDOVER.md — the W20 close-out section. The sha256
list printed above is part of the deliverable.

## What the box may NOT do (the law, mirrored)

- **No re-implementing gates as estimates**: a gate result exists only
  as `w20_gates.py`'s own report from this box; arithmetic, analogy,
  or "it should pass" is not evidence (PROMPT.md §1.1, §9).
- **No editing thresholds**: thresholds live in TASKS.md §4, mirrored
  1:1 into `w20_gates.py`'s THRESHOLDS dict. If one looks wrong,
  report it — the dev box changes it, via a task and a commit.
- **A RED is a finding, not a failure to hide**: file the report,
  transcribe the RED verbatim with its headline numbers; the dev box
  decides the fix. Masking a RED is the protocol's gravest violation
  (PROMPT.md §9).
- **No fabricated artifacts**: fixtures come from the `w20-baseline`
  checkout (Step 2); reports come from real runs of real scripts.
  Anything this box did not itself produce is not evidence.

## Reporting back (the box's deliverable)

Two artifacts, nothing else:
1. the transcribed statuses — every gate's GREEN/RED + headline
   numbers written into TASKS.md §4 (mirrored into HANDOVER.md at the
   final run); the dev-box orchestrator resumes from these files
   alone;
2. the sha256 list of the gate report files
   (`sha256sum env/w20_gate_reports/*.json`) — the tamper-evident
   fingerprint of the report set, quoted in the handback.
A RED is reported exactly the way a GREEN is — with numbers.
