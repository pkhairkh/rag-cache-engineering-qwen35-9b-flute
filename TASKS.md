# TASKS.md — CPU Coding Box: Wave Execution Plan

**Status:** standing orders for the orchestrator (ORCH) and its sub-agents (SUB)
executing on the CPU coding box.

**Scope:** write and verify ALL CODE required by `SPECIFICATION.md` (the
contract), following the design of `PROPOSAL.md`. This box has no GPU and no
CUDA toolchain; every task below is code-complete and CPU-verified. GPU-only
execution (kernel bring-up, real-model measurement, the fine-tune run, 50k
ingestion, timing/VRAM ledgers) is registered in §7 and is explicitly NOT a
task on this box.

**Document hierarchy:** `SPECIFICATION.md` (the what/why — contract) >
`PROPOSAL.md` (the how — decisions D1–D5, phases P0–P7, risks R1–R8) >
`TASKS.md` (this file — execution: waves, tasks, DoD gates).

---

## 0. Charter

**IN (this box):**
- All new code under `src/rag/` per PROPOSAL §3: `codebooks.py`,
  `turboquant.py`, `tq_cache.py`, `m1m2.py`, `hooks.py`, `snapshot.py`,
  `ingest.py`, `index.py`, `install.py`/`query.py`, `finetune.py`,
  `lut_export.py`, `evals.py` (+ `__init__.py`, `_paths.py`, `tests/`).
- The one surgical `modeling.py` edit (M1/M2 wiring behind a config flag).
- The CPU test suite with paper-constant numeric gates (pytest).
- Environment provisioning and the frozen CPU lock.
- Repo hygiene: a commit per task, a push per wave, a worklog entry per task.

**OUT (GPU box — see §7):** the `flute_extended` CUDA build (DEPLOY.md) and
Phase 0 baseline; all real-model measurements; the fine-tune run; the 50k
chunk ingestion; OfficeQA end-to-end; the §9/§10 ledgers.

**Terminal state (all DoD green):** the repo carries the complete RAG build
code for spec §1–§13; every module compiles; imports close; the CPU test
suite is green including the paper-constant gates; every GPU gate in the
proposal's §5 acceptance table maps to a ready-to-run `evals.py` entry point
listed in §7.

---

## 1. Operating protocol

### 1.1 Roles

| Role | Who | Does |
|---|---|---|
| ORCH | the main agent on this box | wave sequencing; DoD gate runs; VCS pushes; cross-file surgery (`modeling.py`); spec-contract modules (`turboquant.py`, `tq_cache.py`, install math, `finetune.py`, `evals.py`); sub-agent dispatch |
| SUB | `general-purpose` sub-agents (Task tool) | leaf tasks: one module and/or one test file with a frozen contract (§5 brief template) |

### 1.2 Context-window discipline

- One SUB task = at most one source module + its test file.
- SUB briefs are self-contained (≤ ~2,000 words): file paths, API contract,
  shapes, numeric constants, test commands, VCS and worklog obligations.
  Never "read the spec and figure it out" — the brief inlines the relevant
  §refs and numbers.
- ORCH keeps no conversation state across waves: the repo and
  `/home/z/my-project/worklog.md` are the only memory.
- A SUB output that fails its DoD line is fixed forward by ORCH in the same
  task before commit. Never commit red. Never re-delegate a fix.

### 1.3 Worklog

Path: `/home/z/my-project/worklog.md` (outside the repo). Every agent reads
it before starting and appends after finishing each task, using exactly:

```markdown
---
Task ID: W<n>.<m>
Agent: <name>
Task: <one line>

Work Log:
- <concrete steps>

Stage Summary:
- <result / decisions / artifacts>
```

### 1.4 Version control

- Commit after EVERY task/subtask: message format `W<n>.<m>: <summary>`.
- Push after EVERY wave, only once its DoD gate is green:
  `git push origin main`, then `git fetch origin && git status -sb` must
  show the branch in sync.
- No stray files (`.gitignore` already covers `__pycache__/`, `*.pyc`).
- Final wave: `git tag cpu-code-complete` and push the tag.

### 1.5 DoD gates

- Each wave's DoD (§4) is a binary checklist. ORCH runs every gate command
  itself, in a fresh shell, after the wave's last commit.
- Any red line blocks the push. Fix forward, re-run the whole gate (not just
  the failed line), then push.
- Gate results (pass/fail per line + command output tails) are appended to
  the worklog as the wave's closing entry.

### 1.6 Return contract

ORCH reports back to the operator ONLY when either (a) every wave DoD in
this file is green and the `cpu-code-complete` tag is pushed, or (b) a DoD
is demonstrably blocked by a missing external input (GPU, model artifacts,
corpus) — with worklog evidence. No partial-progress returns.

### 1.7 Dependency lock

Wave order is topological — no reordering, no skipping. No new third-party
dependency enters the repo without a wave-level decision recorded in the
worklog. Runtime deps live in `src/requirements.txt`; the CPU verification
env is frozen in `src/requirements-cpu.lock.txt`.

### 1.8 House code conventions

- Entry modules (`src/rag/__init__.py`, CLI runners, `src/scripts/loader.py`)
  stay stdlib-only at import; heavy imports (torch, faiss, transformers) go
  inside function bodies. Leaf modules may import torch/numpy at module top.
- Path anchors: `src/rag/_paths.py` inserts `src/rag/`, `src/scripts/`,
  `src/flute_extended/` into `sys.path` (house convention).
- Docstrings cite spec §§ and proposal decisions (D1–D5). Dtype/shape gates
  fail loudly (house pattern: the kernel path dtype-gates promoted masters).
- No handoff prose documents: code + tests + this file. Runbook text is
  `--help` output on CLI entry points.

---

## 2. Verification ladder (CPU)

| L | Check | Command (repo root) |
|---|---|---|
| L1 | compile | `python3 -m py_compile $(git ls-files '*.py')` |
| L2 | import closure | grep sweep: no import of a non-existent module; entry modules stdlib-only at import |
| L3 | unit tests | `python3 -m pytest src/rag/tests -q` |
| L4 | paper-constant numeric gates | inside L3 (codebook centroids, D_mse tolerances — Wave 1) |
| L5 | integration on synthetic stubs | inside L3 (stub cache, stub 32-layer stack, mini-RAG, install math) |

GPU-only measurements are NOT CPU DoD — they are the §7 register, executed
on the A10G box through the `evals.py` entry points this plan builds.

---

## 3. Environment policy (latest-stable rule)

Provision this box with the LATEST STABLE versions of everything
CPU-installable:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install numpy transformers safetensors huggingface_hub faiss-cpu pytest
```

- Always `python3 -m pip` (the bare `pip` on this box targets a different
  interpreter).
- `faiss-cpu` is also a RUNTIME dep (spec §10: the IVFADC index lives in
  CPU RAM, mmap'd) → add it to `src/requirements.txt` with a comment.
- Freeze the verified set into `src/requirements-cpu.lock.txt` (the exact
  `pip freeze` of the installed set, headed by a `python3 --version` comment).
- Documented exception (engineering, not neglect): the CUDA-bound deps
  (`triton`, `flash-linear-attention==0.5.2`, `causal_conv1d==1.7.0`, the
  `flute_extended` extension build) are NOT installable on this CPU box.
  On the GPU box they follow `src/requirements.lock.txt` (known-good,
  hardware-bound pins) and `src/docs/DEPLOY.md`. The latest-stable rule
  governs the CPU verification set only; it does not override known-good
  GPU pins. Recorded here once; do not re-litigate per wave.

---

## 4. Waves

### Wave 0 — Environment provisioning + scaffold + baseline

Objective: provision the CPU verification env, freeze it, scaffold `src/rag/`,
and baseline-verify the existing repo under the new env.

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W0.1 | ORCH | env installed (torch-cpu, numpy, transformers, safetensors, huggingface_hub, faiss-cpu, pytest — latest stable); `src/requirements-cpu.lock.txt` frozen; `faiss-cpu` added to `src/requirements.txt` | `python3 -c "import torch, faiss, transformers, numpy; print(torch.__version__, faiss.__version__)"` |
| W0.2 | ORCH | `src/rag/__init__.py` (stdlib-only; docstring = spec §→module map), `src/rag/_paths.py`, `src/rag/tests/conftest.py` (path setup + tiny-tensor fixtures), `pytest.ini` (repo root, `testpaths = src/rag/tests`) | `python3 -m pytest -q` collects with zero errors on the empty suite |
| W0.3 | ORCH | baseline sweep: L1 over all tracked `.py`; run the five `scripts/poc_toy/` toys (the §12 validated references); run `flute_extended.fht._selfcheck()` (CPU backend); record env versions in the worklog | all toys exit 0; fht selfcheck green |

**DoD (Wave 0):**
- [ ] `python3 -m py_compile $(git ls-files '*.py')` exits 0
- [ ] `python3 -c "import torch, faiss, transformers"` works; lock file committed
- [ ] all five toys run clean; `fht._selfcheck()` green
- [ ] `python3 -m pytest -q` collects with no errors
- [ ] pushed; branch in sync with origin/main

### Wave 1 — TurboQuant core — `codebooks.py`, `turboquant.py`
*(proposal P1, spec §3.1/§3.3)*

Objective: the pure-math quantizer — Lloyd-Max codebooks on the Beta density
and the unit wrapper (norm → FHT → split → bucketize → pack). CPU-verifiable
against the paper's own constants.

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W1.1 | SUB | `src/rag/codebooks.py`: continuous 1-D k-means (Lloyd-Max) on the Beta density `f(x) = Γ(d/2)/(√π·Γ((d−1)/2))·(1−x²)^((d−3)/2)`, x ∈ [−1,1]; solve per bit-width b ∈ {1..4} for d ∈ {2¹⁵, 2¹⁹} (analytic cell integrals between Voronoi midpoints; iterate to fixed point); on-disk cache `src/rag/codebooks/` (npz: centroids + boundaries per (b, d)); validate the centroids ∝ 1/√d scaling law | W1.2 |
| W1.2 | SUB | `tests/test_codebooks.py`: b=1 centroids = ±√(2/π)/√d within 0.5%; b=2 = {±0.453, ±1.51}/√d within 1%; fixed-point convergence; codebook file round-trip; scaling law | pytest |
| W1.3 | ORCH | `src/rag/turboquant.py`: `TurboQuant` class — kinds: S (d=524,288), conv (32,768), M1/M2 (524,288); `quant(x)`: fp norm stored, `r = x/‖x‖₂` → `fht_apply(r, signs)` with `rotation_signs(K, seed)` per kind (seeds `seed_s`, `seed_c`, `seed_m1`, `seed_m2` — D3, one shared rotation per kind) → 3.5-bit split (default 50/50 coords at 3/4 bits; optional outlier partition recorded as metadata — D2) → bucketize to codebook boundaries → pack 3/4-bit indices into uint8; `dequant(codes)`: unpack → centroids → `fht_adjoint` → × norm; `TQCodes` dataclass + npz `save/load`. FHT via `flute_extended/fht.py` (CPU: reference backend) | W1.4 |
| W1.4 | SUB | `tests/test_turboquant.py`: round-trip MSE on random unit vectors at b ∈ {1,2,3,4} ≤ 2.72·4⁻ᵇ·‖x‖² (paper D_mse bound) and b=2 within 5% of 0.117; FHT binding: `fht_apply`/`fht_adjoint` round-trips agree with `build_rotation_matrix` ground truth; pack/unpack lossless; 3.5-bit size gate: one full-scale synthetic chunk (24 S units + 24 conv + M1 + M2) serializes to 5.9–6.1 MiB (spec §3.3/§5); code file round-trip | pytest |

**DoD (Wave 1):** all four test groups green, including the paper-constant
gates (b=1, b=2 centroids; D_mse tolerances) and the 6 MiB/chunk size gate;
pushed.

### Wave 2 — Online cache wrapper — `tq_cache.py`
*(proposal P2, spec §3.2/§2.1/§2.3)*

Objective: quantize-on-write / dequantize-on-read interception so the cache
NEVER holds fp16. Interception points are the exact `modeling.py` call sites:
READ `cache_params.layers[L].recurrent_states[0]`, READ
`cache_params.layers[L].conv_states[0]`, WRITE
`cache_params.update_recurrent_state(state, L)`, WRITE
`cache_params.update_conv_state(...)` — on the `transformers.cache_utils`
`DynamicCache` protocol (`has_previous_state`, `layers[L].record_past`).

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W2.1 | ORCH | `src/rag/tq_cache.py`: `TQCache` — wraps/monkey-patches the cache object passed as `past_key_values` per spec §3.2: recurrent/conv reads dequantize, writes quantize; per-kind `TurboQuant` registry (D3 seeds); M1/M2 code slots exposed as `tq_cache.m1_codes` / `.m2_codes` (spec §5); `quantize-on-snapshot` fallback mode flag (D4, default off) | W2.2, W2.3 |
| W2.2 | SUB | `tests/test_tq_cache.py` (stub cache object with the exact layer protocol): write-then-read returns `dequant(quant(x))` within the Wave-1 MSE budget; every read/write intercepted (counters); model-visible API shape unchanged; **invariant: after a write, no fp16/fp32 state tensor is resident in the code stores** (dtype sweep) | pytest |
| W2.3 | SUB | `tests/test_tq_cache_live.py`: the same contracts against the REAL `transformers` `DynamicCache` (constructed per `modeling.py`'s `DynamicCache(config=...)` with a minimal Qwen3_5 config, CPU). If the live protocol diverges from the stub contract, record the delta in the worklog — the stub remains the binding contract | pytest |

**DoD (Wave 2):** stub-contract tests green (interception, MSE budget,
no-fp16-resident invariant); live-protocol test green or its divergence
documented in the worklog; pushed.

### Wave 3 — M1/M2 addition — `m1m2.py` + `modeling.py` wiring
*(proposal P3, spec §2.2)*

Objective: the two global memories — buffers, gated additive writes,
softmax reads — wired behind a config flag; zero-init = bit-identical no-op.

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W3.1 | SUB | `src/rag/m1m2.py`: `M1M2` module — buffers M1, M2 `(1, 32, 128, 128)` fp16, shared across all 24 linear layers; write `M += write_gate ⊙ w(k, v)` (gated, additive → path-independent deltas, spec §2.2); read `softmax(q @ M1ᵀ) @ M2` folded into the layer output path; write gates zero-init (no-op at init — proposal P3) | W3.3 |
| W3.2 | ORCH | `modeling.py` surgery: wire `M1M2` into `Qwen3_5GatedDeltaNet` behind config flag `use_m1m2` (default OFF for parity; the RAG build turns it on). Flag off = pure guard, zero code-path change (house pattern from the a700100 edits) | W3.3 + L1/L2 |
| W3.3 | SUB | `tests/test_m1m2.py`: (i) build a synthetic random-weight `Qwen3_5GatedDeltaNet` (CPU reference path), run it, attach `M1M2` with flag OFF, run again → bit-identical; (ii) flag ON with zero gates → bit-identical to (i); (iii) nonzero gates: read term `softmax(q @ M1ᵀ) @ M2` matches a numpy reference; (iv) additive-write composability: ΔM from two write sequences sums order-independently | pytest |

**DoD (Wave 3):** bit-identical parity tests green (flag off; flag on with
zero gates); numpy-reference read test green; composability test green;
pushed.

### Wave 4 — Hook capture harness — `hooks.py`
*(spec §1)*

Objective: the 9 capture points. Contract (enumerated from the spec §1
diagram; totals 1+2+7×3 = 24 — the spec's inline "1 + 8×3" line miscounts
its own diagram; the enumerated map below is the authority, and this
discrepancy is recorded in the worklog):

```
hook 0: after layer 0  → S_0
hook 1: after layer 3  → S_1, S_2
hook 2: after layer 7  → S_4, S_5, S_6
hook 3: after layer 11 → S_8, S_9, S_10
hook 4: after layer 15 → S_12, S_13, S_14
hook 5: after layer 19 → S_16, S_17, S_18
hook 6: after layer 23 → S_20, S_21, S_22
hook 7: after layer 27 → S_24, S_25, S_26
hook 8: after layer 31 → S_28, S_29, S_30
```

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W4.1 | SUB | `src/rag/hooks.py`: `CaptureHooks` — registers the 9 forward hooks on the 32-layer stack (layer_types pattern [L,L,L,F]×8); captures each S from the TQ cache stores (raw codes retained, dequantized snapshot provided); returns a `CacheSnapshot` (24 S codes + 24 conv codes + M1/M2 codes, spec §5) | W4.2 |
| W4.2 | SUB | `tests/test_hooks.py`: stub 32-layer stack → hook map EXACTLY the enumerated sets; exactly 24 S tensors captured; capture→quant→dequant round-trip within the Wave-1 MSE budget; idempotent re-run | pytest |

**DoD (Wave 4):** hook-map equality green (24/24 tensors, exact sets);
round-trip gate green; pushed.

### Wave 5 — Ingestion + snapshot serialization — `snapshot.py`, `ingest.py`
*(proposal P5 coding, spec §5/§11/§4)*

Objective: the on-disk chunk format and the delta protocol (D4).

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W5.1 | SUB | `src/rag/snapshot.py`: the npz codec — exact spec §5 schema (`s_codes` per linear layer, `conv_codes`, `m1_codes`, `m2_codes`, `cache_vector` fp32 len 13,631,488); `save/load/verify` (schema + checksum); §11 layout `disk/snapshots/chunk_XXXXX.npz` | W5.3 |
| W5.2 | ORCH | `src/rag/ingest.py`: `ingest_chunk` per spec §5 (online-TQ prefill → snapshot codes → dequantize the retrieval vector) + the D4 delta protocol: system-prompt reset point (quantized once), per-chunk `delta_i = S_chunk − dequant(S_sys)` → `TQ.quant(delta_i)` stored; batch driver with a resume manifest (JSON of completed chunk ids — ingestion is restartable, embarrassingly parallel over chunks); retrieval-vector assembly in the exact §4 order (S sorted by layer, then M1, M2) | W5.3 |
| W5.3 | SUB | `tests/test_ingest.py`: (i) full-scale synthetic single-chunk npz lands 5.9–6.1 MiB and round-trips losslessly at the code level; (ii) delta protocol on a stub model: `dequant(S_sys) + Σ dequant(delta_i)` reconstructs each chunk state exactly (stub deltas are exact); (iii) retrieval vector: length 13,631,488, dtype fp32, §4 order; (iv) driver resume: interrupt after k chunks, restart → no duplicates, manifest consistent | pytest |

**DoD (Wave 5):** size gate, code round-trip, delta math, §4 order, and
resume tests green; pushed.

### Wave 6 — Index + retrieval — `index.py`
*(proposal P5 coding / D5, spec §8)*

Objective: IVFADC build + preselect + rerank with side-metadata.

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W6.1 | SUB | `src/rag/index.py`: `build_index(...)` → `faiss.IndexIVFPQ(faiss.IndexFlatIP(13631488), 13631488, nlist=224, m=64, nbits=8)` with `nprobe=8`; train/add over dequantized cache vectors streamed from snapshots (never materialized en masse — D5); side-metadata JSON persisted beside `disk/ivfadc_cache.index` (per-kind rotation seeds, partitions, codebook hashes, d/nlist/m/nbits) | W6.3 |
| W6.2 | SUB | search path: `preselect(query_vector, k=100)` (IVFADC, nprobe=8) + `rerank(candidates, query_vector, k=3)` — exact cos-sim on full dequantized vectors loaded on demand from TQ codes, one candidate at a time (§6/D5) | W6.3 |
| W6.3 | SUB | `tests/test_index.py` (faiss-cpu, reduced dims): (i) synthetic recall@100 mechanics — query = stored vector + noise → itself in top-100 and reranked into the top-3; (ii) side-metadata round-trip; (iii) API fidelity: the full-scale constants (d=13,631,488, nlist=224, m=64, nbits=8, nprobe=8) are exactly what the builder passes (construct the index object untrained, assert params); (iv) rerank loads candidates lazily (mock loader counts reads) | pytest |

**DoD (Wave 6):** reduced-dim recall mechanics, metadata round-trip,
param fidelity, and lazy-rerank tests green; pushed.

### Wave 7 — Query flow + install — `query.py`
*(proposal P6 coding, spec §6)*

Objective: the eight-step §6 flow and the install math.

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W7.1 | ORCH | `src/rag/install.py`: `sum_turboquant_codes(system, deltas)` = `TQ.quant(dequant(system) + Σ dequant(deltas))` — dequant-sum-requant ONCE (D4: FHT linear + shared per-kind rotation = "sum in the rotated space" from spec §6's note); Lloyd-Max is not additive — raw codes are never summed; conv_state = LAST retrieved chunk verbatim (spec §6); M1/M2 install by the same sum protocol as S | W7.3 |
| W7.2 | ORCH | `src/rag/query.py`: the §6 flow — tokenize → prefill query (online-TQ cache) → snapshot the query vector → preselect → rerank → load top-3 → install → answer → decode; per-step timing hooks (the §9 ledger instrumentation) | W7.3 |
| W7.3 | SUB | `tests/test_query.py`: (i) install math property: `install(sys, [d1, d2])` ≡ `quant(dequant(sys) + dequant(d1) + dequant(d2))` (property test against direct computation); (ii) raw-code summation path does not exist (test asserts the code API offers no code-plus-code op); (iii) conv last-chunk rule; (iv) mini-e2e RAG on the stub model + synthetic corpus (pattern: `scripts/poc_toy/toy_cache_engineered_rag.py`): the query retrieves the correct chunk and installs it (mechanics-level — the toys' §12 ceiling) | pytest |

**DoD (Wave 7):** install property tests, conv rule, and the mini-e2e green;
pushed.

### Wave 8 — Fine-tune loop — `finetune.py`, `lut_export.py`
*(proposal P4 coding, spec §7/§11)*

Objective: the lean loop, CPU-dry-runnable on synthetic tensors.

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W8.1 | ORCH | `src/rag/finetune.py`: next-token loop — trainables: the 24 per-layer linear-attn param groups, the M1/M2 read/write gates, and the LUTs via `PalettizedLinear.make_trainable()` (reference path `forward="reference"` per spec §7 — the straight-through primitive this repo kept); AdamW + cosine, `max_steps≈500` configurable; fp32 LUT masters, everything else fp16/bf16; after training `freeze_lut(snap_fp16=True)` then export | W8.3 |
| W8.2 | SUB | `src/rag/lut_export.py`: export/import codec for fine-tuned LUTs → `pretrained_luts/` (per-layer files, version header, checksums — full-LUT artifacts, spec §11, NOT adapters) | W8.3 |
| W8.3 | SUB | `tests/test_finetune.py`: CPU dry-run — synthetically construct a tiny `PalettizedLinear` (programmatic LUT + residual), attach `M1M2` + stub cache, run 3 steps: loss finite, LUT grads nonzero, linear-attn params receive grads, `freeze_lut` snaps to the fp16 grid, export → import round-trips bit-equal | pytest |

**DoD (Wave 8):** dry-run green (grads flow through the straight-through
path); export round-trip bit-equal; grep clean — no `qlora` import anywhere
under `src/rag/`; pushed.

### Wave 9 — Gate harness + hardening flags — `evals.py`
*(proposal §5 acceptance harness + P7 flags, spec §9/§10 instrumentation)*

Objective: every proposal §5 acceptance row gets a runnable entry point; the
optional P7 paths exist flag-gated, default-off.

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W9.1 | ORCH | `src/rag/evals.py` CLI (stdlib-only import; lazy torch/faiss): subcommands `roundtrip` (P1 items i–iv: Beta-concentration fit, round-trip MSE vs paper, per-head norm spread, outlier mass), `streaming` (P2 greedy token-match vs the fp16-cache baseline), `margin` (P4 same/different-topic cosine margin ≥ 3×), `recall` (P5 recall@100 + top-3), `e2e` (P6 OfficeQA: actual vs no-RAG vs oracle-install), `ledger` (§9 timing + §10 VRAM). Each emits a JSON artifact to `evals_out/` + stdout pass/fail lines (proposal P1 gate format) | W9.3 |
| W9.2 | SUB | hardening flags (P7, all default OFF): `--qjl` prod-variant A/B in `turboquant.py` (structured QJL residual sketch: sign of an FHT-based projection of the residual + one fp γ — D1's flagged A/B, never a rewrite); mmap snapshot-store option in `snapshot.py`; CUDA-graph capture hook points in `tq_cache.py` (GPU-guarded no-ops on CPU) | W9.3 |
| W9.3 | SUB | `tests/test_evals.py`: harness self-tests on synthetic inputs — every subcommand runs end-to-end on stub data and emits schema-valid JSON with a pass/fail line; flag-parity tests (all W9.2 flags off → outputs bit-identical to the unflagged path) | pytest |

**DoD (Wave 9):** all six subcommands self-test green on synthetic input;
parity tests green; every row of the §7 GPU register names a working entry
point; pushed.

### Wave 10 — Final audit + tag

| Task | Exec | Deliverable | Verification |
|---|---|---|---|
| W10.1 | ORCH | full-repo sweep: L1 over all tracked `.py`; full pytest; import-closure grep (no dangling refs to deleted modules; no `qlora`/`eval_common` imports anywhere under `src/`); §6 coverage matrix verified row-by-row against the tree; README updated (TASKS.md link + build status) | all green |
| W10.2 | ORCH | `git tag cpu-code-complete && git push origin main cpu-code-complete`; final worklog entry with the wave-by-wave gate results | tag visible on origin |

**DoD (Wave 10):** every §6 matrix row maps to an existing file + a green
test; the tag is pushed; the worklog is closed out with the complete gate
table.

---

## 5. Sub-agent brief template

Every SUB dispatch uses this skeleton (filled in by ORCH; self-contained):

```markdown
Task ID: W<n>.<m>   |   Sub-agent type: general-purpose   |   Repo: /home/z/my-project/repo

MISSION: <one sentence>

READ FIRST: /home/z/my-project/worklog.md — skim the entries for wave <n>.

CONTEXT KIT (all you need; do not read other files unless listed here):
- Contract: <inline API, shapes, constants, tolerances>
- Repo anchors: <exact file paths + symbol names to import/use>
- House rules: <stdlib-only import? sys.path via src/rag/_paths.py; docstrings cite spec §§; loud dtype gates>

DELIVERABLES: <files to write>

VERIFICATION (must be green BEFORE your commit):
  cd /home/z/my-project/repo && python3 -m pytest src/rag/tests/test_<x>.py -q

VCS: stage and commit ONLY your deliverables as `W<n>.<m>: <summary>`.
Do NOT push — the orchestrator pushes after the wave DoD gate.

WORKLOG: append your entry (format §1.3) to /home/z/my-project/worklog.md.

RETURN: files written, test result, commit hash, worklog appended.
```

---

## 6. Spec coverage matrix

| Spec § | Requirement | Wave(s) |
|---|---|---|
| §1 | model geometry, 9 hooks, 24 S | W4 (+ W3 for the layer it hooks) |
| §2.1 | S — access + composable deltas | W2 (interception), W4 (capture), W5 (delta protocol) |
| §2.2 | M1/M2 added global memories | W3 |
| §2.3 | conv_state | W2 |
| §2.4 | full-attn untouched | no code — guarded by W3/W2 parity tests |
| §3.1 | TurboQuant method (FHT + Lloyd-Max, 3.5-bit) | W1 |
| §3.2 | online: quantize-on-write / dequantize-on-read | W2 |
| §3.3 | compression 4.6×, ~6 MiB/chunk | W1 (size gate), W5 (npz on disk) |
| §4 | retrieval vector (13,631,488 dims, §4 order) | W5 |
| §5 | ingestion + per-chunk disk | W5 |
| §6 | query flow + install (dequant-sum-requant, conv=last) | W7 |
| §7 | fine-tune (straight-through LUTs, full-LUT delivery) | W8 |
| §8 | IVFADC (nlist=224, m=64, nbits=8, nprobe=8) + rerank | W6 |
| §9 | per-query timing ledger | W7 (instrumentation), W9 (`ledger`) |
| §10 | VRAM ledger | W9 (`ledger`) |
| §11 | disk layout (index, snapshots/, pretrained_luts/) | W6, W5, W8 |
| §12 | toy validation | already done — referenced as the ceiling by W7's mini-e2e |
| §13 | repo files used | existing files + W2/W3 surgical edits (no rewrites) |

---

## 7. GPU-box execution register (NOT tasks on this box)

| Proposal phase | GPU gate | CPU-wave code that enables it | `evals.py` entry point |
|---|---|---|---|
| P0 bring-up | build `flute_extended` per DEPLOY.md; kernel spot-check; baseline generations | unchanged loader/docs (repo as-is) | — |
| P1 (real tensors) | the 4-measurement table on real S/conv/M1/M2 → pins norm granularity, 3.5-bit split, variant | W1, W4 | `roundtrip` |
| P2 compounding | ≥95% greedy token-match vs fp16 cache on ≤4k prompts | W2 | `streaming` |
| P4 fine-tune | loss sane; cache-signal margin ≥ 3×; ≤1% token-match regression | W8 | `margin` |
| P5 retrieval | recall@100 ≥ 95%, top-3 correct, prod-variant A/B decision | W5, W6 | `recall` |
| P6 end-to-end | OfficeQA accuracy (actual vs no-RAG vs oracle); §9/§10 ledgers | W7 | `e2e`, `ledger` |
| P7 hardening | CUDA-graph decode path, mmap store, QJL if P5 chose it | W9.2 flags | — |

The CPU box's obligation for every row: the code exists, compiles, passes
its CPU-side tests, and the entry point runs. The GPU box executes the gate
and records the number.

---

## 8. Completion contract

ORCH returns to the operator only when ALL of the following hold:

1. Waves 0–10 executed in order; every task committed (`W<n>.<m>`) and every
   wave pushed after its DoD gate ran green in a fresh shell.
2. The full ladder green: L1 (`py_compile` over all tracked `.py`), L2
   (import closure), L3–L5 (`python3 -m pytest src/rag/tests -q`).
3. §6 coverage matrix verified row-by-row against the tree (W10.1).
4. §7 register fully populated with working entry points (W9).
5. Tag `cpu-code-complete` pushed to origin/main.
6. The worklog closed with the wave-by-wave gate table (W10.2).

Blocked case: a DoD line that requires GPU execution, model artifacts, or
the corpus is reported as blocked with the exact command, its output, and
the missing input named — not silently skipped.

