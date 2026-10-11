# TASKS.md — W20 Execution Ledger: Waves, Tasks, DoD (the machine contract)

Purpose: the execution ledger for **W20 — the TQ kernel family**
(PROPOSAL.md rev 2): 13 waves, 58 tasks, each task one sub-agent, one
commit; each wave one DoD audit, one push, one tag. The orchestrator
iterates this file top-to-bottom under PROMPT.md until §6 closes.
Authority: SPECIFICATION.md > PROPOSAL.md > PROMPT.md > **TASKS.md** >
HANDOVER.md. Line references are **w20-baseline coordinates** (the tree
at tag `w20-baseline`); before editing, re-locate by symbol (`rg`) —
earlier waves move lines.
Status: IN EXECUTION — 4/13 waves closed (W20.0, W20.1, W20.2, W20.3).

---

## 0. The wave ledger (the orchestrator's status table)

| wave | phase (PROPOSAL §8) | title | tasks | gates authored | status |
|---|---|---|---|---|---|
| W20.0 | pre-K | environment + audit infrastructure | 5 | — | CLOSED |
| W20.1 | K1 | baseline tag + frozen probe fixtures | 4 | (capture) | CLOSED |
| W20.2 | K1 | TQCodes device contract + codebook mirror | 3 | — | CLOSED |
| W20.3 | K1 | turboquant.py torch rewrite (the core) | 6 | — | CLOSED |
| W20.4 | K1 | wiring + serving sweep + K1 gates | 7 | G20.P1 | OPEN |
| W20.5 | K2 | fht_forward_large / fht_backward_large | 6 | — | OPEN |
| W20.6 | K2 | debug twins + canary/trap builds | 4 | G20.P2, G20.6 | OPEN |
| W20.7 | K3 | tq_quant_batched (the fused encode) | 5 | — | OPEN |
| W20.8 | K3 | tq_dequant_batched + dispatcher | 4 | G20.P3, G20.1, G20.2, G20.4 | OPEN |
| W20.9 | K4 | QJL kernel chain + tq_l2norm_rows | 4 | — | OPEN |
| W20.10 | K4 | serving-side batched passes | 4 | G20.5 | OPEN |
| W20.11 | K5 | trainer levers (refresh 8, loader pinning) | 3 | G20.2 full, G20.3 | OPEN |
| W20.12 | closure | handover + final all-DoD audit | 3 | — | OPEN |

Status values: OPEN → CLOSED (wave DoD Tier-D green, pushed, tagged) →
BLOCKED (§7.2 PROMPT.md; a disposition note is mandatory).

## 1. Conventions

- **Task ID** `W20.<wave>.<task>`; a context-budget split appends
  `.a/.b` (PROMPT.md §2). Tasks are dispatched in ID order.
- **Checkbox protocol**: `- [ ]` open · `- [x]` done (DO-D verified +
  committed; flipped by the ORCHESTRATOR only, riding the task's commit)
  · `- [~]` BLOCKED (note required).
- **Brief fields** per PROMPT.md §2.1: ROLE / READ (file:line, ≤6
  files, ≤1500 lines) / WRITE (≤2 files) / BRIEF / INVARIANTS / DO-D /
  COMMIT. Sub-agents never run git; the orchestrator commits.
- **DO-D** = deterministic checks (this box; `rg` / `ast` /
  `py_compile` / git facts, via `scripts/w20_static_audit.py`). **Tier
  G** = A10G gate scripts — authored here, executed on the box; PENDING
  until a box report lands (PROMPT.md §5).
- **THE LAW applies to every task**: no test runs, no CPU test
  authoring, no repo-module imports on this box (PROMPT.md §1). The
  auditor parses; it never imports.
- **The two foreign edges** (numpy allowed there and ONLY there, per
  PROPOSAL rev 2): the npz disk edge (`to_arrays/from_arrays`,
  `snapshot.py` save/load) and the faiss boundary (the final
  `.numpy()` at index add/search). The codebook SOLVE
  (`codebooks.py`, offline, cached to disk) is the third documented
  numpy island — offline authoring, never runtime.

## 2. Phase map (PROPOSAL §8 → waves)

| phase | waves | lands what |
|---|---|---|
| K1 | W20.1–W20.4 | frozen fixtures + the torch-CUDA single path (numpy deleted from runtime) + the K1 differential reference |
| K2 | W20.5–W20.6 | `fht_forward_large`/`fht_backward_large` + twins + canary/trap |
| K3 | W20.7–W20.8 | `tq_quant_batched` + `tq_dequant_batched` + the kernel-first dispatcher — TQ leaves the profile |
| K4 | W20.9–W20.10 | QJL chain + `tq_l2norm_rows` + the serving-side batched passes |
| K5 | W20.11 | `--refresh-every 8` regime + loader pinning |
| closure | W20.12 | HANDOVER + final audit + tag `w20-complete` |

---

## WAVE W20.0 — environment + audit infrastructure (pre-K)

Goal: provision both boxes' toolchains at latest stable (evidenced,
recorded) and stand up the audit + test-guard infrastructure every later
wave depends on. No production code is touched.

#### W20.0.1 — CPU-box toolchain audit + latest-stable provisioning
- [x] done
ROLE task-writer · READ `PROMPT.md §6`; `PROPOSAL.md §8` · WRITE
`env/W20_ENV.md` (new), `.gitignore` (only if env artifacts must be
excluded)
BRIEF: probe and record: python3 (3.12.15), pip, git, torch
(2.14.1+cpu → upgrade to the latest stable default PyPI linux wheel,
the CUDA-bundled build — importable without a device), numpy,
transformers, faiss-cpu, ninja (install if absent), nvcc (install only
if obtainable driver-less via the system package manager; else record
ABSENT — legal). For every package: capture `pip index versions <pkg> |
head -2` as latest-stable EVIDENCE, install `-U`, record the installed
version + date + source. Sections: `## CPU box (dev)` per-package
table; `## evidence` (raw index outputs); `## nvcc` (present/ABSENT +
consequence: compile checks local vs deferred).
INVARIANTS: no repo module is imported; no test is run; every version
claim carries its evidence line.
DO-D: `env/W20_ENV.md` exists; contains `pip index versions torch`
evidence; contains `ninja`; contains an nvcc verdict line; auditor
check-set W20.0.1 mirrors these greps.
COMMIT: "W20.0.1 — CPU-box provisioning at latest stable, evidenced
(env/W20_ENV.md)"

#### W20.0.2 — the A10G provisioning script (idempotent, latest-stable)
- [x] done
ROLE task-writer · READ `PROMPT.md §6`; `src/flute_extended/setup.py`
(header comment: FLUTE_CUDA_ARCHES, MAX_JOBS, ninja parallelism) · WRITE
`scripts/gpu/w20_provision.sh` (new)
BRIEF: a bash script the A10G operator runs first: (1) assert
`nvidia-smi` and an Ampere device — fail loudly if capability major !=
8, warn if != 8.6; (2) record driver/CUDA driver versions; (3) pip
install -U (UNPINNED — latest stable at run time): torch, ninja,
transformers, faiss-cpu, numpy; (4) install the CUDA toolkit (nvcc)
matching torch's CUDA build if nvcc is absent or older; (5) build
`flute_extended`: `MAX_JOBS=<cpu-count, cap 16>
FLUTE_CUDA_ARCHES=86 python3 setup.py build_ext --inplace`; (6) run
`python3 -m pytest src/rag/tests -q` (the 237 baseline — ON THE BOX);
(7) write `env/w20_gpu_env.lock` (torch/nvcc/ninja/transformers/faiss
versions + device name + capability + build hash). Idempotent; every
step echoes what it does; no version pins anywhere.
INVARIANTS: unpinned installs (the latest-stable law); never runs on
this box (it is GPU-box-only by content — CUDA asserts).
DO-D: script exists, executable bit set; `rg -n "nvidia-smi|FLUTE_CUDA_ARCHES=86|pytest src/rag/tests|w20_gpu_env.lock" scripts/gpu/w20_provision.sh` all hit; contains no `==` version pins of torch/nvcc.
COMMIT: "W20.0.2 — A10G provisioning script (latest-stable, idempotent,
sm_86-asserting)"

#### W20.0.3 — the static auditor (the DoD vehicle — parses, never imports)
- [x] done
ROLE task-writer · READ `PROMPT.md §1.2, §5` · WRITE
`scripts/w20_static_audit.py` (new)
BRIEF: CLI: `--wave W20.<n> | --all | --list`. Check kinds: `exists`
(path), `grep` (pattern present), `absent` (pattern absent in path),
`ast_def` (module + symbol + expected arg names), `py_compile` (byte-
compile, never import), `git_tag` (tag exists), `git_log` (commit
subject substring exists). Check sets are declared per wave in a
`CHECKS` dict **mirroring TASKS.md DoD items 1:1** (drift = wave-close
failure). Output: one line per check `W20.<n> <kind> <target> …
PASS/FAIL`, exit 0 only if all pass. Module docstring states THE LAW:
"this tool NEVER imports or executes any target module (PROMPT.md
§1.2); it parses and greps only". Seed CHECKS with W20.0's own items
(fixtures of later waves are appended by each wave's close-out task).
INVARIANTS: no `import` of anything under `src/`; only stdlib (ast,
argparse, pathlib, re, subprocess for rg/py_compile/git calls).
DO-D: file exists; `python3 -m py_compile scripts/w20_static_audit.py`
passes; `--list` lists waves; docstring contains "NEVER imports".
COMMIT: "W20.0.3 — the static auditor (rg/ast/py_compile/git check
kinds; parses, never imports)"

#### W20.0.4 — the GPU test guard module (CUDA-only test authoring surface)
- [x] done
ROLE task-writer · READ `pytest.ini`; `src/rag/tests/conftest.py`;
`PROMPT.md §1.3` · WRITE `src/rag/tests/w20/__init__.py` (new, empty),
`src/rag/tests/w20/_gpu_guard.py` (new)
BRIEF: the guard every W20 test file imports at collection time:
`require_cuda()` (module-level: if `torch.cuda.is_available()` is False
→ `pytest.skip("W20 GPU-ONLY: A10G required", allow_module_level=True)`);
`device_report()` (name, capability, torch/cuda versions — for gate
reports; warns if capability != (8, 6), raises if major != 8);
`gate_report_path(gate_id)` helper pointing at
`env/w20_gate_reports/`. NO test functions in this file — it is a
helper, not a test.
INVARIANTS: pure torch/pytest imports; no repo-module imports; no
assertions that execute on CPU beyond the availability probe itself.
DO-D: both files exist; `_gpu_guard.py` contains
`torch.cuda.is_available`, `allow_module_level`, `8, 6`; defines no
`test_` functions (`rg "def test_" src/rag/tests/w20/_gpu_guard.py` ==
0 hits).
COMMIT: "W20.0.4 — the W20 GPU guard (loud CUDA skip + Ampere device
report)"

#### W20.0.5 — wave close W20.0
- [x] done
ROLE orchestrator · READ TASKS.md W20.0 DoD · WRITE
`scripts/w20_static_audit.py` (append CHECKS W20.0), `HANDOVER.md`
(append a "W20 in execution" stub: pointer to PROMPT.md/TASKS.md +
this wave's lines), TASKS.md §0/§5 flips
BRIEF: the standard close-out per PROMPT.md §5 (all seven steps).
Tier-D DoD for the wave = the union of W20.0.1–0.4 DO-D sets + boxes
flipped + commits present.
Tier-G: none.
DO-D: auditor `--wave W20.0` exits 0; `git tag w20-wave-0` exists;
push succeeds; attestation line written.
COMMIT: "W20.0.5 — wave W20.0 closed: infra green, pushed, tagged"

---

## WAVE W20.1 — baseline tag + frozen probe fixtures (K1 start)

Goal: pin the numpy-era semantics BEFORE the rewrite touches anything:
tag the pre-rewrite tree and author the GPU-box capture script + the
box protocol doc. Nothing here executes on this box (the capture is a
GPU-box act — THE LAW).

#### W20.1.1 — tag the pre-rewrite tree
- [x] done
Ledger: `w20-baseline` → e9ea357 (2026-10-11) — the numpy-era runtime
tree (W20.0 landed infra only: env/, scripts/, the test guard; the
runtime modules are untouched since 9af04fe; the doc commits above it
are irrelevant to the capture, as specified).
ROLE orchestrator · READ `git log --oneline -5` · WRITE TASKS.md §3
(ledger note: tag → SHA)
BRIEF: create annotated tag `w20-baseline` at current HEAD (the tree
whose runtime code is still the numpy-era path — W20.3 has not landed;
the doc commits above it are irrelevant to the capture). Record the SHA
in this task's ledger note. The tag is the fixture-capture checkout.
DO-D: `git tag -l w20-baseline` non-empty; ledger note present; tag
pushed with the wave push.
COMMIT: "W20.1.1 — tag w20-baseline (the numpy-era tree, fixture
capture point)"

#### W20.1.2 — the fixture capture script (GPU-box; refuses CPU loudly)
- [x] done
Deviation note: `kind prod` does not exist in the pinned source — it
IS the `qjl=True` flag (paper Alg. 2); split exists only for
non-integer bits (`split = group>1 and not bits.is_integer()`), so the
"plain split" probes are realized as the 3.5-bit lo/hi split (flat at
G1/G3, the W15 outlier-split at G2 — the production conv geometry). 14
probes, seeds 20260001+i.
ROLE task-writer · READ `src/rag/turboquant.py:460-560, :640-710`;
`src/rag/codebooks.py:190-260`; `PROPOSAL.md §5` · WRITE
`scripts/gpu/w20_capture_fixtures.py` (new)
BRIEF: runs ONLY on the A10G box checked out at `w20-baseline` (verify
`git rev-parse HEAD` matches the recorded tag SHA at start; refuse
otherwise). Imports the repo's numpy-era TurboQuant (legal THERE — it
is the box, and the legacy path is the thing being pinned). Probes
(synthetic, seeded, data-oblivious quantizer ⇒ synthetic probes are
valid): G1 d=2^19 ×4 units; G2 d=24,576 ×4 (the conv full-window); G3
d=2^22 ×2; covering kinds mse/prod, plain b∈{3,4}, the 3.5-bit lo/hi
split, qjl on/off; per probe record: input (fp32), codes (packed
bytes), norm (fp32), reconstruction (fp32). Output
`src/rag/tests/w20/fixtures/*.npz` + `fixtures_manifest.json` (per-file
sha256, probe seeds, turboquant config, the codebook digests). `--out`
arg; hard refusal if `torch.cuda.is_available()` is False.
INVARIANTS: refuses CPU; refuses wrong-checkout; deterministic seeds
(torch.Generator, seeds recorded in the manifest).
DO-D: script exists; contains `w20-baseline`, `sha256`,
`torch.cuda.is_available`, `fixtures_manifest.json`; no test functions.
COMMIT: "W20.1.2 — the frozen-probe capture script (GPU-box, pinned to
w20-baseline)"

#### W20.1.3 — the A10G execution protocol doc
- [x] done
ROLE task-writer · READ `PROMPT.md §0, §5`; `HANDOVER.md` (the W19
confirmation-sequence style) · WRITE `env/W20_GPU_PROTOCOL.md` (new)
BRIEF: the ordered, copy-pasteable A10G sequence: (1) run
`scripts/gpu/w20_provision.sh`; (2) `git fetch && git checkout
w20-baseline` (detached) → run `scripts/gpu/w20_capture_fixtures.py`;
(3) `git checkout main && git add src/rag/tests/w20/fixtures …` →
commit `W20.G: fixtures captured from w20-baseline (GPU box)` → push;
(4) as later waves land: run `scripts/gpu/w20_gates.py --gates
G20.P1,…` per the §4 registry and record statuses; (5) the final full
registry run. Notes: fixtures are committed FROM the box; gate reports
land in `env/w20_gate_reports/` (gitignored); statuses are transcribed
into TASKS.md §4.
DO-D: doc exists; contains all five steps in order + `w20-baseline` +
`w20_gates.py`.
COMMIT: "W20.1.3 — the A10G protocol doc (provision → capture → gates)"

#### W20.1.4 — wave close W20.1
- [x] done
ROLE orchestrator · standard close-out (PROMPT.md §5).
Tier-D: tasks 1–3 DO-D sets; tag pushed; ledger updated.
Tier-G: the capture (execution belongs to the box; AUTHORED here).
DO-D: auditor `--wave W20.1` exits 0; tag `w20-wave-1`; push;
attestation.
COMMIT: "W20.1.4 — wave W20.1 closed: baseline pinned, capture
authored"

---

## WAVE W20.2 — TQCodes device contract + codebook mirror (K1)

Goal: the rev-1 codes contract — torch-native codes container (device
tensors, npz ONLY at the disk edge) + codebooks mirrored to device
tensors. `turboquant.py`'s TQCodes (:204) moves to its own module;
callers keep importing the symbol from `turboquant` (re-export).

#### W20.2.1 — TQCodes torch-native (`src/rag/tq_codes.py`)
- [x] done
Ledger amendment (drift fix, additive — the criterion is unchanged):
the DO-D import-pattern `from rag.tq_codes import|from .tq_codes import`
was authored against the wrong import convention — the house style is
FLAT imports anchored by `import _paths` (src/rag on sys.path; cf.
`import codebooks` / `from turboquant import TQCodes` at install.py:47).
The realized, checked form is `from tq_codes import TQCodes`
(turboquant.py:118). The pattern below is corrected to accept the house
form; nothing was deleted from the criterion.
ROLE task-writer · READ `src/rag/turboquant.py:159-260` (TQCodes,
pack_bits, unpack_bits, _packed_len context), `:460-520` (construction
sites) · WRITE `src/rag/tq_codes.py` (new), `src/rag/turboquant.py`
(re-export stub)
BRIEF: the container, torch-native by construction: fields kind, d,
bits_lo/hi, n_lo/hi, seed; `norm` = **0-dim fp32 torch tensor** (never
a python float, never numpy — the `.item()` era ends); `idx_lo/idx_hi`
= uint8 torch tensors (device-resident); methods `to(device)`,
`device` property, `nbytes()`, `__repr__`, `to_arrays()` (THE npz disk
edge — the only numpy in the module), classmethod `from_arrays()`
(the inverse edge). Module docstring: the rev-1 contract (codes are
torch tensors on the quantizing device; numpy exists ONLY at
to_arrays/from_arrays). `turboquant.py` imports TQCodes from here
(symbol-compatible for callers).
INVARIANTS: `rg "\.item\(\)" src/rag/tq_codes.py` == 0; numpy appears
ONLY inside to_arrays/from_arrays (`rg -n "np\.|numpy"` confined to
those two bodies); constructor asserts torch tensors + dtypes.
DO-D: module exists; ast_def TQCodes with expected fields;
py_compile both files; the greps above hold; `rg "from tq_codes
import TQCodes|from rag.tq_codes import|from .tq_codes import"
src/rag/turboquant.py` hits (house flat form accepted — see the ledger
amendment above).
COMMIT: "W20.2.1 — TQCodes torch-native (device tensors; numpy only at
the npz edge)"

#### W20.2.2 — codebooks mirrored to device tensors
- [x] done
Mirror surface landed (W20.3 consumes): `DeviceCodebook` view class —
`get_codebook(bits, d, cache_dir=None, device=None)`; device=None →
legacy numpy Codebook byte-identical; device set → `.boundaries`/
`.centroids` float32 torch tensors + `.numpy_codebook` (the island
original). `device` is keyword-appended AFTER cache_dir (positional-3rd
callers unaffected; W20.3 passes device= by keyword).
ROLE task-writer · READ `src/rag/codebooks.py:1-260` (the Codebook
class :200, the offline solve, the disk cache) · WRITE
`src/rag/codebooks.py`
BRIEF: keep the offline Lloyd-Max solve + its numpy quadrature EXACTLY
as is (documented numpy island: offline authoring, cached to disk, per
PROPOSAL rev 2 — add the island docstring marker). Add the runtime
mirror: `get_codebook(bits, d, device=…)` returns a Codebook whose
`boundaries`/`centroids` are **torch tensors on the requested device**
(cached per (bits, d, device) in a small registry; built once from the
solved tables, `torch.as_tensor(..., device=...)`). Existing numpy
fields stay available under clearly-named legacy accessors for the
snapshot/tests edge ONLY if some caller still needs them — enumerate
callers with `rg "boundaries|centroids" src/rag/` first.
INVARIANTS: the solve path is byte-identical (no numerical change —
`git diff` confined to new mirror code + docstring); runtime consumers
get tensors.
DO-D: `rg "device" src/rag/codebooks.py` hits the registry;
py_compile; the island docstring marker present
(`OFFLINE-NUMPY-ISLAND`); diff shows no change inside the solve
functions.
COMMIT: "W20.2.2 — codebook device mirror (solve stays the offline
numpy island)"

#### W20.2.3 — wave close W20.2
- [x] done
ROLE orchestrator · standard close-out.
Tier-D: 2.1/2.2 DO-D sets; py_compile; auditor green.
Tier-G: none (parity is W20.4's G20.P1).
DO-D: auditor `--wave W20.2` exits 0; tag `w20-wave-2`; push;
attestation.
COMMIT: "W20.2.3 — wave W20.2 closed: the codes contract is
torch-native"

---

## WAVE W20.3 — turboquant.py torch rewrite (K1 core; the six CPU syndromes)

Goal: the whole encode/decode path torch-native and device-parametric
(the differential twin the kernels need; the immediate-relief rung).
PROPOSAL §1's table row-by-row. THE LAW: no behavioral verification on
this box — py_compile + greps only; behavior is G20.P1 on the box.

#### W20.3.1 — quant encode, torch-native (syndromes: `.item()`, `y.cpu().numpy()`, numpy searchsorted)
- [x] done
Note: the ties mapping realized as `np side='left' ≡ torch
right=False` (matches the numpy default the pinned source uses;
TASKS' prose said "side='right' equivalence" — the ledger intent is
"match the numpy ties semantics EXACTLY", which this does; see the
comment at the searchsorted calls). The W15 outlier mask packs
big-endian (np.packbits parity) — deliberately distinct from
`_pack_bits_t`'s little-endian idx stream.
ROLE task-writer · READ `src/rag/turboquant.py:460-560` (quant +
_quant_split), `src/rag/codebooks.py` (mirror API from 2.2) · WRITE
`src/rag/turboquant.py`
BRIEF: rewrite `quant`/`_quant_split`'s encode: norm =
`x32.norm().reshape(())` (a 0-dim tensor — NO `.item()`, NO
`float()`); rotation `fht.fht_apply` (backend auto — device-native);
bucketize with `torch.searchsorted(boundaries_t, y)` (boundaries now
device tensors from 2.2) or an explicit compare ladder — MATCH the
numpy `np.searchsorted` ties semantics (the W16-verified CPU/GPU
parity, side='right' equivalence — document the mapping in a comment);
apply the **boundary-scaling identity** (PROPOSAL §3.3: compare the
UNSCALED rotated coord against norm-scaled boundaries — identical
decisions, monotone exact; the normalized vector is never
materialized); degenerate zero-norm branch builds torch zeros codes.
Emit TQCodes (torch, per 2.1). Cover the split (lo/hi halves) variant.
INVARIANTS: `rg "\.item\(\)|\.cpu\(\)|\.numpy\(\)|np\.searchsorted|np\.packbits" src/rag/turboquant.py` == 0 after 3.1+3.2 land (this task: the encode half); quant's public signature unchanged.
DO-D: greps clean for the encode region; ast_def quant present;
py_compile; the identity documented (`rg "boundary-scaling" …` hits).
COMMIT: "W20.3.1 — quant encode torch-native (norm tensor, searchsorted
parity, boundary-scaling identity)"

#### W20.3.2 — pack/unpack in torch (syndrome: numpy packbits)
- [x] done
Notes: the numpy pair is DELETED (module + __all__); the mask reader
in `_split_channels` (:632) got a mechanical MSB-first torch expansion
(big-endian fixture — deliberately NOT the little-endian _t stream);
the pack domain is asserted to bits∈{3,4} per this brief — W20.4 must
reconcile the blast radius (pre-W20 uniform-bits tests at b∈{1,2};
evals.py bits_list). test_turboquant.py: mechanical rename only
(imports + call sites; the numpy-constructor/dtype adaptation is
W20.4's sweep).
ROLE task-writer · READ `src/rag/turboquant.py:159-203` (pack_bits/
unpack_bits — the byte layout to reproduce), `:800-833` (_packed_len) ·
WRITE `src/rag/turboquant.py`
BRIEF: torch-native `pack_bits_t(idx, bits) -> uint8 tensor` and
`unpack_bits_t(packed, bits, n) -> uint8 tensor` via shifts/masks/
`torch.bitwise_*` (b∈{3,4}; the lo/hi split packs halves separately).
**Byte layout IDENTICAL to the numpy `pack_bits`** (fixture-pinned —
the docstring cites G20.P1 as the pinning gate; layout described
bit-precisely: bit order, byte tail padding). Keep the numpy originals
ONLY if a non-runtime consumer still imports them — enumerate with
`rg "pack_bits|unpack_bits" src/ scripts/` and delete if runtime-only.
INVARIANTS: layout documented bit-precisely; no numpy in the new
functions; indices stay uint8.
DO-D: ast_def both; `rg "np\.packbits|np\.unpackbits" src/rag/turboquant.py` == 0; py_compile; layout docstring present.
COMMIT: "W20.3.2 — torch bit pack/unpack (byte layout fixture-pinned)"

#### W20.3.3 — dequant torch-native (syndromes: numpy gather, from_numpy adjoint, `.to(device)` return)
- [x] done
Notes: the DO-D `from_numpy == 0` is scoped to the decode region for
this task (module-wide zero is 3.4's audit line — the 3 residual hits
are `_qjl_compensation_term` (:847/:850, 3.4's) + the dead `_pad_to`
(:966, deleted at 3.4/3.5). The degenerate zero-norm decodes via the
0-dim multiply (value-exact zero, no host-sync branch). `_qjl_term_`
bridged` (:798) is the sanctioned pre-3.4 bridge — 3.4 collapses it to
the direct call.
ROLE task-writer · READ `src/rag/turboquant.py:640-780` (dequant,
_dequant_mse, _dequant_split_mse, the :770-776 gather+adjoint) · WRITE
`src/rag/turboquant.py`
BRIEF: rewrite the decode: unpack (3.2's torch version); gather =
`centroids_t[idx]` (torch fancy-index, device-resident — the LUT is
already a tensor from 2.2); adjoint rotation `fht.fht_adjoint` on
device; multiply by the **0-dim norm tensor** (no float() — device
residency end-to-end); dtype conversion last; the `device` routing
(`_route_for` :652) preserved. `rg "torch\.from_numpy" src/rag/turboquant.py` → 0. Split + qjl-compensation branches follow in 3.4.
INVARIANTS: public `dequant` signature unchanged; codes in → tensor
out on the codes' device (unless caller overrides); no H2D/D2H sync
points anywhere in the path.
DO-D: greps (from_numpy == 0; `.item()` == 0 module-wide after 3.3+3.4);
ast_def dequant; py_compile.
COMMIT: "W20.3.3 — dequant torch-native (tensor LUT gather, adjoint,
norm tensor)"

#### W20.3.4 — the QJL branch torch-native
- [x] done
The audit line HOLDS: `rg "np\.|numpy" src/rag/turboquant.py` == 0
module-wide (the K1 core's final state — 28 hits swept to zero, prose
included). The qjl γ!=0 host branch is gone (branchless 0-dim
multiply, the 3.3 discipline); the term's scalar prefactor is now
fp32-torch vs legacy host-float64 — last-ulp drift possible on the
qjl=True path only (G20.P1's parity business).
ROLE task-writer · READ `src/rag/turboquant.py:510-560` (the qjl
encode branch), `:695-712` (_qjl_compensation_term) · WRITE
`src/rag/turboquant.py`
BRIEF: the Alg.-2 extras torch-native: residual = x − dequant (torch);
γ = the same 0-dim norm tensor discipline; the sign sketch
`torch.sign` (note: `sign(FHT(r/γ)) ≡ sign(FHT(r))` — the scale
invariance, comment it); the compensation rides `fht_adjoint`. No
numpy anywhere in the branch.
INVARIANTS: the `--qjl` A/B semantics unchanged (flag-off default);
all torch.
DO-D: `rg -n "np\.|numpy" src/rag/turboquant.py` == 0 module-wide
(THIS is the audit line for the whole file); py_compile.
COMMIT: "W20.3.4 — the QJL chain torch-native (numpy-free module)"

#### W20.3.5 — API surface glue + module contract docstring
- [x] done
ROLE task-writer · READ `src/rag/turboquant.py` (full skim ≤400 lines —
the last full read of this file in W20), `src/rag/tq_cache.py:399-430`
(the consumer interface) · WRITE `src/rag/turboquant.py`
BRIEF: reconcile the module: every construction site emits TQCodes
(torch); `_route_for` intact; the module docstring gains the rev-2
runtime contract block (kernel-first dispatch arrives W20.8 —
foreshadow it; torch path = the differential twin + the CPU-tensor
path; numpy exists ONLY at to_arrays/from_arrays). Callers still
compiling (py_compile) is the only box-side guarantee.
INVARIANTS: no consumer-visible signature change (tq_cache/install/
index/query/m1m2 keep importing the same symbols — they are swept in
W20.4).
DO-D: py_compile; `rg "class TurboQuant" src/rag/turboquant.py` hits;
docstring contains "differential twin".
COMMIT: "W20.3.5 — turboquant module contract (torch-native, twin
foreshadow)"

#### W20.3.6 — wave close W20.3
- [x] done
ROLE orchestrator · standard close-out.
Tier-D: the numpy-zero audit for turboquant.py (`rg -c "np\.|numpy"
src/rag/turboquant.py` == 0), the `.item()`/`.cpu()`/`from_numpy`
zero-greps, ast_defs, py_compiles, commits present.
Tier-G: none yet (G20.P1 lands W20.4).
DO-D: auditor `--wave W20.3` exits 0; tag `w20-wave-3`; push;
attestation.
COMMIT: "W20.3.6 — wave W20.3 closed: the core is numpy-free"

---

## WAVE W20.4 — wiring + serving sweep + the K1 gates (K1 end)

Goal: every consumer of the old numpy surface converted (tq_cache,
install, index/query/ingest/snapshot, evals, m1m2) — the 103-touchpoint
elimination — plus the CUDA-gated K1 test file + the gate harness with
G20.P1. The two foreign edges survive BY DESIGN (npz, faiss).

#### W20.4.1 — tq_cache.py conversion
- [x] done
Finding: the module was already the torch surface of the old core
(W15–W18 authored it that way; W18 moved dequant's device param to the
source) — W20.2/3's torch-native TQCodes/quant/dequant landed BENEATH
it, completing the device-native round trip with zero code motion.
The landing = a W20.4.1 attestation paragraph in the docstring (+14
lines, zero code changes); the zero-audit and the hook signatures were
verified, not assumed. `update_recurrent_state` is the layer-level
hook (transformers Cache dispatch), not a TQCache method — true of all
prior waves.
ROLE task-writer · READ `src/rag/tq_cache.py:399-727` (the TQCache
read/write interception region), `src/rag/turboquant.py` (the new
TQCodes contract) · WRITE `src/rag/tq_cache.py`
BRIEF: the cache interception (dequantize-on-read / quantize-on-write,
SPECIFICATION N15/N16) goes torch-native: reads consume torch codes,
writes emit them; the DynamicCache subclass contract (update_recurrent_
state, read_m1/read_m2, update_m1/update_m2 hooks used by
`modeling.py:719-754`) unchanged; zero numpy; zero `.item()`. The
120–192 round trips per prefill now ride device tensors end-to-end.
INVARIANTS: no hook signature changes; `rg -c "np\.|numpy|\.item\(\)"
src/rag/tq_cache.py` == 0.
DO-D: greps; ast_def TQCache intact; py_compile.
COMMIT: "W20.4.1 — TQCache torch-native (the round-trip wiring)"

#### W20.4.2 — install.py requant path
- [x] done
Finding: the file was already numpy-free by construction (W16 authored
it against the quant/dequant surface; W20.2/3 moved the substrate to
torch beneath). The landing: explicit W16-verbatim/CODE-COPY
docstrings + two new co-residency guards (loud ValueError on
cross-device folds — inputs that would otherwise crash late inside
torch's add). install.py opens NO npz — snapshot.py's load_chunk→
from_arrays is the disk edge.
ROLE task-writer · READ `src/rag/install.py:1-260` (sum_turboquant_
codes :53, install_snapshot :150, install_from_disk :245), the W16
verbatim-install notes in `src/rag/tests/test_w16_install.py` (READ
≤80 lines: the pinned contract) · WRITE `src/rag/install.py`
BRIEF: `sum_turboquant_codes` = dequant + sum + requant (N23) on the
torch path — device tensors throughout; the W16 verbatim install
(code-copy, never recompute — bit-exact within a backend) preserved
exactly; npz loads go through `from_arrays` (the disk edge).
INVARIANTS: `rg -c "np\.|numpy" src/rag/install.py` == 0 (loads ride
snapshot.py's edge); the W16 contract functions keep signatures.
DO-D: greps; ast_defs (3 functions); py_compile.
COMMIT: "W20.4.2 — install requant torch-native (W16 verbatim
preserved)"

#### W20.4.3 — index/query/ingest/snapshot sweep (the faiss edge stays)
- [x] done
FAISS-EDGE inventory: 9 markers, 4 actual host hops (index.py :869
add, :949 train, :1087 preselect outbound, :1089 inbound labels).
snapshot.py: numpy confined to the DISK-EDGE save/load codec regions
(digests ride to_arrays — key-for-key identical output, W16 sha256s
byte-preserved; frame npz side-files preserved). Open item for the
box: pre-W20 retrieval tests' strict array-type assertions get
mechanical adaptation at G20.P1 time.
ROLE task-writer · READ `src/rag/index.py:360-520` (the :372/473/510
vector builds), `src/rag/query.py` (the §6 flow), `src/rag/ingest.py`,
`src/rag/snapshot.py:320-460` (save_chunk npz) · WRITE those four
files as needed
BRIEF: the retrieval-frame vector builds (dequant + concat + the
frame's subtract/normalize math) torch-native fp32; the final faiss
hand-off keeps exactly ONE `.numpy()` per build site (the foreign
boundary — allowlist it in a comment); npz save/load rides
to_arrays/from_arrays. `preselect`/`rerank` shapes unchanged
(N30–N32).
INVARIANTS: `rg -n "np\.|numpy" src/rag/index.py` hits ONLY the
documented faiss-edge lines (each carries the `FAISS-EDGE` comment
marker); query/ingest numpy-free.
DO-D: the scoped greps; py_compile ×4; `rg -c "FAISS-EDGE"
src/rag/index.py` ≥ 3.
COMMIT: "W20.4.3 — the vector builds torch-native (faiss edge
allowlisted)"

#### W20.4.4 — evals.py + m1m2.py sweep (the ×5 + :345 sites)
- [x] done
bits_list disposition: FLAGGED, not constrained — no module constant
documents the bits domain (bare default `(1,2,3,3.5,4)`); the W20.3.2
conflict (pack_bits_t asserts bits∈{3,4}) is documented in
measure_roundtrip's docstring; the A10G gate (G20.P1) decides the
domain. Stat-definition notes: population variance (ddof=0), RAW 4th
moment kurtosis, quantile(0.5)=np.median semantics (torch.median drops
the upper middle on even counts — avoided).
ROLE task-writer · READ `src/rag/evals.py:80-145` (the five numpy stat
sites), `src/rag/m1m2.py:330-360` (:345) · WRITE the two files
BRIEF: eval stats in torch (mean/var/kurtosis moments on device;
`.numpy()`/floats only at the final print/report formatting — mark
`REPORT-EDGE`); the m1m2 :345 `.item()` min-position check becomes a
tensor-space assertion (e.g. `(positions >= 0).all()` — gather-free,
no sync).
INVARIANTS: no D2H-then-compute patterns; report formatting may print
floats (formatting, not math).
DO-D: `rg -n "\.item\(\)" src/rag/m1m2.py` == 0; evals compute sites
numpy-free (REPORT-EDGE markers only); py_compile ×2.
COMMIT: "W20.4.4 — evals/m1m2 sweep (stats on device, no sync points)"

#### W20.4.5 — the CPU-elimination audit report
- [x] done
Result at HEAD b9289f8: turboquant/tq_cache/install/query/ingest/
m1m2/evals = 0/0 (np/sync); the allowlist holds — DISK-EDGE 143
(tq_codes + snapshot + lut_export + m1m2_finetune), FAISS-EDGE exactly
4 (index.py, the ceiling), OFFLINE-ISLAND 93 (codebooks solve),
REPORT-EDGE 2 (comments only). FINDINGS: none unclassifiable. 112
audited touchpoints → 0 outside the allowlist.
ROLE task-writer (or orchestrator) · READ the auditor · WRITE
`env/w20_cpu_elimination_audit.md` (new; generated content + the
allowlist narrative)
BRIEF: run the scoped greps (the runtime modules: turboquant, tq_codes,
tq_cache, install, index, query, ingest, m1m2, evals) via
`scripts/w20_static_audit.py` (rg checks — NOT module imports);
produce the report: per-module numpy/sync hit counts (expected: 0,
except the allowlisted edges), the two foreign edges + the offline
island restated, the before/after count vs the 103-touchpoint baseline
(PROPOSAL §2). Committed as the wave's evidence artifact.
INVARIANTS: the report is rg-derived; no module executes.
DO-D: file exists; contains "103"; per-module counts present; the
auditor carries the corresponding grep checks.
COMMIT: "W20.4.5 — the CPU-elimination audit report (rg-derived)"

#### W20.4.6 — the CUDA-gated K1 test file + the gate harness (G20.P1)
SPLIT (context budget, PROMPT.md §2): W20.4.6 → W20.4.6.a (the gate
harness `scripts/gpu/w20_gates.py`) + W20.4.6.b (the CUDA-gated K1
test file + the mechanical pre-W20 test adaptation). Same DoD union,
two commits.

#### W20.4.6.a — the gate harness (`scripts/gpu/w20_gates.py`)
- [x] done
599 lines; `--report` takes a DIR (per-gate JSONs inside); extra
integrity prechecks (fixture sha256, input-rebuild bit-exactness,
pack-layout pin, config consistency) — strictly tighter than the §4
contract, each recorded in metrics. The guard's import-time
pytest-skip makes a direct `_gpu_guard` import unusable from a
standalone script — the assert is replicated inline (discipline kept,
cited by name).
ROLE task-writer · READ the guard, the fixtures manifest layout
(W20.1.2), TASKS.md §4 (the registry + thresholds) · WRITE
`scripts/gpu/w20_gates.py` (new)
BRIEF: `--gates <csv|all> --report env/w20_gate_reports/<id>.json`;
asserts CUDA (the guard discipline), records the device report; each
gate = a function returning {gate, pass, metrics, thresholds};
THRESHOLDS dict mirroring TASKS.md §4 1:1 (drift = wave-close
failure). G20.P1 IMPLEMENTED (the torch path vs the frozen fixtures —
codes equal except boundary ties ≤ 0.01%, norm ≤ 1 ulp, recon
allclose ≤ 1e-5 rel; run-to-run torch.equal on CUDA; the 237-suite
regression via pytest subprocess ON THE BOX). The other 8 gates
registered as NOT-AUTHORED stubs (their waves author them; invoking a
stub exits with a loud NOT-AUTHORED-YET error — the registry grows
per wave).
INVARIANTS: every test the harness runs is CUDA-gated; thresholds
mirror §4 exactly; no CPU fallback.
DO-D: file exists; `rg "G20.P1|THRESHOLDS|--gates" scripts/gpu/
w20_gates.py` hit; py_compile; the guard import present.
COMMIT: "W20.4.6.a — the G20 gate harness (THRESHOLDS 1:1, G20.P1
implemented)"

#### W20.4.6.b — the CUDA-gated K1 tests + mechanical pre-W20 adaptation
- [ ] done
ROLE task-writer · READ `_gpu_guard.py`, the fixtures layout,
PROPOSAL.md §5 (the three-level parity contract), test_turboquant.py +
test_tq_cache.py (the TQ-contract test surfaces) · WRITE
`src/rag/tests/w20/test_w20_torch_path.py` (new), the mechanical
adaptation of test_turboquant.py / test_tq_cache.py (imports/
constructors/dtypes ONLY — no new CPU assertions; no new `def test_`
in pre-W20 files)
BRIEF: the pytest surface — test_w20_torch_path.py is CUDA-gated
(module skip per the guard), fixture-driven, asserting G20.P1's
thresholds; the pre-W20 TQ-contract tests get the mechanical
adaptation the rewrite forces (torch constructors, renamed symbols,
dtype expectations).
INVARIANTS: every test in the W20 file imports `_gpu_guard`; no test
body can pass on CPU (module-level skip); thresholds mirror §4
exactly; pre-W20 diffs mechanical (auditor: no new `def test_`).
DO-D: both files exist; `rg "def test_" src/rag/tests/w20/test_w20_
torch_path.py` > 0; every such file imports `_gpu_guard`; `git diff
w20-wave-3..HEAD -- src/rag/tests/test_turboquant.py src/rag/tests/
test_tq_cache.py` contains no new `def test_`; py_compile all touched.
COMMIT: "W20.4.6.b — G20.P1's pytest surface (CUDA-gated torch-path
tests + mechanical pre-W20 adaptation)"

#### W20.4.7 — wave close W20.4 (K1 complete)
- [ ] done
ROLE orchestrator · standard close-out + HANDOVER's K1 section (the
twin is live; the box runs G20.P1 per env/W20_GPU_PROTOCOL.md).
Tier-D: 4.1–4.6 sets + the audit report + mechanical-only test diffs.
Tier-G: G20.P1 AUTHORED (registry row flips to AUTHORED/PENDING).
DO-D: auditor `--wave W20.4` exits 0; tag `w20-wave-4`; push;
attestation.
COMMIT: "W20.4.7 — wave W20.4 closed: K1 complete (single torch path,
fixtures pinned, G20.P1 authored)"

---

## WAVE W20.5 — fht_forward_large / fht_backward_large (K2, PROPOSAL §3.2)

Goal: the foundational multi-block butterfly for K = 2^19…2^22: the
header contract, the kernels (tile-local 13 stages in a 64 KiB smem
tile + cross-tile global passes), bindings, build registration, and
the fht.py eligibility extension. The existing ≤16,384 kernel stays
VERBATIM (byte-identical file diff — the auditor checks it).

#### W20.5.1 — the fht_large header contract (`include/flute/fht_large.cuh`)
- [ ] done
ROLE task-writer · READ `include/flute/fht.cuh` (the existing math
contract style), `src/kernel_fht.cu:1-80` (the resource accounting
discipline), `fht.py:118-160` (segments/rotation_signs conventions) ·
WRITE `src/flute_extended/include/flute/fht_large.cuh` (new)
BRIEF: the host-side math contract as a header: the stage split
(pair distance `len_` 1…K/2; `2·len_ ≤ 16,384` → tile-local — 13
stages for any K ≥ 2^14; else cross stages — log2(K) − 13 of them: 6
at 2^19, 9 at 2^22); the classification function
`fht_large_stages(K) -> {local, cross}`; the sign conventions
(forward: signs AFTER the butterfly; adjoint: BEFORE — matching
`fht_reference`/`fht_reference_adjoint` exactly); the ownership
discipline (fixed thread→pair mapping, fixed tile/warp/lane
ownership, **no atomics anywhere** — the determinism contract);
1/sqrt(b) scaling per the T-convention in fht.cuh.
INVARIANTS: pure header (no .cu); constants 16384/1024 documented;
the no-atomics rule stated as a contract.
DO-D: header exists; `rg "16384|no atomics|cross" …fht_large.cuh` hit;
py n/a; the auditor `exists`+`grep` checks.
COMMIT: "W20.5.1 — the fht_large math contract header"

#### W20.5.2 — `kernel_tq.cu` born: fht_forward_large
- [ ] done
ROLE task-writer · READ `src/flute_extended/src/kernel_fht.cu:80-200`
(the fht_stage pairing discipline: low = u+w, high = u−w, ascending —
the rounding-order contract), `kernel_gemv_splitk.cu:1-120` (the
workspace/finalizer pattern) · WRITE
`src/flute_extended/src/kernel_tq.cu` (new)
BRIEF: pass 1 (tile-local): grid = rows × (K/16,384) tiles; each CTA
`cp.async`-stages its 64 KiB fp32 tile, runs the 13 local stages with
kernel_fht.cu's EXACT pairing discipline, writes back; pass 2 (cross
stages): elementwise global pair op — thread i reads (u[i], u[i+len_]),
writes (u+w, u−w), one launch per stage or a stage-looped launch.
1024 threads (32 warps), `__launch_bounds__(1024)`, 64 KiB smem via
`cudaFuncSetAttribute` (the checked opt-in — kernel_fht.cu's
precedent); signs applied post-butterfly (forward). Determinism: fixed
ownership, no atomics, run-to-run `torch.equal` assertable.
INVARIANTS: `rg -c "atomic" src/flute_extended/src/kernel_tq.cu` == 0;
`__launch_bounds__` on every instantiation; K asserts (K > 16,384,
K % 32 == 0).
DO-D: file exists; the greps; `rg "cp\.async|__launch_bounds__|cudaFuncSetAttribute" kernel_tq.cu` hit; (nvcc compile-check ONLY if W20.0.1 recorded nvcc present — else deferred, recorded).
COMMIT: "W20.5.2 — kernel_tq.cu: fht_forward_large (13 local stages +
cross passes, deterministic)"

#### W20.5.3 — fht_backward_large (the adjoint orientation)
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (5.2's result), `kernel_fht.cu`
(the backward/adjoint kernel's sign placement) · WRITE
`src/flute_extended/src/kernel_tq.cu`
BRIEF: the adjoint: signs applied BEFORE the butterfly (per
fht_reference_adjoint), same stage split, same staging, same
ownership discipline; out-of-place API; dtype-templated like
kernel_fht.cu (fp16/bf16/fp32 in/out, fp32 smem accumulation).
INVARIANTS: same greps as 5.2 module-wide (no atomics; launch bounds).
DO-D: `rg "fht_backward_large" kernel_tq.cu` hits (kernel + host
wrapper); the no-atomics grep still 0.
COMMIT: "W20.5.3 — fht_backward_large (adjoint orientation, same
discipline)"

#### W20.5.4 — bindings + build registration + canary/trap macros
- [ ] done
ROLE task-writer · READ `src/flute_extended/src/bindings.cpp:76-100`
(the FHT binding pattern), `setup.py` (the TU list + FLUTE_CUDA_ARCHES) ·
WRITE `bindings.cpp`, `setup.py`
BRIEF: bind `fht_forward_large(x, signs)` / `fht_backward_large(g,
signs)` following the existing FHT binding pattern exactly (2-D
contiguous (M, K), dtype dispatch, (K,) fp32 signs, K-shape asserts:
K > 16,384 && K % 32 == 0 — refuse otherwise, loudly); setup.py gains
`kernel_tq.cu` in the TU list (family "tq"; always built);
compile-time macros `FLUTE_TQ_CANARY` / `FLUTE_TQ_TRAP` declared in
the kernel file (default OFF — the W20.6 variants); build flags:
`-gencode arch=compute_86,code=sm_86` + PTX fallback (unchanged
default 86).
INVARIANTS: existing bindings untouched (`git diff` confined to added
blocks); the TU-list append is the only setup.py change.
DO-D: `rg "fht_forward_large|fht_backward_large" bindings.cpp` hit;
`rg "kernel_tq" setup.py` hit; `rg "FLUTE_TQ_CANARY|FLUTE_TQ_TRAP" kernel_tq.cu` hit; compile-check if nvcc present (recorded either way).
COMMIT: "W20.5.4 — bindings + build registration for kernel_tq.cu"

#### W20.5.5 — fht.py wrapper + the eligibility table extension
- [ ] done
ROLE task-writer · READ `fht.py:314-430` (kernel_available, the
dispatch + _kernel_eligible :355, fht_apply :434), `:561-640`
(fht_adjoint) · WRITE `src/flute_extended/fht.py`
BRIEF: extend eligibility: K ≤ 16,384 → the EXISTING single-block
kernel VERBATIM (the dispatch lines untouched — the auditor diffs
kernel_fht.cu to prove it); K > 16,384 && K % 32 == 0 → `fht_large`
(when the extension is built); the conv window (24,576 = two ≤16,384
segments) already routes to the existing kernel — UNTOUCHED; the
loud-refuse discipline when backend="kernel" but the extension is
absent (the file's existing convention); the auto backend unchanged in
spirit. `__all__` gains nothing new (the same fht_apply/fht_adjoint
surface — internal routing only).
INVARIANTS: `git diff w20-wave-4..HEAD -- src/flute_extended/src/kernel_fht.cu` is EMPTY (the byte-identical guarantee); the wrapper refuses ineligible K loudly.
DO-D: the kernel_fht.cu empty-diff check (auditor git fact); `rg "fht_large|16384" fht.py` hits; py_compile.
COMMIT: "W20.5.5 — fht.py routes K>16,384 to fht_large (existing kernel
verbatim)"

#### W20.5.6 — wave close W20.5
- [ ] done
ROLE orchestrator · standard close-out.
Tier-D: 5.1–5.5 sets; the kernel_fht.cu empty-diff; the no-atomics
grep; (conditional) the compile record.
Tier-G: G20.P2's twins are authored in W20.6 (this wave closes on
Tier-D only — noted in the ledger).
DO-D: auditor `--wave W20.5` exits 0; tag `w20-wave-5`; push;
attestation.
COMMIT: "W20.5.6 — wave W20.5 closed: the multi-block butterfly is
authored"

---

## WAVE W20.6 — debug twins + canary/trap builds (K2 end)

Goal: the G-B1..G-B5-style differential surfaces: the scalar debug
twins, the canary/trap build variants, the CUDA-gated fht_large test
file, and gates G20.P2 + G20.6 in the harness.

#### W20.6.1 — the tq_debug_simple twins (scalar fragment assembly)
- [ ] done
ROLE task-writer · READ `src/flute_extended/src/kernel_debug_simple.cu:1-80`
(the existing scalar-twin pattern), `kernel_tq.cu` (5.2/5.3) · WRITE
`kernel_tq.cu` (append)
BRIEF: `tq_debug_fht_large_forward/backward` — scalar, fragment-wise
assembly of the same math (stage by stage, the same order), test-only
(never on the production dispatch path); bound in bindings.cpp behind
the same names; the differential reference the box compares the real
kernel against.
INVARIANTS: test-only (docstring states it); no atomics; same pairing
order as the real kernel (else the twin is meaningless).
DO-D: `rg "tq_debug" kernel_tq.cu bindings.cpp` hit; py n/a; compile
record (conditional).
COMMIT: "W20.6.1 — the fht_large scalar twins (test-only)"

#### W20.6.2 — canary/trap build variants
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (the macros from 5.4),
`setup.py` · WRITE `scripts/gpu/w20_build_variants.sh` (new),
`kernel_tq.cu` (the guarded blocks, if not already in 5.4)
BRIEF: the canary build poisons the tile before encode (surviving
poison = coverage regression — the FLUTE_BWD_CANARY discipline); the
trap build forces a wrong fold order (it MUST fail the determinism
gate — the negative control). Build variants via env macros:
`FLUTE_TQ_CANARY=1 python3 setup.py build_ext --inplace` (and TRAP),
each writing a distinct .so name (e.g. `_C_canary`) so production
stays clean; the script documents the sequence on the box.
INVARIANTS: default builds contain NEITHER variant (macros off); the
guarded code lives behind `#ifdef` only.
DO-D: `rg "FLUTE_TQ_CANARY|FLUTE_TQ_TRAP" kernel_tq.cu` ≥ 2 each; the
variants script exists with both builds; `rg "#ifdef" kernel_tq.cu`
guards present.
COMMIT: "W20.6.2 — canary/trap variants (poison + wrong-fold negative
control)"

#### W20.6.3 — test_w20_fht_large.py (CUDA-gated)
- [ ] done
ROLE task-writer · READ `_gpu_guard.py`, the fixtures manifest,
`kernel_fht.cu`'s test conventions in `test_flute.py` (READ ≤60 lines:
naming/param style) · WRITE `src/rag/tests/w20/test_w20_fht_large.py`
(new)
BRIEF: CUDA-gated suite: differential twins (kernel vs debug twin vs
torch reference, all on CUDA, allclose ≤ 1e-5 — G20.P2); run-to-run
determinism (`torch.equal`, the level-1 hard contract); the
geometry ladder (2^19, 2^20, 2^22; K%32≠0 refusals asserted to RAISE);
canary/trap subtests skip-unless the variant .so is importable
(box-side; they assert canary-zero-poison and trap-must-fail —
G20.6). Every test imports `_gpu_guard` at collection.
INVARIANTS: module-level CUDA skip; no test passes on CPU; thresholds
mirror TASKS.md §4.
DO-D: file exists; `rg "def test_" …` > 0; `rg "_gpu_guard" …` hits;
py_compile.
COMMIT: "W20.6.3 — the fht_large CUDA-gated suite (twins, determinism,
canary/trap)"

#### W20.6.4 — wave close W20.6 (gates G20.P2 + G20.6 authored)
- [ ] done
ROLE orchestrator · standard close-out + the harness extension
(`scripts/gpu/w20_gates.py` gains G20.P2: kernel-vs-twin-vs-torch
allclose 1e-5 + run-to-run torch.equal; G20.6: canary zero-poison +
trap-fails-determinism — each a gate function with thresholds
mirroring §4).
Tier-D: 6.1–6.3 sets.
Tier-G: G20.P2, G20.6 → AUTHORED.
DO-D: auditor `--wave W20.6` exits 0; `rg "G20.P2|G20.6"
scripts/gpu/w20_gates.py` hit; tag `w20-wave-6`; push; attestation.
COMMIT: "W20.6.4 — wave W20.6 closed: K2 complete, G20.P2/G20.6
authored"

---

## WAVE W20.7 — tq_quant_batched (K3, PROPOSAL §3.3 — THE wave begins)

Goal: the fused encode kernel: deterministic norm reduction, the
boundary-scaling identity's compare ladder, the inline multi-tile
butterfly, bfi packing, the batched entry. R20.6's grid-overshoot
guard rides the entry.

#### W20.7.1 — the deterministic norm reduction (the split-K finalizer discipline)
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (5.2's staging), `kernel_gemv_splitk.cu:120-260`
(the last-ticket finalizer + workspace fold) · WRITE `kernel_tq.cu`
BRIEF: grid-stride sum-of-squares; per-warp `shfl.sync` butterfly tree
(fixed masks 16→1 = fixed association order — NOTE fp32
`redux.sync.add` does not exist, PTX §9.7.13.12; the shfl tree IS the
fp32 path); per-CTA partial → a workspace slot owned EXCLUSIVELY by the
CTA; the last-ticket CTA of each unit folds partials in ascending CTA
order — no atomics anywhere; the norm lands as a 0-dim fp32 device
tensor per unit (the workspace is a wrapper-allocated tensor).
INVARIANTS: no atomics (module grep stays 0); fixed fold order;
workspace ownership exclusive.
DO-D: `rg "shfl|finalizer|last" kernel_tq.cu` hit; the atomics grep 0;
compile record (conditional).
COMMIT: "W20.7.1 — the deterministic norm pass (shfl tree + finalizer,
no atomics)"

#### W20.7.2 — boundary scaling + the in-register compare ladder
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (7.1), `src/rag/codebooks.py`
(boundary tables), `turboquant.py` (3.1's identity comment — the SAME
identity) · WRITE `kernel_tq.cu`
BRIEF: scale the ≤16 boundaries ONCE by the unit's norm (16
multiplies, in smem — the identity: argmin|y_j/norm − c_k| ≡
argmin|y_j − norm·c_k|, monotone exact); per coordinate an in-register
linear scan of the ≤16 scaled boundaries (2^b ≤ 16 — a compare
ladder, no search — the vectorization the paper demands); tie
semantics documented to match the torch path's searchsorted parity
(the G20.P3 gate counts ties; boundary-exact flips are level-2
accepted, ≤ 0.01%).
INVARIANTS: the normalized vector is NEVER materialized; the identity
comment cites PROPOSAL §3.3.
DO-D: `rg "16384|compare|ladder|identity" kernel_tq.cu` hits; atomics
grep 0.
COMMIT: "W20.7.2 — the scaled-boundary compare ladder (the
vectorization the paper demands)"

#### W20.7.3 — the multi-tile butterfly + bfi packing (syndrome: np.packbits)
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (5.2's butterfly inline),
`turboquant.py:159-186` (the pinned byte layout), the PTX ISA refs in
`PROPOSAL.md §Sources` · WRITE `kernel_tq.cu`
BRIEF: the multi-tile butterfly inlined (per PROPOSAL §3.3 step 3);
packing: each thread owns a fixed 16-coordinate slice, packs its b-bit
indices into 6 (b=3) or 8 (b=4) bytes via `bfi`/shifts (PTX
§9.7.1.20); tiles are disjoint contiguous coordinate ranges →
tile-aligned packing has NO cross-tile dependencies; the lo/hi split
(n_lo/n_hi) packs the two halves separately, **byte layout identical
to today's pack_bits** (fixture-pinned — the same pin as 3.2, restated
in the docstring); codes out = packed uint8 device tensors.
INVARIANTS: the layout pin is cited in-file; no cross-tile
dependencies (comment).
DO-D: `rg "bfi|n_lo|pack" kernel_tq.cu` hit; atomics grep 0; compile
record.
COMMIT: "W20.7.3 — inlined butterfly + bfi packing (layout
fixture-pinned)"

#### W20.7.4 — the batched entry + bindings (grid Σ tiles, overshoot-guarded)
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (7.1–7.3), `bindings.cpp`
(binding conventions) · WRITE `kernel_tq.cu`, `bindings.cpp`
BRIEF: `tq_quant_batched(units, d, books, out_codes, out_norms)`: one
entry over all of a prefill's 120–192 units — grid = Σ units×tiles +
norm CTAs (≈6,100 CTAs at 2^19 — ~76 waves of the 80 SMs); R20.6's
guard: grid-stride loop per CTA (2^22 M-states = 256-tile units still
~1 wave-pair); fp32 in, uint8 + fp32(0-dim) out; wrapper allocates
workspaces (the norm slots) — the host NEVER touches device memory
content; the 3.5-bit split expressed via (n_lo, bits_lo, n_hi,
bits_hi) book pairs.
INVARIANTS: no `.item()`/`.cpu()` in the wrapper (it is C++ — the
discipline: no D2H sync at all); shape asserts loud.
DO-D: `rg "tq_quant_batched" kernel_tq.cu bindings.cpp` hit; `rg
"grid-stride|grid_stride|overshoot" kernel_tq.cu` hit; atomics 0.
COMMIT: "W20.7.4 — the batched quant entry (two kernel chains per
prefill, not 4,000 round trips)"

#### W20.7.5 — wave close W20.7
- [ ] done
ROLE orchestrator · standard close-out.
Tier-D: 7.1–7.4 sets; the atomics-free grep; compile record.
Tier-G: G20.P3's quant half authored in W20.8 (with dequant).
DO-D: auditor `--wave W20.7` exits 0; tag `w20-wave-7`; push;
attestation.
COMMIT: "W20.7.5 — wave W20.7 closed: the fused encode is authored"

---

## WAVE W20.8 — tq_dequant_batched + the dispatcher (K3 end — TQ leaves the profile)

Goal: the fused decode (bfe unpack, smem LUT gather, adjoint, ×norm)
and the §3.7 kernel-first dispatcher wiring TQCodes to the kernels;
the CUDA-gated TQ kernel suite; gates G20.P3, G20.1, G20.2, G20.4.

#### W20.8.1 — tq_dequant_batched (syndromes: numpy gather + the CPU adjoint)
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (the family so far),
`turboquant.py:765-780` (the decode semantics) · WRITE `kernel_tq.cu`
BRIEF: unpack each packed byte via `bfe.u32` (PTX §9.7.1.19 — b-bit
field extract at the fixed offset; with bfe, unpacking is free — codes
stay packed everywhere, rev 2's K5 note); gather the centroid from the
smem LUT (≤16-entry table broadcast to every CTA — NO global-memory
gathering); apply signs (adjoint orientation — pre-butterfly); run the
multi-tile adjoint butterfly; multiply by the norm; store fp16/fp32
(dtype dispatch). ~230 KiB codes in, 1–2 MiB out per 2^19 unit.
INVARIANTS: the LUT broadcast discipline; no atomics; the byte-layout
pin cited.
DO-D: `rg "bfe|smem|lut|LUT" kernel_tq.cu` hit; atomics 0; compile
record.
COMMIT: "W20.8.1 — tq_dequant_batched (bfe + smem LUT + adjoint)"

#### W20.8.2 — the kernel-first dispatcher (PROPOSAL §3.7)
- [ ] done
ROLE task-writer · READ `turboquant.py` (quant/dequant +
_route_for), `fht.py:314-430` (the auto-backend precedent) · WRITE
`src/rag/turboquant.py`
BRIEF: dispatch: **CUDA tensor + kernel available → the kernel path,
ALWAYS** (codes as device tensors); CPU tensor → the torch reference
path (the differential twin this box authors, the GPU-less path);
extension presence probed once (the fht.py `kernel_available` pattern
applied at the TurboQuant level); loud diagnostics on mismatched
devices; NO numpy path on ANY box at runtime — the npz edge and the
faiss hand-off remain the only foreign boundaries. TQCodes emitted by
either path (the contract from 2.1 unchanged).
INVARIANTS: no consumer-visible API change; the dispatcher table
documented (which path serves which device class); `rg -c "np\.|numpy"
src/rag/turboquant.py` stays 0.
DO-D: `rg "kernel_available|dispatch" src/rag/turboquant.py` hit;
py_compile; the numpy grep 0.
COMMIT: "W20.8.2 — the kernel-first dispatcher (torch = the twin, the
CPU path)"

#### W20.8.3 — test_w20_tq_kernels.py (CUDA-gated; G20.P3/G20.4 thresholds)
- [ ] done
ROLE task-writer · READ `_gpu_guard.py`, the fixtures, `PROPOSAL.md §5`
· WRITE `src/rag/tests/w20/test_w20_tq_kernels.py` (new),
`scripts/gpu/w20_gates.py` (extend: G20.P3 — kernel quant+dequant
parity vs torch on CUDA (codes equal except ties ≤ 0.01%, norm ≤ 1
ulp, recon allclose ≤ 1e-5); G20.4 — the full parity contract
(level-1 run-to-run torch.equal on kernel outputs; level-2
cross-backend thresholds); G20.1/G20.2 phase-1 — the capture-cycle +
step-time phase timers read from a `finetune_m1m2.py` short-run
history (thresholds per §4; full confirmation after K4/K5))
BRIEF: the kernel suite: parity vs the torch twin (both on CUDA),
run-to-run determinism, the batched-grid shapes (120–192 units),
the tie counter, geometry coverage (2^19, 24,576, 2^22), refusal
paths (bad K, bad dtype). All CUDA-gated; thresholds mirror §4 1:1.
INVARIANTS: module-level CUDA skip; no CPU-passable test.
DO-D: file exists + `_gpu_guard` import; `rg "G20.P3|G20.4|G20.1|G20.2"
scripts/gpu/w20_gates.py` hit; py_compile.
COMMIT: "W20.8.3 — the TQ kernel suite + G20.P3/P4/1/2 authored"

#### W20.8.4 — wave close W20.8 (K3 complete)
- [ ] done
ROLE orchestrator · standard close-out + HANDOVER's K3 section (the
two-kernel-chains-per-prefill note; the box runs G20.P3/P4 and the
G20.1/2 phase-1 timers).
Tier-D: 8.1–8.3 sets.
Tier-G: G20.P3, G20.4, G20.1, G20.2 → AUTHORED.
DO-D: auditor `--wave W20.8` exits 0; tag `w20-wave-8`; push;
attestation.
COMMIT: "W20.8.4 — wave W20.8 closed: K3 complete (kernel-first
dispatch live)"

---

## WAVE W20.9 — the QJL kernel chain + tq_l2norm_rows (K4)

Goal: the Alg.-2 kernel chain (flag-off, kernel-complete) and the
serving normalize kernel.

#### W20.9.1 — the tq_qjl chain kernels
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (the family),
`turboquant.py` (3.4's torch qjl semantics — the contract to match) ·
WRITE `kernel_tq.cu`, `bindings.cpp`
BRIEF: `tq_qjl_quant/dequant`: the residual r = x − dequant_mse as a
fused quant→dequant chain; γ via 7.1's norm pass; the sketch
`sign(FHT(r))` (scale-invariant — sign(FHT(r/γ)) ≡ sign(FHT(r)), the
same identity); the compensation rides fht_backward_large; no CPU
step anywhere in the A/B. Flag-off: the entries exist and are bound;
the runtime keeps `--qjl` default off.
INVARIANTS: the flag-off completeness (bound + callable on the box);
the identity comments.
DO-D: `rg "tq_qjl" kernel_tq.cu bindings.cpp` hit; atomics 0;
compile record.
COMMIT: "W20.9.1 — the QJL kernel chain (flag-off, complete)"

#### W20.9.2 — tq_l2norm_rows
- [ ] done
ROLE task-writer · READ `kernel_tq.cu` (7.1's shfl tree) · WRITE
`kernel_tq.cu`, `bindings.cpp`
BRIEF: rows × 256 threads; per-warp shfl.sync tree + the fixed-order
cross-warp fold (the same discipline, smaller tile); the serving
index frame's normalize step; bound and callable.
INVARIANTS: no atomics; fixed fold order.
DO-D: `rg "tq_l2norm_rows" kernel_tq.cu bindings.cpp` hit; atomics 0.
COMMIT: "W20.9.2 — tq_l2norm_rows (the frame normalize kernel)"

#### W20.9.3 — the QJL/l2norm CUDA-gated tests + gate stubs
- [ ] done
ROLE task-writer · READ `_gpu_guard.py`, `turboquant.py` (qjl branch) ·
WRITE `src/rag/tests/w20/test_w20_qjl_l2norm.py` (new)
BRIEF: qjl kernel-vs-torch parity (allclose 1e-5) incl. the residual
chain; l2norm vs torch reference (1 ulp); run-to-run determinism;
flag-off non-invasiveness (default path unchanged when the extension
is absent). CUDA-gated throughout.
INVARIANTS: module skip; thresholds mirror §4.
DO-D: file exists + guard import; py_compile.
COMMIT: "W20.9.3 — the QJL/l2norm CUDA-gated suite"

#### W20.9.4 — wave close W20.9
- [ ] done
ROLE orchestrator · standard close-out.
Tier-D: 9.1–9.3 sets.
Tier-G: the QJL gates fold into G20.P3's scope (noted in §4).
DO-D: auditor `--wave W20.9` exits 0; tag `w20-wave-9`; push;
attestation.
COMMIT: "W20.9.4 — wave W20.9 closed: the QJL chain + l2norm
authored"

---

## WAVE W20.10 — the serving-side batched passes (K4 end)

Goal: W20.K4's serving side: the install requant, the index frame
vector builds, the eval harness — all batched GPU passes calling the
same batched entries; faiss receives GPU-produced vectors once at its
boundary.

#### W20.10.1 — the install requant as one batched pass
- [ ] done
ROLE task-writer · READ `install.py` (4.2's torch path),
`tq_cache.py` (the install hooks), `PROPOSAL.md §3.6` · WRITE
`src/rag/install.py`, `src/rag/tq_cache.py` (if the hook needs the
batched entry)
BRIEF: the install requant (dequant Σ dequant → requant over 24 S
units + M1/M2) as ONE batched chain (tq_dequant_batched over the
units, torch sum, tq_quant_batched back); the W16 verbatim install
stays bit-exact WITHIN a backend (it copies codes, never recomputes —
untouched); `sum_turboquant_codes` signature unchanged.
INVARIANTS: the W16 copy-path untouched; the requant path is the
batched chain; no numpy.
DO-D: `rg "tq_quant_batched|tq_dequant_batched" src/rag/install.py`
hit; `rg -c "np\.|numpy" src/rag/install.py` == 0; py_compile.
COMMIT: "W20.10.1 — the install requant batched (one chain over 26
units)"

#### W20.10.2 — the index frame vectors batched (index.py:372/473/510)
- [ ] done
ROLE task-writer · READ `index.py:360-520` (4.3's torch builds),
`PROPOSAL.md §3.6` · WRITE `src/rag/index.py`
BRIEF: the retrieval-frame vector builds (dequant + concat +
L2-normalize) as batched passes: tq_dequant_batched over the units,
torch concat, tq_l2norm_rows for the frame normalize; the faiss add
receives GPU-produced vectors once at its boundary (the single
`.numpy()` per build — the FAISS-EDGE marker from 4.3, now feeding
from the batched chain).
INVARIANTS: exactly one FAISS-EDGE `.numpy()` per build site; no
other numpy.
DO-D: `rg "tq_l2norm_rows|tq_dequant_batched" src/rag/index.py` hit;
the FAISS-EDGE count ≥ 3 and the numpy grep confined to those lines;
py_compile.
COMMIT: "W20.10.2 — the frame vectors batched (faiss fed once at the
boundary)"

#### W20.10.3 — the eval harness batched
- [ ] done
ROLE task-writer · READ `evals.py` (4.4's torch stats + the ×5
sites), `m1m2.py` (the :345 region, post-4.4) · WRITE `src/rag/evals.py`
BRIEF: the five eval sites call the batched entries (round trips via
tq_quant/dequant_batched; stats in torch; REPORT-EDGE formatting
only); m1m2's remaining runtime sites re-checked torch-clean.
INVARIANTS: no D2H-then-compute; REPORT-EDGE markers only.
DO-D: `rg "tq_quant_batched|tq_dequant_batched" src/rag/evals.py`
hit; the scoped numpy grep clean; py_compile.
COMMIT: "W20.10.3 — the eval harness batched"

#### W20.10.4 — wave close W20.10 (G20.5 authored)
- [ ] done
ROLE orchestrator · standard close-out + the harness extension
(G20.5: protocol integrity — TRAIN → RE-INGEST → INDEX → EVAL → QUERY
end-to-end on the kernel path; npz digests stable run-to-run;
cross-backend decode allclose ≤ 1e-5; the W16 verbatim install
bit-exact within the backend — each a gate function).
Tier-D: 10.1–10.3 sets.
Tier-G: G20.5 → AUTHORED.
DO-D: auditor `--wave W20.10` exits 0; `rg "G20.5" scripts/gpu/w20_gates.py` hit; tag `w20-wave-10`; push; attestation.
COMMIT: "W20.10.4 — wave W20.10 closed: K4 complete, G20.5 authored"

---

## WAVE W20.11 — the trainer levers (K5)

Goal: `--refresh-every 8` as a first-class A/B regime + the loader
pinning. (The "unpacked hot codes" lever is DROPPED per PROPOSAL §8 —
with bfe, unpacking is free.)

#### W20.11.1 — the refresh-8 regime
- [ ] done
ROLE task-writer · READ `scripts/gpu/finetune_m1m2.py` (the W19 flags
--refresh-every/--gate-init/--absolute-loss), `src/rag/m1m2_finetune.py`
(refresh_every docstring), HANDOVER.md's W19 section (the staleness
arithmetic) · WRITE both files (flag documentation + the regime notes)
BRIEF: make refresh 8 a documented A/B regime: the driver's --help
text, the finetune docstring's staleness bound (lr × (N−1) = lr × 7),
the HANDOVER confirmation sequence updated (step 3's "A/B
--refresh-every 8"). Default stays 4 (quality-first; the regime is
the operator's call on the box). No semantic code change to the pool.
INVARIANTS: default 4 unchanged; docstring/threshold text only + any
plumbing already present from W19.
DO-D: `rg "refresh" scripts/gpu/finetune_m1m2.py` hits the help +
docstring; py_compile both.
COMMIT: "W20.11.1 — the refresh-8 regime documented (staleness lr×7)"

#### W20.11.2 — the loader pinning
- [ ] done
ROLE task-writer · READ `src/rag/m1m2_finetune.py` (the capture pool's
batch feed), `scripts/gpu/finetune_m1m2.py` (the pairs-file loading) ·
WRITE either
BRIEF: pin the capture-loop's host→device path: batches built as
pinned-memory tensors (`pin_memory()`), H2D copies non_blocking, the
prefill's inputs pre-staged per pool refresh; zero semantics change
(bit-identical values — pinning is transport, not math); the phase
timers (W19 instrumentation) will show the effect on the box.
INVARIANTS: values bit-identical (no math change); pinning confined to
the capture loop.
DO-D: `rg "pin_memory|non_blocking" src/rag/m1m2_finetune.py` or the
driver hits; py_compile.
COMMIT: "W20.11.2 — the capture-loop loader pinning (pinned H2D,
non-blocking)"

#### W20.11.3 — wave close W20.11 (G20.2 full + G20.3 authored)
- [ ] done
ROLE orchestrator · standard close-out + the harness extension
(G20.2 full: step ≤ 2 s after K3, ≤ 1 s at refresh 8, 300 steps ≤ 10
min end-to-end — read from the phase-timer history; G20.3: the W19
convergence gates — loss leaves ln(8) at step 0, gk/gv_absmax > 0,
held-out spread positive).
Tier-D: 11.1/11.2 sets.
Tier-G: G20.2 (full), G20.3 → AUTHORED.
DO-D: auditor `--wave W20.11` exits 0; `rg "G20.2|G20.3"
scripts/gpu/w20_gates.py` hit; tag `w20-wave-11`; push; attestation.
COMMIT: "W20.11.3 — wave W20.11 closed: K5 complete, G20.2/G20.3
authored"

---

## WAVE W20.12 — closure (the handover + the all-DoD audit)

Goal: the human-facing close-out: HANDOVER's W20 section, the
SPECIFICATION status addendum, the final audit, the tag.

#### W20.12.1 — HANDOVER.md W20 + the SPECIFICATION addendum
- [ ] done
ROLE task-writer (or orchestrator) · READ `HANDOVER.md` (the W19
shape), `SPECIFICATION.md` (§13 + the status line), TASKS.md §0/§4 ·
WRITE `HANDOVER.md`, `SPECIFICATION.md`
BRIEF: HANDOVER's W20 section: the wave table (what landed where), the
GPU-box confirmation sequence (pointer to env/W20_GPU_PROTOCOL.md +
the gate registry state), the expected phase-timer arithmetic (the
§6 ladder: ~15–25 µs/unit; capture ~5–8 s model-bound), open items
(the read-gate training question, cross-box portability rules), and
the no-CPU-test attestation summary. SPECIFICATION gains a short "W20
implementation status" addendum (the kernel-first runtime, the
dispatcher, the two foreign edges) — authority-preserving, doc-only.
INVARIANTS: HANDOVER stays the human narrative; no gate result is
claimed beyond AUTHORED/PENDING/box-reported.
DO-D: `rg "W20" HANDOVER.md` section present; SPECIFICATION addendum
present (`rg "W20 implementation status" SPECIFICATION.md` hits).
COMMIT: "W20.12.1 — the W20 handover + specification status addendum"

#### W20.12.2 — the final all-DoD audit + attestation
- [ ] done
ROLE orchestrator · READ TASKS.md (full — the one deliberate full
read), the auditor output · WRITE TASKS.md §0/§4/§5 final states,
`/home/z/my-project/worklog.md`
BRIEF: run `python3 scripts/w20_static_audit.py --all` (every wave's
Tier-D re-checked in one pass); walk the §6 closure checklist; verify
13/13 attestation lines; reconcile §4 statuses (AUTHORED at minimum;
GREEN/RED only from box reports); write the final worklog entry.
Tier-D: the --all exit 0 + the checklist below.
Tier-G: the registry states are the handoff truth.
DO-D: auditor --all exits 0; §0 shows 13/13 CLOSED; §5 shows 13 lines.
COMMIT: "W20.12.2 — the final all-DoD audit green"

#### W20.12.3 — the final commit, push, tag
- [ ] done
ROLE orchestrator · WRITE the git remote state
BRIEF: final commit (this task's), `git push origin main`, annotated
tag `w20-complete`, push the tag; then — and only per PROMPT.md §8 —
return with the handoff report.
DO-D: `git tag -l w20-complete` non-empty; origin/main == HEAD; the
worklog final entry exists.
COMMIT: "W20.12.3 — W20 complete: pushed, tagged, handed to the A10G
box"

---

## 4. The GPU-box gate registry (statuses: NOT-AUTHORED → AUTHORED → PENDING → GREEN/RED from box reports ONLY)

Thresholds are the gate's source of truth (mirrored 1:1 into
`scripts/gpu/w20_gates.py`'s THRESHOLDS dict — drift is a wave-close
failure). A gate turns GREEN/RED **only** from
`env/w20_gate_reports/<id>.json` produced on the A10G box.

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

The capture (W20.1) and the 237-suite baseline are protocol steps, not
gates; they are recorded in `env/W20_GPU_PROTOCOL.md`.

## 5. The attestation ledger (the no-CPU-test law, per wave)

One line per wave close, written by the orchestrator, only when
truthful (PROMPT.md §1.4):

| wave | attestation |
|---|---|
| W20.0 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.0 exit 0 (26/26, re-verified post-tag); push w20-wave-0 |
| W20.1 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.1 exit 0 (25/25, re-verified post-tag); push w20-wave-1 |
| W20.2 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.2 exit 0 (24/24, re-verified post-tag); push w20-wave-2 |
| W20.3 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.3 exit 0 (24/24, re-verified post-tag; incl. the module-wide numpy-zero audit line); push w20-wave-3 |
| W20.4 | — (pending) |
| W20.5 | — (pending) |
| W20.6 | — (pending) |
| W20.7 | — (pending) |
| W20.8 | — (pending) |
| W20.9 | — (pending) |
| W20.10 | — (pending) |
| W20.11 | — (pending) |
| W20.12 | — (pending) |

Line format: `CLOSED <date> — no test executed, no CPU test authored,
no repo module imported on this box; auditor --wave W20.<n> exit 0;
push w20-wave-<n>`.

## 6. The final closure checklist (the orchestrator's return condition)

- [ ] §0: 13/13 waves CLOSED (no `- [ ]`/`- [~]` without disposition)
- [ ] `python3 scripts/w20_static_audit.py --all` exits 0
- [ ] §4: every gate AUTHORED (script + thresholds committed; statuses
      honest: PENDING or box-reported)
- [ ] §5: 13/13 truthful attestation lines
- [ ] `env/W20_ENV.md` + `env/w20_gpu_env.lock` protocol committed
      (the lock is written by the box; its absence is recorded, not
      fatal)
- [ ] HANDOVER.md W20 section complete; SPECIFICATION addendum present
- [ ] final commit + push; tag `w20-complete` on origin
- [ ] the worklog's final entry written (PROMPT.md §8 report follows)

The orchestrator returns ONLY when every box above is checked
(PROMPT.md §8). The A10G box executes `env/W20_GPU_PROTOCOL.md` next.
