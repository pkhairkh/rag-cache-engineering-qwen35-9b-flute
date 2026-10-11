# TASKS.md — W20 Execution Ledger: Waves, Tasks, DoD (the machine contract)

Purpose: the execution ledger for **W20 — the TQ kernel family**
(PROPOSAL.md rev 2): 13 waves, 58 tasks, each task one sub-agent, one
commit; each wave one DoD audit, one push, one tag. The orchestrator
iterates this file top-to-bottom under PROMPT.md until §6 closes.
Authority: SPECIFICATION.md > PROPOSAL.md > PROMPT.md > **TASKS.md** >
HANDOVER.md. Line references are **w20-baseline coordinates** (the tree
at tag `w20-baseline`); before editing, re-locate by symbol (`rg`) —
earlier waves move lines.
Status: COMPLETE — 13/13 waves closed (W20.0–W20.12); the post-close box-feedback waves W20.13 (5 tasks) and W20.14 (5 tasks, 2026-10-12) closed on top, the post-finetune-regime wave W20.15 (5 tasks, 2026-10-12) on top of those, the content-addressing wave W21 (5 tasks, 2026-10-12 — the position-lock + anisotropy-pedestal mitigation, gates + global caches only) on top, and the box-run repair W21.6 (2026-10-12 — the meta-mix self-heal + the deterministic content scatter; the kernel was right, the delivery was not) on top; all Tier-D DoD green; gates AUTHORED (PENDING box reports by design; §4.1 records the W20.13.4 threshold re-derivation).

---

## 0. The wave ledger (the orchestrator's status table)

| wave | phase (PROPOSAL §8) | title | tasks | gates authored | status |
|---|---|---|---|---|---|
| W20.0 | pre-K | environment + audit infrastructure | 5 | — | CLOSED |
| W20.1 | K1 | baseline tag + frozen probe fixtures | 4 | (capture) | CLOSED |
| W20.2 | K1 | TQCodes device contract + codebook mirror | 3 | — | CLOSED |
| W20.3 | K1 | turboquant.py torch rewrite (the core) | 6 | — | CLOSED |
| W20.4 | K1 | wiring + serving sweep + K1 gates | 7 | G20.P1 | CLOSED |
| W20.5 | K2 | fht_forward_large / fht_backward_large | 6 | — | CLOSED |
| W20.6 | K2 | debug twins + canary/trap builds | 4 | G20.P2, G20.6 | CLOSED |
| W20.7 | K3 | tq_quant_batched (the fused encode) | 5 | — | CLOSED |
| W20.8 | K3 | tq_dequant_batched + dispatcher | 4 | G20.P3, G20.1, G20.2, G20.4 | CLOSED |
| W20.9 | K4 | QJL kernel chain + tq_l2norm_rows | 4 | — | CLOSED |
| W20.10 | K4 | serving-side batched passes | 4 | G20.5 | CLOSED |
| W20.11 | K5 | trainer levers (refresh 8, loader pinning) | 3 | G20.2 full, G20.3 | CLOSED |
| W20.12 | closure | handover + final all-DoD audit | 3 | — | CLOSED |
| W20.13 | box-feedback | the post-compile stabilization: pack domain, rerank device, test adaptation, parity re-derivation | 5 | — | CLOSED |

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
- [x] done
4 CUDA-gated tests (manifest-load, parametrized fixture parity,
run-to-run determinism, device report); manifest-ABSENT = module skip
(present-but-empty fails loudly — deliberate). Guard import: `from
w20 import _gpu_guard` + sys.path insert of src/rag/tests (pytest
prepend resolves w20.test_… with basedir src/rag/tests — the dotted
src.rag form would NOT resolve; conftest anchors src/rag, not the
repo root). The W20.3.2 blast radius resolved mechanically:
parametrize constrained to b∈{3,4} + the in-code domain comment; the
value-range rejection rows kept for b∈{1,2} (the idx VALUE check was
removed with the numpy era — dead rows dropped, not re-pinned).
Open: sibling test files (test_turboquant_split.py etc.) carry
np.array_equal sites that break on the box — flagged for the owning
sweep (G20.P1 time).
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
- [x] done
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
- [x] done
Boundary resolution (orchestrator, binding for 5.2): the PROPOSAL's
canonical numbers (13 local stages; cross = log2 K − 13 → 6 at 2^19, 9
at 2^22) are the authority — the split criterion is the HALF-TILE line
(2·len_ ≤ 8,192, i.e. sh 0..12 local). The TASKS prose "2·len_ ≤
16,384" would yield 14/5/8 (the len_=8,192 stage's pair window equals
exactly one full tile) — a prose off-by-one in this ledger; the
len_=8,192 stage is classified cross (degenerate, intra-tile-capable;
the header's Boundary note records that 5.2 MAY execute it
intra-tile while keeping the 13/6/9 accounting). 13/6/9 shipped.
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
- [x] done
The len_=8,192 stage is LOWERED intra-tile (pass 1's fht_block runs
sh 0..13 — the header's Boundary note sanctions it; global cross
launches = log2 K − 14; the accounting constant stays 13). Pass 1
reuses kernel_fht.cu:90's fht_block primitive (the pairing discipline
by construction); cp.async 16 B raw on fp32, the converting loop on
fp16/bf16. int64 pair indexing (M·K can pass 2^31 at the envelope
edge). CANARY = NaN poison of out; TRAP = clock64-based pair flip;
both #ifdef, default OFF.
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
- [x] done
Orientation (three concordant authorities: kernel_fht.cu's adjoint,
fht.py:236-237, fht_large.cuh:94-95): signs at the FIRST LOAD (the
input-side sign multiply), the 1/sqrt(K) rsqrtf fold at the FINAL
STORE — the butterfly itself is orientation-free (H symmetric ⇒
self-adjoint). Pass 1 CLONED (the sign multiply can't ride cp.async —
raw 16 B copies — so the adjoint's local pass uses the converting
loop for ALL dtypes, kernel_fht.cu's own adjoint precedent); the
transit cross stages REUSED VERBATIM (unsigned ⇒ orientation-free,
inherits TRAP); the final cross stage is a new kernel (no kBackward
retrofit — additions-only). W20.6 must arm CANARY/TRAP on the
backward's local+final stages.
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
- [x] done
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
- [x] done
Three bands landed: K≤16,384 single-block VERBATIM (segments serve the
24,576 conv window — untouched, verified); 16,384<K≤2^22 pow2 →
fht_large when the built extension exposes the names (lazy
_large_kernel_available, the _awq convention; explicit kernel+old
build → the loud refusal, "rebuild to gain them"); everything else →
auto reference / explicit loud refusal. NOTE: the pre-W20 file
silently fell back for kernel+ineligible K (its comment claimed
validation); the refusal makes that literal truth — no in-repo
consumer passes backend="kernel" (verified). fht.py +94/0 purely
additive; kernel_fht.cu empty-diff verified (the auditor's git fact).
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
- [x] done
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
- [x] done
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
- [x] done
Backward coverage was MISSING (5.3's ledger note) — extended
additions-only (+71/0): the backward's local kernel TRAP (smem pair
re-store on clock64), its final cross TRAP, and the backward runner's
CANARY poison-launch; the transit stages were already trapped (the
verbatim reuse). Artifact mechanism: variant builds in separate build
dirs (env-macro via a torch CUDAExtension monkeypatch wrapper —
setup.py byte-identical), .so copied aside as _C_canary.so/_C_trap.so,
production restored bit-exact from a pre-run backup (EXIT-trap; built
clean if absent). W20.6.3's load contract: ExtensionFileLoader on the
named .so. TRAP's proof is behavioral (G20.6's determinism gate must
FAIL).
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
- [x] done
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
- [x] done
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
- [x] done
Route (b) realized: the SECOND-LAUNCH fold (splitk's control-only
ticket precedent documented but rejected — this module's grep-zero
invariant is stricter; the launch boundary is the only cross-CTA
sync). Association: ascending-j fmaf → xor-butterfly 16→1 → warps
0..7 ascending → partials 0..63 ascending → __fsqrt_rn (fast-math
independent). NOT bit-identical to torch.norm (different tree) —
run-to-run determinism + tolerance gates carry it.
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
- [x] done
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
- [x] done
Composition decision (banner-documented): the "inlined butterfly" is
realized as 7.4's ORCHESTRATION — 5.2's existing (M,K)-batched
butterfly passes run over the unit batch, then the new kernel is a
pure post-butterfly pass (norm broadcast → 7.2's smem scaling →
ladder → pack). Cross stages require launch boundaries (the family's
only cross-CTA sync) — a single fused kernel would need banned
persistent-CTA machinery; re-inlining would fork the
fixture-pinned butterfly. Packing: word = Σ idx[c]<<(c·bits) — the
flat-stream segment as a little-endian integer (the pack_bits_t pin;
b=4 nibble and b=3 general COINCIDE); 16·bits ≡ 0 (mod 8) ⇒ every
byte single-owned (no cross-tile deps); plain shift/mask over PTX bfi
(cited, simpler to verify). codes_lo/codes_hi separate streams. The
launcher refuses n_lo % 16 ≠ 0 loudly (the alignment contract —
misaligned fractions route to the torch twin).
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
- [x] done
Returns std::tuple<codes_lo, codes_hi, norms>. The zero-norm branch =
a separate appended sweep kernel launched LAST (7.3's banner assigned
the branch to 7.4; exact IEEE norms[m]==0; the underflow-degenerate
case is the sweep's real work). Signs passed IN by the dispatcher
(never generated in C++). Open for W20.8: full-length boundary
mirrors even for empty halves; y workspace 2 MiB/unit (chunk if
VRAM-tight).
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
- [x] done
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
- [x] done
The mirror of 7.4: (a) the unpack+LUT kernel (word = Σ src[b0+w]<<8w,
idx = (word>>(c·bits))&mask — the exact inverse of 7.3's pack, bfe
semantics; smem LUT broadcast, zero global gather); (b) fht_backward
large launched VERBATIM (the same kernel the twin's auto-backend
routes to — bit-equal on CUDA); (c) the norms broadcast (one
mul.rn); the single cast at the store. Parity: stage (a) bit-exact
data movement, (b) the same kernel, (c) one mul — documented.
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
- [x] done
The REV-3 routing table (evaluated in order): CPU input → torch
always; kernel pair absent → torch (performance fallback, never
correctness); ineligible → torch (the W15 outlier split, d≤16,384/
non-pow2/%32, n_lo%16, bits∉{3,4}, qjl-ON this wave, non-fp32/fp16
dequant dtypes); CUDA + present + eligible → KERNEL, ALWAYS. The
probe delegates to fht._load_kernel (one extension, one loader) +
the TQ-name hasattr check. Signs mirrored per instance/device (the
_dev_signs idiom); boundaries/centroids passed UNSCALED (the kernel
applies the identity). Mixed-device inputs raise an explicit
RuntimeError naming both devices. NOTE: the binding's actual
signature carries `d` (bindings.cpp:125) — the ledger sketch omitted
it; the code is authority.
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
- [x] done
Twin mechanism: a contextmanager monkeypatch of the dispatcher's ONE
probe (turboquant._tq_kernel_available → False) — the same
TurboQuant instance serves both legs (same signs/codebooks/seed; only
routing flips). The phase-1 gates G20.1/G20.2 run a SHORT finetune
(finetune_m1m2.py --self-test --max-steps 8) and read the history
phase timers; refresh-8 + 300-step legs recorded not-evaluated
(W20.11 full) — never silently passed. G20.1's TQ-share leg is timed
in-process (CUDA events) since phase timers cannot isolate TQ from
the model forward. Remaining stubs: G20.3, G20.5.
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
- [x] done
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
- [x] done
The chains reuse the committed family verbatim (dequant_batched →
residual → norm_rows → fht_forward_large → sign; compensate:
widen → fht_backward_large → scale+add). QJL_C = √(π/2) pinned as
the hex-float constant + static_assert; .rn intrinsics defeat
--use_fast_math; √K via exponent arithmetic (no library sqrt). The
second-draw param is named qjl_rotation (disambiguated from the
first-draw signs).
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
- [x] done
ROLE task-writer · READ `kernel_tq.cu` (7.1's shfl tree) · WRITE
`kernel_tq.cu`, `bindings.cpp`
BRIEF: rows × 256 threads; per-warp shfl.sync tree + the fixed-order
cross-warp fold (the same discipline, smaller tile); the serving
index frame's normalize step; bound and callable.
INVARIANTS: no atomics; fixed fold order.
DO-D: `rg "tq_l2norm_rows" kernel_tq.cu bindings.cpp` hit; atomics 0.
COMMIT: "W20.9.2 — tq_l2norm_rows (the frame normalize kernel)"

#### W20.9.3 — the QJL/l2norm CUDA-gated tests + gate stubs
- [x] done
5 tests: qjl kernel-vs-twin (x_mse allclose 1e-5 + gamma ≤ 1 ulp +
the int8 sketch's tie budget ≤ 0.01% — the faithful form of §4's
codes threshold for a ±1 vector; the compensated output allclose);
l2norm vs torch (3 rungs incl. non-pow2 + fp16; ≤ 1 ulp + the zero
row + a two-call determinism pair); run-to-run; flag-off
non-invasiveness (qjl_signs is None on the default path); device
report. Honest-RED note: if small-N rungs show 2-3 ulps of
tree-association drift on the box, that is the gate owner's call.
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
- [x] done
ROLE orchestrator · standard close-out.
Tier-D: 9.1–9.3 sets.
Tier-G: the QJL gates fold into G20.P3's scope (noted in §4).
DO-D: auditor `--wave W20.9` exits 0; tag `w20-wave-9`; push;
attestation.
COMMIT: "W20.9.4 — wave W20.9 closed: the QJL chain + l2norm
authored"

#### W20.9.5 — box-side fix: the QJL constants' bit-exact rewrite (post-close)
- [x] done
The A10G box's first nvcc 13.2 build (rm -rf build; MAX_JOBS=8
build_ext) failed at the QJL constants, and the handover blamed
"--use_fast_math compile-time FP evaluation" — wrong. W20.9.1's
hexfloat literals were mis-authored: the mantissa INTEGER was pasted
after "0x1." as if it were the fraction (hex digits after the point
carry the fraction x 2^24 = the mantissa left-shifted by one), so
0x1.206C99p+0 is 1.1266... and 0x1.3504F3p+0 is 1.2071... — neither
its sqrt — and the bracket static_asserts correctly REFUSED them: a
static_assert on a false condition is ill-formed. The asserts were
the alarm, not the fire; deleting them to "fix the compile" would
have run the chain at silently wrong constants (1.1266 in QJL_C's
seat, 1.2071 in sqrt(2)'s) — the numerical-catastrophe path the
asserts exist to bar. THE FIX (kernel_tq.cu, the constants block):
the two constants become IEEE-754 integer bit patterns (kTqQjlCBits
= 0x3FA06C99u = fl32(sqrt(pi/2)), kTqQjlSqrt2Bits = 0x3FB504F3u =
fl32(sqrt(2)) — both verified by pure-Python struct round-trip, no
repo import); the brackets become INTEGER comparisons (positive
IEEE floats order exactly as their bits: fl32(1.2533) = 0x3FA06C22,
fl32(1.2534) = 0x3FA06F69, fl32(1.4142) = 0x3FB50481, fl32(1.4143)
= 0x3FB507C8); a __host__ __device__ tq_qjl_bits_as_float serves
both code paths (device: __uint_as_float — a register
reinterpretation, no PTX instruction, folds into the consumer's
immediate; PTX itself spells fp32 constants as their bits; host: the
memcpy idiom, strict-aliasing-safe, folded to a register move). ZERO
compile-time floating-point evaluation remains in the TU — immune to
any fast-math/constexpr-folding behavior in either nvcc pass — and
the runtime bits are IDENTICAL to the twin's fl32 constants by
construction. Riding the same commit (the re-audit's push-
completeness repairs, all box-facing): (a) w20_provision.sh and
w20_build_variants.sh exec bits recorded in git (100644 → 100755 —
a fresh clone could not execute them); (b) the w20-baseline tag
recreated at e9ea357 and PUSHED (the W20.1 original was local-only
on the authoring box and never reached origin — a fresh clone's
auditor --all failed 3/13 before these repairs; 13/13 after). DO-D:
auditor --all exit 0 on the fixed tree; the W20.9 checks (tq_qjl
greps, the absent check, the ledger, the git facts) stay green;
python3 struct-verified bit patterns; no test executed, no CPU test
authored, no repo module imported (the GPU-ONLY law, unchanged).
COMMIT: "W20.9.5 — the QJL constants' bit-exact fix (box-side:
hexfloat authoring bug)"

---

## WAVE W20.10 — the serving-side batched passes (K4 end)

Goal: W20.K4's serving side: the install requant, the index frame
vector builds, the eval harness — all batched GPU passes calling the
same batched entries; faiss receives GPU-produced vectors once at its
boundary.

#### W20.10.1 — the install requant as one batched pass
- [x] done
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
- [x] done
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
- [x] done
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
- [x] done
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
- [x] done
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
- [x] done
Realization: the driver's pair/system token-id tensors built
host-side then pinned (_host_pinned, guarded — plain tensors without
CUDA, values identical); the H2D moved INTO the capture loop
(non_blocking, the pinned sources ride async DMA); train_gates
pre-fetches the NEXT pool's ids during the current refresh cycle (the
staging window); already-device ids pass through as the same objects;
non-tensor id placeholders pass verbatim (the W19 test shims must
stay green). Bit-identical values, batch order, termination, RNG.
Same-stream async (µs-scale copies — a side stream would be
over-engineering, noted).
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
- [x] done
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
- [x] done
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
- [x] done
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
- [x] done
ROLE orchestrator · WRITE the git remote state
BRIEF: final commit (this task's), `git push origin main`, annotated
tag `w20-complete`, push the tag; then — and only per PROMPT.md §8 —
return with the handoff report.
DO-D: `git tag -l w20-complete` non-empty; origin/main == HEAD; the
worklog final entry exists.
COMMIT: "W20.12.3 — W20 complete: pushed, tagged, handed to the A10G
box"

---

## WAVE W20.13 — the box-feedback stabilization (post kernel-compile)

Goal: the A10G box's first full post-compile 237-suite run (225
passed / 35 failed / 3 skipped — the 2026-10-12 handover) decomposed
and fixed at the ROOT on the dev box, one commit per task: THE LOOP's
feedback half. The box's five categories resolved: the pack domain
(15 tests), the rerank device contract (9), the pre-W20 tests'
mechanical torch idioms (the ledger's carried open item, landed at
gate time), the parity-threshold re-derivation (3 kernel-suite
tests — arithmetic, not defects; §4.1 is the record), and the wave
close. The production semantics (m1m2.py, the trainer, the W17/W18
parity contracts) are untouched.

#### W20.13.1 — the pack domain restored (the legacy 1..8)
- [x] done
ROLE kernel-twin owner · READ `turboquant.py` (the W20.3.2 pack
bodies), `evals.py` (the OPEN ITEM note), the box's Category-1
evidence · WRITE `turboquant.py`, `evals.py`
BRIEF: the W20.3.2 {3,4}-only guard broke the evals bits_list=(1,2,
3,3.5,4) sweep and the uniform-bits tests [1,2] — the W20.4.4 open
item, now resolved by the box's evidence. A shared
_check_pack_width domain cop on pack_bits_t/unpack_bits_t (integer
widths 1..8; int-like inputs incl. x.0 floats per the old guard's
float-equality semantics; bools/non-integral/out-of-domain rejected
loudly); the general shift/mask path already covers every width; the
bits==4 nibble fast path unchanged. The KERNEL pack stays {3,4} by
its own TORCH_CHECKs + the dispatcher's _kernel_geometry_eligible —
non-production widths can never reach a kernel.
INVARIANTS: zero behavioral change at {3,4}; the byte layout
width-generic by construction (G20.P1's fixture pin owns it).
DO-D: py_compile green; `rg "_check_pack_width" turboquant.py` hits
twice; the old `bits not in (3, 4)` guards absent; evals' OPEN ITEM
marked RESOLVED.
COMMIT: "W20.13.1 — pack_bits_t/unpack_bits_t: the legacy 1..8
domain restored"

#### W20.13.2 — the rerank device contract (loader.device + the ONE query hop)
- [x] done
ROLE index owner · READ `index.py` (rerank, ChunkVectorLoader,
RetrievalFrame), the box's Category-2 evidence (the live-model CUDA
query vs the npz-edge CPU corpus — the W20.4.3 regression a CUDA-less
dev box could never surface) · WRITE `index.py`
BRIEF: ChunkVectorLoader.device (the reset point's own codes' device,
metadata-only, mixed-device refuses loudly); rerank's frame=None path
hops the ONE query to loader.device (the same hop center_query does
for the frame path; device-less mock loaders keep q put and the
per-candidate belt aligns — a no-op on the real path);
RetrievalFrame.check_loader gains the device belt (named refusal,
mock-safe via getattr). preselect's faiss-edge already crossed
correctly — untouched.
INVARIANTS: numerics unchanged (the fp64 divide, the score order, the
tie discipline verbatim); the dots run where the corpus lives (the
pre-W20 host-era placement).
DO-D: py_compile green; `rg "def device" index.py` hits;
`rg "W20.13.2" index.py` hits (the hop + the belt).
COMMIT: "W20.13.2 — the rerank device contract: loader.device + the
ONE query hop"

#### W20.13.3 — the pre-W20 tests' mechanical torch adaptation
- [x] done
ROLE test-sweeper · READ the box's failure decomposition (beyond its
Category-1 attribution: the np.array_equal/np.all/.copy()/dtype
sites on torch tensors — the open item carried since W20.4) · WRITE
`test_evals.py`, `test_ingest.py`, `test_query.py`, `test_index.py`,
`test_turboquant_split.py`, `test_w16_retrieval.py`
BRIEF: the house idiom (test_tq_cache/test_turboquant's precedent —
torch.equal + a .cpu() belt, np.array_equal cannot cross the device
boundary): ~50 sites swept — array_equal on codes/vectors,
np.all(bool-tensor) (numpy's method dispatch calls Tensor.all(axis=,
out=) — torch rejects: the box's test_end_to_end_mini signature),
Tensor .copy() -> .clone(), the qjl parity's tobytes/np.unique/
np.int8/np.float32 sites, the mmap-root walk's .numpy() hop
(from_arrays KEEPS the memmap memory — the flag still cannot lie),
dtype checks -> torch dtypes, the W20.4.3 loader contract on the
test mocks (torch.as_tensor, zero-copy; the counting mock's call
count untouched), test_query's layout test on device-resident
streams.
INVARIANTS: evidence-driven, not a blind regex — every site still
comparing REAL numpy (the codebooks dataclass, the fixture builders,
the faiss-side brute force, the dict .copy()s) untouched;
comparisons/tolerances/gates verbatim.
DO-D: py_compile green on all six files; the `assert np.array_equal`
residue zero in the swept files.
COMMIT: "W20.13.3 — the pre-W20 tests' mechanical torch adaptation
(the ledger's open item, landed at gate time)"

#### W20.13.4 — the parity contract re-derived (norm 4 ulps + the TIE-AWARE round-trip envelope)
- [x] done
ROLE gate-owner (the dev box IS the gate owner; §1.1: thresholds are
re-derived by task+commit, never masked) · READ the box's Category-4
measurements (d2p22 round-trip beyond 1e-5 with the tie fraction
GREEN; M120/M192 norm 2 ulp > 1), the W20.7.1 design note ("NOT
bit-identical to torch.norm by tree shape") · WRITE
`test_w20_tq_kernels.py`, `scripts/gpu/w20_gates.py`, TASKS.md §4/§4.1
BRIEF: §4.1 is the decision record. (1) norm 1 -> 4 ulps (G20.P1/
P3/G20.4-level2): two different reduction trees, measured max 2 ulps
over 120–192 rows; 4 = 2x the measured max; determinism stays
torch.equal ABSOLUTE. (2) the round-trip recon leg TIE-AWARE: one
admitted flip smears ±Δc·norm/√d over the WHOLE vector via the
adjoint Hadamard — any nonzero tie count breaks per-element
allclose BY ARITHMETIC; the envelope |a−b| ≤ atol + rtol·|b| +
(mm_lo·gap_lo + mm_hi·gap_hi)·norm/√d reduces EXACTLY to the
authored allclose at mm == 0. The isolated decode leg UNCHANGED at
1e-5. G20.P1's envelope is partition-aware (the outlier sub-books
through the codebooks registry; the sub-units' own √n and norms) and
covers the QJL sign-flip source (2·QJL_C·γ/d).
INVARIANTS: ties budget, determinism contracts, decode legs, timer
gates — all UNCHANGED.
DO-D: py_compile green; THRESHOLDS' three norm rows read 4; the test
constant NORM_MAX_ULPS = 4; §4.1 present in TASKS.md.
COMMIT: "W20.13.4 — the parity contract re-derived: norm 4 ulps + the
TIE-AWARE round-trip envelope"

#### W20.13.5 — the wave close (ledger + handover + audit + tag)
- [x] done
ROLE orchestrator · READ TASKS.md §0/§4/§5, the auditor output ·
WRITE TASKS.md, HANDOVER.md, `scripts/w20_static_audit.py`,
`/home/z/my-project/worklog.md`
BRIEF: the §0/§5 rows; HANDOVER's W20.13 section (the box's re-run
sequence — pull ≥ this commit, rebuild NOT needed for the python-side
fixes, re-run the 237-suite, then the protocol's gate sequence);
the auditor's W20.13 CHECKS (mirroring this DoD 1:1); the worklog
entry; `--all` green (14/14); push + tag w20-wave-13.
DO-D: auditor --wave W20.13 exit 0; --all exit 0; the tag pushed;
§5's W20.13 line written.
COMMIT: "W20.13.5 — wave W20.13 closed: the box-feedback fixes
landed, the suite re-armed"

---

## 4. The GPU-box gate registry (statuses: NOT-AUTHORED → AUTHORED → PENDING → GREEN/RED from box reports ONLY)

Thresholds are the gate's source of truth (mirrored 1:1 into
`scripts/gpu/w20_gates.py`'s THRESHOLDS dict — drift is a wave-close
failure). A gate turns GREEN/RED **only** from
`env/w20_gate_reports/<id>.json` produced on the A10G box.

| gate | contract | thresholds | authored in | status |
|---|---|---|---|---|
| G20.P1 | K1: the torch single path vs the frozen fixtures + the 237-suite regression | codes equal except ties ≤ 0.01%; norm ≤ 4 ulps; recon allclose ≤ 1e-5 rel (TIE-AWARE); run-to-run `torch.equal` (CUDA); 237-suite 0 failures | W20.4 | AUTHORED — PENDING (awaiting the A10G box report per env/W20_GPU_PROTOCOL.md) |
| G20.P2 | K2: fht_large differential twins + determinism | kernel vs debug twin vs torch allclose ≤ 1e-5; run-to-run `torch.equal`; K%32≠0 and K≤16,384 refusals raise | W20.6 | AUTHORED — PENDING (awaiting the A10G box report; build the variants first or G20.6 FAILs honestly) |
| G20.P3 | K3+K4: TQ kernel parity (quant+dequant+qjl) | same as G20.P1 thresholds on kernel outputs (norm ≤ 4 ulps, recon TIE-AWARE); batched-grid shapes 120–192 units | W20.8 (qjl folds in W20.9) | AUTHORED — PENDING (awaiting the A10G box report; the qjl leg completes W20.9) |
| G20.1 | capture cycle + TQ share | capture ≤ 10 s/cycle after K3; TQ ≤ 5 ms/prefill (phase timers) | W20.8 (phase-1) | AUTHORED — PENDING (phase-1: short-run history timers; full confirmation after K4/K5) |
| G20.2 | step time + the 300-step run | step ≤ 2 s after K3; ≤ 1 s at refresh 8; 300 steps ≤ 10 min | W20.8 (phase-1) / W20.11 (full) | AUTHORED — PENDING (FULL: two 300-step runs at refresh 4/8, the loop clock end-to-end; honest box cost ≈ 25+ min) |
| G20.3 | W19 convergence (re-run on the kernel path) | loss leaves ln(8) at step 0; gk/gv_absmax > 0; held-out spread positive | W20.11 | AUTHORED — PENDING (the exact history keys: loss/gk_absmax/gv_absmax/cos_pos/cos_neg; the ln(8)=ln(8-batch) tolerance 1e-4, the W19 suite's own) |
| G20.4 | the parity contract (levels 1–2) | level 1: run-to-run `torch.equal` (codes + recon, kernel); level 2: ties ≤ 0.01%, norm ≤ 4 ulps, allclose ≤ 1e-5 (TIE-AWARE) | W20.8 | AUTHORED — PENDING (awaiting the A10G box report) |
| G20.5 | protocol integrity | TRAIN → RE-INGEST → INDEX → EVAL → QUERY on the kernel path; npz digests stable run-to-run; cross-backend decode allclose ≤ 1e-5; W16 verbatim install bit-exact within backend | W20.10 | AUTHORED — PENDING (awaiting the A10G box report; the digest surface excludes only the meta's saved_at provenance field — the format's own determinism surface) |
| G20.6 | canary/trap | canary build: zero poison survives; trap build: FAILS the determinism gate (the negative control must fail) | W20.6 | AUTHORED — PENDING (the variants must be built on the box first — scripts/gpu/w20_build_variants.sh) |

The capture (W20.1) and the 237-suite baseline are protocol steps, not
gates; they are recorded in `env/W20_GPU_PROTOCOL.md`.

### §4.1 — the W20.13.4 threshold re-derivation (the record, never a mask)

The box's first full kernel run (225 passed / 35 failed / 3 skipped,
2026-10-12 handover) measured two authored-threshold failures inside
G20.P3's pytest surface that are ARITHMETIC, not kernel defects, and
the registry rows above were re-derived accordingly (this section is
the decision record; the THRESHOLDS dict and the test constants mirror
it 1:1):

1. **norm 1 → 4 ulps** (`test_batched_grid_shapes_kernel_vs_twin`
   M120/M192: "norm 2 ulp > 1"): the comparison crosses two different
   reduction trees (the kernel's shfl-xor 16→1 butterfly + ascending
   CTA fold vs torch's cub tree). Measured max 2 ulps over 120–192
   rows — the statistical drift of tree association, exactly as the
   W20.7.1 design note predicted ("NOT bit-identical to torch.norm by
   tree shape — the determinism + tolerance gates carry it"). The
   budget: 4 = 2× the measured max. Run-to-run determinism stays
   `torch.equal` — ABSOLUTE, untouched; the norm's downstream effect
   (boundary ties) is carried by TIES_MAX_FRACTION.
2. **the round-trip recon leg becomes TIE-AWARE**
   (`test_geometry_ladder_kernel_vs_torch_twin[d2p22-u1]`: "kernel
   round trip drifts vs twin round trip beyond 1e-5" while the codes
   tie-fraction PASSED): the authored allclose(1e-5) contradicted the
   tie budget by arithmetic — ONE admitted flip changes the
   pre-butterfly stream at one flat position by ≤ gap·norm, and the
   adjoint Hadamard smears it as ±Δ/√d over the WHOLE output vector,
   so ANY nonzero tie count breaks per-element allclose at 2^22 (the
   smaller rungs happened to flip zero ties). The re-derived
   predicate: |a−b| ≤ atol + rtol·|b| + (mm_lo·gap_lo + mm_hi·gap_hi)
   ·norm/√d — mm == 0 reduces EXACTLY to the authored allclose. The
   ISOLATED DECODE leg (the same codes into both decoders) stays at
   the authored 1e-5: no tie involvement, pure rounding drift. G20.P1
   gains the same envelope (per-partition sub-books + the QJL
   sign-flip term 2·QJL_C·γ/d — the sketch's FHT tree ties are a flip
   source of their own).

The ties budget (≤ 0.01%), the determinism contracts (`torch.equal`),
the decode legs (1e-5) and every timer gate are UNCHANGED — the
re-derivation touched exactly the two lines the box measured false,
with the arithmetic written down. The box re-runs the suite + the
gates as the only verdicts there are.

## WAVE W20.14 — the M1/M2 discrimination metric (post-finetune handover)

Goal: the A10G box's M1/M2 gate fine-tune completes (300 steps, 573 s,
no OOM, live gradients, gates moving) but reports cos(pos) ≡ cos(neg)
EXACTLY at every step and held-out spread 0.0000 — "fails to learn
discrimination". Root cause: NOT a learning failure. The held-out
metric `cosine_spread` v1 computed the (B,) TRUE-pair cosine vector and
reported the "negatives" as that SAME vector rolled by one — a roll is
a permutation, and a permutation's mean is identical, so pos == neg
EXACTLY by construction, for ANY model, at ANY step (the 0.615/0.615
... 0.4801/0.4801 table is this identity, not evidence). The LOSS was
always correct (infonce_pairs_loss: the full B×B sim matrix, diagonal
targets, in-batch negatives off-diagonal — unchanged, pinned by the
W17/W19 gates), and the box's below-ln(8) losses (1.40, 0.97, ...) were
real ranking signal the metric could not see. The loss oscillation is
the capture pool cycling (refresh_every=4) sampled at log_every=25 —
25 mod 4 = 1 walks the pool phases; an artifact, not divergence.

Also landed: the data-quality audit of `data/pairs.jsonl` (33,595 rows
— clean: 0 schema problems, 0 duplicate/conflicting/cross-order pairs,
47.9%/52.1% pos/neg balance, uniform decile profile so the positional
val split is sound, 0.1% in-batch text-sharing windows, 3.2% val↔train
exact-text overlap, 13 self-pairs; the 17,519 label-0 rows are UNUSED
by the v1 loader — free eval capacity, recorded below). And the H6
knob (`--train-layers`) for the box's isolate-single-layer experiment.

#### W20.14.1 — cosine_spread: the roll bug
- [x] done
ROLE finetune owner · READ `m1m2_finetune.py` (the v1 metric body),
the box handover's exact-equality table · WRITE `m1m2_finetune.py`
BRIEF: negatives are now the OFF-DIAGONAL mean of the same B×B sim
matrix the InfoNCE trains on (every other pair's b-side) — the metric
scores exactly the discrimination the loss optimizes. Shared
`_pair_sim_stats` helper; the b<2 branch keeps NaN loudness.
INVARIANTS: `infonce_pairs_loss` byte-untouched; the public
cosine_spread signature unchanged (the driver and train_gates call
sites stay).
DO-D: `rg "shifted = sim.roll" m1m2_finetune.py` zero hits;
`rg "_pair_sim_stats"` hits; py_compile green.
COMMIT: "W20.14.1 — cosine_spread: the roll bug (a permutation's mean
is identical — pos==neg at every step by construction)"

#### W20.14.2 — the trainer instrumentation (the box can finally see)
- [x] done
ROLE finetune owner · WRITE `m1m2_finetune.py`
BRIEF: every history entry carries the _pair_sim_stats fields
(cos_pos/cos_neg/spread/off_max/diag_min — the hardest negative and
the weakest true pair ride along) + the pool phase (the 1-based
refresh-cycle position). The log line prints spread + worst-neg +
weakest-pos + pool — the loss-oscillation explainer is now visible
per step.
INVARIANTS: the capture/replay/backward cadence, the W18 exactness
contract, the W19 pool contract all untouched; `del` of the replay
graphs stays after the metrics are read.
DO-D: py_compile green; `rg '"pool": steps_on_batch'` hits.
COMMIT: "W20.14.2 — the trainer instrumentation: _pair_sim_stats into
every history entry (...) + the pool phase"

#### W20.14.3 — the H6 knob: train_layers
- [x] done
ROLE finetune owner · WRITE `m1m2_finetune.py`, `scripts/gpu/finetune_m1m2.py`
BRIEF: GatesTrainConfig.train_layers (None = all) + driver
--train-layers ('all' / '5' / '0-11,16'): the write gates outside the
subset get zeroed grads after backward (BEFORE the |g| read, so the
instrumentation reports the applied gradient; weight_decay=0 default
keeps masked gates bit-anchored). The box's "isolate single layer /
does the shared signal average out" experiment is one flag per run.
Out-of-range indices are a loud refusal.
INVARIANTS: default path (train_layers=None) is behavior-identical to
W20.13.5.
DO-D: py_compile green on both files.
COMMIT: "W20.14.3 — the H6 knob: GatesTrainConfig.train_layers +
--train-layers (...)"

#### W20.14.4 — the box-run regression gates (AUTHORED, PENDING)
- [x] done
ROLE test author · WRITE `src/rag/tests/test_w19_finetune.py`
BRIEF: test_w19f_7 — the rigged batch (true pairs share directions,
cross pairs independent): pos≈1, neg≈0, spread>0.8, off_max<0.6
(v1 reported neg == pos == 1.0 here — the regression pin);
test_w19f_8 — the history fields + pool values + the train_layers
bit-anchor contract (masked layers torch.equal, unmasked moves) +
the out-of-range ValueError. Statuses: AUTHORED on this box, PENDING
on the A10G box (the only place they run).
DO-D: py_compile green; the two test names grep-hit.
COMMIT: "W20.14.4 — the box-run regression gates (AUTHORED, PENDING)"

#### W20.14.5 — the wave close
- [x] done
ROLE orchestrator · WRITE `scripts/w20_static_audit.py` (the W20.14
CHECKS: the roll absent, the stats/grep + py_compile fleet, the
ledger, the five commit prefixes, the tag; PLUS the W20.12 stale
check re-pinned — the box operator's post-finetune push 8e73674
replaced the W20 summary handover with the post-finetune handover,
the retired heading stays a git fact on w20-complete), `TASKS.md`
(this section + the §5 line), `HANDOVER.md` (the box re-run sequence,
rebased atop the operator's handover)
DO-D: `python3 scripts/w20_static_audit.py --wave W20.14` exit 0 AND
`--all` exit 0; tag `w20-wave-14` pushed.
COMMIT: "W20.14.5 — wave W20.14 closed: the discrimination was always
there; the metric can finally see it"

Box open items carried (NOT this wave's scope, recorded for the next):
the 3.2% val↔train exact-text overlap (a dedup pass, if the box wants a
cleaner held-out number); the read_gate contrastive path (needs the
aux-NLL term, the driver's --aux-nll-weight was specced in W17 and
deferred); the label-0 rows as TRAINING hard negatives (their eval
use landed in W20.15 — the curated control line — the training use
needs a masked-InfoNCE loss change, its own wave).

## WAVE W20.15 — the fine-tune regime: batch reconstitution + the honest ceiling instruments (post-fix handover)

Goal: the box re-ran the 300-step gate fine-tune at W20.14.5 — the
metric now sees, and what it saw was a SAWTOOTH: loss block means
0.40..2.98 (chance ln 8 = 2.079; 53/75 blocks below, 22 above), the
BETWEEN-capture-block loss variance 7,595x the within-block variance
(the learning slope -0.003 nats/step drowned 199x by the batch jumps),
pre-clip |gv| swinging 160x with the batch (0.016..2.578 block means),
600 of 14,469 train pairs consumed (4.1% — one epoch never completed),
held-out gap +0.1297 (init +0.080), and the training spread PLATEAUED
at ~0.12 with cos_neg stable (~0.35 background) while cos_pos swings
per batch (0.35..0.74) — the block difficulty IS the batch's own
pairs' cross-side alignment. Root cause decomposition: (R1) the v1
`_cycle_batches` iterator fixed every pair's in-batch negatives to
its file-layout neighbors for the whole run and starved coverage;
(R2) the observability gap — 300 |g| values, zero gate VALUE
locations; (R3) the missing instrument — the write is
position-addressed (slot = p mod mem_size, ONE token per slot at
T < mem_size, so the state delta IS the key sequence and state-cos =
position-ALIGNED key correlation), predicting the discrimination
lives in the surface-aligned pairs; nothing measured that. The wave
lands the regime fix + the instruments; the box's next run decides
between "regime" (fixable by training) and "ceiling" (architectural —
the §2.2 conversation, NOT more gate steps).

#### W20.15.1 — the trainer observability + the regime/instrument library
- [x] done
ROLE finetune owner · WRITE `src/rag/m1m2_finetune.py`
BRIEF: (a) `_gate_value_stats` (gk/gv min/max/absmean, post-update) +
the capture-batch ORDINAL ("batch" — the block id the loss sawtooths
on) into every history entry, the done line printing the final gate
ranges — the box's run logged 300 gradient magnitudes and zero
parameter locations; (b) the library the driver wires next:
`epoch_batches` (per-epoch re-batching — Random(seed*100003+epoch),
the wrap tail full-size and DISJOINT, n <= batch one batch, batch<2 /
bool / single pair loud refusals), `lexical_overlap` (word-multiset
containment — the surface-alignment proxy), `bucketed_spread` (the
per-bucket diagonal against the overall off-diagonal background).
INVARIANTS: the capture/replay/backward cadence, the W18 exactness
contract, the W19 pool contract, the W20.14 field set all untouched
(only additive fields); cosine_spread/infonce_pairs_loss
byte-untouched.
DO-D: py_compile green; `rg '"batch": batch_idx'` +
`rg 'def epoch_batches'` + `rg 'def lexical_overlap'` +
`rg 'def bucketed_spread'` hit.
COMMIT: "W20.15.1 — the trainer observability (the gate VALUE stats +
the batch ordinal) + the regime/instrument library (epoch_batches /
lexical_overlap / bucketed_spread)"

#### W20.15.2 — the driver wiring: the regime, the ladder, the coverage
- [x] done
ROLE finetune owner · WRITE `scripts/gpu/finetune_m1m2.py`
BRIEF: the seeded val split (a random sample, was the file's head
block) + epoch_batches as the train iterator + the consumed-batch
tally (the coverage number the v1 run never printed: 4.1%) +
`--fixed-batches` (the v1 regime A/B) + `--shuffle-seed`; the [5]
report becomes the LADDER — cos(true pairs) vs cos(label-0 curated
pairs) (the 17,519 rows the v1 loader skipped: the datasets' own
non-pairs, a cleaner floor than in-batch randoms; they never enter
training) vs cos(in-batch neg), bucketed by lexical-overlap thirds
(`--val-pairs` 64 default, was a hard 32).
INVARIANTS: the trainer consumes ANY iterator unchanged (the
prefetch contract intact — the generator is endless); `--fixed-batches`
is behavior-identical to v1's batch order; the self-test path
touched only by the empty-text guards.
DO-D: py_compile green; `rg -- '--fixed-batches'` +
`rg 'curated negatives'` (the driver) hit.
COMMIT: "W20.15.2 — the driver wiring: per-epoch re-batching, the
seeded split, the coverage tally, the held-out ladder (true /
label-0 / in-batch) + the overlap buckets"

#### W20.15.3 — the box-run regression gates (AUTHORED, PENDING)
- [x] done
ROLE test author · WRITE `src/rag/tests/test_w20_15_regime.py`
BRIEF: five gates — (a) epoch_batches re-batches per epoch,
deterministic per seed, seed-sensitive, full coverage; (b) the
contracts: full-size batches, the wrap's in-batch distinctness, the
loud refusals (batch<2 / bool / single pair); (c) the
lexical-overlap math (containment, case-blind, multiset); (d)
bucketed_spread on a rigged rig (aligned half vs independent half
must split; gap == bucket diagonal - overall off-diagonal; label
mismatch refuses); (e) the history fields (batch ordinals 1,1,2,2,3
at refresh 2 over 5 steps; gk/gv min/max/absmean present and equal to
the module's actual gates in the final entry). Statuses: AUTHORED on
this box, PENDING on the A10G box (the only place they run).
DO-D: py_compile green; the five test names grep-hit.
COMMIT: "W20.15.3 — the box-run regression gates (AUTHORED, PENDING)"

#### W20.15.4 — the auditor registration
- [x] done
ROLE orchestrator · WRITE `scripts/w20_static_audit.py`
BRIEF: the W20.15 CHECKS block — the function/flag greps + py_compile
on the three files + the five commit prefixes + the ledger row +
the tag (modeled on the W20.14 block; the --all roll-up becomes the
16-wave pass).
DO-D: py_compile green on the auditor; the block's greps verified
against the real targets before commit.
COMMIT: "W20.15.4 — the auditor registration (the W20.15 checks: the
library/flag greps + py_compile + the commit prefixes + the tag)"

#### W20.15.5 — the wave close
- [x] done
ROLE orchestrator · WRITE `TASKS.md` (this section + the §5 line + the
header status), `HANDOVER.md` (the W20.15 box section: the analysis,
the re-run command, the decision tree)
DO-D: `python3 scripts/w20_static_audit.py --wave W20.15` exit 0 AND
`--all` exit 0; tag `w20-wave-15` pushed.
COMMIT: "W20.15.5 — wave W20.15 closed: the sawtooth decomposed; the
regime fixed; the ceiling made measurable"

Box open items carried (NOT this wave's scope, recorded for the next):
the label-0 rows as TRAINING hard negatives (the eval control line
landed; the training use needs a masked-InfoNCE loss change); the
read_gate aux-NLL path (specced W17, deferred); if the box's bucketed
report shows the high-overlap-only signature, the NEXT wave is the
§2.2 slot-addressing conversation (position-locked states — an
architecture question, not a gate-training run).

---

## WAVE W21 — the content-addressed global cache (the position-lock, answered)

The box's W20.15 re-run delivered the predicted signature: mechanics
clean (sawtooth decomposed, coverage tallied, gates moved to
k[-0.087..+1.122]), spread still ~0.12-0.13, cos_neg PARKED at ~+0.36
between RANDOM cross pairs — a representation-anisotropy pedestal plus
the position-locked write (slot = p mod mem_size makes the state
cosine a position-ALIGNED key correlation: surface overlap, not
content; the low-overlap buckets flat, the W20.15 close note's "next
wave" trigger). The run also OOM-crashed INSIDE the held-out control
stage and lost the trained artifact (the 4x 1.07 GiB stacked states
plus the retained capture pool). The mitigation is confined to exactly
the two surfaces the handover allows — the GATES and the GLOBAL
CACHES' write — no model, wiring, quantization, or pipeline change:
content-addressed slots (simhash of the per-(b,h) mean-centered key
over a fixed Sylvester-Hadamard mixing, kappa independent
sign-blocks), centering at the write (the pedestal removed inside the
cache write, train and serve identical), per-(layer, head) write
gates. Paper lineage: Lample et al. 2019 (arXiv:1907.05242, PKM —
content addressing), Wu et al. 2022 (arXiv:2203.08913, Memorizing
Transformers), Su et al. 2021 (centering/whitening), Charikar 2002
(simhash).

#### W21.1 — m1m2.py: the content address + centering + per-head gates
- [x] done
ROLE module-owner · WRITE `src/rag/m1m2.py`: `write_addr`
("content" default / "positional" — the v1 path kept bit-exact for the
A/B), `center_write` (per-(b,h) token-mean removal; T=1 decode writes
its centered zero — the memory freezes at prefill end), `addr_blocks`
(kappa sign-blocks; the stride/bits geometry derived + validated
loudly), `_content_slots` (Hadamard-mixed sign bits — deterministic,
no RNG, the mixing is a plain attribute NEVER a buffer), the content
scatter (per-(head, block) index_add into the (H*mem, D) flat layout),
and the write gates to (num_linear_layers, num_heads) rows (read_gate
stays per-layer). P3 zero-gate no-op holds in BOTH modes; the state
geometry/quantization path/cache glue untouched.
DO-D: `python3 -m py_compile src/rag/m1m2.py` exit 0; `rg -n
"write_addr|_content_slots|_sylvester_hadamard" src/rag/m1m2.py` hits;
`rg -n "num_linear_layers, num_heads" src/rag/m1m2.py` hits (the gate
shape).
COMMIT: "W21.1 — the content-addressed write: simhash slots over the
Hadamard-mixed centered key + per-(layer, head) gates (the position
lock and the pedestal, answered at the cache)"

#### W21.2 — the trainer-side consequences (per-head grads, one sim site)
- [x] done
ROLE trainer-owner · WRITE `src/rag/m1m2_finetune.py`: the W18
Function's backward accumulates PER-HEAD rows (one inner product per
head, sum over slots+dims); the B x B sim matrix moves to ONE public
construction site (`pair_sim_matrix` — the loss, `_pair_sim_stats`,
`bucketed_spread`, and the driver's ladder all read it; the W20.14
"the metric scores what the loss trains" rule, now shared code);
`flat_unit` public; `sim_matrix_stats` / `bucketed_spread_from_sim`
for the ladder's streamed matrices; `load_gates` refuses pre-W21 (L,)
artifacts (shape) and cross-regime artifacts (write_addr meta) loudly,
BEFORE any parameter mutates. Loss/metric semantics unchanged.
DO-D: `python3 -m py_compile src/rag/m1m2_finetune.py` exit 0; `rg -n
"sum(dim=(1, 2))" src/rag/m1m2_finetune.py` hits (the per-head
backward); `rg -n "def pair_sim_matrix" src/rag/m1m2_finetune.py`
hits.
COMMIT: "W21.2 — the trainer side: per-head gate gradients + the one
pair_sim_matrix site (the W20.14 rule by construction)"

#### W21.3 — the driver: addressing flags, artifact-first, the streaming ladder
- [x] done
ROLE driver-owner · WRITE `scripts/gpu/finetune_m1m2.py`: `--write-addr
/ --no-center-write / --addr-blocks` (applied to the shared module
BEFORE the system prefill; the artifact records the regime);
`[3.5]` the data-quality profile (the lexical-overlap distribution of
the positives vs the label-0 controls — "is the data good" answered
with numbers); the gates artifact + history save BEFORE any held-out
report (the box's run lost 575 s of training to a report-stage OOM);
`[6]` the held-out LADDER — one capture, every addressing config
replayed offline from the same k/v (the run's config, kappa 1/2/8,
uncentered content, centered positional, the v1 corner), each row
reporting cos_pos/cos_neg/spread/off_max/diag_min + the overlap
buckets + the label-0 control line; per-pair-side streaming units
(never a stacked (B, dim) state), no_grad replays, del + empty_cache
per config, warn-and-continue try/except per section.
DO-D: `python3 -m py_compile scripts/gpu/finetune_m1m2.py` exit 0;
`rg -n "_ADDR_LADDER|--write-addr" scripts/gpu/finetune_m1m2.py` hits.
COMMIT: "W21.3 — the driver: the addressing A/B flags, the artifact-
first ordering (an OOM never loses the gates), the one-capture
held-out ladder + the data profile"

#### W21.4 — the box-run regression gates (AUTHORED, PENDING)
- [x] done
ROLE test-author · WRITE `src/rag/tests/test_w21_content_addr.py`
(A. order invariance — the permuted multiset lands in the same slots
while the positional state flips; B. the kernel — same content high /
independent low, content vs positional contrast; C. centering removes
the common direction at module level; D. simhash collision
monotonicity; E. per-head gate shapes + one-hot rows + P3 no-op; F.
the T=1 decode zero; G. write_deltas bit-parity + the replay Function
vs plain-chain per-head gradient parity; H. two fresh modules derive
the same slots + the state_dict stays exactly the 3 gates; I. loud
guards + the v1-artifact and cross-regime refusals) + UPDATE the
W17/W18 shape pins (the linspace view, the per-head manual answer)
and test_m1m2 D.5 to pin the POSITIONAL reference explicitly. NOT run
on this box (the no-CPU-test law) — the box runs the suite.
DO-D: `python3 -m py_compile` all four files exit 0; the W21 test
names present.
COMMIT: "W21.4 — the W21 contracts (AUTHORED, PENDING): order
invariance, the kernel contrast, the pedestal, simhash quality,
per-head parity"

#### W21.5 — the wave close
- [x] done
ROLE orchestrator · WRITE `TASKS.md` (this section + the header
status), `SPECIFICATION.md` (§2.2's N6/N7 amendments — the content
address + centering + per-head gates), the auditor registration
(`scripts/w20_static_audit.py` CHECKS W21), `HANDOVER.md` (the W21 box
section: the analysis, the re-run commands, the decision tree)
DO-D: `python3 scripts/w20_static_audit.py --wave W21` exit 0 AND
`--all` exit 0; tag `w21-content-addr` pushed.
COMMIT: "W21.5 — wave W21 closed: the position lock and the pedestal
answered at the cache; the ladder makes the rows measurable"

Box open items carried (NOT this wave's scope, recorded for the next):
the kappa choice is a box decision (the [6] ladder prints the rows —
pick the config with the best low-overlap gap and retrain if it is
not the default 4); the label-0 rows as TRAINING hard negatives (a
masked-InfoNCE loss change); the read_gate aux-NLL path (specced W17,
deferred); if the ladder shows the content rows separating but the
training spread still flat, the next lever is the LOSS temperature or
the hard-negative mining, not the architecture.

## WAVE W21.6 — the box-run repair (the W21 delivery was broken, not the
kernel)

- [x] done
ROLE orchestrator · CONTEXT the box's first W21 run (the pasted
HANDOVER): 17 failed / 258 passed / 3 skipped / 1 error AND the
finetune driver CRASHES at [2] prefill_system — while the W21 kernel
gates themselves PASSED (w21a/b/c/e/f: order invariance, the kernel
contrast, the pedestal, per-head gates, decode freeze). Three
mechanics bugs, none of them the architecture:
  * P0 — `_get_mix` meta crash: `from_pretrained(low_cpu_mem_usage)`
    constructs the model inside accelerate's meta-device context; the
    gates get materialized by the missing-key re-init, the Sylvester
    master — a PLAIN attribute — stays meta with no data. Fix:
    `_get_mix` rebuilds it from the recursion (a pure function of the
    geometry — bit-identical; state_dict still exactly the 3 gates).
    Restores: the 5 loader-test failures + the error + THE FINETUNE
    CRASH.
  * P1 — `torch.index_add` on CUDA accumulates through atomics:
    run-to-run nondeterministic at >= 3 colliders per index; the
    content mode's kappa-block collisions are 8-deep at the test
    geometry. Fix: `_scatter_delta_content` is now a stable-sort
    segment sum (argsort(stable) -> index_select -> fp32 cumsum ->
    unique_consecutive boundaries -> index_copy on distinct rows; no
    atomics on the value path). Restores: write == write_deltas
    bit-exact (w18f_1, w21g), 3 replays bit-identical (w19f_5b),
    memopt forward/grad parity (w18f_2/3), the cross-module slots
    (w21h), the quant-store composability (d6), and w17f_2's
    replay-vs-cache rel-MSE (the box measured 0.4630 — nondeterminism
    amplified by 3.5b quant bucket flips against the 0.12 budget).
    Standalone numpy verification (dev box, no repo imports):
    scripts/w21_6_segment_check.py — the segment sums equal the
    reference scatter within fp re-association (f64 ~1e-13; fp32
    <= 1e-5 at the production geometry — far inside the 3.5-bit
    quant noise), bit-stable across invocations, the (j, n) flat
    layout == src.repeat, empty guard, blocks=1 == ordered
    accumulation.
  * P2 — test-side + library-side defects the W21.4 authoring pass
    missed: (a) `epoch_batches` was a bare generator — the refusal
    body never ran at call time (test_w20_15b DID NOT RAISE); now a
    plain function that validates eagerly and hands back the engine
    (iteration unchanged). (b) `float(gk[0].abs())`-style idioms on
    the new (L, H) gate rows raise ValueError (w18f_3's f-string,
    w19f_3, w19f_8) — now `.abs().max()` (the row magnitude). (c)
    w21d's `.view(1, H, 1, D)` on a (D,) key crashes (16 elements into
    64 slots) — the key broadcasts via (1, 1, 1, D); the independent
    bound was also a coin flip at the any-of-kappa statistic (0.23
    expected vs the 0.25 bound) — now 0.35 with the derivation.
WRITE `src/rag/m1m2.py` (the two fixes + the docstring contracts),
`src/rag/m1m2_finetune.py` (epoch_batches eager), the three test
files, `SPECIFICATION.md` (N7.3), this section + the §5 row, the
auditor's W21.6 block, `HANDOVER.md` (the W21.6 box section).
DO-D: py_compile all five touched files exit 0; `python3
scripts/w20_static_audit.py --wave W21.6` exit 0 AND `--all` exit 0;
the W21 pins held (every W21.1-W21.5 pattern still matches).
COMMIT: "W21.6 — the box-run repair: the meta mix self-heal + the
deterministic content scatter (stable-sort segment sum); the kernel
was right, the delivery was not"

## WAVE W22 — the proven-memory wave: the ladder made runnable, the
curated hard negatives, the sparse read

The box's W21.6 run (the pasted HANDOVER) is the FIRST REAL SIGNAL in
the M1/M2 line — content addressing answered the position lock and the
pedestal: spread +0.20..0.35 sustained over 300 steps (v1: exactly
0.0000), cos(pos) +0.30..0.46 against cos(neg) +0.07..0.16, gates open
k[-0.080..+1.183], zero OOM in training. But the run is NOT closable:
(1) every [6] ladder row OOM'd (16 MiB request against a full 22 GiB —
the device-resident captures + the 4x1.07 GiB unit stacks + their
lists on top of the model), then the script CRASHED with
UnboundLocalError (`del val_group` UNBINDS the name the trailing
report reads) — held-out generalization was never MEASURED; (2) the
loss swung 0.28..1.64 block-to-block on SEVEN random in-batch
negatives while 17.5k curated label-0 non-pairs rode along unused;
(3) coverage 4.2%. The wave adopts the three remaining PROVEN pieces
of the reference key-value-memory lineage (the same lineage W21
cites — no new concept, the RAG flow untouched): the sparse top-k
READ (PKM top-32, Lample et al. 2019, arXiv:1907.05242; Memorizing
Transformers kNN, Wu et al. 2022, arXiv:2203.08913; RETRO, Borgeaud
et al. 2022, arXiv:2112.04426), the curated HARD-NEGATIVE columns
(DPR, Karpukhin et al. 2020, arXiv:2004.04906; MoCo's shared queue,
He et al. 2020, arXiv:1911.05722; Robinson et al. 2021,
arXiv:2011.12031), and the host-staged ladder (the measurement
instrument — bounded GPU footprint, < 1 GiB beyond the model at any
pair count or row count).

#### W22.1 — m1m2_finetune.py: the curated hard negatives in the loss
- [x] done
ROLE trainer-owner · WRITE `src/rag/m1m2_finetune.py`:
`infonce_pairs_loss(..., m1_n=None, m2_n=None)` — N curated negative
COLUMNS appended to the logits (sim(a_i, n_l) never a target; built
through the SAME pair_sim_matrix one-construction site with the
b-operands swapped — the W20.14 metric/loss parity rule holds by
construction); gradients flow into the negatives' replays (push-away
into the same (L, H) gates). `GatesTrainConfig.neg_pairs` (int >= 0,
default 0 = the exact W21 regime bit-identically); `train_gates(...
negatives=...)` — the pool captures neg_pairs curated SIDES per
refresh (staged like pair sides; one-time loud warning + W21 fallback
on iterator exhaustion, never a mid-pool crash); `_pair_sim_stats`
gains cos_hneg (NaN when off); the capture log names the neg sides.
DO-D: `python3 -m py_compile src/rag/m1m2_finetune.py` exit 0; `rg -n
"m1_n|neg_pairs|cos_hneg" src/rag/m1m2_finetune.py` hits.
COMMIT: "W22.1 — the curated hard negatives: N shared label-0 columns
in the InfoNCE (DPR/MoCo's small-batch fix; neg_pairs=0 is the exact
W21 regime)"

#### W22.2 — m1m2.py: read_topk (the sparse retrieval, opt-in)
- [x] done
ROLE module-owner · WRITE `src/rag/m1m2.py`: `read_topk` (None = the
spec §2.2 dense softmax, bit-identical — the DEFAULT; an int k in
[1, mem_size] softmaxes the top-k score row only, the unselected slots
EXACT zero probability — topk -> softmax -> scatter). Validated loudly
at construction AND at read time (bool/0/>mem/non-int raise; the
write knobs' loudness contract). The §7 fine-tune never reads (it
scores WRITTEN states), so gates artifacts stay regime-valid at any
k; the knob is a deployment-time retrieval-quality A/B measured by
the retrieval evals, not the pair ladder. P3 holds at any k (zero
memories read exact zeros).
DO-D: `python3 -m py_compile src/rag/m1m2.py` exit 0; `rg -n
"read_topk|_read_k|torch.topk" src/rag/m1m2.py` hits.
COMMIT: "W22.2 — read_topk: the sparse PKM/Memorizing-Transformers
read (default None = the spec §2.2 dense contract, bit-identical)"

#### W22.3 — the driver: the ladder OOM + the crash, fixed; the feed
- [x] done
ROLE driver-owner · WRITE `scripts/gpu/finetune_m1m2.py`: (a) THE
CRASH — `del val_group` rebinds to None (never del: the box's
UnboundLocalError at the trailing `if val_group is not None`), with
`ladder_ran` carrying the report's condition; (b) THE OOM —
`_capture_group(host_stage=True)` moves every recorded k/v + m_init to
(pinned) HOST at prefill end, `_side_units` streams one side's calls
to the module device and frees them, `_group_units` stacks the four
unit matrices on the HOST, `_sim_from_units` stages only (chunk, dim)
row/col blocks onto the device — the ladder's GPU footprint beyond
the model is bounded at < 1 GiB at any pair count; (c) the [4] residue
release before [6] (iterators, the [2] system cache; gc.collect +
empty_cache — val_pairs stays, [6] captures from it); (d)
`--neg-pairs` (default 8; 0 = the W21 A/B) — the label-0 feed drawn
from the rows EXCLUDING the held-out control slice (the first n_ctrl
after the seeded shuffle), epoch-rebatched at seed+1, capped at 2048
encoded b-sides; the artifact's meta records neg_pairs.
DO-D: `python3 -m py_compile scripts/gpu/finetune_m1m2.py` exit 0;
`rg -n "host_stage|ladder_ran|--neg-pairs|val_group = None"
scripts/gpu/finetune_m1m2.py` hits.
COMMIT: "W22.3 — the box-run repair: the ladder host-staged (bounded
GPU), the UnboundLocalError crash fixed, the --neg-pairs feed (the
control slice stays held out)"

#### W22.4 — the box-run regression gates (AUTHORED, PENDING)
- [x] done
ROLE test-author · WRITE `src/rag/tests/test_w22_hardneg_topk.py`
(A. the default read == the manual dense spec reference bit-exact;
B. k=mem == dense up to softmax order + the top-k support is EXACT
zeros (one-hot-M2 probe turns the read output INTO the prob row),
rows sum to 1, top-1 == score argmax; C. k=1 returns the argmax M2
row EXACTLY; D. P3 at any k — zero memories read exact zeros; E. loud
guards at init AND post-init; F. the loss OFF is bit-identical to the
W21 loss AND the manual cross-entropy; G. a HARD (near-duplicate)
negative raises the loss above an EASY one, gradients reach the
negative states, and 20 SGD steps PUSH APART; H. the loss's guards —
half a pair / N=0 / N-mismatch raise; I. cos_hneg == the shared-site
mean, NaN when off, and the square-metric fields do not move; J. the
trainer with a negatives feed — finite cos_hneg in history, the
exact pop accounting, exhaustion falls back loudly (NaN after),
neg_pairs=0 with a live iterator is BIT-IDENTICAL to negatives=None).
NOT run on this box (the no-CPU-test law) — the box runs the suite.
DO-D: `python3 -m py_compile src/rag/tests/test_w22_hardneg_topk.py`
exit 0; the W22 test names present.
COMMIT: "W22.4 — the W22 contracts (AUTHORED, PENDING): the sparse-
read support/exactness/P3/guards + the hard-negative loss semantics
and the W21-regime bit-parity"

#### W22.5 — the serve-side knob + the wave close
- [x] done
ROLE orchestrator · WRITE `scripts/gpu/run_query.py` +
`scripts/gpu/run_ingestion.py` (`--m1m2-read-topk`, default 0 = the
dense spec contract; set on the module post-load, validated loudly
inside M1M2.read; the SAME k at ingest and query for read
consistency); `TASKS.md` (this section + the §5 row),
`SPECIFICATION.md` (N7.4 — the read_topk amendment), the auditor
registration (`scripts/w20_static_audit.py` CHECKS W22), `HANDOVER.md`
(the W22 box section: the honest verdict on the W21.6 run, the re-run
commands, the decision tree).
DO-D: `python3 scripts/w20_static_audit.py --wave W22` exit 0 AND
`--all` exit 0; tag `w22-proven-memory` pushed.
COMMIT: "W22.5 — wave W22 closed: the sparse read + the curated hard
negatives + the runnable ladder (the three proven pieces, adopted)"

Box open items carried (NOT this wave's scope, recorded for the next):
the kappa choice stays a box decision (the ladder finally RUNS — pick
the config with the best low-overlap gap; the W21.5 note applies); the
read_topk VALUE is a retrieval-eval A/B on the box (start k=32, the
PKM choice at a comparable slot budget; measure with eval_retrieval.py
before serving with it); the loss temperature at the new negative-pool
size (tau 0.07 was tuned for large-batch SimCLR — try 0.1); coverage
(4.2% at 300 steps — the runbook's re-run raises steps, not width).

## WAVE W23 — the generation repair: the proven context build (prefill answer mode)

The user's box paste (run_query on ragga_ingested_test100): retrieval
returns [70, 80, 82] at scores 0.63/0.60/0.59 and the answer is
multilingual word salad. Root cause is NOT the model, NOT the TQ
quantization, NOT the decode loop — the W14/W15 true-doc control
measured conf 0.895 answering over a text-prefilled state on this exact
machinery. It is the §6 INSTALL: a gated-delta-net state is a
SEQUENCE state, S_t = (I − β k kᵀ)·S_{t−1} + β k vᵀ with per-head decay
— NON-ADDITIVE. sum_absolute_codes(sys, [abs₁..absₙ]) = Σabs −
(n−1)·sys is exact arithmetic on the codes but NOT a valid state
composition: it superposes n chunks' associations (each at full
strength, no decay, no conflict resolution) — the linear-attention
readout then returns the superposed mush, and the answer is garbage.
This is the same finding the CacheBlend (EMNLP 2024) / Block-Attention
(2024) line published for KV caches: reused chunk caches still need
the TEXT (or a recompute fraction) to rebuild coherent cross-chunk
state. A second, smaller bug: run_query.py fed the model a 1-D token
tensor (the user's sed hack worked around it — the crash, plus a lost
`.to(dev)`); now the repo's proper fix.

#### W23.1 — query.py: the prefill answer mode (`context_token_ids`)
- [x] done
ROLE core author · WRITE `src/rag/query.py`: `answer_query(...,
context_token_ids)` — one 2-D tensor per retrieved chunk, ascending
corpus order (the ingest stream order). The branch: fresh
`cache_factory()` cache (or the passed cache, reseeded — semantically
fresh), `reseed_cache`, sequential context prefills, the W16
`_reset_full_attention` (docs ride the LINEAR state, full-attn fresh —
spec §2.4), then the existing [7]/[8]. Steps 5-6 (codes load +
install) skipped: `t["context_prefill"]` replaces load_codes/install
in the §9 ledger; `install_report = {"mode": "prefill",
"n_chunks", "context_tokens"}`. Loud refusal: contexts without
retrieved_ids. `None` (default) = the §6 install flow VERBATIM (the
A/B, bit-identical).
DO-D: `python3 -m py_compile src/rag/query.py` exit 0.
COMMIT: "W23.1 — the prefill answer mode: context_token_ids rebuilds
the answer state from the retrieved chunks' text (the proven build)"

#### W23.2 — the m1m2 answer-time A/B (`enabled`)
- [x] done
ROLE core author · WRITE `src/rag/m1m2.py` (`self.enabled = True` — a
runtime flag, never a parameter, never serialized) +
`src/scripts/modeling.py` (the block guard gains
`getattr(_m1m2, "enabled", True)`). The memory's GENERATION effect
(trained gates reading into the stream during prefill+decode) is now
a measured A/B, not an assumption — the true-doc control (0.895) ran
without M1/M2, and the finetune trained retrieval discrimination,
never generation quality.
DO-D: `python3 -m py_compile src/rag/m1m2.py src/scripts/modeling.py`
exit 0.
COMMIT: "W23.2 — the m1m2 enabled knob: the answer-time memory A/B
(the true-doc regime is one flag away)"

#### W23.3 — run_query.py: the 2-D fix + the driver knobs
- [x] done
ROLE driver author · WRITE `scripts/gpu/run_query.py`: query ids are
2-D (1, T) long on the model's device (the proper fix of the box's sed
hack); `--answer-mode {prefill,install}` (DEFAULT prefill — the
proven path); `--corpus` (documents.jsonl, chunk i = line i, the
ingest convention; missing file/lines fail LOUDLY);
`--no-answer-m1m2` (disables the M1/M2 block for the ANSWER phase
only — after the retrieval prefill, so the §4 query vector keeps the
M1/M2 codes); the gen-gate printout (mean conf + repetition, the
bisect's own bar: conf >= 0.20, rep <= 0.6; the true-doc control
measured 0.895).
DO-D: `python3 -m py_compile scripts/gpu/run_query.py` exit 0.
COMMIT: "W23.3 — run_query: 2-D ids (the sed hack's proper fix),
--answer-mode prefill (default), --corpus, --no-answer-m1m2, the gen
gate printout"

#### W23.4 — the box-run regression gates (AUTHORED, PENDING)
- [x] done
ROLE test author · WRITE `src/rag/tests/test_query.py` (the 6e
section): single-context (steps 5-6 skipped — the timings prove the
path; the argmax is the context chunk's topic; the W12 confidences
flow), multi-context sequential composition (majority topic wins —
ordering the install-sum cannot express), no-disk (a loader without
.disk_dir is FINE in prefill mode — the install path refuses that
exact case), the loud refusal.
DO-D: `python3 -m py_compile src/rag/tests/test_query.py` exit 0 (the
suite executes on the box, W22.4's precedent).
COMMIT: "W23.4 — the W23 contracts (AUTHORED, PENDING): the prefill
answer-mode gates"

#### W23.5 — the wave close: docs, auditor, handover
- [x] done
ROLE orchestrator · WRITE `TASKS.md` (this section + the §5 row),
`scripts/w20_static_audit.py` (CHECKS W23), `HANDOVER.md` (the W23 box
section: the honest verdict, the OOM recap, the re-run commands, the
decision tree).
DO-D: `python3 scripts/w20_static_audit.py --wave W23` exit 0 AND
`--all` exit 0; tag `w23-prefill-answer` pushed.
COMMIT: "W23.5 — wave W23 closed: the generation repair — the proven
prefill context build (the install sum stays the A/B)"

Box open items carried (NOT this wave's scope, recorded for the next):
the prefill vs install gen-gate comparison on the real corpus (the
A/B the flags exist for); the M1/M2 on/off answer A/B (one run each
with --no-answer-m1m2); retrieval still needs the re-ingest with the
trained gates (the corpus codes were written under the OLD gates
regime — M1/M2 carry ~zeros; the W22 decision tree's re-ingest step);
the ladder's held-out spread (the W22 box sequence owns it).

## 5. The attestation ledger (the no-CPU-test law, per wave)

One line per wave close, written by the orchestrator, only when
truthful (PROMPT.md §1.4):

| wave | attestation |
|---|---|
| W20.0 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.0 exit 0 (26/26, re-verified post-tag); push w20-wave-0 |
| W20.1 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.1 exit 0 (25/25, re-verified post-tag); push w20-wave-1 |
| W20.2 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.2 exit 0 (24/24, re-verified post-tag); push w20-wave-2 |
| W20.3 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.3 exit 0 (24/24, re-verified post-tag; incl. the module-wide numpy-zero audit line); push w20-wave-3 |
| W20.4 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.4 exit 0 (43/43, re-verified post-tag); push w20-wave-4 |
| W20.5 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.5 exit 0 (27/27, re-verified post-tag; incl. the kernel_fht.cu byte-identical git fact via the new git_diff_empty kind); compile deferred to the A10G box (nvcc ABSENT, recorded); push w20-wave-5 |
| W20.6 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.6 exit 0 (20/20, re-verified post-tag); push w20-wave-6 |
| W20.7 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.7 exit 0 (17/17, re-verified post-tag; the module-wide atomics-free absent-check green); compile deferred to the A10G box; push w20-wave-7 |
| W20.8 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.8 exit 0 (21/21, re-verified post-tag); push w20-wave-8 |
| W20.9 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.9 exit 0 (15/15, re-verified post-tag); push w20-wave-9 |
| W20.10 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.10 exit 0 (17/17, re-verified post-tag); push w20-wave-10 |
| W20.11 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.11 exit 0 (12/12, re-verified post-tag); push w20-wave-11 |
| W20.12 | CLOSED 2026-10-11 — no test executed, no CPU test authored, no repo module imported on this box; auditor --wave W20.12 exit 0 (8/8, re-verified post-tag) AND --all exit 0 (the full 13-wave re-check in one pass); push w20-wave-12 implicit in the w20-complete tag sequence |
| W20.13 | CLOSED 2026-10-12 — no test executed, no CPU test authored, no repo module imported on this box (the g++/py_compile/static-audit vehicles only); auditor --wave W20.13 exit 0 (26/26, re-verified post-tag) AND --all exit 0 (the full 14-wave re-check in one pass); push w20-wave-13 |
| W20.14 | CLOSED 2026-10-12 — no test executed, no CPU test authored, no repo module imported on this box (py_compile/static-audit vehicles + a pure-stdlib jsonl read of the training DATA, no repo code); auditor --wave W20.14 exit 0 (22/22, re-verified post-tag) AND --all exit 0 (the full 15-wave re-check in one pass); push w20-wave-14 |
| W20.15 | CLOSED 2026-10-12 — no test executed, no CPU test authored, no repo module imported on this box (py_compile/static-audit vehicles + a pure-stdlib json/history read of the box-run artifacts, no repo code); auditor --wave W20.15 exit 0 (re-verified post-tag) AND --all exit 0 (the full 16-wave re-check in one pass); push w20-wave-15 |
| W21 | CLOSED 2026-10-12 — no test executed, no CPU test authored, no repo module imported on this box (py_compile/static-audit vehicles only; the uploaded HANDOVER.md training log read as data); auditor --wave W21 exit 0 (36/36, re-verified post-tag) AND --all exit 0 (the full 17-wave re-check in one pass — the W20.15 "curated negatives" vocabulary pin held after the driver rewrite); push w21-content-addr |
| W21.6 | CLOSED 2026-10-12 — no test executed, no CPU test authored, no repo module imported on this box (py_compile + the static auditor only; the pure-numpy segment-sum verification in /home/z/my-project/scripts/w21_6_segment_check.py imports nothing from the repo); auditor --wave W21.6 exit 0 AND --all exit 0 (the W21 pins re-verified post-edit); push main (a62ffa8..W21.6) |
| W22 | CLOSED 2026-10-12 — no test executed, no CPU test authored, no repo module imported on this box (py_compile + the static auditor only; the uploaded HANDOVER.md box log read as data); auditor --wave W22 exit 0 AND --all exit 0 (the full 18-wave re-check in one pass); push w22-proven-memory |
| W23 | CLOSED 2026-10-12 — no test executed, no CPU test authored on this box (the 6e gates authored + py_compile'd only, W22.4's precedent — the suite executes on the box), no repo module imported on this box; auditor --wave W23 exit 0 AND --all exit 0 (the full 19-wave re-check in one pass); push w23-prefill-answer |

Line format: `CLOSED <date> — no test executed, no CPU test authored,
no repo module imported on this box; auditor --wave W20.<n> exit 0;
push w20-wave-<n>`.

## 6. The final closure checklist (the orchestrator's return condition)

- [x] §0: 13/13 waves CLOSED (zero open, zero blocked-undisposed boxes)
- [x] `python3 scripts/w20_static_audit.py --all` exits 0
- [x] §4: every gate AUTHORED (script + thresholds committed; statuses
      honest: PENDING or box-reported)
- [x] §5: 13/13 truthful attestation lines
- [x] `env/W20_ENV.md` + `env/w20_gpu_env.lock` protocol committed
      (the lock is written by the box; its absence is recorded, not
      fatal)
- [x] HANDOVER.md W20 section complete; SPECIFICATION addendum present
- [x] final commit + push; tag `w20-complete` on origin
- [x] the worklog's final entry written (PROMPT.md §8 report follows)

The orchestrator returns ONLY when every box above is checked
(PROMPT.md §8). The A10G box executes `env/W20_GPU_PROTOCOL.md` next.
