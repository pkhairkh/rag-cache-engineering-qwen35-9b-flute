# RAGGA Handover — W19 (the saddle + the capture pool)

## Tests: 237 PASSED

```
python3 -m pytest src/rag/tests -q
237 passed
```

229 (the W18 state) + 8 new W19 gates (test_w19_finetune.py). The W17/W18
parity gates are UNTOUCHED and green — the replay Function's bit-exactness
contracts survived every W19 change.

## The box's three findings, answered

### 1. "The gates are not learning" — ROOT CAUSE: the P3 zero-init is an EXACT saddle of the InfoNCE

The 50-step flat line (loss 2.0794 = ln(8), cos +0.000, gap 0.000) is
EXACT ARITHMETIC, not a weak signal. At write gates = 0:

- every pair-side replays to `m_init + 0 = m_init` — all 16 states in the
  batch are the SAME tensor; with the system reset point built at P3
  gates (the driver's order: load, `prefill_system`, train) the system
  M1/M2 codes are quantized zeros, so `m_init` is EXACTLY zero;
- zero states: `u = 0/||0||_clamped = 0` in the cosine operands, so
  `dsim/dstate = (du/dstate)^T · u_other = 0` exactly — the huge clamped
  1/1e-12 Jacobian multiplies a ZERO vector;
- identical NONZERO states (an open-gate system state): every cosine sits
  AT its maximum (`cos(v,v)`, local gradient 0) AND the cross-entropy row
  gradient `(p - y)` sums to exactly 0 over each row, so the shared
  direction cancels exactly;
- the write path's LOCAL Jacobian `dm/dg = Delta` is nonzero (P3's
  branch-free add — the W17 docstring's claim is true LOCALLY), but the
  gate gradient is `<dL/dm, Delta>` and `dL/dm = 0` at the saddle.

So `write_gate_k.grad` is IDENTICALLY zero — exactly what the box
suspected ("check if write_gate_k actually has non-zero gradients" — it
does not, and cannot, from that init). Pinned permanently by
`test_w19f_1` (grads `torch.equal` zeros; 3 AdamW steps leave the gates
at exactly 0.0) and `test_w19f_1b` (the open-gate-system regime is the
same cancellation). The W17 test that "proved" learning worked
(test_w17f_5) manually `fill_(0.1)`-ed the gates before training — the
driver never did; the tests covered the fixed case, not the shipped
path.

### 2. "22 s/step" — ROOT CAUSE: the 16 capture prefills, ~95% of the wall clock

The replay forward + the W18 backward are ~0.5 s combined (24 calls x 16
sides of small scatter/dot ops). The cost center is the CAPTURE: each
prefill runs the production online path, and the wiring pushes
S + M1 + M2 through ~50 TurboQuant quants of 2^19-wide units per forward
(24 S updates + 24 M1 + 24 M2 pushes at `update_m1/update_m2` per layer)
— and d = 524,288 > 32,768 is NOT CUDA-FHT-eligible, so every one of
those runs the 19-stage torch-reference butterfly on device.

The qlora repo's `flute_train_kernels` (fused backward GEMM +
dL/dLUT scatter) were examined for the "reuse the backward kernel"
question: they are the WEIGHT-side LUT-GEMM backward
(`grad_X = grad_Y @ W_dequant`, dL/dLUT over codebook codes) — a
different op family from the replay's state-side slot-scatter. No
literal kernel reuse. What transfers is the fusion discipline
(one deterministic launch, fixed association order) — noted for the
W20 candidate below, where it actually applies.

### 3. "Read-gate not trainable" — confirmed an architecture gap, not a bug

The contrastive loss never sees `read_gate` (the state is write-built).
Still frozen by default (`freeze_read_gate=True`); training it needs a
next-token auxiliary objective through the LIVE read path (a full
differentiable forward — the OOM-cost regime). NOT implemented this
wave; the two candidate designs are in "Open for the next wave".

## The W19 fixes (all trainer-side; the production m1m2.py semantics are UNTOUCHED — P3 default, zero code change)

1. **`gate_init` (the saddle escape)** — `GatesTrainConfig.gate_init`
   (default 0.1, driver `--gate-init`) fills both write gates before
   step 0. Any nonzero value escapes (the cosine is scale-invariant —
   the value only sets where optimization starts). `--gate-init 0` keeps
   the loaded values and the loop prints the loud saddle WARNING.
   `test_w19f_2/3`: the escaped loop has LIVE grads (|g| ~ 1e+0 at the
   stub scale) and DESCENDS the InfoNCE below ln(B) - 0.5 with the
   signal layer out-weighting a planted common component.
2. **`delta_loss` (the loss geometry)** — default True: the InfoNCE
   scores `(m_final - m_init)` per side — the SAME deltas the W16
   centered retrieval frame scores at serving time (the absolute form
   would let a nonzero system state dominate every direction; W16's
   common-component lesson applied to the trainer). A/B:
   `--absolute-loss`. `test_w19f_4`: the history loss equals the manual
   delta-InfoNCE bit-for-bit; False recovers the absolute form.
3. **`refresh_every` (the capture pool)** — default 4: ONE batch of
   pairs prefilled, then trained on for 4 GRADIENT STEPS. The replay
   re-runs every step, so each gradient is taken at the CURRENT gates;
   only the captured k/v freeze (detached constants — the W18 gradient
   contract is untouched; staleness bounded by lr x (N-1) gate
   movement, and the read-path feedback the k/v feel is second-order
   small). Expected wall clock: ~22 s/step -> ~6 s/step at
   refresh 4 (the phase timers will confirm). `test_w19f_5/5b`:
   refresh 2 halves the prefill count exactly; pool reuse replays
   bit-identically.
4. **Instrumentation** — every history entry now carries
   `gk_absmax`/`gv_absmax` (the box's exact ask:
   `write_gate_k.grad.abs().max()`, recorded pre-clip) +
   `capture_s`/`replay_s`/`backward_s` (per-phase cumulative timers).
   The step log prints them; the run's final line prints the phase
   totals + the capture share. The 22 s/step black box is now
   measurable ON THE BOX — if capture is NOT ~95% of it there, the
   timers will say what is.

## The driver changes (`scripts/gpu/finetune_m1m2.py`)

New flags: `--gate-init` (0.1), `--refresh-every` (4),
`--absolute-loss`. The [W19] line prints the active regime; the step log
carries |g| and the phase times; the held-out spread [5] now scores the
DELTAS (the centered frame's geometry — cos of full states is ~+1
trivially off the shared m_init).

## GPU box: the W19 confirmation sequence

```
# same pairs file as the W18 run (33,595 pairs, 16,076 positive — build_pairs is fixed and green)
python3 scripts/gpu/finetune_m1m2.py --pairs-file pairs.jsonl \
    --gates-out disk/m1m2_gates.npz --m1m2-mem-size 1024
```

Read the run off the new instrumentation:
1. step 0 log: loss should leave ln(8)=2.0794 IMMEDIATELY (any movement
   = the escape worked); |g| should be > 0 from step 0;
2. cos(pos) vs cos(neg) should SEPARATE within ~25 steps (the delta loss
   + the escape). If loss still sits at 2.0794 with |g| > 0, that is a
   NEW bug — file the history json;
3. the final phase line: expect capture ~90-97% of the time (confirming
   the diagnosis). At refresh 4 the amortized step should be ~6 s; if
   acceptable, A/B `--refresh-every 8` (~3 s/step, staleness lr x 7);
   if quality degrades at higher refresh, drop back;
4. 300 steps at ~6 s/step ≈ 30 min (was ~110 min);
5. after training: the deployment protocol is UNCHANGED — save ->
   RE-INGEST with the gates (`--m1m2-gates ... --m1m2-mem-size SAME`) ->
   run_index -> eval_retrieval (the M1/M2 units now carry trained
   discrimination) -> run_query.

## Open for the next wave (measured/diagnosed, not fixed)

- **The S/M1/M2 online-quant FHT is the throughput ceiling**: d = 2^19
  units run the 19-stage torch-reference butterfly (~50 per prefill).
  A CUDA FHT extension to multi-block K > 32,768 (the qlora repo's
  `kernel_fht.cu` pattern, flute_extended is frozen — a deliberate
  fork) would cut capture cost at the source, compounding with the
  pool. This is where the qlora repo's kernel discipline actually
  applies.
- **read_gate training**: needs the aux next-token objective through
  the live read path (candidate designs: (a) a short differentiable
  decode window over the replayed state — bounded activations, not the
  full prefill; (b) a distillation target from the frozen no-read
  output). The gates artifact already roundtrips read_gate values.
- **--aux-nll-weight** is referenced in m1m2_finetune's docstring but
  was never plumbed (W17 aspirational text); do not cite it until
  implemented.
- Far-future cross-project lever: the capture prefills could run on the
  FLUTE-palettized build of the same base model (the qlora repo: 0.32x
  weight bytes, 5.4 GiB streams) — a model-loading integration, NOT a
  trainer change; only worth it if the pool + FHT work is still not
  enough.

## Files changed this wave

- `src/rag/m1m2_finetune.py` — the saddle proof (docstring),
  `gate_init`/`delta_loss`/`refresh_every`, the capture pool loop, the
  phase timers + grad instrumentation in history.
- `scripts/gpu/finetune_m1m2.py` — the three new flags, the [W19]
  regime line, the delta-aware held-out spread.
- `src/rag/m1m2.py` — docstring ONLY (the W19 saddle caveat on the P3
  paragraph; zero production code change).
- `src/rag/tests/test_w19_finetune.py` — 8 gates (the saddle, the
  escape, descent with a planted common component, the delta wiring,
  the pool counts + bit-identity, the history fields).

---

# W20 in execution — the TQ kernel family (PROPOSAL.md rev 2)

W20 runs as a wave-centric, sub-agent orchestration from the CPU dev
box. The binding contracts, in authority order: SPECIFICATION.md >
PROPOSAL.md (rev 2, the KERNEL directive) > **PROMPT.md** (the W20
standing orchestration prompt — the GPU-ONLY law: no CPU test runs, no
CPU test authoring, no repo-module imports on the dev box) >
**TASKS.md** (the execution ledger: 13 waves, 58 tasks, per-wave DoD)
> this file (the human narrative). Machine state: the un-committed
worklog on the dev box.

## W20.0 — environment + audit infrastructure (CLOSED 2026-10-11)

- `env/W20_ENV.md` — the CPU box provisioned at latest stable,
  evidence-backed (`pip index versions` outputs embedded): torch
  2.14.1 (the default PyPI linux wheel = the CUDA-bundled cu130 build,
  importable without a device — matches the A10G authoring semantics),
  numpy 2.5.4, transformers 5.19.0, faiss-cpu 1.15.1, ninja 1.13.2;
  nvcc ABSENT (driver-less unobtainable) → compile checks defer to the
  A10G box.
- `scripts/gpu/w20_provision.sh` — the A10G operator's entry point:
  sm_86 assert, unpinned latest-stable installs, toolkit fallback,
  flute_extended build (FLUTE_CUDA_ARCHES=86), the 237-suite baseline
  banner, `env/w20_gpu_env.lock` writer. Idempotent.
- `scripts/w20_static_audit.py` — the Tier-D DoD vehicle (parses,
  never imports): exists/grep/absent/ast_def/py_compile/git_tag/
  git_log/ledger kinds, CHECKS per wave mirroring TASKS.md DoD 1:1.
- `src/rag/tests/w20/_gpu_guard.py` — the CUDA-only authoring surface:
  require_cuda() loud module-level skip, device_report() (warn != 8.6,
  raise major != 8), gate_report_path() → env/w20_gate_reports/.

Wave DoD: Tier-D green (26/26, auditor --wave W20.0 exit 0 post-tag);
commits W20.0.1–W20.0.5 on origin; annotated tag `w20-wave-0` pushed.
Tier-G: none in this wave. No test executed, no CPU test authored, no
repo module imported on the dev box (the attestation ledger, TASKS.md
§5). Next: W20.1 — the baseline tag + frozen probe fixtures.

## W20.1 — baseline tag + frozen probe fixtures (CLOSED 2026-10-11)

- `w20-baseline` (annotated tag) → e9ea357 — the numpy-era runtime
  tree pinned BEFORE the rewrite (W20.0 was infra-only; runtime
  byte-identical to 9af04fe). This is the fixture-capture checkout.
- `scripts/gpu/w20_capture_fixtures.py` — the capture vehicle,
  refusal-first: pinned-checkout assert (exit 2) → CUDA assert
  (exit 3) → only then the pinned repo imports. 14 seeded probes
  (G1 d=2^19 ×4, G2 d=24,576 ×4 the conv geometry with the W15
  outlier-split, G3 d=2^22 ×2; mse/prod(=qjl)/b∈{3,4}/3.5-bit split
  covered; source-true deviations pinned in the TASKS.md ledger).
- `env/W20_GPU_PROTOCOL.md` — the A10G operator's five-step sequence
  (provision → capture → commit-from-box → per-wave gates → final
  registry run), with the §4 gate registry transcribed 1:1.
  `env/w20_gate_reports/` gitignored (box-written, never committed).

Wave DoD: Tier-D green (25/25, auditor --wave W20.1 exit 0 post-tag);
Tier-G: the capture is AUTHORED, PENDING until the A10G box runs it
(protocol step 2). Attestation truthful (no test executed, no CPU
test authored, no repo module imported on this box). Next: W20.2 —
the TQCodes device contract + codebook mirror.

## W20.2 — the codes contract is torch-native (CLOSED 2026-10-11)

- `src/rag/tq_codes.py` — TQCodes by construction: norm is a 0-dim
  fp32 torch tensor, idx_lo/idx_hi device-resident uint8 tensors, the
  W15 outlier trio + W9.2 qjl pair as optional tensors;
  to(device)/device/nbytes()/sync-free repr; to_arrays/from_arrays are
  THE npz disk edge (numpy structurally confined to those two bodies;
  the legacy key set/dtypes/conditional presence replicated 1:1 — old
  readers and old digests unaffected).
- `src/rag/turboquant.py` — the old numpy TQCodes class removed;
  house flat re-export at :118 keeps `from turboquant import TQCodes`
  symbol-compatible (install/tq_cache/snapshot untouched imports).
- `src/rag/codebooks.py` — the OFFLINE-NUMPY-ISLAND markers (solve +
  quadrature + npz cache byte-identical, diff-verified) + the runtime
  mirror: get_codebook(device=...) → the DeviceCodebook view
  (boundaries/centroids fp32 tensors, identity-keyed
  _DEVICE_TENSOR_CACHE, .numpy_codebook for the digest edge).
- Staged breakage (intentional, resolves in W20.3): quant() still
  constructs with numpy → the torch-asserting constructor refuses;
  nothing executes until the A10G box gates.

Wave DoD: Tier-D green (24/24, auditor --wave W20.2 exit 0 post-tag);
Tier-G none (parity is W20.4's G20.P1); attestation truthful. Next:
W20.3 — the turboquant.py torch rewrite (the core, the six CPU
syndromes).

## W20.3 — turboquant.py torch rewrite, the K1 core (CLOSED 2026-10-11)

The six CPU syndromes are dead, one per task: the `.item()` norm sync
(3.1 — norm is a 0-dim fp32 tensor end-to-end), the numpy searchsorted
(3.1 — torch.searchsorted with the boundary-scaling identity: argmin
|y/n − c| ≡ argmin |y − n·c|, the normalized vector never materialized;
ties np-left ≡ torch right=False), numpy packbits (3.2 — pack_bits_t/
unpack_bits_t, byte layout fixture-pinned), the numpy gather + the
from_numpy adjoint butterfly (3.3 — torch LUT gather + fht_adjoint
device-native; the degenerate zero-norm decodes via the 0-dim multiply),
and the QJL chain (3.4 — torch.sign(FHT(r)) with the scale-invariance
identity; the compensation device-native; the bridge collapsed).

**THE AUDIT LINE**: `rg "np\.|numpy" src/rag/turboquant.py` == 0
module-wide — numpy exists NOWHERE in the K1 core (the two sanctioned
offline edges: TQCodes.to_arrays/from_arrays + the codebooks solve
island). The `.item()`/`.cpu()`/`.numpy()`/from_numpy greps: 0.

3.5 reconciled the module: the rev-2 runtime contract docstring (the
single device-parametric path; the DIFFERENTIAL TWIN role — kernel-first
dispatch foreshadowed for W20.8), all five construction sites verified
torch-clean, the consumer surface intact (tq_cache/install/index/
snapshot/evals keep their imports; the mechanical test sweep is W20.4's).

Carried to W20.4 (open items, recorded in the TASKS ledger): the
pre-W20 test files' dtype adaptations (pack tests + evals qjl host-
dtype assertions); the pack domain bits∈{3,4} blast radius (uniform-
bits tests at b∈{1,2}; evals bits_list); G20.P1 is the behavioral
authority for all of it. Wave DoD: Tier-D 24/24 post-tag; Tier-G none
(G20.P1 lands W20.4). Attestation truthful. Next: W20.4 — wiring +
serving sweep + the K1 gates.
