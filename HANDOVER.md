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
6. W20 — the refresh-8 A/B row (step 3's complement; now a first-class
   regime — the W20.11 lever): after the refresh-4 run reads clean, run
   the SAME command with `--refresh-every 8` — the throughput regime.
   The staleness bound is the tight one: lr x (N - 1) = lr x 7 (the
   frozen k/v trail the gates by at most 7 optimizer steps; the replay
   re-runs per step, so the gradients themselves stay current). Judge
   it on the same step-log gauges — loss off ln(8) at step 0, |g| > 0,
   cos(pos)/cos(neg) separating by ~25 steps; if the separation is
   materially worse than the refresh-4 run, drop back. The default 4 is
   quality-first; the regime is the operator's call on the box (G20.2's
   refresh-8 targets: step <= 1 s, 300 steps <= 10 min end-to-end).

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

## W20.4 — K1 COMPLETE: the single torch path, the gates authored (CLOSED 2026-10-11)

The 103-touchpoint elimination is landed and evidenced: the whole
runtime (turboquant, tq_cache, install, index, query, ingest, m1m2,
evals) is torch device-parametric — `env/w20_cpu_elimination_audit.md`
(rg-derived, 17 module rows): **112 audited touchpoints → 0 outside
the allowlist** (DISK-EDGE 143 across the npz codec surfaces;
FAISS-EDGE exactly 4 in index.py; OFFLINE-ISLAND 93 in codebooks;
REPORT-EDGE 2 comment-only). FINDINGS: none.

G20.P1 is AUTHORED — PENDING, honestly: `scripts/gpu/w20_gates.py`
(599 lines: THRESHOLDS mirroring the §4 registry 1:1, G20.P1 fully
implemented — fixture parity [ties ≤ 0.01%, norm ≤ 1 ulp, recon
allclose 1e-5, run-to-run torch.equal] + the 237-suite subprocess on
the box; the 8 later gates as loud NOT-AUTHORED stubs) +
`src/rag/tests/w20/test_w20_torch_path.py` (4 CUDA-gated,
fixture-driven tests; manifest-absent = module skip). **The A10G box
runs G20.P1 next** (env/W20_GPU_PROTOCOL.md steps 2-4: capture the
fixtures first, then `--gates G20.P1`).

The K1 open items (the box's plate, recorded in the ledger): the
pack-domain decision (bits ∈ {3,4} vs the legacy 1..8 — evals'
bits_list + the pre-W20 uniform-bits tests constrained mechanically,
G20.P1 arbitrates); the sibling test files' np.array_equal sites
(mechanical adaptation at gate time); the qjl=True path's fp32
prefactor (last-ulp drift vs legacy host float64 — the parity gate
covers it). Wave DoD: Tier-D 43/43 post-tag; attestation truthful.
Next: W20.5 — fht_forward_large / fht_backward_large (K2 begins).

## W20.5 — fht_forward_large / fht_backward_large, K2 begins (CLOSED 2026-10-11)

The multi-block butterfly for K = 2^19…2^22 is authored:
`include/flute/fht_large.cuh` (the math contract: 13 tile-local
stages at the half-tile line — the PROPOSAL's canonical 13/6/9, the
ledger's off-by-one resolved and recorded; sign conventions cited;
the NO-ATOMICS ownership contracts); `src/kernel_tq.cu` (born, 941
lines: pass 1 tile-local via kernel_fht.cu's own fht_block primitive
— the pairing discipline by construction, cp.async staging; pass 2
cross stages one-launch-per-stage; the adjoint with signs at first
load + rsqrtf(K) at the final store, transit stages reused
orientation-free; 5 kernels all __launch_bounds__(1024), zero
atomics; the CANARY/TRAP negative-control scaffolding, default OFF);
`bindings.cpp` + `setup.py` (the pair bound, kernel_tq.cu always
built, the variant-build guidance); `fht.py` (the three-band
eligibility table: ≤16,384 single-block VERBATIM — **git-verified
byte-identical via the auditor's new git_diff_empty kind**; the
pow2 large band when the extension is built; everything else
reference/loud-refusal; the 24,576 conv window untouched via the
segments machinery).

Compile + run defer to the A10G box (nvcc ABSENT on this box —
recorded): the box's provision step builds kernel_tq.cu with
FLUTE_CUDA_ARCHES=86, then G20.P2 (W20.6) is the differential
authority. Tier-G: G20.P2's twins land in W20.6 (this wave closed on
Tier-D only — noted in the ledger). Attestation truthful. Next:
W20.6 — the debug twins + canary/trap builds + the K2 gates.

## W20.6 — K2 COMPLETE: the twins, the negative controls, the gates (CLOSED 2026-10-11)

The differential surfaces: the scalar debug twins
(tq_debug_fht_large_forward/backward — flat pair loops in the kernels'
exact enumeration order, the G20.P2 references); the canary/trap
build variants (the backward's coverage armed — the 5.3 ledger note
closed; `scripts/gpu/w20_build_variants.sh`: env-macro variant builds
with .so artifacts set aside and production restored bit-exact from
backup); the CUDA-gated suite (test_w20_fht_large.py: 5 tests —
ladder parity vs twin+reference, run-to-run determinism, the
refusals, canary zero-poison + trap-must-fail via fresh-subprocess
variant probes).

G20.P2 + G20.6 are AUTHORED — PENDING: the harness's gate_g20_p2
(per-rung twin+reference parity + per-rung run-to-run determinism +
the refusals) and gate_g20_6 (canary zero-poison; trap must fail;
absent variants = honest FAIL with the build pointer — a gate is
never skippable). **The box runs G20.P2 + G20.6 after building the
variants** (`bash scripts/gpu/w20_build_variants.sh canary` then
`trap`, then `--gates G20.P2,G20.6`).

Wave DoD: Tier-D 20/20 post-tag; attestation truthful. K2 is
complete. Next: W20.7 — tq_quant_batched (K3, THE wave: the fused
encode kernel).

## W20.7 — tq_quant_batched, THE wave's encode half (CLOSED 2026-10-11)

The fused encode is authored (kernel_tq.cu now ~2,800 lines, every
piece additions-only, the module atomics-free): the deterministic
norm pass (7.1 — the shfl-xor butterfly 16→1 + the second-launch
ascending fold; NOT bit-identical to torch.norm by tree shape — the
determinism + tolerance gates carry it); the boundary-scaling
identity in-kernel (7.2 — the ≤15 scaled boundaries, the in-register
linear compare ladder, searchsorted-left tie parity); the
ladder+pack pass (7.3 — the word-shift packing IS pack_bits_t's
little-endian flat-stream pin; every byte single-owned); the batched
entry (7.4 — tq_quant_batched: norm → the batched butterfly →
ladder+pack → the zero-norm sweep; ONE host call, zero D2H, the
6,144-CTA / ~76-waves budget line; signs passed in by the coming
dispatcher).

Composition note (banner-documented): the "inlined butterfly" is the
entry's orchestration over 5.2's proven (M,K)-batched kernels —
re-inlining would fork the fixture-pinned butterfly. W20.8's plate
(recorded): the dispatcher supplies the signs + full-length boundary
mirrors; the y workspace is 2 MiB/unit (chunk if VRAM-tight);
G20.P3's quant half lands with dequant next wave. Tier-D 17/17
post-tag; attestation truthful. Next: W20.8 — tq_dequant_batched +
the kernel-first dispatcher (K3 end — TQ leaves the profile).

## W20.8 — K3 COMPLETE: the decode + the dispatcher, TQ leaves the profile (CLOSED 2026-10-11)

tq_dequant_batched (the mirror of quant: bfe-semantics unpack — the
exact inverse of the pack — the smem LUT broadcast with zero global
gather, fht_backward_large launched verbatim, the norms broadcast,
one cast); the kernel-first dispatcher (REV-3 routing: CUDA +
eligible → the KERNEL path, ALWAYS; CPU/absent/ineligible → the torch
twin — the performance fallback, never correctness; the W15 outlier
split and the 24,576 conv window stay twin-side; mixed-device inputs
raise by name); the CUDA-gated TQ kernel suite (7 tests — the ladder
vs the twin via the one-probe monkeypatch, the batched-grid shapes,
run-to-run determinism, the refusals).

**7 of 9 gates are now AUTHORED — PENDING** (G20.P1, P2, P3, P4, G6 +
the phase-1 G20.1/G20.2 whose full legs evaluate at W20.11; the
remaining stubs: G20.3 → W20.11, G20.5 → W20.10). The two
kernel-chains-per-prefill design is live in code: the ~4,000 host
round trips are structurally gone on the CUDA path. The box runs
G20.P3/P4 + the phase-1 timers per the protocol (the finetune
short-run history; the TQ-share CUDA-event timing).

Wave DoD: Tier-D 21/21 post-tag (incl. turboquant's numpy-zero still
0 and the module-wide atomics-free check); attestation truthful.
Next: W20.9 — the QJL kernel chain + tq_l2norm_rows (K4).

## W20.9 — the QJL chain + tq_l2norm_rows (CLOSED 2026-10-11)

The Alg.-2 kernel chain, flag-off and complete: tq_qjl_quant (the
residual rides the fused decode verbatim → the 7.1 norm pass for γ →
fht_forward_large → the trichotomy sign kernel; the scale-invariance
identity, QJL_C = √(π/2) pinned as the hex-float constant, .rn
intrinsics against --use_fast_math) and tq_qjl_compensate (widen →
fht_backward_large → the exact twin scaling chain). tq_l2norm_rows
(the serving frame's normalize: one CTA per row, the 7.1 association
discipline, branchless zero). The CUDA-gated suite (5 tests: the qjl
parity with the tie budget as the faithful codes threshold, l2norm ≤
1 ulp, determinism, flag-off non-invasiveness).

Tier-G: the QJL gates fold into G20.P3's scope (noted in §4). Wave
DoD: Tier-D 15/15 post-tag; attestation truthful. Next: W20.10 — the
serving-side batched passes (K4 end, G20.5).

## W20.10 — the serving side batched, K4 COMPLETE (CLOSED 2026-10-11)

The serving sites ride the batched entries: install's requant folds
(one chain per unit-kind geometry — the W16 copy paths untouched,
the CPU legacy fold verbatim for non-kernel boxes); index's frame
vector builds (grouped by quantizer identity → one batched dequant
per group; the frame normalize via tq_l2norm_rows with the torch
fallback; the 4 FAISS-EDGE hops unchanged — faiss fed once at the
boundary); evals' sweep blocks (one quant + one dequant over the
whole block, the per-vector mse a single device reduction; the CPU
fallback verbatim).

G20.5 AUTHORED — PENDING: the protocol-integrity gate (TRAIN →
RE-INGEST ×2 → INDEX → EVAL → QUERY as subprocess stages; the npz
digest sets stable run-to-run — the digest surface excludes only
meta's saved_at, the format's own determinism surface; the
cross-backend decode allclose 1e-5; the W16 verbatim install's
digest equality within the backend). 8 of 9 gates authored (G20.3
remains — W20.11). Wave DoD: Tier-D 17/17 post-tag; attestation
truthful. Next: W20.11 — the trainer levers (K5).

## W20.11 — K5 COMPLETE: the trainer levers, all 9 gates authored (CLOSED 2026-10-11)

Refresh 8 is a first-class A/B regime (the tight staleness bound
lr×7 documented in the help + the config + HANDOVER's confirmation
sequence step 6; default 4 stays quality-first); the capture loop's
loader is pinned (host-pinned token-id tensors, non-blocking H2D
inside the capture loop, the next pool's ids pre-fetched during the
current refresh cycle — transport, not math: bit-identical values).

The LAST two gates are authored — **9/9 AUTHORED, all PENDING the
box**: G20.2 FULL (two 300-step runs at refresh 4/8; the step
thresholds + the loop-clock 10-min end-to-end; honest box cost ≈
25+ min green) and G20.3 (the W19 convergence shape re-run on the
kernel path: the loss leaving ln(8) at step 0, gk/gv_absmax > 0, the
held-out spread — the exact history keys pinned in the registry).
Wave DoD: Tier-D 12/12 post-tag; attestation truthful. Next: W20.12
— closure (the handover + the final all-DoD audit + the tag).
