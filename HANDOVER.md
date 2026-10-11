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
