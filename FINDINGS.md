# FINDINGS.md — what truly makes the query pipeline produce garbage

**Scope**: the §6 query flow (`scripts/gpu/run_query.py` → `src/rag/query.py` →
install → answer prefill → decode), at repo HEAD `ad0d926` (W24.1: reprefill
path already removed). No architecture changes were made for this document
and none are proposed — only failure classification, evidence, and minimal
code-level remedies.

**Method**: 4 code waves (query/generation; model+weights+install; retrieval;
ingest/finetune consistency) + 2 literature waves (arXiv: linear-attention
state semantics & memory reads; quantization-induced generation degradation)
+ one synthetic algebra demonstration of the install math (CPU, numpy, no
model, no GPU results claimed — it isolates the ALGEBRA only).

**The observed failure** (the uploaded HANDOVER run, after the tokenizer
2-D hotfix, W24 code): retrieval is sane (top 3 = [70, 80, 82], scores
0.6286 / 0.5986 / 0.5889 over 100 chunks) but the answer is garbage from
the very first token — `"by byoims车 车cse na & wife。 （..."` — mixed-script
near-random tokens with heavy whitespace. First-token garbage means the
corruption is already present in the state the answer prefill/decode
consumes, not something that develops during sampling.

---

## Executive summary (ranked, with confidence)

| # | Finding | Class | Confidence | One-line evidence |
|---|---------|-------|-----------|-------------------|
| **F1** | The n≥2 S-state install (`Σ abs_i − (n−1)·sys`) is **structurally invalid algebra for the GatedDeltaNet recurrence** — the sum is not a state any forward pass could ever produce | **ROOT CAUSE** | **Certain** (math + demo) | `install.py:278` builds `dequant(sys)·(1−n)`; GDN recurrence has erase+decay (`modeling.py:686,693-715`); synthetic delta-rule demo: n=1 rel-MSE 0.0000, n=3 rel-MSE **2.63**, cos 0.56; the W14 bisect budget for S corruption was **0.15** |
| **F2** | Quantized write-path noise on S may still exceed the model's tolerance even at n=1 (reseed round + online quant round ≈ 0.03–0.10 rel-MSE) | Contributor (measured margin) | Medium | W15 TRUE-dist budget 0.15; `w16_probe_noise` decomposition 0.030 reseed + 0.070 capture+install; true-doc control (RAW, zero quant) = conf 0.895 vs quantized install = garbage in the same bisect |
| **F3** | The answer-phase M1/M2 read is **untrained, uncalibrated, and was never validated for generation** — read_gate frozen at 1.0, dense softmax over a summed 1024-slot memory, output added to every token's core attention | Contributor (high suspicion at n≥2) | High | `m1m2_finetune.py:893` `freeze_read_gate=True` ("the contrastive loss never sees it"); user's finetune passed no `--aux-nll-weight`; `modeling.py:758` adds `_read_out` to `core_attn_out`; the W14/15 true-doc control ran with **zero-init gates** → memory reads returned exact zeros → the control never exercised this path |
| **F4** | conv-state install is an arbitrary stale window (the retrieval-order LAST chunk = the *lowest*-scoring one), quantized at 3.5 bits | Minor contributor | High | `install.py:210-216` `_last_conv`; run order [70, 80, 82] → chunk 82's tail installed |
| **F5** | Loader metadata debt (GS mismatch gs 16 vs derived 512 on 132 tensors; palettized group inconsistency) | Watch item — NOT the garbage cause | Certain-not-cause | The true-doc control passes through the identical loader/ROT/AWQ path and scores 0.895; loader policy trusts LUT geometry (correct per its own audit) |

**Ruled out with evidence** (see §R): decode kernels / ROT / AWQ /
palettization, tokenizer batching, reprefill (already gone at HEAD),
full-attn query-seen-twice, retrieval, cache keying/geometry, position
handling.

**Bottom line**: the architecture is sound at **n=1** (verbatim install is
algebra-exact and bit-exact) and the retrieval side works. The pipeline
fails because the §6 install **sums GatedDeltaNet recurrent states**, which
the delta rule's erase+decay recurrence makes non-additive — at the user's
`--top-k 3` the installed S state is ~20× outside the corruption budget the
model tolerates — and the answer then reads through an M1/M2 path whose
read gate was never trained and whose amplitude was never validated.

---

## F1 — the n≥2 S-state install is structurally invalid algebra (ROOT CAUSE)

### Where it lives
- `src/rag/install.py:219-306` — `sum_absolute_codes`:
  `target = Σ_i abs_i − (n−1)·sys`, one requant. The literal code:
  `acc = q.dequant(system) * (1 - len(absolutes))` (line 278/303).
- Consumed at query time: `src/rag/query.py:217` → `install_snapshot`
  (`install.py:309-408`) → `cache.set_s_codes(L, summed)` per linear layer.
- The state it produces is read by the model as the GDN **initial state**
  of the answer prefill: `src/scripts/modeling.py:691`
  (`recurrent_state = ...recurrent_states[0] if use_precomputed_states`) →
  `torch_chunk_gated_delta_rule(..., initial_state=recurrent_state, ...)`
  (line 705-715).

### Why the algebra is wrong (the code-level truth)
The GatedDeltaNet recurrence carried by this repo's own kernels
(`modeling.py`, FLA `chunk_gated_delta_rule` / recurrent twin) is

```
S_t = λ_t · (I − β_t k_t k_tᵀ) · S_{t−1} + β_t k_t v_tᵀ        (k l2-normalized)
```

Per chunk i, captured independently from the system reset point
(`ingest.py:325-336` — each chunk is reseeded from `S_sys` and prefilled
alone), the end state is

```
abs_i = A_i · S_sys + B_i ,   A_i = Π_t λ_t(I − β k kᵀ)  (a contraction ≠ I)
```

The install computes

```
S_install = Σ_i abs_i − (n−1)·S_sys
          = (Σ_i A_i − n + 1)·S_sys + Σ_i B_i .
```

Two independent structural errors:

1. **The spurious negated system term.** Over a chunk-length sequence the
   system state decays to almost nothing (the synthetic demo measures
   cos(abs_1, S_sys) = 0.056; the TRUE sequential end state carries ~no
   system content), so the term `−(n−1)·S_sys` injects **pure spurious,
   sign-flipped system energy** into the installed state: 12% of the
   install's energy at n=2, **28% at n=3** (demo numbers). No real
   trajectory state contains a negative system component; every downstream
   layer (gated RMSNorm `modeling.py:765`, `out_proj`, the MLPs) consumes
   a direction that no pretraining forward pass ever produced.

2. **The B_i superposition keeps stale content at full strength.** In a
   true sequential read, chunk 1's associations B_1 are decayed and
   partially erased by chunks 2 and 3's transitions (A_3A_2 applied to
   them). The sum keeps every B_i at end-of-own-chunk strength. Even
   granting the venture's superposition intent, the resulting tensor has
   ~2× the norm of any single-chunk state (demo: ‖install‖ = 28.3 vs
   ‖true‖ = 14.5) and a direction that lies off the trajectory manifold.
   The gated RMSNorm normalizes the *scale* per token, so it is precisely
   the *direction* error that survives into the residual stream, and it
   compounds through 24 linear layers.

### The numbers (synthetic delta-rule demo — algebra only, no model)
`scripts/demo_install_algebra.py` (CPU, numpy; the
exact recurrence above, D=128, 256-token chunks, 5 seeds):

| n retrieved | rel-MSE(install vs true sequential) | cos(install, true) | spurious −(n−1)·sys energy |
|---|---|---|---|
| 1 | **0.0000** | 1.0000 | 0% |
| 2 | 0.9568 | 0.7414 | 12.2% |
| 3 | **2.6332** | **0.5617** | 28.3% |

Reference tolerance from this repo's own bisection history: the W15
TRUE-dist budget for S was **0.15 rel-MSE** (`bisect_install.py:429-431`,
"the online ingestion drift compounds through the recurrence") — and the
W14 matrix observed garbage *below* that budget. The n=3 install error is
**~20× over budget**. Note the demo *overlays no quantization at all* —
this is the pure algebra, before TurboQuant's additional rounds.

### Why this exactly matches the observed symptom
- Garbage **from the first decoded token**: the answer prefill's logits
  (query.py step [7]) are computed against S_install in every linear layer
  — the distribution is corrupted before the first argmax.
- **n=1 is clean by the same math** — which is why the W16 verbatim
  install claims ("bit-exact cache reuse") and the W23.6 note are both
  true *and* compatible with the garbage: the failure only exists for
  n≥2, and the user's run retrieved top-3.
- The W23.5 "generation repair — the proven prefill context build"
  worked precisely because re-prefilling the retrieved TEXT rebuilt S by
  actual sequential recurrence (an on-manifold state). W24 removed it by
  order (it bypassed the installed codes); the install-only path then
  re-exposes F1.

### Literature cross-check (wave 5)
- The delta rule's defining feature is the **erase term** — DeltaNet
  (Yang et al., arXiv:2406.06484) and Gated Delta Networks (Yang, Kautz,
  Hatami, ICLR 2025, arXiv:2412.06464) formulate the recurrence exactly
  as above; GatedDeltaNet-2 (2026) goes as far as *decoupling erase and
  write* because the coupling is the algebraic crux. A sum of
  independently-evolved states is not closed under this recurrence: the
  erase operators of one chunk do not act on the other chunks' content in
  the sum.
- Every proven external-memory design reads from a **separate store**
  rather than composing recurrent states: Memorizing Transformers (Wu et
  al. 2022, arXiv:2203.08913 — kNN into a (key,value) store, mixed into
  attention through a **learned gate**), Product-Key Memory (Lample et
  al. 2019, arXiv:1907.05242 — sparse top-k slot reads). The additive
  sum-instead-of-read is exactly the one thing none of them do —
  consistent with the M1/M2 design in this repo (additive write, exact
  superposition) but NOT with summing the GDN S states.
- Cache-Augmented Generation (arXiv:2412.15605 line of work) reuses
  document context as a **KV cache** — exact because softmax attention's
  cache is per-token content-addressed with no cross-token state mixing;
  the recurrent state has no such property.

### Minimal remedies (no architecture change)
1. **Config-only**: `--top-k 1` → the install takes the verbatim path
   (n==1 hand-off, `install.py:262-277`) — algebra-exact, bit-exact,
   zero requant. Cost: single-chunk answer context.
2. **3-line patch (recommended)**: in `install_snapshot`, install S (and
   conv) **verbatim from one chunk** (best-scoring, or the last) and sum
   **only M1/M2**, whose write is additive and whose superposition is
   exact (see F3 for the read-side caveat). This keeps the multi-chunk
   retrieval signal flowing through the one memory that mathematically
   supports it, and keeps the GDN state on-manifold. Everything else in
   the flow stays untouched.
3. Longer-term (flagged, not architecture): decay-corrected composition —
   persist each chunk's decay product so the install can weight B_i the
   way a sequential read would; or per-chunk S reads mixed at the output.
   Both are install-math changes, not layer changes.

---

## F2 — quantized write-path noise may still exceed tolerance even at n=1

Even with the verbatim n=1 install, the installed codes are the product of
the **online TurboQuant regime**: the chunk was prefilled at ingest with
quantize-on-write (`tq_cache.py:416-433` — every
`update_recurrent_state` requantizes at bits=3.5), starting from a
**reseeded quantized system state** (`ingest.py:332` —
`reseed_cache` installs `dequant(sys)` codes). The residual distortion vs
the raw state the true-doc control materializes is therefore ≈ the reseed
round + the ingest write round.

Measured anchors in the code history: `scripts/w16_probe_noise.py`'s
decomposition (reseed-drift 0.030, capture+install 0.070, total 0.0994 at
the rig); the W15 TRUE-dist budget S ≤ 0.15; and the W14 bisect matrix,
where the **raw** true-doc control decoded at conf 0.895 while the
**quantized** install variants produced garbage in the same run — i.e. the
observed corruption threshold sits somewhere between 0 and ~0.1 rel-MSE on
this model. The W16 absolute protocol removed the capture+install rounds
at n=1, leaving ~0.03–0.1 — inside the budget on paper but with a margin
thin enough that it has never been measured end-to-end at the current
stack (mem_size 1024 + trained gates + W15 conv split).

Literature cross-check (wave 6): KV-cache quantization work (KIVI
arXiv:2402.02750, GEAR, the adaptive-precision line) tolerates 2–4-bit KV
because errors are **per-token and transient**; a recurrent state is a
**single persistent tensor whose error propagates through every subsequent
step of the recurrence** — structurally tighter, which is why the same
bits that are safe for KV can be fatal for S. AWQ (arXiv:2306.00978)
shows ~1% salient weights dominate quantization error for *weights*; the
S state has no such protected subset.

Decisive test: E3 below (`--top-k 1 --no-answer-m1m2`). If THAT is still
garbage, F2 is the floor and the S write path needs bits ≥ 4 (or the
reseed round removed). If E3 is clean, F2 is currently within tolerance
and F1/F3 own the failure.

---

## F3 — the answer-phase M1/M2 read is untrained, uncalibrated, unvalidated

The wiring (`src/scripts/modeling.py:720-760`): at every linear layer,
every token of the answer phase, the model reads the cache's M1/M2
memories (`read_m1/read_m2` — dequantized from the installed codes),
computes `softmax(q @ M1ᵀ/√D) @ M2`, scales by `read_gate[layer]`, and
**adds the result directly to the GDN core attention output** before the
gated RMSNorm.

Three independent problems:

1. **read_gate was never trained.** `m1m2_finetune.py:893`:
   `freeze_read_gate: bool = True  # the contrastive loss never sees it`.
   The §7 loss scores WRITTEN states (pair similarity) and never reads
   (`m1m2.py:96-102` — "the §7 fine-tune never reads"). The user's
   training command (uploaded HANDOVER) passes no `--aux-nll-weight`, so
   `read_gate` stands at its **1.0 init** on every layer — the amplitude
   of the injection into every token's output was chosen by an initializer,
   not by any measurement.
2. **The read was never validated in the control.** The W14/15 true-doc
   control (conf 0.895) ran through `bisect_install.py`'s
   `load_model(...)` with **no gates npz** → zero-init write gates → the
   memories stayed exactly zero → `read` returned exact zeros (the P3
   no-op). The one flow that proved generation clean never exercised the
   trained-read path. "The architecture works" is therefore proven for
   the model+decode+retrieval — NOT for the answer-phase memory read.
3. **The read consumes the n≥2 summed memory through a dense softmax.**
   The M1/M2 *install* sum is algebra-exact (the write is additive,
   `m1m2.py:163-175` — "chunk deltas sum losslessly"), so this is not an
   algebra bug; it is an amplitude/quality problem: at n=3 the slot
   contents are ~3× larger than any single-chunk regime, the dense
   softmax over 1024 accumulating slots is the "blur" the module's own
   W22 docstring warns about, and read_topk defaults to 0 (dense) at
   query unless `--m1m2-read-topk` is passed (the user's run didn't).

Literature cross-check (wave 5): Memorizing Transformers inject external
retrieval through a **learned gate** and PKM reads **sparse top-k** —
both mechanisms exist precisely because an ungated, dense average over a
large accumulated memory degrades the output distribution.

Decisive test: E2 below (`--no-answer-m1m2`) — one flag, already
implemented for exactly this A/B (W23). Remedies (no architecture
change): train read_gate via the existing `--aux-nll-weight` path (the
finetune already supports it), or deploy with
`--m1m2-read-topk 32` (W22) and/or a small read_gate constant.

---

## F4 — the conv-state install is an arbitrary stale window (minor)

`install.py:210-216` (`_last_conv`) + `install.py:370-374`: the conv
window installed for the answer is the **retrieval-order last** chunk's
end-of-chunk window. In the user's run the retrieved order was
[70, 80, 82] by descending score, so the "last" chunk is 82 — the
**lowest-scoring** of the three — and the window holds its final ~4
tokens' mixed-qkv (the causal-conv state that feeds the very first query
tokens, `modeling.py:645-668`). The true-doc flow's conv window holds
the actual final doc tokens instead. Combined with the W15 split's
~0.02–0.03 rel write-path noise, this is a bounded but real mismatch at
the most position-sensitive spot in the decode path. Trivial remedy when
touching install anyway: take the conv from the same chunk whose S is
installed verbatim.

---

## F5 — loader metadata debt (watch item, not the garbage cause)

The load log prints a GS metadata mismatch (gs 16 vs derived 512 on 132
tensor components) and a palettized-group inconsistency; the loader's
policy is to trust the LUT geometry and it says so ("Fix the writer").
This is real debt for the palettizer's writer, but it cannot be the
garbage differentiator: the true-doc control loads the identical
artifacts through the identical ROT/AWQ compensation and FusedPalettizedMLP
route and decodes at conf 0.895. Keep it on the list; do not chase it for
this failure.

---

## R — ruled out, with the evidence that rules them out

| Suspect | Ruling | Evidence |
|---|---|---|
| Decode kernels / FLA / Triton / FusedPalettizedMLP / ROT / AWQ | Clean | The true-doc control (conf 0.895) runs the same loader, same kernels, same greedy loop (`bisect_install.py::true_doc_flow`; `query.py:24-26` cites it as the control) |
| Tokenizer tensor batching | Fixed | `run_query.py:160-162` builds 2-D ids (the W23 repo fix = the box's sed hotfix); the HANDOVER run already includes it |
| The reprefill path | Gone | Removed at HEAD (W24.1); `run_query.py:175-179` documents the removal; install is the only answer-context build |
| Full-attn "query seen twice" | Fixed | `_reset_full_attention` (`query.py:103-131`) uses the control's exact `DynamicLayer.reset()`; M1/M2 token counters zeroed |
| Retrieval | Functional | Top-3 [70, 80, 82] / 0.63-0.59 scores over 100 chunks; `index.py` rerank is exact cosine on unit vectors; centered frame optional and OFF in this run (absolute metric) |
| Cache keying / geometry drift | Clean | S/conv keyed by full-layer-list indices consistently (`tq_cache.py:651-665`, `snapshot_codes:783-788`); loud geometry guards at load (`ingest.py:188-214`, `tq_cache.py:743-762`) |
| Position handling at answer | Clean | Fresh full-attn after reset → positions restart at 0 for the answer prefill; `_m1m2_tokens` reset (`query.py:129-130`); decode continues from cache length |
| M1/M2 install algebra | Exact (unlike S) | The write is additive and per-call path-independent (`m1m2.py:578-644`, D4) — the sum formula is *correct* for M1/M2; the problem is the read, not the install (F3) |

---

## E — the decisive experiment ladder (one command each, GPU box)

Ordered to bisect F1 vs F3 vs F2 in at most three runs:

```bash
# E1 — kills the n>=2 S-sum AND the M1/M2 sum (verbatim install, read still ON)
python3 scripts/gpu/run_query.py --disk-dir /home/ubuntu/ragga_ingested_test100 \
    --question "How do I deploy perf-canary?" --m1m2-gates m1m2_gates_trained.npz \
    --top-k 1 --max-new-tokens 128
#   clean  => F1 confirmed (the S sum is the killer; n=1 is exact)
#   garbage=> go E2

# E2 — kills only the M1/M2 answer read (n=3 sum still installed)
python3 scripts/gpu/run_query.py ... --top-k 3 --no-answer-m1m2 --max-new-tokens 128
#   clean  => F3 confirmed (the memory read is the killer)
#   garbage=> F1 confirmed (the n=3 S sum is the killer)

# E3 — the cleanest regime: verbatim S install + no memory read
python3 scripts/gpu/run_query.py ... --top-k 1 --no-answer-m1m2 --max-new-tokens 128
#   clean  => the floor is set by F1/F3; ship top-k 1 + calibrated read
#   garbage=> F2 is the floor: quantization tolerance — raise S bits /
#            remove the reseed round; re-run bisect_install.py for TRUE-dist

# E4 (if E2 was clean) — the sparse-read candidate fix
python3 scripts/gpu/run_query.py ... --top-k 3 --m1m2-read-topk 32

# E5 — the standing bisection matrix at the current stack (true-doc row
#      re-validates the control WITH the current loader+gates regime)
python3 scripts/gpu/bisect_install.py --chunk 0 --question "How do I deploy perf-canary?" --max-new-tokens 32
```

Read the `[gen gate]` line the drivers print (mean conf ≥ 0.20, rep ≤ 0.6;
the true-doc control measured 0.895).

---

## Recommended minimal fix order (no architecture change)

1. **Immediately**: run E1–E3 (three commands, ~minutes each on the A10G)
   to convert this classification from "math-certain + high-confidence" to
   "measured".
2. **Ship-safe patch** (≈3 lines in `install_snapshot`): S + conv
   verbatim from the best-scoring retrieved chunk; M1/M2 summed (exact);
   keep everything else. This is fully within the existing §6 mechanism —
   it only stops summing the one tensor whose recurrence is non-additive.
3. **Calibrate the read**: `--m1m2-read-topk 32` and/or train read_gate
   with `--aux-nll-weight` (already supported by the finetune driver);
   until then `--no-answer-m1m2` is the honest default for answer quality.
4. **Then** reassess F2 (bits on S / the reseed round) only if E3 comes
   back garbage.

---

## Appendix — method and reproduction

- Code waves: (1) `run_query.py`, `query.py` (prompt build, install,
  answer prefill, greedy decode, tokenizer); (2) `modeling.py` GDN forward
  + `tq_cache.py` online cache + `install.py`/`snapshot.py` + `m1m2.py` +
  `loader.py`; (3) `index.py` retrieval/frame + `hooks.py` capture; (4)
  `ingest.py` capture protocol + `m1m2_finetune.py` gate-training regime +
  `run_ingestion.py` tokenization regime.
- Literature waves: Gated Delta Networks (ICLR 2025, arXiv:2412.06464),
  DeltaNet (arXiv:2406.06484), GatedDeltaNet-2 (2026, erase/write
  decoupling), Memorizing Transformers (arXiv:2203.08913), Product-Key
  Memory (arXiv:1907.05242), CAG/KV-reuse line (arXiv:2412.15605),
  KIVI (arXiv:2402.02750), AWQ (arXiv:2306.00978), Outlier-Safe
  Pre-Training (ACL 2025).
- Algebra demo: `scripts/demo_install_algebra.py` in this repo
  (numpy, CPU; no model, no GPU artifacts — pure recurrence simulation).
  Output is reproduced in F1's table.
- Evidence files: uploaded HANDOVER.md (the garbage run), this repo at
  `ad0d926`.

---

## Errata (W26 — post-publication audit, before the box run)

1. **F3's remedy line overstates the driver surface.** "train
   read_gate via the existing `--aux-nll-weight` path (the finetune
   already supports it)" is TRUE at the module docstring level
   (`m1m2_finetune.py` names `--aux-nll-weight` as "the driver's"
   flag) but **FALSE at the driver level**:
   `scripts/gpu/finetune_m1m2.py` has NO `--aux-nll-weight` argument
   (checked against the argparse surface at W26). The driver's
   `--train-read-gate` merely unfreezes the parameter — the §7
   contrastive loss has no gradient path to it (it scores WRITTEN
   states), so unfreezing alone trains nothing. Read_gate currently
   CANNOT be trained by any driver flag; the read-gate training path
   is a TODO. The interim read mitigations that DO exist today:
   `--no-answer-m1m2` (the honest answer-quality default until
   calibrated) and `--m1m2-read-topk` (sparse read, W22).
2. **The F1 remedy is IMPLEMENTED (opt-in, W26).** The 3-line patch of
   "Recommended minimal fix order" item 2 now ships as
   `install_snapshot(..., gdn_install="best")` /
   `run_query.py --install-gdn best`: S + conv verbatim from the
   highest-scoring retrieved chunk, M1/M2 summed (unchanged). The
   default stays `"sum"` (the §6 contract) so the E1–E3 ladder above
   measures the broken path exactly as ranked; switch to `best` only
   after E1 confirms. Gates: `test_query.py::
   test_install_gdn_best_mode` (AUTHORED, PENDING the box).
3. Line-number caveat: the `install.py:219-306` / `query.py:217`
   references above were written at `ad0d926`; W26 shifts them
   (install_snapshot is now at `install.py:344+`, the query call at
   `query.py:227`). The *shapes* of the claims are unchanged.

---

## Errata (W27 — the box's E-ladder attempt, classified)

The W26 HANDOVER's E1–E3 ladder ran on the box (the uploaded hermes
session log). Verdict per leg, with the new evidence:

- **E1 (top-k 1, gates loaded): GARBAGE** — multilingual soup from
  token 1 (`"by theness原则anImmke BIfetimes..."`) while the printed
  gen gate *passed* its own bar (conf 0.382 ≥ 0.20, rep 0.25) — the
  conf bar cannot see script-mixing; W27 adds the non-ascii ratio and
  a verdict band to the printout. **CONTAMINATED**: the "gates" it ran
  with were not a trained artifact at all — see Erratum 4.
- **E2 (top-k 3, --no-answer-m1m2): CRASHED before the install** —
  `torch.cat` in `query_cache_vector` raised "Expected all tensors to
  be on the same device" (host M1/M2 codes vs model-device S codes;
  the disabled M1/M2 block never requantizes the memory codes onto the
  model device, and the reseed leaves them host-side off disk).
  **THE F1-vs-F3 ISOLATION NEVER RAN.**
- **E3 (top-k 1, --no-answer-m1m2): CRASHED** — same site, same cause.

4. **The box's gates artifact is a hand-broadcast forgery, and it was
   the E1 contamination.** The box's only artifact
   (`m1m2_gates_trained.npz`) is pre-W21 per-layer `(24,)`; the
   on-box hermes session "repaired" it by `np.repeat`-ing each layer's
   scalar across all 32 heads (`m1m2_gates_trained_fixed.npz`, pushed
   to this repo at `a9bbda2` and deleted in W27) — correct shapes,
   correct geometry meta, **zero per-head addressing signal** (a
   content-addressed kernel with per-layer-constant gate rows is
   noise-by-construction: the read injects content-blind signal at 24
   layers × every decode step). That file loaded cleanly under the
   W26 checks and rode the entire E1 run. **W27 closes the hole**:
   `load_gates` now (a) refuses artifacts without `meta_write_addr`
   on a content-addressing module (every save_gates file since W21
   carries it; W27 additionally bakes it in), and (b) refuses the
   broadcast signature (write_gate_k rows constant across heads).
   The E-ladder is re-specified GATELESS (Erratum 5).
5. **The corrected ladder drops `--m1m2-gates` on the box's corpus.**
   The corpus was ingested with zero-init write gates (the W17-era
   ingest predates the finetune; the W22 log itself said "re-ingest
   with --m1m2-gates" and that was never done), so its chunk M1/M2
   codes dequantize to exact zeros — the memory subspace is INERT and
   F3 (the read's answer-phase effect) is **untestable on this
   corpus** (an empty memory returns nothing the read could blur).
   E1-gateless and E3 are therefore expected to agree; the ladder's
   discriminating legs are E2 (F1, the n≥3 sum) and E3 (F2, the
   quantized write-path floor). Retrieval-side regime consistency is
   now automatic: `run_query.py` introspects the corpus
   (`index.corpus_memory_state`) and runs the retrieval prefill
   M1/M2-OFF when the memories are empty and gates are loaded (the
   W27 regime match — the gateless runs were already consistent).
6. **The retrieval complaint ("rerank fails badly on old snapshots")
   decomposes into three measured, non-architectural causes**: (a) the
   absolute-frame common-component floor (W16's own measurement: a
   ~0.70 score floor with ~0.016 spread — the centered frame
   `run_index.py` writes exists to remove it, and the box has never
   run it for test100: every log prints "centered retrieval frame
   OFF"); (b) the inert memory subspace diluting the §4 metric (on
   the box geometry 8,388,608 of 8,486,912 dims are M1/M2 — inert
   mass that compresses the S signal's score spread; the centered
   frame's sys-subtraction removes the shared part); (c) the
   query-time regime mismatch when gates are loaded against a
   zero-write corpus (fixed by the W27 regime match). The ground
   truth question ("is chunk 70 the right doc for perf-canary?") is
   answerable with `scripts/gpu/eval_retrieval.py` (hit@1/3/5 in all
   three frames against the benchmark's gold doc ids) — run it after
   `run_index.py`.
7. **The E2/E3 crash is fixed at the crash site** (W27.1:
   `query_cache_vector` aligns every §4 piece to the S pieces' device
   — the live model's; uniform-device flows are bit-identical no-ops,
   pinned by `test_query_cache_vector_device_alignment`). No flow
   contract changed: oracle mode still computes the (now
   crash-free) query vector, the install still overwrites the codes
   the prefill left.
