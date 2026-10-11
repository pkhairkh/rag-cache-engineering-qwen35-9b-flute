# PROPOSAL.md — W20: The TQ Kernel Family — Kernel Paths for All Parts

Purpose: the design layer for **CUDA kernel paths for EVERY part of
TurboQuant** on the A10G Ampere box — quantize, dequantize, the large
rotation, the bit packing, the QJL sketch, the batched serving passes.
One sentence: the paper's core selling point is that TurboQuant is
"highly accelerator-friendly" (arXiv-2504.19874 main.tex:168) — it
attacks prior work whose "binary search algorithm is also computationally
slow and particularly inefficient on accelerators like GPU because of
their algorithm's inherent lack of vectorization" (main.tex:250) — and
our current numpy/CPU implementation IS that anti-pattern. This
document specifies the fix as kernels, not as torch ops.
Authority: subordinate to `SPECIFICATION.md`; execution waves live in
`HANDOVER.md`. Supersedes PROPOSAL rev 0 (GPU twin) and rev 1
(device-parametric torch single-path): rev 2 makes the CUDA kernel the
RUNTIME path; torch remains the differential twin and the GPU-less dev
box's path, exactly as `flute_extended` already treats `kernel_fht.cu`
vs `fht_reference` (the auto backend).
Status: **rev 2 — the KERNEL directive** (the user's directive, verbatim:
"we need kernel paths for all parts! We are on A10G ampere with 23GB
VRAM! We need TQ kernels for all paths!"). Proposed; this wave is
doc-only (no code changed, no tests run).

Sources read for this document (all in `upload/` + the repo):
- **arXiv-2504.19874v1.tar** — the TurboQuant paper source (main.tex,
  Alg. 1 `TurboQuant_mse`, Alg. 2 `TurboQuant_prod`, §"Practical
  Implementation" (empty — the paper leaves the kernel layer to the
  engineer; this document is that layer), the accelerator-friendliness
  quotes above, A100 as the paper's own hardware).
- **ptx_isa_9.2.pdf** — the PTX ISA 9.2 reference: `bfe` Bit Field
  Extract (§9.7.1.19, p.152), `bfi` Bit Field Insert (§9.7.1.20, p.154),
  `redux.sync` (§9.7.13.12, p.405 — note: `.add` exists only for
  `.u32/.s32`; an fp32 sum reduction must use a `shfl.sync` tree +
  fixed-order fold), `cp.async` data movement (§9.7.8), `ld.global.nc`
  vectorized loads.
- **the in-house kernel precedents** — `flute_extended/src/kernel_fht.cu`
  (the existing single-tile FHT kernels, their shared-memory discipline)
  and the qlora repo's `flute_train_kernels` (kernel_backward_gemm.cu,
  kernel_gemv_splitk.cu, kernel_lut_grad.cu + docs/KERNELS.md): the
  deterministic split-K finalizer, `__launch_bounds__`, canary/debug
  twins, no-atomics discipline — all specified for THE SAME A10G box
  ("SM_86, 80 SM, 600 GB/s" per their KERNELS.md).

---

## 1. What must become a kernel (the paper's own op list)

`TurboQuant_mse` (paper Alg. 1) per unit x ∈ R^d on the sphere:

| paper line | operation | today (CPU!) | rev 2 kernel |
|---|---|---|---|
| `y ← Π·x` | the random rotation | FHT torch butterfly (quant: GPU; dequant adjoint: **CPU**) | `fht_forward_large` / `fht_backward_large` (§3.2) |
| `idx_j ← argmin_k \|y_j − c_k\|` | nearest-centroid encode | **np.searchsorted on CPU** | in-register bucketize fused in `tq_quant` (§3.3) — a ≤16-entry compare, no search |
| store `‖x‖` fp | the norm | **`.item()` sync + CPU float** | fixed-order block reduction fused in `tq_quant` |
| pack idx to b bits | bit packing | **np.packbits on CPU** | `bfi`/shift pack fused in `tq_quant` |
| `ỹ_j ← c_idx_j` | centroid gather | **numpy fancy-index on CPU** | smem LUT gather fused in `tq_dequant` (§3.4) |
| `x̃ ← Πᵀ·ỹ` | the inverse rotation | **CPU adjoint butterfly** | `fht_backward_large` fused into `tq_dequant` |
| `Q_prod` extras (Alg. 2) | residual + QJL sign sketch | **np.sign on CPU** | the `tq_qjl` chain (§3.5) |

The 3.5-bit split (lo/hi halves, D2) and the conv window's segment
structure ride the same kernels — the segment table (`fht.segments`)
maps every production unit onto 16,384-coord tiles:
S/M1/M2 at mem_size=128: d=2^19 → 32 tiles; M1/M2 at mem_size≥1024:
d=2^22 → 256 tiles; conv: d=24,576 → 2 tiles (16,384 + 8,192 — both
segments already kernel-eligible today).

## 2. The measured problem (unchanged from rev 1, one paragraph)

Per prefill the wiring (`modeling.py:715–754`, `tq_cache.py:399–727`)
performs 120–192 TQ round trips; each pays six CPU syndromes
(turboquant.py: `.item()` norm sync :491, `y.cpu().numpy()` :501, numpy
searchsorted/pack :503–508, numpy gather :770–774, the from_numpy CPU
adjoint butterfly :775–776 — the monster, the return `.to(device)`
:705–706). The box measured **263 s per capture cycle** (~16.4 s per
prefill, ~95% CPU). The audit counted **103 numpy/sync touchpoints in
turboquant.py** plus the serving-side sites (index.py:372/473/510,
evals.py ×5, m1m2.py:345). Rev 2 deletes them by replacing the whole
pipeline with the kernel family below — the torch ops are not "twinned
alongside"; the CUDA runtime dispatch goes kernel → (fallback) torch
only when no CUDA context exists.

## 3. The design: the TQ kernel family

New file `src/flute_extended/src/kernel_tq.cu` (same module, same build,
same bindings pattern as `kernel_fht.cu`), entries bound in
`bindings.cpp`:

### 3.1 The kernel inventory

| entry point | grid | threads | smem | core ops (PTX) | consumers |
|---|---|---|---|---|---|
| `fht_forward_large(x, signs, out, K)` | K/16,384 tiles × rows | 1024 (32 warps) | 64 KiB tile | butterfly stages in-tile; cross stages global; `cp.async` staging, `ld.global.nc.v4` | standalone rotations (QJL, frame math) + inside quant |
| `fht_backward_large(g, signs, out, K)` | same | same | same | the adjoint orientation (signs applied pre-butterfly, per `fht_reference_adjoint`) | standalone + inside dequant |
| `tq_quant_batched(units[], d, books, out_codes[])` | Σ units×tiles + norm CTAs | 1024 | 64 KiB tile + LUT | norm: `shfl.sync` tree + fixed-order workspace fold; bucketize: in-register compare vs ≤16 scaled boundaries; pack: `bfi`/shift, `prmt`-assisted | `TurboQuant.quant` on CUDA (S/conv/M1/M2 writes) |
| `tq_dequant_batched(codes[], books, out[], dtype)` | Σ units×tiles | 1024 | 64 KiB tile + LUT | unpack: `bfe.u32`; gather: smem LUT (centroids broadcast); adjoint butterfly; ×norm | `TurboQuant.dequant` on CUDA (every read + round trip) |
| `tq_qjl_quant/dequant` (flag-off, complete) | as above | 1024 | as above | sign sketch via `fht_forward_large` on the residual; compensation via `fht_backward_large` | the Alg.-2 A/B (`--qjl`) |
| `tq_l2norm_rows(x, out)` | rows | 256 | small | `shfl.sync` tree | the index frame's normalize step (serving batch) |
| `tq_debug_simple_*` twins | — | — | — | scalar fragment assembly | the G-B2-style differential gates |

Debug surfaces (test-only builds, production unaffected): a canary
build poisons the tile before encode — surviving poison = coverage
regression (the qlora `FLUTE_BWD_CANARY` discipline); a "trap" build
forces the wrong fold order — must fail the determinism gate.

### 3.2 `fht_large` — the multi-block butterfly for K = 2^19…2^22

The existing kernel's ceiling is one 16,384-coord segment per 64 KiB
shared-memory tile (`fht.py:352`); K=2^19 is a single 2^19 segment —
impossible in smem as one tile. The multi-block decomposition:

- **Stage split.** The K-stage butterfly has pair distances
  `len_ = 1…K/2`. Stages with `2·len_ ≤ 16,384` (13 of them for any
  K ≥ 2^14) have every pair inside an aligned 16,384-wide tile →
  **tile-local**. Stages with `len_ > 16,384` cross tiles →
  **global passes** (log2(K) − 13 of them: 6 at 2^19, 9 at 2^22).
- **Pass 1 (tile-local, one launch per batch):** grid =
  rows × (K/16,384) tiles; each CTA `cp.async`-stages its 64 KiB tile,
  runs the 13 local stages in smem with the EXACT pairing discipline of
  `kernel_fht.cu`'s `fht_stage` (low = u+w, high = u−w, ascending — the
  rounding-order contract the reference and kernel already share),
  writes back.
- **Pass 2 (cross stages, one launch per stage or a stage-looped
  launch):** elementwise global pair op — thread i reads
  (u[i], u[i+len_]), writes (u+w, u−w). Bandwidth-bound: ~8 MiB traffic
  per stage at ~600 GB/s ≈ 15 µs; the whole cross pass at 2^19 ≈ 0.1 ms.
  The unit's 2 MiB fits the A10G's L2, so effective bandwidth is higher.
- **Signs:** forward applies s AFTER the butterfly, adjoint BEFORE
  (matching `fht_reference` / `fht_reference_adjoint` exactly).
- **Determinism:** fixed thread→pair mapping, fixed tile/warp/lane
  ownership, no atomics — run-to-run `torch.equal` assertable (the
  qlora G-B5b discipline).
- **Eligibility table extended:** K ≤ 16,384 keeps today's single-block
  kernel verbatim (binaries unchanged); K > 16,384 and multiple of 32
  routes to `fht_large`. The conv window (24,576 = two ≤16,384
  segments) already routes to the EXISTING kernel — untouched.

### 3.3 `tq_quant_batched` — the fused encode (paper Alg. 1's Quant)

Per unit x (d coords, fp32 staged via `cp.async` in tiles):

1. **Norm pass (deterministic):** grid-stride sum-of-squares; per-warp
   `shfl.sync` butterfly tree (fixed masks 16→1 = fixed association
   order), per-CTA partial to a workspace slot owned exclusively by the
   CTA; the CTA holding the last ticket of the unit folds the partials
   in ascending CTA order — the split-K finalizer discipline
   (kernel_gemv_splitk.cu) — **no atomics anywhere**. Note: fp32
   `redux.sync.add` does not exist (PTX §9.7.13.12: `.add` is u32/s32
   only) — the shfl tree IS the fp32-reduction instruction path.
2. **Boundary scaling identity (saves a whole-unit pass):** the paper's
   encode is `argmin_k |y_j/norm − c_k|`; multiplying the comparisons
   by norm > 0 gives `argmin_k |y_j − norm·c_k|` — IDENTICAL decisions
   (monotone, exact). So the kernel bucketizes the UNSCALED rotated
   coordinate against the ≤16 boundaries scaled once by norm (16
   multiplies, in smem). Same for QJL: `sign(FHT(x/norm)) ≡
   sign(FHT(x))`. The normalized vector is never materialized.
3. **Rotate + encode + pack pass:** the multi-tile butterfly (§3.2
   inline), then per coordinate an in-register linear scan of the ≤16
   scaled boundaries (2^b ≤ 16 — a compare ladder, no search; this is
   the vectorization the paper demands), then pack: each thread owns a
   fixed 16-coordinate slice, packs its b-bit indices into 6 (b=3) or 8
   (b=4) bytes via `bfi`/shifts (PTX §9.7.1.20), writes the packed byte
   range of its tile — tiles are disjoint contiguous coordinate ranges,
   so tile-aligned packing has NO cross-tile dependencies. The lo/hi
   split (n_lo/n_hi) packs the two halves separately, byte layout
   identical to today's `pack_bits` (fixture-pinned).
4. **Codes out:** packed uint8 tensors + the fp32 norm (0-dim) as
   device tensors — the rev-1 TQCodes torch-native contract unchanged;
   `to_arrays()` remains the npz disk edge.

### 3.4 `tq_dequant_batched` — the fused decode (Alg. 1's DeQuant)

Per unit: unpack each packed byte via `bfe.u32` (PTX §9.7.1.19 — b-bit
field extract at the fixed offset), gather the centroid from the smem
LUT (the ≤16-entry table broadcast to every CTA — no global-memory
gathering at all), apply signs (adjoint orientation), run the multi-tile
adjoint butterfly, multiply by the norm, store fp16/fp32. Reads ~230 KiB
codes per 2^19 unit, writes 1–2 MiB — ~15 µs of memory time per unit.

### 3.5 The QJL kernels (Alg. 2, flag-off but kernel-complete)

`--qjl` rides the same entries: the residual r = x − dequant_mse is a
fused quant→dequant chain; γ = ‖r‖ (the same deterministic norm pass);
the sketch `sign(FHT(r/γ))` = `sign(FHT(r))` (scale-invariant — §3.3's
identity); the compensation term `FHT_adjoint(qjl)` rides
`fht_backward_large`. No CPU step anywhere in the A/B.

### 3.6 Batched entries + the serving side

The runtime calls the BATCHED entries: one `tq_quant_batched` over all
of a prefill's ~120–192 units (grid = Σ tiles ≈ 6,100 CTAs at 2^19 —
~76 waves of the 80 SMs, each wave microseconds), one
`tq_dequant_batched` likewise — **two kernel chains per prefill
instead of ~4,000 host round trips.** The serving side (W20.K4): the
install requant (multi-chunk sums + requant over 24 S units + M1/M2),
the index loader's retrieval-frame vector builds
(index.py:372/473/510 — dequant + concat + L2-normalize via
`tq_l2norm_rows`), and the eval harness all call the same batched
entries; the faiss add receives GPU-produced vectors once at its
boundary (faiss-cpu stays — foreign library, tiny index, few queries).

### 3.7 The dispatcher (kernel-first, one runtime path per device class)

`TurboQuant.quant/dequant` dispatch: **CUDA tensor + kernel available →
the kernel path, always** (codes as device tensors); CPU tensor → the
torch reference path (the same code this dev box and the test twin
run). This is exactly `fht.py`'s proven auto-backend discipline —
applied at the TurboQuant level. There is no numpy path on ANY box at
runtime (rev 1's deletion stands); the npz disk edge (`to_arrays` /
`from_arrays`) and the faiss hand-off are the only foreign boundaries.

### 3.8 Build

`kernel_tq.cu` joins `flute_extended`'s existing extension build (the
same setup.py that already compiles kernel_fht.cu on the box):
`-gencode arch=compute_86,code=sm_86` + PTX fallback;
`__launch_bounds__(1024)` on every instantiation; the canary/trap
builds behind compile-time macros (test-only). Bindings:
`tq_quant_batched`, `tq_dequant_batched`, `fht_forward_large`,
`fht_backward_large`, `tq_l2norm_rows`, the debug twins.

## 4. The A10G budget (sm_86, the user's box)

| resource | value | use |
|---|---|---|
| SMs | 80 (per the qlora KERNELS.md box table) | 6,100-CTA batched grids = ~76 waves |
| VRAM | 23 GiB usable (user's figure) | units 2–16 MiB; codes KBs; workspaces MBs — trivial |
| HBM | ~600 GB/s | ~5–6 MiB traffic per unit round trip → µs-scale |
| L2 | 6 MiB | a 2^19 unit's cross-stage scratch is L2-resident |
| smem/CTA | 48 KiB default, ~99 KiB opt-in (sm_86) | the 64 KiB tile (16,384 fp32), same opt-in as kernel_fht.cu today |
| threads/CTA | 1024 (32 warps) | tile-wide staging + the 13 local stages |

## 5. Parity + determinism contract (REVISED — and why)

Rev 1 gated "byte-identical codes across backends". That is
mathematically unsatisfiable for kernel paths: the fp32 norm (a sum of
2^19 terms) has backend-dependent summation order, so the stored fp32
norm can differ by ≤ 1 ulp (and a boundary-exact coordinate can flip
its bucket — measure zero on real states). The honest contract, three
levels:

1. **Run-to-run determinism (hard):** same backend, same inputs →
   `torch.equal` on codes AND reconstruction. Guaranteed by design:
   no atomics, fixed tile/warp/lane ownership, fixed-order folds.
   (The qlora kernels assert exactly this — G-B5b.)
2. **Cross-backend (kernel vs torch-CUDA/CPU):** codes identical
   except boundary ties (gate: tie count ≤ 0.01% on frozen probe
   sets), norm within 1 ulp, reconstruction allclose ≤ 1e-5 rel.
3. **Snapshot compatibility:** an npz written by either backend loads
   and decodes allclose ≤ 1e-5; the W16 verbatim install stays
   bit-exact WITHIN a backend (it copies codes, never recomputes).
   Protocol rule (unchanged in spirit): run TRAIN → RE-INGEST → INDEX
   → EVAL → QUERY on ONE box — cross-box is allclose-level portable.

The frozen probe fixtures (rev 1) remain the accountably for the torch
path; the kernel differential twins (kernel vs torch, both on CUDA)
are the kernel's accountably — the `flute_train_kernels` gate family
(G-B1..G-B5 style) ported to TQ.

## 6. Performance arithmetic (the ladder)

Per d=2^19 unit round trip (quant + dequant):

| path | per unit | per prefill (120–192 units) | capture cycle (16 prefills) | s/step (refresh 4 / 8) |
|---|---|---|---|---|
| today — CPU numpy (measured) | ~80–160 ms | ~16.4 s | **263 s** | ~66 / ~33 |
| torch-CUDA single path (K1, the twin + immediate relief) | ~4–8 ms | ~0.5–1.5 s | ~10–25 s | ~3–7 / ~1.5–3.5 |
| **the kernels (K2+K3)** | **~15–25 µs** | **~2.5–5 ms** | **~5–8 s (model-FLOPs bound)** | **~1.3–2 / ~0.7–1** |

300 steps ≈ 4–5 minutes at the kernel rung (vs half a day today). The
capture cycle becomes bound by the model forward itself (LUT-GEMM
prefills, ~0.3–0.5 s each) — TurboQuant drops out of the profile
entirely, which is the paper's own design intent for streaming KV
(main.tex:139, "minimal latency… reduced computational and
communication resources"). Ingestion and query TQ costs likewise
collapse to µs per unit.

## 7. What the qlora kernel project contributes (the reuse map, precise)

| asset | origin | reuse in kernel_tq.cu |
|---|---|---|
| `flute_extended` build + bindings | both repos share it | kernel_tq.cu compiles in the existing extension — no new build system |
| `kernel_fht.cu` (fht_forward/backward) | both repos, identical | the ≤16,384-segment kernels stay verbatim; `fht_large` extends the same file family |
| the split-K finalizer pattern | kernel_gemv_splitk.cu | the norm reduction's workspace fold (last-ticket CTA, ascending order, no atomics) |
| the deterministic two-pass scatter | kernel_lut_grad.cu | the discipline: private partials → fixed warp-shuffle tree → exclusive workspace slot |
| canary/trap debug builds + twins | FLUTE_BWD_CANARY / FLUTE_BWD_TRAP / G-B1..G-B5 | the TQ debug twins and the coverage canary |
| `__launch_bounds__`, tile discipline, `cp.async` double-buffer | kernel_streaming.cu | the tile pipeline of pass 1 |
| the backward GEMM itself | kernel_backward_gemm.cu | **not reused** — weight-side LUT-GEMM backward, a different op family from the state-side replay (W19's finding; the replay backward is already exact at ~0.5 s/step, ~2% of wall) |

## 8. Build order (the phases)

- **W20.K1** — fixtures + the torch-CUDA single path (rev 1's §3.1–3.2
  implemented): the frozen probe fixtures captured from the numpy era;
  the torch path verified on CUDA. Lands the immediate relief
  (~3–7 s/step) AND builds the differential reference the kernels need.
- **W20.K2** — `fht_forward_large` / `fht_backward_large` (§3.2) +
  their differential twins + canary/trap builds. The foundational
  kernel; standalone-usable.
- **W20.K3** — `tq_quant_batched` + `tq_dequant_batched` (§3.3–3.4) +
  the dispatcher (§3.7) + the codes device contract wired to kernels.
  THE wave: TQ leaves the profile (G20.1/2 confirmed on the box).
- **W20.K4** — the QJL kernel chain (§3.5, flag-off) + the serving-side
  batched passes (install requant, index frame vectors, evals) +
  `tq_l2norm_rows`.
- **W20.K5** — the trainer flags: `--refresh-every 8` (staleness bound
  lr×N, W19), the loader pinning. (The "unpacked hot codes" lever from
  rev 1 is DROPPED — with `bfe` unpacking is free; codes stay packed
  everywhere.)

## 9. Risk register

| ID | risk | mitigation |
|---|---|---|
| R20.1 | kernel butterfly rounding differs from torch reference | the shared pairing discipline (low=+/high=−, ascending) — same family as kernel_fht today; twin gate allclose 1e-5 |
| R20.2 | boundary ties flip codes vs the torch path | measure-zero; gate counts ties on frozen probes (≤ 0.01%); §5 level-2 contract |
| R20.3 | the norm's last-ulp divergence | accepted in §5 (level 2/3); run-to-run torch.equal still hard-guaranteed |
| R20.4 | non-determinism (atomics, scheduling) | none anywhere: fixed ownership + finalizer folds; the trap build must fail the determinism gate |
| R20.5 | smem opt-in / occupancy regressions on sm_86 | 64 KiB tile = the kernel_fht.cu precedent; `__launch_bounds__`; occupancy checked at build |
| R20.6 | batched grid overshoot (units × tiles at 2^22 M-states) | grid-stride loop per CTA; 256-tile units still ~1 wave-pair at 80 SMs |
| R20.7 | the 23 GiB ceiling (capture pool + model + codes) | codes are KBs; workspaces MBs; the W18 `_calls_bytes` tracker rides along |
| R20.8 | cross-module drift (torch twin vs kernel) | twins run in CI on the box; the dev box runs the torch path in the suite |

## 10. Acceptance gates (GPU box)

- **G20.1** — `finetune_m1m2.py --refresh-every 4`: the W19 phase line
  shows capture ≤ 10 s/cycle after K3; TQ time ≤ 5 ms/prefill.
- **G20.2** — step time ≤ 2 s after K3 (≤ 1 s at K5's refresh 8);
  300 steps ≤ 10 min end-to-end.
- **G20.3** — the W19 convergence gates: loss leaves ln(8) at step 0,
  `gk/gv_absmax > 0`, held-out spread positive.
- **G20.4** — parity: run-to-run `torch.equal` (kernel outputs, codes
  + reconstruction); kernel vs torch: tie count ≤ 0.01%, norm ≤ 1 ulp,
  reconstruction allclose ≤ 1e-5 on the frozen probes.
- **G20.5** — protocol integrity: TRAIN → RE-INGEST → INDEX → EVAL →
  QUERY end-to-end on the kernel path; npz digests stable run-to-run;
  cross-backend decode allclose ≤ 1e-5; the W16 verbatim install stays
  bit-exact within the backend.
- **G20.6** — canary/trap builds behave (canary survives zero poison;
  the trap fold-order build fails the determinism gate).

The standing protocol (TRAIN → RE-INGEST → INDEX → EVAL → QUERY,
corpus-independent gate training, no traditional-RAG fallback) is
untouched by every phase; this wave is doc-only and changed no code.
