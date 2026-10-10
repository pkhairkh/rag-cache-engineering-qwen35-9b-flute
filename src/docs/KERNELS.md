# CUDA kernels

Purpose: the kernel family of `src/flute_extended` — entry points, regimes, design points, and the shared device primitives.
Authority: authoritative for the kernel contracts; subordinate to `SPECIFICATION.md` for RAG semantics.
Status: synced from the main project @ ab78893, scoped to this repo Wv2-8 — the inference family is carried here in full; the training-kernel package (`flute_train_kernels/`, §6) is main project, not part of this repo.

The kernel family of `src/flute_extended` (inference; the training
package `flute_train_kernels` is main project — not part of this repo).
All kernels compute `C[M, N] = A[M, K] @ W[N, K]^T` against LUT-palettized
weights (`src/docs/QUANTIZATION.md`) — W never exists as FP16. Entry
points are bound in `src/flute_extended/src/bindings.cpp`, declared in
`src/flute_extended/include/flute/entrypoints.h`.

| regime | entry point | source | when |
|---|---|---|---|
| batched / prefill, M ≥ 16 | `qgemm_per_group_lut` | `kernel_streaming.cu` | tensor-core streaming pipeline |
| M ≤ 16, two-stream weights | `qgemm_dual_stream` | `kernel_streaming.cu` | dual-stream fused decode |
| M = 1, wide modules | `qgemm_gemv_stream` / `qgemm_gemv_fht_stream` | `kernel_gemv.cu` | the memory-bound streamer |
| M = 1, everything else | `qgemm_gemv_splitk_stream` / `qgemm_gemv_splitk_fht_stream` | `kernel_gemv_splitk.cu` | split-K + double-buffer |
| M = 1, QKV groups | `qgemm_gemv_multi` | `kernel_gemv_multi.cu` | grouped multi-blob launch |
| M = 1, MLP blocks | `qgemm_gemv_mlp` | `kernel_gemv_mlp.cu` | merged gate+up + SiLU·mul |
| Hadamard fold | `fht_forward` / `fht_backward` / `fht_inplace_` / `fht_forward_awq` | `kernel_fht.cu` | see §5 |
| differential twin | `qgemm_debug_simple` | `kernel_debug_simple.cu` | bring-up / debugging |
| dense baseline | `qgemm_cutlass_dense` | `kernel_cutlass_dense.cu` | benchmarking (needs CUTLASS) |

Which route a module actually takes at M = 1 is decided by the routing
layer (`src/scripts/palettized_modules.py`) — that policy and its
switches live in [ROUTING.md](ROUTING.md).

## 1. The streaming family (prefill, M ≥ 16)

`flute_kernel_streaming` (`kernel_streaming.cu`): a one-barrier,
double-buffered tensor-core pipeline over tiles (BM, BN, BK, GS) — A
staged by `cp.async` with tile i+1's copy issued before tile i's MMA;
dequant is a paired-LUT lookup per packed byte straight into the MMA
register file (no shared-memory W tile). Details, tile table, and the
register budget: `src/flute_extended/README.md` and [BUILD.md](BUILD.md) §4.

The sub-4-bit instantiations
(`flute_kernel_streaming_sub4<Cfg, B>` legacy layout,
`flute_kernel_streaming_fd_sub4<Cfg, B>` idxN blob path) extend the
same pipeline to b ∈ {1,2,3}; the 4-bit binaries are unchanged.
`qgemm_cutlass_dual_stream` is the two-stream fused decode entry
(M ≤ 16): both streams' GEMMs in one launch.

## 2. The plain GEMV streamer (M = 1, wide modules)

`flute_kernel_gemv_dual` (`src/flute_extended/src/kernel_gemv.cu`):
one launch per module, a
plain warp-per-row streamer whose only job is to move the weight stream
once at as close to DRAM speed as the shape allows. The routing layer
prefers it for **wide** modules (≥ 160 output tiles — `lm_head` with its
1940 tiles) where the row-parallel grid fills the machine by itself; it
measures ~492 GB/s there. `qgemm_gemv_fht_stream` is the variant that
takes the FHT operands (signs / AWQ scales) and folds the boundary
rotation inside the launch.

## 3. The split-K GEMV (M = 1, the workhorse)

`flute_kernel_gemv_splitk` (`src/flute_extended/src/kernel_gemv_splitk.cu`)
— the default decode kernel for the 248 layer modules. Design points,
in the order they matter:

- **Split-K grid**: `grid = (N/128, SPLIT)` with `SPLIT` chosen at
  launch so `(N/128)·SPLIT ≥ 160` (two waves of the 80 SMs; powers of
  two 1..16; `G % (4·SPLIT) == 0` so every j4 warp keeps ≥ 1 g-tile).
  Narrow modules deepen the split (k/v: 8 tiles → SPLIT 16 = 128 CTAs;
  q_proj 64 → 256; gate/up 96 → 192; lm_head 1940 → SPLIT 1, which is
  the wide-preference crossover).
- **Double-buffered K loop**: the `g+4` code chunks load before the `g`
  compute; with split-K the per-warp chain is `K/(256·SPLIT)` g-tiles,
  so the pipeline plus the deeper split hides most of the latency the
  plain GEMV exposed. The loop is palette-parameterized at runtime
  (kRegPal: gs 16/32 → the shared-palette path, gs ≥ 64 → the
  register/shuffle path).
- **Deterministic split reduction**: every CTA writes its partial to
  the workspace `P [SPLIT, N+R]` fp32; the CTA that draws ticket
  `SPLIT-1` is the FINALIZER — it folds the partials in fixed order,
  adds the residual + bias, and rounds ONCE to FP16. Fixed association
  order end to end: k-pairs accumulate ascending-k per lane, the 4
  j-words fold red[row][0..3], the SPLIT partials fold 0..SPLIT-1.
  The workspace self-resets during the write-back pass — the invariant
  that makes CUDA-graph replay legal. `SPLIT == 1` skips the workspace
  entirely.
- **Fused residual, rank ≤ 256**: the low-rank branch
  `(x @ resB^T) @ resA^T` rides the epilogue (4 ranks per warp), the
  partials stride `N+R`. Ranks above 256 ride the two-launch fallback
  (the kernel refuses loudly, never silently degrades).
- **The wide 20-pair table**: the paired-LUT decode uses a 20-entry
  register table covering the k-pair positions of the widest
  configurations.
- Coverage: group_size 16..2048, residual rank 1..256, both stream
  widths, K multiple of 256·SPLIT/4 enforced at dispatch.

`qgemm_gemv_splitk_fht_stream` folds the FHT boundary rotation into the
same launch (fp32 staging + butterfly + per-segment signs/AWQ into the
shared-memory x row) — the A-side rotation the plain routes pay as a
separate step.

## 4. The merged launches (M = 1)

Both merges replace several launches with one split-K launch over a
concatenated tile space, keeping the single-module numerics contract
(the kloop arms are the same template instantiations the single-module
split-K GEMV runs). Decode-only (M == 1); prefill/PPL never routes
here.

**`qgemm_gemv_multi` — the QKV merge** (`kernel_gemv_multi.cu`):
2-4 modules sharing one input row in one launch. The deployed
mixed-radix palette gives every tensor its own
`(bitwidth, bitwidth2, group_size)` and every AWQ-folded tensor its own
sign vector — all of that is **per segment at runtime**: a
CTA-uniform switch (`FLUTE_HET_KLOOP_SWITCH`) into the shared kloop
(zero intra-CTA divergence), per-segment group-size paths, per-segment
sign/AWQ pointers (a segment with `s == 0` skips the AWQ compensation).
Components with different rotation seeds merge in one launch — each
CTA's FHT prologue applies its own segment's signs. The segment table
is a **persistent CPU pointer table** (pointers stable across
CUDA-graph replays); the grid resolves each CTA's segment from the
cumulative tile table. What it removes: the split-QKV `torch.cat` and
2-4 separate launches.

**`qgemm_gemv_mlp` — the merged MLP** (`kernel_gemv_mlp.cu`): one
split-K launch computes `silu(gate(x)) * up(x)`. Each CTA computes its
128-row tile of both blobs (the gate K-loop pass, then the up pass —
same x slice, same FHT, same palette structure); the epilogue folds
both rows in fixed split order and writes the product with ONE FP16
round. What it removes: the gate/up launches, the SiLU and mul
elementwise kernels, and the `[1, N]` transients.

**Known regression (measured, live)**: the merged MLP launch carries
the two LUT/code streams of gate+up in one kernel and currently
measures 97-135 GB/s where the two separate split-K launches measure
170-285 GB/s — a net ~8.7 ms/token cost across the 32 MLP groups. The
A/B is `FLUTE_NO_MERGE=1` (routes gate/up back to separate launches).
Fixing the merged kernel's stream locality is open work; the QKV merge
is a measured win (186-286 GB/s merged vs the parts' separate sum) and
stays.

## 5. The FHT kernels

`src/flute_extended/src/kernel_fht.cu` +
`src/flute_extended/include/flute/fht.cuh` + `src/flute_extended/fht.py`:
the Hadamard boundary fold
`x_rot = x @ T` with `T = blockdiag_b(H_b · diag(s_b) / sqrt(b))`
computed as an O(K log K) butterfly instead of an O(K²) explicit
matrix multiply. The sign vector scales the **columns** of T (the
`_rot_matrix` convention `hadamard(k) * s.view(1, k) / sqrt(k)` the
artifacts encode), so the forward is `(x @ H) * s / sqrt(b)` per
power-of-two segment, and the adjoint (H symmetric, H@H = b·I) is
`(y * s) @ H / sqrt(b)` — also the exact autograd gradient.

- `fht_forward(x, signs)` — out-of-place; `fht_inplace_` — in-place
  variant; `fht_backward(g, signs)` — the adjoint;
  `fht_forward_awq(x, signs, s)` — `((x*s) @ T) / s`, the AWQ-folded
  form.
- Block-diagonal K: descending power-of-two segments (production
  down_proj K=12288 = 8192 + 4096).
- Storage: a `(K,)` sign vector (~16 KB at K=4096), GPU-resident after
  the first per-device cache fill — versus the old explicit-matrix
  cache of ~1.5 GB across the 8 unique (seed, k) pairs and a pageable
  CPU→GPU copy of up to 576 MB per forward at K=12288.
- `fht_block` (the kernel-block form used inside the decode GEMV
  prologues): shift/mask pair math + register-local early stages —
  12 → 9 barriers at K=4096, used by all five FHT call sites.
- Round-trip exactness: `fht_adjoint(fht(x)) == x` bit-exact; the unit
  tests compare against the explicit `x @ T` ground truth
  (`fht.build_rotation_matrix`).
- Dispatch eligibility (W11, `fht._kernel_eligible`): the CUDA kernel
  auto-engages only for K it can tile — a multiple of 32, within
  [32, 65,504], and every descending-power-of-two segment ≤ 16,384
  coordinates (one 64 KiB shared-memory tile, the opt-in budget every
  sm_70+ device honors; the launcher queries the device attribute and
  raises loudly if a larger tile is ever requested). Larger first
  segments — K = 32,768 (one 128 KiB block, over the ~99 KiB consumer
  opt-in) or the RAG plane's S units at K = 524,288 — run the pure-torch
  reference butterfly ON DEVICE instead (correct, moderately fast; the
  RAG conv unit K = 24,576 = 16,384 + 8,192 stays kernel-eligible).

## 6. The training kernels (`flute_train_kernels/`, main project — not part of this repo)

The backward side for QLoRA-style training on palettized weights.
Every entry takes `(bitwidth, indices_layout)` — the 4-bit path is the
byte-identical legacy one; 1/2/3-bit run the sub-4-bit instantiations.
`indices_layout`, when non-empty, must equal `idx{bitwidth}` (the same
agreement gate as the inference package).

| entry | source | computes |
|---|---|---|
| `fused_backward_gemm` | `kernel_backward_gemm.cu` (main project) | `grad_X[M, K] = grad_Y[M, N] @ W[N, K]` with on-the-fly dequant (tensor cores) |
| `backward_simple_twin` | `kernel_backward_gemm.cu` (main project) | the differential twin |
| `lut_grad_scatter` | `kernel_lut_grad.cu` (main project) | `dL/dLUT[g, c] = Σ_{n∈g} Σ_{k: idx[n,k]=c} dL/dW[n, k]` — the scatter from `dL/dW = grad_Y^T X` into the codebook |
| `lut_grad_scatter_twin` | `kernel_lut_grad.cu` (main project) | the differential twin |

Python surface (main project: `scripts/qlora_gemm.py`) wraps these as
`FusedQLoRAGEMMTrainLUT` (single-stream) and
`FusedQLoRAGEMMTrainLUTTwoStreams` (two-stream autograd path — two
single-stream backwards, the frozen-stream-2 drop fixed) with the
pure-torch references (`train_lut_reference_*`) as ground truth. The
CPU gate suite (main project: `tests/test_lut_gradients.py`,
`tests/test_two_stream_training.py`, `tests/test_dual_stream.py`) is
not part of this repo.

## 7. Shared device primitives (`src/flute_extended/include/flute/gemv.cuh`)

Everything the GEMV families share lives here once:

- `gemv_chunk_word` / `gemv_pair_codes` — the idxN blob pair decode
  (the normative `idxN.py` blob map transcribed verbatim);
- `gemv_kloop` — the double-buffered K loop (B1/B2 templates,
  runtime-GS palette selection);
- `flute::GemvSegTab` — the grouped-segment table (per-segment
  bitwidths, group sizes, rotation signs, AWQ scales) for the
  heterogeneous multi/MLP kernels;
- `FLUTE_HET_KLOOP_SWITCH` — the runtime-pair dispatch for the
  heterogeneous kernels (a uniform per-CTA switch over the segment's
  `(b1, b2)`);
- `gemv_fht_prologue` — the shared FHT boundary-fold prologue (fp32
  staging + butterfly + per-segment signs/AWQ into the smem x row).

The split-K workspace lives in `src/flute_extended/src/gemv_host.cpp`
(`gemv_pick_split(tiles, G)` is the launch-side SPLIT policy of §3).
