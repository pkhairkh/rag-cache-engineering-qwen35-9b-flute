# docs/KERNEL_SPEC_DLDLUT.md — the dL/dLUT scatter kernel (W5-T01)

The spec of record for WAVE 5 (PROPOSAL WP5 / §3.2). Written BEFORE any
CUDA exists (TASKS.md W5-T01); W5-T02 implements against this document
and the PTX review checklist (docs/PTX_NOTES.md §4) audits the code
against it item by item. Every kernel-level constant below carries its
arithmetic; every budget row cites docs/PTX_NOTES.md.

## 1. The math (PROPOSAL §3.2, verbatim)

With `C = X W^T`, group `g = n // gs`, code `c = idx[n,k]`:

```
dL/dW[n,k] = Σ_m grad_Y[m,n] · X[m,k]          (a GEMM — the existing
                                                backward kernel's shape)
dL/dLUT[g,c] = Σ_{n: n//gs = g} Σ_{k: idx[n,k] = c} dL/dW[n,k]
                                                (a scatter-add keyed by
                                                the 4-bit nibbles)
```

Indices receive no gradient (integers, frozen). The kernel computes
`dL/dLUT` from `(grad_Y, X, idx4 blob)` directly: the GEMM contraction
`Σ_m` runs on Tensor Cores into an fp32 register accumulator, the
`(N,K)`-shaped `dL/dW` tile never touches DRAM, and the epilogue
scatter-adds it into the `(n_groups, 16)` fp32 output keyed by the
nibbles. One fused pass, no intermediate allocation beyond the
per-block partials (§4).

Entry point (flute_train_kernels, the package's existing host-validation
style):

```
lut_grad_scatter(grad_y, x, indices, N, K, group_size) -> grad_lut
    grad_y   [M, N]  float16 | bfloat16, contiguous, CUDA
    x        [M, K]  float16 | bfloat16, contiguous, CUDA (same dtype
                     as grad_y — dispatch mirrors fused_backward_gemm)
    indices  flat uint8 idx4 blob, N*K/2 bytes, contiguous, CUDA
    N % 128 == 0, K % 64 == 0, group_size in {32, 64} (the package
             contract; loud TORCH_CHECKs, never silent clamps)
    returns  [ceil(N/group_size), 16] float32, zero-initialized
```

## 2. The determinism decision (recorded once, binding)

**Deterministic two-pass.** Pass 1 (the main kernel): every block
computes its `(n_groups_local, 16)` fp32 partial and writes it to a
global workspace slot owned exclusively by that block — no global
atomics, no cross-block ordering hazards. Pass 2 (the reduce kernel):
sums the workspace over the block axis in a FIXED sequential order
(block index ascending; one thread per `(g, c)` output cell) into
`grad_lut`.

Inside pass 1 the reduction is deterministic at both levels: each
thread accumulates a private register partial over ITS bijection-owned
`(n_local, k)` pairs (§3, the blob-segment walk), then a fixed
warp-shuffle tree, then a fixed warp-order smem sum — no smem atomics.

Gate consequences (why atomics were rejected): `atomicAdd` on fp32 is
order-nondeterministic, so the G-B5b bit-exact twin (two runs,
identical bytes) would flake by fp32 non-associativity — an
unfalsifiable gate. The two-pass costs one extra `n_blocks x
(n_groups_local x 16) fp32` workspace (§4 sizing: ~2.4 MiB at the worst
real geometry) and one tiny reduce launch; it buys a gate that can
actually fail. G-B5b therefore asserts BIT-equality, not tolerance.

## 3. Memory layout and tile plan

Geometry (mirrors kernel_backward_gemm.cu with the output/contraction
roles swapped — the dW GEMM is the backward GEMM's transpose problem):

| Constant | Value | Role (vs the backward kernel's) |
|---|---|---|
| BM | 64 | dW tile n-rows per block (was: grad_X m-rows) |
| BN | 64 | m-contraction steps (was: n-contraction steps) |
| BK | 64 | dW tile k-cols per block (unchanged) |
| THREADS | 128 | 4 warps in a 2x2 grid (unchanged) |
| WARP tile | 32 x 32 | (WARP_M x WARP_N; 2 m-tiles x 4 n-tiles per warp) |

Shared memory (all fp16 operands fp32-accumulated; dtype ladder
PROPOSAL §3.3 / TASKS §3.3):

| Buffer | Shape | Bytes |
|---|---|---|
| sA (double-buffered) | grad_Y tile [2][BN=64 m][BM=64 n] fp16 | 16,384 |
| sB (double-buffered) | X tile [2][BN=64 m][BK=64 k] fp16 | 16,384 |
| sDW | accumulated dW tile [BM=64 n][BK=64 k] fp32 | 16,384 |
| per-thread partial | (BM/gs) x 16 fp32 in registers | — |
| block partial | (BM/gs) x 16 fp32 in smem (warp staging) | 64–128 |
| **DYN_SMEM** | | **48 KiB + partial** |

Occupancy (docs/PTX_NOTES.md §1 arithmetic): 2 CTAs/SM ->
2 x 48.125 KiB = 96.25 KiB <= 100 KiB/SM (smem-limited, the binding
resource — the backward kernel's own verdict class); threads
2 x 128 = 256 <= 1,536; registers 65,536 / 256 = 256 regs/thread
ceiling >= the 255 practical cap — `__launch_bounds__(128, 2)` declares
it. A third CTA is impossible (3 x 48 > 100), exactly like the worked
example (docs/PTX_NOTES.md §3).

Fragment sourcing (all helpers exist in-repo, flute/mma*.cuh):
* A-fragments `A'[n, m] = grad_Y[m, n]` — sA stages grad_Y tiles
  UNCHANGED (contiguous row segments, the backward's `stage_sA` pattern
  with the m/n roles swapped), fragments loaded `.trans`
  (`ldmatrix_x4_trans` + the A-style address math) — the transpose is
  consumed by the ldmatrix, never by the staging.
* B-fragments `B'[m, k] = X[m, k]` — sB stages X tiles row-major; the
  fragment load is the backward's own B pattern
  (`ldmatrix_bT_addr` + `.trans`) applied to sB.
* Epilogue: the warp accumulator fragments are written into sDW at the
  16 B-block XOR-8-swizzled positions (`flute::swz_word`, the sW
  convention), guarded only on the ragged tail rows (K % 64 == 0 makes
  the k guard vacuous — the backward's own epilogue rule).
* Scatter pass: `dequant_w_tile`'s INVERSE walk. Each thread reads its
  one `uint4` blob segment (`blob_segment_offset(n_base, k0, K, tid)`
  verbatim — 128 threads x 16 B cover the 2048 B half-tile bijectively)
  and for every byte j computes `(n_local, k)` and `(n_local, k+1)` with
  the documented mapping `n_local = v*16 + d*8 + (chunk>>2)`,
  `k = 2*(seg*8 + (chunk&3) + 4*s2)`, `v = j>>2`, `d = (j>>1)&1`,
  `s2 = j&1`; the two nibbles `(b & 0x0F)` at k and `(b >> 4)` at k+1
  key the accumulation `partial[g, nibble] += sDW[n_local, k]` (values
  read at the same swizzled word position the dequant would have
  written — the walk is byte-for-byte the decoder's, with the dataflow
  reversed: gather becomes scatter).

Grid (the backward's L2 rule, mirrored): `x = k-blocks (fastest),
y = n-blocks` — a scheduling wave covers all k-blocks of one n-strip,
so the grad_Y n-strip (M x 64) stays L2-resident; X tiles are re-read
once per n-strip (N/64 strips — the same re-read class as the backward
kernel's grad_Y re-reads, mitigated by the same wave discipline).

Workspace: `[n_blocks][BM/gs][16]` fp32 with
`n_blocks = (N/BM) x (K/BK)`. Worst real geometry (N = K = 12288,
gs = 64): 192 x 192 blocks x 64 B = 2.25 MiB — allocated by the host
wrapper, freed on return; never a residency-bearing allocation (the
vram_ledger needs no new row: it is transient, sub-0.01 GiB).

## 4. The host wrapper contract (the package's existing style)

* Full TORCH_CHECK validation: CUDA residency, dtypes (grad_y/x fp16 or
  bf16, SAME dtype; indices uint8; the LUT is NOT an input — this
  kernel never reads it), contiguity, `N % 128 == 0`, `K % 64 == 0`,
  `group_size in {32, 64}`, `indices.numel() == N*K/2`,
  `M > 0`; 16 B alignment with transparent clones
  (the backward's `backward_gemm_impl` pattern).
* Output pre-zeroed with `torch::zeros` (G-B5e's contract: unused codes
  and never-touched groups read exactly 0.0).
* Pass 2 launched on the same stream after pass 1; the workspace tensor
  is a local; the return is `[ceil(N/gs), 16]` fp32.
* Dispatch: template on the activation dtype (`__half` /
  `__nv_bfloat16`) and `GS in {32, 64}` — the backward's 4-way
  instantiation shape.
* Debug surface: `FLUTE_LUTGRAD_CANARY` poisons sDW with the fp32 NaN
  pattern before the epilogue write (coverage proof: the scatter walk
  must consume every sDW cell — a surviving NaN is a decoder/scatter
  regression). Mirrors `FLUTE_BWD_CANARY`.

## 5. The autograd Function (W5-T03, contract recorded here)

`FusedQLoRAGEMMTrainLUT` (scripts/qlora_gemm.py), the `_LoRABranchFn`
pattern applied to the codebook:

* forward: `Y = qgemm_per_group_lut(xh, indices, lut16,
  bitwidth, group_size, "idx4")` where `lut16 = lut_master_fp32.half()`
  — the fp32 master is the trainable Parameter, the fp16 cast is the
  kernel operand (never saved: PROMPT §10 rule 5 — recomputed in
  backward). `x` is saved in its fp16 operand form (the kernel input
  itself, not a derived cast). `indices` and the geometry travel as ctx
  attributes (the FusedQLoRAGEMM convention).
* backward(grad_y): `grad_x = fused_backward_gemm(grad_y16, indices,
  lut16', N, K, gs)` (the EXISTING kernel, .to(ctx.x_dtype) on return)
  and `grad_lut = lut_grad_scatter(grad_y16, x16, indices, N, K, gs)`
  — cast `.to(fp32)` into the master's dtype. `grad_indices = None`
  (frozen integers). The grad w.r.t. the bitwidth/group_size/N/K
  integer args is None.
* `fused_gemm_eligible(module, lut_trainable=False)`: the new
  `lut_trainable=True` mode returns True only when BOTH kernels are
  available (the forward AND flute_train_kernels.lut_grad_scatter) —
  the trainer's `--lut-path kernel` selects on it (W5-T05).
* CPU reference twin (the oracle's implementation): the pure-torch
  composition over the same inputs — `W = reference_dequant(blob, lut,
  N, K, gs, bw)`, `Y = X @ W.t()` (forward), `dW = grad_Y.t() @ X`,
  `grad_lut = zeros(...).scatter_add_` keyed by the logical indices
  (the closed form of §1) — fp32 end-to-end. This twin IS the G-B5a
  oracle; it lives in scripts/qlora_gemm.py next to the Function.

## 6. The differential gates (defined BEFORE the code — W5-T04)

| Gate | Check | Pass criterion | Where |
|---|---|---|---|
| G-B5a | oracle parity: the kernel's grad_lut vs the closed-form scatter on random inputs at the REAL module shapes ((12288,4096), (4096,12288), (4096,4096); gs 32 and 64; M in {2048, 32768}) | `torch.testing.assert_close(rtol=0, atol=0)` — fp32-exact on the CPU twin arm (the reference IS the definition); the CUDA arm compares against the twin with rtol 2e-6 (fp32 GEMM-order tolerance), pinned in the test | CPU always (twin); CUDA on the box |
| G-B5b | bit-exact twin: two consecutive runs of `lut_grad_scatter` on identical inputs | `torch.equal` — byte-identical outputs (the deterministic two-pass is what makes this assertable; atomics would flake it) | CUDA (box); the determinism DECISION is asserted on CPU by construction |
| G-B5c | NaN canary: inject one NaN into grad_y; every (g,c) whose receptive (n,k) set touches it must propagate NaN; untouched cells stay finite | exact propagation map (no `isfinite` clamps inside the math — the tripwire's EMA canary depends on honest NaNs) | CUDA (box) + the twin arm on CPU |
| G-B5d | finite differences: for sampled LUT entries (the forward Y is a function of LUT), `(L(x, lut + eps e_gc) - L(x, lut - eps e_gc)) / (2 eps)` vs the analytic scatter of dW | relative error <= 1e-3 at eps = 1e-2 (fp16 forward rounding sets the floor) | CPU (the Function's reference path); CUDA arm optional |
| G-B5e | zero semantics: a zero grad_y gives an all-zero grad_lut; codes never occurring in the blob give exact 0.0 rows; groups beyond the touched n-range stay 0.0 | `torch.equal(zero)` / exact 0.0 comparisons | CPU (twin) + CUDA |
| G-B5f | perf budget: wall time of `lut_grad_scatter` vs `fused_backward_gemm` at the box geometry (M=32768, N=K=12288, gs=64, fp16), warm caches, 20-run median | <= 1.3x the backward kernel's time (same mma FLOP count + the scatter epilogue; the L2-wave discipline mirrors the backward's) | box only |

Gate-to-test mapping is fixed at W5-T04 (tests/test_lut_gradients.py);
the CUDA arms are `skipif(not torch.cuda.is_available())` with the
CPU arms always running (the PROMPT §10 rule 9 discipline: the oracle
is the definition, the kernel must match it).

## 7. Out of scope (recorded to prevent drift)

* No changes to `flute_extended` (frozen — the forward kernel is used
  as-is).
* No bf16 LUT support (the artifact contract pins the LUT fp16).
* No dL/dW materialization to DRAM (the (N,K) transient is exactly
  what this kernel removes).
* No trainer default flip: `--lut-path` stays `reference` until the box
  G-B5 verdict (W5-T05 records the flip rule in RUNBOOK.md).

## 8. The idxN extension (W9 — sub-4-bit widths, DEQUANT_SPEC section 8)

The W9 backward-kernel family extends BOTH kernels of this package to
the unified idxN blob (flute_extended/flute_extended/idxN.py) at bit
widths B in {1, 2, 3}, following the forward campaign's conventions:
new `_sub4` kernels templated on `<T, B, GS, kTwin>`, the 4-bit kernels
and their binaries UNCHANGED (the regression contract), runtime
dispatch through a host-side `bitwidth` argument with the layout
agreement gate (`indices_layout == "idx{bitwidth}"`).

### 8.1 The width-parameterized blob walk

The invariant that makes the extension a "same style" one: the pair
positions of the 4-bit walk are WIDTH-INDEPENDENT. Each thread's share
of one N-step is always 16 k-PAIRS; the width only changes how many
bytes those 16 pairs occupy and how many codes the LUT carries:

| quantity | b=4 (legacy) | general B |
|---|---|---|
| tile (128 rows x 64 k) | 4096 B | 1024*B B |
| half-tile (one N-step) | 2048 B | 512*B B |
| (wx, lane) chunk       | 64 B   | 16*B B |
| thread segment         | 16 B (uint4) | 4*B B (B u32 words) |
| pair field             | byte j (nibbles) | 2*B bits at bit 2*B*j, LSB-first |
| LUT row / scatter codes| 16 | 2^B |

The segment offset `blob_segment_offset_sub4<B>(n_base, k0, K, tid)` is
the 4-bit arithmetic with every byte count scaled by B/4 (at B=4 it
reduces to the legacy formula verbatim — asserted by the simulator's
level D and the partition gate). Pair j of the segment maps to the SAME
`(n_local, k)` as byte j of the 4-bit walk:
`n_local = v*16 + d*8 + (chunk>>2)`, `k = 2*(seg*8 + (chunk&3) + 4*s2)`
with `v = j>>2, d = (j>>1)&1, s2 = j&1`.

Field extraction (`decode_pair_sub4<B>`): b=1 reads bits `2j, 2j+1` of
word 0; b=2 reads the 4-bit field at `4j` (word `j>>3`, shift `4j&31`,
never spanning); b=3 reads the 6-bit field at `6j` — it spans a word
only when `shift+6 > 32` (shift in {28, 30}), and a spanning field
ends strictly inside the segment (the last pair ends exactly at bit
32*B), so `q[w0+1]` is always in range. This is the same arithmetic as
the forward's `dequant_tile_sub4` (flute_extended
kernel_cutlass_streaming.cu).

### 8.2 The kernels

* `fused_backward_gemm_sub4_kernel<T, B, GS, kTwin>` — the 4-bit
  kernel's geometry, pipeline, mma phase, epilogue and debug surfaces
  (FLUTE_BWD_CANARY / FLUTE_BWD_TRAP / the scalar twin) verbatim; the
  segment prefetch is B u32 `.nc` loads (the forward's
  `q_fd_prefetch_sub4` discipline) and the dequant gathers a 2^B-entry
  LUT row. sW, the swizzle, the mma phase and the epilogue are
  width-independent (the forward legacy-sub4 contract).
* `lut_grad_scatter_sub4_kernel<T, B, GS, kTwin>` — the deterministic
  two-pass UNCHANGED in structure: the scatter keys become 2^B codes
  (`partial[LBM/GS][2^B]`), the warp_part layout and the workspace slot
  stride scale with 2^B, and `lut_grad_reduce_sub4_kernel<GS, B>`
  writes `[n_groups, 2^B]` (the 16-thread-per-group granularity kept;
  threads with c >= 2^B retire). No atomics anywhere — the §2 decision
  is width-independent, so G-B5b's torch.equal stays assertable.

Instantiation matrix: T in {__half, __nv_bfloat16} x B in {1,2,3} x GS
in {32, 64} x kTwin — 24 scatter-kernel instances per file, plus the
frozen 4-bit ones. Register pressure and smem are IDENTICAL to the
4-bit kernels (same 48 KiB + 512 B dynamic smem, same
`__launch_bounds__(128, 2)`; the partials are smaller at B < 4).

### 8.3 The host contract

`fused_backward_gemm(grad_y, indices, lut, N, K, group_size,
bitwidth=4, indices_layout="")` — same for `backward_simple_twin`,
`lut_grad_scatter`, `lut_grad_scatter_twin`. Validation (loud, never a
silent remap): `bitwidth in 1..4`; `indices.numel() == N*K*bitwidth/8`;
`lut [ceil(N/GS), 2^bitwidth]`; a non-empty layout string must equal
`"idx" + bitwidth`. bitwidth=4 runs the byte-identical legacy path
(the 6-argument call form still works — the pybind defaults).

The Python wrapper (`flute_train_kernels/__init__.py`) re-validates the
width/layout pair and translates a stale pre-idxN extension's
TypeError into the loud rebuild message (`idxn_available()` probes the
pybind signature for the bitwidth parameter — the path resolution in
qlora_gemm.fused_gemm_eligible gates sub-4 kernel-path attach on it).

### 8.4 The reference layer (the oracle lineage)

`scripts/qlora_fallback.py::dequant_idxn_torch` — the flat-PAIR walk
(position algebra worked backwards from the kernel's segment
arithmetic, the same lineage discipline as `dequant_idx4_torch`, NOT
the packer's map): `byte0 = (chunk_base*8 + 2*B*r3) >> 3`, the 2*B-bit
field from the two-byte little-endian window (span-guarded — the last
pair's field ends at the blob's bit end). At bitwidth=4 the walk
degenerates to the byte walk exactly and delegates to the frozen
`dequant_idx4_torch` (the test-suite identity gate).

`scripts/qlora_gemm.py::_lut_grad_scatter_reference(..., bitwidth)` —
the closed-form scatter over 2^B codes, filling the logical index array
by the pair walk and accumulating in ONE flat `(n*K + k)`-order
index_put_ — the same order as the 4-bit oracle (bit-exact against any
equally-ordered oracle). `train_lut_reference_forward/backward` and the
autograd Functions (`FusedQLoRAGEMM`, `FusedQLoRAGEMMTrainLUT`) carry
the bitwidth end-to-end (the forward call uses
`indices_layout=f"idx{bitwidth}"`, the backward passes it to the
kernels; the reference fallback arms dequantize at the same width).

### 8.5 The gates (W9)

| Gate | Check | Where |
|---|---|---|
| G-B1n | sub-4 fused_backward_gemm vs the pair-walk reference GEMM (cos > 0.9999, rel < 1e-3/5e-3-bf16) at real shapes, ragged M, adversarial LUT | CUDA (box) |
| G-B2n | the scalar-fragment twin bit-exact vs the production path at every width | CUDA (box) |
| G-B5an | lut_grad_scatter vs the pair-walk reference oracle (the 4-bit tolerance) | CUDA (box) |
| G-B5bn | consecutive runs byte-identical + the twin bit-exact at every width | CUDA (box) |
| IdxnRefusal | blob byte count / LUT width / layout disagreement all raise | CUDA (box) |
| G-B4n | finite differences through FusedQLoRAGEMM at sub-4 widths | CUDA (box) |
| IdxnReferenceLayerCPU | dequant == canonical gather; the scatter == the naive bincount oracle; the b=4 pair-walk identity; the wrapper refusals | CPU (always) |
| IdxnFunctionContractCPU | the reference forward/backward twins at sub-4 widths | CPU (always) |
| simulator | scripts/lutgrad_sim.py levels A/D at every width (the exact value semantics + the packer inverse + the segment partition) | CPU (always) |

The 4-bit gates (G-B1..G-B5f) run unchanged — the regression contract.
