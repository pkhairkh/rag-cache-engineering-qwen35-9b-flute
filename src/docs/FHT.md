# docs/FHT.md — the Fast Hadamard Transform kernel (W13, FLUTE extension)

## 1. What it is

`src/kernel_fht.cu` + `include/flute/fht.cuh` + `fht.py` (project root)
replace the Hadamard boundary fold's explicit K x K rotation multiply

```
x_rot = x @ T,    T = blockdiag_b( H_b @ diag(s_b) / sqrt(b) )
```

with an O(K log K) butterfly. The requirement's numbers (measured on the
A10G box, pre-W13): the old class-level `_rot_cache` held ~1.5 GB of
fp16 rotation matrices across the 8 unique (seed, k) pairs, and every
forward paid a pageable CPU -> GPU copy of up to 576 MB (K=12288). The
FHT stores a (K,) sign vector (~16 KB at K=4096), never leaves the GPU
after the first per-device cache fill, and turns O(K^2) compute into
12-13 butterfly stages.

## 2. The math (pinned to THIS repo's conventions)

The production `_rot_matrix` builds `hadamard(k) * s.view(1, k) /
sqrt(k)` — the sign vector scales the COLUMNS:

```
T[i, j] = H[i, j] * s[j] / sqrt(b)          per power-of-two segment b
```

so

```
forward:   x @ T  = ((x @ H) * s) / sqrt(b)     signs on the OUTPUT
adjoint:   y @ T^T = ((y * s) @ H) / sqrt(b)    signs on the INPUT
```

The requirement document's sketch "(x * s) @ H / sqrt(K)" is the
TRANSPOSED convention (diag(s) @ H = T^T, not T). This implementation
pins the repo's column-scaled T — the one the artifacts encode; the
unit tests compare against the explicit `x @ T` ground truth
(`fht.build_rotation_matrix`, byte-identical to `_rot_matrix`).

H is symmetric and H @ H = b * I, so the adjoint is also the gradient:
the autograd backward of `fht_apply` is one more butterfly with the
signs on the input side (`fht_backward`), and `fht_adjoint(fht(x)) ==
x` exactly.

Block-diagonal K (production down_proj: K=12288 = 8192 + 4096, mirroring
`_rot_matrix`'s descending binary decomposition): each power-of-two
segment transforms independently; the kernel iterates segments in
shared memory, reusing the buffer.

## 3. The kernel

One CUDA block per row (grid.x = M); THREADS in {128, 256, 512, 1024}
chosen from the largest segment (>= 4 elements per thread); the active
segment is staged in dynamic shared memory as fp32 (the requirement's
"FP16 input/output, FP32 accumulation") and walked by the classic
iterative FWHT (stage length doubling, sums into the low element,
differences into the high element — the EXACT pairing of the Sylvester
Hadamard, verified bit-level against the matmul in tests/test_fht.py).

Resource accounting (sm_86, docs/PTX_NOTES.md section 1 — the W5
checklist):

* dynamic smem = 4 * b_max bytes: 16 KiB (K=4096), 32 KiB (K=12288's
  8192-segment) — under the 48 KiB default; a 16384-element segment
  (64 KiB) opts in via cudaFuncSetAttribute (checked, loud);
* `__launch_bounds__(THREADS)` on every instantiation; the working set
  is smem, registers are a handful;
* occupancy at THREADS=1024 / 16 KiB: thread-limited 1 CTA/SM for the
  one-row decode launch (the < 10 us latency target is met by launch +
  12 stages), smem admits 2 CTAs/SM at the 32 KiB segment;
* global traffic: K scalars in + K scalars out per row + the K signs
  read once per row-block;
* bank behaviour: stage len=1 pairs read stride-2 words (2-way
  conflict); the remaining stages are conflict-free or 2-way — noise
  across 12 stages (the kernel is launch-latency bound at decode M).

In-place variant: safe because a segment is fully staged in smem before
any of its columns are written back (segments touch disjoint column
ranges). Autograd: `fht_apply` routes through a custom Function whose
backward is the adjoint kernel; in-place refuses `requires_grad` input
loudly.

## 4. The Python layer

`flute_extended/fht.py` (PROJECT ROOT — deliberately outside the
`flute_extended` package: the package __init__ imports `_C` at module
import time, and the FHT must stay importable on CPU boxes where the
extension is not built; `scripts/palettized_modules.py` loads it
standalone through the same spec_from_file_location pattern as
idxN.py):

* `rotation_signs(K, seed)` — the per-tensor sign draw, bitwise the
  palettizer's (seed + 4242);
* `fht_apply(x, signs, backend=...)` / `fht_adjoint(v, signs, ...)`
  / `fht_apply_` — autograd-aware entry points; `backend` in
  {"auto" (CUDA kernel when built + CUDA input, torch butterfly
  otherwise), "kernel" (loud refusal when unavailable), "reference",
  "matmul" (the explicit x @ T — differential testing only)};
* `segments(K)`, `hadamard_matrix(K)`, `build_rotation_matrix(K, seed)`,
  `fht_reference` / `fht_reference_adjoint` (the Phase-1 deliverable).

Backends are also reachable from the module layer through the
`FLUTE_ROTATION` env knob (auto | reference | matmul).

## 5. Integration (the requirement's Phase 3)

`scripts/palettized_modules.py`:

* `PalettizedLinear.__init__` stores `rot_signs` (K,) fp32 +
  `awq_scale` + `fold_order` — the K x K `_rot_cache` is GONE
  (`_explicit_rot_T()` materializes the matrix on demand for the
  matmul fallback / differential tests only);
* `forward` rotates the input once, on BOTH execution paths (the
  pre-W13 reference path GEMMed the un-rotated x against the
  fold-space weight — a rotated-space output; its docstring claimed
  `_quantized_weight` handled it, but forward calls
  `_quantized_weight_stream`, the per-stream fold-space dequant);
* `_quantized_weight` un-rotates through the adjoint FHT (the old
  `w @ rot_T` returned W * s_column — the sign-scaled original);
* the W10 train route no longer post-multiplies the OUTPUT by T (the
  double-rotation — also a shape error whenever N != K, e.g.
  down_proj);
* `PalettizedEmbedding` (the TIED head case) un-rotates the gathered
  rows through the adjoint before the final cast — the pre-W13
  consumer returned fold-space (scrambled) embedding rows;
* `load_tied_pair` accepts `embed_meta=None` (the head pass writes ONE
  artifact set under the lm_head name) and `resolve_module` resolves
  `model.lm_head.weight` / `model.embed_tokens.weight` (previously
  int("weight") — head artifacts were unloadable).

## 6. The AWQ composition (fold_order) — the greedy-decode fix

The 2026-10-04 box run folded the rotation BEFORE the AWQ scale
(W' = W @ T @ diag(s)). The deployed pipeline (norm folds diag(s)^-1
into the gain, the module rotates by T) then computes

```
y = x @ (D^-1 @ T @ D @ T^T) @ W^T      != x @ W^T
```

an O(1) input scrambling on every alpha>0 AWQ group (46/82 groups in
the production log) that the per-layer cosine gates are BLIND to (both
gate operands carry the same scrambler — the gate target is the
scrambled output). That is the reported "0.999 cos but garbage greedy
decode with repetitions". Proof: `scripts/diagnose_greedy_bug.py`
(old deploy cos 0.90-0.99 at alpha 0.083..0.333 while the old GATE
reads 1.000000).

Fixes:

* PRODUCER (`palettize_qwen3_5_9b.py::_palettize_tensor_core`): the
  AWQ transform now runs FIRST — W' = (W @ diag(s)) @ T — composing
  exactly with the deployed pipeline; `awq_info["fold_order"] =
  "awq_then_rotate"` is the provenance key. The gate at the same site
  becomes the exact deployment-fidelity measure (its target is the
  true output; the stale comment claiming "the AWQ scales already
  cancel the legacy way" is replaced by the corrected derivation).
* LOADER, for the already-produced 19.6 GPU-hour artifacts:
  `_recover_awq_scales` re-derives the per-channel s from
  norm_gain_edits.json captured against the PRISTINE from_pretrained
  model (s = (1+w_orig)/(1+w_edited); cross-checked against the
  tensor meta's s_min/s_max/s_mean at 2%), and the module serves the
  compensated input rotation M = D T D^-1 (scale -> FHT -> unscale —
  exact; proof claim 4). Load-time console evidence:
  `[ROT] recovered AWQ scales ...` and `[ROT] legacy rotate-then-AWQ
  fold order compensated on N tensor group(s)`.
* QKV splits: the fused meta's awq record governs every component;
  the per-component gates now rotate the retained sample (gate
  honesty, `rot_T` passed into `palettize_qkv_weight_split`).

## 7. Verification (the gates of record)

CPU (this box, green):

```
python3 scripts/diagnose_greedy_bug.py                  # claims 1-4
python3 -m pytest tests/test_fht.py -q                  # 21 passed (4 CUDA skips)
python3 -m pytest tests/test_rotation_awq_composition.py -q   # 15 passed
python3 -m pytest tests/test_toy_rotation_fold.py -q    # 10 passed (unchanged)
python3 -m pytest tests/test_palettized_modules2.py tests/test_palettized_embedding.py \
                   tests/test_toy_heads.py tests/test_merge.py -q
python3 flute_extended/fht.py                           # selfcheck PASS
```

GPU box (after `cd flute_extended && python setup.py build_ext
--inplace`):

```
cd flute_extended && python test_flute.py            # unchanged GEMM gates
cd flute_extended && python fht.py                   # CPU selfcheck on the box
python3 -m pytest tests/test_fht.py -q               # the 4 CUDA arms
   (kernel vs reference at 512/4096/12288 fp16+fp32, backward,
    in-place, latency smoke vs the <10us / <30us targets x3 margin)
python3 eval_greedy_match.py --artifacts-dir /home/ubuntu/qwen3_5_9B_palettized \
    --residual --forward kernel    # the requirement's integration gate:
                                   # exact_match_fraction should now be
                                   # high (the compensation serves the
                                   # 2026-10-04 artifacts correctly)
```

## 8. File map (vs EXTENSION_REQUIREMENT.md's list)

| requirement | repo (this round) | why |
|---|---|---|
| `flute_extended/kernel/fht.cuh` | `flute_extended/include/flute/fht.cuh` | the repo's header home (mma.cuh / dequant.cuh live there) |
| `flute_extended/kernel/fht.cu` | `flute_extended/src/kernel_fht.cu` | the repo's source home (setup.py's sources list) |
| `flute_extended/fht.py` | `flute_extended/fht.py` | as specified (project root — CPU-importable without the built `_C`) |
| `tests/test_fht.py` | `tests/test_fht.py` | as specified |
| `palettized_modules.py` | `scripts/palettized_modules.py` | the repo's module layer lives under scripts/ |
| `flute_extended/setup.py` | `flute_extended/setup.py` | + `src/kernel_fht.cu` in sources |
