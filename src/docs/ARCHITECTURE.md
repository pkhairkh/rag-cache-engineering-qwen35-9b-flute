# Architecture

How the pieces of this repository fit together, from a dense Hugging Face
checkpoint to CUDA-graph decode on the A10G.

```
Qwen/Qwen3.5-9B (dense FP16)
        │
        │  scripts/capture.py + scripts/calibrate_real_text.py
        ▼
calibration artifacts (Hessians, Grams, activation samples)
        │
        │  scripts/palettize_qwen3_5_9b.py  (recipe per weight, Lloyd/GPTQ/GPTVQ,
        │                                     mixed-radix, residual SVD, idxN blobs)
        ▼
palettized artifacts (idx{b} blobs, lut_scalar, resA/resB, metadata.json)
        │
        │  scripts/palettized_modules.py  (PalettizedLinear / FusedPalettizedMLP
        │                                   routing → kernel dispatch)
        ▼
flute_extended._C  (CUDA kernels: streaming GEMM, decode GEMV family,
                    FHT, merged QKV/MLP launches)
        │
        ▼
scripts/modeling.py (hybrid linear/full attention forward, CUDA-graph decode)
        │
        ├── scripts/probe_decode_routing.py   (per-module route + bandwidth)
        └── scripts/eval_greedy_match.py      (greedy match, tok/s, PPL, VRAM)
```

## 1. The three code packages

**`flute_extended/` — inference kernels (CUDA + Python).**
The runtime heart. The C++/CUDA extension `_C` exposes one family of
entry points per regime (bound in `src/bindings.cpp`, declared in
`include/flute/entrypoints.h`):

- `qgemm_per_group_lut` — the batched/prefill tensor-core GEMM family
  (`kernel_streaming.cu`), M ≥ 16, with the `cutlass_streaming`,
  `debug_simple` (differential twin), `naive`, and optional `cutlass_dense`
  reference backends;
- `qgemm_gemv_stream` / `qgemm_gemv_fht_stream` — the M = 1 memory-bound
  streamer (`kernel_gemv.cu`), plain and FHT-fused;
- `qgemm_gemv_splitk_stream` / `qgemm_gemv_splitk_fht_stream` — the
  split-K + double-buffered decode GEMV (`kernel_gemv_splitk.cu`),
  residual rank ≤ 256 fused;
- `qgemm_gemv_multi` — the grouped multi-blob launch (QKV merge,
  `kernel_gemv_multi.cu`);
- `qgemm_gemv_mlp` — the merged gate+up launch with the SiLU·mul epilogue
  (`kernel_gemv_mlp.cu`);
- `fht_forward` / `fht_backward` / `fht_inplace_` / `fht_forward_awq` — the
  Fast Hadamard Transform kernels (`kernel_fht.cu`).

Python side: `flute_extended/idxN.py` (the ONLY producer/consumer of
sub-4-bit packed blobs), `flute_extended/idx4.py` (canonical 4-bit
packer, byte-identical special case of idxN), `fht.py`, `example.py`,
`benchmark_kernel.py`, and the GPU-side standalone gates
`test_flute.py` / `test_qwen_weights.py`.

**`flute_train_kernels/` — training kernels (CUDA).**
The backward side for QLoRA-style training on palettized weights:
`kernel_backward_gemm.cu` (fused `grad_X = grad_Y @ W` with on-the-fly
dequant) and `kernel_lut_grad.cu` (the dL/dLUT scatter). Every entry
takes `(bitwidth, indices_layout)`; 1/2/3-bit paths share the family.

**`scripts/` — the pipeline and the measurement kit.**
Python modules (imported by each other and by the tests) plus CLI entry
points: `capture.py`, `calibrate_real_text.py`,
`palettize_qwen3_5_9b.py` (the engine), `palettized_modules.py` (the
runtime routing/wiring layer), `modeling.py` (the forward),
`probe_decode_routing.py`, `eval_greedy_match.py`, `eval_ppl.py`,
`measure_decode.sh`, `doctor.py`, and supporting tools. The full map is
in [RUNBOOK.md](RUNBOOK.md).

## 2. The data contract between the stages

Every palettized tensor is **idxN streams + LUT (+ optional residual)**,
consumed by exactly one kernel call per stream (`docs/QUANTIZATION.md`
defines the byte-level format):

```
W[n, k]  =  LUT[n // group_size, idx[n, k]]            # one stream
W_eff    =  W1 + W2                                     # two-stream tensor
y        =  x @ W_eff^T + (x @ resB^T) @ resA^T         # + residual branch
```

The palettizer writes blobs in the kernel-consumable permutation
(`flute_extended/idxN.py`), records per-stream sha256 + geometry in
`metadata.json`, and the loader (`palettized_modules.py`) verifies both
before the first forward. A mismatch is a hard error; there is no
silent fallback to dense.

Two model-specific folds sit between the checkpoint and the kernels and
are compensated exactly at load time (never numerically):

- **rotation + AWQ norm-gain composition** — recovered from
  `norm_gain_edits.json`, applied as the exact matrix identity
  `M = D T D⁻¹` on the affected tensors;
- **FHT boundary fold** — the K×K Hadamard rotation of the FLUTE scheme is
  either pre-applied to the activation (plain GEMV routes) or fused inside
  the decode kernel (`_fht` routes, see [ROUTING.md](ROUTING.md)).

## 3. The runtime layer (`scripts/palettized_modules.py`)

`PalettizedLinear` swaps in for every `nn.Linear` whose weight has a
palettized artifact set. At decode (M = 1) each module walks a gate chain
and picks one of: the merged-group launch (if its group merged), the
FHT-fused split-K GEMV, the plain split-K GEMV, the wide-preference
streamer (lm_head), or the two-launch fallback. Every route decision is
recorded with the reason that the alternatives were refused — the probe
prints exactly this census. `FusedPalettizedMLP` wraps gate+up of each
MLP block into the merged SiLU·mul launch when the specs allow.

Decode itself runs inside a CUDA graph: one capture, then replays per
token — the host dispatch cost is out of the measurement entirely
(`docs/EVALUATION.md`).

## 4. Hybrid attention wiring (`scripts/modeling.py`)

Qwen3.5-9B interleaves 24 linear-attention layers (gated delta net:
`in_proj_qkv`, `in_proj_z`, `out_proj`, short conv) with 8 full-attention
layers every 4th position (`q/k/v/o_proj`, GQA 16/4 heads, head_dim 256,
Q+gate packed in `q_proj`). `scripts/modeling.py` implements this forward
on top of the palettized modules, with the exact geometry pinned in
[MODEL_GEOMETRY.md](MODEL_GEOMETRY.md). `scripts/attn_sm86.py` carries the
SM_86 attention kernels used by the full-attention path.

## 5. Verification spine

Every stage carries its own differential gate, and the chain composes:

1. **packers**: `idxN.self_test()` — pack/unpack round-trip, all widths;
2. **kernels vs math**: `tests/test_gemv*.py`, `tests/test_dequant_reference.py`
   (CPU ground truth), `flute_extended/test_flute.py` (GPU, bit-exact
   three-way `idxN == kernel-internal == debug_simple`);
3. **real weights**: `flute_extended/test_qwen_weights.py` (GPU, against
   the pure-torch reference on actual checkpoint tensors);
4. **runtime wiring**: `tests/test_palettized_modules.py`,
   `tests/test_eval_*` (loader, routing, graph capture, eval protocol);
5. **box numerics**: `scripts/verify_gemv.py` (the deployed kernel vs the
   fp32 reference on every deployed rank/split class).

`docs/TESTING.md` maps the suite; `conftest.py` keeps the CPU box green by
excluding the two GPU-only standalone scripts from collection.
