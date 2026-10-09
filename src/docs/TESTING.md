# Testing

The gate suite: ~190 CPU-executable tests (no GPU, no CUDA toolkit
required) plus two GPU-side standalone scripts that run only on the
box where the extension is built. `conftest.py` (root and `tests/`)
implements the collection policy that keeps a CPU-only checkout green.

## 1. Running

```bash
python -m pytest -q                 # the whole CPU suite
python -m pytest tests/test_gemv_splitk.py -q    # one file
python -m pytest -q -k "toy"        # the toy/differential family
```

On the GPU box, additionally:

```bash
python flute_extended/test_flute.py           # multi-backend + differential gates
python flute_extended/test_qwen_weights.py    # real-weights oracle
python scripts/verify_gemv.py                 # deployed split-K GEMV numerics
```

## 2. The suite map

| File | Gates |
|---|---|
| `test_gemv.py`, `test_gemv_fht.py`, `test_gemv_splitk.py`, `test_gemv_merge.py` | the decode GEMV family: numerics vs the CPU reference, split regimes, residual ranks, the merges, FHT fusion |
| `test_dequant_reference.py` | the pure-torch dequant oracle for every width/layout |
| `test_attn_kernel.py` | the SM_86 attention kernel (parity vs reference) |
| `test_fht.py` | FHT round-trips, the explicit-matrix ground truth, block-diagonal K |
| `test_merge.py` | merged-launch equivalence (QKV / MLP vs separate launches) |
| `test_gs_grid_parallel.py` | the full group-size grid × kernel paths |
| `test_palettized_modules.py`, `test_palettized_embedding.py` | the routing/wiring layer: loader verification, route gates, graph capture |
| `test_capture_store.py`, `test_streaming_calibration.py`, `test_interleave_resume.py` | the calibration pipeline: store formats, windowed capture, crash-safe resume |
| `test_auto_compression.py`, `test_toy_joint_selection.py`, `test_toy_lut2bit_verify.py`, `test_toy_nmse_verify.py`, `test_toy_unified_verify.py` | the auto resolver's decisions on pinned spectra |
| `test_toy_heads.py`, `test_toy_head_capture.py`, `test_toy_real_geometry.py`, `test_toy_rotation_fold.py`, `test_toy_hybrid_integration.py` | the toy drivers: head geometry, real-ratio cells, the rotation/AWQ fold, hybrid integration |
| `test_lut_gradients.py`, `test_two_stream_training.py`, `test_dual_stream.py`, `test_joint_trainer.py`, `test_training_fold_consistency.py`, `test_layerwise_loss.py` | the training stack: LUT gradients, two-stream autograd, the joint trainer, fold consistency |
| `test_eval_common.py`, `test_eval_greedy_match_units.py`, `test_eval_greedy_match_capture.py`, `test_eval_greedy_match_graphs.py`, `test_eval_ppl.py`, `test_eval_verdicts.py` | the eval plane: protocol units, graph capture, PPL, verdicts |
| `test_rotation_awq_composition.py` | the load-time fold compensation (the `M = D T D⁻¹` identity) |
| `test_rank_alloc.py`, `test_sensitivity_rank.py`, `test_auto_compression.py` | rank allocation + sensitivity |
| `test_qlora_wrapper.py`, `test_optimizer.py`, `test_data.py`, `test_dual_stream.py` | QLoRA wrapper, Muon, dataset plumbing |
| `test_vram_ledger.py`, `test_check_gpu_contract.py`, `test_kernel_status.py`, `test_head_only_pass.py` | the ledgers and mechanical contract gates |
| `_toy.py` | the shared toy fixtures |

The two GPU-only scripts (`flute_extended/test_flute.py`,
`flute_extended/test_qwen_weights.py`) import `_C` at module scope;
collecting them without a built extension is a collection error by
design, so the root `conftest.py` excludes them on CPU boxes — they
stay byte-identical and run explicitly on the box.

## 3. What the tests pin (and what they deliberately do not)

- **Numerics**: every kernel's output vs the fp32 reference chain on
  CPU-computable inputs — the same ground truth `verify_gemv.py`
  re-proves on the box with the real deployed configs.
- **Determinism**: fixed association order (split-K partial folds,
  j-word folds) is pinned, not just tolerance — decode must be
  reproducible across CUDA-graph replays.
- **Formats**: pack/unpack round-trips at every width; idxN(4) ≡ idx4
  byte-identity; the blob map vs the kernel's decode.
- **Protocols**: the eval's graph capture/verify loop, the PPL
  windowing, the OOM ladder.
- **Decisions**: the auto resolver's choices on fixed spectra (the
  ledger, the budgets, the refine gate).

They do NOT pin wall-clock performance (clocks, L2 state, and driver
skew make that a box concern — the probe owns it), and they do not
exercise the CUTLASS dense baseline (build-time optional).

## 4. Adding a gate

Follow the existing shape: deterministic tensors from a fixed seed,
the pure-torch reference as ground truth, one behavior per test, and a
failure message that names the contract clause it pins. GPU-requiring
gates belong in the box scripts, not in `tests/`.
