# idxN Training Support - VERIFIED ✅

## ✅ ALL WIDTHS SUPPORTED (GPU-box verified 2026-10-03)

The idxN kernel family (1/2/3/4-bit) now supports **both inference and training**.

---

## W14: Training vs the Fold (rotation + AWQ + residual) — VERIFIED ✅

The training pipeline is fold-consistent with the deployment on every
seam. Grounded in palettized-folder JSONs of the box's exact legacy
shape (`metadata.json` rotation/awq/residual records without
`fold_order`, `norm_gain_edits.json` with alpha>0 consumers):

| Seam | Contract | Gate |
|------|----------|------|
| `materialize_student_layer` | recovers the AWQ scales from the PRISTINE gains (pre-edit window) and hands them to the module swap — the student trains against the SAME compensated fold (M = D T D⁻¹) the deployment serves; `awq_compensation=False` reproduces the scrambled arm | `test_training_fold_consistency` G0/G1 |
| `QLoRALinear.forward` | computes the FOLD input once (`base._rotate_input`); the fused-flute and torch-cached GEMMs, the resA/resB branch and the LoRA branch ALL consume it — the pre-W14 code GEMMed the raw input against the fold-space codebook and trained the lora in a frame the merge cannot reproduce | G3 |
| export norm gains | `_export_norm_edits` PINS the frozen compensation (`awq_scales/<norm>.npy`, sha-pinned, `entry["awq_scale_file"]`); `_recover_awq_scales` prefers the record over the (1+w)/(1+w') diff, which the trained gain silently moves | G5 |
| `_verify_layer_roundtrip` | receives the resident modules' scale map — the roundtrip compares the compensated quantity | G6 |
| export polish | `_fold_polish_frame`: the target is the FOLDED teacher weight minus the fold-space residual, the Gram rides the fold congruence (`pmod.fold_input_gram`) | G7 |
| `qlora_merge` | reads + ABSORBS resA/resB into the merged weight and strips the stale records (the pre-W14 merge silently dropped the residual and left dangling records); the re-palettization Gram is frame-transformed; a rotated+AWQ tensor without a pinned s refuses loudly | G4/G4b |

The trainables under a fold: LUT masters + norm gains + lora, with the
compensation s FROZEN (a fixed linear reparameterization of the norm
gain — gradient-sound); indices, residual factors and the dense
remainder stay frozen.

---

## Compatibility Matrix

| Operation           | 1-bit | 2-bit | 3-bit | 4-bit |
|---------------------|-------|-------|-------|-------|
| Forward (inference) | ✅    | ✅    | ✅    | ✅    |
| Backward (training) | ✅    | ✅    | ✅    | ✅    |
| LUT gradients       | ✅    | ✅    | ✅    | ✅    |
| QLoRA fine-tuning   | ✅    | ✅    | ✅    | ✅    |

---

## What Changed

### W9 Campaign (2026-10-03)

**Implemented backward/gradient kernels for sub-4-bit:**
- `flute_train_kernels/src/kernel_backward_gemm.cu` — Fused backward GEMM for B=1,2,3
- `flute_train_kernels/src/kernel_lut_grad.cu` — LUT gradient scatter for B=1,2,3
- Python bindings extended with `bitwidth` parameter
- All kernels verified on NVIDIA A10G (sm_86)

**Performance (box-measured):**
- 1-bit: 52.2 TFLOPS (29% faster than 4-bit)
- 2-bit: 54.3 TFLOPS (34% faster than 4-bit)
- 3-bit: 46.0 TFLOPS (14% faster than 4-bit)
- 4-bit: 40.5 TFLOPS (baseline)

**Why sub-4 is faster:** Less blob data to load (4*B vs 16 bytes per N-step)

---

## What You Can Do

### ✅ All Operations Supported (All Widths)

- **Inference**: Forward pass through palettized layers
- **Evaluation**: Perplexity, accuracy metrics, quality gates
- **Export**: Save/load palettized models
- **QLoRA fine-tuning**: Gradient computation through quantized weights
- **LUT gradient descent**: Updating LUT entries via backprop
- **Training scripts**: `trainer.py`, `qlora.py` (all widths)

---

## Technical Implementation

### Backward Kernel Design

The backward kernels extend the 4-bit design with template specialization:

```cpp
// Kernel signature
__global__ void fused_backward_gemm_sub4_kernel<T, B, GS, kTwin>(
    const T* grad_y,
    const T* x,
    const uint8_t* blob,
    float* workspace,
    int M, int N, int K,
    int bitwidth  // 1, 2, 3, or 4
)
```

**Key innovations:**
- Width-independent (n_local, k) walk
- Template specialization for each B ∈ {1,2,3}
- 2^B-entry LUT rows
- Same 0-spill guarantee as 4-bit

---

## Usage

### Python API

```python
import flute_train_kernels as ftk

# Backward GEMM (any bitwidth)
grad_x = ftk.fused_backward_gemm(
    grad_y, x, indices, lut, group_size,
    bitwidth=2,  # 1, 2, 3, or 4
    indices_layout="idx2"  # must match bitwidth
)

# LUT gradients
lut_grad = ftk.lut_grad_scatter(
    grad_y, x, indices, group_size,
    bitwidth=2
)
```

### Training Example

```python
import qlora_gemm

# QLoRA with 2-bit quantization
y = qlora_gemm.FusedQLoRAGEMMTrainLUT.apply(
    x, indices, lut, bitwidth=2, group_size=32, N, K
)
loss = y.sum()
loss.backward()  # Gradients flow through 2-bit quantized weights
```

---

## Performance Comparison

### Backward Pass TFLOPS (A10G, M=N=K=12288, GS=64)

| Width | Time (ms) | TFLOPS | Speedup vs 4-bit |
|-------|-----------|--------|------------------|
| 1-bit | 189.52    | 52.2   | 1.29×            |
| 2-bit | 182.40    | 54.3   | 1.34×            |
| 3-bit | 215.05    | 46.0   | 1.14×            |
| 4-bit | 244.34    | 40.5   | 1.00× (baseline) |

**Sub-4 kernels are faster** due to reduced memory bandwidth (smaller blobs).

---

## Verification

**All gates pass:**
- GB1nRealShapeOracle: ✅ Shape coverage
- GB2nDifferentialTwin: ✅ Bit-exact twins
- GB5anKernelParity: ✅ Kernel vs reference
- GB5bnBitExactTwin: ✅ Determinism
- IdxnRefusalGatesGPU: ✅ Validation
- GB4nAutogradFiniteDifference: ✅ Autograd contract

**See:** `reports/w9_backward_gates.md` for full evidence

---

## Migration Guide

### From 4-bit to Sub-4-bit Training

No code changes needed - just change the bitwidth parameter:

```python
# Old (4-bit only)
y = qlora_gemm.FusedQLoRAGEMMTrainLUT.apply(
    x, indices, lut, 4, group_size, N, K
)

# New (any width)
y = qlora_gemm.FusedQLoRAGEMMTrainLUT.apply(
    x, indices, lut, 2, group_size, N, K  # Just change bitwidth
)
```

---

## Historical Note

**Before 2026-10-03:** Only 4-bit training was supported (forward pass worked for all widths).

**After 2026-10-03:** Full idxN ecosystem complete - training and inference supported across all bit widths (1/2/3/4-bit).

**This document previously described a limitation that has been resolved.**
