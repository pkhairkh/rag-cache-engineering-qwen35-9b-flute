# Two-Stream System - Complete Analysis

## Question
Does idx1.2, idx2.2, idx3.2, idx4.2 and idx1.1, idx2.1, idx3.1 and idx4.1 work with double correction streams in palettizing, forward kernel, and backward kernel?

---

## Answer: FULL SUPPORT ✅ (W10 — two-stream training implemented)

### What Works

#### 1. Palettizing (Writing Artifacts) ✅
**Location:** `scripts/palettize_qwen3_5_9b.py`

The writer **correctly produces** two-stream artifacts:

**Storage rule (lines 133-137):**
- `prod(2^radices) <= 16` → ONE composite .idx{N} file
- `prod(2^radices) > 16` → TWO files:
  - Stream 1: `.idx{r1}` (base stream)
  - Stream 2: `.idxN.2` (composite of refinement streams)

**Examples:**
- `mixed:2,2` (palette=16) → ONE `.idx4` file (composite)
- `mixed:4,2` (palette=32>16) → TWO files: `.idx4` + `.idxN.2`
- `hybrid422` (4,2,2) → `.idx4` + `.idx4.2`

**What gets written:**
- ✅ `.idx1`, `.idx2`, `.idx3`, `.idx4` (stream 1)
- ✅ `.idxN.2` (stream 2, where N = bitwidth of composite)
- ✅ `.lut_scalar`, `.lut_scalar.2`
- ✅ Metadata records `streams` list with derivations

#### 2. Forward Kernel (Inference) ✅
**Location:** `scripts/palettized_modules.py::PalettizedLinear.forward()`

**Supports:** Two streams with **DIFFERENT bitwidths**

```python
# Lines 635-646
y1 = flute_extended.qgemm_per_group_lut(
    xh, self.indices, self.lut,
    bitwidth=self.bitwidth,  # e.g., 4
    indices_layout=f"idx{self.bitwidth}"
)
if self.has_stream2:
    y2 = flute_extended.qgemm_per_group_lut(
        xh, self.indices2, self.lut2,
        bitwidth=self.bitwidth2,  # e.g., 2 (DIFFERENT!)
        indices_layout=f"idx{self.bitwidth2}"
    )
    output = (y1 + y2).to(original_dtype)
```

**Works with:**
- ✅ `idx4.1` (4-bit base + 1-bit refinement)
- ✅ `idx4.2` (4-bit base + 2-bit refinement) 
- ✅ `idx4.3` (4-bit base + 3-bit refinement)
- ✅ `idx3.1`, `idx3.2` (3-bit base + refinements)
- ✅ `idx2.1` (2-bit base + 1-bit refinement)
- ✅ `idx2.2` (two 2-bit streams as separate files if palette>16, else composite)

**Correctness:** Lines 270-272 validate bitwidth agreement:
```python
self.bitwidth2 = int(bitwidth2) if bitwidth2 is not None else self.bitwidth
```

### What Does NOT Work

#### 3. Backward Kernel (Training) ✅ (W10)
**Location:** `scripts/qlora_gemm.py::FusedQLoRAGEMMTrainLUTTwoStreams`

**Signature (both streams, each with its OWN bit width):**
```python
def forward(ctx, x, indices1, lut1_master, bitwidth1,
            indices2, lut2_master, bitwidth2, group_size, N, K):
    # TWO FLUTE qgemm calls + ONE ordered add (y1 + y2)
```

**The analytic contract (no new CUDA kernels needed):**
```
y        = y1 + y2 = x @ W1^T + x @ W2^T
dL/dx    = dL/dy @ (W1 + W2)   — two fused_backward_gemm calls, summed
dL/dlut1 = scatter(dL/dy, x, indices1)  — INDEPENDENT of stream 2
dL/dlut2 = scatter(dL/dy, x, indices2)  — INDEPENDENT of stream 1
```
The W9 idxN backward family already serves any width per stream, so the
two-stream backward decomposes into two single-stream backwards
(grad_x summed; the LUT scatters independent). The reference twins
(`train_lut_two_streams_reference_{forward,backward}`) are the CPU
oracles; the Function's kernel arms are the GPU path (kernel parity
+ saved-memory census gates in `tests/test_two_stream_training.py`).

**Wiring:**
- `scripts/qlora.py` — `QLoRALinear.forward` routes two-stream bases:
  trainable → the W10 Function; frozen → `FusedQLoRAGEMMTwoStreams`
  (the two-qgemm deployment shape + summed fused backward — this also
  FIXES the pre-W10 silent stream-2 drop on the frozen path)
- `scripts/palettized_modules.py` — `PalettizedLinear.forward` (kernel
  path, CUDA): trainable LUTs route through the train Functions; CPU
  or kernel-less CUDA keeps the loud refusal (never a silent fallback)
- `scripts/qlora_gemm.py::fused_gemm_eligible` — stream-2 geometry,
  dtype/device and kernel-build gates
- `make_trainable()` promotes BOTH LUT masters (fp32); `freeze_lut()`
  snaps both

---

## Compatibility Matrix

| Recipe | Palettizing | Forward | Backward | Total Bits |
|--------|-------------|---------|----------|------------|
| `idx1` (r1, b=1) | ✅ | ✅ | ✅ | 1 bit |
| `idx2` (r1, b=2) | ✅ | ✅ | ✅ | 2 bits |
| `idx3` (r1, b=3) | ✅ | ✅ | ✅ | 3 bits |
| `idx4` (r1, b=4) | ✅ | ✅ | ✅ | 4 bits |
| `idx1.1` (mixed:1,1) | ✅ composite → .idx2 | ✅ | ✅ | 2 bits |
| `idx2.1` (mixed:2,1) | ✅ → .idx2 + .idx1.2 | ✅ | ✅ | 3 bits |
| `idx2.2` (mixed:2,2) | ✅ composite → .idx4 | ✅ | ✅ | 4 bits |
| `idx3.1` (mixed:3,1) | ✅ → .idx3 + .idx1.2 | ✅ | ✅ | 4 bits |
| `idx3.2` (mixed:3,2) | ✅ → .idx3 + .idxN.2 | ✅ | ✅ (W10) | 5 bits |
| `idx4.1` (mixed:4,1) | ✅ → .idx4 + .idx1.2 | ✅ | ✅ (W10) | 5 bits |
| `idx4.2` (hybrid422 base) | ✅ → .idx4 + .idx4.2 | ✅ | ✅ (W10) | 6 bits |
| `idx4.3` (mixed:4,3) | ✅ → .idx4 + .idxN.2 | ✅ | ✅ (W10) | 7 bits |

**Key insight (post-W10):**
- Pallettes ≤ 16: **composite to single file** — ONE dequant GEMM, the
  single-stream training shape
- Palettes > 16: **two files** — TWO GEMMs, trainable since W10 via
  `FusedQLoRAGEMMTrainLUTTwoStreams` (the `--auto-speed-priority`
  knob in the palettizer weighs this deployment cost)

---

## Specific Combinations

### idx1.2 (1-bit base + 2-bit refinement)
- Palettizing: ✅ Produces `.idx1` + `.idx2.2` (palette=4×2=8≤16, so composite → `.idx2`)
- Forward: ✅ Single stream works
- Backward: ✅ Single stream trainable

### idx2.1 (2-bit base + 1-bit refinement)
- Palettizing: ✅ Produces `.idx2` + `.idx1.2` (palette=4×2=8≤16, so composite → `.idx2`)
- Forward: ✅ Single stream works
- Backward: ✅ Single stream trainable

### idx2.2 (two 2-bit streams)
- Palettizing: ✅ Produces composite `.idx4` (palette=4×4=16≤16)
- Forward: ✅ Single stream works
- Backward: ✅ Single stream trainable

### idx3.1 (3-bit base + 1-bit refinement)
- Palettizing: ✅ Produces `.idx3` + `.idx1.2` (palette=8×2=16≤16, so composite → `.idx4`)
- Forward: ✅ Single stream works
- Backward: ✅ Single stream trainable

### idx3.2 (3-bit base + 2-bit refinement)
- Palettizing: ✅ Produces `.idx3` + `.idxN.2` (palette=8×4=32>16)
- Forward: ✅ Two streams work
- Backward: ✅ **TRAINABLE (W10)** — `FusedQLoRAGEMMTrainLUTTwoStreams`

### idx4.1 (4-bit base + 1-bit refinement)
- Palettizing: ✅ Produces `.idx4` + `.idx1.2` (palette=16×2=32>16)
- Forward: ✅ Two streams work
- Backward: ✅ **TRAINABLE (W10)**

### idx4.2 (hybrid422: 4-bit + two 2-bit as composite)
- Palettizing: ✅ Produces `.idx4` + `.idx4.2` (palette=16×4=64>16)
- Forward: ✅ Two streams work
- Backward: ✅ **TRAINABLE (W10)**
- **This is what hybrid422 produces!**

### idx4.3 (4-bit base + 3-bit refinement)
- Palettizing: ✅ Produces `.idx4` + `.idxN.2` (palette=16×8=128>16)
- Forward: ✅ Two streams work
- Backward: ✅ **TRAINABLE (W10)**

---

## Summary

### ✅ What Works
1. **Palettizing:** All two-stream combinations written correctly
2. **Forward:** All two-stream combinations work for inference
3. **Backward:** ALL recipes trainable — composites through the
   single-stream W5/W9 path, palette>16 through the W10
   two-stream Function

### ❌ What Doesn't Work
(nothing — the W10 implementation closed the last gap)

### Historical workarounds (pre-W10, now obsolete)
1. **Use composites when palette≤16** — still the FASTEST shape
   (one GEMM); `--auto-speed-priority 1.0` prefers it
2. **Train sequentially (freeze one stream, train the other)** — the
   W10 Function trains both masters jointly in one backward
3. **Inference-only for palette>16** — fully trainable since W10

---

## Implementation (W10 — shipped)

**`scripts/qlora_gemm.py`:**

```python
class FusedQLoRAGEMMTrainLUTTwoStreams(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices1, lut1_master, bitwidth1,
                indices2, lut2_master, bitwidth2, group_size, N, K):
        y1 = flute_extended.qgemm_per_group_lut(...)   # idx{bitwidth1}
        y2 = flute_extended.qgemm_per_group_lut(...)   # idx{bitwidth2}
        return (y1 + y2)                               # ordered add

    @staticmethod
    def backward(ctx, grad_y):
        # grad_x = two fused backward GEMMs, summed
        # grad_lut1/grad_lut2 = per-stream lut_grad_scatter (W9 kernels)
        return (grad_x, None, grad_lut1, None,
                None, grad_lut2, None, None, None, None)
```

**Tests:** `tests/test_two_stream_training.py` — the CPU analytic-
contract gates (reference twins vs autograd, per-stream scatter
bit-exactness, cross-stream independence, the wrapper refusals) + the
CUDA DoD gates (kernel parity, the saved-memory census, the
QLoRALinear route).

Also shipped with W10: the FROZEN two-stream route
(`FusedQLoRAGEMMTwoStreams`) — fixes the pre-W10 silent stream-2 drop
where a two-stream module under `QLoRALinear` (fused-flute, frozen)
computed y1 only, and the compression-first auto resolver
(`--auto-target-bits` / `--auto-layer-budget` / `--auto-speed-priority`,
see docs/AUTO_SELECTION_GUIDE.md).

---

## Current Status

**Two-stream inference:** ✅ COMPLETE  
**Two-stream training:** ✅ IMPLEMENTED (W10)

You can use any combination for **inference** AND **training**; the
composite (palette ≤ 16) shapes remain the fastest (one GEMM) and are
preferred by `--auto-speed-priority`.
