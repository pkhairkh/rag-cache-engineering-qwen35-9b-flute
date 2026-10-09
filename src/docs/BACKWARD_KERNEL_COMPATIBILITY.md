# Backward Kernel Compatibility — W9 resolution

## Status: IMPLEMENTED for idxN (1/2/3-bit), pending GPU-box verification

This document originally recorded the incompatibility (the 4-bit-only
backward kernels) as the blocking item for sub-4-bit training. W9
resolves it: the idxN backward-kernel family is implemented in
`flute_train_kernels`, CPU-verified, and runs on the GPU box after one
rebuild (the W9 campaign gates in `HANDOVER.md`).

The original analysis — the hardcoded nibble walk, the missing bitwidth
parameter, the 4-bit-only Python surface — is preserved verbatim below
as the historical record of WHAT was extended. The normative spec of
the extension is `docs/KERNEL_SPEC_DLDLUT.md` §8.

---

## What W9 implements (the point-by-point resolution)

| The gap (below) | The W9 resolution |
|---|---|
| `blob_segment_offset` hardcoded 4096/2048/64/16 | `blob_segment_offset_sub4<B>`: tile 1024*B, half 512*B, chunk 16*B, segment 4*B — at B=4 the legacy arithmetic verbatim |
| nibble extraction `b & 0x0F / b >> 4` | `decode_pair_sub4<B>`: the 2*B-bit pair field at bit 2*B*j, LSB-first (b=3 spans a word only when shift+6 > 32, in-range by construction) |
| no bitwidth in the kernel signature | `fused_backward_gemm_sub4_kernel<T, B, GS, kTwin>` / `lut_grad_scatter_sub4_kernel<T, B, GS, kTwin>` — compile-time B, runtime host dispatch (the forward campaign's switch style) |
| partials/workspace/reduce hardcoded 16 codes | `[LBM/GS][2^B]` partials, 2^B-strided workspace, `lut_grad_reduce_sub4_kernel<GS, B>` writing `[n_groups, 2^B]` |
| Python bindings expose 4-bit only | every entry takes `(bitwidth=4, indices_layout="")` with agreement gates; the wrapper re-validates; `idxn_available()` probes stale builds |

Determinism, geometry, pipeline, smem budget and register pressure are
the 4-bit kernels' verbatim (the §2 two-pass decision is
width-independent — G-B5b's torch.equal stays assertable at every
width). The 4-bit kernels and the 6-argument call form are UNCHANGED.

## The original analysis (preserved)

### 1. Kernel Source Analysis (at d6741a7)

**File:** `flute_train_kernels/src/kernel_lut_grad.cu`

**Line 289:** Hardcoded comment
```cpp
const uint8_t* __restrict__ blob,    // flat idx4 blob (N*K/2 bytes)
```

**Lines 276-277:** Hardcoded 4-bit nibble extraction
```cpp
partial[n_local / GS][b & 0x0F] += w0;  // Extract low nibble
partial[n_local / GS][b >> 4]   += w1;  // Extract high nibble
```

This is **4-bit specific** - the kernel assumes each byte contains
exactly 2 nibbles (4-bit indices).

### 2. No Bitwidth Parameter (at d6741a7)

```cpp
__global__ void lut_grad_scatter_kernel(
    const T* __restrict__ grad_y,        // [M, N]
    const T* __restrict__ x,             // [M, K]
    const uint8_t* __restrict__ blob,    // flat idx4 blob
    float* __restrict__ workspace,       // [n_blocks][LBM/GS][16]
    int M, int N, int K                  // No bitwidth!
)
```

---

## Impact (resolved state)

### What Works:
- Forward pass for all widths (1/2/3/4-bit) — W8, box-verified
- Inference with idxN format — W8, box-verified
- 4-bit training (backward pass) — unchanged, the regression contract
- 1/2/3-bit training (backward + LUT gradients) — W9: implemented,
  CPU-verified; the box gates (G-B1n/G-B2n/G-B5an/G-B5bn/G-B4n +
  the refusal gates) are the remaining verification step

### Migration (the 4-bit-only call form keeps working)

```python
# legacy (unchanged, still valid):
ftk.fused_backward_gemm(grad_y, blob4, lut16, N, K, GS)

# idxN (new):
ftk.fused_backward_gemm(grad_y, blob2, lut4, N, K, GS,
                        bitwidth=2, indices_layout="idx2")
```

Stale pre-idxN extensions refuse loudly (`idxn_available()` gates the
kernel-path attach; a mid-backward TypeError from an old .so is
translated into the rebuild message).
