# DEQUANT_SPEC.md

Specification of the Qwen3.5-9B palettized weight format and the exact
dequantization semantics implemented by every backend. This document is
the contract between the model format and the kernels; the test
suites (test_flute.py, test_qwen_weights.py) enforce every clause.

Sections 1-6 define the LOGICAL format (nibble order, grouping, numerical
semantics). Section 7 defines the idx4 artifact layout — the canonical
on-disk byte permutation of the logical packed indices that every
`<name>.idx4` file carries and the production kernel consumes.

---

## 1. Tensors

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| A (activations X) | [M, K] | FP16 | row-major, contiguous |
| indices (Q) | [N, ceil(K/2)] | uint8 | 2 x 4-bit indices per byte |
| lut (LUT) | [ceil(N/group_size), 16] | FP16 | one 16-entry row per group |
| C (output Y) | [M, N] | FP16 | row-major; Y = X @ W^T |

The GEMM computed is  **C[m, n] = sum_k A[m, k] * W[n, k]**  with
**W [N, K]** the dequantized weight matrix (weights are stored transposed
relative to the GEMM: W rows = output channels).

## 2. Packed 4-bit indices — LSB_FIRST nibble order (FIXED format)

Two consecutive K-dimension indices share one byte:

    byte = indices[n, k >> 1]

    even k (k % 2 == 0):  index =  byte        & 0x0F   // LOW nibble  = FIRST
    odd  k (k % 2 == 1):  index = (byte >> 4)  & 0x0F   // HIGH nibble = SECOND

Bit pattern of one byte (little-endian nibble order):

    bits [7:4] = index of the ODD k (second)
    bits [3:0] = index of the EVEN k (first)

Example: byte 0xA7 -> k=2j gets LUT value #7 (0x7), k=2j+1 gets LUT value
#10 (0xA).

## 3. Grouping and the LUT gather

    group_idx(n) = n // group_size            // groups run along N (rows of W)
    W[n, k]      = LUT[group_idx(n), index(n, k)]

* One LUT row (16 FP16 values) is shared by every `group_size` CONSECUTIVE
  ROWS of W (N dimension), across the ENTIRE K dimension.
* group_size = 32 for MLP layers (gate_proj/up_proj), 64 for attention
  layers (q/k/v/o_proj, down_proj) of Qwen3.5-9B.
* W12 (full-GS-range kernels): every kernel of this tree —
  cutlass_streaming (4-bit + sub-4, legacy + fragment-direct) and
  debug_simple — accepts group_size in {16, 32, 64, 128, 256, 512}.
  GS > BN=128 is legal: a 128-row N-tile may straddle one group
  boundary; the in-kernel group index is computed from the ABSOLUTE
  row (n0 + n_local)/GS, which degenerates to the pre-W12
  n_local/GS arithmetic verbatim whenever BN % GS == 0. K must be a
  multiple of 32 (the BK constraint); K % group_size is no longer
  required (BK is decoupled from GS at every width).
* LUT must have exactly ceil(N / group_size) rows (the host wrappers
  validate this).
* NOTE: the grouping is along N only; K never crosses a group boundary.
  Every K-tile of a row n uses the single LUT row n // group_size. The
  streaming kernel exploits this by staging the LUT once per block and
  decoupling the K-tile depth (BK) from group_size entirely.

Example shapes from the model metadata:

    model.layers.0.mlp.gate_proj.weight : dense [12288, 4096], gs=32, groups=384 (= 12288/32)
    model.layers.0.mlp.down_proj.weight : dense [4096, 12288], gs=64, groups=64  (= 4096/64)
    model.layers.31.self_attn.q_proj    : dense [8192, 4096],  gs=64, groups=128 (= 8192/64)

("groups" in the metadata counts N/group_size LUT rows, matching
ceil(N/group_size) for these aligned shapes.)

## 4. Numerical semantics

* Dequantized values are EXACTLY the FP16 LUT entries (no scaling, no
  zero-point, no asymmetry — pure codebook lookup).
* Products A[m,k] * W[n,k] are computed in FP16 (mma.sync
  f32.f16.f16.f32: FP16 multiplies), accumulated in FP32.
* Final C is rounded once to FP16.
* Boundary rules: contributions with k >= K or n >= N are exactly zero
  (kernels zero-fill the shared-memory tiles), so padding never biases
  results.
* The naive backend emulates the same semantics with __half2float
  multiplies and an FP32 accumulator — it is the golden reference.

## 5. Where each clause lives in the code

| Clause | Implementation |
|---|---|
| LSB-first nibble split | dequant.cuh decode_nibble/decode_pair; kernels' `idx = (k&1) ? hi : lo`; test_flute.py's nibble-order gate (bit-exact) |
| group index n//group_size | dequant.cuh group_of(); streaming kernel `g = (n / GS) - grp_first` |
| LUT row cache | streaming kernel Step 0 (sLUT32 in shared memory, synced); optimized kernel sLUT |
| BK decoupled from group_size | TileConfig<BM,BN,BK,GS>; dispatch on group_size + FLUTE_GS32_BK |
| FP32 accumulate | streaming mma f32 accumulators; naive/optimized float sum |
| zero boundary fill | streaming a_copy_async else-branch and q_zero; debug_simple zero rows |
| idx4 artifact layout | flute_extended/idx4.py (pack/unpack/self_test); producer scripts/palettize_qwen3_5_9b.py |

## 6. Reference implementation (Python, used by the tests)

```python
lo = (indices & 0x0F).long()                    # even-k indices
hi = ((indices >> 4) & 0x0F).long()             # odd-k indices
idx_full = torch.stack([lo, hi], -1).reshape(N, K)
groups = torch.arange(N) // group_size
W = torch.gather(lut[groups].float(), 1, idx_full)   # W[n, k]
C_ref = (A.float() @ W.T).half()                # fp32 accumulate
```

## 7. The idx4 artifact layout (canonical on-disk format)

Sections 1-6 define the CANONICAL logical format. Every on-disk
`<name>.idx4` artifact stores the packed `indices` matrix as a byte
PERMUTATION of that logical layout — same size, same LUT, same numerical
semantics, bit-identical GEMM output — called **idx4** (producer and
normative implementation: `flute_extended/idx4.py`; consumer:
`qgemm_per_group_lut(..., indices_layout="idx4")`, kernel dispatch flag
q_layout=1).

* Eligibility: `N % 128 == 0` and `K % 64 == 0` (every Qwen3.5-9B layer
  qualifies). `flute_extended.idx4` refuses other shapes and the kernel
  rejects an idx4 blob for an ineligible shape with a hard error — there
  is no silent fallback.
* Layout: the blob is cut into 4096-byte tiles, one per (128-row tile t,
  64-k-value tile g), at byte offset `((t * K/64) + g) * 4096`. Within a
  tile, thread (wx in {0,1}, lane in [0,32)) owns a contiguous 64-byte
  chunk at `wx*2048 + lane*64`; chunk byte `kt*16 + v*4 + d*2 + s2` holds
  the ORIGINAL byte `indices[n, kp]` with
  `n = t*128 + wx*64 + v*16 + d*8 + (lane>>2)` and
  `kp = g*32 + kt*8 + (lane&3) + 4*s2`.
* Why: one packed byte is exactly one `mma.m16n8k16` B-fragment u32
  (`{W[k], W[k+1]}` of one output row, LSB-first matching section 2), so
  after the permutation each thread's Q load lands directly in the
  registers the tensor cores consume — dequantization needs a single
  256-entry paired-LUT lookup per byte and no shared-memory round trip
  (FLUTE, arXiv 2407.10960, sections 3.1-3.2).
* Q loads use plain `ld.global.nc` (see include/flute/mma.cuh): an
  `evict_first` L2 hint is illegal on Ampere (SM_80-89), so it is not
  emitted.
* Verification path: `flute_extended.idx4.self_test()` proves the
  pack/unpack round-trip on the CPU; `test_flute.py` re-proves
  bit-exactness on the GPU
  (`idx4 == kernel-internal legacy == debug_simple`, `torch.equal`), and
  `test_qwen_weights.py` re-proves it against the section-6 reference on
  real weights. The kernel's internal q_layout=0 byte order exists solely
  as that differential-test reference.

## 8. The idxN family (sub-4-bit storage: b in {1, 2, 3})

The idxN extension generalizes sections 1-7 to bit widths 1/2/3 at every
layer of the stack — the SAME approach, the SAME style:

* Logical format (generalizes section 2): b-bit indices packed
  LSB-first along each row; row byte count = K*b/8 (exact for every
  artifact: K % 64 == 0). b=1: 8 values/byte; b=2: 4 values/byte; b=3:
  8 values per 3 bytes (24-bit little-endian groups). b=4 is section 2
  verbatim.
* LUT (generalizes section 1/3): [ceil(N/group_size), 2^b] FP16 — 2, 4
  or 8 entries per group. The grouping, the pure-lookup numerical
  semantics (section 4) and the zero-boundary rules are unchanged.
* Blob layout (generalizes section 7): tiles of 1024*b bytes per
  (128-row, 64-k) tile; thread (wx, lane) owns a contiguous 16*b-byte
  chunk at `wx*512*b + lane*16*b`. The chunk holds the thread's 64
  k-PAIRS (128 elements) in pair order p = kt*16 + v*4 + d*2 + s2 (the
  byte-position formula of section 7 generalized from bytes to pairs):
  pair p sits at bit 2*b*p, LSB-first, covering
  n = t*128 + wx*64 + v*16 + d*8 + (lane>>2) and
  k = g*64 + kt*16 + 2*(lane&3) + 8*s2 (and k+1). At b=4 this reduces
  EXACTLY to section 7 (one byte per pair) — `idxN.pack_idxn(idx, 4)`
  is byte-identical to `idx4.pack_idx4(idx)` (asserted by the tests).
* The k-PAIR is the unit that matters: two consecutive values
  (idx[n,k], idx[n,k+1]) dequantize into ONE mma.m16n8k16 B-fragment
  u32 {W[k], W[k+1]} (low half = W[k]) — the same fragment register the
  4-bit kernel feeds, so the mma phase, the accumulation order and the
  epilogue are width-independent.
* Kernels: `flute_kernel_streaming_sub4<Cfg, B>` (legacy q_layout=0
  path) and `flute_kernel_streaming_fd_sub4<Cfg, B>` (idxN blob path,
  the production path) for B in {1,2,3}; the 4-bit kernels of sections
  1-7 are UNTOUCHED (their binaries are the deployed, gated artifacts).
  `flute_kernel_debug_simple_sub4<Cfg, B>` extends the differential
  twin, so the section-7 chain — `idxN == kernel-internal legacy ==
  debug_simple`, `torch.equal` — holds at every width
  (test_flute.py --bits all).
* Fragment-direct dequant at width b (why the fd path stays one LDS per
  fragment u32): the 256-entry byte-indexed paired-LUT trick of section
  7 generalizes as b=2 -> two tables (pairs at nibbles 0/1),
  b=1 -> four tables (pairs at bit pairs (2t, 2t+1)), and b=3 -> one
  64-entry table indexed by the 6-bit pair field itself.
* Producer/verifier: `flute_extended/idxN.py` (pack_idxn / unpack_idxn /
  self_test; the ONLY producer of the family — the palettizer, the test
  suite and the loaders all defer to it). `flute_extended/idx4.py`
  stays untouched as the canonical b=4 producer.
* Python surface: `qgemm_per_group_lut(A, indices, lut, bitwidth=b,
  group_size=gs, indices_layout=f"idx{b}")` (layout and bitwidth must
  agree — the wrapper refuses mismatches loudly); C++ dispatch flag
  q_layout=1 with `lut.size(1) == 2^b` and `indices.numel() ==
  N*K*b/8`.
* Eligibility, group sizes, dispatch (BK/GS), environment toggles and
  the m-major rasterization are identical to section 7 at every width.

The artifact naming convention: `<name>.idx1` / `.idx2` / `.idx3` /
`.idx4` for the blob, `<name>.lut_scalar` with 2^b entries per group;
stream-tagged variants `<name>.idx{b}.<tag>` follow the W4 pair rule
(e.g. the mixed-recipe rest-composite lands as `.idx2.2`).
