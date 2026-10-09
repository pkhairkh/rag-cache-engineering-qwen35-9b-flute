# Quantization format

The storage contract between the palettizer and the kernels: what sits in
an artifact directory, what every byte means, and what a dequantization
costs at runtime. The packers (`flute_extended/idxN.py`,
`flute_extended/idx4.py`) are the normative implementation; the tests
enforce every clause (CPU: `tests/test_dequant_reference.py` and friends;
GPU: `flute_extended/test_flute.py`, `test_qwen_weights.py`).

## 1. Artifact set of one palettized tensor

```
<name>.idx{b}          packed b-bit indices, N*K*b/8 bytes   (idxN blob)
<name>.lut_scalar      (ceil(N/group_size), 2^b) FP16 LUT
<name>.idx{b2}.2       optional SECOND stream (suffix ".2")
<name>.lut_scalar.2    its LUT (ceil(N/group_size), 2^b2) FP16
<name>.resA / .resB    optional rank-r residual factors (FP16)
```

`b ∈ {1, 2, 3, 4}`. The GEMM convention throughout is
`C[M, N] = A[M, K] @ W[N, K]^T` — weights are stored transposed relative
to the GEMM (W rows are output channels), with A FP16 row-major.

Dequantization semantics:

```
W[n, k]   = LUT[n // group_size, idx[n, k]]     # pure codebook lookup
W_eff     = W1 + W2                             # two-stream tensor
y         = x @ W_eff^T + (x @ resB^T) @ resA^T  # + residual branch
```

No scales, no zero points, no asymmetry: a dequantized weight is exactly
the FP16 LUT entry. Products are FP16 multiplies (`mma.sync` with FP32
accumulate), the output is rounded to FP16 once.

## 2. Logical packing (LSB-first along each row)

Indices pack LSB-first along the K dimension of each row:

- `b = 4`: two indices per byte — even k takes the low nibble, odd k the
  high nibble (`byte = idx[k>>1]`, `idx[k even] = byte & 0x0F`);
- `b = 1`: eight values per byte;
- `b = 2`: four values per byte;
- `b = 3`: eight values per 3 bytes (24-bit little-endian groups).

Row byte count is exactly `K*b/8` for every artifact (all model shapes
have `K % 64 == 0`).

## 3. Grouping

```
group(n) = n // group_size        # groups run along N only
```

One LUT row of `2^b` FP16 values is shared by `group_size` **consecutive
rows of W** across the whole K dimension; K never crosses a group
boundary. `ceil(N/group_size)` LUT rows exactly — the host wrappers
validate this.

Kernels accept `group_size ∈ {16, 32, 64, 128, 256, 512}`. A 128-row
N-tile may straddle a group boundary when `group_size < 128`: the
in-kernel group index is computed from the absolute row
(`(n0 + n_local) / group_size`), which degenerates to the simple form
when `BN % group_size == 0`. K must be a multiple of 32; K % group_size
is not required (BK is decoupled from group size at every width).

The deployed body sweep records `group_size` per tensor in
`metadata.json`. Known wrinkle: 132 deployed tensor components carry a
**stale `group_size` field** from the calibration sweep (the recorded
sweep value, not the deployed composition's value). The loader trusts
the LUT geometry, derives the true value, and warns once per run with
the count; the artifact set is unaffected. Fix belongs in the palettizer
writer, not the loader.

## 4. The blob layout (kernel-consumable permutation)

The on-disk blob is a byte **permutation** of the logical packed layout —
same size, same LUT, bit-identical GEMM output — arranged so that each
thread's Q loads land directly in the fragment registers the tensor
cores consume (the FLUTE paired-LUT trick, arXiv 2407.10960 §3.1-3.2):

- Eligibility: `N % 128 == 0` and `K % 64 == 0`. Every Qwen3.5-9B
  projection layer and both heads qualify. Ineligible shapes are refused
  loudly by the packers and by the kernel dispatch — no silent fallback.
- Tiles: `1024*b` bytes per (128-row tile t, 64-k tile g) at offset
  `((t * K/64) + g) * 1024*b`.
- Within a tile, thread (wx ∈ {0,1}, lane ∈ [0,32)) owns a contiguous
  `16*b`-byte chunk at `wx*512*b + lane*16*b`; the chunk holds that
  thread's 64 k-**pairs** in pair order `p = kt*16 + v*4 + d*2 + s2`,
  pair p at bit `2*b*p`, LSB-first, covering
  `n = t*128 + wx*64 + v*16 + d*8 + (lane>>2)` and
  `k = g*64 + kt*16 + 2*(lane&3) + 8*s2` (and k+1).
- At `b = 4` this reduces exactly to the classic idx4 layout (one byte
  per pair); `idxN.pack_idxn(idx, 4)` is byte-identical to
  `idx4.pack_idx4(idx)` (asserted by the tests).

The **k-pair** is the unit that matters: two consecutive values
(`idx[n,k]`, `idx[n,k+1]`) dequantize into one `mma.m16n8k16`
B-fragment u32 `{W[k], W[k+1]}` (low half = W[k]) via a byte-indexed
paired-LUT lookup — 256-entry table at b=4, two tables at b=2, four at
b=1, one 64-entry table at b=3. The mma phase, accumulation order and
epilogue are width-independent. Q loads use plain `ld.global.nc` (an
`evict_first` L2 hint is illegal on SM_80-89 and is not emitted).

## 5. Mixed-radix recipes

A recipe is a stream-width tuple `r1, r2, …`; the composite palette is
`prod(2^ri)`.

- Palette ≤ 16 → **one composite artifact** at the palette's storage
  width (`(2,2)` → one idx4; `(3,1)` → one idx4; `(1,1)` → one idx2);
- palette > 16 → **two streams**: the base at width r1, the composite of
  the rest (≤ 16 required) as `.idxN.2` (`(4,2)` → idx4 + idx2.2;
  `(4,2,2)` → idx4 + idx4.2; `(2,2,2)` → idx2 + idx4.2).

Storage bits per element = sum of stream widths (the LUT and residual
add ~`512*(1/N + 1/K)` bits/elt at model scale — under 0.2 everywhere):

| recipe | streams stored | bits/elt | vs FP16 | notes |
|---|---|---|---|---|
| `r1` | one idx4 | 4.0 | 27% | classic single-stream 4-bit |
| `hybrid422` | idx4 + idx4.2 | 8.0 | 51% | accuracy recipe: the two 2-bit refinements compose to a 16-entry palette, so there is no 2-bit storage to recover — it is stored as a 4-bit composite |
| `mixed:2` | one idx2 | 2.0 | 14% | true 2-bit |
| `mixed:3` | one idx3 | 3.0 | 20% | true 3-bit |
| `mixed:4,1` | idx4 + idx1.2 | 5.0 | 33% | |
| `mixed:4,2` | idx4 + idx2.2 | 6.0 | 39% | honest "4-bit + 2-bit refinement" |
| `mixed:2,2` | one idx4 | 4.0 | 27% | composite 16-palette at r1's storage |
| `auto` | per-tensor ladder | 2..8 | 14..51% | see [PALETTIZATION.md](PALETTIZATION.md) |

QKV fused tensors keep 4-bit bases (the split machinery's per-component
contract): their pool is `(4,)`, `(4,1)`, `(4,2)`, `(4,1,1)`, `(4,2,2)`.

## 6. Runtime cost

- One idxN stream = one `qgemm_per_group_lut`-family call at any width
  (the same register-direct tensor-core path; narrower streams cut HBM
  Q-traffic proportionally).
- A two-stream tensor = two calls + one FP16 add.
- The residual = one rank-r GEMM chain (`(x @ resB^T) @ resA^T`); the
  split-K decode GEMV fuses it up to **r ≤ 256** in the epilogue.
- Paired-LUT shared-memory cost: b=1 uses four tables (20 KB at gs=32) —
  within the SM_86 budget at every width.

## 7. Integrity

Every stream carries `sha256_idx` / `sha256_lut` in `metadata.json`; the
loaders verify on read and refuse mismatches. Resume rules: the RECORDED
streams in metadata decide (per-component `index_file`/`lut_file` pairs
for QKV); an artifact set without metadata is refused (fresh output
directory required — never a silent overwrite). A sub-4-bit set that
disagrees with the recorded geometry aborts before the model load.

## 8. Reference implementation

The pure-torch oracle the kernels are differentially tested against:

```python
# unpack (b=4 shown; the general b case unpacks LSB-first)
lo = (indices & 0x0F).long()
hi = ((indices >> 4) & 0x0F).long()
idx_full = torch.stack([lo, hi], -1).reshape(N, K)
groups = torch.arange(N) // group_size
W = torch.gather(lut[groups].float(), 1, idx_full)      # W[n, k]
C_ref = (A.float() @ W.T).half()                        # FP32 accumulate
```
