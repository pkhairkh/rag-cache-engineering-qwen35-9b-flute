# docs/QUANTIZATION_FORMAT.md — the storage-format contract of record

**Audience.** The coding agent (CPU-only context), the GPU-box agent
(A10G context) and the operator. This document is the honest accounting
of what every quantization recipe in this repository STORES on disk, in
bits per element, and what each format's dequantization costs at
runtime. It supersedes the storage claims of PROPOSAL.md §1 wherever
the two disagree (the proposal's §1 mixed "optimization objective" with
"storage format"; this file separates them).

Provenance: the bits/elt numbers below are the formula
`sum(stream bit widths) + 512*(1/N + 1/K)` (the residual/LUT overhead
law of reports/lut2bit_integration.md §1.4, PROPOSAL.md §1.2), with the
stream bit widths now taken from the ACTUAL stored artifacts.

---

## 1. The artifact families

Every palettized tensor is a set of one or two **idxN streams**, each
consumed by one `qgemm_per_group_lut` call (b = the stream's width in
{1,2,3,4}):

```
<name>.idx{b}          # packed b-bit indices, N*K*b/8 bytes (idxN blob)
<name>.lut_scalar      # (n_groups, 2^b) fp16 LUT
<name>.idx{b2}.2       # optional SECOND stream (tagged ".2")
<name>.lut_scalar.2    # its LUT (n_groups, 2^b2)
<name>.resA / .resB    # optional rank-r residual factors (fp16)
```

Dequantization semantics of one stream (DEQUANT_SPEC §3/§8):
`W[n, k] = LUT[n // group_size, idx[n, k]]` — a pure codebook lookup.
A two-stream tensor's effective weight is `W1 + W2` (two GEMMs plus one
ordered add — the module layer's deployment shape); the residual adds
the low-rank branch `(x resB^T) resA^T`.

## 2. The recipes and their TRUE storage

| recipe (CLI) | streams stored | bits/elt (indices) | + r32 residual | total vs FP16 | notes |
|---|---|---|---|---|---|
| `r1` | one idx4 (16-entry LUT) | 4.0 | +0.19 (attn_q) | ~27% | the legacy single-stream 4-bit path, byte-identical |
| `hybrid422` | idx4 base + idx4.2 pair-composite | 8.0 | +0.19 | ~51% | the E8 winner `hyb_4_2_2_joint_r32`; the "2-bit refinement" is STORED as a 4-bit composite — this is the accuracy recipe, NOT a compression recipe (see §4) |
| `mixed:2` | one idx2 | 2.0 | +0.19 | ~14% | true 2-bit storage |
| `mixed:1,1` | one idx2 (composite palette 4) | 2.0 | +0.19 | ~14% | refined 2-bit (joint selection over 2×1-bit stages) |
| `mixed:3` | one idx3 | 3.0 | +0.19 | ~20% | true 3-bit storage |
| `mixed:2,1` | one idx3 (composite palette 8) | 3.0 | +0.19 | ~20% | refined 3-bit |
| `mixed:2,2` | one idx4 (composite palette 16) | 4.0 | +0.19 | ~27% | the E8 `lut2_S2`-class recipe at r1's storage |
| `mixed:3,1` | one idx4 (composite palette 16) | 4.0 | +0.19 | ~27% | another 4-bit-composite variant |
| `mixed:4,2` | idx4 base + idx2.2 refinement | 6.0 | +0.19 | ~39% | the honest "4-bit + 2-bit refinement" split |
| `mixed:4,2,2` | idx4 base + idx4.2 composite | 8.0 | +0.19 | ~51% | the lloyd-stage-1 variant of hybrid422's layout |
| `mixed:4,1` | idx4 base + idx1.2 refinement | 5.0 | +0.19 | ~33% | 5-bit total |
| `auto` | per-weight, from the ladder | 2..8 | +0.19 | 14..51% | selective per-weight refinement (see §5) |

The LUT overhead is `2^b * 2 B / group_size / (b/8)` bits per element
(gs=32, b=2: 8/32 bytes = 2 bits/elt? no — the LUT rows scale with
N/group_size, and the per-element LUT cost at model scale is the
`512*(1/N + 1/K)` law's small tail; measured at the toy and model
geometries it stays well under 0.2 bits/elt for every width). The
residual rank-32 term is the `512*(1/N+1/K)` law verbatim.

## 3. The optimization machinery vs the storage format

The quantization side was ALWAYS width-parameterized
(`_lloyd_sorted(bitwidth)`, `select_joint_mixed_radix` with arbitrary
per-stream radices, `_mixed_radix_decode`). What was 4-bit-locked was
the STORAGE: every stream landed in an idx4 artifact, so a
"4-bit base + two 2-bit refinements" pipeline stored
`4 + composite(2+2 → 4) = 8 bits/elt`. The idxN family (kernels +
packers + writers + loaders, DEQUANT_SPEC §8) removes that lock: the
stored width now equals the palette the optimization actually uses
(2/4/8/16 entries → idx1/idx2/idx3/idx4).

Mixed-radix composition law: a spec `r1, r2, ...` (stream bit widths)
has composite palette `prod(2^ri)`.
- palette ≤ 16 → ONE composite artifact at
  `palette_storage_bits(palette)` width (e.g. `2,2` → one idx4);
- palette > 16 → the two-stream layout: the base stream at width r1
  plus the composite-of-the-rest (palette ≤ 16 required) as `.idxN.2`
  (e.g. `4,2,2` → idx4 + idx4.2; `4,2` → idx4 + idx2.2; `2,2,2` →
  idx2 + idx4.2). This is exactly the hybrid422 write rule, generalized
  to sub-4 stream widths.

## 4. Why hybrid422 stores 8 bits/elt (the corrected understanding)

The two 2-bit refinement streams of hybrid422 compose into a
Minkowski-sum palette of 16 entries — a PERFECT fit for one idx4
stream. Storing the two refinements separately (idx2 + idx2) would cost
the same 4 bits; there is no 2-bit compression to recover. The real
sub-4-bit savings come from SMALLER TOTAL radices: a 2-bit base
(`mixed:2`), a 3-bit single stream (`mixed:3`), or short base+refine
splits (`mixed:4,1` = 5 bits, `mixed:4,2` = 6 bits). The accuracy cost
at a given total width is the campaign's measured trade (the E8 toy
data: `rot_lut2_S3_joint_r32` at 6 bits misses the 1e-4 nmse gate;
`hyb_4_2_2_joint_r32` at 8 bits passes) — with the residual and the
joint selection as the compensating levers.

## 5. `--recipe auto` (selective per-weight refinement)

Per tensor (per QKV component), the resolver walks the ladder in
ascending total storage bits — single-tensor:
`(2,) (1,1) (3,) (2,1) (4,) (2,2) (3,1)`; QKV (4-bit-base contract):
`(4,) (4,1) (4,2) (4,1,1) (4,2,2)` — and picks the FIRST candidate
whose calibration cosine meets `--auto-cos` (default 0.9995, the
preview band's floor); none passing → the best-cos candidate. The
resolved spec is recorded in metadata.json (`"recipe": "auto:<spec>"`,
`"mixed_spec": [...]`) and printed per tensor. `--bits-plan plan.json`
overrides individual weights: `{"model.layers.0.mlp.down_proj.weight":
"mixed:2,2", "q_proj.weight": "mixed:4,2"}` (exact name, then longest
suffix). Sub-4-bit base stages run the Lloyd assignment engine
(`--assign lloyd`); gptq/gptvq are 16-palette designs and are refused
loudly for sub-4 bases.

## 6. Runtime cost

One idxN stream costs one `qgemm_per_group_lut` call — the SAME
register-direct tensor-core path at every width (one paired-LUT LDS per
mma B-fragment u32; DEQUANT_SPEC §8). A two-stream tensor costs two
calls plus one fp16 add; the residual adds one rank-r GEMM chain.
Narrower streams cut the Q-load bytes proportionally
(2 bits → half the HBM traffic of 4 bits at equal N*K) and shrink the
artifact set the same way; the paired-LUT tables grow with the count
(b=1: 4 tables = 20 KB smem at gs=32 — still within the SM_86 budget).

## 7. Resume and integrity

Legacy rule (r1/hybrid422): base-pair existence, plus the W4-T02/T03
pair rule and sha256 verification against metadata.json. idxN runs:
the RECORDED streams decide (metadata's index_file/lut_file pairs, per
component for QKV); a sub-4-bit artifact set without metadata is
refused loudly (fresh output directory required — never a silent
overwrite). Every stream carries sha256_idx/sha256_lut; the loaders
verify on read.
