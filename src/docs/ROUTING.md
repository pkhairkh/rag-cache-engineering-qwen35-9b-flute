# Decode routing

How a module picks its kernel at M = 1. The policy lives in
`scripts/palettized_modules.py` (`_gemv_decode`, `_gemv_fht_decode`,
the merge wrappers); the probe (`scripts/probe_decode_routing.py`)
prints the decision, and the reason every alternative was refused, for
every module — this doc is the map to that output.

## 1. The route chain

For each `PalettizedLinear` module, in order, the first gate that says
yes wins:

| # | route | kernel | gate |
|---|---|---|---|
| 1 | **merged-group launch** | `qgemm_gemv_multi` (QKV) / `qgemm_gemv_mlp` (MLP) | the module belongs to a merge-eligible group, same-geometry specs (per-segment heterogeneity is allowed — see below), `M == 1`, and `FLUTE_NO_MERGE` unset |
| 2 | **wide preference** | `qgemm_gemv_stream` (plain streamer) | ≥ 160 output tiles (lm_head: 1940) and `FLUTE_NO_WIDE_PREF` unset — the streamer carries wide modules at ~490 GB/s, SPLIT would be 1 anyway |
| 3 | **split-K + double-buffer** | `qgemm_gemv_splitk_stream` / `qgemm_gemv_splitk_fht_stream` | the workhorse: group_size 16..2048, residual rank ≤ 256, K-multiple constraints met; the FHT-fused variant when the boundary fold applies and `FLUTE_NO_FHT_FUSE` is unset |
| 4 | **dual-stream fused** | `qgemm_dual_stream` | two-stream weights at M ≤ 16 |
| 5 | **two-launch fallback** | two GEMV calls + add | anything the fused paths refuse (rank > 256, exotic geometry, a disabled switch) |

The merge gates are **per-group**: a QKV group merges when its
components' blobs are compatible at the segment level (the multi
kernel takes per-segment bitwidths/group sizes/signs at runtime, so
"same spec" is not required — heterogeneous components merge); an MLP
group merges when gate/up share the geometry the merged kernel assumes.
Every refusal records a reason string; the probe prints them all.

## 2. The environment switches

Every switch is read once per process, `1`/`true` enables, and exists
to attribute a measurement (A/B without a rebuild):

| switch | off (default) | on |
|---|---|---|
| `FLUTE_NO_MERGE=1` | merged QKV/MLP launches where eligible | every module routes to its single-module kernel — attributes the merge effect |
| `FLUTE_NO_FHT_FUSE=1` | FHT boundary fold inside the GEMV launch | the rotation runs as a separate step + plain GEMV — attributes the FHT fusion |
| `FLUTE_NO_SPLITK=1` | split-K GEMV (routes 3) | the plain streamer / two-launch path — attributes the split-K effect |
| `FLUTE_NO_WIDE_PREF=1` | wide modules on the plain streamer | wide modules forced onto split-K — the wide-preference A/B |
| `FLUTE_GS32_BK=64` | BK=32 tiles for gs=32 | deep K-tiles (`<64,128,64,32>`) where `K % 64 == 0` — the prefill A/B |
| `FLUTE_LAUNCH_DEBUG_SIMPLE=1` | production kernels | the differential twin — debugging only |
| `FLUTE_MERGE_DEBUG=1` | quiet | per-group merge decision prints |

Modeling/training side (same convention):

| switch | effect |
|---|---|
| `FLUTE_NO_FLA=1` (or `FLUTE_FLA=0`) | restores the reference linear-attention path (the `flute_sm86` kernel A/B) |
| `FLUTE_ATTN_IMPL` | the ONLY override of the full-attention implementation choice (default `flute_sm86`) |
| `FLUTE_FUSED_BWD=0` | the fused backward escape hatch (falls to reference backwards) |
| `FLUTE_WCACHE_GIB` | process-wide dequant W cache cap for the CPU fallback path (default 6 GiB) |

Build-time switches (`FLUTE_CUDA_ARCHES`, `FLUTE_CUTLASS_HOME`,
`FLUTE_DENSE_STREAMK`, `FLUTE_TRAIN_PTXAS_V`): [BUILD.md](BUILD.md).

## 3. Reading the probe table

```
module                       shape       gs   pair        route          grid      MB        us    GB/s
model.layers.0.mlp.gate_proj 12288x4096  512 (4, 1) splitk_fht_gemv     96x2    30.50    140.73   227.3
lm_head                      248320x4096 2048 (3, 3)         gemv         0     789.13   1681.70   492.0
```

- `pair` is the mixed-radix stream spec `(b1, b2)`;
- `grid` is `tiles × SPLIT` (the launch grid product is the CTA count —
  the policy targets ≥ 160 CTAs = 2 waves of the 80 SMs); `0` on the
  wide streamer row;
- `MB` is the code-stream bytes per forward (indices + LUT + residual);
- `us` / `GB/s` are the graph-replayed forward time and the effective
  bandwidth `MB / us` — the number to compare against the 600 GB/s wall;
- the composite section lists the merged launches the decode graph
  actually replays (QKV/MLP groups), with the merged bandwidth.

`--timing graph` (default on CUDA) captures the reps into one CUDA
graph and times replays — no host dispatch. `--timing eager` adds the
per-module host floor (~42 µs/module) if you want to see what the graph
is saving.

## 4. The routing census of the deployed model

249 PalettizedLinear modules, all on kernel routes (zero eager
fallback):

- 32 MLP groups → `mlp_merge` (one launch each, gate+up);
- 24 linear-attention QKV groups → `qkv_merge`; their `in_proj_z` and
  `out_proj` run single-module split-K;
- 8 full-attention layers: q/k/v/o as four single-module split-K
  launches each (the heterogeneous QKV merge covers the
  linear-attention groups; the full-attention layers' mixed
  shapes 8192/1024/1024/4096 are the current gap);
- `lm_head` → the wide streamer.

The current measured state of every route: [PERFORMANCE.md](PERFORMANCE.md).
