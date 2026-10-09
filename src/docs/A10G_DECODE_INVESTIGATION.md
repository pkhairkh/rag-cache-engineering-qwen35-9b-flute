# A10G_DECODE_INVESTIGATION.md — the W29 deep audit: hardware, kernels, modeling, and where the remaining 5.8% gap (and the next 2x) actually live

Provenance: generated 2026-10-09 from the repo at `1630ffb` (W28), after the
W28 box report (22.287 tok/s = 0.942x dense, PPL +3.52% unchanged, FHT
fusion a wash). Every claim is anchored to a file:line or an arithmetic
identity; nothing is asserted from memory alone. Companion artifact:
`scripts/probe_decode_routing.py` (the box probe that turns §4/§5 into
measured numbers in one run).

Scope answers the operator's three questions, in order:

1. **"Make thorough investigation for our exact hardware — read all
   kernel files and our modeling logic. How can we even get faster?"**
   → §2 (hardware), §3–§6 (kernel + modeling audit), §8 (the roadmap).
2. **"Do thorough investigation on A10G too"** → §2, including a
   repo-internal doc conflict this audit settles (72 vs 80 SMs).
3. **"Why don't we use flash attention?"** → §7 — the short version:
   at M=1 decode it is worth <1 ms/token of the 44.9 ms, PyTorch's flash
   backend refuses this geometry on sm_86 anyway, and the repo's own
   Triton kernel (which exists and is correct) deliberately short-circuits
   S=1 to reference math. Attention is not where the missing 2x lives.

---

## 0. Executive summary

| # | Finding | Evidence | Consequence |
|---|---------|----------|-------------|
| 1 | **The decode-GEMV path runs at ~90–110 GB/s effective — 5–6.5x below the A10G's 600 GB/s streaming wall.** The whole remaining gap is inside the module GEMMs, not the composition, not launches, not attention. | §3: W26/W27/W28 arithmetic cross-check; §4 kernel audit | W29 kernel work has ~4.5x recoverable headroom — enough to reach ~2x dense by itself |
| 2 | **The GEMV grid is `N/128` CTAs with no K-split.** k/v_proj: 8 CTAs on 80 SMs (10%); Q/K: 16 (20%); V/Z/out/down: 32 (40%). Only gate/up (96) and lm_head (1940) fill the machine. | kernel_cutlass_streaming.cu:2331 (`grid(N / 128)`); §4.1 table | 2–3x of the gap is raw SM idling on narrow modules |
| 3 | **Per-CTA streaming is latency-bound: no prefetch/double-buffer in the K-loop, 8–10 warps/SM, and every warp serially walks `K/256` g-iterations.** The W28 FHT prologue made it worse for down_proj (smem 26.8→57.3 KB ⇒ 1 CTA/SM). | §4.2–4.4; the W28 kernel's own smem table (cu:3012–3017) | The other ~2.5x of the gap; this is why W28 gained nothing |
| 4 | **The W28 no-op is now fully explained.** The 281 standalone FHT launches cost ~0.4–0.6 ms/token (not the 2.47 ms hypothesized), and the fused prologue ADDS ~1–4 µs of serial butterfly per CTA — a net wash, inside measurement noise. | §3.3 | FHT fusion was the right idea at the wrong margin; keep it, it becomes strictly positive once the GEMV streams at ≥300 GB/s |
| 5 | **Two routing holes outside the GEMV table:** GS=16/32 modules (6 instances) take the full pre-W26 chain (FHT + two mma launches + residual GEMMs + casts), and **lm_head is the (4,4) pair + rank-32 residual — in NEITHER the GEMV nor the DUAL pair tables** (both cap at the same 7 pairs and R≤16), so the single biggest module (1.03 GB/step, ~25% of all traffic) runs the slowest path. | `__init__.py:193–204` (pair tables), `DUAL_STREAM_GROUP_SIZES={64,128,256,512}`; kernel GS gate cu:2819–2825; R≤16 at cu:2889; MODEL_GEOMETRY.md §3 | ~3–5 ms/token recoverable; verify on box with the probe (§9) |
| 6 | **The linear-attention decode runs pure-torch fallbacks** (`torch_recurrent_gated_delta_rule`, torch `causal_conv1d_update`) — ~35–45 tiny kernels per layer, ~2–4 ms/token — unless the transformers hub-kernel wiring picks up `fla` (installed in the lockfile but not obviously wired). | modeling.py:566–578, :189–206; requirements.lock.txt:20–21 | Second-tier fix worth 1.5–3 ms/token; also shortens the graph |
| 7 | **A10G = 80 SMs, 6 MB L2, 600 GB/s** (AWS: 320 tensor cores ÷ 4; TechPowerUp: 6 MB L2). `docs/GPU_SPEC.md`'s "72 SMs / 9,216 cores" is the A10 *non-G* die and should be corrected; `flute_extended/docs/PERFORMANCE.md` §1 (80 SMs) is right. | §2 | The occupancy arithmetic in §4 uses 80 |

The one-line physics: **dense decodes at 1.58x its floor; quant decodes at
7.3x its floor; the entire difference is GEMV kernel efficiency.** Fix
findings 2+3+5 and the quant arm lands at ~10.5–13 ms/token
(77–95 tok/s ≈ **1.8–2.25x dense**) with the same artifacts, the same PPL,
and no change to the model.

---

## 1. What was audited

Every file on the decode path, in full or in the relevant sections:

| File | Role | Sections read |
|------|------|---------------|
| `flute_extended/src/kernel_cutlass_streaming.cu` (3,804 lines) | all production qgemm kernels | W27 GEMV 2039–2341; dispatch 2740–2960; W28 FHT-GEMV 2965–3300; dual mma structure 1459+; helpers 199–880 |
| `flute_extended/flute_extended/__init__.py` | Python gates, pair/GS tables | 190–583 |
| `scripts/palettized_modules.py` (2,760 lines) | PalettizedLinear composition | forward 1134–1307; `_gemv_decode` 887–953; `_gemv_fht_decode` 955–1064; `_dual_stream_decode` 1066–1132 |
| `scripts/modeling.py` (1,168 lines) | Qwen3.5-9B hybrid architecture | GatedDeltaNet 444–603; recurrent/chunk torch fallbacks 240–437; conv fallbacks 189–229; attention dispatch 664–756 |
| `scripts/eval_greedy_match.py` | the decode harness | graph runner 305–350; `_sdpa_math_ctx` 370–395; capture/verify 260–420 |
| `scripts/attn_sm86.py` | the repo's Triton flash attention | header/contract 1–80 |
| `flute_extended/fht.py`, `include/flute/fht.cuh`, `src/kernel_fht.cu` | the rotation | semantics + prologue transcription |
| `docs/GPU_SPEC.md`, `flute_extended/docs/PERFORMANCE.md`, `docs/MODEL_GEOMETRY.md`, `docs/QUANTIZATION_FORMAT.md`, `flute_extended/docs/DEQUANT_SPEC.md`, `INSPECTION.md` | the contracts of record | all |
| `scripts/eval_common.py`, `scripts/generate.py`, `flute_extended/benchmark_kernel.py` | loaders / bench | all / decode-sweep |

---

## 2. The A10G, settled

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA A10G (AWS g5), GA102, sm_86 | AWS g5 page |
| **SMs** | **80** | AWS News blog: "80 [RT] cores and 320 third-generation Tensor Cores" per A10G; 3rd-gen tensor = 4/SM on GA10x ⇒ 320/4 = 80. Cross-checked: PERFORMANCE.md §1 says 80 |
| CUDA cores | 10,240 | 80 × 128 |
| L2 | 6 MB | TechPowerUp A10G database entry |
| DRAM | 24 GB GDDR6, 384-bit, **600 GB/s** | A10 datasheet mirrors; consistent everywhere |
| FP16/BF16 tensor, FP32-acc | 69.8 TFLOPS @1710 MHz (sustained ~62.5) | PERFORMANCE.md §1 (the FP16-acc 125/139.6 figures are the 2x-rate, not this kernel's regime) |
| Visible memory | ≈22.35 GiB | AWS g5 instance page |
| **In-repo conflict** | GPU_SPEC.md §1 says "72 SMs / 9,216 CUDA cores" — that is the **A10 non-G** count; PERFORMANCE.md says 80 | ACTION: correct GPU_SPEC.md (one-line fix, next commit that touches it) |

What 80 SMs / 600 GB / 6 MB L2 means for this workload:

* Per-SM bandwidth share = 600/80 = **7.5 GB/s**. A kernel that leaves an
  SM without a resident CTA simply donates its 7.5 GB/s to nobody.
* DRAM latency on GDDR6 through L2 ≈ 600–900 ns. To saturate 7.5 GB/s an
  SM needs ~(7.5 GB/s × 700 ns) ≈ **5.3 KB of loads in flight, sustained**.
  With 16 B/thread loads that is ~330 in-flight lanes — i.e. **≥10 warps
  issuing back-to-back, or explicit prefetch**. This single number is the
  hinge of §4.
* L2 = 6 MB: every LUT (2–24 KB per module) and every resB (≤131 KB) is
  L2-resident after first touch; the code streams (3.4 GB/token) can never
  be. So LUT/residual traffic is noise; **the code stream is the only
  memory story that matters at M=1.**

Box verification (one line, settles any residual doubt):

```
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, p.multi_processor_count)"
```

---

## 3. The decode time budget — where 44.9 ms/token actually goes

### 3.1 The measured anchors

| Run | tok/s | ms/token | module-GEMM path |
|---|---|---|---|
| W26 box | 13.514 | 74.0 | dual mma (BM=32) for everything, fused residual |
| W27 box | 22.361 | 44.7 | GEMV for 97% of instances |
| W28 box | 22.287 | 44.9 | W28 FHT-fused GEMV (this commit) |
| dense | 23.668 | 42.3 | cuBLAS skinny GEMM/GEMV |

### 3.2 The floor arithmetic (the artifact census, INSPECTION.md §3.2/§5)

Per-token weight traffic at M=1:

| Stream | Bytes/token | @600 GB/s |
|---|---|---|
| layer code streams (2.292 GiB packed) | 2.46 GB | 4.1 ms |
| head code streams (~2 idx4 blobs, MODEL_GEOMETRY §3) | 1.03 GB | 1.7 ms |
| LUTs + resA/resB (L2-resident, counted once) | ~0.15 GB | 0.25 ms |
| KV (8 full-attn layers, 4×256 GQA, ~2k ctx) + GDN states | ~75 MB | 0.13 ms |
| **quant floor** | **~3.7 GB** | **~6.2 ms ⇒ 161 tok/s ceiling** |
| dense floor (13.8 GB body + 2.03 GB lm_head fp16) | ~16.0 GB | ~26.7 ms ⇒ 37.5 tok/s |

Dense measured 42.3 ms = 1.58× its floor ⇒ the dense arm streams at
~63% overall (~420 GB/s through cuBLAS — a healthy eager baseline).
**Quant measured 44.9 ms = 7.3× its floor.**

### 3.3 The W26 cross-check that pins the blame

The W26 box datum (74.0 ms/token) was taken with the *same* machinery,
*same* graphs, *same* everything — only the module GEMMs differed (the
dual-mma kernel, measured "~12x above the code-stream floor" at M=1,
kernel_cutlass_streaming.cu:2044–2046). Check:
3.7 GB × 12 ÷ 600 GB/s = **74 ms — the W26 number is almost exactly the
mma path alone.** Therefore the non-GEMM machinery (attention, linear
attention, norms, RoPE, logits, host loop) is **~3–4 ms/token**, both arms.

W27 swapped 97% of module instances onto the GEMV: 74.0 → 44.7 ms. So the
GEMV path (3.6 GB over 242 modules + whatever the fallbacks cost) now runs
~41 ms ⇒ **~90–105 GB/s effective = 15–17% of the DRAM wall.**

W28 then fused the FHT into the GEMV and moved nothing (−0.3%). That is
consistent with two facts, both now derivable rather than hypothesized:

* the 281 standalone FHT launches were worth ~0.4–0.6 ms/token (butterfly
  kernels of ~1–2 µs each — the "562 vs 282 launches" framing over-counted
  because a CUDA-graph node costs ~1 µs, not the ~8 µs eager-launch figure
  the 2.47 ms estimate assumed);
* the fused prologue **adds** work to every CTA's serial path (the
  butterfly runs on the GEMV's own 256 threads, cu:2995–2998) and pushed
  down_proj's smem from 26.8 KB to 57.3 KB — 3 CTAs/SM → **1 CTA/SM**
  (cu:3012–3017, the kernel's own table). Net: a wash, exactly as measured.

**Conclusion: 100% of the remaining gap to dense — and all of the headroom
beyond it — is the GEMV kernel's achieved bandwidth.** Not launches (graphs
already amortized them), not the composition (W26/W27 fused everything),
not attention (§7), not the FHT (§3.3).

---

## 4. Why the GEMV runs at ~15% of the wall — four structural causes

The kernel: `flute_kernel_gemv_dual` (cu:2142–2306). One CTA = one 128-row
N-tile, 256 threads = 8 warps; warp (wx, j4) owns rows [wx·64, +64) and
k-tiles g ≡ j4 (mod 4); each lane owns 8 rows via `acc[8]`; codes load as
`uint4` per stream per g-iteration; the palette lives in one register per
lane, served by `shfl.idx`; per-CTA smem = 2K (staged x) + 2112 B.

### 4.1 Cause 1 — the grid: `N/128` CTAs, no K-split (the big one)

`launch_gemv_dual`, cu:2331: `dim3 grid(N / 128);` — one CTA per 128-row
tile, and **nothing splits K**. Against 80 SMs (§2):

| module (per layer) | shape (N×K) | CTAs | SM coverage | code MB/token |
|---|---|---|---|---|
| in_proj_qkv → Q | 2048×4096 | 16 | **20%** | 2.99 |
| in_proj_qkv → K | 2048×4096 | 16 | **20%** | 2.99 |
| in_proj_qkv → V | 4096×4096 | 32 | **40%** | 5.98 |
| in_proj_z | 4096×4096 | 32 | **40%** | 5.98 |
| out_proj | 4096×4096 | 32 | **40%** | 5.98 |
| gate_proj | 12288×4096 | 96 | 100% (1.2 CTA/SM) | 17.9 |
| up_proj | 12288×4096 | 96 | 100% (1.2 CTA/SM) | 17.9 |
| down_proj | 4096×12288 | 32 | **40%** | 17.9 |
| full-attn q_proj | 8192×4096 | 64 | 80% | 11.96 |
| full-attn k/v_proj | 1024×4096 | **8** | **10%** | 1.50 |
| lm_head (if routed) | 248320×4096 | 1940 | full | 1030 |

A linear-attention layer is ~489 µs of pure stream time at 100% BW, but
with these grids the narrow modules donate their SMs to idle: the
element-weighted utilization across a layer is **~45–50%** — before any
per-CTA inefficiency. The full-attn k/v_proj pair is the worst case in the
model: 3 MB of codes on 8 SMs = ~8% of the machine. (The W27 kernel
comment even names the mechanism — cu:3012–3017: "the block count, not
smem, is the occupancy limiter" — but the launch never grew a K-split.)

### 4.2 Cause 2 — latency-boundedness: no prefetch, one g-tile in flight

The K-loop (cu:2223–2271):

```c
for (int g = j4; g < G; g += 4) {
    uint4 qv1[B1];  ... ldg_nc_evict_first_v4(qv1[u], ...);   // load g
    uint4 qv2[B2>0?B2:1]; ...                                   // load g
    // fully-unrolled 64-pair compute, FIRST USE of qv is a shfl (dep)
}
```

There is **no double-buffer, no cp.async, no next-tile prefetch** — the
`g += 4` loop is not unrolled, and the first dependent consumer of each
load batch is the very next statement block. Each warp-iteration is
therefore `[issue 6–8 loads → stall ~600–900 ns on DRAM → ~1.5–2 k
instructions]`, and with only 8–10 warps resident per SM (1–1.2 CTAs × 8
warps vs 48 warp slots) the stall cannot be hidden: the resident warps
enter the loop in phase (post-`__syncthreads`) and stall together.

Per-warp serial chain: `G/4 = K/256` iterations. At K=4096 that is 16
iterations; at K=12288, 48. Measured module times ≈ 20–30 µs for every
K=4096 module regardless of N (see §9's probe — this is its most
predictable signature: **Q, K, V, Z, out, gate, up, k_proj, v_proj all
taking nearly identical µs despite 8–40x different byte counts**), and
~60–75 µs for down_proj. That signature is latency×iteration-count, not
bandwidth×bytes — the model a DRAM-bound kernel would show.

### 4.3 Cause 3 — the W28 prologue made down_proj worse

Shared memory per CTA: W27 = `2K + 2112` (26.8 KB at K=12288 → 3 CTAs/SM);
W28 = `2K + max(4·b_max, 2112)` = **57.3 KB at K=12288 → 1 CTA/SM**
(cu:3000–3028). down_proj is simultaneously the deep-K module (48 serial
iterations), a 32-CTA module (40% SM coverage), and now a 1-CTA/SM module.
The prologue also adds the segment butterfly (12 stages at K=4096) to the
front of every CTA. Small per CTA, but it lands on the critical path of
the worst-occupied module in the model.

### 4.4 Cause 4 — per-SM warp budget by construction

256 threads/CTA with grid ≤ 96 means ≤ 9.6 warps/SM average even on
gate/up — **20% of the 48-warp capacity** — and the kernel has no
mechanism (prefetch, cp.async, deeper ILP) to compensate. The A10G's
5.3 KB/SM in-flight requirement (§2) needs ~10 warps × 3 KB loads issued
back-to-back; the kernel issues one 3 KB batch per warp then stalls.

### 4.5 What is NOT wrong (checked and cleared)

* **Coalescing**: warp w reads `(wx·512 + lane·16)·B` consecutive bytes
  per tile — 512·B contiguous per warp, 16 B/thread aligned. Textbook.
* **LUT serving**: register palette + `shfl.idx` — zero smem, zero bank
  conflicts, ≤4 shfl per pair; shfl unit is not the limiter at ≤10
  warps/SM.
* **x staging**: one 8 KB global read per CTA (L2-broadcast), then smem
  broadcast reads (4-lane groups share addresses) — conflict-free.
* **resB re-reads**: every CTA re-reads resB (≤131 KB) — pure L2 traffic,
  ~4.6 GB/s equivalent across a layer, noise.
* **Register/spill risk**: ~60–80 regs incl. the `uint4` code buffers —
  far from the 255 wall (PERFORMANCE.md §5 gate).
* **Numerics/determinism**: the fixed-order butterfly + `red[]` tree —
  intact, and W29 must preserve exactly this contract (§8).

---

## 5. The routing census — what runs what at M=1 today

`PalettizedLinear.forward` (palettized_modules.py:1134–1307) tries, in
order: W28 FHT-fused GEMV (:1159) → W27 GEMV (:1263) → W26 dual (:1272) →
two-launch (:1276–1297). The gates:

| Gate | Value | Source |
|---|---|---|
| GEMV width pairs | (1,4),(4,1),(2,3),(2,4),(3,3),(3,0),(4,0) | `__init__.py` ~310–330; cu:2948–2954 |
| GEMV GS | {64,128,256,512,1024,2048} — **16/32 refused** | cu:2819–2825 |
| DUAL width pairs | the same 7 | `__init__.py:198–200` |
| DUAL GS | {64,128,256,512} — **16/32 refused** | `__init__.py:204` |
| Residual rank | R ≤ 16 — **rank-32 refused (heads!)** | cu:2889; palettized :942, :1050 |
| M | GEMV only M=1; dual M ≤ 16 | cu:2812; `DUAL_STREAM_MAX_M` |

Deployed census (the `__init__.py:193–197` comment, from the layer
metadata): (1,4)×45, (4,1)×24, (3,3)×61, (2,3)×3, (2,4)×6 two-stream +
single-stream riders — all inside the 7-pair table. The holes:

1. **6 modules at GS=16/32** (the W27 box report: 4× GS16, 2× GS32) fall
   through BOTH fused gates to the two-launch mma path **plus** the
   unfused residual, plus the explicit FHT (W28 only fuses inside the
   GEMV), plus casts. Identity unknown from the repo — the probe prints
   it. If any of them is mid-size the cost is the "~12x floor" regime.
2. **The heads**: MODEL_GEOMETRY.md §3 records lm_head as R2 hybrid =
   **2×idx4 blobs (the (4,4) pair) + rank-32 residual, gs=64-or-32**.
   (4,4) is in **neither** pair table, and R=32 exceeds the R≤16 cap —
   so lm_head (the single largest module: 1.03 GB/step, ~25% of all
   quant traffic; grid would be a healthy 1940 CTAs) runs the **full
   pre-W26 chain**: explicit FHT + two mma launches + ordered add + two
   cuBLAS residual GEMMs (resA alone is 15.9 MB fp16) + casts. Estimated
   3–5 ms/token. The W27 "97%" coverage number almost certainly counts
   instances, not bytes — the probe's per-route byte totals settle it.
   (Note the same two reasons, whichever fires, have the same fix.)

---

## 6. The modeling-side audit (what both arms pay)

The architecture is hybrid: **24 linear-attention (GatedDeltaNet) + 8
full-attention layers**, head_dim 256, GQA 4 KV, vocab 248,320, plus an
MTP layer (modeling.py; MODEL_GEOMETRY.md §1–2). At M=1:

* **GatedDeltaNet decode** (modeling.py:566–578) calls
  `torch_recurrent_gated_delta_rule` — the **pure-torch per-token loop**
  (:419–431: ~8–10 small kernels on the [1,32,128,128] state) — and the
  conv update (:515–524) calls `causal_conv1d_update`'s **torch fallback**
  (cat + copy + `F.conv1d` with groups=8192, :189–206). The decorators
  (`use_kernel_func_from_hub_with_fallback(..., "fla")`) route to hub
  kernels only if the transformers kernel-hub wiring is active;
  `flash-linear-attention==0.5.2` is in requirements.lock.txt:20–21 but
  nothing in the repo verifies the hub path actually engages. Per layer
  this is ~35–45 tiny kernels ≈ 75–110 µs; ×24 ≈ **2–3 ms/token**.
* **Full attention** (8 layers): SDPA with the W24 math-forced context
  under graphs (`_sdpa_math_ctx`, eval_greedy_match.py:370–395 — the
  StaticCache boolean-mask capture-verify fix). At q_len=1 the math
  backend materializes [1,16,1,S] scores — trivial bytes, ~10–14 small
  kernels × 8 layers ≈ 0.3–0.5 ms/token.
* **Norms/RoPE/gates/logits**: fp32-cast RMSNorm chains, the q+gate
  sigmoid split, [1,248320] logits + argmax — ~1 ms/token combined.
* **The eval loop**: one graph replay per token + argmax + one D2H sync —
  ~0.3–0.5 ms/token of host time, currently hidden inside 44.9 ms; will
  become visible (5–8%) at ~11 ms/token. See §8 step W30-d.

**Shared machinery total ≈ 3–4 ms/token.** This is the second-tier
optimization surface (worth ~2–3 ms once the GEMV is fixed), not the
first.

---

## 7. "Why don't we use flash attention?" — the complete answer

1. **At M=1 decode it cannot matter.** The 8 full-attn layers read
   4 KV heads × 256 × 2 (K,V) × 8 layers × S ≈ 33 MB/token at S=2048 —
   **0.05 ms at the wall**. Even replacing the entire math-SDPA chain
   with a perfect flash kernel saves ~0.3–0.5 ms of the 44.9 ms (§6).
   The 2x target lives in §4, not here.
2. **PyTorch's flash SDPA backend refuses this geometry on this box.**
   head_dim=256 + GQA + fp16 on sm_86 dispatches to the MATH backend —
   this is precisely why `attn_sm86.py` exists (its header: "where torch
   SDPA dispatches to the math backend and materializes (B,H,S,S) fp32
   scores"). On top of that, W24 **deliberately** forces math under graph
   capture: the flash/efficient dispatch on the box's torch/cuDNN pair
   did not honor the StaticCache [1,1,1,max_cache_len] boolean mask (the
   capture-verify gate failure; eval_greedy_match.py:62–79). Swapping
   that back in without re-proving the gate would reintroduce the exact
   W24 defect.
3. **We already have our own flash attention** — `scripts/attn_sm86.py`,
   an FA1-style Triton kernel with GQA (no repeat_kv), head_dim 256,
   causal, forward+backward, registered as the `flute_sm86` interface
   (modeling.py:692–756). It is the student path in the distill/training
   plane. It is not pinned in eval_greedy_match because (a) the eval
   asserts sdpa for parity between arms, and (b) **its own contract
   short-circuits S==1 to reference math** — decode would not use it
   even if pinned.
4. **Where flash WOULD pay**: prefill and the PPL arm (S≫1, where the
   math backend's (B,H,S,S) fp32 materialization is quadratic traffic —
   the GPU_SPEC.md §3 row (d) "wall"). If prefill latency or PPL wall
   time ever becomes the metric, pinning `flute_sm86` for the quant arm
   (after re-proving the capture gate, or restricting it to the
   non-graphed prefill pass) is the ready-made lever — the kernel is
   built, tested (tests/test_attn_kernel.py), and its absence of
   repeat_kv is strictly better than eager's materialized path.

---

## 8. The W29+ roadmap — ranked by (expected ms/token) ÷ (risk)

### W29 — GEMV v2: "fill the machine, hide the latency" (the headline round)

One kernel rewrite of `flute_kernel_gemv[_fht]_dual` + dispatch changes,
no artifact changes, no numerics changes on any existing route:

* **(a) Split-K grid.** `grid = (N/128, SPLIT)` with SPLIT chosen at
  launch so `(N/128)·SPLIT ≥ 160` (2 waves of 80): k/v_proj SPLIT=8+,
  Q/K 4, V/Z/out/down 2–4, gate/up 2. Partials in fp32, deterministic
  fixed-order reduction — NOT atomicAdd (the W27 decode-determinism
  contract, cu:2070–2079). Cleanest deterministic shape: each split-CTA
  writes `P[split][n]` (≤ 8×12288×4 B = 393 KB staging), and the LAST
  CTA per n-tile (atomicInc ticket — the ticket order is irrelevant to
  the sum order) folds P in **fixed split order** + resA·xBf + bias and
  rounds once to fp16. Same accumulation contract as W27's `red[]`,
  generalized. Expected: kills Cause 1 (~2–3x on narrow modules).
* **(b) Register double-buffer + two g-tiles per iteration.** `qv_cur /
  qv_nxt` (B1+B2 uint4 each), load g+4 before computing g; with split-K
  the per-warp chain is `K/(256·SPLIT)` iterations, so the pipeline
  covers the stall. Expected: kills Cause 2 (~1.5–2.5x per resident CTA).
* **(c) smem trim.** With split-K, the staged x row shrinks to the
  split's k-range: down_proj returns to ≤3 CTAs/SM even with the FHT
  staging; the FHT prologue stays per-CTA (it is O(K log K), ~2–4 µs —
  duplicated across splits, still cheaper than the launch it replaced;
  each split-CTA needs only its own k-slice of the rotated row, which it
  computes from the full butterfly it already runs).
* **(d) Table extensions (routing holes):** add the **(4,4)** pair (one
  more instantiation — lm_head) and raise the residual cap to **R ≤ 32**
  (xBf[32] + a 32-deep epilogue loop — the heads' r32); optionally GS=32
  and GS=16 (the warp's 64 rows span 2 or 4 groups — the palette select
  is a compile-time-constant function of the accumulator index `v`, so
  it costs registers only, zero per-code work). This closes §5's holes
  without touching the artifacts.
* **(e) Probe-driven acceptance:** W29 lands only if
  `probe_decode_routing.py` shows ≥300 GB/s aggregate (stretch: 450+),
  i.e. module-GEMM time ≤ ~12 ms; decode ≥ ~55 tok/s ≥ 1.3x dense on the
  box. Numerics gate: bit-identical decode tokens on the capture-verify
  prompts (the fixed-order reduction is what makes this provable), PPL
  unchanged at +3.52% (PPL never routes through the GEMV).

Expected end state after W29: **~10.5–13 ms/token ⇒ 77–95 tok/s ⇒
1.8–2.25x dense** (6.2 ms stream at 450–550 GB/s + 3–4 ms machinery +
host).

### W30 — machinery + composition (the tail, after the GEMV is honest)

* (a) Merge the per-layer independent GEMVs: QKV+Z as one launch (12352
  rows ⇒ 96+ CTAs, one x staging; four sign-vectors share ONE butterfly —
  `(x@H)·s_i` — the rotations differ only in the sign epilogue), gate+up
  similarly (24576 rows). Module launches 248 → ~90; tails disappear.
* (b) Wire/verify the `fla` kernels for the GDN decode
  (`fused_recurrent_gated_delta_rule` + `causal_conv1d_update`): the
  lockfile already carries fla 0.5.2 — verify the hub-kernel decorator
  actually engages on the box, else call `fla.ops` directly behind an
  env gate with the torch loop as fallback. Worth ~1.5–2.5 ms/token.
* (c) The logits chain: argmax in-graph / double-buffered pinned copy —
  removes the per-token D2H sync from the critical path (~0.3–0.5 ms).
* (d) `torch.compile(mode="reduce-overhead")` on the non-GEMM residue
  (norms, RoPE, gates) — only after (a)–(c), and only if the graph node
  count is still the visible tail.

Expected end state after W30: ~9–10.5 ms/token ⇒ **95–110 tok/s ⇒
2.2–2.6x dense.**

### Beyond — the levers that change the physics instead of the constant

* **MTP / speculative decode**: the checkpoint carries
  `mtp_num_hidden_layers: 1` (config). Accept-and-verify at 2 tokens per
  weight pass ⇒ up to ~1.8x effective tok/s at unchanged per-step bytes.
  The largest single lever after the GEMV is fixed; also the largest
  engineering risk (a verify loop + the MTP head's own quantization).
* **Batched decode (M=2–16)**: the same 3.7 GB stream serves every
  sequence — but M≥2 routes to the dual-mma path (12x floor at M≤16),
  so batching needs a GEMV-v2 variant with M=2–8 lanes or the W29 design
  extended to a skinny-M mma. The eval metric stays M=1; this is a
  serving-time option, not an eval-target item.
* **The dense arm itself will move**: dense runs at ~420 GB/s effective;
  if it were tightened (or run under the same graphs discipline), the
  2x bar shifts with it. The honest target is the wall, not the
  baseline: **dense floor 26.7 ms vs quant post-W29 ~11 ms — the
  asymptotic ratio is ~2.4x.**

---

## 9. Box verification plan (one run, three numbers)

```bash
# 1) The routing + bandwidth census (§4/§5 → measured):
python scripts/probe_decode_routing.py \
    --artifacts-dir /home/ubuntu/qwen3_5_9B_palettized \
    --heads-dir   /home/ubuntu/qwen3_5_9B_palettized_heads \
    --residual --model Qwen/Qwen3.5-9B --json probe_w29.json
#    Read: per-route {ms, GiB, GB/s}; the aggregate GB/s number is THE
#    baseline for W29 acceptance. Expect ~90–110 GB/s aggregate today,
#    every K=4096 module at near-identical µs (the latency signature),
#    and lm_head's route line showing two_launch or the GS fallback.

# 2) The stall proof (one narrow module, ncu):
ncu -k "regex:flute_kernel_gemv" \
    --metrics gpu__time_duration.sum,dram__bytes.sum, \
sm__warps_active.avg.pct_of_peak_sustained_active, \
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct, \
launch__grid_size,launch__occupancy_limit_shared_mem \
    python scripts/probe_decode_routing.py <same flags> --only out_proj --reps 3
#    Expect: grid 32 (vs 80 SMs), warps_active <25%, long-scoreboard
#    stalls dominant — Causes 1+2 in one screenshot.

# 3) The device facts (§2):
python -c "import torch; p=torch.cuda.get_device_properties(0); \
print(p.name, p.multi_processor_count)"
```

The probe is CPU-safe to import, modifies nothing, and its per-module
µs column is directly comparable against W29's per-module target
(bytes ÷ 450 GB/s).

---

## 10. W29 addendum (2026-10-09): the implementation landed — and one audit correction

The §8a-d plan is implemented (kernel_cutlass_streaming.cu:
`flute_kernel_gemv2_dual`, entries `qgemm_gemv2_stream` /
`qgemm_gemv2_fht_stream`; the wide tables in
flute_extended/flute_extended/__init__.py; the routing ladder in
scripts/palettized_modules.py tries the v2 first; the probe's route
census gained the w29 routes + the tiles×SPLIT grid column — see
docs/AUTO_SELECTION_GUIDE.md's W29 entry for the full bill). Two facts
recorded here because they correct §5/§6 of THIS document:

1. **The deployed GS census (§5's hole #1, refined).** The QKV
   components' metadata group_size (16/32) is STALE — the loader
   trusts the LUT geometry (W17 policy) and deploys them at the
   derived 512/1024, so the QKV modules are NOT GS 16/32 stragglers;
   the true GS 16/32 set is the 6 non-QKV modules (out_proj ×2,
   down_proj, v_proj ×2, up_proj). The QKV routing hole is the PAIR
   table, not GS: the components deploy as (4,3)/(3,4)/(4,4)/(2,2) —
   pairs the 7-entry tables never carried. The v2's 20-pair table
   closes both classes at once.

2. **The fla claim in §6 was wrong in a useful direction.** The
   transformers 5.17 `@use_kernel_func_from_hub_with_fallback`
   decorator resolves the kernel FROM THE INSTALLED PACKAGE at import
   (priority 2: "Original package" — no env opt-in needed), and the
   box lockfile carries flash-linear-attention 0.5.2 + causal_conv1d
   1.7.0, so the GDN decode and the conv update ALREADY run the
   installed kernels on the box. The "~2-4 ms/token" of §6's root
   cause 6 was mostly already recovered; W29 adds the direct in-body
   wiring as defense-in-depth (envs without the packages), the
   FLUTE_NO_FLA=1 A/B switch, and a one-shot failure latch. The §8e
   acceptance arithmetic is unchanged (the GEMV kernel efficiency is
   the whole remaining story).

Box verification: §9's probe run is the acceptance gate (aggregate
>= 300 GB/s; the W29 entry in AUTO_SELECTION_GUIDE.md carries the
full read-out instructions and the fallback diagnostics ladder).
