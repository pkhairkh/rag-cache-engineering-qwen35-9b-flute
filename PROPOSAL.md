# PROPOSAL.md — W20: The GPU-Resident TurboQuant Pipeline

Purpose: the design layer for **getting the M1/M2 fine-tune (and every
online cache path) fully on the GPU — zero CPU interrupts, zero device
round-trips in the hot loop**. This re-opens the `PROPOSAL.md` slot
(SPECIFICATION.md's authority chain cites it); the pre-W20 design record
(D1–D5, P0–P7) was retired in the doc sweep — historical decision IDs
still cited in code comments refer to that retired record. The IDs below
are fresh: the design is **G-TQ** (GPU-resident TurboQuant), its build
phases **W20.1–W20.5**, risks **R20.1–R20.6**, gates **G20.1–G20.5**.
Status: **rev 1 — the ALL-GPU directive** (the user's follow-up,
verbatim: "WE NEED ALL OF TURBOQUANT ON GPU! WE NEED NO CPU PATHS AS
FAR AS POSSIBLE!"): rev 0 kept the CPU numpy path as a live runtime
reference; rev 1 removes it from the runtime entirely. One
device-parametric torch path serves every box — CUDA tensors in
production (never a numpy array, never a sync), CPU tensors only where
the surrounding box has no GPU (this dev box, the test twin). The legacy
numpy behavior is pinned by frozen probe fixtures inside the TEST
SUITE, not by a live code path. Proposed; no code changed in this
commit.

Trigger — the GPU box's W19-post report (verbatim findings):

1. `d = 2^19 = 524,288 > 32,768` (the CUDA FHT kernel's eligibility
   ceiling) — every S/M1/M2 unit misses the kernel.
2. `~50` online TQ quants per prefill (24 S + 24 M1 + 24 M2 updates per
   layer wiring; plus 24 conv writes and the per-chunk S reads — the
   full count is 120–192 TQ units per prefill, §1).
3. Each round trip runs a 19-stage torch **reference** butterfly on the
   CPU (the dequant side; §2/S5).
4. This owns **~95% of the 263 s capture cycle** (16 prefills per
   refresh batch, ~16.4 s per prefill — pure CPU TQ arithmetic).
5. User directive: "ALSO TURBOQUANT NEEDS TO FULLY GET ON GPU. NO CPU
   INTERRUPTS. WE NEED TO GET EVERYTHING ON GPU."

The W19 amortization (refresh_every) did its arithmetic job — the
capture pool cut prefill count 4× — but it divides a CPU-bound constant:
263 s / 4 ≈ 66 s/step of capture amortized on top of ~0.5 s of
replay+backward. The trainer is not slow because of the backward; it is
slow because **every quantize-on-write and dequantize-on-read crosses
the PCIe bus twice and runs its reconstruction math on one CPU thread**.
This proposal removes the crossings and the CPU math — not by touching
the backward (it is already exact and cheap, W18/W19), but by making
the quant/dequant round trip a resident GPU operation.

**REV 1, THE HARDENED DIRECTIVE**: rev 0 spoke of a "GPU twin with the
CPU reference kept intact". That is withdrawn. The runtime keeps ONE
path — device-parametric torch — and the numpy implementations are
DELETED from it, not branched around. Every quant, dequant, codebook
access, pack/unpack, QJL sketch, snapshot install and retrieval-vector
build in the serving/training stack runs torch ops on the caller's
device. What remains outside torch is exactly two FOREIGN boundaries
(§3.5): the npz file format at the disk edge and the faiss CPU index at
the retrieval edge — and even there, all MATH is torch/GPU; only the
final bytes hand off.

---

## 1. The measured hot path (deep code analysis, HEAD 99e3914)

One capture prefill drives the model forward through 24 linear-attention
layers. Per layer, per forward call, the wiring executes this sequence
(`src/scripts/modeling.py:715–754`):

| # | op | code site | TQ work |
|---|---|---|---|
| 1 | S read (per fla chunk) | `tq_cache.py:198–205` (`_StateView.__getitem__`) | 1 dequant |
| 2 | S write | `tq_cache.py:402–419` (`update_recurrent_state`) | quant + dequant |
| 3 | conv write | `tq_cache.py:399` (`update_conv_state`) | quant |
| 4 | M1 read | `modeling.py:727` → `tq_cache.py:689` (`read_m1`) | dequant |
| 5 | M2 read | `modeling.py:728` → `tq_cache.py:720` (`read_m2`) | dequant |
| 6 | M1 write | `modeling.py:752` → `tq_cache.py:676` (`update_m1`) | quant + dequant |
| 7 | M2 write | `modeling.py:753` → `tq_cache.py:710` (`update_m2`) | quant + dequant |

At T≈512 with the fla chunked scan, ops 2–7 fire 24× per prefill and op
1 fires once per chunk per layer: **120–192 TQ round trips per prefill**,
each on a d=2^19 unit (S/M1/M2) or d=24,576 (conv). Every one of them
pays the CPU toll below.

## 2. Root-cause inventory: the six CPU syndromes

The quant/dequant pair (`turboquant.py:470–512` and `667–777`), pinned
line-by-line for a d=2^19 CUDA input:

| ID | what | code pin | cost shape |
|----|------|----------|-----------|
| S1 | `norm = float(x32.norm().item())` — a device→host sync that stalls the CUDA stream | `turboquant.py:491` | pipeline drain per unit |
| S2 | `y.cpu().numpy()` — the rotated vector leaves the GPU as a pageable 2 MiB copy | `turboquant.py:501` | D2H per unit |
| S3 | `np.searchsorted` + `pack_bits` — CPU numpy bucketize + bit-pack | `turboquant.py:503–508` | single-thread CPU per unit |
| S4 | `unpack_bits` + `centroids[idx]` — CPU numpy unpack + fancy-index gather | `turboquant.py:770–774` | single-thread CPU per unit |
| S5 | `fht.fht_adjoint(y_t, ...)` with `y_t = torch.from_numpy(...)` — the adjoint 19-stage butterfly runs **on a CPU tensor** because the reconstruction enters as numpy | `turboquant.py:775–776` | **the monster**: ~95 tensor ops on a (1, 2^19) CPU row, ~50–150 ms |
| S6 | the reconstruction `.to(device)` — 2 MiB H2D back to where the math started | `turboquant.py:705–706, 753–754` | H2D per unit |

The asymmetry that explains the box's profile: the **quant-side** FHT
(`turboquant.py:500`) receives a CUDA tensor and its butterfly runs on
the GPU (the torch reference is device-agnostic; ~114 kernel launches,
~1–2 ms) — so quant pays S1–S3 but not a CPU butterfly. The
**dequant-side** adjoint receives a numpy-born CPU tensor (S4's
`from_numpy`) and runs the whole 19-stage butterfly on the CPU — S5.
With 120–192 round trips per prefill × ~80–160 ms of CPU-dominated work
each, the arithmetic lands exactly on the measured 16.4 s/prefill,
263 s per capture cycle.

Why the CUDA FHT kernel does not already save us
(`src/flute_extended/fht.py:345–364`): one FHT segment stages its tile
in shared memory; `_MAX_KERNEL_SEGMENT = 16,384` coords = 64 KiB — the
opt-in budget. K=2^19 is a single 2^19 segment = 2 MiB of shared memory
— physically impossible. `_kernel_eligible` therefore rejects it and
`auto` mode falls back to the torch butterfly **on the input's device**
— CUDA for quant, CPU for dequant. The conv window (24,576 =
16,384 + 8,192) is the only production unit that is kernel-eligible
today.

## 3. The design: G-TQ — GPU-resident TurboQuant

The contract (rev 1): **ONE device-parametric torch path. When the
input tensor is CUDA, `quant()` and `dequant()` never leave the device,
never sync the stream, and never construct a numpy array. When the
input tensor is CPU (the GPU-less dev box, the test twin), the SAME
code runs the SAME torch ops on CPU tensors — there is no second,
numpy implementation anywhere in the runtime.** The legacy numpy
semantics (searchsorted buckets, packed-bit layout, QJL signs, the
npz field layout) are pinned by frozen probe fixtures in the test
suite — assertions about outputs, not a live path that can drift (and
that is the point: two live paths drift; one path plus fixtures cannot).
The FHT backend keeps `fht.py`'s auto discipline (CUDA kernel when
eligible + CUDA input, torch butterfly otherwise) — that is a KERNEL
dispatch, not a CPU detour: on a CUDA input the butterfly is never on
the CPU.

### 3.0 The CPU-elimination audit (what gets deleted, module by module)

Measured at HEAD c4c1091 — every numpy / sync / transfer touchpoint in
the TQ runtime stack and its fate under rev 1:

| module | CPU touchpoints | fate under G-TQ rev 1 |
|---|---|---|
| `turboquant.py` | **103** (`.item()` norms, `y.cpu().numpy()`, `np.searchsorted`, `pack_bits`/`unpack_bits`, `centroids[idx]` gathers, `from_numpy` adjoints, `np.sign`/`np.array` in the QJL path, the split path's `einsum/argsort/packbits`) | quant/dequant/QJL/split rewritten as torch ops on the input's device; numpy calls DELETED (not bypassed) |
| `index.py` | 3 (retrieval vectors → `.numpy()` at `:372/:473/:510`) | vector math stays torch/GPU; ONE hand-off at the faiss boundary (§3.5) |
| `evals.py` | 5 (`.item()` MSE reads, concentration/norm `.numpy()`, the CRC id) | measurement harness goes torch; `.numpy()` only where numpy IS the consumer (faiss/matplotlib) |
| `m1m2.py` | 1 (`positions.min().item()` in a guard message, `:345`) | replaced by a torch-native format of the same value (no stream sync) |
| `snapshot.py` | 1 (`raw.item()` metadata decode, `:464`) | stays — the npz disk edge (§3.5) |
| `hooks.py` | 0 live sites (a comment mentions np.packbits padding) | comment updated; the popcount itself becomes a torch op if used |
| `tq_cache.py`, `install.py` | 0 direct sites (they call into turboquant) | inherit the fix through the single path |
| `codebooks.py` | the solver (pure numpy, run once per (b, d), artifacts committed) | stays OFFLINE — an artifact build, not a runtime path; the LOADED books mirror to device tensors at first use (§3.1) |

### 3.1 W20.1 — the pure-torch GPU round trip (no new CUDA code)

The math of quant/dequant is already torch for the heavy parts; the
numpy parts are REPLACED (deleted) by their exact torch equivalents,
executed on the input's device:

| numpy step today (DELETED) | torch replacement (input's device) | parity |
|---|---|---|
| `float(x32.norm().item())` | `n = x32.norm()` — a 0-dim CUDA tensor carried in the codes object; `.item()` only at the disk edge (§3.6) | exact |
| `np.searchsorted(boundaries, y)` | `torch.searchsorted(b_gpu, y_gpu)` | **left-side semantics verified identical, ties included** (both default side=left; probe below) |
| `np.packbits` bit-packing | torch shift/or on uint8 views — **or deferred entirely** (§3.4) | bit-identical gate G20.4 |
| `centroids[idx]` gather | `torch.index_select(centroids_gpu, 0, idx_gpu)` | exact |
| `np.unpackbits` | torch bit ops (or skipped — unpacked indices ride the hot loop) | bit-identical gate G20.4 |
| `fht.fht_adjoint(from_numpy(y))` | `fht.fht_adjoint(y_cuda)` — the butterfly is already device-agnostic; with a CUDA input it runs ~114 GPU launches, ~1–2 ms | allclose ≤ 1e-5 rel (sum-order only, G20.4) |
| `.to(device)` at return | nothing — the tensor never left | — |
| QJL `np.sign`, `np.array` sketches | `torch.sign`, direct tensor ops (qjl flag off in production — the path is unified regardless) | exact (sign of ±fp32) |
| split path `np.einsum/argsort/packbits` | `torch.argsort` (stable), gather/scatter, torch pack | stable-sort + tie gates pin it (R20.2) |

Codebooks (boundaries + centroids, ≤ 16 fp32 entries each at b ≤ 4) are
LOADED from the committed npz once and mirrored as device tensors at
first use per quantizer instance — kilobytes, immutable thereafter. No
numpy object is held by a live quantizer; `Codebook` gains a
`device_tensors()` accessor and the numpy fields become load-time-only.

**The deletion discipline (why fixtures, not a twin):** before any
numpy line is deleted, its exact behavior on a frozen probe set
(reference states at d=2^19 and 24,576, fixed seeds) is captured — codes
byte-for-byte, reconstructions to the fp32 bit — into the test suite.
The numpy implementation then dies; the fixtures hold it accountable
forever. This is the same discipline `flute_train_kernels` uses for its
G-B1…G-B5 gates (debug twins + canaries pinning behavior, not shipping
a second production path).

**Expected effect (arithmetic, to be measured by the W19 phase timers
already riding every history entry):** per round trip, CPU ~80–160 ms →
GPU ~2–5 ms (two 19-stage torch butterflies ≈ 1–2 ms each + sub-ms
searchsorted/gather/pack). Capture cycle 263 s → **~8–20 s** (16
prefills × ~0.5–1.2 s, now dominated by the model FLOPs themselves);
at refresh_every=4 that is **~2–5 s/step**, and 300 steps ≈ **10–25
minutes** instead of half a day. No new CUDA, no build step, no kernel
risk — this is the phase that lands first.

### 3.2 W20.2 — the codes device contract (torch-native codes)

`TQCodes.idx_lo/idx_hi/norm` become torch tensors BY CONSTRUCTION — the
quantizer emits device-resident uint8/0-dim tensors, full stop. The QJL
sketch fields likewise (int8 tensor + 0-dim norm). There is no numpy
carrier anywhere in the object graph at runtime. The ONLY conversion is
the disk edge: `to_arrays()` (npz write) performs one batched
`.cpu()` per unit and packs ON GPU before the copy; `from_arrays()`
(npz read) lifts the bytes back to the caller's device in one H2D per
unit. Snapshot write/read, install, the index loader — all consumers go
through these two edges, and their MATH (multi-chunk sums, requant,
retrieval-vector assembly) is torch on the caller's device (W20.5).

### 3.3 W20.3 — `fht_large`: the multi-block FHT kernel for K = 2^19

After W20.1 the 2^19 butterflies are torch-on-GPU: correct, but ~114
launches each, and the prefill issues 192+ of them (S/M1/M2 quant and
adjoint, per layer) — ~22,000 kernel launches per prefill whose
launch-queue overhead alone is ~100+ ms. The kernel fix (the handover's
W20 candidate, now specified):

**Stage decomposition.** A Hadamard butterfly over K=2^19 has 19 stages
with pair distance `len_ = 1, 2, …, 2^18`. Every stage with
`2·len_ ≤ 16,384` has all its pairs fully inside an aligned 16,384-wide
tile → 13 tile-local stages; the remaining 6 stages (`len_ ≥ 16,384`)
cross tiles.

**Two-pass kernel** (`kernel_fht_large.cu`, new file in
`flute_extended/src`, bound alongside the existing entries):

- **Pass 1 (tile-local, one launch):** grid = K/16,384 = 32 blocks;
  each block stages its tile in 64 KiB shared memory (the existing
  opt-in budget — same as today's kernel) and runs the 13 local stages
  in-register, fixed pairing discipline preserved (low = u+w,
  high = u−w, ascending — byte-order identical to `fht_reference` and
  `fht_stage`, the contract `kernel_fht.cu` already pins).
- **Pass 2 (cross-tile, ≤ 6 launches):** for each cross stage, an
  elementwise global-memory pass — thread i reads (u[i], u[i+len_]),
  writes (u+w, u−w). Bandwidth-bound: ~8 MiB traffic per stage at
  ~600 GB/s ≈ 15 µs; six stages ≈ 0.1 ms.

Total ≈ **0.3–0.5 ms per 2^19 FHT** (vs ~1–2 ms torch, vs 50–150 ms
CPU), and the launch count collapses 22,000 → ~700 per prefill.
Eligibility is decided by the existing `_kernel_eligible` table
extended with the large path: any K that is a multiple of 32 with
K > 16,384 routes to `fht_large` (K ≤ 16,384 keeps today's single-block
kernel unchanged — its binaries and gates are untouched).

**The differential-twin discipline (borrowed verbatim from
`flute_train_kernels` — see §5):** `fht_large` is pinned against
`fht_reference` on random probes (allclose, rel ≤ 1e-5, fp32 sum-order
only); a poison-canary debug build flags any uncovered tile cell; the
deterministic two-pass order (fixed thread→pair mapping, no atomics)
makes runs repeatable to the bit.

Expected: capture cycle ~8–20 s → **~5–10 s**; step time → **~1.3–2.5
s**; combined with §3.4, under 1 s.

### 3.4 W20.4 — trainer-level levers (already half-built by W19)

- `m1m2.py:345`'s guard `.item()` dies with W20.1 (a torch-native
  format call replaces the sync).
- `evals.py`'s measurement sites (5) go torch — the MSE reads and
  concentration stats become tensor reductions; `.numpy()` appears
  only at a true external consumer.

- `refresh_every` 8 (the driver flag exists) — capture amortizes
  another 2×; the staleness bound stays lr × N (the W19 argument; the
  held-out spread is the gate).
- Keep the codes **unpacked** in the hot loop: the round trip's dequant
  skips `unpack_bits` entirely (the indices are already material); VRAM
  cost ~50 MiB for ~96 units — trivial on the A10G 24 GB. Packing
  happens once, on GPU, at snapshot time.
- The pairs loader tokenizes up front (driver-side, already) — no
  per-step CPU work remains beyond the `log_every` prints.

### 3.5 W20.5 — the serving side goes GPU too (ALL of TurboQuant)

"All of TurboQuant" includes the deployment path, not just the
trainer. Today `install_snapshot`'s requant round, the index loader's
per-chunk vector builds (`index.py:372/473/510`), and the eval harness
dequantize unit-by-unit through the CPU path. Under rev 1 they become
BATCHED GPU passes: one launch shape over all 24 S units + M1 + M2 (the
install requant), one over every chunk's units (the retrieval-frame
build), with the codes lifted to device once (W20.2's read edge). The
faiss add/search stays on its CPU index (faiss-cpu is the pinned
dependency) — but its INPUT vectors are produced on GPU and handed off
once, already normalized, already centered (the W16 frame math on
device).

### 3.6 The two remaining numpy edges (foreign formats, not CPU paths)

1. **The npz disk format** (`to_arrays`/`from_arrays`, snapshot read/write,
   the committed codebook artifacts): numpy is the FILE FORMAT. The pack,
   unpack, sum and requant math on both sides of the bytes is torch on
device; the conversion is one batched copy at the disk edge. There is
no per-unit CPU math anywhere on the way in or out.
2. **The faiss CPU index** (add/search at the retrieval edge): an external
   library boundary. Vectors arrive GPU-computed; the hand-off is a
   single `.numpy()` per batch. (A faiss-gpu swap is explicitly out of
   scope — the CPU index is tiny, the queries are few, and the W16
   centered rerank dominates cost.)

Everything else in the TQ stack is torch. The dev box (this machine,
CPU-only) runs the SAME single path on CPU tensors — that is not a CPU
path, that is the same code with a different device argument. **This
wave (the proposal) changes zero code — no tests were run or needed;
the last green suite (237) pins the pre-W20 tree.**

## 4. Numerics and parity contract (fixtures, not twins)

1. **Codes are ints — they must be IDENTICAL to the frozen fixtures.**
   `torch.searchsorted` and `np.searchsorted` share left-side semantics
   (verified on ties: both place a boundary-exact value in the upper
   bucket — probe: `np/torch searchsorted([-1,-.5,0,.5], .5) → 3/3`).
   The pre-deletion numpy outputs on the frozen probe set (codes,
   byte-for-byte) become the test fixtures; the single torch path must
   reproduce them exactly (G20.4).
2. **Reconstruction is fp32 sum-order — allclose ≤ 1e-5 rel** vs the
   fixtures. The only divergence source is the butterfly's summation
   order; this is the existing kernel-vs-reference discipline
   (`fht.py`'s differential twins), extended to G-TQ.
3. **Snapshots stay byte-identical.** The npz artifacts are the
   protocol's interchange format (TRAIN → RE-INGEST → INDEX → EVAL →
   QUERY); the disk-edge packing produces the same packed bytes (gate
   G20.4 pins bit-identity vs the frozen fixtures).
4. **One path means no drift.** rev 0's "CPU reference kept intact"
   is withdrawn precisely because two live paths drift; rev 1 keeps ONE
   runtime path and holds it to fixtures captured from the numpy era.
   A corpus re-ingest is not required by the change itself (same codes,
   same bytes — gate G20.5); the protocol's re-confirm sequence runs
   whenever gates are re-confirmed as usual.

## 5. The qlora backward-kernel question, answered precisely

The directive: "maybe we can reuse / alternate the backward kernel from
`qwen3_5_9B_flute_qlora_v1.3`". Deep-read of that repo's
`flute_train_kernels` (`kernel_backward_gemm.cu`, `kernel_lut_grad.cu`,
`docs/KERNELS.md`):

- **What they are:** the fused WEIGHT-side LUT-GEMM backward —
  `grad_X[M,K] = grad_Y[M,N] @ W[N,K]` with `W` dequantized from LUT
  codes, plus the `dL/dLUT` scatter (mma-computed dW tiles scattered
  into per-(group, code) partials, deterministic two-pass reduce). That
  is the QLoRA `nn.Linear` backward: gradients flowing through
  LUT-gathered **weights**.
- **What the M1/M2 replay needs:** gradients through a STATE-side gated
  scatter — `∂m_final/∂g_L = Δ_L`, accumulated as
  `<grad_m, Δ_L>` per call (`m1m2_finetune.py:378–397`). Different op
  family: no GEMM against W, no LUT rows, no mma fragments. The W18
  `_GatedMemoryReplay` backward is already exact and costs ~0.5 s per
  step — it is not the bottleneck (the W19 phase timers settle this on
  the box: backward ≈ 2% of wall).
- **What IS literally shared:** `flute_extended` itself — the same tree
  in both projects. `kernel_fht.cu`'s `fht_forward` / **`fht_backward`**
  are the kernels both stacks use, and the dequant adjoint is exactly
  `fht_backward`'s math (`g @ T^T`). The moment G-TQ feeds it a CUDA
  tensor (W20.1) the existing kernel engages for every eligible K, and
  W20.3 extends that same kernel file for K=2^19. The reuse the
  directive asks for is real — it is the **FHT family**, not the GEMM
  backward.
- **What transfers as discipline** (into W20.3, verbatim from
  `flute_train_kernels`): deterministic two-pass accumulation — private
  register partials → fixed warp-shuffle tree → fixed-order fold, **no
  atomics anywhere**; one-launch fusion with `__launch_bounds__`
  pinned; the debug-twin + poison-canary build gates (their G-B1…G-B5
  family) that make coverage an arithmetic identity; fixed association
  order end-to-end so torch.equal is assertable. `kernel_fht_large.cu`
  adopts this whole discipline.

## 6. The speedup ladder

| step | change | capture cycle (16 prefills) | s/step (refresh 4) | risk |
|---|---|---|---|---|
| W19 (measured) | CPU TQ round trips | **263 s** | ~66 + 0.5 replay | — |
| W20.1 | torch-native quant/dequant, numpy DELETED (§3.0–3.2) | ~8–20 s (est.) | ~2–5 | low |
| W20.3 | `fht_large` multi-block kernel (§3.3) | ~5–10 s (est.) | ~1.3–2.5 | medium |
| W20.4 | refresh 8 + unpacked hot codes (§3.4) | halves the amortized share | ~0.7–1.3 | low |
| W20.5 | serving-side batched GPU install/index/evals (§3.5) | — (ingest/index/query path) | — | low |

(All post-W20.1 numbers are arithmetic estimates anchored on the
measured 16.4 s/prefill CPU constant and the ~1–2 ms GPU butterfly
measurements; the W19 phase timers in every history entry verify each
rung on the box.)

## 7. Risk register

| ID | risk | mitigation |
|---|---|---|
| R20.1 | GPU bit-packing diverges from numpy `packbits` | differential gate: bit-identical packed bytes on probe sets (G20.4) |
| R20.2 | `searchsorted` tie divergence CPU↔GPU | left-side semantics verified identical incl. ties; gate asserts equal codes on probes |
| R20.3 | VRAM growth (unpacked uint8 codes, mirrors, capture pool) | ~50–100 MiB against 24 GB; the W18 pool already tracks `_calls_bytes` |
| R20.4 | the single-path rewrite ripples into every consumer | that is the POINT (no dual-path drift); the frozen fixtures + the 237-suite (same code, CPU tensors) + GPU gates hold it |
| R20.5 | `fht_large` numerics / coverage | differential twin vs `fht_reference` (1e-5), poison canary, no-atomics determinism — the flute_train_kernels gate family |
| R20.6 | capture staleness at refresh 8/16 | bounded by lr × N (W19); the held-out spread is the acceptance metric, unchanged |

## 8. Acceptance gates (GPU box)

- **G20.1** — `finetune_m1m2.py --refresh-every 4`: the phase line
  shows capture ≤ 20 s/cycle after W20.1, ≤ 10 s after W20.3 (the W19
  timers already measure it).
- **G20.2** — step time ≤ 3 s (W20.1) → ≤ 2 s (W20.3); 300 steps ≤ 15
  min end-to-end.
- **G20.3** — the W19 convergence gates still hold: loss leaves ln(8)
  at step 0, `gk/gv_absmax > 0`, held-out spread positive.
- **G20.4** — parity: the single torch path reproduces the frozen
  fixtures exactly — codes byte-identical on CUDA AND CPU tensors,
  reconstruction allclose ≤ 1e-5 rel, packed bytes bit-identical.
- **G20.5** — protocol integrity: TRAIN → RE-INGEST → INDEX → EVAL →
  QUERY on the G-TQ path produces snapshot digests identical to the
  pre-W20 CPU artifacts (byte-identical npz) AND the serving side
  (install requant, index vectors, eval metrics) matches the frozen
  fixtures on device.

## 9. Build order

W20.1 + W20.2 (one wave: torch-native quant/dequant with numpy
DELETED + torch-native codes + the frozen fixtures) → box confirmation
(G20.1/2/3/4) → W20.3 (`fht_large` + its gate family) → box
confirmation → W20.4 flags (refresh 8, unpacked hot codes) → W20.5
(the serving-side batched GPU passes) → the unchanged deployment
protocol with the trained gates. The implementation waves run the full
suite on the single path (CPU tensors here, CUDA tensors on the box —
same code) plus the new GPU gates; this proposal wave is doc-only and
changed no code.
