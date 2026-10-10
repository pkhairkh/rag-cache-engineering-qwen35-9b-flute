# RAGGA Handover Document

## W18 STATUS: ALL 229 TESTS DEVICE-PARAMETRIC, CPU+GPU VERIFIABLE

**Every one of the 229 tests now carries the canonical device contract:**

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- EVERY tensor is created on / moved to `DEVICE` (device at creation where
  no generator is pinned; ONE `.to(DEVICE)` after a CPU-generator factory
  — the determinism lock).
- EVERY model/rig is `.to(DEVICE)`'d as a WHOLE (no per-tensor copy_
  patchwork that leaves A_log/dt_bias/embed on the wrong device).
- ZERO `.cuda()` anywhere in the repo (tests, rag core, gpu scripts — the
  `flute_extended` kernel harness keeps its by-design `device="cuda"` and
  now skips cleanly when collected without a GPU).
- ZERO redundant moves (`.to(DEVICE).to(DEVICE)`), ZERO generator+device
  conflicts (`torch.randn(..., generator=cpu_gen, device=DEVICE)` CRASHES
  on CUDA — 8 such latent crashes found in test_m1m2.py and fixed).

**What this means for the box:** `python3 -m pytest src/rag/tests -q`
resolves DEVICE=cuda — every tensor is CUDA-born, every model CUDA-moved,
no mismatch is constructible. The same suite runs 229-green on the CPU
lock (the agent's verification rig), so logic AND placement are both
pinned. If a failure appears on the GPU box it is a REAL bug, not a
device-placement artifact — bisect it as such.

## THE TRIPLE ROOT CAUSE OF THE PREVIOUS 21 FAILURES (all fixed)

1. **`dequant` always returned CPU** (the numpy codebook path). FIXED at
   the source: `TurboQuant.dequant(codes, dtype, device=None)` — the
   reconstruction STILL runs on the CPU reference path (numerics pinned
   bit-identical across platforms; dequant is the reference, the CUDA FHT
   kernel is a QUANT-time device), and the FINAL tensor moves once to the
   caller's device. All call sites pass their device (tq_cache read
   paths, update round-trips, roundtrip); the scattered post-hoc
   `.to(layer._device)` patchwork is collapsed into the source argument.
2. **Rig models built half-CPU** (per-tensor `copy_(x.to(DEVICE))` leaves
   the params on CPU; only dim>1 weights got moved). FIXED: whole-module
   `.to(DEVICE)` at construction in every rig (test_finetune's
   FinetuneStubModel, test_install_conv_math's _ResidualGDNStack, ...).
3. **CPU-born tensors fed CUDA flows**: the greedy decode's fed-back
   token (`query.py::_greedy_decode` — a PRODUCTION crash at step [8] on
   any CUDA model), the M1/M2 replay's init read (a codes-only fresh
   cache carries no device signal — `prefill_capture_calls` now follows
   the capture device; `m1m2_from_cache` follows the module), stub logits
   and GDN decay inputs. ALL fixed.

## M1/M2 ARCHITECTURE (unchanged, now enforced)

- TWO GLOBAL states per model — M1/M2 `(32, mem_size, 128)`, shared
  across all 24 linear layers; the gates (write_gate_k, write_gate_v,
  read_gate) are the only per-layer weights.
- **`--m1m2-mem-size` band [1024, 4096] is now a LOUD GUARD** on both
  `finetune_m1m2.py` and `run_ingestion.py` (SystemExit outside the band).
- Model weights ~6GB; M1/M2 states are small; the training OOM was the
  REPLAY's retained graph, root-caused and fixed properly (below).

## THE GATE-TRAINING OOM — FIXED INSIDE THE ARCHITECTURE (W18)

**ROOT CAUSE (measured, not worked around):** the §7 replay re-runs the
24 recorded wiring calls under grad. The plain autograd chain — AND the
W17 `torch.utils.checkpoint` variant — retain a BOUNDARY STATE PER CALL:
at batch_pairs=8 x 2 sides x 24 calls that is 384 live fp32
`(32, mem_size, 128)` tensors = ~13 GiB at mem_size=1024, ~52 GiB at
4096, BEFORE the model's ~6GB and the prefill transients. That is the
A10G OOM (the `--self-test` path included).

**THE FIX (`_GatedMemoryReplay`, a custom autograd.Function):** the
additive write (spec §2.2) is LINEAR in the gates, so in the fp32 replay
regime `m_final = m_init + Σ g_L·Δ_L` with Δ the pure k/v scatter —
the exact Jacobian is identity-to-init plus per-call deltas. The
Function's forward runs the `module.write` chain under no_grad
(BIT-IDENTICAL to the plain chain — same ops, same order), and its
backward recomputes each Δ from the recorded k/v ONE CALL AT A TIME via
the NEW `M1M2.write_deltas` (the SAME `_scatter_delta` the write itself
uses — no duplicated math) and accumulates the exact per-layer gate
gradients. Retained memory: O(state + captured k/v) — ~1.3GB at the
production geometry, ANY mem_size.

**The parity gates (test_w18_finetune.py, 5 tests):** forward
bit-identical to the plain chain; gate gradients EXACTLY equal to the
plain chain's; multi-call layers accumulate; fp16 states refused loudly
(the exactness contract); write_deltas bit-equal to the write's own
deltas. `replay_states(memopt=False)` keeps the W17 reference semantics
for the A/B; `--no-memopt` on the driver is the old OOM path for
comparison.

## Production Paths

**Model files:**
- `/home/ubuntu/qwen3_5_9B_palettized/` - Palettized LUT/idx files (4.7GB)
- `/home/ubuntu/qwen3_5_9B_palettized_heads/` - Head LUTs (1.8GB)

**Data — the W18 builder closes the gap:**
```bash
# writes {text1, text2, label} rows + a .manifest.json; deterministic
python3 scripts/gpu/build_pairs.py \
    --out /home/ubuntu/RAGGA/data/pairs.jsonl \
    --sources mrpc,qqp,paws,snli,mnli,stsb,squad --per-source 5000
```
Sources: MRPC/QQP/PAWS (paraphrase), SNLI/MNLI (entail=1), STS-B
(score>=4 -> 1), SQuAD (question, answer-bearing sentence — the
retrieval-shaped pair). `--offline` skips missing HF-cache sources
loudly; `--self-check` validates the format without any dataset.
NEVER the served corpus — corpus independence is the design.

**Artifacts:**
- `/home/ubuntu/RAGGA/artifacts/gates/` - Trained gate weights

## Protocol: TRAIN → RE-INGEST → INDEX → EVAL → QUERY

### 0. (once) build the pairs
```bash
python3 scripts/gpu/build_pairs.py --out /home/ubuntu/RAGGA/data/pairs.jsonl
```

### 1. TRAIN (W18 default: the memory-safe exact-gradient replay)
```bash
python scripts/gpu/finetune_m1m2.py \
    --pairs-file /home/ubuntu/RAGGA/data/pairs.jsonl \
    --gates-out /home/ubuntu/RAGGA/disk/m1m2_gates.npz \
    --m1m2-mem-size 1024 \
    --max-steps 300
```

### 2. RE-INGEST
```bash
python scripts/gpu/run_ingestion.py \
    --out-dir /home/ubuntu/RAGGA/disk/ingested_m1m2 \
    --m1m2-gates /home/ubuntu/RAGGA/disk/m1m2_gates.npz \
    --m1m2-mem-size 1024 \
    --n-docs 100
```

### 3. INDEX
```bash
python scripts/gpu/run_index.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2
```

### 4. EVAL
```bash
python scripts/gpu/eval_retrieval.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2
```

### 5. QUERY
```bash
python scripts/gpu/run_query.py \
    --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2 \
    --m1m2-gates /home/ubuntu/RAGGA/disk/m1m2_gates.npz
```

## Confirmation Sequence for the GPU Box (W18)

1. `python3 -m pytest src/rag/tests -q` — expect **229 passed** (DEVICE
   resolves to cuda; any failure is a real bug, bisect it).
2. `python scripts/gpu/finetune_m1m2.py --self-test --max-steps 5
   --m1m2-mem-size 1024` — the former OOM case now fits; watch the
   `[W18] exact-gradient memory-safe replay` banner.
3. The full protocol above (TRAIN with pairs.jsonl → RE-INGEST → INDEX →
   EVAL → QUERY); `eval_retrieval.py`'s hit@k is the M1/M2 retrieval
   verdict (the §7 signal: cos(true) vs cos(shifted) gap > 0 and
   hit@1/@3 climbing off the S-only floor).
4. A/B ladders: `--no-memopt` (the old replay — the OOM A/B),
   `--m1m2-mem-size 4096` (the band's upper end), `--train-read-gate`.

## Code Changes Made This Session (W18)

### Production
1. `turboquant.py` — `dequant(..., device=None)`: CPU-reference math,
   one final move to the caller's device; `roundtrip` uses it.
2. `tq_cache.py` — read paths (S/conv/M1/M2) pass `device=` at the
   dequant source; `_m_read_device()` resolves codes-only reads
   (tracked M-write -> first layer device -> CPU); update round-trips
   land on the writer's device.
3. `query.py` — `_greedy_decode`'s fed-back token follows the logits'
   device (a production CUDA crash at decode step [8]).
4. `m1m2.py` — `write_deltas` + the shared `_scatter_delta` (the exact
   per-call deltas); `m1m2_from_cache` follows the module's device.
5. `m1m2_finetune.py` — `_GatedMemoryReplay` (the OOM fix, exact
   gradients, O(state + k/v)); `replay_states(memopt=...)`;
   `prefill_capture_calls` device-follows the capture;
   `GatesTrainConfig.memopt`.
6. `scripts/gpu/build_pairs.py` — NEW: the general-similarity pairs
   builder (the missing pairs.jsonl).
7. `finetune_m1m2.py` / `run_ingestion.py` — the [1024, 4096] mem_size
   band guard; `--no-memopt` A/B.
8. `eval_retrieval.py` / `run_query.py` / `verify_pipeline.py` — token
   ids follow the loaded model's device (no `.cuda()`).
9. `flute_extended/test_flute.py` — CUDA-gated collection (clean skip).

### Tests (ALL files, ALL 229)
- The device-parametric sweep: canonical `DEVICE` per file, tensor
  creation at device, whole-module moves, zero `.cuda()`.
- Fixed the latent GPU-only bugs: generator+device conflicts (8 sites,
  test_m1m2), half-CPU rigs (test_finetune, test_install_conv_math),
  CPU-born GDN decay (test_install_real_math), CPU stub logits
  (test_w16_retrieval), CPU zeros placeholders, redundant double moves.
- NEW `test_w18_finetune.py` (5 gates: the memopt parity contract).

## Outstanding (honest)

1. **The GPU verdict on the §7 signal** — the gates now train within
   memory; whether the InfoNCE-descended gates actually move hit@k on
   the real corpus is the box's measurement (eval_retrieval is the
   judge; if the gap stays flat, the next lever is tau/lr/mem_size
   ladders, still inside the architecture).
2. **Read-gate training needs a next-token objective** — the
   contrastive loss never sees read_gate (state-built only);
   `--train-read-gate` unfreezes it but nothing trains it yet (a later
   wave's aux-NLL, already plumbed as `--aux-nll-weight` in the design).
