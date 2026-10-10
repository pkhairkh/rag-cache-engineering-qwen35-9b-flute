# RAGGA Handover Document

## Current State

W17 deployed: **M1/M2 are ACTUALLY attached now** (the loader instantiates the vendored Qwen3.5 classes with the wiring), the §7 gate trainer exists end-to-end (capture/replay/InfoNCE/artifact), and the full pipeline (ingest → snapshot → index → query → install) is M1/M2-aware with loud geometry guards. **224 CPU tests green** (204 + 20 W17 gates). The box must run the TRAIN → RE-INGEST → EVAL sequence below.

| Component | Status |
|-----------|--------|
| M1/M2 activation (the loader) | ✓ FIXED (W17) — vendored class + text-config flags + `key_mapping` |
| Gate fine-tune (§7) | ✓ NEW — `scripts/gpu/finetune_m1m2.py` + `src/rag/m1m2_finetune.py` |
| m1m2_mem_size experiments | ✓ plumbed (ingest 1024 default; query auto-resolves from disk) |
| Pipeline M1/M2 flow | ✓ snapshot/loader/index/install all geometry-adaptive |
| CPU test suite | ✓ 224 passed (the c37f20e hard-coded `cuda` regression fixed) |
| Retrieval quality with TRAINED gates | ⏳ PENDING the GPU box's fine-tune + re-ingest + eval |

---

## The W17 Root-Cause Diagnosis (why "M1/M2 NOT ACTIVATED")

The W16-post handover said "loader uses default model class" — the full
picture is a **triple silent no-op**, each layer of which was pinned by a
CPU gate:

1. **The native class has no wiring at all.** `AutoModelForCausalLM` on
   `Qwen/Qwen3.5-9B` resolves the NATIVE transformers
   `Qwen3_5ForCausalLM`. The M1/M2 read/write block exists ONLY in this
   repo's vendored `src/scripts/modeling.py` (the byte-faithful text-only
   copy + the wiring). Setting `config.use_m1m2 = True` on the native
   class does nothing — there is no code that reads it.

2. **The flags were set on the wrong config object.** The hub checkpoint
   is COMPOSITE (`Qwen3_5Config` wrapping `text_config`). The vendored
   `Qwen3_5TextModel.__init__` does `_get_text_config(config)` and reads
   `use_m1m2`/`m1m2_mem_size` from the **TEXT** config. The c37f20e fix
   set them on the composite wrapper — the second silent no-op.

3. **The vendored class silently re-initializes the text weights** (found
   by the W17 prototype, would have been a garbage-model generator).
   `from_pretrained`'s conversion table maps the composite checkpoint's
   `model.language_model.*` keys → `model.*` for LIBRARY classes only:
   the vendored module counts as "custom code" (`is_custom_code()`:
   `__module__` not under `"transformers."`) and the lookup is SKIPPED —
   19/28 text weights re-initialize at random. Fix: the OFFICIAL
   `key_mapping` kwarg carries the same prefix strip explicitly
   (`_TEXT_FROM_COMPOSITE_KEY_MAPPING` in palettized_modules.py).

Plus one found by the new tests: **from_pretrained re-initializes the
gate vectors through `_init_weights`** (they are always MISSING keys) —
the vendored `_init_weights` now restores the P3 zeros/ones (without it
the loader served denormal garbage gates).

## The Fix (all within the architecture, no fallbacks)

`load_palettized_model(..., use_m1m2=True, m1m2_mem_size=128,
m1m2_gates_path=None)`:

- instantiates `modeling.Qwen3_5ForCausalLM.from_pretrained(...,
  config=config, key_mapping=_TEXT_FROM_COMPOSITE_KEY_MAPPING)`;
- sets the flags on the TEXT config (mirrored on the composite);
- **verifies the wiring landed** (loud RuntimeError — shared module,
  ordinals, geometry; the W16 failure was SILENT);
- loads the trained-gates artifact when given (geometry-validated).

Zero-init gates are bit-unchanged no-ops: `use_m1m2=True` reproduces the
native class's outputs EXACTLY (pinned by test_w17_2 through a TQCache
forward) — activation is safe for every flow; the retrieval signal
appears only after the §7 fine-tune opens the gates.

## The §7 Gate Trainer (corpus-INDEPENDENT by design)

**The gradient problem it solves**: the production online loop is
gradient-dead for the WRITE gates BY DESIGN (quantize-on-write severs
autograd — "the cache states act as constants in the graph"). The
trainer builds the differentiable state OUTSIDE the quantized loop:

1. **CAPTURE** (`prefill_capture_calls`): the text prefills under
   `no_grad` through the production TQCache (reseeded from the system
   reset point); forward hooks on the shared m1m2 module record every
   layer's call (k, v, layer_idx, positions) detached. `m_init` is read
   BEFORE the prefill (the reseeded system state — reading it after
   hands the replay the final state; the test rig caught exactly this).
2. **REPLAY** (`replay_states`): the recorded calls re-run under
   `enable_grad` through the module's own `write` on a live chain — the
   additive write makes the result exactly
   `m_init + Σ_L g_L·Δ_L` (the §2.2 path-independence property) —
   the noise-free, gate-differentiable proxy of the cache's held state
   (pinned within quant-rel-MSE by test_w17f_2).
3. **LOSS**: InfoNCE with in-batch negatives over the B pair-states
   (sim = 0.5·(cos_m1 + cos_m2), temperature tau) on GENERAL similarity
   pairs — MRPC/QQP/PAWS/SNLI/MNLI/STS-B/SQuAD converted to a pairs
   JSONL (`{"text1", "text2", "label": 1}`); the served corpus is NEVER
   a training input.

Trained: ONLY the 3 gate vectors (`freeze_all_but_gates`, fp32 masters,
everything else frozen). `read_gate` stays at the P3 one-init by default
(the contrastive loss never sees it — the READ affects generation, not
the retrieval state; the installed memories steering generation through
the read IS the spec's design).

**Artifacts**: `save_gates`/`load_gates` (.npz: the 3 gate vectors +
geometry identity; a mem_size-mismatched file is a LOUD error). Serving:
`load_quant_model(..., m1m2_gates_path=...)`.

## The Deployment Protocol (ORDER MATTERS — the box sequence)

```bash
# 1. TRAIN the gates on general pairs (NOT the served corpus)
#    (convert MRPC/QQP/SNLI/... offline to pairs.jsonl; --self-test for
#    a mechanics smoke first)
python3 scripts/gpu/finetune_m1m2.py --pairs-file /home/ubuntu/pairs.jsonl \
    --gates-out /home/ubuntu/RAGGA/disk/m1m2_gates.npz \
    --m1m2-mem-size 1024 --max-steps 300

# 2. RE-INGEST with the trained gates (the system state + every chunk
#    delta then live in the trained-gate regime) — fresh out-dir
python3 scripts/gpu/run_ingestion.py \
    --out-dir /home/ubuntu/RAGGA/disk/ingested_m1m2 \
    --m1m2-gates /home/ubuntu/RAGGA/disk/m1m2_gates.npz \
    --m1m2-mem-size 1024 --n-docs 100

# 3. INDEX (disk-side; the loader auto-includes the m1/m2 units; the
#    codebook digests pin the ACTUAL M1/M2 unit dims)
python3 scripts/gpu/run_index.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2

# 4. EVAL retrieval (mem auto-resolved from system_state.npz)
python3 scripts/gpu/eval_retrieval.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2

# 5. QUERY (same gates; geometry drift fails loudly)
python3 scripts/gpu/run_query.py --disk-dir /home/ubuntu/RAGGA/disk/ingested_m1m2 \
    --m1m2-gates /home/ubuntu/RAGGA/disk/m1m2_gates.npz

# the A/B ladders:
#   --m1m2-mem-size 128|1024|4096|8192 (re-ingest per value; decode cost
#     grows ~linearly with mem — 24 M1/M2 quantizations per token)
#   --no-m1m2 (the W16-exact behavior, bit-identical)
#   untrained gates (omit --m1m2-gates: zero-norm m1/m2 units, retrieval
#     unchanged — isolates the gate-training effect)
```

## What the W17 Gates Pin (src/rag/tests/test_w17_loader.py, test_w17_finetune.py)

1. vendored load == native load (every text weight bit-equal, both
   directions) + zero-gate forward parity through TQCache;
2. the wiring (attached/shared/ordered/P3 init) + flags-on-TEXT-config
   (the composite-only placement must NOT wire);
3. the loud verifiers (unwired model, mem_size drift, codes-vs-module
   geometry at the first forward);
4. the e2e flow: ingest → snapshot m1/m2 units → loader dims →
   query vector → verbatim install restores m1/m2 codes bit-exact;
5. the gates artifact roundtrip through the loader + geometry-mismatch
   loud; mem auto-resolution from disk;
6. the trainer: recorder sees every layer; replay == the cache's held
   state within quant noise (and == the manual write chain exactly);
   gradients reach ONLY the write gates; InfoNCE descends on the model
   (before/after on the same pairs); train_gates smoke; freeze; the
   artifact roundtrip.

## Verification (CPU box, the committed state)

```
python3 -m pytest src/rag/tests -q          # 224 passed
python3 -m pytest src/rag/tests/test_w17_loader.py src/rag/tests/test_w17_finetune.py -v
```

## Files Reference

| file | role |
|---|---|
| `src/scripts/palettized_modules.py` | `load_palettized_model` — the W17 fix: vendored class + text-config flags + `key_mapping` + `_verify_m1m2_attached` + gates load |
| `src/scripts/loader.py` | `load_quant_model(..., use_m1m2, m1m2_mem_size, m1m2_gates_path)` |
| `src/scripts/modeling.py` | the vendored Qwen3.5 (byte-faithful + M1/M2 wiring); `_init_weights` restores the P3 gates |
| `src/rag/m1m2_finetune.py` | NEW — capture/replay/InfoNCE/freeze/train_gates/save_gates/load_gates |
| `scripts/gpu/finetune_m1m2.py` | NEW — the §7 driver (pairs JSONL, self-test, val spread, gates artifact) |
| `scripts/gpu/_bootstrap.py` | `load_model` threads the M1/M2 params (the post-load flag-set no-op removed) |
| `src/rag/ingest.py` | `m1m2_mem_size_from_system` + `check_m1m2_geometry` (the loud drift guards) |
| `src/rag/tq_cache.py` | `read_m1/read_m2` geometry validation (loud, actionable) |
| `src/rag/index.py` | `build_index(unit_dims=...)` — the codebook digests pin the actual M1/M2 units |
| GPU tools | run_ingestion/run_query/eval_retrieval/bisect/verify: `--m1m2-mem-size`/`--m1m2-gates`/`--no-m1m2`; query side auto-resolves from disk |

## Expected Results After the Box Runs the Protocol

1. The gates artifact trains (loss descends; the held-out spread
   cos(true) − cos(shifted) > 0 on general pairs);
2. re-ingested snapshots carry nonzero m1/m2 units whose content differs
   per chunk (the §4 vector's last two units become discriminative);
3. eval_retrieval's hit@k improves over the S-only 50% baseline (the
   centered frame now scores m1/m2 content, not just S);
4. If retrieval improves but answers degrade: read_gate stays 1 (P3) —
   the memories steer generation through the read; that is the spec's
   design, but a read-gate A/B (--train-read-gate + an NLL aux in a
   later wave) is the in-architecture lever.

## Design Notes (the W17 decisions)

- **mem_size default 1024 in the tools** (the handover's first
  recommendation; the loader default stays 128 = the spec §2.2
  geometry). The quantizer resolves any mem (codebooks solved+committed
  at d = 2^22/2^24/2^25); power-of-two mem keeps the unit pow2.
- **The trainer trains on "zeros + deltas"** (the system reset point is
  built at P3 zero gates) while deployment's states are
  "system-at-trained-gates + deltas" — consistent because the W16
  centered frame subtracts the system vector at scoring time.
- **Capture memory scales with T** (k/v per layer per prefill), not
  mem_size; `--max-tokens 256` + `--batch-pairs 8` keeps the rig inside
  A10G headroom; `replay_states` checkpoints per call on CUDA when the
  state exceeds 1M elements.
- **decode cost**: every decoded token writes M1/M2 through 24 layers
  (one quantization of the full memory each) — the cost grows ~linearly
  with mem_size; 128 ≈ one extra S-layer per token, 1024 ≈ 8x that.
- The 3 gates are the ONLY trained parameters (the spec §7/N28 "train
  the M1/M2 read/write gates"); the LUT/linear-attn groups of
  `finetune.py` remain the next-token N28 path (a later wave).
