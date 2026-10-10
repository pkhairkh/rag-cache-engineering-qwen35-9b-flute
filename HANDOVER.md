# RAGGA Handover Document

## Current State (post-W11)

| item | status |
|---|---|
| CPU suite | 175 tests green (`python3 -m pytest src/rag/tests -q`; CPU lock: torch 2.14.1+cpu, transformers 5.19.0, faiss-cpu 1.15.1) |
| S-install math | ADJUDICATED CLEAN at real GDN fidelity (`src/rag/tests/test_install_real_math.py`): install ≈ ground truth 0.048 < 0.06; query-output tracking 0.019; contractivity 0.0014; reseed-vs-install separation 0.71 vs 0.019 |
| conv geometry | FIXED: full-window policy, d = 24,576 (segments 16,384 + 8,192, every FHT tile ≤ 64 KiB) — the GPU session's 16,384 truncation (which zeroed the last 8,192 coordinates of every conv read: every ingestion delta, every installed window, every decode window ran on a one-third-blinded conv state) is gone, with a pinned regression test |
| FHT dispatch | kernel auto-engages only for K it can tile (`fht._kernel_eligible`: multiple of 32, in [32, 65,504], every segment ≤ 16,384); K = 524,288 (S) and K ≥ 32,768 run the torch reference on-device — the 32,768-unit's 128 KiB single-segment tile no longer traps consumer GPUs |
| GPU tools | consolidated under `scripts/gpu/` (see below); the 15 root-level scripts and the committed `query_output*.txt` are gone |

## The W11 Findings (what the "generation garbage after install" was)

1. The pushed GPU-session code (62a3e2a) TRUNCATED the 24,576-coordinate conv
   window to 16,384 at `_init_conv` ("just truncate to 16384 elements for
   now" + TODO) while its own edited tests expected 24,576 — 7 tests failed
   on the pushed tree (the "165 tests passing" claim predates the truncation
   edit). Every conv read returned the window with coordinates 16,384:24,576
   (channels 4,096–6,143) ZEROED: ingestion deltas, chunk conv installs and
   decode windows were all built on that blinded state. The answer prefill
   consumes the installed (blinded) conv window at its boundary — that is
   the pipeline's generation corruption.
2. The S path is exonerated by construction: the install sum, the codes
   setters, the read path and the quantizer identity are bit-clean
   (test_install_real_math.py gate 1), the values the debug script verified
   were correct, and the delta rule is CONTRACTIVE in the state (5% state
   noise moves query outputs by 0.14%) — quantization-scale noise cannot
   produce garbage. Any residual GPU-side corruption after the conv fix is
   a kernel-route defect, not the math: run `bisect_install.py` (below).
3. The handover's isolation table (layer-0-S-only install → garbage) could
   not distinguish the conv corruption (present in BOTH its ✅ and ❌ rows —
   the ✅ "immediate `<|im_end|>`" is itself the degraded mode) from the
   state-scale difference; the fixed conv geometry removes the common
   defect.

## REQUIRED GPU-Box Procedure (after pulling W11)

```bash
cd /home/ubuntu/RAGGA && git pull

# 1. RE-INGEST — the pre-W11 snapshots carry 16,384-truncated conv codes;
#    installing them into a full-window layer raises the frame guard
#    ("...must be RE-INGESTED") by design.
python3 scripts/gpu/run_ingestion.py            # defaults: 100 docs, ingested_50k

# 2. Index (flat IP under 256 chunks; IVFADC above)
python3 scripts/gpu/run_index.py

# 3. Staged verification ladder (G1 pure model → G6 e2e; conf/rep metrics)
python3 scripts/gpu/verify_pipeline.py

# 4. If any stage fails: bisect one axis at a time (install content ×
#    kernel route), incl. the conv-tail dead-energy detector
python3 scripts/gpu/bisect_install.py           # and/or: --fla-off
```

`--fla-off` sets `FLUTE_NO_FLA=1` (pure-torch decode kernels, bit-identical
contract per modeling.py's wiring block) — the A/B that rules the FLA Triton
decode route in or out of any residual corruption.

## Tools (scripts/gpu/)

| script | purpose |
|---|---|
| `run_ingestion.py` | the §5 ingestion driver (argparse: corpus, n-docs, system prompt, bits, out-dir; resumable) |
| `run_index.py` | flat IndexFlatIP under 256 chunks, IVFADC above |
| `run_query.py` | the §6 flow: query prefill → vector → rerank → install → decode |
| `verify_pipeline.py` | the G1–G6 generation ladder with confidence/repetition metrics (exit code = failures) |
| `bisect_install.py` | install-content × kernel-route bisection: S-read drift check, conv-tail energy check, per-variant decode |

## Reference

- Contract: `SPECIFICATION.md` (N16.1 = the conv full-window policy; §6 = the install math).
- Design: `PROPOSAL.md` (D3 rotation frames, D4 delta protocol).
- RAG-plane page: `src/docs/RAG_PIPELINE.md` (module map, run matrix, disk layout).
- Execution record: `TASKS.md` (W10, W11 rows); narrative: the repo worklog.
