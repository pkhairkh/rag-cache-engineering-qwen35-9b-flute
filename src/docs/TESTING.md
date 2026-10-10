# Testing

Purpose: the gate suite — what runs where, what each gate pins, and the collection policy that keeps a CPU-only checkout green.
Authority: authoritative for the test layout; subordinate to `SPECIFICATION.md` for RAG semantics.
Status: synced from the main project @ ab78893, scoped to this repo Wv2-8 — this repo's suite is `src/rag/tests/` (161 CPU tests) plus two GPU-side standalone gates under `src/flute_extended/`; the main project's ~190-test kernel/training/eval suite is not part of this repo.

## 1. Running

```bash
python3 -m pytest src/rag/tests -q              # this repo's whole CPU suite (161 tests)
python3 -m pytest src/rag/tests/test_query.py -q  # one file
python3 -m pytest src/rag/tests -q -m "not slow"  # skip the full-scale CPU tests
```

On the GPU box, additionally (extension built first — [BUILD.md](BUILD.md) §2/§5):

```bash
cd src/flute_extended
python test_flute.py           # multi-backend + differential gates
python test_qwen_weights.py    # real-weights oracle
```

## 2. The suite map

| File (all under `src/rag/tests/`) | Gates |
|---|---|
| `test_turboquant.py` | the quantizer core: round-trips, per-width constants, concentration, qjl flag parity |
| `test_codebooks.py` | the Lloyd-Max codebook constants + the production unit dims |
| `test_tq_cache.py` | the online TurboQuant cache wrapper (stub route): read/write interception, quantize-on-write |
| `test_tq_cache_live.py` | the cache wrapper against the real `transformers` cache machinery |
| `test_m1m2.py` | the two global memories M1/M2 + their model wiring |
| `test_hooks.py` | the 9-hook capture contract + the snapshot capture vector order |
| `test_ingest.py` | the delta-protocol ingestion + the restartable driver |
| `test_index.py` | the IVFADC index build + preselect/rerank |
| `test_query.py` | the §6 eight-step query flow + install math |
| `test_finetune.py` | the lean fine-tune loop + the LUT artifact codec + the trainer-plane grep gate |
| `test_evals.py` | the phase-gate harness: every evals subcommand, GPU-refusal hardening, flag parity |

The module↔test↔spec-section map with the entry points:
[RAG_PIPELINE.md](RAG_PIPELINE.md). The main project's suite map is
main project, not part of this repo: the decode GEMV family + dequant
oracle + attention/FHT/merge/GS-grid tests (main project:
`tests/test_gemv*.py`, `tests/test_dequant_reference.py`,
`tests/test_attn_kernel.py`, `tests/test_fht.py`, `tests/test_merge.py`,
`tests/test_gs_grid_parallel.py`), the routing/wiring layer (main
project: `tests/test_palettized_modules*.py`), the calibration
pipeline (main project: `tests/test_capture_store.py`,
`tests/test_streaming_calibration.py`,
`tests/test_interleave_resume.py`), the auto resolver + toy
drivers (main project: `tests/test_auto_compression.py`,
`tests/test_toy_*.py`), the training
stack (main project: `tests/test_lut_gradients.py`,
`tests/test_two_stream_training.py`,
`tests/test_dual_stream.py`, `tests/test_joint_trainer*.py`), the eval
plane (main project: `tests/test_eval_*.py`),
`tests/test_vram_ledger.py` and the contract ledgers.

The two GPU-only scripts (`src/flute_extended/test_flute.py`,
`src/flute_extended/test_qwen_weights.py`) import `_C` at module scope;
collecting them without a built extension is a collection error by
design. Here `pytest.ini` (testpaths `src/rag/tests`) never collects them —
they stay byte-identical and run explicitly on the box (in the main
project, `conftest.py` excludes them).

## 3. What the tests pin (and what they deliberately do not)

- **Numerics**: every kernel's output vs the fp32 reference chain on
  CPU-computable inputs (main project suite) — and, here, the TurboQuant
  round-trips, the snapshot digests, and the install/requant math against
  the same pure-torch references.
- **Determinism**: fixed association order (split-K partial folds,
  j-word folds) is pinned, not just tolerance — decode must be
  reproducible across CUDA-graph replays; the RAG plane pins the same
  property on codes (identical digests call after call).
- **Formats**: pack/unpack round-trips at every width; idxN(4) ≡ idx4
  byte-identity; the blob map vs the kernel's decode; the snapshot npz
  layout and its sha256 integrity member.
- **Protocols**: the eval's graph capture/verify loop, the PPL
  windowing, the OOM ladder (main project suite); here — the ingestion
  delta protocol, the query eight-step flow, the evals subcommand
  contracts.
- **Decisions**: the auto resolver's choices on fixed spectra (the
  ledger, the budgets, the refine gate; main project suite).

They do NOT pin wall-clock performance (clocks, L2 state, and driver
skew make that a box concern — the probe owns it), and they do not
exercise the CUTLASS dense baseline (build-time optional).

## 4. Adding a gate

Follow the existing shape: deterministic tensors from a fixed seed,
the pure-torch reference as ground truth, one behavior per test, and a
failure message that names the contract clause it pins. GPU-requiring
gates belong in the box scripts, not in `src/rag/tests/`.
