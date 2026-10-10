# rag-cache-engineering-qwen35-9b-flute

Purpose: cache-engineered RAG on Qwen3.5-9B (FLUTE idxN hybrid palettization) — the LUT model's cache IS the retrieval vector; no separate embedder, no chunk text on disk, no re-prefill, TurboQuant online (3.5-bit, quality-neutral).
Authority: SPECIFICATION.md is the contract (24 linear-attention + 8 full-attention layers; per-layer S + global M1/M2 + conv_state caches; online TurboQuant cache API; retrieval vector; §5 ingestion / §6 query flows; fine-tune); PROPOSAL.md is the build plan (arXiv:2504.19874 mapped onto the tensors, design decisions D1–D5, phased build with gates, risk register); TASKS.md is the execution plan (waves 0–9, tasks, DoD gates).
Status: RAG plane implemented and green — 175 tests, tag `base-synced-v1.3` (base synced from main @ ab78893; v1 build tagged `cpu-code-complete`); production runs + SPECIFICATION §9–§10 measurements happen on the GPU box.

## Layout (current tree)

| path | contents |
|---|---|
| `SPECIFICATION.md`, `PROPOSAL.md`, `TASKS.md`, `README.md`, `HANDOVER.md` | contract / build plan / execution plan / this file / the GPU-box session handover (state, fixes, next steps) |
| `pytest.ini` | pytest config: testpaths `src/rag/tests`, marker `slow` (full-scale CPU tests) |
| `scripts/poc_toy/` | the validated CPU toys (SPECIFICATION §12): `toy_cache_as_vector`, `toy_kimi_two_caches`, `toy_ivfadc_caches`, `toy_cache_engineered_rag`, `toy_1000_chunks` |
| `scripts/gpu/` | the GPU-box operational tools (W11): `run_ingestion.py`, `run_index.py`, `run_query.py` (the pipeline stages, argparse), `verify_pipeline.py` (the G1–G6 generation ladder), `bisect_install.py` (install-content × kernel-route bisection), `_bootstrap.py` (paths + model load) |
| `src/scripts/` | model side: `modeling.py` (Qwen3.5-9B hybrid + M1/M2 wiring), `loader.py` (`load_quant_model`), `palettized_modules.py` (idxN LUT forward), `attn_sm86.py` (SM86 attention kernel vehicle) |
| `src/rag/` | the RAG plane (SPECIFICATION §1–§8): 13 modules — `turboquant`, `codebooks` (+ `codebooks/`), `tq_cache`, `m1m2`, `hooks`, `ingest`, `snapshot`, `install`, `query`, `finetune`, `lut_export`, `index`, `evals`; `tests/` = 12 test files (175 tests) |
| `src/flute_extended/` | the W4 inference CUDA extension: `src/` kernels (`kernel_gemv.cu` + `_mlp/_multi/_splitk`, `kernel_streaming.cu`, `kernel_fht.cu`, `kernel_debug_simple.cu`, `kernel_cutlass_dense.cu`, `bindings.cpp`, `gemv_host.cpp`), `include/flute/` headers, nested `flute_extended/` package (`idxN.py`), `fht.py` (pure-torch FHT, CPU-legal), `setup.py`, `tools/`, `test_flute.py` + `test_qwen_weights.py`, `SHA256SUMS` |
| `src/docs/` | 12 format/build contracts: ARCHITECTURE, BUILD, EVALUATION, HARDWARE, KERNELS, MODEL_GEOMETRY, PALETTIZATION, PERFORMANCE, QUANTIZATION, ROUTING, RUNBOOK, TESTING (+ `qwen3_5_9b_config.json`) |
| `src/requirements*.txt` | minimal deps + the known-good pin set (`requirements.lock.txt`) + the CPU lock (`requirements-cpu.lock.txt`) |

## Run

| goal | command / entry point |
|---|---|
| run the RAG test suite | `python3 -m pytest src/rag/tests -q` (175 tests; `-m "not slow"` skips full-scale) |
| run evals self-tests (CPU) | `python3 src/rag/evals.py <roundtrip\|streaming\|margin\|recall\|e2e\|ledger> --self-test` (artifacts under `evals_out/`; without `--self-test` the four real-data commands refuse — GPU box only) |
| load the pre-built model (GPU box) | `PYTHONPATH=src/scripts python3 -c "from loader import load_quant_model"` → `load_quant_model(artifacts_dir, model_name, device, forward="kernel"\|"reference")` (idxN hybrid LUT model + heads; `reference` is the CPU-legal route) |
| ingest a corpus (GPU box) | `python3 scripts/gpu/run_ingestion.py` (argparse: corpus, n-docs, system prompt, bits, out-dir) → `disk/snapshots/chunk_XXXXX.npz` (TurboQuant codes only; the driver API stays `src/rag/ingest.py::IngestDriver`) |
| build the index / answer a query (GPU box) | `python3 scripts/gpu/run_index.py` + `python3 scripts/gpu/run_query.py` (FAISS flat under 256 chunks, IVFADC above; the API stays `src/rag/index.py::build_index` + `src/rag/query.py::answer_query`) |
| verify / bisect generation on the GPU box | `python3 scripts/gpu/verify_pipeline.py` (G1 pure model → G6 e2e, conf/rep metrics) · `python3 scripts/gpu/bisect_install.py` (S-read drift, conv-tail energy, FLA on/off A/B) |
| kernel bring-up (GPU box) | `src/docs/BUILD.md` + `src/docs/RUNBOOK.md` |

## Scope

| item | in this repo? | reason |
|---|---|---|
| palettizer (dense → LUT export) | NO | the model + heads artifacts are provided pre-built and loaded via `src/scripts/loader.py::load_quant_model` |
| parent trainer (QLoRA/distillation stack: `qlora`, `qlora_gemm`, `qlora_fallback`, `trainer`, `capture`/`loss`/`muon_optimizer`, `flute_train_kernels`) | NO | cut in the parent project (a700100), stays cut; the SPECIFICATION §7 fine-tune is the lean GPU-box loop in `src/rag/finetune.py` — straight-through LUTs via `PalettizedLinear.make_trainable()` on `forward="reference"`, served as full LUT artifacts (`pretrained_luts/`, §11), not QLoRA adapters |
| eval plane (production measurement) | NO | measurement gates run on the GPU box via `src/rag/evals.py` real-data mode; CPU boxes run `--self-test` fixtures only |
| RAG plane + CPU toys + W4 kernels + docs | YES | this repo's deliverable: SPECIFICATION §1–§8 under `src/rag/`, the §12 toys, the FLUTE W4 extension, the format/build contracts |

Authority chain: SPECIFICATION.md > PROPOSAL.md > TASKS.md.
