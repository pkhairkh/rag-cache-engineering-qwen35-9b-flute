# RAG pipeline — the RAG-box page

Purpose: the cache-engineered RAG plane under `src/rag/` in one page — module map, run matrix, disk layout, entry points.
Authority: subordinate to `SPECIFICATION.md` (§1–§8 define the semantics; this page restates the operational facts); authoritative for nothing.
Status: written Wv2-8 for THIS repo (the one page with no main-project counterpart); every number restated inline.

## 1. Module map (13 modules under `src/rag/`)

| module | one line | test file (under `src/rag/tests/`) |
|---|---|---|
| `turboquant.py` | the TurboQuant quantizer core: FHT rotation + per-coordinate Lloyd-Max, 3.5-bit codes | `test_turboquant.py` |
| `codebooks.py` (+ `codebooks/`) | the precomputed Lloyd-Max codebooks, bits × dims (1024/32768/524288) | `test_codebooks.py` |
| `tq_cache.py` | `TQCache` (a `DynamicCache` subclass): quantize-on-write, dequantize-on-read; owns the conv full-window geometry policy (N16.1: d = flat size rounded to 32) | `test_tq_cache.py`, `test_tq_cache_live.py` |
| `m1m2.py` | the two global memories M1/M2 and their read/write wiring | `test_m1m2.py` |
| `hooks.py` | the 9-hook capture harness (hook 0 after layer 0, 1–8 after layers 3,7,…,31) | `test_hooks.py` |
| `ingest.py` | the delta-protocol ingestion + the restartable `IngestDriver` | `test_ingest.py` |
| `snapshot.py` | the codes-only per-chunk npz codec + sha256 integrity | `test_ingest.py` |
| `install.py` | the §6 install math: `sum_turboquant_codes`, `install_from_disk` | `test_query.py`, `test_install_real_math.py` (W11: install-vs-ground-truth at real GDN fidelity, S-path isolation), `test_install_conv_math.py` (W13: the S+conv COMBINATION through the real GDN forward — install-vs-truth, no-variant-asymmetry, decode stability) |
| `query.py` | the §6 eight-step query flow: `answer_query` | `test_query.py` |
| `finetune.py` | the §7 lean fine-tune loop (straight-through LUTs, ~500 steps A10G) | `test_finetune.py` |
| `lut_export.py` | the fine-tuned LUT artifact codec (`pretrained_luts/`) | `test_finetune.py` |
| `index.py` | the IVFADC index: `build_index` / `preselect` / `rerank` | `test_index.py` |
| `evals.py` | the phase-gate measurement harness (P1–P6 + the ledgers) | `test_evals.py` |

Housekeeping, not in the 13: `__init__.py`, `_paths.py` (the sys.path
anchor — no dedicated test file).

Suite: `python3 -m pytest src/rag/tests -q` → 179 tests (13 files), CPU-only.

GPU-box operational tools live under `scripts/gpu/` (W11, revised W13):
`run_ingestion.py`, `run_index.py`, `run_query.py` (the pipeline stages,
argparse over the box defaults), `verify_pipeline.py` (the G1–G6
generation ladder with confidence/repetition metrics — G6 judges conf
too since W13) and `bisect_install.py` (install-content × kernel-route
bisection: the per-variant S-read check against the CORRECT reference
—the W12 "DRIFT" rows were this script's own reference bug, retracted—,
the CUDA-kernel-vs-reference FRAME check at the live conv geometry, conv
read fidelity + tail energy, gen conf numbers in the matrix, and the
TRUE-DOC raw control that separates semantic document-continuation from
corruption; FLA on/off A/B) — see HANDOVER.md for the post-W13
procedure.

## 2. Run matrix

| box | runs |
|---|---|
| CPU | `python3 -m pytest src/rag/tests -q` (179 tests; `-m "not slow"` skips full-scale) |
| CPU | `python3 src/rag/evals.py <roundtrip\|streaming\|margin\|recall\|e2e\|ledger> --self-test` (artifacts under `evals_out/`) |
| GPU | kernel build + the ladder gates: `src/docs/BUILD.md` §2/§5 (`test_flute.py`, `test_qwen_weights.py`) |
| GPU | model load: `load_quant_model(artifacts_dir, model_name, device, forward="kernel")` (`forward="reference"` = the CPU-legal route) |
| GPU | ingestion: `python3 scripts/gpu/run_ingestion.py` (argparse; the driver API stays `IngestDriver(model, system_token_ids, chunks, out_dir)`) → `disk/snapshots/` |
| GPU | index + query: `python3 scripts/gpu/run_index.py` + `python3 scripts/gpu/run_query.py` (the APIs stay `build_index(vectors_iter, IndexConfig())` → `answer_query(model, query_token_ids, system, index, loader)`) |
| GPU | generation verification / bisection: `python3 scripts/gpu/verify_pipeline.py` (G1–G6) · `python3 scripts/gpu/bisect_install.py` (`--fla-off` = `FLUTE_NO_FLA=1`) |
| GPU | the evals real-data gates: the four commands `streaming`, `margin`, `recall`, `e2e` (they refuse loudly without a GPU / without real data; `roundtrip`/`ledger` degrade to the self-test stub when no model/disk dir is given) |
| GPU | fine-tune: `src/rag/finetune.py::train` (~500 steps, A10G) |

## 3. Disk layout (one `disk/` root; codes only — no chunk text, no token IDs, no fp16)

| path | contents | size |
|---|---|---|
| `disk/ivfadc_cache.index` (+ `.meta.json`) | the FAISS IVFADC index on dequantized cache vectors | — |
| `disk/snapshots/system_state.npz` | the persisted system-prompt reset point (absolute codes) | ~6 MiB |
| `disk/snapshots/chunk_XXXXX.npz` | 24 S codes + 24 conv codes + M1 + M2 codes (DELTA for S/M1/M2 vs the system state; ABSOLUTE for conv) | ~6 MiB each |
| `disk/snapshots/ingest_manifest.json` | the restart manifest (completed chunk ids) | — |
| `disk/pretrained_luts/` (+ `manifest.json`) | the fine-tuned LUT artifacts | ~6.35 GiB |

The retrieval vector (dims 24 × 524,288 + 2 × 524,288 = 13,631,488) is
dequantized from the codes on demand for index building and rerank —
never stored. Index constants (`IndexConfig` defaults): `d = 13,631,488`,
`nlist = 224`, `m = 64`, `nbits = 8`, `nprobe = 8`, `preselect_k = 100`,
`rerank_k = 3`. Per chunk: 24 S codes 5.4 MiB + 24 conv codes 0.34 MiB +
M1 0.22 MiB + M2 0.22 MiB ≈ 6.0 MiB (4.6× vs the 27.5 MiB fp16 state);
50,000 chunks ≈ 300 GiB.

## 4. Entry points

| entry point | module | signature (essentials) |
|---|---|---|
| `load_quant_model` | `src/scripts/loader.py` | `(artifacts_dir, model_name, device, residual=False, dtype=None, forward="kernel", awq_compensation=True, heads_dir=None) -> (model, metadata)` |
| `ingest_chunk` | `src/rag/ingest.py` | `(model, chunk_token_ids, cache, system)` — one chunk, online TurboQuant writes |
| `IngestDriver` | `src/rag/ingest.py` | `(model, system_token_ids, chunks, out_dir)` — restartable batch; writes the npz set + manifest |
| `build_index` / `preselect` / `rerank` | `src/rag/index.py` | `(vectors_iter, config=IndexConfig())` / `(index, query_vector, k=100)` / `(loader, query_vector, candidate_ids, k=3)` |
| `answer_query` | `src/rag/query.py` | `(model, query_token_ids, system, index=None, loader=None, preselect_k=100, rerank_k=3, max_new_tokens=0)` — the §6 eight-step flow |
| `sum_turboquant_codes` / `install_from_disk` | `src/rag/install.py` | the §6 install: dequant, sum, requant once — onto the system codes |
| `evals.py` subcommands | `src/rag/evals.py` | `roundtrip` (P1 MSE+concentration), `streaming` (P2 token-match), `margin` (P4 cos margin), `recall` (P5 recall@100+top-3), `e2e` (P6 no-RAG vs actual vs oracle), `ledger` (§9 timing + §10 VRAM) |
