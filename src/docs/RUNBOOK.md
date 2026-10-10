# Runbook — the box workflow

Purpose: the exact command sequence for a session on this repo, from a fresh clone to the gates — CPU box (suite + self-tests) and GPU box (kernel build, model load, the RAG pipeline).
Authority: authoritative for operations; subordinate to `SPECIFICATION.md` for RAG semantics.
Status: synced from the main project @ ab78893, scoped to this repo Wv2-8 — the palettization/probe/verdict/training operations are main project (one scoping-table row each, §6); this repo's operations are §0–§4.
Workstation (GPU box): one AWS g5.xlarge (A10G, 24 GB), Ubuntu,
CUDA 12/13, a venv.

## 0. First session on a box

```bash
# environment (once)
python3 -m pip install -r src/requirements.txt     # CPU box: src/requirements-cpu.lock.txt
python3 -m pip install ninja

# kernel extension (GPU box; CPU boxes skip — pure-torch fallbacks carry the suite)
cd src/flute_extended && python setup.py build_ext --inplace && cd ../..
```

Build verification ladder (new hardware or after a compiler change):
[BUILD.md](BUILD.md) §4-5.

## 1. CPU box: the suite + the self-tests

```bash
python3 -m pytest src/rag/tests -q          # 169 tests
python3 src/rag/evals.py <roundtrip|streaming|margin|recall|e2e|ledger> --self-test
```

Without `--self-test` the four real-data commands refuse loudly (exit
nonzero) — GPU box only. Self-test artifacts land under `evals_out/`
(git-ignored).

## 2. GPU box: kernel gates + model load

```bash
cd src/flute_extended
python test_flute.py --backend debug_simple   # ladder 5.1 (BUILD.md §5)
python test_flute.py                          # ladder 5.2
python test_qwen_weights.py                   # ladder 5.3 — needs the artifact set
cd ../..

PYTHONPATH=src/scripts python3 -c "
from loader import load_quant_model
model = load_quant_model('<artifacts_dir>', 'Qwen/Qwen3.5-9B',
                          device='cuda', forward='kernel')"   # 'reference' = CPU-legal
```

A/B switches for attribution — set per run, no rebuild
(`FLUTE_NO_MERGE`, `FLUTE_NO_FHT_FUSE`, `FLUTE_NO_SPLITK`,
`FLUTE_NO_WIDE_PREF`; [ROUTING.md](ROUTING.md) §2). The probe runs that
exercise them per-module are main project (§6).

## 3. GPU box: the RAG pipeline

Module map, run matrix, disk layout, and entry-point signatures:
[RAG_PIPELINE.md](RAG_PIPELINE.md). The flow: load the model
(`load_quant_model`) → ingest the corpus (`src/rag/ingest.py::IngestDriver`
→ `disk/snapshots/chunk_XXXXX.npz` + `system_state.npz` +
`ingest_manifest.json`) → build the index (`src/rag/index.py::build_index`
→ `disk/ivfadc_cache.index`) → answer queries
(`src/rag/query.py::answer_query`) → the evals gates
(`src/rag/evals.py`, real-data mode).

## 4. GPU box: deep measurement

```bash
sudo bash src/flute_extended/tools/lock_clocks.sh 1530 <maxmem>   # reproducible A/B
bash src/flute_extended/tools/ncu_profile.sh                     # the ncu success gates
```

The measurement kit around them (dmon clocks, nsys capture, the FHT A/B,
per-module ncu follow-ups) is main project — §6; the bands and gates it
reports: [PERFORMANCE.md](PERFORMANCE.md) §6.

## 5. Operational warnings (what is noise, what is not)

| Message | Meaning | Action |
|---|---|---|
| `metadata group_size ... inconsistent with the LUT geometry` + `[GS] ... 132 tensor component(s)` | the stale metadata field; the loader trusts the LUT geometry | none (writer-side fix is main project) |
| unauthenticated HF Hub requests | no `HF_TOKEN` | `export HF_TOKEN=...` (config fetches only) |
| CUDA 13.x vs torch 13.0 minor-mismatch warning | toolkit/torch skew | none in practice |
| `[flute_extended] CUTLASS: NOT FOUND` | no CUTLASS checkout | only needed for the dense baseline |
| faiss train-size notes at small corpus scale | `nlist` vs vector count in self-tests | none (documented; scales away at production corpus size) |
| expandable-segments OOM recovery warnings | allocator headroom ran out mid-run; the ladder recovers by halving the batch (main project: the PPL ladder) | none unless it aborts; then lower the batch |

## 6. Main-project operations (scoping table)

| operation | commands (main project — not part of this repo) |
|---|---|
| session settling | main project: `scripts/doctor.py` (imports, path table, numerics triple-check, attention parity) |
| environment provision | main project: `scripts/provision_env.sh` / `scripts/provision_env_cpu.sh` |
| palettization runs | main project: `scripts/palettize_qwen3_5_9b.py` (body + `--only-heads` passes; recipe/knob reference [PALETTIZATION.md](PALETTIZATION.md)) |
| routing + bandwidth census | main project: `scripts/probe_decode_routing.py` (reading the table: [ROUTING.md](ROUTING.md) §3) |
| end-to-end verdict | main project: `scripts/eval_greedy_match.py` (protocol: [EVALUATION.md](EVALUATION.md) §2; expected numbers: [PERFORMANCE.md](PERFORMANCE.md) §1) |
| deep measurement kit | main project: `scripts/measure_decode.sh`, `scripts/measure_energy.py` |
| box numerics gate | main project: `scripts/verify_gemv.py` |
| GPU-contract bans | main project: `scripts/check_gpu_contract.py` + `gpu_contract_allowlist.txt` |
| training stack | main project: `scripts/train_two_phase.sh`, `scripts/trainer.py`, `scripts/data.py`, `scripts/loss.py`, `scripts/muon_optimizer.py`, `scripts/qlora.py` / `qlora_gemm.py` / `qlora_merge.py` / `qlora_fallback.py` (the SPECIFICATION §7 fine-tune HERE is `src/rag/finetune.py`, served via `src/rag/lut_export.py`) |
| analysis plane | main project: `scripts/spectrum.py`, `sensitivity_rank.py`, `distill_rank_alloc.py`, `distill_eval.py`, `report.py`, `o1_baseline_check.py`, `vram_ledger.py`, `diagnose_greedy_bug.py` |
| toy/differential drivers | main project: `scripts/toy_common.py`, `toy_e5_ladder.py`, `toy_e7_recipes.py`, `toy_e8_lut2bit.py`, `toy_real_geometry.py`, `lutgrad_sim.py` (this repo's validated toys: `scripts/poc_toy/`) |
| PPL/eval noise rows | main project: the `Token indices sequence length ... > 262144` corpus note (windowed), the PPL-ladder OOM warnings |
