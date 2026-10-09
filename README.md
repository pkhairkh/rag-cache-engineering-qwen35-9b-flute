# rag-cache-engineering-qwen35-9b-flute

Cache-engineered RAG on Qwen3.5-9B (FLUTE idxN W4+r32): the LUT model's
cache IS the retrieval vector. No separate embedder, no chunk text on
disk, no re-prefill, TurboQuant online (3.5-bit, quality-neutral).

**The contract: SPECIFICATION.md.** Read it first — it defines the model
geometry (24 linear-attention + 8 full-attention layers), the snapshotted
caches (per-layer S + global M1/M2 + conv_state), the online TurboQuant
cache API, the retrieval vector, the ingestion / query flows, and the
fine-tune that makes the caches carry information. **PROPOSAL.md** is the
build plan (how): the TurboQuant paper (arXiv:2504.19874) mapped onto our
tensors, five design decisions, phased build with gates, risk register.

## Layout

```
SPECIFICATION.md            the definitive spec (the contract)
scripts/poc_toy/            the validated CPU toys (§12) + their deps
src/
  scripts/                  model-side code: modeling (Qwen3.5-9B hybrid),
                            loader (the pre-built-model loader),
                            palettized_modules (W4 LUT forward),
                            attn_sm86 (SM86 attention kernel vehicle)
  flute_extended/           the W4 inference CUDA extension
                            (cutlass_streaming + debug_simple + FHT kernels)
  docs/                     format + build contracts (DEQUANT_SPEC,
                            QUANTIZATION_FORMAT, FHT, MODEL_GEOMETRY,
                            HARDWARE, DEPLOY)
  requirements*.txt         minimal deps + the known-good pin set
```

## Scope notes (post-cleanup)

- The palettized model + heads artifacts are provided pre-built and
  loaded via `src/scripts/loader.py::load_quant_model`; the palettizer
  and the eval plane are intentionally NOT part of this repo.
- No QLoRA: the parent project's training stack (qlora, qlora_gemm,
  qlora_fallback, trainer, capture/loss/muon_optimizer,
  flute_train_kernels) is not part of this repo. The §7 fine-tune is
  written on the GPU box as part of the RAG build — the straight-through
  LUT primitive is `PalettizedLinear.make_trainable()` via the reference
  path; fine-tuned LUTs are served as full LUT artifacts
  (`pretrained_luts/`, SPECIFICATION.md §11).
- Kernel bring-up on the GPU box: `src/docs/DEPLOY.md` (build, ptxas
  gate, differential spot-check).
- The RAG pipeline itself (TurboQuant online cache wrapper, M1/M2
  architectural addition, 9-hook capture, ingestion -> IVFADC ->
  install) is specified but NOT yet implemented — SPECIFICATION.md
  sections 2-9 are the build list.
