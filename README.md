# rag-cache-engineering-qwen35-9b-flute

Cache-engineered RAG on Qwen3.5-9B (FLUTE idxN W4+r32): the LUT model's
cache IS the retrieval vector. No separate embedder, no chunk text on
disk, no re-prefill, TurboQuant online (3.5-bit, quality-neutral).

**The contract: SPECIFICATION.md.** Read it first — it defines the model
geometry (24 linear-attention + 8 full-attention layers), the snapshotted
caches (per-layer S + global M1/M2 + conv_state), the online TurboQuant
cache API, the retrieval vector, the ingestion / query flows, and the
fine-tune that makes the caches carry information.

## Layout

```
SPECIFICATION.md            the definitive spec (the contract)
scripts/poc_toy/            the validated CPU toys (§12) + their deps
src/
  scripts/                  model-side code: modeling (Qwen3.5-9B hybrid),
                            loader (eval_common), W10 QLoRA training stack
                            (qlora, qlora_gemm, trainer, capture, loss,
                            muon_optimizer), palettized_modules, attn_sm86
  flute_extended/           the W4/W10 inference CUDA extension
                            (cutlass_streaming + debug_simple + FHT kernels)
  flute_train_kernels/      the W10 training CUDA extension
                            (backward GEMM + LUT-grad)
  docs/                     format + build contracts (DEQUANT_SPEC,
                            KERNEL_SPEC_DLDLUT, QUANTIZATION_FORMAT, FHT,
                            MODEL_GEOMETRY, TRAINING_LIMITATION, DEPLOY)
  requirements*.txt         minimal deps + the known-good pin set
```

## Scope notes (post-cleanup)

- The palettized model + heads artifacts are provided pre-built and
  loaded via `src/scripts/eval_common.py::load_quant_model`; the
  canonical palettizer is intentionally NOT part of this repo.
- Fine-tuned weights are served through the QLoRA adapter channel
  (`qlora_adapters.pt` + `qlora_config.json`), not re-palettization.
- Kernel bring-up on the GPU box: `src/docs/DEPLOY.md` (build, ptxas
  gate, differential spot-check).
- The RAG pipeline itself (TurboQuant online cache wrapper, M1/M2
  architectural addition, 9-hook capture, ingestion -> IVFADC ->
  install) is specified but NOT yet implemented — SPECIFICATION.md
  sections 2-9 are the build list.
