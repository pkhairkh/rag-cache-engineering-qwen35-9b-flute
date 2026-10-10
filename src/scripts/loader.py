#!/usr/bin/env python3
"""loader.py — the one quantized-model loader.

Loads the pre-built palettized (idxN LUT, W4+r32) Qwen3.5-9B from its
artifacts dir, optionally with separate heads artifacts. The artifacts
are PROVIDED — neither the palettizer nor the parent project's
training stack is part of this repo.

Module contract: STDLIB ONLY at import time — torch, transformers and
palettized_modules load lazily inside the function body.
"""
from __future__ import annotations

__all__ = ["load_quant_model"]


def load_quant_model(artifacts_dir, model_name, device,
                     residual=False, dtype=None, forward="kernel",
                     awq_compensation=True, heads_dir=None,
                     use_m1m2=True, m1m2_mem_size=128,
                     m1m2_gates_path=None):
    """The quant loader. Returns (model, metadata).

    forward: "kernel" (the box default — the FLUTE fused path) or
    "reference" (the torch dequant path — the only CPU-legal route;
    the frozen-path resolver keeps off-CUDA runs on reference
    numerics, so a harness that must run on CPU passes this).

    awq_compensation (W13): serve the legacy rotate-then-AWQ fold
    with the exact compensated rotation M = D T D^-1 (s recovered
    from norm_gain_edits.json on the pristine model).

    heads_dir: optional separate heads artifacts dir. When set, loads
    transformer layers from artifacts_dir and embed_tokens/lm_head from
    heads_dir (the separate head-pass output with higher-quality LUT).

    use_m1m2 (W17): attach the SPECIFICATION §2.2 global memories by
    instantiating through the VENDORED Qwen3.5 classes (the native
    AutoModelForCausalLM class has no wiring — the flag was a silent
    no-op before W17). Zero-init gates (P3) leave the model's outputs
    bit-unchanged until the §7 fine-tune opens them.

    m1m2_mem_size: the memories' slot count (spec §2.2 default 128 =
    the canonical 524,288-element unit; 1024/4096/8192 are the
    retrieval-capacity experiments — the §4 vector and the snapshot
    grow accordingly; INGEST and QUERY must agree on it, the loaders
    fail loudly on drift).

    m1m2_gates_path: optional .npz of trained gates (§7 fine-tune,
    rag/m1m2_finetune.py::save_gates) — loaded with geometry
    validation; without it the gates stay at the P3 zero/one init.
    """
    if dtype is None:
        import torch
        dtype = torch.float16
    import palettized_modules as pmod
    return pmod.load_palettized_model(
        artifacts_dir, model_name, device=device, dtype=dtype,
        residual=residual, reference=(forward == "reference"),
        awq_compensation=awq_compensation, heads_dir=heads_dir,
        use_m1m2=use_m1m2, m1m2_mem_size=m1m2_mem_size,
        m1m2_gates_path=m1m2_gates_path)
