#!/usr/bin/env python3
"""eval_common.py — the shared eval-plane helpers.

PROPOSAL.md §4.1 rule 4 (one loader, one git_head, one cosine measure,
one eval scaffold, one atomic-JSON helper) made concrete: every helper
that the eval scripts previously copied per-file lives here once.

Module contract: STDLIB ONLY at import time. Importing this module must
not import torch, transformers, or any scripts/ peer — the verification
orchestrator imports it while staying torch-free, and the engine-plane
modules (capture, spectrum, report, trainer) import it for the atomic
writer. The model scaffold functions import their heavy dependencies
lazily inside the function body.
"""
from __future__ import annotations

import json
import os
import subprocess

__all__ = [
    "atomic_json_dump",
    "cosine_of_activations",
    "git_head",
    "load_dense_fp16",
    "load_quant_model",
    "paired_mean",
    "release_model_memory",
]

_HERE = os.path.dirname(os.path.abspath(__file__))


def git_head() -> str:
    """HEAD sha of this repository, "unknown" when git is unavailable.

    The provenance field of every eval report comes from here; the
    working directory is the repository root (the parent of scripts/).
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=os.path.join(_HERE, "..")).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def atomic_json_dump(obj, path: str) -> None:
    """Write obj as JSON to path atomically (tmp file + os.replace).

    A reader never observes a half-written report and a crash never
    truncates one. Layout matches the repo convention: indent=2, no
    trailing newline.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def paired_mean(a, b) -> float:
    """Mean of the paired per-doc deltas d_i = b_i - a_i (negative = b
    better) — the O-1 paired probe's mean_delta semantics (PROPOSAL
    1.1's metric of record). Pure python on parsed JSON numbers; the
    caller owns the length validation (the paired noise floor only
    holds doc-for-doc)."""
    n = len(a)
    if not n:
        return 0.0
    return sum(bi - ai for ai, bi in zip(a, b)) / n


def cosine_of_activations(a, b) -> float:
    """Cosine of two activation tensors, each flattened to one vector.

    The eval plane's single cosine measure: both inputs fp32, shapes
    broadcast-compatible after flatten, returns a python float. Equals
    F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item().
    Chunked accumulations that compute the same quantity without ever
    materializing their operands are separate memory contracts, not
    copies of this function.
    """
    import torch.nn.functional as F
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def load_dense_fp16(model_name, device, dtype=None):
    """The eval scaffold's dense reference model.

    from_pretrained (trust_remote_code, low_cpu_mem_usage) at fp16 by
    default, moved to device, eval mode. The caller owns the release:
    delete its reference, then call release_model_memory().
    """
    import torch
    from transformers import AutoModelForCausalLM
    if dtype is None:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True).to(device)
    model.eval()
    return model


def release_model_memory() -> None:
    """The scaffold's release tail: full collection, then the CUDA cache.

    Call AFTER deleting the model names in the caller's scope — this
    function cannot drop the caller's references. CPU-safe:
    torch.cuda.empty_cache() is a no-op without a CUDA context.
    """
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def load_quant_model(artifacts_dir, model_name, device,
                     qlora_adapters=None, residual=False, dtype=None,
                     forward="kernel", awq_compensation=True,
                     heads_dir=None):
    """The eval plane's one quant loader. Returns (model, metadata).

    With qlora_adapters: the palettized model with the trained QLoRA
    adapters attached (kernel eval path, per the saved adapter
    config's geometry). Without: the plain palettized idx4 model.

    forward: "kernel" (the box default — the FLUTE fused path) or
    "reference" (the torch dequant path — the only CPU-legal route;
    the frozen-path resolver keeps off-CUDA runs on reference
    numerics, so an eval harness that must run on CPU passes this).

    awq_compensation (W13): serve the legacy rotate-then-AWQ fold
    with the exact compensated rotation M = D T D^-1 (s recovered
    from norm_gain_edits.json on the pristine model). The
    differential debugging arm passes False to reproduce the
    pre-W13 broken composition deliberately.

    heads_dir: optional separate heads artifacts dir. When set, loads
    transformer layers from artifacts_dir and embed_tokens/lm_head from
    heads_dir (the separate head-pass output with higher-quality LUT).
    """
    if dtype is None:
        import torch
        dtype = torch.float16
    if qlora_adapters:
        import qlora
        model, metadata = qlora.load_qlora_model(
            artifacts_dir, adapters_dir=qlora_adapters,
            model_name=model_name, device=device, dtype=dtype,
            residual=residual, eval_kernel=(forward == "kernel"))
        if forward == "reference":
            # every PalettizedLinear, wrapped or bare (rank-0 modules
            # stay unwrapped — their reference flag is the only lever)
            import palettized_modules as pmod
            for _, mod in pmod.iter_palettized_linears(model):
                mod.reference = True
        return model, metadata
    import palettized_modules as pmod
    return pmod.load_palettized_model(
        artifacts_dir, model_name, device=device, dtype=dtype,
        residual=residual, reference=(forward == "reference"),
        awq_compensation=awq_compensation, heads_dir=heads_dir)
