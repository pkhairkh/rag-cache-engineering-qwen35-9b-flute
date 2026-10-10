"""_bootstrap.py — shared path setup + model load for the GPU tools.

Every script under scripts/gpu/ does:

    from _bootstrap import boot, load_model, DEFAULTS

`boot()` anchors src/rag, src/scripts and src/flute_extended on sys.path
(relative to this file — the tools run from anywhere); `load_model()` wraps
src/scripts/loader.load_quant_model with the box's default artifact paths
(all overridable per script).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def boot() -> str:
    """Anchor the three source roots; returns the repo root."""
    for p in ("src/rag", "src/scripts", "src/flute_extended"):
        sys.path.insert(0, os.path.abspath(os.path.join(_REPO, p)))
    return _REPO


# the GPU-box defaults (the W10/W11 session geometry)
DEFAULTS = {
    "artifacts_dir": "/home/ubuntu/qwen3_5_9B_palettized",
    "heads_dir": "/home/ubuntu/qwen3_5_9B_palettized_heads",
    "model_name": "Qwen/Qwen3.5-9B",
    "disk_dir": "/home/ubuntu/RAGGA/disk/ingested_50k",
    "corpus": "/home/ubuntu/enterprise_rag_bench/documents/documents.jsonl",
    "questions": "/home/ubuntu/enterprise_rag_bench/questions/questions.jsonl",
    "device": "cuda",
}


def load_model(artifacts_dir=None, heads_dir=None, model_name=None,
               device=None):
    """load_quant_model with the box defaults (loader.py's entry point)."""
    from loader import load_quant_model
    model, meta = load_quant_model(
        artifacts_dir=artifacts_dir or DEFAULTS["artifacts_dir"],
        model_name=model_name or DEFAULTS["model_name"],
        device=device or DEFAULTS["device"],
        heads_dir=heads_dir or DEFAULTS["heads_dir"])
    model = model.eval()
    return model, meta
