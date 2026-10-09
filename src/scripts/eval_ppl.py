#!/usr/bin/env python3
"""
eval_ppl.py — WikiText-2 perplexity evaluation.

Protocol (GPTQ evaluation convention):
  * WikiText-2 raw-v1 **test** split — disjoint from the **train** split
    used for calibration (CALIB_SPLIT/EVAL_SPLIT in calibrate_real_text.py
    are the single source of truth; the calibration suite checks
    the disjointness);
  * concatenate the corpus, tokenize, cut into non-overlapping 2048-token
    windows;
  * PPL = exp(total NLL / total tokens) over all windows.

Evaluates the dense FP16 model and the palettized idx4 model (optionally
with the whitened-SVD residual branch). Output: reports/ppl_<stage>.json
(written by this script only).

W22 (the chunked-CE round): evaluate_nll no longer routes labels
through the model. transformers' ForCausalLMLoss upcasts the FULL
(B, T-1, V) logits to fp32 and cross_entropy then materializes the
log-softmax twin — at the box geometry (T=2048, V=248320,
MODEL_GEOMETRY.md) that is 2 x 1.89 GiB of fp32 temps per window
(2 x 7.57 GiB at batch 4) ON TOP of the resident weights, which on
the 22.06 GiB A10G OOM'd the dense arm at its very first window
(the observed ask, 2034237440 bytes, is exactly 2048 x 248320 x 4).
evaluate_nll now scores the window in 256-row chunks (<= 254 MiB of
fp32 live at once) with the IDENTICAL arithmetic, and halves its
working batch on a CUDA OOM instead of crashing.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# W22: set BEFORE the first CUDA allocation (torch reads it when the
# caching allocator initializes). Expandable segments let the big asks
# (the dense reload, the full-vocab PPL logits) find contiguous room
# after a release instead of fragmenting; setdefault so an explicit
# user setting always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True")

import torch
import torch.nn.functional as F

# W22: rows of the shifted window scored per cross_entropy call. 256
# rows x 248320 vocab x 4 B = 254 MiB of fp32 live per chunk — vs the
# 2 x 1.89 GiB full-window fp32 twins of the labels path (see the
# module docstring). Small enough that the chunk itself is never the
# OOM constraint, large enough that the per-chunk syncs are noise.
_CE_CHUNK_ROWS = 256

# torch.OutOfMemoryError exists from 2.1 on (a RuntimeError subclass);
# the fallback keeps older torch importable, it just loses the halving.
_OOM_ERROR = (getattr(torch, "OutOfMemoryError", None)
              or getattr(torch.cuda, "OutOfMemoryError", RuntimeError))

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from calibrate_real_text import EVAL_SPLIT  # noqa: E402
from eval_common import (  # noqa: E402
    git_head,
    load_dense_fp16,
    load_quant_model,
    release_model_memory,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--artifacts-dir", required=True,
                   help="the palettized artifacts dir (usage example: "
                        "/home/ubuntu/qwen3_5_9b_palettized)")
    p.add_argument("--residual", action="store_true",
                   help="attach the whitened-SVD residual branch when the "
                        "artifacts carry one (W4 configuration)")
    p.add_argument("--qlora-adapters", default=None,
                   help="QLoRA adapters directory")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="evaluation window length (GPTQ convention: 2048)")
    p.add_argument("--max-windows", type=int, default=0,
                   help="cap on evaluation windows (0 = all)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--forward", choices=["kernel", "reference"],
                   default="kernel",
                   help="quant forward route: kernel = FLUTE fused path "
                        "(the box default; requires CUDA) or reference "
                        "= torch dequant path (the CPU-legal route)")
    p.add_argument("--stage", default="gptq",
                   help="stage tag for the output file name")
    p.add_argument("--skip-dense", action="store_true",
                   help="evaluate only the palettized model (dense numbers "
                        "already committed from a previous run)")
    p.add_argument("--output", default=None,
                   help="output path (default reports/ppl_<stage>.json)")
    return p.parse_args()


def load_eval_windows(tokenizer, seq_len, max_windows):
    """Tokenized non-overlapping windows of the WikiText-2 test split."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split=EVAL_SPLIT["wikitext2"])
    blob = "\n".join(t for t in ds["text"] if len(t.strip()) > 0)
    ids = tokenizer(blob, add_special_tokens=False)["input_ids"]
    n_windows = len(ids) // seq_len
    if max_windows:
        n_windows = min(n_windows, max_windows)
    windows = torch.tensor(
        [ids[i * seq_len:(i + 1) * seq_len] for i in range(n_windows)],
        dtype=torch.long)
    return windows, len(ids)


def _window_nll(model, batch):
    """One window's (nll_sum, n_valid) over the SHIFTED positions — the
    ForCausalLMLoss arithmetic (logits[:, :-1] vs labels[:, 1:], fp32
    cross entropy) computed WITHOUT the full-window fp32 upcast (W22:
    the labels route OOM'd the dense arm; see the module docstring).
    The per-chunk sums accumulate in Python float (float64), so the
    window total is at least as exact as the single-kernel fp32 sum.
    """
    out = model(batch)
    logits = out.logits if hasattr(out, "logits") else out
    shift_logits = logits[:, :-1, :]
    shift_labels = batch[:, 1:]
    vocab = shift_logits.shape[-1]
    nll_sum = 0.0
    for j in range(0, shift_logits.shape[1], _CE_CHUNK_ROWS):
        rows = shift_logits[:, j:j + _CE_CHUNK_ROWS]
        # fp32 upcast of the CHUNK only: 256 x V x 4 B, not (T-1) x V x 4 B
        chunk = rows.reshape(-1, vocab).float()
        tgt = shift_labels[:, j:j + _CE_CHUNK_ROWS].reshape(-1)
        nll_sum += float(F.cross_entropy(chunk, tgt,
                                         reduction="sum").item())
    del shift_logits, logits, out
    return nll_sum, shift_labels.numel()


@torch.no_grad()
def evaluate_nll(model, windows, device, batch_size=4):
    """Returns (total_nll, total_tokens) over all windows.

    W22: labels are no longer passed to the model (the full-window
    fp32 upcast + log-softmax twin OOM'd the dense arm next to its
    ~18.1 GiB resident — 2034237440 B = 2048 x 248320 x 4, twice).
    The accounting is the pre-W22 one VERBATIM so every number stays
    comparable: HF's loss is the mean over the B x (T-1) shifted
    positions and this loop re-inflates it by batch.numel() (B x T),
    exactly as `out.loss.item() * n_tok` did. A CUDA OOM halves the
    working batch (floor 1) and retries the window; at the floor the
    error re-raises — the crash stays faithful.
    """
    total_nll, total_tok = 0.0, 0
    b = int(batch_size)
    i = 0
    while i < windows.shape[0]:
        batch = windows[i:i + b].to(device)
        try:
            nll_sum, n_valid = _window_nll(model, batch)
        except _OOM_ERROR:
            if b <= 1:
                raise
            torch.cuda.empty_cache()   # CPU-safe no-op without a context
            b = max(1, b // 2)
            print(f"  [PPL] window {i + 1}: OOM — retrying at batch {b}",
                  flush=True)
            continue
        # the pre-W22 accounting, verbatim: mean over the shifted
        # positions, re-inflated by the window's own token count
        total_nll += (nll_sum / n_valid) * batch.numel()
        total_tok += batch.numel()
        if (i // max(1, int(batch_size))) % 10 == 0:
            print(f"    {i + batch.shape[0]}/{windows.shape[0]} windows",
                  flush=True)
        i += b
    return total_nll, total_tok


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    windows, n_corpus_tokens = load_eval_windows(
        tokenizer, args.seq_len, args.max_windows)
    print(f"eval split: wikitext2 {EVAL_SPLIT['wikitext2']} — "
          f"{windows.shape[0]} windows x {args.seq_len} tokens "
          f"(corpus {n_corpus_tokens} tokens)", flush=True)

    results = {"dense_fp16": None, "palettized_idx4": None}

    if not args.skip_dense:
        print("[1/2] Dense FP16 reference", flush=True)
        dense = load_dense_fp16(args.model, args.device)
        nll, tok = evaluate_nll(dense, windows, args.device)
        results["dense_fp16"] = {"total_nll": nll, "n_tokens": tok,
                                 "ppl": float(torch.exp(torch.tensor(nll / tok)))}
        print(f"  dense PPL = {results['dense_fp16']['ppl']:.4f}", flush=True)
        del dense
        release_model_memory()

    print(f"[2/2] Palettized idx4 model"
          f"{' + residual' if args.residual else ''}"
          f"{' + qlora(' + args.qlora_adapters + ')' if args.qlora_adapters else ''}",
          flush=True)
    quant, metadata = load_quant_model(
        args.artifacts_dir, args.model, args.device,
        qlora_adapters=args.qlora_adapters, residual=args.residual,
        dtype=torch.float16, forward=args.forward)
    nll_q, tok_q = evaluate_nll(quant, windows, args.device)
    results["palettized_idx4"] = {
        "total_nll": nll_q, "n_tokens": tok_q,
        "ppl": float(torch.exp(torch.tensor(nll_q / tok_q))),
        "residual": bool(args.residual),
        "qlora_adapters": args.qlora_adapters,
    }
    print(f"  palettized PPL = {results['palettized_idx4']['ppl']:.4f}",
          flush=True)
    del quant
    release_model_memory()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "model": args.model,
        "artifacts_dir": args.artifacts_dir,
        "stage": args.stage,
        "eval_split": f"wikitext2/{EVAL_SPLIT['wikitext2']}",
        "calibration_split": "wikitext2/train (disjoint; see calibrate_real_text)",
        "seq_len": args.seq_len,
        "n_windows": int(windows.shape[0]),
        "n_eval_tokens": int(windows.numel()),
        "results": results,
        "delta_vs_fp16": (
            {"ppl": results["palettized_idx4"]["ppl"]
             - results["dense_fp16"]["ppl"]}
            if results["dense_fp16"] else None),
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "none"),
        },
    }

    out = args.output or f"reports/ppl_{args.stage}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
