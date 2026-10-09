#!/usr/bin/env python3
"""
o1_baseline_check.py — O-1 paired same-document probe.

Separates the two confounded quantities behind the FineWeb-Edu
"oscillation without progress" investigation
(reports/training_oscillation_analysis.md):

  * the STATIC quantization gap   (dense fp16 - palettized idx4)
  * the TRAINING movement         (adapter@N - adapter@0)

Both are measured as paired per-document deltas on the exact same
documents, in the exact same order, so the noise floor is the std of the
DELTAS — not the std of document difficulty (~0.5 nats). This detects
changes far below what two independent eval means can resolve.

Stages (one document list for all of them):
  [dense]   (optional, --dense) dense FP16 HF model        — reference
  [base+0]  palettized idx4 + freshly attached QLoRA (B=0) — by the
            B-zero-init contract this is bit-identical to the palettized
            base; it doubles as the canary that the attach path is a
            faithful pass-through
  [base+N]  (optional, --adapters-dir) same model with the trained
            adapter loaded — what training actually changed. The fresh
            attach uses the adapter dir's saved qlora_config.json
            geometry (r/alpha/dropout/scope/init/rank_map/alpha_mode,
            mirroring qlora.load_qlora_model); CLI --r/--alpha are
            ignored (a note is printed if they were set)

Document replication contract: the probe rebuilds the trainer's eval
holdout when --dataset / --max-samples / --seq-len / --eval-holdout /
--seed match the training invocation. Defaults mirror the run recorded
recorded in reports/o1_baseline_fineweb.json (FineWeb-Edu, 10k
samples, seq 256, holdout 64, seed 0).
If the probed training used a larger holdout, the probe's docs are the
first N of the same permutation (a subset — still exactly comparable).

Usage:
  python scripts/o1_baseline_check.py \
      --artifacts-dir /home/ubuntu/qwen3_5_9b_palettized \
      --dense \
      --output reports/o1_baseline_fineweb.json
  # after (re-)running training with an output checkpoint:
  python scripts/o1_baseline_check.py \
      --artifacts-dir /home/ubuntu/qwen3_5_9b_palettized \
      --adapters-dir <output-dir of the run> \
      --dense \
      --output reports/o1_baseline_fineweb.json

Interpretation guide:
  quantization_gap  mean 0.2-0.4 nats  = expected frozen 4-bit/AWQ error
  training_movement |t| < 2             = below the paired noise floor
                                      t <= -2 = real improvement (nats < 0)
                                      t >= +2 = the run made the model worse
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from datetime import datetime, timezone

# Allocator default (expandable_segments) — applied before torch import
# so it is honored at CUDA initialization; an explicit environment
# override wins.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import palettized_modules as pmod
import qlora
from data import load_sft_dataset, _report_status, DATASET_DEFAULTS
from eval_common import git_head, load_dense_fp16, release_model_memory


# attach-parameter defaults (the --r/--alpha argparse defaults; kept in
# sync so the ignored-flags note below only fires on explicit overrides)
_R_DEFAULT = 64
_ALPHA_DEFAULT = 16

# attach geometry that must round-trip from a saved qlora_config.json
# when --adapters-dir is given (mirrors qlora.load_qlora_model's
# re-attach; cfg.tensors is regenerated per attach, so not compared)
_GEOMETRY_FIELDS = ("r", "alpha", "dropout", "scope",
                    "include_residual_branch", "init_a", "init_b",
                    "rank_map", "alpha_mode")


def _attach_geometry(args, saved_cfg):
    """attach_qlora kwargs for the fresh (B=0) attach.

    With --adapters-dir: the saved config's geometry (the same fields
    qlora.load_qlora_model re-attaches with) so the probe measures the
    run that was actually trained — a CLI-guessed r/alpha would silently
    mismatch the adapter shapes/scales. Without it: the CLI flags,
    exactly as before. A non-default --r/--alpha next to --adapters-dir
    is overridden — the printed note is the loud (not silent) outcome."""
    if saved_cfg is None:
        return dict(r=args.r, alpha=args.alpha, dropout=args.dropout,
                    scope=args.scope,
                    include_residual_branch=args.residual,
                    init_a=args.init_a, init_b="zero",
                    base_model=args.model,
                    artifacts_dir=args.artifacts_dir)
    if args.r != _R_DEFAULT or args.alpha != _ALPHA_DEFAULT:
        print(f"  [adapter] note: --adapters-dir given — CLI --r/--alpha "
              f"({args.r}/{args.alpha}) ignored; using the saved config's "
              f"geometry (r={saved_cfg.r}, alpha={saved_cfg.alpha}, "
              f"rank_map={'on' if saved_cfg.rank_map else 'none'}, "
              f"alpha_mode={saved_cfg.alpha_mode})", flush=True)
    return dict(r=saved_cfg.r, alpha=saved_cfg.alpha,
                dropout=saved_cfg.dropout, scope=saved_cfg.scope,
                include_residual_branch=saved_cfg.include_residual_branch,
                init_a=saved_cfg.init_a, init_b=saved_cfg.init_b,
                base_model=args.model, artifacts_dir=args.artifacts_dir,
                rank_map=saved_cfg.rank_map,
                alpha_mode=saved_cfg.alpha_mode)


def _check_attach_geometry(saved_cfg, attach_cfg):
    """Loud guard: the fresh attach must have used the saved adapter
    geometry — the strict adapter load only catches shape-level drift,
    and a same-shape wrong-scale geometry would silently corrupt the
    paired movement stage."""
    for field in _GEOMETRY_FIELDS:
        got, want = getattr(attach_cfg, field), getattr(saved_cfg, field)
        if got != want:
            raise RuntimeError(
                f"o1_baseline_check: attach used {field}={got!r} but the "
                f"adapter dir's qlora_config.json records {want!r} — "
                f"refusing to silently probe a mismatched geometry")


def parse_args():
    p = argparse.ArgumentParser(
        description="O-1 paired same-doc probe: quantization gap vs training movement")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--artifacts-dir", required=True,
                   help="idx4 artifacts directory (same one training used)")
    p.add_argument("--adapters-dir", default=None,
                   help="trained QLoRA checkpoint dir (qlora_adapters.pt + "
                        "qlora_config.json); omit to skip the movement "
                        "stage. When given, the saved config's attach "
                        "geometry (r/alpha/rank_map/...) is used — CLI "
                        "--r/--alpha are ignored")
    p.add_argument("--dense", action="store_true",
                   help="also score the dense FP16 reference (needs ~18 GiB "
                        "more VRAM; run once)")
    # document replication: defaults mirror the run recorded in
    # reports/o1_baseline_fineweb.json
    p.add_argument("--dataset", default="FineWeb-Edu", choices=sorted(DATASET_DEFAULTS))
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--eval-holdout", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    # attach parameters: must match the probed run's adapter geometry
    # (ignored when --adapters-dir is given: the saved config wins)
    p.add_argument("--r", type=int, default=_R_DEFAULT)
    p.add_argument("--alpha", type=int, default=_ALPHA_DEFAULT)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--scope", default="all")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--init-a", default="kaiming_uniform",
                   choices=["kaiming_uniform", "normal"])
    # forward configuration: mirror the trainer's eval path
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16",
                   help="autocast during scoring, as in the trainer's eval")
    p.add_argument("--forward", choices=["kernel", "reference"], default="kernel",
                   help="kernel = FLUTE fused path (training parity); "
                        "reference = torch dequant path (parity cross-check)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="fineweb")
    p.add_argument("--output", default=None,
                   help="output JSON (default reports/o1_baseline_<tag>.json)")
    return p.parse_args()


def paired_diff(a, b):
    """Paired per-doc stats for d_i = b_i - a_i (negative = b better)."""
    n = len(a)
    d = [bi - ai for ai, bi in zip(a, b)]
    mean = sum(d) / n if n else 0.0
    var = (sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    std = var ** 0.5
    se = (std / math.sqrt(n)) if n else 0.0
    if se > 0:
        t = mean / se
    else:
        t = 0.0 if mean == 0 else (math.inf if mean > 0 else -math.inf)
    ds = sorted(d)
    if n == 0:
        median = 0.0
    elif n % 2:
        median = ds[n // 2]
    else:
        median = 0.5 * (ds[n // 2 - 1] + ds[n // 2])
    return {"n": n, "mean_delta": mean, "std": std, "stderr": se, "t_stat": t,
            "n_improved": sum(1 for x in d if x < 0),
            "n_worsened": sum(1 for x in d if x > 0),
            "median_delta": median,
            "p5_delta": ds[max(0, int(0.05 * n) - 1)] if n else 0.0,
            "p95_delta": ds[min(n - 1, int(0.95 * n))] if n else 0.0}


@torch.no_grad()
def score_docs(model, examples, device, amp_dtype, tag):
    """Per-document mean CE (batch of one), token-weighted aggregate."""
    model.eval()
    per_doc, n_toks = [], []
    t0 = time.time()
    for i, ex in enumerate(examples):
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        labels = ex["labels"].unsqueeze(0).to(device)
        attn = ex["attention_mask"].unsqueeze(0).to(device)
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        else:
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        per_doc.append(float(out.loss))
        n_toks.append(int((labels != -100).sum().item()))
        if (i + 1) % 32 == 0 or (i + 1) == len(examples):
            print(f"    [{tag}] {i + 1}/{len(examples)} docs ({time.time() - t0:.0f}s)",
                  flush=True)
    total_tok = sum(n_toks)
    wmean = sum(l * t for l, t in zip(per_doc, n_toks)) / max(1, total_tok)
    stats = {"n_docs": len(per_doc), "label_tokens": total_tok,
             "doc_mean_loss": sum(per_doc) / max(1, len(per_doc)),
             "token_weighted_mean_loss": wmean,
             "ppl": math.exp(wmean),
             "per_doc": per_doc}
    print(f"  [{tag}] mean loss {stats['doc_mean_loss']:.4f} | token-weighted "
          f"{wmean:.4f} (ppl {stats['ppl']:.2f}, {total_tok} label tokens)",
          flush=True)
    return stats


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    amp_dtype = None
    if args.amp == "bf16": amp_dtype = torch.bfloat16
    elif args.amp == "fp16": amp_dtype = torch.float16

    print("=" * 78, flush=True)
    print("O-1 baseline probe — paired same-doc: gap vs movement", flush=True)
    print(f"  dataset={args.dataset} max_samples={args.max_samples} seq={args.seq_len} "
          f"holdout={args.eval_holdout} seed={args.seed}", flush=True)
    print(f"  forward={args.forward} dtype={args.dtype} amp={args.amp} "
          f"adapters={args.adapters_dir or '(none)'} dense={args.dense}", flush=True)
    print("=" * 78, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- documents: replicate the trainer's eval holdout -----------------
    examples = load_sft_dataset(args.dataset, args.seq_len, tokenizer, args.max_samples)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(examples))
    eval_examples = [examples[i] for i in perm[:args.eval_holdout]]
    if not eval_examples:
        raise SystemExit("empty holdout — check --eval-holdout / --max-samples")
    print(f"[data] holdout replicated: {len(eval_examples)} docs "
          f"(trainer holdout subset: "
          f"{'yes' if args.eval_holdout <= len(examples) else 'no'})", flush=True)

    results, deltas, verdicts = {}, {}, []

    # --- optional dense FP16 reference ----------------------------------
    if args.dense:
        print("[dense] loading FP16 dense reference (~18 GiB) ...", flush=True)
        dense = load_dense_fp16(args.model, args.device)
        results["dense_fp16"] = score_docs(
            dense, eval_examples, args.device, amp_dtype, "dense")
        del dense
        release_model_memory()

    # --- palettized idx4 + QLoRA ----------------------------------------
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print("[palettized] loading idx4 model + attaching fresh (B=0) QLoRA ...",
          flush=True)
    model, metadata = pmod.load_palettized_model(
        args.artifacts_dir, args.model, device=args.device, dtype=dtype,
        residual=args.residual, reference=True)
    saved_cfg = None
    if args.adapters_dir:
        # Re-attach with the adapter dir's saved geometry (mirrors
        # qlora.load_qlora_model) so the trained weights land in a model
        # shaped exactly like the one that produced them.
        saved_cfg = qlora.QLoRAConfig.from_json(
            os.path.join(args.adapters_dir, "qlora_config.json"))
    model, qlora_cfg = qlora.attach_qlora(
        model, metadata, **_attach_geometry(args, saved_cfg))
    if saved_cfg is not None:
        _check_attach_geometry(saved_cfg, qlora_cfg)
    model.eval()
    switch = (lambda m: m.eval_kernel()) if args.forward == "kernel" \
        else (lambda m: m.eval_reference())
    for _, mod in qlora.iter_qlora_modules(model):
        switch(mod)
    _report_status(model)

    results["base_adapter0"] = score_docs(
        model, eval_examples, args.device, amp_dtype, "base+0")

    if args.adapters_dir:
        print(f"[adapter] loading trained adapter: {args.adapters_dir}", flush=True)
        qlora.load_qlora(model, args.adapters_dir, strict=True,
                         config=saved_cfg)
        results["base_adapterN"] = score_docs(
            model, eval_examples, args.device, amp_dtype, "base+N")

    # --- paired deltas ----------------------------------------------------
    if "base_adapterN" in results:
        deltas["training_movement"] = paired_diff(
            results["base_adapter0"]["per_doc"], results["base_adapterN"]["per_doc"])
    if "dense_fp16" in results:
        deltas["quantization_gap"] = paired_diff(
            results["dense_fp16"]["per_doc"], results["base_adapter0"]["per_doc"])

    if "quantization_gap" in deltas:
        qg = deltas["quantization_gap"]
        verdicts.append(
            f"QUANTIZATION GAP: dense is better by {qg['mean_delta']:.4f} nats/doc "
            f"(t={qg['t_stat']:.1f}) — "
            + ("within the expected 0.2-0.4 frozen-4bit band" if qg["mean_delta"] <= 0.45
               else "LARGER than expected: the idx4 artifacts deserve a closer look"))
    if "training_movement" in deltas:
        tm = deltas["training_movement"]
        if abs(tm["t_stat"]) < 2.0:
            verdicts.append(
                f"NO RESOLVABLE MOVEMENT: paired delta {tm['mean_delta']:+.4f} nats "
                f"(|t|={abs(tm['t_stat']):.1f} < 2, stderr {tm['stderr']:.4f}) — "
                f"the trained adapter is indistinguishable from no-op on this "
                f"holdout at this sample size")
        elif tm["mean_delta"] < 0:
            verdicts.append(
                f"TRAINING MOVED THE MODEL: improved by {abs(tm['mean_delta']):.4f} "
                f"nats/doc (t={tm['t_stat']:.1f}, {tm['n_improved']}/{tm['n']} docs "
                f"improved)")
        else:
            verdicts.append(
                f"TRAINING WORSENED THE MODEL: +{tm['mean_delta']:.4f} nats/doc "
                f"(t={tm['t_stat']:.1f}, {tm['n_worsened']}/{tm['n']} docs worse) — "
                f"regime problem (LR/clip/data), not a silent no-op")

    print("\n" + "=" * 78, flush=True)
    print("VERDICTS", flush=True)
    for v in verdicts:
        print(f"  * {v}", flush=True)
    print("=" * 78, flush=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "purpose": "O-1 paired same-doc probe: static quantization gap vs "
                   "training movement (see reports/training_oscillation_analysis.md)",
        "args": vars(args),
        "n_docs": len(eval_examples),
        "results": {k: v for k, v in results.items()},
        "deltas": deltas,
        "verdicts": verdicts,
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "none"),
        },
    }
    out = args.output or f"reports/o1_baseline_{args.tag}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
