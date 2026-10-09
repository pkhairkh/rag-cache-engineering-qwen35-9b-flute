#!/usr/bin/env python3
"""scripts/report.py — the alignment producers.

Two subcommands, both pure measurement (no training, no artifact
writes): `report` — the per-layer down_proj alignment table (student
propagated through the captured boundary rows vs the teacher, layer by
layer — the "activations degrade layer to layer" metric); `weights` —
the per-module activation-space alignment (X @ W_student vs
X @ W_teacher on the calibration capture, exactly like the palettizer)
whose JSON feeds distill_rank_alloc.py, the QLoRA rank-decision input.

Import design (one-way): from-imports the engine's trainer plane; the
engine dispatches lazily. Unwinds at W2-T07/T08 and W2-T10.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import palettized_modules as pmod          # noqa: E402  (local, same dir)
from eval_common import atomic_json_dump as _atomic_json_dump  # noqa: E402
from eval_common import cosine_of_activations  # noqa: E402
from calibrate_real_text import (          # noqa: E402  (local, same dir)
    CalibrationCapture,
    load_calibration_sequences,
)
from loss import distill_loss              # noqa: E402  (local, same dir)
from capture import (                      # noqa: E402  (local, same dir)
    CaptureStore,
    _text_config,
)
from trainer import (                       # noqa: E402  (the trainer plane)
    XCache,
    _forward_layer,
    _layer_type,
    _pos_batch,
    _position_embeddings,
    propagate_layer,
)


def _flute_kernel_available() -> bool:
    """True if the compiled flute_extended extension imports and CUDA is
    up (ported engine-local helper: the report's --propagate-kernel
    gate; checks the actual symbol — a stray directory can import as an
    empty namespace package and fail later with a confusing
    AttributeError)."""
    import sys as _sys
    if not torch.cuda.is_available():
        return False
    try:
        flute_path = os.path.join(_HERE, "..", "flute_extended")
        if flute_path not in _sys.path:
            _sys.path.insert(0, flute_path)
        import flute_extended  # noqa: F401
        return hasattr(flute_extended, "qgemm_per_group_lut")
    except Exception:
        return False


def build_student(model_ref: str, artifacts_dir: str, reference: bool = True,
                  device: str = "cpu"):
    """Load the checkpoint + swap palettized modules — the exact
    deployment path (ported engine-local helper; the report subcommand
    propagates through the FULL student, unlike the layer-scoped
    trainer plane)."""
    model = _load_model_for_causal_lm(model_ref, device=device)
    metadata = pmod.load_metadata(artifacts_dir)
    pmod.apply_norm_gain_edits(model, artifacts_dir)
    pmod.replace_linear_with_palettized(
        model, metadata, artifacts_dir, reference=reference)
    model.eval()
    model.requires_grad_(False)
    model = model.to("cpu")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model, metadata

# ---------------------------------------------------------------------------
# report subcommand — the "activations degrade layer to layer" table
# ---------------------------------------------------------------------------

def cmd_report(args):
    device = args.device
    store = CaptureStore(capture_dir=args.capture_dir)
    model, metadata = build_student(args.model, args.artifacts_dir,
                                    reference=True)
    cfg = _text_config(model)
    layers = pmod.get_layers(model)
    kernel = args.propagate_kernel and _flute_kernel_available()

    rows = min(args.rows, store.rows)
    xcache = XCache(os.path.join(args.artifacts_dir, "report_cache"),
                    store.rows, store.seq, store.hidden, len(layers),
                    keep_every=10 ** 9)
    xcache.seed_from_capture(store)

    print("=" * 78)
    print("Per-layer activation alignment vs dense teacher (down_proj tap)")
    print("=" * 78)
    print(f"{'layer':>6} {'type':>16} {'tok_cos':>10} {'flat_cos':>10} "
          f"{'rel_mse':>12}")
    results = []
    pos16 = _position_embeddings(model, store.seq, torch.float16, device)
    for L in range(len(layers)):
        layer = layers[L]
        block_type = _layer_type(cfg, L)
        tap = {}
        h = layer.mlp.down_proj.register_forward_hook(
            lambda m, i, o: tap.__setitem__(
                "y", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                layer.to(device)
                cos_sum = mse_sum = flat_n = flat_d = 0.0
                CH = 8
                for i0 in range(0, rows, CH):
                    i1 = min(i0 + CH, rows)
                    x = xcache.read(L, i0, i1, device, torch.float16)
                    tgt = store.target(L, i0, i1, device, torch.float32)
                    _forward_layer(layer, x, _pos_batch(pos16, x.shape[0]),
                                   model)
                    _, met = distill_loss(tap["y"].float(), tgt)
                    n = i1 - i0
                    cos_sum += met["tok_cos"] * n
                    mse_sum += met["rel_mse"] * n
                    p = tap["y"].float().reshape(1, -1)
                    t = tgt.reshape(1, -1)
                    flat_n += float((p @ t.T).item()) * n
                    flat_d += float((p.norm() * t.norm()).item()) * n
                R = rows
                row = (L, block_type, 1 - cos_sum / R, flat_n / flat_d,
                       mse_sum / R)
                results.append(row)
                print(f"{L:>6} {block_type:>16} {row[2]:>10.6f} "
                      f"{row[3]:>10.6f} {row[4]:>12.4g}", flush=True)
        finally:
            h.remove()
            layer.to("cpu")
        # advance the trajectory through every layer (x_layer{N} = output of
        # the last layer, needed for the final-hidden comparison)
        if L + 1 <= len(layers):
            propagate_layer(L, layers[L], model, xcache, device, kernel)
    final_cos = None
    if store.has_final_hidden:
        with torch.no_grad():
            xN = xcache.read(len(layers), 0, rows, "cpu", torch.float16)
            xs = model.model.norm(xN).to(device, torch.float32)
            tf = store.final_hidden(0, rows, device, torch.float32)
            final_cos = cosine_of_activations(xs, tf)
            print(f"\nFinal hidden-state cosine vs teacher: {final_cos:.6f}")
    worst = min(results, key=lambda r: r[2])
    mean = sum(r[2] for r in results) / len(results)
    print(f"\nMean layer tok_cos: {mean:.6f}   worst: L{worst[0]} "
          f"({worst[2]:.6f})")
    if getattr(args, "json", None):
        # machine-readable mirror of exactly the numbers printed above
        # (per-layer tok_cos / flat_cos / rel_mse, the mean, the worst
        # layer, and the final-hidden cosine when captured) — consumed by
        # the W6 verification orchestrator.
        doc = {
            "layers": [{"layer": L, "type": bt, "tok_cos": tc,
                        "flat_cos": fc, "rel_mse": rm}
                       for (L, bt, tc, fc, rm) in results],
            "mean_tok_cos": mean,
            "worst_layer": int(worst[0]),
            "worst_tok_cos": worst[2],
        }
        if final_cos is not None:
            doc["final_hidden_cos"] = final_cos
        _atomic_json_dump(doc, args.json)
        print(f"report JSON: {args.json}")


# ---------------------------------------------------------------------------
# weights subcommand — per-module weight-space alignment (rank decision)
# ---------------------------------------------------------------------------

# metadata component key -> rank_map/adapter suffix (QLoRASplitQKV attrs)
_QKV_COMPONENTS = (("Q", "q"), ("K", "k"), ("V", "v"))


def _dense_shape(tmeta: dict, var: str):
    """(N, K) of a metadata tensor entry, validated positive."""
    shape = tmeta.get("dense_shape")
    try:
        N, K = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError):
        raise RuntimeError(
            f"{var}: metadata dense_shape {shape!r} is missing or invalid")
    if N <= 0 or K <= 0:
        raise RuntimeError(
            f"{var}: non-positive dense_shape (N={N}, K={K})")
    return N, K


def _student_effective_weight(meta: dict, artifacts_dir: str, shape, ctx: str):
    """Effective weight of one student module: reference dequant of the
    idx4 artifacts (LUT + indices) plus the stored residual factors
    (resA @ resB) when present — the same weight qlora_merge.py's
    _materialize_weight computes before re-palettizing. Returns an fp32
    (N, K) tensor.

    `shape=(N, K)` is required for QKV components (their metadata carries
    no dense_shape of its own; the parent tensor does — component N
    follows from packed_len_bytes, the load_palettized_weight convention).
    Plain tensors read dense_shape from their own metadata."""
    indices, lut_t, bw, gs, N, K = pmod._read_indices_and_lut(
        meta, artifacts_dir, ctx=ctx, shape=shape)
    if N <= 0 or K <= 0:
        raise RuntimeError(f"{ctx}: non-positive module shape (N={N}, K={K})")
    W = pmod.reference_dequant(indices, lut_t, N, K, gs, bw).float()
    resA, resB = pmod._read_residual(meta, artifacts_dir, ctx, N, K)
    if resA is not None and resB is not None:
        W = W + resA.float() @ resB.float()
    return W


def _weight_alignment_metrics(W_student, W_teacher, module: str, device: str):
    """(cos, rel_mse) of two (N, K) weight matrices, fp32 inputs on
    `device`, float64-accumulated reductions.

    rel_mse = ||Ws - Wt||^2 / ||Wt||^2. A zero-norm teacher weight is a
    hard error (rel_mse would be undefined), never a silent inf.

    Precision note: the weights are compared in fp32, but the reductions
    accumulate in float64 (1M-element chunks). A plain fp32 reduction over
    a real-scale (12288 x 4096 = 50M-element) weight was measured to read
    cos = 1.008 for IDENTICAL tensors — 16x the width of the r=0 tier
    threshold (0.9995) — which would corrupt the rank decision; fp64
    accumulation keeps the metric exact (identical tensors give cos = 1.0
    and rel_mse = 0.0 bit-exactly) at any scale."""
    a = W_student.detach().to(device=device, dtype=torch.float32).reshape(-1)
    b = W_teacher.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if a.shape != b.shape:
        raise RuntimeError(
            f"{module}: student/teacher weight shape mismatch "
            f"{tuple(W_student.shape)} vs {tuple(W_teacher.shape)}")
    dot = sq_a = sq_b = sq_d = 0.0
    CH = 1 << 20
    for i in range(0, a.numel(), CH):
        ac = a[i:i + CH].double()
        bc = b[i:i + CH].double()
        dot += float((ac * bc).sum().item())
        sq_a += float(ac.pow(2).sum().item())
        sq_b += float(bc.pow(2).sum().item())
        sq_d += float((ac - bc).pow(2).sum().item())
    if sq_b == 0.0:
        raise RuntimeError(
            f"{module}: teacher weight has zero norm — rel_mse undefined")
    cos = min(1.0, max(-1.0, dot / ((sq_a * sq_b) ** 0.5)))
    rel_mse = sq_d / sq_b
    return cos, rel_mse


def cmd_weights(args):
    """Activation-space alignment: X @ W_student vs X @ W_teacher.
    
    This matches the palettizer's measurement exactly: cosine and MSE
    are computed on output activations, not on weights.
    
    For AWQ norm-edit consumers, the input X is scaled by 1/s where
    s = (1 + gamma) / (1 + gamma_new), exactly like the palettizer.
    """
    device = args.device
    dtype = torch.float16
    
    # Loud guard: this measurement needs a full HF checkpoint (config +
    # weights + tokenizer for the calibration stream); a bare
    # safetensors dir (a toy teacher) is refused with an explanation
    # instead of a transformers traceback.
    if not os.path.isdir(args.teacher) or \
            not os.path.exists(os.path.join(args.teacher, "config.json")):
        raise SystemExit(
            f"--teacher must be a LOCAL full HF checkpoint directory "
            f"(config.json + safetensors + tokenizer files for the "
            f"calibration stream); {args.teacher!r} is not one — the "
            f"'weights' measurement cannot run on it")
    
    # Load metadata
    metadata = pmod.load_metadata(args.artifacts)
    tensors = metadata.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise SystemExit(f"{args.artifacts}/metadata.json: no tensors")
    
    # Load model
    print(f"Loading model from {args.teacher}")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    config = AutoConfig.from_pretrained(args.teacher, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, config=config, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    
    # Get layers (handles both text-only and multimodal checkpoints)
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise AttributeError("Cannot find layers")
    
    num_layers = len(layers)
    print(f"  {num_layers} layers")
    
    # Load calibration data
    print(f"Loading calibration sequences from {args.calib_source}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    seqs = load_calibration_sequences(
        tokenizer, args.calib_source, args.calib_seqs, args.calib_seq_len, seed=args.seed)
    print(f"  token stream: {tuple(seqs.shape)}")
    
    # Calibration capture - match palettizer's behavior exactly
    print(f"Running calibration capture (retain_rows={args.retain_rows})")
    
    def should_capture(name, weight):
        # name is like "model.layers.0.linear_attn.out_proj"
        # tensors keys are "model.layers.0.linear_attn.out_proj.weight"
        return weight.numel() > 0 and (name + ".weight") in tensors
    
    cap = CalibrationCapture(
        model, should_capture,
        retain_rows=args.retain_rows,
        full_gram=False,
        gram_device="cuda")
    cap.run(seqs, device=device, batch_size=args.batch_size, log_every=32)
    cap.remove_hooks()
    
    # Load norm edits
    norm_edits_path = os.path.join(args.artifacts, "norm_gain_edits.json")
    norm_edits = {}
    if os.path.exists(norm_edits_path):
        with open(norm_edits_path) as f:
            norm_edits_doc = json.load(f)
        norm_edits = norm_edits_doc.get("edits", {})
        print(f"  [norm edits] loaded {len(norm_edits)} entries")
    
    def get_norm_edit_for_weight(var_name):
        """Return (norm_name, norm_entry) if var_name is a norm-edit consumer."""
        for norm_name, entry in norm_edits.items():
            if var_name in entry.get("consumers", []):
                return norm_name, entry
        return None, None
    
    # Compute cosine per tensor
    print("\n" + "=" * 78)
    print("Activation-space alignment (X @ W_student vs X @ W_teacher)")
    print("=" * 78)
    print(f"{'module':<56} {'N':>7} {'K':>7} {'cos':>10} {'rel_mse':>12}")
    
    modules = {}
    
    for tensor_name, tmeta in tensors.items():
        var = tmeta.get("var", tensor_name)
        if not var.endswith(".weight"):
            continue
        
        # Parse layer and module name
        parts = var.split(".")
        if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
            continue
        layer_idx = int(parts[2])
        module_name = ".".join(parts[3:-1])  # e.g., "linear_attn.in_proj_qkv"
        
        # Get captured activation sample - key is (layer_idx, module_name)
        X_sample = cap.sample((layer_idx, module_name))
        if X_sample is None:
            print(f"  {var}: no captured activation, skipping")
            continue
        
        X_sample = X_sample.to(device=device, dtype=torch.float32)
        
        # Get teacher weight
        layer = layers[layer_idx]
        W_teacher = dict(layer.named_parameters())[module_name + ".weight"].detach().float()
        
        # Check if this weight has AWQ norm edit
        norm_name, norm_entry = get_norm_edit_for_weight(var)
        X_f = X_sample  # default: unscaled
        if norm_entry is not None:
            # AWQ scaling: apply the transform exactly like palettizer
            # gamma_new = (1 + gamma) / s - 1, so s = (1 + gamma) / (1 + gamma_new)
            # X' = X / s, W' = W * s
            gamma_new = np.load(os.path.join(args.artifacts, norm_entry["file"]))
            gamma_new = torch.from_numpy(gamma_new).float().to(device)
            
            # Load teacher's gamma (pristine)
            if "input_layernorm" in norm_name:
                norm = layer.input_layernorm
            elif "post_attention_layernorm" in norm_name:
                norm = layer.post_attention_layernorm
            else:
                norm = None
            
            if norm is not None:
                gamma = norm.weight.detach().float().to(device)
                s = (1.0 + gamma) / (1.0 + gamma_new)
                # Transform X: X' = X / s
                X_f = X_sample / s.unsqueeze(0)  # (R, K) / (1, K)
                # Transform W: W' = W * s (for teacher output in scaled space)
                W_teacher = W_teacher * s.unsqueeze(0)
        
        if "components" in tmeta:
            # SplitQKV - handle each component
            K = int(tmeta.get("dense_shape", [0, 4096])[1])
            off = 0
            for comp, tag in [("Q", "q"), ("K", "k"), ("V", "v")]:
                if comp not in tmeta["components"]:
                    continue
                cm = tmeta["components"][comp]
                comp_N = int(cm["packed_len_bytes"]) * (8 // int(cm["bitwidth"])) // K
                
                # Dequantize student weight
                indices, lut, bw, gs, N_, K_ = pmod._read_indices_and_lut(
                    cm, args.artifacts, ctx=f"weights:{comp}", shape=(comp_N, K))
                W_student = pmod.reference_dequant(indices, lut, comp_N, K, gs, bw).float().to(device)
                
                # Compute outputs
                Y_student = X_f @ W_student.t()
                Y_teacher = X_f @ W_teacher[off:off + comp_N, :].t()
                
                cos = cosine_of_activations(Y_student, Y_teacher)
                rel_mse = float(((Y_student - Y_teacher) ** 2).sum() / (Y_teacher ** 2).sum())
                
                full_key = f"{var[:-7]}.{tag}"
                modules[full_key] = {"cos": cos, "rel_mse": rel_mse, "N": comp_N, "K": K}
                print(f"  {full_key:<56} {comp_N:>7} {K:>7} {cos:>10.6f} {rel_mse:>12.4e}")
                off += comp_N
        else:
            # Plain tensor
            N, K = int(tmeta["dense_shape"][0]), int(tmeta["dense_shape"][1])
            
            # Dequantize student weight
            indices, lut, bw, gs, N_, K_ = pmod._read_indices_and_lut(
                tmeta, args.artifacts, ctx="weights", shape=(N, K))
            W_student = pmod.reference_dequant(indices, lut, N, K, gs, bw).float().to(device)
            
            # Compute outputs
            Y_student = X_f @ W_student.t()
            Y_teacher = X_f @ W_teacher.t()
            
            cos = cosine_of_activations(Y_student, Y_teacher)
            rel_mse = float(((Y_student - Y_teacher) ** 2).sum() / (Y_teacher ** 2).sum())
            
            full_key = var[:-7]  # strip .weight
            modules[full_key] = {"cos": cos, "rel_mse": rel_mse, "N": N, "K": K}
            print(f"  {full_key:<56} {N:>7} {K:>7} {cos:>10.6f} {rel_mse:>12.4e}")
    
    # Write output
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.artifacts)), "weights_alignment.json")
    doc = {
        "modules": modules,
        "artifacts": os.path.abspath(args.artifacts),
        "teacher": os.path.abspath(args.teacher),
        "calib_source": args.calib_source,
        "retain_rows": args.retain_rows,
    }
    _atomic_json_dump(doc, out_path)
    
    n = len(modules)
    if n > 0:
        best = max(modules.items(), key=lambda kv: kv[1]["cos"])
        worst = min(modules.items(), key=lambda kv: kv[1]["cos"])
        worst_mse = max(modules.items(), key=lambda kv: kv[1]["rel_mse"])
        print("-" * 78)
        print(f"  modules={n}  best cos={best[1]['cos']:.6f} ({best[0]})")
        print(f"  worst cos={worst[1]['cos']:.6f} ({worst[0]})")
        print(f"  worst rel_mse={worst_mse[1]['rel_mse']:.4e} ({worst_mse[0]})")
    print(f"  wrote {out_path} (feed to scripts/distill_rank_alloc.py)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Alignment producers for the FLUTE-palettized "
                    "Qwen3.5-9B layerwise pipeline: the per-layer report "
                    "table and the per-module weights alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default="Qwen/Qwen3.5-9B",
                        help="HF id or LOCAL checkpoint dir (required for "
                             "--target-mode teacher-on-student)")
        sp.add_argument("--device", default=(
            "cuda:0" if torch.cuda.is_available() else "cpu"))

    # ---- report ------------------------------------------------------------
    r = sub.add_parser(
        "report", help="per-layer down_proj cosine vs teacher — the "
                       "layer-to-layer degradation table")
    common(r)
    r.add_argument("--artifacts-dir", required=True)
    r.add_argument("--capture-dir", required=True)
    r.add_argument("--rows", type=int, default=64)
    r.add_argument("--propagate-kernel", action="store_true")
    r.add_argument("--json", default=None,
                   help="also dump the per-layer table (the exact numbers "
                        "printed: tok_cos / flat_cos / rel_mse per layer, "
                        "mean, worst, final-hidden cosine) as JSON to this "
                        "path — consumed by distill_eval.py (W6)")

    # ---- weights -----------------------------------------------------------
    wgt = sub.add_parser(
        "weights", help="per-module activation-space alignment (X @ W_student vs X @ W_teacher) "
                        "using calibration capture, exactly like the palettizer — the QLoRA "
                        "rank-decision input; feed the JSON to scripts/distill_rank_alloc.py")
    wgt.add_argument("--artifacts", required=True,
                     help="palettized STUDENT artifacts dir (metadata.json)")
    wgt.add_argument("--teacher", required=True,
                     help="LOCAL DENSE teacher checkpoint dir")
    wgt.add_argument("--out", default=None,
                     help="output JSON path (default: "
                          "<artifacts>/../weights_alignment.json)")
    wgt.add_argument("--device", default="cuda:0",
                     help="device for model and computation")
    # Calibration args (match palettizer defaults)
    wgt.add_argument("--calib-source", default="fineweb",
                     help="calibration text source")
    wgt.add_argument("--calib-seqs", type=int, default=32,
                     help="calibration sequences (default 32 for memory)")
    wgt.add_argument("--calib-seq-len", type=int, default=1024,
                     help="tokens per sequence (default 1024 for memory)")
    wgt.add_argument("--retain-rows", type=int, default=128,
                     help="activation rows retained per tensor")
    wgt.add_argument("--batch-size", type=int, default=1,
                     help="calibration batch size (default 1 for memory)")
    wgt.add_argument("--seed", type=int, default=42)
    wgt.add_argument("--teacher-prefix", default=None,
                     help="prefix for teacher tensor names (e.g. 'model.language_model.' "
                          "for Qwen3_5ForConditionalGeneration checkpoints)")
    


    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "weights":
        cmd_weights(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown subcommand {args.cmd}")


if __name__ == "__main__":
    main()
