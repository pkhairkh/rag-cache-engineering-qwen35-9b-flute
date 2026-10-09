"""finetune.py — the lean §7 fine-tune loop (SPECIFICATION §7; PROPOSAL P4).

Next-token prediction with the online-TQ cache ACTIVE and M1/M2 attached.
Trainables (three groups, spec §7):
  1. linear_attn — the 24 per-layer linear-attention parameter groups
     (in_proj_qkv/z/b/a, conv1d, out_proj, norm, A_log, dt_bias) — "so S
     is discriminative";
  2. m1m2_gates — the global memories' read/write gates — "so the global
     caches carry info";
  3. luts — the W10 LUTs via `PalettizedLinear.make_trainable()` on the
     REFERENCE path (`forward="reference"` per the loader) — the
     straight-through primitive this repo kept for exactly this (the
     parent project's QLoRA/distillation trainer is NOT part of this repo).

~500 steps, AdamW + cosine (fp32 LUT masters — make_trainable promotes
them; everything else stays fp16/bf16). After training: `freeze_lut()`
(fp16 snap) and export to `pretrained_luts/` (rag/lut_export.py — full-LUT
artifacts per spec §11, NOT adapters).

The online-TQ cache is a per-batch FRESH cache: the loop trains the
weights AROUND the quantized-state regime (the R1 mitigation — the cache
states act as constants in the graph; the FHT reads are autograd-aware
where differentiable, the code lookups are not, by design).

The loop is model-agnostic: it needs `model(input_ids=..., past_key_values=
..., use_cache=True) -> logits` (the stub dry-run uses the same contract)
and a cache_factory.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch

import _paths  # noqa: F401

__all__ = ["FinetuneConfig", "build_trainables", "freeze_all_luts",
           "train", "next_token_loss"]


@dataclass
class FinetuneConfig:
    lr: float = 2e-5
    lut_lr: float = 1e-4            # straight-through masters get their own LR
    weight_decay: float = 0.0
    max_steps: int = 500            # spec §7: ~500 steps
    warmup_steps: int = 20
    min_lr_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 1234
    log_every: int = 50


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard shifted next-token cross-entropy (fp32 logits)."""
    return torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1))


def build_trainables(model) -> List[Dict]:
    """The three spec §7 param groups. `luts` are PROMOTED in place
    (make_trainable — fp32 masters, straight-through)."""
    groups: List[Dict] = []

    # 1. linear-attn params (the 24 Qwen3_5GatedDeltaNet groups)
    lin_params = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
            lin_params.extend(p for p in mod.parameters() if p.requires_grad)
    if lin_params:
        groups.append({"name": "linear_attn", "params": lin_params})

    # 2. the M1/M2 gates (the shared module attached by the W3.2 wiring)
    m1m2 = getattr(model, "m1m2", None)
    if m1m2 is not None:
        gate_params = [p for p in m1m2.parameters() if p.requires_grad]
        if gate_params:
            groups.append({"name": "m1m2_gates", "params": gate_params})

    # 3. the LUTs — promote to fp32 straight-through masters (reference path)
    import palettized_modules as pm
    lut_params = []
    for name, lin in pm.iter_palettized_linears(model):
        lin.make_trainable()
        lut_params.append(lin.lut)
        if getattr(lin, "has_stream2", False) and isinstance(lin.lut2, torch.Tensor):
            lut_params.append(lin.lut2)
    if lut_params:
        groups.append({"name": "luts", "params": lut_params,
                       "lr": "lut_lr"})
    return groups


def freeze_all_luts(model) -> int:
    """freeze_lut(snap_fp16=True) on every palettized linear (post-training
    export step). Returns the count."""
    import palettized_modules as pm
    n = 0
    for _name, lin in pm.iter_palettized_linears(model):
        lin.freeze_lut(snap_fp16=True)
        n += 1
    return n


def _cosine(step: int, total: int, cfg: FinetuneConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    t = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t))


def train(model, batches: Iterable[torch.Tensor], cache_factory: Callable,
          cfg: Optional[FinetuneConfig] = None,
          param_groups: Optional[List[Dict]] = None,
          loss_fn: Callable = next_token_loss,
          log_fn: Optional[Callable] = None) -> Dict:
    """The lean loop. `batches` yields (1, T) input-id tensors; a FRESH
    online-TQ cache is built per batch (the quantized-state training
    regime). Returns {"history": [(step, loss)], "seconds": float,
    "param_groups": [names]}."""
    cfg = cfg or FinetuneConfig()
    log = log_fn or (lambda *a: None)
    torch.manual_seed(cfg.seed)

    groups = param_groups if param_groups is not None \
        else build_trainables(model)
    if not groups:
        raise ValueError(
            "finetune.train: no trainable groups — the model has no "
            "linear-attn params, no M1/M2 gates, and no palettized linears "
            "(pass param_groups= for stub dry-runs)")
    defaults = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    opt_groups = []
    for g in groups:
        og = {"params": g["params"], "weight_decay": cfg.weight_decay}
        if g.get("lr") == "lut_lr":
            og["lr"] = cfg.lut_lr
        opt_groups.append(og)
    opt = torch.optim.AdamW(opt_groups, **defaults)

    history = []
    t0 = time.time()
    step = 0
    for batch in batches:
        if step >= cfg.max_steps:
            break
        cache = cache_factory()
        out = model(input_ids=batch, past_key_values=cache, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        if logits is None or not torch.is_tensor(logits):
            raise ValueError(
                "finetune.train: the model must return logits (tuple[0] or "
                "bare) — the stub dry-run contract")
        loss = loss_fn(logits, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], cfg.grad_clip)
        # cosine schedule on every group (relative LRs preserved)
        frac = _cosine(step, cfg.max_steps, cfg)
        for og, g in zip(opt.param_groups, groups):
            base = cfg.lut_lr if g.get("lr") == "lut_lr" else cfg.lr
            og["lr"] = base * frac
        opt.step()
        history.append((step, float(loss.item())))
        if step % cfg.log_every == 0:
            log(f"step {step}: loss={history[-1][1]:.4f} lr_frac={frac:.3f}")
        step += 1
    return {"history": history, "seconds": time.time() - t0,
            "param_groups": [g["name"] for g in groups], "steps": step}
