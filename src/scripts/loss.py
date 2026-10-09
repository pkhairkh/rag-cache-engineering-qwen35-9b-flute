#!/usr/bin/env python3
"""scripts/loss.py — the layerwise distillation loss, standalone.

One implementation of the per-layer objective every layerwise consumer
shares (PROPOSAL §2.4): the engine's train/eval paths, the
``distill_loss.layerwise_loss`` shim, and the trainer that replaces them.
Pure torch, CPU-testable; no CUDA, no model files, no I/O.

Contract (see ``distill_loss`` for the full statement):

* objective — ``cos_weight * (1 - tok_cos) + mse_weight * rel_mse``
* mask — a (B, S) weight (1 = keep, 0 = drop) broadcast to both the
  per-token cosine map and the elementwise MSE terms
* aggregation — weight-means everywhere (never sums)
* units — rel_mse and cosine are dimensionless; metrics carry
  ``tok_cos`` (the cosine LOSS, 1 - cos), ``rel_mse`` and ``flat_cos``
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["distill_loss", "_masked_mean"]


def _masked_mean(x, weight):
    """Weighted mean over all elements of ``x``.

    weight: None (plain mean), a broadcastable scalar, or a tensor of
    ones/zeros (1 = keep, 0 = drop). The denominator is the weight sum
    clamped at 1e-12 (zero-mask safety). Units: the units of ``x``.
    """
    if weight is None:
        return x.mean()
    return (x * weight).sum() / weight.sum().clamp_min(1e-12)


def distill_loss(pred, target, weight=None, mse_weight: float = 0.5,
                 cos_weight: float = 1.0, sync: bool = True):
    """Per-token cosine + weighted relative MSE (both masked).

    pred/target: (B, S, H) fp32. weight: None, a scalar multiplier, or a
    (B, S) mask (1 = keep, 0 = drop) — a 2-D mask is broadcast
    internally against both the per-token cosine map (B, S) and the
    elementwise MSE terms (B, S, H). Aggregation is weight-MEAN in every
    term (row-weighted mean; never a sum), so a dropped token
    contributes to neither numerator nor denominator. Both rel_mse and
    the cosine are dimensionless; ``metrics['tok_cos']`` carries the
    cosine LOSS (1 - cos) and ``metrics['flat_cos']`` the true flattened
    cosine, reported for comparability — the trained objective is the
    per-token form: it gives every token a dense gradient instead of one
    rank-1 direction over the flattened elements.

    Weights: the layerwise CLI passes ``mse_weight=1.0,
    cos_weight=0.05`` (rel_mse primary and first-order,
    magnitude-preserving; the cosine a light regularizer — 1 - cos is
    ~rel_mse/2 for a near-orthogonal residual, so driving MSE down
    carries cosine to ~1.0 as a byproduct). The signature defaults
    (0.5/1.0) are the historical Stage-2 mirror, kept verbatim.

    sync (default True): the metrics dict carries Python floats. When
    False the values stay as 0-d tensors on the computation device: the
    train loop appends them to its telemetry window and materializes
    ONE batch of floats at window close, so a step forces ZERO device
    syncs from the loss. Values are identical; only their
    materialization is deferred.

    Returns (loss, metrics dict).
    """
    weight_tok = weight_el = weight
    if weight is not None and torch.is_tensor(weight) and weight.ndim == 2:
        weight_tok = weight              # (B, S)   vs cos_tok (B, S)
        weight_el = weight.unsqueeze(-1)  # (B, S, 1) vs (B, S, H)
    cos_tok = F.cosine_similarity(pred, target, dim=-1)          # (B, S)
    loss_cos = 1.0 - _masked_mean(cos_tok, weight_tok)
    diff2 = _masked_mean((pred - target).pow(2), weight_el)
    ref2 = _masked_mean(target.pow(2), weight_el)
    rel_mse = diff2 / ref2.clamp_min(1e-12)
    loss = cos_weight * loss_cos + mse_weight * rel_mse
    flat_cos = F.cosine_similarity(
        pred.reshape(1, -1), target.reshape(1, -1))
    if sync:
        metrics = {
            "tok_cos": float(loss_cos.item()),
            "rel_mse": float(rel_mse.item()),
            "flat_cos": float(flat_cos.item()),
        }
    else:
        metrics = {
            "tok_cos": loss_cos.detach(),
            "rel_mse": rel_mse.detach(),
            # cosine_similarity(1, -1) yields shape (1,); squeeze to a
            # uniform 0-d contract for the deferred-materialization path
            "flat_cos": flat_cos.detach().squeeze(),
        }
    return loss, metrics
