#!/usr/bin/env python3
"""muon_optimizer.py — vendored Muon optimizer for the QLoRA adapter
factors (the PROPOSAL.md §3.3 spec block; the engine's private
``torch.optim._muon`` import is replaced by this module — stock CPU
wheels do not ship the private API, so before this file the ``muon``
path could not even be exercised on the coding box).

Muon IS stochastic gradient descent: a momentum buffer, ONE
preconditioning step (Newton–Schulz orthogonalization of the update),
and a step of size lr — just with the update orthogonalized instead of
diagonally rescaled. It is built for 2D matrix parameters, which is
exactly the QLoRA path's trainable set: {lora_A (r, K), lora_B (N, r)}
per module — no embeddings, no 1D gains (the norm folds are frozen), no
param-group surgery.

Design decisions (all recorded because they are load-bearing):
  * fp32 throughout (the adapter parameters are fp32 by construction in
    qlora.py; fp32 NS is bit-reproducible on CPU and GPU for the tests).
  * Newton–Schulz, 5 iterations, coefficients (3.4445, -4.7750, 2.0315),
    Frobenius pre-normalization; the iteration always runs on the
    orientation whose Gram side is the smaller dimension (lora_B is
    (N, r) with N >> r, so it orthogonalizes as (r, N) — O(r^2 * N) per
    NS step, microseconds at these sizes).
  * momentum 0.95, Nesterov ON: buf <- m * buf + g; d <- g + m * buf
    (Nesterov) else d <- buf. The Nesterov form is the classical
    lookahead — the reference semantics the unit tests hand-roll.
  * weight decay 0 (the deployment format carries no optimizer state
    and decay would fight the warm-start magnitude).
  * step scale = 0.2 * sqrt(max(shape)) — the Moonlight/Kimi-K2 RMS
    matching: after orthogonalization the update's element RMS is
    1/sqrt(max(m, n)), so the applied step's RMS is 0.2 * lr regardless
    of factor shape. THIS is why Muon lr is NOT transferable from
    AdamW's calibration — the pilot sweep {1e-3, 3e-3, 1e-2, 3e-2}
    (gate G1-O) selects the peak lr on evidence.
  * Startup assert: every param in a muon group is 2D — a non-2D param
    is a LOUD refusal, never a silent skip. Optimizer state is created
    fresh per layer job by the engine (one optimizer per layer) — no
    cross-layer leakage by construction.

Scope boundaries (PROPOSAL.md §3.3): the LUT escalation path keeps
AdamW (its codebook tensors are (n_groups, 16)-shaped and
orthogonalization discards the per-group scale structure that path
needs); Stage-2 polish keeps its spec-exact optimizer. Muon is the
Stage-1/1.5 default on the 2D adapter factors only, pilot-gated with
AdamW one flag away.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch

# Newton–Schulz coefficients (the modded-nanogpt / Keller Jordan values,
# tuned for ~5-iteration convergence to an orthogonal matrix).
_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_NS_STEPS = 5
_FROBENIUS_EPS = 1e-7
# RMS-matching step scale: applied step element RMS == STEP_RMS * lr.
STEP_RMS = 0.2


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = _NS_STEPS,
                                eps: float = _FROBENIUS_EPS) -> torch.Tensor:
    """Approximate the orthogonalization (UV^T of the SVD) of a 2D matrix
    with a quintic Newton–Schulz iteration, fp32.

    The iteration X -> aX + (bA + cA^2) X with A = X X^T converges to an
    orthogonal matrix with the same singular directions as G and flat
    singular values (every direction equal norm). It always runs on the
    orientation whose Gram side is the SMALLER dimension (X is stored
    transposed when rows > cols, transposed back at the end) — the
    smaller Gram matrix is what keeps the cost at O(r^2 * N).

    G must be 2D (a loud assertion — this function has no meaning on any
    other shape class). Returns a NEW tensor; G is not modified.
    """
    if G.ndim != 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5: expected a 2D matrix, got "
            f"shape {tuple(G.shape)} (ndim={G.ndim}) — Muon orthogonalizes "
            f"matrices only")
    a, b, c = _NS_COEFFICIENTS
    X = G.detach().to(torch.float32).clone()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T                       # Gram side = the smaller dimension
        transposed = True
    X = X / (X.norm() + eps)          # Frobenius pre-normalization
    for _ in range(int(steps)):
        A = X @ X.T                   # (k, k) with k = min(m, n)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def muon_step_scale(shape) -> float:
    """The shape factor of one Muon step: STEP_RMS * sqrt(max(m, n)).

    After NS the update's element RMS is 1/sqrt(max(m, n)), so the
    applied update p -= lr * scale * update has element RMS exactly
    STEP_RMS * lr — shape-independent, matching AdamW's ~lr update RMS
    at a 0.2 conservatism factor (the Moonlight/Kimi-K2 calibration)."""
    m, n = int(shape[0]), int(shape[1])
    return STEP_RMS * math.sqrt(max(m, n))


class Muon(torch.optim.Optimizer):
    """Muon on 2D matrix parameters (see the module docstring).

    ``params`` may be any iterable of Parameters or param groups (the
    torch.optim.Optimizer contract); EVERY param must be 2D — the
    constructor refuses loudly otherwise (a non-2D param in a muon group
    is a wiring bug, never a silently-skipped tensor).

    lr is per-group (the engine's warmup+cosine schedule writes
    param_group['lr'] every step — that surface is respected)."""

    def __init__(self, params: Iterable, lr: float = 0.02,
                 momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = _NS_STEPS):
        if lr <= 0:
            raise ValueError(f"Muon: lr must be positive (got {lr})")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"Muon: momentum must be in [0, 1) (got {momentum})")
        defaults = dict(lr=lr, momentum=momentum, nesterov=bool(nesterov),
                        weight_decay=weight_decay, ns_steps=int(ns_steps))
        super().__init__(params, defaults)
        # Startup assert: the whole trainable set is 2D matrices.
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise RuntimeError(
                        f"Muon: parameter with shape {tuple(p.shape)} "
                        f"(ndim={p.ndim}) is not a 2D matrix — Muon "
                        f"orthogonalizes matrix factors only; put non-2D "
                        f"tensors in an AdamW group (refusing loudly, "
                        f"never silently skipping)")

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError(
                        f"Muon.step: gradient with shape {tuple(g.shape)} "
                        f"is not 2D (param shape {tuple(p.shape)}) — "
                        f"refusing loudly")
                if tuple(g.shape) != tuple(p.shape):
                    raise RuntimeError(
                        f"Muon.step: gradient shape {tuple(g.shape)} does "
                        f"not match param shape {tuple(p.shape)} — a "
                        f"shape-mismatched grad is a wiring bug, refusing "
                        f"loudly")
                if g.dtype != torch.float32:
                    g = g.float()
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, dtype=torch.float32)
                    state["step_count"] = 0
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    d = g.add(buf, alpha=momentum)   # lookahead form
                else:
                    d = buf
                update = zeropower_via_newtonschulz5(d, steps=ns_steps)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update.to(p.dtype),
                       alpha=-lr * muon_step_scale(p.shape))
                state["step_count"] += 1
        return loss


__all__ = ["Muon", "zeropower_via_newtonschulz5", "muon_step_scale",
           "STEP_RMS", "_NS_COEFFICIENTS"]
