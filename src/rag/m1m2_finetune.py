"""m1m2_finetune.py — the §7 contrastive fine-tune of the M1/M2 gates.

THE OBJECTIVE (SPECIFICATION §7 / the W17 handover): make the two global
memories' STATE a general semantic-similarity space, so a question and
its relevant document land close together in cos-sim over the dequantized
M1/M2 codes (the §4 vector's last two units) — retrieval by content
overlap, no embedder, corpus-INDEPENDENT (trained on general paraphrase/
entailment/QA pairs, never on the served corpus).

WHY A SPECIAL TRAINER (the gradient path): the production online loop is
gradient-dead for the WRITE gates BY DESIGN — the wiring's write goes
`m_new -> TurboQuant.quant -> codes -> dequant -> next read`, and the
code lookups sever autograd (finetune.py's R1: "the cache states act as
constants in the graph"). The READ gate still trains through any
next-token loss (read_gate multiplies the read output live), but the
write gates only ever receive gradient through a DIFFERENTIABLE state —
which this module builds outside the quantized loop:

  1. CAPTURE: the text is prefilled under no_grad through a normal
     TQCache (the exact production path — reseeded from the system reset
     point, the wiring writes codes as always); forward hooks on the
     SHARED m1m2 module record every layer's call (k, v, layer_idx,
     positions), detached.
  2. REPLAY: the 24 recorded calls are re-run under enable_grad through
     the module's own `write` on a live state chain
     (state <- state + g_L * scatter(k_L)) starting from the reseeded
     m_init. The additive write makes the final state exactly
     m_init + sum_L g_L * Delta_L in real arithmetic (the §2.2
     path-independence property), so the replay state equals the cache's
     held state up to the inter-layer quantization round-trips — the
     noise-free, gate-differentiable proxy of the deployed state.

THE LOSS: InfoNCE with in-batch negatives over the B pair-states
(sim = 0.5*(cos_m1 + cos_m2), temperature tau) — positives on the
diagonal, every other pair in the batch a negative. Positive pairs come
from GENERAL similarity datasets (MRPC/QQP/PAWS/SNLI/MNLI/STS-B/SQuAD —
the handover's list); the served corpus is NEVER a training input.

WHAT IS TRAINED: ONLY the M1M2 module's gate vectors
(write_gate_k/write_gate_v per linear layer, optionally read_gate) —
`freeze_all_but_gates` turns off every other parameter of the model
(the LUTs, the linear-attention weights, everything). The gates are
promoted to fp32 masters in place (3 x num_linear_layers floats —
negligible); the write path's promote/accumulate semantics keep fp16
k/v inputs exact in fp32 states.

READ GATE NOTE: read_gate multiplies the read output (softmax(q@M1)@M2)
which the wiring adds to every linear layer's core attention output —
it affects GENERATION, not the retrieval state. The contrastive loss
does not touch it (the state is built by writes only); it stays at the
P3 one-init unless a next-token auxiliary term is used (the driver's
--aux-nll-weight; its gradient flows through the live read_gate
multiply) or --freeze-read-gate pins it.

DEPLOYMENT ORDER (the protocol): train gates on general pairs ->
save_gates(path) -> INGEST with the trained gates (the system state +
every chunk delta then live in the trained-gate regime) -> index ->
query with the SAME gates + mem_size (load_quant_model(
m1m2_gates_path=..., m1m2_mem_size=...); geometry drift fails loudly).

Artifacts: save_gates/load_gates (.npz: the 3 gate vectors + the
geometry identity + provenance; load validates the geometry EXACTLY —
a gates file trained at another mem_size/head-count is a loud error,
never a silent reshape).

This module imports torch + numpy only at top (transformers stays out —
`TQCache`/`reseed_cache` import lazily inside the functions, house rule).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import _paths  # noqa: F401  (house sys.path anchor — must precede sibling imports)

__all__ = [
    "GatesTrainConfig", "save_gates", "load_gates",
    "freeze_all_but_gates", "M1M2CallRecorder",
    "prefill_capture_calls", "replay_states", "infonce_pairs_loss",
    "cosine_spread", "train_gates",
]


# ------------------------------------------------------------- gates I/O ---
_GATE_KEYS = ("write_gate_k", "write_gate_v", "read_gate")
_GEOM_KEYS = ("num_heads", "head_dim", "mem_size", "num_linear_layers")


def _module_geometry(module) -> Dict[str, int]:
    return {k: int(getattr(module, k)) for k in _GEOM_KEYS}


def save_gates(module, path: str, extra_meta: Optional[dict] = None) -> str:
    """Persist the gate vectors + geometry identity as one .npz.

    The three gate vectors are the ONLY trainable state of the §7
    fine-tune (the memories themselves are runtime cache state, never
    weights). The geometry keys ride along so `load_gates` can refuse a
    module-mismatched file loudly (a gates file trained at mem_size=1024
    applied to a mem_size=128 module would otherwise silently scatter
    into the wrong slots).
    """
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path = path + ".npz"
    arrays = {k: getattr(module, k).detach().cpu().numpy().astype(np.float64)
              for k in _GATE_KEYS}
    meta = _module_geometry(module)
    meta["created_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    meta["torch_version"] = torch.__version__
    if extra_meta:
        meta.update({k: v for k, v in extra_meta.items()
                     if isinstance(v, (str, int, float, bool))})
    np.savez(path, **arrays, **{f"meta_{k}": v for k, v in meta.items()})
    return path


def load_gates(module, path: str) -> Dict[str, object]:
    """Load a save_gates artifact INTO the module (in place, geometry
    validated exactly). Returns the file's metadata dict."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"load_gates: {path} not found")
    with np.load(path) as z:
        missing = [k for k in _GATE_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"load_gates: {path} misses keys {missing}")
        file_meta = {k[len("meta_"):]: z[k].item() if hasattr(z[k], "item")
                     else z[k] for k in z.files if k.startswith("meta_")}
        arrays = {k: np.asarray(z[k], dtype=np.float64) for k in _GATE_KEYS}
    want = _module_geometry(module)
    for k in _GEOM_KEYS:
        if k not in file_meta:
            raise ValueError(
                f"load_gates: {path} carries no meta_{k} — the geometry "
                f"identity is missing; refusing an unvalidated file")
        if int(file_meta[k]) != want[k]:
            raise ValueError(
                f"load_gates: {path} was trained at {k}="
                f"{int(file_meta[k])}, this module has {k}={want[k]} — "
                f"geometry drift; refusing a silent reshape")
    for k in _GATE_KEYS:
        param = getattr(module, k, None)
        if param is None or not torch.is_tensor(param):
            raise TypeError(
                f"load_gates: module has no parameter {k!r} — pass the "
                f"M1M2 module, not the model")
        want_shape = tuple(param.shape)
        if arrays[k].shape != want_shape:
            raise ValueError(
                f"load_gates: {k} shape {arrays[k].shape} != the module's "
                f"{want_shape} — the gate-vector geometry disagrees")
        with torch.no_grad():
            param.copy_(torch.from_numpy(arrays[k]).to(
                param.device, param.dtype))
    return file_meta


# ------------------------------------------------------------- freezing ---
def freeze_all_but_gates(model, include_read_gate: bool = True
                         ) -> List[str]:
    """Turn off EVERY parameter of `model` except the M1/M2 gate vectors
    (promoted to fp32 masters in place — 3 x num_linear_layers floats).

    `model` is either the CausalLM (gates at model.model.m1m2) or the
    inner TextModel (gates at model.m1m2). Returns the trainable names.
    """
    inner = getattr(model, "model", model)
    m1m2 = getattr(inner, "m1m2", None)
    if m1m2 is None:
        raise ValueError(
            "freeze_all_but_gates: no m1m2 module found — load the model "
            "with use_m1m2=True (the W17 loader path)")
    for p in model.parameters():
        p.requires_grad_(False)
    names = ["write_gate_k", "write_gate_v"]
    if include_read_gate:
        names.append("read_gate")
    for name in names:
        p = getattr(m1m2, name)
        p.requires_grad_(True)
        if p.dtype != torch.float32:
            p.data = p.data.float()
    return [f"m1m2.{n}" for n in names]


# ------------------------------------------------------------- capture ----
class M1M2CallRecorder:
    """Forward hooks on the shared M1M2 module: records every wiring call
    (k, v, layer_idx, positions) DETACHED — the replay inputs.

    The wiring (modeling.py) calls the module once per linear-attention
    layer per forward: forward(q, k, v, m1, m2, layer_idx, positions=...).
    The recorder keeps k/v (B, H, T, D) detached clones (memory scales
    with T, not mem_size) + the layer ordinal + the write positions.
    """

    def __init__(self, module):
        self.module = module
        self.calls: List[Tuple[torch.Tensor, torch.Tensor, int,
                               Optional[torch.Tensor]]] = []
        self._handle = module.register_forward_hook(
            self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, output):
        # args: (q, k, v, m1, m2, layer_idx); kwargs: {"positions": ...}
        q, k, v, m1, m2, layer_idx = args
        positions = kwargs.get("positions")
        self.calls.append((
            k.detach().clone(), v.detach().clone(), int(layer_idx),
            positions.detach().clone() if positions is not None else None))

    def detach_calls(self) -> List[Tuple[torch.Tensor, torch.Tensor, int,
                                         Optional[torch.Tensor]]]:
        return list(self.calls)

    def remove(self) -> None:
        self._handle.remove()


def prefill_capture_calls(model, input_ids: torch.Tensor, cache,
                          system) -> Tuple[List, Optional[torch.Tensor],
                                           Optional[torch.Tensor]]:
    """Prefill `input_ids` under no_grad through the production TQ path
    (reseeded from the system reset point) while recording every M1/M2
    wiring call. Returns (calls, m1_init, m2_init) — the replay inputs.

    m1_init/m2_init are the reseeded memories (dequantized system codes,
    fp32, detached); None system M1/M2 (a corpus ingested without M1/M2,
    or the pre-system zero state) falls back to the module's zeros — the
    spec §3.2 loop's first-forward contract.
    """
    from ingest import reseed_cache
    inner = getattr(model, "model", model)
    module = getattr(inner, "m1m2", None)
    if module is None:
        raise ValueError(
            "prefill_capture_calls: the model has no m1m2 wiring — load "
            "with use_m1m2=True")
    reseed_cache(cache, system)
    # m_init BEFORE the prefill: the reseeded (system) state the first
    # layer's write starts from — reading it AFTER would hand the replay
    # the FINAL state (init + all deltas twice over; the W17 test rig
    # caught exactly this).
    m1_init = cache.read_m1(dtype=torch.float32)
    m2_init = cache.read_m2(dtype=torch.float32)
    if m1_init is None:
        m1_init = module.init_state(dtype=torch.float32,
                                    device=input_ids.device)
    else:
        m1_init = m1_init.detach().to(torch.float32)
    if m2_init is None:
        m2_init = module.init_state(dtype=torch.float32,
                                    device=input_ids.device)
    else:
        m2_init = m2_init.detach().to(torch.float32)
    recorder = M1M2CallRecorder(module)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache,
                  use_cache=True)
    finally:
        recorder.remove()
    return recorder.detach_calls(), m1_init, m2_init


# -------------------------------------------------------------- replay ----
def replay_states(module, calls: Sequence, m1_init: torch.Tensor,
                  m2_init: torch.Tensor,
                  checkpoint: Optional[bool] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-run the recorded wiring calls under GRAD through the module's
    own `write` — the gate-differentiable state chain.

    state <- state + g_L * scatter(k_L) per call, starting from m_init.
    The additive write makes the result m_init + sum_L g_L*Delta_L —
    exactly the cache's held state up to the inter-layer quantization
    round-trips (the deployed state is the quantized version of this
    sum; the replay is the noise-free proxy the loss trains on).

    checkpoint: None = auto (True on CUDA with large states), True =
    torch.utils.checkpoint per call (recomputes the scatter in backward —
    memory scales with the captured k/v, not 24 x state temporaries),
    False = the plain graph (CPU tests).
    """
    m1, m2 = m1_init, m2_init
    if checkpoint is None:
        checkpoint = (m1.is_cuda and m1.numel() > 1_000_000)
    for (k, v, layer_idx, positions) in calls:
        if checkpoint:
            def _write(k, v, m1, m2, layer_idx=layer_idx, positions=positions):
                return module.write(k, v, m1, m2, layer_idx,
                                    positions=positions)
            m1, m2 = torch.utils.checkpoint.checkpoint(
                _write, k, v, m1, m2, use_reentrant=False)
        else:
            m1, m2 = module.write(k, v, m1, m2, layer_idx,
                                  positions=positions)
    return m1, m2


# --------------------------------------------------------------- loss -----
def _flat_unit(x: torch.Tensor) -> torch.Tensor:
    """(B, ...) -> (B, dims) fp32, L2-normalized (the cosine operand)."""
    b = x.shape[0]
    v = x.reshape(b, -1).to(torch.float32)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v


def infonce_pairs_loss(m1_a: torch.Tensor, m2_a: torch.Tensor,
                       m1_b: torch.Tensor, m2_b: torch.Tensor,
                       tau: float = 0.07) -> torch.Tensor:
    """InfoNCE with in-batch negatives over the pair states.

    sim(i, j) = 0.5 * (cos(m1_a[i], m1_b[j]) + cos(m2_a[i], m2_b[j]));
    logits = sim / tau; the diagonal (the true pairs) are the targets.
    B >= 2 required (in-batch negatives); the matrix is symmetric so the
    a->b direction covers both.
    """
    if tau <= 0 or not isinstance(tau, float):
        raise ValueError(f"infonce_pairs_loss: tau must be > 0, got {tau!r}")
    b = m1_a.shape[0]
    if b < 2:
        raise ValueError(
            f"infonce_pairs_loss: batch of {b} pair has no in-batch "
            f"negatives — use B >= 2")
    sim = 0.5 * (_flat_unit(m1_a) @ _flat_unit(m1_b).T
                 + _flat_unit(m2_a) @ _flat_unit(m2_b).T)
    logits = sim / tau
    targets = torch.arange(b, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def cosine_spread(m1_a, m2_a, m1_b, m2_b) -> Tuple[float, float]:
    """(mean cos over the TRUE pairs, mean cos over the SHIFTED pairs) —
    the held-out discrimination metric (no grad; the gap is the signal)."""
    with torch.no_grad():
        sim = 0.5 * ((_flat_unit(m1_a) * _flat_unit(m1_b)).sum(-1)
                     + (_flat_unit(m2_a) * _flat_unit(m2_b)).sum(-1))
        b = sim.shape[0]
        if b < 2:
            return float(sim.mean()), float("nan")
        shifted = sim.roll(1)
        return float(sim.mean()), float(shifted.mean())


# ------------------------------------------------------------- the loop ---
@dataclass
class GatesTrainConfig:
    lr: float = 1e-2                # gates are 3 x 24 scalars — a hot lr
    weight_decay: float = 0.0
    max_steps: int = 300
    warmup_steps: int = 20
    min_lr_frac: float = 0.05
    grad_clip: float = 1.0
    tau: float = 0.07               # the InfoNCE temperature
    batch_pairs: int = 8            # B (>= 2: in-batch negatives)
    freeze_read_gate: bool = True   # the contrastive loss never sees it
    seed: int = 1234
    log_every: int = 25
    history: List[Dict[str, float]] = field(default_factory=list)


def _cosine_lr(step: int, total: int, cfg: GatesTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    frac = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_frac
                     + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(
                         math.pi * min(1.0, frac))))


def train_gates(model, pairs: Iterable[Sequence[torch.Tensor]], system,
                cache_factory: Callable, cfg: Optional[GatesTrainConfig] = None
                ) -> GatesTrainConfig:
    """The §7 gate fine-tune loop (general similarity pairs, NEVER the
    served corpus — corpus independence is the design).

    pairs: an iterable of (ids_a, ids_b) positive-pair batches of length
    cfg.batch_pairs (token-id tensors (1, T); the DRIVER tokenizes). Each
    step: capture both sides under no_grad (production path), replay
    under grad, InfoNCE, backward, AdamW step on the gate vectors ONLY.

    The model must be loaded with use_m1m2=True (the wiring) — the same
    mem_size as the intended ingestion; freeze_all_but_gates runs first.
    """
    cfg = cfg or GatesTrainConfig()
    inner = getattr(model, "model", model)
    module = inner.m1m2
    trainables = freeze_all_but_gates(model,
                                      include_read_gate=not cfg.freeze_read_gate)
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(
        [p for n, p in module.named_parameters()
         if any(n.endswith(t.split(".")[-1]) for t in trainables)],
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    t0 = time.perf_counter()
    for step, batch in enumerate(pairs):
        if step >= cfg.max_steps:
            break
        if len(batch) < 2:
            raise ValueError(
                f"train_gates: batch {step} holds {len(batch)} pairs — "
                f"the InfoNCE needs >= 2 (in-batch negatives)")
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, cfg.max_steps, cfg)
        states = []
        for ids_a, ids_b in batch:
            pair_states = []
            for ids in (ids_a, ids_b):
                cache = cache_factory()
                calls, m1_i, m2_i = prefill_capture_calls(
                    model, ids, cache, system)
                m1, m2 = replay_states(module, calls, m1_i, m2_i)
                pair_states.append((m1, m2))
            states.append(pair_states)
        m1_a = torch.stack([s[0][0] for s in states])
        m2_a = torch.stack([s[0][1] for s in states])
        m1_b = torch.stack([s[1][0] for s in states])
        m2_b = torch.stack([s[1][1] for s in states])
        loss = infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b, tau=cfg.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in module.parameters() if p.requires_grad],
            cfg.grad_clip)
        opt.step()
        pos, neg = cosine_spread(m1_a, m2_a, m1_b, m2_b)
        cfg.history.append({"step": step, "loss": float(loss.item()),
                            "cos_pos": pos, "cos_neg": neg,
                            "lr": opt.param_groups[0]["lr"]})
        if cfg.log_every and step % cfg.log_every == 0:
            print(f"    [gates] step {step:4d} loss {loss.item():.4f} "
                  f"cos(pos) {pos:+.3f} cos(neg) {neg:+.3f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return cfg
