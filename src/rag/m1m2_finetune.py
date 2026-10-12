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

W19 — THE SADDLE (why the first training run never moved): the P3
zero-init write gates sit at an EXACT saddle of this loss. At g=0
every pair-side replays to the SAME state (m_init + 0 = m_init), and
with the system reset point built at P3 gates m_init is exactly zero:
all pair-states are the zero vector, every cosine is 0, every logit is
equal, and the loss is ln(B) — the observed flat 2.0794 at B=8. The
gradient w.r.t. the gates is then IDENTICALLY zero, in both regimes:
  * zero states: sim = cos(u_a, u_b) with u = 0/||0||_clamped = 0, so
    dsim/dstate = (du/dstate)^T · u_other = 0 exactly (the huge clamped
    1/1e-12 Jacobian multiplies a ZERO vector);
  * identical nonzero states (an open-gate system state): every cosine
    sits AT its maximum (cos(v, v)), whose local gradient is 0, AND the
    cross-entropy row gradient (p - y) sums to exactly 0 over each row,
    so the common-direction component cancels exactly.
The write path's own Jacobian dm/dg = Delta is nonzero (the branch-free
gated add, PROPOSAL P3) — but <dL/dm, Delta> needs dL/dm, and dL/dm = 0
at the saddle. The 50-step flat line on the GPU box is the exact
arithmetic, not a weak signal. THE FIX: `GatesTrainConfig.gate_init`
(0.1 by default) opens both write gates before step 0 — any nonzero
value escapes (the cosine is scale-invariant, so the VALUE only sets
where optimization starts, not whether signal exists); the P3 default
in M1M2.__init__ is untouched (deployment semantics). gate_init=0/None
keeps the loaded values and prints the loud saddle WARNING.

W19 — THE LOSS GEOMETRY (delta-centering): the retrieval side scores
CHUNK-vs-QUERY M1/M2 units in the W16 centered frame (chunk vector
minus the system vector — the common component collapses the absolute
cosine). The trainer now matches that geometry: `delta_loss=True`
(default) computes the InfoNCE over (m_final - m_init) per side — the
same deltas retrieval compares, with the system state's common
direction removed. The absolute form stays available for the A/B
(`GatesTrainConfig(delta_loss=False)` / the driver's --absolute-loss).

W19 — THE SPEED (the 22 s/step): the per-step cost is ~95% the 16
capture prefills (each prefill pushes S/M1/M2 through ~50 online
TurboQuant quants of 2^19-wide units — the torch-reference FHT
butterfly, 19 stages each, d > 32,768 is not CUDA-kernel-eligible);
the replay forward + the W18 backward are ~0.5 s combined. THE FIX:
`GatesTrainConfig.refresh_every` (4 by default) — the capture pool:
ONE batch of pairs is prefilled, then trained on for refresh_every
GRADIENT STEPS (the replay re-runs every step, so each gradient is
taken at the CURRENT gates; only the captured k/v are frozen), and
the prefill count drops refresh_every-fold. The k/v are DETACHED
constants either way (the W18 gradient contract is unchanged —
gradients remain exact w.r.t. the calls in hand); the only
approximation is capture STALENESS: the k/v depend on the gates
through the read-path feedback, and the gates move <= lr per step,
so a retained batch is at most (refresh_every - 1) optimizer steps
stale. Set --refresh-every 1 for the exact-recapture regime (the W17
semantics, one gradient step per fresh capture). Phase timers
(capture/replay/backward/step) and the gate-gradient magnitudes ride
in every history entry — the box can now SEE where the seconds go.

W20.11 — THE LOADER PINNING (transport, not math): the capture loop's
host->device path is pinned — the driver hands (pinned) HOST token-id
tensors, and each pool refresh stages its prefill inputs on the
capture device with a non_blocking H2D. The staging window IS the
refresh cycle: the NEXT pool's ids stage while the current pool
trains, resident on the device before the next prefill wave needs
them. Zero semantics change — the values are bit-identical (pinning
moves bytes, not math); the W18 exact-gradient and W19 pool contracts
are untouched, and the training step's data path (replay over the
captured k/v) sees nothing new.

W20.14 — THE METRIC BUG (the box's "spread 0.0000 at every step"): the
held-out discrimination metric reported the "negatives" as the TRUE-pair
cosine vector ROLLED by one — but a roll is a permutation, and a
permutation's mean is identical, so cos(pos) == cos(neg) EXACTLY at
every step, for ANY model, by construction (the box's 0.615/0.615 ...
0.4801/0.4801 table was this identity, never a learning failure; the
loss itself was always correct — the full B x B matrix with diagonal
targets — and the below-ln(B) losses the box logged were real ranking
signal it could not see). cosine_spread now measures the negatives the
InfoNCE actually trains against — the OFF-DIAGONAL mean of the same
sim matrix — and the per-step history carries spread/off_max/diag_min
plus the pool phase (the refresh-cycle position that explains the loss
oscillation), all shared through _pair_sim_stats. The H6 investigation
knob (GatesTrainConfig.train_layers) freezes every write gate outside
a layer subset (zeroed grads; the default weight_decay=0 keeps them
bit-anchored) — the "isolate single layer" experiment without code
changes on the box.

W21 — THE CEILING, ANSWERED (the architecture, not the regime): the
box's post-W20.15 run trained clean mechanics (sawtooth decomposed,
coverage tallied, gates moved to k[-0.087..+1.122]) yet the spread
still plateaued ~0.12-0.13 with cos_neg PARKED at ~+0.36 — random
cross pairs at +0.36 cosine is a representation-ANISOTROPY pedestal
(the keys' shared direction; Su et al. 2021's centering fix), and the
residual signal is the position lock: the v1 write addressed slots by
POSITION (slot = p mod mem_size), so the state cosine is a
position-aligned key correlation — surface overlap, not content. The
fix lives in m1m2.py W21 (content-addressed slots via simhash of the
cENTERED key + per-(layer, head) write gates); THIS module's changes
are the trainer-side consequences only: the W18 backward accumulates
per-HEAD gate rows ((L, H) — one inner product per head), and the
sim-matrix construction moved to ONE public site (pair_sim_matrix —
the loss, the stats, the bucketing, and the driver's held-out ladder
all read the same matrix; the W20.14 rule, now shared code). The
loss/metric SEMANTICS are unchanged.

W20.15 — THE REGIME (what the fixed metric revealed on the box's
300-step re-run): a SAWTOOTH. The between-capture-block loss variance
measured 7,595x the within-block variance (the block difficulty IS
the batch — its 8 pairs' own cross-side alignment; the background
cos_neg is stable while cos_pos swings 0.35..0.74), the pre-clip
|gv| swung 160x with the batch, only 600 of 14,469 train pairs were
consumed (4.1% — one pass never completed), and the spread plateaued
at ~0.12-0.13. Three instruments + one regime fix landed:
  * epoch_batches — per-epoch RE-BATCHING: the v1 cycle re-yielded the
    SAME file-order slices forever, so a pair's in-batch negatives
    were fixed by the file layout for the whole run; every epoch now
    draws fresh negative sets (deterministic per seed+epoch).
  * the gate VALUE stats (gk/gv min/max/absmean) + the capture batch
    ordinal ride every history entry — 300 steps of |g| said nothing
    about WHERE the gates actually were.
  * lexical_overlap + bucketed_spread — the held-out gap bucketed by
    the pair's SURFACE overlap: the write is position-addressed
    (slot = p mod mem_size, ONE token per slot at T < mem_size, so the
    state delta IS the key sequence and state-cos = position-aligned
    key correlation); the bucket report makes the predicted ceiling
    (high-overlap pairs separate, low-overlap pairs do not) measurable
    instead of arguable. The label-0 rows (17,519 curated non-pairs,
    unused by the v1 loader) become the driver's CONTROL LINE.

W22 — THE HARD NEGATIVES (the proven small-batch fix): the W21 box
run trained REAL signal for the first time (spread +0.20..0.35, no
plateau) but with B=8 the InfoNCE's negative pool is SEVEN in-batch
cross pairs — random draws whose difficulty swings the loss 0.28..1.64
between blocks (the W20.15 sawtooth's residue), while the 17.5k
curated label-0 non-pairs ride along UNUSED. Every reference design
at this batch scale mines curated/hard negatives into the candidate
set (DPR's in-batch + hard negatives, Karpukhin et al. 2020, arXiv:
2004.04906; MoCo's queue of shared negatives, He et al. 2020, arXiv:
1911.05722; Robinson et al. 2021, arXiv:2011.12031 — the contrastive
shortcut-solution analysis). `train_gates(negatives=...)` +
GatesTrainConfig.neg_pairs now attach N curated negative SIDES to
every capture pool: the loss logits gain N extra columns —
sim(a_i, n_l) never a target — so each anchor must rank its OWN pair
above the curated non-pairs too. The negatives are replayed under
grad like any side (their gates receive the push-away gradient), the
metric gains cos_hneg (the curated-negative column mean — the same
number the driver's label-0 control line reports on held-out data),
and neg_pairs=0 (the library default) is bit-identical to the W21
regime (the driver's --neg-pairs 8 is the recommended run).

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
import random
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
    "cosine_spread", "epoch_batches", "lexical_overlap",
    "bucketed_spread", "train_gates",
    "flat_unit", "pair_sim_matrix", "sim_matrix_stats",
    "bucketed_spread_from_sim",
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

    W27 — the WRITE KERNEL REGIME IS BAKED IN (write_addr, and
    center_write/addr_blocks when the module carries them): every
    artifact produced by this function is self-describing about its
    addressing regime, so load_gates can (a) refuse artifacts that
    predate W21 content addressing (no meta_write_addr — the pre-W21
    (L,) era, and any hand-broadcast forgery of it) and (b) refuse
    regime-mismatched loads. The finetune driver ALSO passes the regime
    in extra_meta (W21); the baked copy is authoritative and identical.
    """
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path = path + ".npz"
    arrays = {k: getattr(module, k).detach().cpu().numpy().astype(np.float64)
              for k in _GATE_KEYS}
    meta = _module_geometry(module)
    for regime_key in ("write_addr", "center_write", "addr_blocks"):
        if hasattr(module, regime_key):
            meta[regime_key] = getattr(module, regime_key)
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
    # W21 — the addressing regime must match (validated BEFORE any
    # parameter mutates): a content-trained artifact loaded into a
    # positional module (or vice versa) would silently score the wrong
    # kernel; the shapes alone cannot catch it.
    if ("write_addr" in file_meta
            and hasattr(module, "write_addr")
            and str(file_meta["write_addr"]) != str(module.write_addr)):
        raise ValueError(
            f"load_gates: {path} was trained at write_addr="
            f"{file_meta['write_addr']!r}, this module runs "
            f"{module.write_addr!r} — the addressing regimes disagree; "
            f"refusing a silent kernel mismatch")
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
                f"{want_shape} — the gate-vector geometry disagrees "
                f"(a pre-W21 (L,) artifact is meaningless under content "
                f"addressing; retrain)")
    # W27.2 — THE REGIME META IS MANDATORY on a content-addressing module
    # (has write_addr): save_gates has baked meta_write_addr in since W27,
    # and the W21+ finetune driver passed it in extra_meta before that.
    # An artifact WITHOUT it either predates W21 content addressing or was
    # hand-assembled from one — the box's hermes session built exactly
    # such a file (a pre-W21 (24,) vector np.repeat-ed to (24, 32)) and
    # it slipped past both the shape and regime checks above while
    # carrying zero per-head addressing signal.
    if hasattr(module, "write_addr") and "write_addr" not in file_meta:
        raise ValueError(
            f"load_gates: {path} carries no meta_write_addr — the "
            f"artifact predates W21 content addressing (or was "
            f"hand-assembled from one); its gate semantics cannot address "
            f"this module's kernel regime; retrain with the W21+ "
            f"finetune driver (scripts/gpu/finetune_m1m2.py)")
    # W27.2 — the secondary regime keys, when BOTH sides carry them
    # (mismatch = a kernel the artifact was not trained under)
    for regime_key in ("center_write", "addr_blocks"):
        if (regime_key in file_meta and hasattr(module, regime_key)
                and str(file_meta[regime_key]) != str(
                    getattr(module, regime_key))):
            raise ValueError(
                f"load_gates: {path} was trained at {regime_key}="
                f"{file_meta[regime_key]!r}, this module runs "
                f"{getattr(module, regime_key)!r} — write-kernel regime "
                f"drift; refusing a silent mismatch")
    # W27.2 — THE BROADCAST SIGNATURE: a genuine (L, H) training run never
    # produces a write_gate_k whose rows are CONSTANT across every head
    # (measure-zero under SGD); a hand-broadcast pre-W21 (L,) artifact is
    # exactly that. Only write_gate_k is checked — write_gate_v is
    # legitimately uniform in tests/calibration (fill_(c) is a real
    # pattern), and the KEY gate is the one whose per-head structure IS
    # the content addressing signal.
    a_k = arrays["write_gate_k"]
    if (a_k.ndim == 2 and a_k.shape[1] > 1
            and bool(np.all(a_k == a_k[:, :1]))):
        raise ValueError(
            f"load_gates: {path} write_gate_k rows are constant across "
            f"all {a_k.shape[1]} heads — the broadcast signature of a "
            f"pre-W21 (L,) artifact reshaped to (L, H); it carries no "
            f"per-head addressing signal (the box's "
            f"m1m2_gates_trained_fixed.npz was exactly this and produced "
            f"garbage); retrain under content addressing")
    for k in _GATE_KEYS:
        param = getattr(module, k, None)
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
    # caught exactly this). W18: the read follows the CAPTURE device — a
    # codes-only fresh cache carries no device signal (no write yet), so
    # the dequant may land on CPU while the recorded k/v (and the model)
    # live on CUDA; the replay must run where the capture ran.
    dev = input_ids.device
    m1_init = cache.read_m1(dtype=torch.float32)
    m2_init = cache.read_m2(dtype=torch.float32)
    if m1_init is None:
        m1_init = module.init_state(dtype=torch.float32, device=dev)
    else:
        m1_init = m1_init.detach().to(torch.float32).to(dev)
    if m2_init is None:
        m2_init = module.init_state(dtype=torch.float32, device=dev)
    else:
        m2_init = m2_init.detach().to(torch.float32).to(dev)
    recorder = M1M2CallRecorder(module)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache,
                  use_cache=True)
    finally:
        recorder.remove()
    return recorder.detach_calls(), m1_init, m2_init


# -------------------------------------------------------------- replay ----
class _GatedMemoryReplay(torch.autograd.Function):
    """The W18 memory-safe replay: the sequential module.write chain over
    the recorded calls, with EXACT gradients and O(state + captured k/v)
    retained memory.

    WHY (the A10G OOM, root-caused): the plain autograd chain retains a
    boundary state PER CALL — at batch_pairs=8 x 2 sides x 24 calls and
    fp32 (32, mem_size, 128) states that is 384 live state tensors
    (~13 GiB at mem_size=1024, ~52 GiB at 4096) BEFORE the full-attn
    prefill activations — the checkpointed variant retains the same 24
    boundary states as saved segment inputs. The architectural fact the
    chain hides: the additive write (spec §2.2) is LINEAR in the gates,
    so in the fp32 replay regime

        m_final = m_init + sum_i g_{L(i)} · Δ_i,   Δ_i = scatter(k_i, v_i)

    holds with NO per-layer state retention — the exact Jacobian is
      ∂m_final/∂m_init = I   (the chain is pure adds),
      ∂m_final/∂g_L  = Δ_i   (per call, state-independent),
    so backward recomputes each Δ_i FROM THE RECORDED k/v ONE AT A TIME
    (module.write_deltas — the SAME _scatter_delta code the forward's
    write uses, bit-identical by construction) and accumulates
    <grad_out, Δ_i> into the two gate vectors. The forward itself runs
    the module.write chain in op order — bit-identical to the plain
    chain (pinned by test_w18_finetune gates).

    EXACTNESS CONTRACT: requires the fp32 replay regime (m_init/states
    fp32) — at fp16 states the per-layer cast-backs make the Jacobian
    non-identity and this Function's gradients would be inexact; the
    trainer always replays fp32 (read_m1(dtype=float32)); the plain
    chain stays available for reference (replay_states(memopt=False)).
    """

    @staticmethod
    def forward(ctx, m1_init, m2_init, write_gate_k, write_gate_v,
                module, calls):
        with torch.no_grad():
            m1, m2 = m1_init, m2_init
            for (k, v, layer_idx, positions) in calls:
                m1, m2 = module.write(k, v, m1, m2, layer_idx,
                                      positions=positions)
        ctx.module = module
        ctx.calls = calls
        ctx.save_for_backward(write_gate_k, write_gate_v)
        return m1, m2

    @staticmethod
    def backward(ctx, grad_m1, grad_m2):
        module, calls = ctx.module, ctx.calls
        write_gate_k, write_gate_v = ctx.saved_tensors
        gk = (torch.zeros_like(write_gate_k)
              if ctx.needs_input_grad[2] else None)
        gv = (torch.zeros_like(write_gate_v)
              if ctx.needs_input_grad[3] else None)
        with torch.no_grad():
            for (k, v, layer_idx, positions) in calls:
                d1, d2 = module.write_deltas(
                    k, v, layer_idx, positions=positions,
                    state=grad_m1)
                # W21: the gates are per-(layer, head) — one inner
                # product per head (sum over the slot/dim axes), the
                # exact Jacobian row for the (L, H) gate vectors.
                if gk is not None:
                    gk[layer_idx] += (grad_m1 * d1).sum(dim=(1, 2))
                if gv is not None:
                    gv[layer_idx] += (grad_m2 * d2).sum(dim=(1, 2))
        # the chain is pure adds: the identity gradient for the inits
        grad_m1_init = grad_m1 if ctx.needs_input_grad[0] else None
        grad_m2_init = grad_m2 if ctx.needs_input_grad[1] else None
        return grad_m1_init, grad_m2_init, gk, gv, None, None


def replay_states(module, calls: Sequence, m1_init: torch.Tensor,
                  m2_init: torch.Tensor,
                  memopt: Optional[bool] = None,
                  checkpoint: Optional[bool] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-run the recorded wiring calls under GRAD through the module's
    own `write` — the gate-differentiable state chain.

    state <- state + g_L * scatter(k_L) per call, starting from m_init.
    The additive write makes the result m_init + sum_L g_L*Delta_L —
    exactly the cache's held state up to the inter-layer quantization
    round-trips (the deployed state is the quantized version of this
    sum; the replay is the noise-free proxy the loss trains on).

    memopt (W18, the OOM fix): None = auto (the exact-gradient
    _GatedMemoryReplay Function — O(state + captured k/v) memory, no
    24-boundary-state chain; bit-identical forward, exact gradients in
    the fp32 regime); True = force the Function (loud ValueError on
    non-fp32 states — the exactness contract); False = the plain
    autograd chain (the reference path the CPU parity gates pin).
    `checkpoint` (the W17 torch.utils.checkpoint variant) is retained
    for the A/B only — it retains the boundary states as saved segment
    inputs and is NOT the memory fix (deprecated default).
    """
    if memopt is None:
        # an explicit W17-style `checkpoint=` request (True or False)
        # selects the legacy chain — the pinned reference semantics; the
        # unadorned call (what train_gates makes) takes the W18 default.
        memopt = checkpoint is None
    if memopt:
        for name, t in (("m1_init", m1_init), ("m2_init", m2_init)):
            if t.dtype not in (torch.float32, torch.float64):
                raise ValueError(
                    f"replay_states(memopt=True): {name} has dtype "
                    f"{t.dtype} — the exact-gradient replay requires the "
                    f"fp32 regime (the per-layer fp16 cast-backs would "
                    f"break the identity Jacobian; use memopt=False for "
                    f"the reference chain)")
        return _GatedMemoryReplay.apply(
            m1_init, m2_init, module.write_gate_k, module.write_gate_v,
            module, list(calls))
    m1, m2 = m1_init, m2_init
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
def flat_unit(x: torch.Tensor) -> torch.Tensor:
    """(B, ...) -> (B, dims) fp32, L2-normalized (the cosine operand).

    W21: public (was _flat_unit) — the driver's held-out ladder streams
    per-side units through the same normalization the loss uses."""
    b = x.shape[0]
    v = x.reshape(b, -1).to(torch.float32)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v


# W20.14 kept a private alias so internal callers stay stable:
_flat_unit = flat_unit


def pair_sim_matrix(m1_a: torch.Tensor, m2_a: torch.Tensor,
                    m1_b: torch.Tensor, m2_b: torch.Tensor
                    ) -> torch.Tensor:
    """sim(i, j) = 0.5 * (cos(m1_a[i], m1_b[j]) + cos(m2_a[i],
    m2_b[j])) — the B x B matrix EVERY reader scores: the InfoNCE loss,
    the history stats, the bucketing, and the driver's held-out ladder
    (W21: ONE construction site, shared code — the W20.14 rule that
    the metric and the loss see the same discrimination, now enforced
    by construction rather than by convention). Operands may be any
    (B, ...) tensors of equal B; grad flows as the callers need (the
    loss trains through this; the stats read it under no_grad)."""
    return 0.5 * (flat_unit(m1_a) @ flat_unit(m1_b).T
                 + flat_unit(m2_a) @ flat_unit(m2_b).T)


def infonce_pairs_loss(m1_a: torch.Tensor, m2_a: torch.Tensor,
                       m1_b: torch.Tensor, m2_b: torch.Tensor,
                       tau: float = 0.07, m1_n: Optional[torch.Tensor] = None,
                       m2_n: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
    """InfoNCE with in-batch negatives (+ W22 curated hard-negative
    columns) over the pair states.

    sim(i, j) = 0.5 * (cos(m1_a[i], m1_b[j]) + cos(m2_a[i], m2_b[j]));
    logits = sim / tau; the diagonal (the true pairs) are the targets.
    B >= 2 required (in-batch negatives); the matrix is symmetric so the
    a->b direction covers both.

    W22 — `m1_n`/`m2_n` (N curated non-pair side-states, stacked like
    the b-operands) append N negative COLUMNS: sim(a_i, n_l) is never
    a target, so every anchor must ALSO rank its own pair above the
    curated non-pairs (DPR/MoCo's shared hard negatives; the box's B=8
    in-batch pool is 7 random draws — the loss's dominant noise). The
    columns flow gradient into the negatives' replays (push-away).
    Both must be given together, N >= 1, and B/N operands share the
    flattened dims (any (N, ...) states — flat_unit normalizes).
    """
    if tau <= 0 or not isinstance(tau, float):
        raise ValueError(f"infonce_pairs_loss: tau must be > 0, got {tau!r}")
    b = m1_a.shape[0]
    if b < 2:
        raise ValueError(
            f"infonce_pairs_loss: batch of {b} pair has no in-batch "
            f"negatives — use B >= 2")
    sim = pair_sim_matrix(m1_a, m2_a, m1_b, m2_b)
    if (m1_n is None) != (m2_n is None):
        raise ValueError(
            "infonce_pairs_loss: m1_n and m2_n are a pair — give both "
            f"(got {'m1_n' if m2_n is None else 'm2_n'} only)")
    if m1_n is not None:
        n = m1_n.shape[0]
        if n < 1:
            raise ValueError(
                f"infonce_pairs_loss: {n} curated negative column(s) — "
                f"at least 1 (or None to disable)")
        if m2_n.shape[0] != n:
            raise ValueError(
                f"infonce_pairs_loss: m1_n has {n} rows but m2_n has "
                f"{m2_n.shape[0]} — the two memories' negatives must "
                f"share the side list")
        # the SAME one construction site (the W20.14 rule): the
        # negative columns are pair_sim_matrix with the b-operands
        # swapped for the curated sides.
        neg = pair_sim_matrix(m1_a, m2_a, m1_n, m2_n)     # (B, N)
        sim = torch.cat([sim, neg], dim=1)
    logits = sim / tau
    targets = torch.arange(b, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def sim_matrix_stats(sim: torch.Tensor) -> Dict[str, float]:
    """The B x B stats from a PRECOMPUTED pair_sim_matrix (W21: the
    driver's ladder passes the streamed matrix; call under no_grad for
    metric use):

      cos_pos   the DIAGONAL mean — the true pairs,
      cos_neg   the OFF-DIAGONAL mean — the in-batch negatives,
      spread    cos_pos - cos_neg — the discrimination signal,
      off_max   the hardest negative (the tightest false pair),
      diag_min  the weakest true pair.

    B < 2 (no in-batch negatives) returns NaN for the negative-side
    fields — the same loudness the loss itself has."""
    diag = torch.diagonal(sim)
    b = sim.shape[0]
    if b < 2:
        return {"cos_pos": float(diag.mean()), "cos_neg": float("nan"),
                "spread": float("nan"), "off_max": float("nan"),
                "diag_min": float(diag.min())}
    off = sim[~torch.eye(b, dtype=torch.bool, device=sim.device)]
    return {"cos_pos": float(diag.mean()), "cos_neg": float(off.mean()),
            "spread": float(diag.mean() - off.mean()),
            "off_max": float(off.max()), "diag_min": float(diag.min())}


def _pair_sim_stats(m1_a, m2_a, m1_b, m2_b, m1_n=None, m2_n=None
                    ) -> Dict[str, float]:
    """W20.14 — the B x B pair-similarity stats the history and the
    held-out report share (no grad; sim_matrix_stats over the shared
    pair_sim_matrix — EXACTLY the matrix infonce_pairs_loss trains
    on, so the metric scores the same discrimination the loss
    optimizes).

    W22 — `m1_n`/`m2_n` (the curated negative sides, when the pool
    carries them) add `cos_hneg`: the mean sim(a_i, n_l) over the
    hard-negative columns the loss ALSO trains against (the same
    number the driver's label-0 control line reports held-out). NaN
    when no negatives ride the pool (the W21 regime)."""
    with torch.no_grad():
        st = sim_matrix_stats(pair_sim_matrix(m1_a, m2_a, m1_b, m2_b))
        if m1_n is not None and m2_n is not None \
                and m1_n.shape[0] > 0 and m2_n.shape[0] > 0:
            neg = pair_sim_matrix(m1_a, m2_a, m1_n, m2_n)  # (B, N)
            st["cos_hneg"] = float(neg.mean())
        else:
            st["cos_hneg"] = float("nan")
        return st


def cosine_spread(m1_a, m2_a, m1_b, m2_b) -> Tuple[float, float]:
    """(mean cos over the TRUE pairs, mean cos over the IN-BATCH
    negatives) — the held-out discrimination metric (no grad; the gap
    is the signal).

    W20.14 — THE ROLL BUG (the box's "spread 0.0000 at every step"):
    v1 built the (B,) true-pair cosine vector and reported the
    "negatives" as that SAME vector rolled by one — but a roll is a
    permutation and a permutation's mean is IDENTICAL, so cos(pos) ==
    cos(neg) EXACTLY at every step, for ANY model, by construction
    (the box's 0.615/0.615 ... 0.4801/0.4801 table was this identity,
    never a learning failure). The metric now returns the off-diagonal
    mean of the same B x B sim matrix the InfoNCE builds — the exact
    negatives the loss trains against (see _pair_sim_stats)."""
    s = _pair_sim_stats(m1_a, m2_a, m1_b, m2_b)
    return s["cos_pos"], s["cos_neg"]


# ------------------------------------------------- W20.15 the regime -----
def epoch_batches(pairs: Sequence, batch: int, seed: int = 1234):
    """W20.15 — the per-epoch RE-BATCHING generator (the trainer's
    `pairs` iterator: yields lists of `batch` (ids_a, ids_b) tuples,
    forever).

    WHY (the box's 300-step run): the v1 `_cycle_batches` walked the
    pair list in ONE fixed order forever, so a pair's in-batch
    negatives — the InfoNCE's ENTIRE negative signal — were fixed by
    the file layout for the whole run; 300 steps at B=8/refresh 4
    consumed 600 of 14,469 train pairs (4.1%, never one full epoch),
    and longer runs would have looped the SAME 75 batches' neighbor
    sets indefinitely. Every epoch here re-shuffles with
    Random(seed*100003 + epoch) and partitions into fresh batches —
    new negative combinations per epoch, deterministic and
    reproducible run to run.

    THE TAIL: the last slice of an epoch wraps with the epoch's FIRST
    pairs so every batch holds exactly `batch` pairs (the loss's own
    B >= 2 contract). The wrap is DISJOINT from the tail slice (the
    wrap takes indices [0, batch - tail_len), which are all < the
    last slice's start whenever batch < n), so no pair appears TWICE
    in ONE batch — the duplicated members only straddle two batches
    of the same epoch (harmless: a batch's negatives are its other
    members, which remain distinct pairs). n <= batch yields the
    whole shuffled set as ONE batch (nothing to wrap).

    W21.6 — the refusals are EAGER: this is a plain function that
    validates (batch, the pair count) and hands back the generator,
    so `epoch_batches(pairs, 1)` raises AT THE CALL — the W21.0 box
    run caught the v1 shape, a bare `yield` function, deferring the
    body (and the raise) to the first `next()`, which made the
    refusal unobservable at the call site (test_w20_15b: DID NOT
    RAISE). Iteration behavior is unchanged for every well-formed
    caller (the driver, the tests, the trainer).
    """
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 2:
        raise ValueError(
            f"epoch_batches: batch must be an int >= 2 (the InfoNCE "
            f"needs in-batch negatives), got {batch!r}")
    items = list(pairs)
    n = len(items)
    if n < 2:
        raise ValueError(
            f"epoch_batches: {n} pair cannot form a batch — the "
            f"InfoNCE needs >= 2 (in-batch negatives)")
    return _epoch_batches_gen(items, batch, seed)


def _epoch_batches_gen(items: list, batch: int, seed: int):
    """epoch_batches' engine (validated inputs; the forever loop)."""
    n = len(items)
    epoch = 0
    while True:
        order = list(range(n))
        random.Random(seed * 100003 + epoch).shuffle(order)
        if n <= batch:
            yield [items[i] for i in order]
            epoch += 1
            continue
        for s in range(0, n, batch):
            take = order[s:s + batch]
            if len(take) < batch:
                take = take + order[:batch - len(take)]
            yield [items[i] for i in take]
        epoch += 1


def lexical_overlap(text_a: str, text_b: str) -> float:
    """W20.15 — the pair's SURFACE alignment: the word-multiset
    containment |A ∩ B| / min(|A|, |B|) over lowercased whitespace
    words (1.0 = the shorter text is lexically contained in the
    longer; 0.0 = disjoint vocabularies).

    THE DIAGNOSSTIC ROLE: the M1/M2 write is position-addressed
    (slot = p mod mem_size) and at T < mem_size every slot holds
    EXACTLY ONE token, so a pair-side's state delta IS its key/value
    sequence and the state cosine is the POSITION-ALIGNED key
    correlation — a pair separates only if its two sides carry
    similar tokens at the same positions (surface alignment), no
    matter the semantics. Bucketing the held-out gap by this overlap
    makes that ceiling MEASURABLE: a gap that lives only in the
    high-overlap bucket is the position-lock signature; a flat bucket
    profile means the signal is not surface-locked and the question
    moves to capacity/optimization."""
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        raise TypeError(
            f"lexical_overlap: both members must be str, got "
            f"{type(text_a).__name__} / {type(text_b).__name__}")
    from collections import Counter
    ca = Counter(text_a.lower().split())
    cb = Counter(text_b.lower().split())
    inter = sum((ca & cb).values())
    denom = min(sum(ca.values()), sum(cb.values()))
    return float(inter) / denom if denom else 0.0


def bucketed_spread_from_sim(sim: torch.Tensor,
                             buckets: Sequence[str]) -> Dict[str, dict]:
    """W21 — bucketed_spread over a PRECOMPUTED pair_sim_matrix (the
    driver's ladder path): the overall row plus, per bucket label, the
    members' mean TRUE-pair cosine (the diagonal) and their gap
    against the OVERALL off-diagonal mean (the shared
    in-batch-negative background). buckets[i] labels pair i
    (len(buckets) == B required). The report reads: bucket cos_pos -
    overall cos_neg. A high-overlap bucket at gap ~ +0.3 with a
    low-overlap bucket at ~ 0.0 is the position-locked-write
    signature (see lexical_overlap); a flat profile is the negative
    result that redirects the investigation. Labels are reported
    verbatim — the caller owns the bucketing rule."""
    b = int(sim.shape[0])
    if len(buckets) != b:
        raise ValueError(
            f"bucketed_spread_from_sim: {len(buckets)} bucket labels "
            f"for {b} pairs — one label per pair, aligned with the sim "
            f"matrix")
    diag = torch.diagonal(sim)
    if b >= 2:
        off = sim[~torch.eye(b, dtype=torch.bool, device=sim.device)]
        cos_neg = float(off.mean())
    else:
        off = diag
        cos_neg = float("nan")
    diag_list = [float(x) for x in diag]
    out: Dict[str, Dict[str, float]] = {
        "overall": {
            "cos_pos": float(diag.mean()), "cos_neg": cos_neg,
            "spread": float(diag.mean() - off.mean()) if b >= 2
            else float("nan"),
            "off_max": float(off.max()) if b >= 2 else float("nan"),
            "diag_min": float(diag.min()),
        },
        "buckets": {},
    }
    for label in dict.fromkeys(buckets):          # first-seen order
        members = [diag_list[i] for i, x in enumerate(buckets) if x == label]
        pos = sum(members) / len(members)
        out["buckets"][label] = {
            "n": len(members), "cos_pos": pos,
            "gap": pos - cos_neg,
        }
    return out


def bucketed_spread(m1_a, m2_a, m1_b, m2_b,
                    buckets: Sequence[str]) -> Dict[str, dict]:
    """W20.15 — the held-out stats BUCKETED per pair (tensor operands;
    the ladder's from-sim twin is bucketed_spread_from_sim — the same
    numbers over a shared pair_sim_matrix)."""
    b = int(m1_a.shape[0])
    if len(buckets) != b:
        raise ValueError(
            f"bucketed_spread: {len(buckets)} bucket labels for {b} "
            f"pairs — one label per pair, aligned with the stacked "
            f"sides")
    with torch.no_grad():
        return bucketed_spread_from_sim(
            pair_sim_matrix(m1_a, m2_a, m1_b, m2_b), buckets)


# ------------------------------------------------------------- the loop ---
@dataclass
class GatesTrainConfig:
    lr: float = 1e-2                # gates are 3 small vectors — a hot lr
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
    memopt: Optional[bool] = None   # W18: None = the exact-gradient replay
    # W19 — the saddle escape: nonzero fill of both write gates before
    # step 0 (0.1 default; 0/None = keep the loaded values — at P3 zero
    # that is the PROVABLY dead exact saddle, and the loop warns loudly)
    gate_init: Optional[float] = 0.1
    # W19 — the loss geometry: True = InfoNCE over (m - m_init) per side
    # (the W16 centered retrieval frame); False = the absolute states
    delta_loss: bool = True
    # W19 — the capture pool: ONE batch of pairs prefilled, trained on
    # for this many GRADIENT STEPS (1 = fresh capture every step, the
    # W17 exact regime; the replay re-runs per step so gradients sit at
    # the current gates — only the k/v are frozen, stale by <= lr x N).
    # W20.11 — the regimes: 4 (the default) is quality-first; 8 is the
    # THROUGHPUT regime, a first-class A/B row — the staleness bound is
    # lr x (N - 1) = lr x 7 at refresh 8 (the tight form of the N bound
    # above: the captured k/v trail the gates by at most 7 optimizer
    # steps; the prefill count halves again vs 4). The operator's call
    # on the box — A/B it and watch the loss/cos separation.
    refresh_every: int = 4
    # W22 — the curated hard negatives: how many label-0 single sides
    # ride EACH capture pool as extra InfoNCE columns (shared across
    # the batch — DPR/MoCo's design; the box's B=8 in-batch pool is 7
    # random draws whose difficulty swings the loss block-to-block).
    # 0 (the default) = the exact W21 regime (bit-identical); the
    # driver's recommended run passes 8. The `negatives` iterable to
    # train_gates yields batches of single-side ids (host tensors);
    # exhaustion falls back to the plain regime for that pool (one
    # loud warning, never a crash).
    neg_pairs: int = 0
    # W20.14 — the H6 investigation knob: train the write gates for ONLY
    # these layer indices (None = all, the default). Layers outside the
    # set get their gate gradients ZEROED after backward — with the
    # default weight_decay=0.0 they stay bit-anchored at their fill
    # value (the "isolate single layer / layer subset" experiment the
    # box handover asks for, no code changes needed there). The box's
    # per-layer question ("do per-layer gates help, or does the shared
    # signal average out?") is now one CLI flag per run.
    train_layers: Optional[Sequence[int]] = None
    history: List[Dict[str, float]] = field(default_factory=list)


def _cosine_lr(step: int, total: int, cfg: GatesTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    frac = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_frac
                     + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(
                         math.pi * min(1.0, frac))))


def _calls_bytes(calls) -> int:
    """The retained bytes of a captured call list (k/v + positions)."""
    n = 0
    for (k, v, _layer, pos) in calls:
        n += k.numel() * k.element_size() + v.numel() * v.element_size()
        if pos is not None:
            n += pos.numel() * pos.element_size()
    return n


def _grad_absmax(param) -> float:
    """|grad|.max of a (possibly None / zero) parameter — the saddle
    detector (0.0 = the gradient path is dead; the W19 history field)."""
    if param is None or param.grad is None:
        return 0.0
    return float(param.grad.detach().abs().max())


def _gate_value_stats(module) -> Dict[str, float]:
    """W20.15 — WHERE THE GATES ARE (not just their gradients): min /
    max / absmean of both write-gate vectors (W21: (L, H) per-head
    rows — the stats flatten; per-head locations ride the npz). The
    box's 300-step run logged 300 |g| absmax values and ZERO parameter
    locations — the gates could have gone anywhere (negative =
    subtractive writes are legal and unbounded) and the history could
    not say where they ended; these fields pin it per step,
    post-update."""
    gk = module.write_gate_k.detach().float()
    gv = module.write_gate_v.detach().float()
    return {"gk_min": float(gk.min()), "gk_max": float(gk.max()),
            "gk_absmean": float(gk.abs().mean()),
            "gv_min": float(gv.min()), "gv_max": float(gv.max()),
            "gv_absmean": float(gv.abs().mean())}


def _stage_ids(ids, dev):
    """W20.11.2 — the loader pinning: one prefill input onto the capture
device (non_blocking H2D — a pinned host source rides async DMA;
bit-identical values, transport only). An id tensor already on `dev`
passes through unchanged (.to() no-op); a NON-tensor id (the W19 test
shim's None placeholders) passes through verbatim — the capture call
sees exactly what it saw before."""
    if isinstance(ids, torch.Tensor):
        return ids.to(dev, non_blocking=True)
    return ids


def _stage_prefill_ids(batch, dev):
    """W20.11.2 — the loader pinning: pre-stage a pair batch's prefill
inputs on the capture device (per pool refresh — the staging window
is the refresh cycle; see train_gates). Values bit-identical."""
    staged = []
    for ids_a, ids_b in batch:
        staged.append((_stage_ids(ids_a, dev), _stage_ids(ids_b, dev)))
    return staged


def train_gates(model, pairs: Iterable[Sequence[torch.Tensor]], system,
                cache_factory: Callable,
                cfg: Optional[GatesTrainConfig] = None,
                negatives: Optional[Iterable[Sequence[torch.Tensor]]] = None
                ) -> GatesTrainConfig:
    """The §7 gate fine-tune loop (general similarity pairs, NEVER the
    served corpus — corpus independence is the design).

    pairs: an iterable of (ids_a, ids_b) positive-pair batches of length
    cfg.batch_pairs (token-id tensors (1, T); the DRIVER tokenizes).

    W22 — negatives: an iterable of SINGLE-side id batches of length
    cfg.neg_pairs (the driver's curated label-0 sides, the same seeded
    epoch-rebatch regime). Each capture pool attaches them as shared
    InfoNCE hard-negative columns (see infonce_pairs_loss); the
    negatives' replays are grad-carrying sides like any other, so their
    push-away gradient lands in the SAME (L, H) gates. neg_pairs=0 or
    negatives=None is the exact W21 regime (bit-identical).

    W19 step anatomy: a CAPTURE POOL — one batch of pairs prefilled under
    no_grad (the production path), then trained on for
    cfg.refresh_every gradient steps (the replay re-runs each step, so
current-gate gradients; frozen k/v) — capture/replay/backward phase
times and the gate-gradient magnitudes ride every history entry (the
22 s/step black box is now instrumented). gate_init opens the write
gates off the P3 exact saddle BEFORE step 0; delta_loss scores the
states the centered retrieval frame scores (m - m_init).

W20.14 — the discrimination is finally VISIBLE: every history entry
carries the _pair_sim_stats fields (cos_pos/cos_neg/spread/off_max/
diag_min — read off the same B x B matrix the loss trains on; the v1
cos(neg) was the rolled true-pair mean, identically equal to cos(pos),
so the box's 300-step run could never see a spread) plus the pool
phase (steps_on_batch — the refresh-cycle position that makes the
loss oscillate on the log grid: log_every=25 against refresh_every=4
aliases the pool cycle, an artifact, not non-convergence).

W20.15 — the entry also carries the capture-batch ORDINAL ("batch" —
the block id the loss sawtooths on; the post-hoc analysis groups by
it) and the gate VALUE stats (gk/gv min/max/absmean, read post-update:
WHERE the gates are, not just how hard they are pushed — the box's
run logged 300 |g| values and zero parameter locations). The `pairs`
iterable is normally the driver's epoch_batches generator (per-epoch
re-batching: fresh in-batch negative sets every epoch, deterministic
per seed).

    W20.11.2 — the loader pinning: each pool refresh stages its prefill
inputs on the capture device (a non_blocking H2D — the driver hands
pinned HOST id tensors; ids already on the device pass through), and
the NEXT pool's ids pre-stage during the current refresh cycle (the
staging window), resident before the next prefill wave. Transport
only — values bit-identical; the training step's data path (replay
over the captured k/v) is untouched.

    The model must be loaded with use_m1m2=True (the wiring) — the same
    mem_size as the intended ingestion; freeze_all_but_gates runs first.
    """
    cfg = cfg or GatesTrainConfig()
    if cfg.refresh_every < 1:
        raise ValueError(
            f"train_gates: refresh_every must be >= 1, got "
            f"{cfg.refresh_every}")
    if isinstance(cfg.neg_pairs, bool) or not isinstance(cfg.neg_pairs, int) \
            or cfg.neg_pairs < 0:
        raise ValueError(
            f"train_gates: neg_pairs must be an int >= 0, got "
            f"{cfg.neg_pairs!r}")
    neg_pending = (iter(negatives)
                   if negatives is not None and cfg.neg_pairs > 0 else None)
    neg_exhausted = {"warned": False}
    inner = getattr(model, "model", model)
    module = inner.m1m2
    # W20.11.2 — the loader pinning: the capture device (where the
    # gates/model live — the pool's prefill inputs stage onto it per
    # refresh; ids already there pass through .to() unchanged)
    cap_dev = module.write_gate_k.device
    trainables = freeze_all_but_gates(model,
                                      include_read_gate=not cfg.freeze_read_gate)
    # W19 — the saddle escape: the P3 zero-init write gates are an exact
    # saddle of the InfoNCE (dL/dg identically 0; the module docstring's
    # proof). Open them before step 0 unless explicitly told not to.
    if cfg.gate_init:
        with torch.no_grad():
            module.write_gate_k.fill_(cfg.gate_init)
            module.write_gate_v.fill_(cfg.gate_init)
        print(f"    [gates] W19 saddle escape: write gates opened to "
              f"{cfg.gate_init:+.3f} (P3 zero-init is an exact saddle — "
              f"loss ln(B), gradient identically zero)", flush=True)
    elif (float(module.write_gate_k.detach().abs().max()) == 0.0
          and float(module.write_gate_v.detach().abs().max()) == 0.0):
        print("    [gates] WARNING: zero write gates (P3) — the EXACT "
              "saddle: the loss will sit at ln(B) and the gates will "
              "NOT move; set gate_init/--gate-init != 0 to escape",
              flush=True)
    # W20.14 — the layer mask (the H6 knob): validate once, zero the
    # masked gate grads after every backward (weight_decay=0 keeps the
    # masked gates bit-anchored — they never move)
    keep_mask = None
    if cfg.train_layers is not None:
        idx = sorted({int(i) for i in cfg.train_layers})
        if not idx:
            raise ValueError(
                "train_gates: train_layers=[] would train nothing — "
                "pass None for all layers")
        outside = [i for i in idx
                   if not 0 <= i < module.num_linear_layers]
        if outside:
            raise ValueError(
                f"train_gates: train_layers indices {outside} outside "
                f"[0, {module.num_linear_layers}) — the write gates are "
                f"per-linear-layer")
        keep_mask = torch.zeros(module.num_linear_layers, dtype=torch.bool,
                                device=cap_dev)
        keep_mask[idx] = True
        print(f"    [gates] W20.14 layer mask: training write gates for "
              f"layers {idx} only ({module.num_linear_layers - len(idx)} "
              f"frozen at their current value)", flush=True)
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(
        [p for n, p in module.named_parameters()
         if any(n.endswith(t.split(".")[-1]) for t in trainables)],
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    t0 = time.perf_counter()
    phase = {"capture_s": 0.0, "replay_s": 0.0, "backward_s": 0.0}
    cap_batch: Optional[List[List[Tuple]]] = None   # the retained sides
    steps_on_batch = 0
    batch_idx = 0       # W20.15 — the capture-batch ordinal (the block id)
    pool_bytes = 0
    pending = iter(pairs)
    step = -1
    # W20.11.2 — the loader pinning: `staged` holds the NEXT pool's
    # prefill inputs, already on the capture device (the staging window
    # IS the refresh cycle — the next pool's ids stage while the
    # current pool trains, resident before the next prefill wave).
    # None = nothing pre-staged (the first refresh, or the pairs ran
    # out at the prefetch).
    staged: Optional[List] = None
    while True:
        step += 1
        if step >= cfg.max_steps:
            break
        # ---- the capture pool (W19): one batch of pairs is prefilled,
        # then trained on for cfg.refresh_every GRADIENT STEPS (the
        # replay re-runs each step, so the gradient is always taken at
        # the CURRENT gates; only the captured k/v are frozen). This is
        # the amortization: prefill count drops refresh_every-fold.
        # The k/v are detached constants in the replay either way — the
        # W18 exactness contract is untouched; the traded quantity is
        # capture staleness (the gates moved <= lr x refresh_every).
        if cap_batch is None or steps_on_batch >= cfg.refresh_every:
            tc = time.perf_counter()
            if staged is None:
                try:
                    staged = _stage_prefill_ids(next(pending), cap_dev)
                except StopIteration:
                    break    # pairs exhausted (the for-loop semantics)
            batch = staged
            staged = None
            if len(batch) < 2:
                raise ValueError(
                    f"train_gates: batch {step} holds {len(batch)} "
                    f"pairs — the InfoNCE needs >= 2 (in-batch "
                    f"negatives)")
            cap_batch = []
            cap_negs: List[Tuple] = []       # W22 — the pool's curated sides
            pool_bytes = 0
            for ids_a, ids_b in batch:
                sides = []
                for ids in (ids_a, ids_b):
                    cache = cache_factory()
                    calls, m1_i, m2_i = prefill_capture_calls(
                        model, ids, cache, system)
                    pool_bytes += _calls_bytes(calls)
                    sides.append((calls, m1_i, m2_i))
                    # W18: the calls' k/v stay alive inside the
                    # replay graph until backward; the CACHE itself
                    # is droppable now — codes only.
                    del cache
                cap_batch.append(sides)
            # W22 — the curated hard-negative sides ride the SAME pool
            # (one prefill each, the same retention contract; staged on
            # the capture device like the pair sides). Exhaustion is a
            # one-time loud warning + the plain W21 regime for the rest
            # of the run — never a crash mid-pool.
            if neg_pending is not None:
                try:
                    neg_batch = [_stage_ids(t, cap_dev)
                                 for t in next(neg_pending)]
                except StopIteration:
                    if not neg_exhausted["warned"]:
                        print("    [gates] [W22] the negatives iterator is "
                              "exhausted — the remaining pools train on "
                              "in-batch negatives only (the W21 regime)",
                              flush=True)
                        neg_exhausted["warned"] = True
                    neg_pending = None
                    neg_batch = []
                if neg_pending is not None and len(neg_batch) != cfg.neg_pairs:
                    raise ValueError(
                        f"train_gates: the negatives batch holds "
                        f"{len(neg_batch)} side(s), cfg.neg_pairs is "
                        f"{cfg.neg_pairs} — the driver's iterator must "
                        f"yield exactly cfg.neg_pairs per batch")
                for ids in neg_batch:
                    cache = cache_factory()
                    calls, m1_i, m2_i = prefill_capture_calls(
                        model, ids, cache, system)
                    pool_bytes += _calls_bytes(calls)
                    cap_negs.append((calls, m1_i, m2_i))
                    del cache
            # W20.11.2 — pre-stage the NEXT pool's prefill inputs NOW:
            # the staging window is the rest of this refresh cycle (the
            # tiny non_blocking copies ride the current stream during
            # this pool's gradient steps; same-stream ordering has them
            # complete long before any use — values bit-identical).
            try:
                staged = _stage_prefill_ids(next(pending), cap_dev)
            except StopIteration:
                staged = None
            steps_on_batch = 0
            batch_idx += 1      # W20.15 — a new capture block starts
            phase["capture_s"] += time.perf_counter() - tc
            if cfg.log_every:
                neg_note = (f" + {len(cap_negs)} neg sides"
                            if cap_negs else "")
                print(f"    [gates] capture batch prefilled: "
                      f"{len(cap_batch)} pairs{neg_note} "
                      f"({pool_bytes / 2**20:.0f}"
                      f" MiB retained k/v)", flush=True)
        steps_on_batch += 1

        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, cfg.max_steps, cfg)
        # ---- replay under grad (the W18 exact-gradient Function default)
        tr = time.perf_counter()
        states = []
        for sides in cap_batch:
            pair_states = []
            for (calls, m1_i, m2_i) in sides:
                m1, m2 = replay_states(module, calls, m1_i, m2_i,
                                       memopt=cfg.memopt)
                if cfg.delta_loss:
                    # W19: score the DELTAS the centered retrieval frame
                    # scores — the system state's common direction out
                    m1, m2 = m1 - m1_i, m2 - m2_i
                pair_states.append((m1, m2))
            states.append(pair_states)
        m1_a = torch.stack([s[0][0] for s in states])
        m2_a = torch.stack([s[0][1] for s in states])
        m1_b = torch.stack([s[1][0] for s in states])
        m2_b = torch.stack([s[1][1] for s in states])
        # W22 — the curated sides replay under grad exactly like the
        # pair sides (the push-away gradient must reach their gates);
        # delta_loss centers them in the same retrieval frame.
        m1_n = m2_n = None
        if cap_negs:
            neg_states = []
            for (calls, m1_i, m2_i) in cap_negs:
                m1, m2 = replay_states(module, calls, m1_i, m2_i,
                                       memopt=cfg.memopt)
                if cfg.delta_loss:
                    m1, m2 = m1 - m1_i, m2 - m2_i
                neg_states.append((m1, m2))
            m1_n = torch.stack([s[0] for s in neg_states])
            m2_n = torch.stack([s[1] for s in neg_states])
            del neg_states
        phase["replay_s"] += time.perf_counter() - tr
        loss = infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b, tau=cfg.tau,
                                  m1_n=m1_n, m2_n=m2_n)
        opt.zero_grad(set_to_none=True)
        tb = time.perf_counter()
        loss.backward()
        # W20.14 — the layer mask (the H6 knob): masked gates get zero
        # grad BEFORE the |g| read, so the instrumentation reports the
        # gradient that will actually be applied
        if keep_mask is not None:
            for p in (module.write_gate_k, module.write_gate_v):
                if p.grad is not None:
                    p.grad[~keep_mask] = 0
            if (not cfg.freeze_read_gate
                    and module.read_gate.grad is not None):
                module.read_gate.grad[~keep_mask] = 0
        # W20.14 — the discrimination the loss optimizes, finally
        # measurable: the same B x B sim matrix, read as stats (the v1
        # metric reported the negatives as the ROLLED true-pair mean —
        # a permutation, identical mean, spread structurally 0)
        stats = _pair_sim_stats(m1_a, m2_a, m1_b, m2_b, m1_n=m1_n,
                                m2_n=m2_n)
        # W19 — the saddle/health detector: the gate-grad magnitudes the
        # box asked for (recorded BEFORE the clip; 0.0 = dead path)
        gk_absmax = _grad_absmax(module.write_gate_k)
        gv_absmax = _grad_absmax(module.write_gate_v)
        # W18: the replay graphs (the recorded k/v refs + the state
        # outputs) free here; explicit del after the metrics are read
        # keeps the peak flat across steps on the A10G.
        del states, m1_a, m2_a, m1_b, m2_b
        if m1_n is not None:
            del m1_n, m2_n
        torch.nn.utils.clip_grad_norm_(
            [p for p in module.parameters() if p.requires_grad],
            cfg.grad_clip)
        opt.step()
        phase["backward_s"] += time.perf_counter() - tb
        cfg.history.append({
            "step": step, "loss": float(loss.item()),
            "lr": opt.param_groups[0]["lr"],
            "gk_absmax": gk_absmax, "gv_absmax": gv_absmax,
            "pool": steps_on_batch,
            "batch": batch_idx,          # W20.15 — the block id
            **_gate_value_stats(module),  # W20.15 — where the gates ARE
            **stats, **phase})
        if cfg.log_every and step % cfg.log_every == 0:
            hneg_note = ("" if not math.isfinite(stats["cos_hneg"])
                         else f" hneg {stats['cos_hneg']:+.3f}")
            print(f"    [gates] step {step:4d} pool {steps_on_batch}/"
                  f"{cfg.refresh_every} loss {loss.item():.4f} "
                  f"cos(pos) {stats['cos_pos']:+.3f} "
                  f"cos(neg) {stats['cos_neg']:+.3f}{hneg_note} "
                  f"spread {stats['spread']:+.3f} "
                  f"[worst neg {stats['off_max']:+.3f} "
                  f"weakest pos {stats['diag_min']:+.3f}] "
                  f"lr {opt.param_groups[0]['lr']:.2e} "
                  f"|g| {gk_absmax:.2e}/{gv_absmax:.2e} "
                  f"cap {phase['capture_s'] / (step + 1):.2f}s "
                  f"rep {phase['replay_s'] / (step + 1):.2f}s "
                  f"bwd {phase['backward_s'] / (step + 1):.2f}s "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    if cfg.history:
        gv_stats = _gate_value_stats(module)
        print(f"    [gates] done: {len(cfg.history)} steps in "
              f"{time.perf_counter() - t0:.1f}s — capture "
              f"{phase['capture_s']:.1f}s / replay "
              f"{phase['replay_s']:.1f}s / backward "
              f"{phase['backward_s']:.1f}s "
              f"({100 * phase['capture_s'] / max(1e-9, sum(phase.values())):.0f}% "
              f"capture), final |gk| "
              f"{_grad_absmax(module.write_gate_k):.2e}, gates "
              f"k[{gv_stats['gk_min']:+.3f}..{gv_stats['gk_max']:+.3f}] "
              f"v[{gv_stats['gv_min']:+.3f}..{gv_stats['gv_max']:+.3f}] "
              f"(|k| {gv_stats['gk_absmean']:.3f} / |v| "
              f"{gv_stats['gv_absmean']:.3f})", flush=True)
    return cfg
