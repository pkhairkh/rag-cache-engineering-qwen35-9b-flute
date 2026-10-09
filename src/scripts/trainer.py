#!/usr/bin/env python3
"""trainer.py — the layerwise distillation trainer: the engine's ported
home (W2-T07 part 1: the layer-residency plane + the `train` driver;
W2-T08 part 2: the control plane, the resident-eval pair, the export
machinery — the trainer is now ENGINE-FREE; W4-T01: the joint trainable
set — `--train luts,norms,lora` enumerates LUT masters (make_trainable),
the layer's norm gains, and the attached LoRA A/B at attach, with the
one-line step-0 census; W4-T02: the two-channel optimizer — the lora
group (AdamW or Muon at --lr-lora) plus the codes channel (LUT masters
at --lr-lut, norm gains their own group at --lr-norm, always AdamW),
the schedule per group, one global clip; the anchor/banked-best/tripwire
restores are the JOINT true clone — PROPOSAL §2.5/§2.6; W4-T06: the
export path — `export --run <dir> --out <dir>` writes the deployment
artifacts COPY from the banked-best joint snapshots (fp16 LUT snap +
the polish grid re-snap, the norm-gain edits file, the G-J3 reload
gate, the merge hook — PROPOSAL §2.8); W4-T07: the 2-config pilot
— `pilot --layers 0,3 --alt lr-lut 1e-4` runs the base and the alt
config through the SAME layer jobs (one shared teacher + target cache
per layer) and records the §7.1 decision to pilot_joint.json.

Contract (PROPOSAL.md §2.2-2.3, §2.5-§2.6, §2.8, §7.2; TASKS.md
W2-T07/T08, W4-T01/T02):
  * two-layer residency — per layer, ONE pristine dense teacher layer
    (TeacherLayerSource, streamed from the checkpoint shards) and ONE
    layer-scoped student layer (materialize_student_layer) are resident;
    the full student model is never built;
  * `train` is the adapter-only channel — parity with the engine's
    `finetune --train-target qlora`: inputs are the captured boundary
    states h_i, targets are the resident teacher layer's forward, only
    lora_A/lora_B train, and the output is an adapter dir (the
    artifacts dir is never written);
  * the control plane is PROPOSAL §2.6 verbatim (ported, not
    reinvented): banking (holdout rel_mse primary, tok_cos tiebreak,
    TRUE-CLONE `_lora_snapshot` banking, final restore-best), patience
    stop, the F10 tripwire (2.0x EMA rule, restore-best, halve LR,
    2 firings max — the 3rd STOPS), the before-training anchor, the
    warm-start gate verdicts, provenance-fingerprinted resume; the
    resident-eval pair and the export machinery (PROPOSAL §2.8) are
    ported alongside;
  * SELF-SUFFICIENT (W2-T08): this module imports ONLY the leaf
    modules (capture/loss/qlora/palettized_modules/modeling via lazy
    accessors) plus spectrum via a function-level lazy import for the
    warm-start factors. The ENGINE imports nothing from here and this
    module imports nothing from the engine — both live until W2-T10
    deletes the engine (its copies die with it).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import types
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import (  # noqa: E402
    create_causal_mask,
    create_recurrent_attention_mask,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import palettized_modules as pmod          # noqa: E402  (local, same dir)
from qlora import (                        # noqa: E402  (local, same dir)
    QLoRAConfig,
    QLoRALinear,
    attach_qlora,
    freeze_all_non_qlora,
    iter_qlora_modules,
    report_frozen_paths,
)
from eval_common import atomic_json_dump as _atomic_json_dump  # noqa: E402
from eval_common import paired_mean  # noqa: E402
from capture import (                      # noqa: E402  (local, same dir)
    CaptureStore,
    _memmap_rw,
    _reader_amp,
    _reader_counters_reset,
    _split_runs,
    _text_config,
)
from loss import distill_loss              # noqa: E402  (local, same dir)

# ---------------------------------------------------------------------------
# Lazy import of the vendored text-only modeling file
# ---------------------------------------------------------------------------

_MODELING = None


def _modeling():
    """Import the vendored modeling.py (same directory)."""
    global _MODELING
    if _MODELING is None:
        import modeling as M
        _MODELING = M
    return _MODELING


# ---------------------------------------------------------------------------
# The layer-residency plane: ONE teacher layer + ONE layer-scoped student
# layer resident at a time; the input is the captured boundary state h_i,
# the targets are computed on the fly by the resident teacher layer.
# ---------------------------------------------------------------------------

_QLORA_DEFAULT_RANK = 64        # uniform fallback when no rank map is given
_QLORA_DEFAULT_ALPHA = 16       # r=64 / alpha=16 — the proven kappa=0.25


def _position_embeddings(model, seq: int, dtype, device):
    """cos/sin for text positions 0..S-1 (all grid rows identical — the
    text-only convention of Qwen3_5TextModel.forward). Computed with
    batch=1 (values are batch-independent); expand with _pos_batch()."""
    rotary = model.model.rotary_emb
    # Ensure rotary embedding is on the target device (model may be CPU-only
    # when layers are trained one at a time and moved individually)
    rotary = rotary.to(device)
    pos = torch.arange(seq, device=device).view(1, 1, seq).expand(3, 1, seq)
    dummy = torch.zeros(1, seq, _text_config(model).hidden_size,
                        dtype=dtype, device=device)
    cos, sin = rotary(dummy, pos)
    return cos, sin


def _text_config_from(model_or_cfg):
    cfg = getattr(model_or_cfg, "config", model_or_cfg)
    return _text_config(cfg) if hasattr(cfg, "text_config") else cfg


class TeacherLayerSource:
    """Loads one pristine dense decoder layer at a time from the local
    checkpoint safetensors shards (O(one-layer) VRAM). The teacher is the
    untouched fp16 checkpoint — never norm-edited (targets must come from
    the unmodified dense model)."""

    def __init__(self, model_ref: str, device: str = "cpu"):
        from safetensors import safe_open
        self.device = device
        self.model_ref = model_ref
        if not os.path.isdir(model_ref):
            raise SystemExit(
                f"--teacher-on-student requires a LOCAL checkpoint directory "
                f"(got {model_ref!r}); pre-download the model or point "
                f"--model at the local path")
        index = os.path.join(model_ref, "model.safetensors.index.json")
        shards = []
        if os.path.exists(index):
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted(set(weight_map.values()))
            self.weight_map = weight_map
        else:
            single = os.path.join(model_ref, "model.safetensors")
            if not os.path.exists(single):
                raise SystemExit(f"no safetensors found under {model_ref}")
            shards = ["model.safetensors"]
            self.weight_map = None
        # layer -> {tensor_name: shard}
        self.layer_shards = {}
        for shard in shards:
            path = os.path.join(model_ref, shard)
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    # Support both model.layers.* and model.language_model.layers.*
                    layer_match = None
                    if key.startswith("model.layers."):
                        layer_match = int(key.split(".")[2])
                    elif key.startswith("model.language_model.layers."):
                        layer_match = int(key.split(".")[3])
                    if layer_match is not None:
                        self.layer_shards.setdefault(layer_match, {})[key] = shard
        # flat tensor-name -> shard map over every model.layers.* tensor;
        # the `weights` subcommand streams single dense tensors through it
        # (same shard machinery, no layer construction). The model's
        # FINAL norm gain is indexed alongside the layer tensors (the
        # W4-T05 final-norm job's student base — the dense checkpoint's
        # own gain; both text-only and multimodal key layouts).
        self.tensor_shards = {k: s for d in self.layer_shards.values()
                              for k, s in d.items()}
        for shard in shards:
            path = os.path.join(model_ref, shard)
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    if key in ("model.norm.weight",
                               "model.language_model.norm.weight"):
                        self.tensor_shards[key] = shard
        self._cached_L = None
        self._cached_layer = None

    def layer_state(self, L: int):
        from safetensors.torch import load_file
        shard_map = self.layer_shards.get(L, {})
        if not shard_map:
            raise KeyError(f"no tensors found for layer {L} in {self.model_ref}")
        # Derive prefix from actual key (handles model.layers.* AND model.language_model.layers.*)
        first_key = next(iter(shard_map.keys()))
        # Find where the layer index ends (after "model.layers.{L}." or "model.language_model.layers.{L}.")
        parts = first_key.split(".")
        # Parts are like ["model", "layers", "0", ...] or ["model", "language_model", "layers", "0", ...]
        # Find the part that is the layer number
        for i, p in enumerate(parts):
            if p.isdigit() or (p.lstrip('-').isdigit()):
                prefix = ".".join(parts[:i+1]) + "."
                break
        else:
            prefix = f"model.layers.{L}."
        by_shard = {}
        for key, shard in shard_map.items():
            by_shard.setdefault(shard, []).append(key)
        sd = {}
        for shard, keys in by_shard.items():
            tensors = load_file(os.path.join(self.model_ref, shard),
                                device=self.device if self.device != "cpu"
                                else "cpu")
            for key in keys:
                sd[key[len(prefix):]] = tensors[key].to(torch.float16)
            del tensors
        return sd

    def load_layer(self, L: int, config, sd=None):
        """Build the resident teacher layer for L. `sd` may carry a
        pre-loaded layer state (the shared single shard read of the
        Stage-R pass / the qlora layer job — read the safetensors layer
        shard ONCE, use it for both the teacher and the student's dense
        parts)."""
        if self._cached_L == L:
            return self._cached_layer
        M = _modeling()
        if sd is None:
            sd = self.layer_state(L)
        layer_cfg = _text_config_from(config)
        if getattr(layer_cfg, "_attn_implementation", None) is None:
            # a standalone layer from a fresh config silently runs eager
            # attention; direct calls would then be uncausal (see
            # _attn_impl) — the teacher must inherit the pinned backend
            raise SystemExit(
                "TeacherLayerSource: config._attn_implementation is None "
                "(fresh/standalone config). Build the student with "
                "from_pretrained(..., attn_implementation='sdpa') so the "
                "teacher layer inherits a pinned backend.")
        layer = M.Qwen3_5DecoderLayer(layer_cfg, L)
        layer.load_state_dict(sd, strict=True)
        layer.eval().requires_grad_(False)
        layer = layer.to(self.device, torch.float16)
        self._cached_L = L
        self._cached_layer = layer
        return layer

    def drop_cache(self):
        self._cached_L = None
        self._cached_layer = None

    def load_dense_tensor(self, name: str, device: str = None):
        """Stream ONE dense tensor from the checkpoint shards by its full
        checkpoint name (lazy per-tensor read via safe_open().get_tensor —
        no full-shard materialization).

        The student-var-path -> dense-tensor-name mapping is the identity:
        metadata var paths (model.layers.<i>.<path>.weight) ARE checkpoint
        tensor names — the same contract load_layer applies when it slices
        a layer's state dict. SplitQKV components read row slices of the
        single dense in_proj_qkv tensor (Q rows first, then K, then V — the
        palettizer's split order; see cmd_weights). A missing tensor is a
        loud RuntimeError, never a silent skip."""
        from safetensors import safe_open
        shard = self.tensor_shards.get(name)
        if shard is None:
            raise RuntimeError(
                f"teacher checkpoint {self.model_ref!r} has no tensor "
                f"{name!r} (required by a student module) — the dense "
                f"teacher and the palettized artifacts do not correspond; "
                f"refusing to silently skip this module")
        with safe_open(os.path.join(self.model_ref, shard),
                       framework="pt") as f:
            t = f.get_tensor(name)
        return t.to(self.device if device is None else device)


def _attn_impl(model):
    """Pinned attention implementation of the model's text config.

    Direct decoder-layer calls bypass Qwen3_5TextModel.forward, so nothing
    builds the masks the full path creates. Under 'sdpa' a None mask is
    still causal (sdpa_attention_forward derives is_causal=True for
    q_len > 1), but under 'eager' — including the silent eager fallback a
    standalone/fresh config gets (config._attn_implementation None ->
    ALL_ATTENTION_FUNCTIONS.get_interface(None, eager)) — a None mask
    means bidirectional attention (measured: logits cosine 0.943 on the
    tiny hybrid model). Fail loudly instead of corrupting trajectories."""
    cfg = _text_config(model)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl is None:
        raise SystemExit(
            "config._attn_implementation is None: a layer built from this "
            "config silently runs EAGER attention, and a direct layer call "
            "without a mask is then UNCAUSAL. Load the model via "
            "from_pretrained(..., attn_implementation='sdpa') (or 'eager') "
            "so _forward_layer can replicate the full forward's masks.")
    return impl, cfg


_LAYER_MASK_CACHE = {}


def _layer_mask(layer, x, model):
    """The mask Qwen3_5TextModel.forward would hand THIS layer for packed
    rows (2D attention_mask=None, positions 0..S-1):

      full_attention   -> create_causal_mask: None under sdpa (implicit
                         is_causal), explicit 4D causal mask under eager
      linear_attention -> create_recurrent_attention_mask: None for
                         packed rows (no padding states to zero)

    Replicating the TextModel construction makes the direct call causal
    under any backend — and bit-identical to the full forward (sweep_03
    T-B1/B2/B4: max|d|=0 both backends)."""
    _, cfg = _attn_impl(model)
    block = layer.block_type
    B, S = x.shape[0], x.shape[1]
    pos = torch.arange(S, device=x.device).view(1, S).expand(B, S)
    key = None
    if block == "full_attention":
        mask = create_causal_mask(config=cfg, inputs_embeds=x,
                                  attention_mask=None,
                                  past_key_values=None, position_ids=pos)
    else:
        mask = create_recurrent_attention_mask(
            config=cfg, inputs_embeds=x, attention_mask=None,
            past_key_values=None, position_ids=pos)
    if mask is not None:
        # packed causal mask depends only on (shape, dtype, device) —
        # cache the eager 4D tensor instead of rebuilding it every step
        key = (block, B, S, str(x.dtype), str(x.device))
        mask = _LAYER_MASK_CACHE.setdefault(key, mask)
    return mask


def _forward_layer(layer, x, position_embeddings, model):
    """Call the real decoder layer exactly like Qwen3_5TextModel.forward
    does for packed rows (positions 0..S-1) — including its per-backend
    mask construction, so the call is causal under any attention backend
    (sdpa: implicit is_causal; eager: explicit 4D causal mask)."""
    return layer(x,
                 position_embeddings=position_embeddings,
                 attention_mask=_layer_mask(layer, x, model),
                 position_ids=None,
                 past_key_values=None)


class _ShellText(nn.Module):
    """`model` sub-shell: .layers (the ONE layer at its true index) and
    .rotary_emb — exactly the attributes _position_embeddings/get_layers
    touch on a real model."""

    def __init__(self, layers, rotary_emb):
        super().__init__()
        self.layers = layers
        self.rotary_emb = rotary_emb


class StudentLayerShell(nn.Module):
    """Layer-scoped student container — the W4 seam.

    `model.layers.<i>` holds the ONE materialized decoder layer (None
    padding keeps the true index), `model.rotary_emb` the shared rotary
    module and `config` the pinned text config. Consequences:
      * named_modules()/named_parameters() carry FULL checkpoint paths
        (model.layers.<i>.mlp.gate_proj...) — attach_qlora / save / load
        keys then match load_qlora_model's expectations exactly;
      * _position_embeddings / _attn_impl / _layer_mask / _forward_layer
        work on the shell exactly as on a full model (only those surfaces
        are ever consulted);
      * build_student / full-model instantiation is never needed in qlora
        mode (the two-layer-residency invariant).
    """

    def __init__(self, layer, layer_idx, config, device=None):
        super().__init__()
        cfg = _text_config_from(config)
        if getattr(cfg, "_attn_implementation", None) is None:
            # same guard as TeacherLayerSource: a standalone layer from a
            # fresh config silently runs eager attention and a None mask is
            # then UNCAUSAL — pin the backend instead of guessing
            raise SystemExit(
                "StudentLayerShell: config._attn_implementation is None "
                "(fresh/standalone config) — a layer built from it silently "
                "runs eager attention and direct calls would be uncausal. "
                "Pass a config pinned via from_pretrained(..., "
                "attn_implementation='sdpa') or set the attribute.")
        self.config = cfg
        M = _modeling()
        rotary = M.Qwen3_5TextRotaryEmbedding(cfg, device)
        layers = nn.ModuleList([None] * layer_idx + [layer])
        self.model = _ShellText(layers, rotary)

    @property
    def layer(self):
        return self.model.layers[
            [i for i, m in enumerate(self.model.layers)
             if m is not None][0]]


def _apply_layer_norm_edits(layer, layer_idx, artifacts_dir,
                            verify_sha: bool = True) -> dict:
    """Layer-scoped norm-gain edits: the entries of norm_gain_edits.json
    belonging to layer `layer_idx`, applied to the (dense) layer parameters
    BEFORE the palettized swap — the order build_student mandates.

    Keys carry the full parameter name of the loaded HF model, which may
    nest the text stack either way (model.layers.* or
    model.language_model.layers.*); the layer index is parsed out and the
    remainder is mapped onto the layer's own named parameters. A key that
    parses to this layer but has no matching parameter is a loud error;
    other layers' entries are simply not this layer's business."""
    path = os.path.join(artifacts_dir, "norm_gain_edits.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    edits = doc.get("edits", {})
    named = dict(layer.named_parameters())
    applied = {}
    for pname, entry in edits.items():
        toks = pname.split(".")
        if toks and toks[0] == "model":
            toks = toks[1:]
        if len(toks) >= 2 and toks[0] == "language_model":
            toks = toks[1:]
        if len(toks) < 2 or toks[0] != "layers" or not toks[1].lstrip("-").isdigit():
            raise RuntimeError(
                f"norm_gain_edits: cannot parse parameter name {pname!r} "
                f"(expected model[.language_model].layers.<i>.<param>)")
        if int(toks[1]) != layer_idx:
            continue
        rel = ".".join(toks[2:])
        if rel not in named:
            raise RuntimeError(
                f"norm_gain_edits: {pname!r} parses to layer {layer_idx} "
                f"but the layer has no parameter {rel!r} — the artifacts "
                f"and the layer structure disagree")
        edit_path = os.path.join(artifacts_dir, entry["file"])
        pmod._check_sha(edit_path, entry.get("sha256"),
                        f"norm_edit:{pname}", verify=verify_sha)
        arr = np.load(edit_path)
        param = named[rel]
        if tuple(arr.shape) != tuple(param.shape):
            raise ValueError(
                f"norm_edit:{pname}: shape {arr.shape} != parameter shape "
                f"{tuple(param.shape)}")
        param.data.copy_(torch.from_numpy(arr).to(param.dtype))
        applied[pname] = entry
    return applied


class _FullNamedShim:
    """Presents {full_param_name: tensor} as a model-like object for
    pmod._recover_awq_scales (which consumes model.named_parameters())."""

    def __init__(self, named):
        self._named = dict(named)

    def named_parameters(self):
        return list(self._named.items())


def _layer_norm_edit_slice(norm_doc, layer_idx):
    """The entries of norm_gain_edits.json that belong to layer
    `layer_idx` (full names parsed the _apply_layer_norm_edits way).
    Non-layer entries (e.g. model.norm.weight) are NOT this helper's
    business and are dropped."""
    edits = (norm_doc or {}).get("edits", {}) or {}
    mine = {}
    for pname, entry in edits.items():
        toks = pname.split(".")
        if toks and toks[0] == "model":
            toks = toks[1:]
        if len(toks) >= 2 and toks[0] == "language_model":
            toks = toks[1:]
        if len(toks) < 2 or toks[0] != "layers" \
                or not toks[1].lstrip("-").isdigit():
            continue
        if int(toks[1]) == layer_idx:
            mine[pname] = entry
    return mine


def _layer_awq_scales(named_full, layer_idx, artifacts_dir, metadata,
                      norm_doc=None):
    """The W13 compensation input, LAYER-SCOPED: the per-consumer AWQ
    scale map recovered exactly the way load_palettized_model does
    model-wide (pmod._recover_awq_scales — including the recorded-scale
    preference and every validation/cross-check), but from ONE layer's
    pristine parameter map.

    `named_full` maps FULL parameter names (model.layers.<i>....weight)
    to the PRISTINE (pre-edit) gains — the caller must capture it
    BEFORE _apply_layer_norm_edits lands, exactly like the model-wide
    loader recovers from the pristine from_pretrained model. Entries of
    other layers are filtered out (they would trip the shim's KeyError
    guard); consumers on OTHER layers still cross-check against the
    full metadata, same as the model-wide path.

    Returns {consumer_tensor_name: (K,) fp32 tensor} — the map
    load_palettized_weight(awq_scales=...) consumes."""
    if norm_doc is None:
        norm_doc = pmod._read_norm_gain_doc(artifacts_dir)
    if not norm_doc:
        return {}
    mine = _layer_norm_edit_slice(norm_doc, layer_idx)
    if not mine:
        return {}
    return pmod._recover_awq_scales(
        _FullNamedShim(named_full), artifacts_dir, metadata,
        {"edits": mine})


def materialize_student_layer(layer_idx, config, base_stream, artifacts_dir,
                              metadata, device="cpu",
                              dtype=torch.float16,
                              awq_compensation: bool = True):
    """Materialize ONLY layer `layer_idx` of the student (the W4 seam).

    Builds Qwen3_5DecoderLayer(layer_idx), loads its DENSE parts from
    `base_stream.layer_state(layer_idx)` (a TeacherLayerSource over the same
    local checkpoint — the student's dense base IS the teacher checkpoint),
    applies this layer's norm-gain edits from the artifacts dir, then swaps
    this layer's palettized modules in from the frozen artifacts dir
    (reference dequant path). Never instantiates the full student model.

    R1 (recovery campaign): the swap now attaches the stored resA/resB
    residual branch (`residual=True`) — the SAME effective weight
    `_student_effective_weight` models for the spectrum and qlora_merge
    materializes for deployment. Before R1 the deployed student silently
    DROPPED the residual whenever the artifacts carried one, so the
    spectrum's E (and every warm-start factor derived from it) was fit
    against a student that never existed — the Stage-1 halt's warm-start
    mis-fit class. Artifacts without residuals are unaffected (resA=None
    is a no-op); the attach count is PRINTED so the box can see whether
    its artifacts carry residuals at all.

    Toy tests drive the same seam with any object exposing
    `layer_state(i) -> dict` (module-relative names -> fp16 tensors) plus
    toy idx4 artifacts + a tiny Qwen3_5TextConfig.

    W14 (compensation parity): the AWQ scales of this layer's edit
    entries are recovered from the PRISTINE gains (captured between
    load_state_dict and _apply_layer_norm_edits — the exact window the
    model-wide loader uses) and handed to load_palettized_weight, so the
    student trains against the SAME compensated fold (M = D T D^-1 for
    legacy rotate-then-AWQ artifacts) the deployment serves. Before
    W14 the seam swapped modules with awq_scales=None: on the box's
    legacy artifacts every alpha>0 group trained against the scrambled
    fold — the distillation would fit LUT/norms to a model that never
    deploys. `awq_compensation=False` reproduces that arm for
    differential tests only.

    Returns (layer, n_swapped): the layer (eval, requires_grad False, on
    `device`/`dtype`) and the number of palettized modules swapped in."""
    M = _modeling()
    layer_cfg = _text_config_from(config)
    if getattr(layer_cfg, "_attn_implementation", None) is None:
        raise SystemExit(
            "materialize_student_layer: config._attn_implementation is "
            "None — pin the backend (from_pretrained(..., "
            "attn_implementation='sdpa')) before building layers")
    layer = M.Qwen3_5DecoderLayer(layer_cfg, layer_idx)
    sd = base_stream.layer_state(layer_idx)
    # order mirrors build_student: dense state -> norm edits -> swap
    layer.load_state_dict(sd, strict=True)
    # W14: the scales are recovered while the gains are STILL pristine
    # (the diff (1+w_orig)/(1+w_edit) is only readable pre-edit; the
    # export's pinned awq_scale_file is preferred when present).
    awq_scales = _layer_awq_scales(
        {f"model.layers.{layer_idx}.{rel}": p
         for rel, p in layer.named_parameters()},
        layer_idx, artifacts_dir, metadata) if awq_compensation else {}
    # Apply norm edits to student's layernorms
    # This matches the measurement in weights alignment
    _apply_layer_norm_edits(layer, layer_idx, artifacts_dir)
    tensors = metadata.get("tensors") if isinstance(metadata, dict) else None
    if not isinstance(tensors, dict) or not tensors:
        raise RuntimeError(
            f"{artifacts_dir}/metadata.json carries no 'tensors' mapping — "
            f"not a palettized artifacts dir")
    swapped = 0
    n_residual = 0
    n_awq = 0
    for tensor_name, tmeta in tensors.items():
        parts = tensor_name.split(".")
        if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers" \
                or not parts[2].lstrip("-").isdigit() \
                or parts[-1] != "weight":
            raise RuntimeError(
                f"{tensor_name}: metadata var path is not "
                f"'model.layers.<i>.<module path>.weight' — the layer-scoped "
                f"materializer cannot resolve it")
        if int(parts[2]) != layer_idx:
            continue
        parent = layer
        attrs = parts[3:-1]
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        old = getattr(parent, attrs[-1])
        bias = getattr(old, "bias", None)
        new = pmod.load_palettized_weight(
            tmeta, artifacts_dir,
            bias=bias.data if bias is not None else None, residual=True,
            reference=True, awq_scales=awq_scales)
        for _pm in new.modules() if isinstance(new, pmod.SplitQKV) \
                else (new,):
            if isinstance(_pm, pmod.PalettizedLinear) and _pm.resA is not None:
                n_residual += 1
            if isinstance(_pm, pmod.PalettizedLinear) \
                    and _pm.awq_scale is not None:
                n_awq += 1
        setattr(parent, attrs[-1], new)
        del old
        swapped += 1
    if swapped == 0:
        raise RuntimeError(
            f"layer {layer_idx}: the metadata carries no palettized var "
            f"paths for this layer — the artifacts set does not cover it "
            f"(no silent dense fallback)")
    # R1: the residual fact is OBSERVABLE — a box whose artifacts carry
    # residuals sees the count once per materialized layer; "none" is
    # the honest answer for residual-free artifacts (the toy fixtures).
    if n_awq:
        print(f"  [L{layer_idx:02d}] [ROT] legacy rotate-then-AWQ fold "
              f"compensated on {n_awq} module(s) — the student trains "
              f"against the DEPLOYED fold (M = D T D^-1)", flush=True)
    print(f"  [L{layer_idx:02d}] student materialized: {swapped} palettized "
          f"module(s) swapped, {n_residual} residual branch(es) attached "
          f"(spectrum/merge parity)", flush=True)
    layer.eval().requires_grad_(False)
    layer.to(device=device, dtype=dtype)
    return layer, swapped


def _rank_slice_for_layer(rank_map, layer_idx):
    """The rank_map entries of one layer (module paths are
    model.layers.<i>....); keys for other layers are simply not this
    layer's slice (subset runs keep a full map)."""
    prefix = f"model.layers.{layer_idx}."
    return {k: v for k, v in rank_map.items() if k.startswith(prefix)}


def _attach_layer_adapters(shell, layer_idx, rank_map, args,
                           r_default=_QLORA_DEFAULT_RANK,
                           alpha_default=_QLORA_DEFAULT_ALPHA):
    """Attach FRESH adapters (B=0 identity init) for this layer's modules
    from the rank map slice and freeze everything but lora_A/lora_B.

    Returns (qlora config of the attach, rank slice). A layer whose whole
    slice is r=0 returns (None, slice): already aligned, no adapter (the
    caller skips it loudly instead of erroring)."""
    rank_slice = _rank_slice_for_layer(rank_map, layer_idx) \
        if rank_map is not None else None
    if rank_slice is not None and rank_slice and \
            all(int(v) == 0 for v in rank_slice.values()):
        return None, rank_slice
    try:
        _, cfg = attach_qlora(
            shell, r=r_default, alpha=alpha_default, dropout=0.0,
            scope="all", include_residual_branch=True,
            init_a="kaiming_uniform", init_b="zero",
            base_model=args.model, artifacts_dir=args.artifacts_dir,
            rank_map=rank_slice, alpha_mode="proportional")
    except RuntimeError as e:
        # name the layer: attach_qlora's unknown-key guard fires per layer
        raise RuntimeError(f"layer {layer_idx}: {e}") from e
    freeze_all_non_qlora(shell)
    return cfg, rank_slice


def _pos_batch(pos, batch):
    cos, sin = pos
    return (cos.expand(batch, -1, -1), sin.expand(batch, -1, -1))


# ---------------------------------------------------------------------------
# The joint trainable set (W4-T01, PROPOSAL §2.3) — enumerated at attach
# ---------------------------------------------------------------------------

_TRAIN_GROUPS = ("luts", "norms", "lora")

# The norm-gain inventory, matched against the layer's ACTUAL module
# tree by name suffix (PROPOSAL §2.3 row 2): the two RMSNorms every
# decoder layer carries, the per-head q/k norms of full-attention
# layers, and the gated norm inside linear attention. The final
# model-level norm is NOT in a layer shell (it is the W4-T05 job).
_NORM_GAIN_SUFFIXES = (
    ".input_layernorm",
    ".post_attention_layernorm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    ".linear_attn.norm",
)


def _parse_train_groups(spec):
    """`--train`: the comma list of trainable groups. Returns a frozenset
    of _TRAIN_GROUPS tokens; an empty selection or an unknown token is
    refused loudly (the trainable set is a contract, never a guess)."""
    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]
    unknown = sorted(set(tokens) - set(_TRAIN_GROUPS))
    if unknown:
        raise SystemExit(
            f"--train {spec!r}: unknown group(s) {unknown} — legal "
            f"groups: {', '.join(_TRAIN_GROUPS)} (comma list, default "
            f"all three)")
    if not tokens:
        raise SystemExit(
            f"--train {spec!r}: the empty selection trains nothing — "
            f"legal groups: {', '.join(_TRAIN_GROUPS)}")
    return frozenset(tokens)


def _train_groups_of(args):
    """The layer job's trainable-set selection: `train_groups` from the
    namespace (a frozenset passes through, a comma string parses);
    a missing attribute means the CLI default — all three groups."""
    raw = getattr(args, "train_groups", None)
    if raw is None:
        return frozenset(_TRAIN_GROUPS)
    if isinstance(raw, str):
        return _parse_train_groups(raw)
    return frozenset(raw)


def _census_record(n_modules, params):
    """One census record: module count, parameter elements, bytes,
    device, dtype — computed from the LIVE tensors (the census reports
    the world, never the wish)."""
    devices = sorted({str(p.device) for p in params})
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in params})
    return {
        "modules": int(n_modules),
        "params": int(sum(p.numel() for p in params)),
        "bytes": int(sum(p.numel() * p.element_size() for p in params)),
        "device": ",".join(devices) if devices else "-",
        "dtype": ",".join(dtypes) if dtypes else "-",
    }


def _joint_trainable_census(shell):
    """The per-group trainable-set census over the CURRENT shell state:
    {group: {modules, params, bytes, device, dtype}}.

    Groups (PROPOSAL §2.3):
      * luts — every PalettizedLinear whose LUT is a trainable fp32
        master (isinstance nn.Parameter and requires_grad), counted per
        module (a QLoRALinear wrapper's base and an unwrapped module
        are the same one module);
      * norms — the inventory gains (suffix match) that require grad;
      * lora — non-empty lora_A/lora_B parameters that require grad,
        counted per wrapped module (r=0 empties contribute nothing and
        are not counted)."""
    luts_mods, luts_params = 0, []
    for _name, mod in pmod.iter_palettized_linears(shell):
        if isinstance(mod.lut, nn.Parameter) and mod.lut.requires_grad:
            luts_mods += 1
            luts_params.append(mod.lut)
    norms_mods, norms_params = 0, []
    for name, mod in shell.named_modules():
        if not name or not any(name.endswith(s)
                               for s in _NORM_GAIN_SUFFIXES):
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, nn.Parameter) and w.requires_grad:
            norms_mods += 1
            norms_params.append(w)
    lora_by_mod = {}
    for name, p in shell.named_parameters():
        if not (name.endswith(".lora_A") or name.endswith(".lora_B")):
            continue
        if not p.requires_grad or p.numel() == 0:
            continue
        lora_by_mod.setdefault(name.rsplit(".", 1)[0], []).append(p)
    return {
        "luts": _census_record(luts_mods, luts_params),
        "norms": _census_record(norms_mods, norms_params),
        "lora": _census_record(len(lora_by_mod),
                               [p for ps in lora_by_mod.values()
                                for p in ps]),
    }


def _enumerate_joint_trainable(shell, layer_idx, train_groups,
                                lut_path="reference"):
    """Enumerate the joint trainable set at attach (W4-T01, PROPOSAL
    §2.3) — AFTER `_attach_layer_adapters` froze everything but the
    adapters:

      * luts: `PalettizedLinear.make_trainable()` on every palettized
        module in the layer (fp32 master, cached logical indices,
        indices frozen — constants by construction);
      * norms: `requires_grad_(True)` on the shell's norm gains, per
        the layer's actual module inventory (suffix match);
      * lora: the attached A/B keep their grad; an UNSELECTED lora
        group is re-frozen here (the trainable set is exactly the
        selection).

    Everything else stays frozen: indices, resA/resB, the dense
    remainder, the teacher. Forward values are invariant under the
    enumeration (the fp32 master is the fp16 buffer's exact cast and
    the cached-index gather is reference_dequant's own arithmetic), so
    the baseline/anchor evals are unchanged.

    CUDA contract (PROPOSAL §2.3): the reference path uses
    lut[row_groups].float().gather(1, idx_logical) which is fully
    differentiable on both CPU and CUDA (the straight-through pattern).
    The kernel path (--lut-path kernel) requires both the FLUTE forward
    kernel and the dL/dLUT scatter kernel (PROPOSAL §3).

    Prints the one-line step-0 census; returns the census dict."""
    groups = _parse_train_groups(train_groups) \
        if isinstance(train_groups, str) else frozenset(train_groups)
    if "luts" in groups:
        import qlora_gemm
        kernel_unservable = []
        for name, mod in pmod.iter_palettized_linears(shell):
            mod.make_trainable()
            if mod.lut.device.type != "cuda":
                continue
            if lut_path == "kernel" and qlora_gemm.fused_gemm_eligible(
                    mod, lut_trainable=True):
                continue    # the W5 route: the Function carries dL/dLUT
            if lut_path == "kernel":
                kernel_unservable.append(name)
            # reference path: the gather is differentiable on any device
        if kernel_unservable:
            raise RuntimeError(
                f"[joint-set] layer {layer_idx}: --lut-path kernel was "
                f"selected but {len(kernel_unservable)} CUDA-resident "
                f"module(s) (first {kernel_unservable[0]!r}) are not "
                f"kernel-servable with a trainable LUT "
                f"(fused_gemm_eligible lut_trainable=True is False: the "
                f"FLUTE forward kernel or lut_grad_scatter is "
                f"unavailable) — rebuild both kernels (RUNBOOK §2) or "
                f"run the reference path. Refusing to silently train "
                f"nothing.")
    for name, mod in shell.named_modules():
        if not name or not any(name.endswith(s)
                               for s in _NORM_GAIN_SUFFIXES):
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, nn.Parameter):
            continue
        if "norms" in groups and w.dtype != torch.float32:
            # The dtype ladder (GPU_SPEC §3.3 / PROPOSAL §2.3): trainable
            # parameters are fp32 MASTERS — the LUT channel
            # (make_trainable) and the lora channel (QLoRALinear's A/B)
            # already are; the norm gains must be too. An fp16 gain under
            # AdamW keeps fp16 moments: the second moment v ~ grad^2
            # underflows the fp16 subnormal floor (6e-8) and the FIRST
            # step explodes to NaN (the W4-T04 delta-L2 census caught
            # exactly this on the toy). The forward is bit-invariant
            # under the promotion: the RMSNorm paths cast the gain
            # (.float() / .to(input_dtype), modeling's explicit cast).
            mod.weight = nn.Parameter(w.detach().float().clone())
            w = mod.weight
        w.requires_grad_("norms" in groups)
    for name, p in shell.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            p.requires_grad_("lora" in groups)
    census = _joint_trainable_census(shell)
    print(f"  [L{layer_idx:02d}] step-0 census: "
          + " ".join(
              f"{g}={census[g]['modules']}mod/{census[g]['params']}par/"
              f"{census[g]['bytes']}B/{census[g]['device']}/"
              f"{census[g]['dtype']}"
              for g in _TRAIN_GROUPS)
          + f" train={'+'.join(sorted(groups))}",
          flush=True)
    return census




# ---------------------------------------------------------------------------
# The control plane (W2-T08, PROPOSAL §2.6 — ported, not reinvented)
# ---------------------------------------------------------------------------

def _layer_type(cfg, layer_idx: int) -> str:
    types = getattr(cfg, "layer_types", None) or []
    if layer_idx < len(types):
        return types[layer_idx]
    return "full_attention" if (layer_idx % 4 == 3) else "linear_attention"


def _make_optimizer(names_params, lr: float, kind: str = "adamw",
                    eps: float = 1e-8):
    """Optimizer factory for ONE channel's trainable set.

    adamw: unchanged (betas 0.9/0.999, wd 0); eps is now LIVE from
    --adam-eps (W2-T08 — the engine's factory pinned 1e-8; the R13
    Adam-noise floor makes 1e-15 the real-run value: small parameter
    groups die at eps 1e-8).
    muon: the VENDORED implementation (scripts/muon_optimizer.py —
    fp32 Newton-Schulz5, momentum 0.95, Nesterov, wd 0, RMS-matched
    step scale 0.2*sqrt(max(shape)); see its module docstring). The
    private torch-internal Muon import is GONE: stock CPU wheels do not
    ship it, so this path could never run on the coding box.
    Scope boundary (W4-T02, mechanically enforced): muon's 2-D
    matrix-factor semantics apply ONLY to lora_A/lora_B factors — a
    group carrying anything else (a (groups, 16) LUT master would lose
    its per-group scale structure; a norm gain is not a matrix factor
    at all) is REFUSED with a ValueError here, at construction.
    Muon lr is NOT transferable from AdamW (different step calibration)
    — the G1-O pilot selects the peak lr on evidence."""
    params = [p for _, p in names_params]
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                 eps=float(eps), weight_decay=0.0)
    if kind == "muon":
        non_lora = [n for n, _ in names_params
                    if not (n.endswith(".lora_A")
                            or n.endswith(".lora_B"))]
        if non_lora:
            raise ValueError(
                f"muon is the LORA channel's optimizer only — its group "
                f"carries {len(non_lora)} non-lora parameter(s) (first "
                f"{non_lora[0]!r}): matrix-factor semantics do not apply "
                f"to (groups, 16) LUT codebooks or norm gains (the scope "
                f"boundary, PROPOSAL §2.5)")
        from muon_optimizer import Muon
        return Muon(params, lr=lr, momentum=0.95, nesterov=True,
                    weight_decay=0.0)
    raise ValueError(f"unknown optimizer {kind!r}")


def _joint_param_groups(shell):
    """The layer job's trainable parameters grouped by channel
    (PROPOSAL §2.3/§2.5), read off the LIVE requires_grad state (the
    W4-T01 enumeration already made the state equal the selection):
      * lora — lora_A/lora_B (non-empty; r=0 empties are skipped —
        nothing to train);
      * luts — the trainable fp32 LUT masters (module paths ending in
        `.lut`);
      * norms — the inventory gains (the _NORM_GAIN_SUFFIXES match).
    A requires_grad parameter NO channel claims is a loud error: the
    optimizer's domain and the census must agree exactly (G-J1's
    spirit — grads flow to the three groups and to nothing else)."""
    groups = {"lora": [], "luts": [], "norms": []}
    for name, p in shell.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            if p.numel() > 0:
                groups["lora"].append((name, p))
        elif name.endswith(".lut"):
            groups["luts"].append((name, p))
        elif any(name.endswith(s + ".weight") for s in _NORM_GAIN_SUFFIXES):
            groups["norms"].append((name, p))
        else:
            raise RuntimeError(
                f"_joint_param_groups: {name!r} requires grad but no "
                f"channel claims it (legal channels: lora_A/B, lut "
                f"masters, the norm-gain inventory) — the optimizer and "
                f"the census disagree")
    return groups


def _make_joint_optimizer(param_groups, lr_lora, lr_lut, lr_norm,
                          opt_lora: str = "adamw",
                          adam_eps: float = 1e-8):
    """The joint two-channel optimizer plan (W4-T02, PROPOSAL §2.5):

      * the LORA channel: AdamW (betas 0.9/0.999, eps adam_eps, wd 0)
        at `lr_lora`, or Muon at `lr_lora` when `opt_lora == 'muon'`
        (the pilot-tied alternative) — via `_make_optimizer`, whose
        scope boundary refuses muon for anything but lora factors;
      * the CODES channel (LUT codebooks + norm gains): ALWAYS AdamW
        (same betas/eps/wd) — the codebooks at `lr_lut`, the norm
        gains in their OWN group at `lr_norm`.

    Returns [(optimizer, [base_lr per param group]), ...]:
      * opt_lora='adamw': ONE AdamW whose param_groups are the live
        channels in order (lora, luts, norms) — three groups when all
        channels are selected;
      * opt_lora='muon': [Muon(lora), AdamW(codes)].
    A dead channel contributes no group. The caller applies the
    schedule to every group as base_lr * multiplier, steps every
    optimizer, and clips ONCE over the union of all params (§2.5
    rule 3)."""
    if opt_lora not in ("adamw", "muon"):
        raise ValueError(f"unknown optimizer {opt_lora!r}")
    lora_np = param_groups.get("lora") or []
    code_specs = []
    if param_groups.get("luts"):
        code_specs.append({"params": [p for _, p in param_groups["luts"]],
                           "lr": float(lr_lut)})
    if param_groups.get("norms"):
        code_specs.append({"params": [p for _, p in param_groups["norms"]],
                           "lr": float(lr_norm)})
    plans = []
    if opt_lora == "adamw":
        specs = [{"params": [p for _, p in lora_np], "lr": float(lr_lora)}] \
            if lora_np else []
        specs += code_specs
        if specs:
            plans.append((torch.optim.AdamW(
                specs, lr=float(lr_lora), betas=(0.9, 0.999),
                eps=float(adam_eps), weight_decay=0.0),
                [float(s["lr"]) for s in specs]))
        return plans
    # muon: the lora channel only (codes are always AdamW — the scope
    # boundary refuses matrix-factor semantics for codebooks/gains)
    if lora_np:
        plans.append((_make_optimizer(lora_np, float(lr_lora), opt_lora,
                                      eps=float(adam_eps)),
                      [float(lr_lora)]))
    if code_specs:
        plans.append((torch.optim.AdamW(
            code_specs, lr=float(lr_lut), betas=(0.9, 0.999),
            eps=float(adam_eps), weight_decay=0.0),
            [float(s["lr"]) for s in code_specs]))
    return plans


# ---------------------------------------------------------------------------
# W4-T04 telemetry (PROPOSAL §2.6 + §8 risk 2): the step-0 delta-L2
# census, the per-module code-usage histogram, the VRAM pair line
# ---------------------------------------------------------------------------

# PROPOSAL §8 risk 2: a group is LOW-USAGE when fewer than 14 of its 16
# codes occur in its cached logical indices; a module flags COLLAPSE
# when more than 10% of its groups are low-usage (the mid-run
# icm_reassign_sweep trigger — observed, never auto-applied).
_CODE_USAGE_MIN_DISTINCT = 14
_CODE_USAGE_GROUP_FRACTION = 0.10


def _module_code_usage(mod):
    """Per-group 16-bin code-usage histograms from the module's CACHED
    logical indices (the straight-through cache make_trainable built):
    (bins, n_low, n_groups) with bins[g, c] = how often code c occurs
    in group g's rows, n_low the number of groups using fewer than
    _CODE_USAGE_MIN_DISTINCT distinct codes. None when the module has
    no cached indices (the luts channel is off — the histogram is the
    LUT channel's instrument and never recomputes indices)."""
    idx = getattr(mod, "_idx_logical", None)
    if idx is None or idx.numel() == 0:
        return None
    gs, N = int(mod.group_size), int(mod.N)
    palette = 1 << int(mod.bitwidth)
    n_groups = (N + gs - 1) // gs
    bins = torch.zeros(n_groups, palette, dtype=torch.int64)
    n_low = 0
    for g in range(n_groups):
        rows = idx[g * gs:min((g + 1) * gs, N)]
        b = torch.bincount(rows.reshape(-1), minlength=palette)
        bins[g] = b
        if int((b > 0).sum()) < _CODE_USAGE_MIN_DISTINCT:
            n_low += 1
    return bins, n_low, n_groups


def _code_usage_line(shell, layer_idx, step):
    """One line per eval (W4-T04, PROPOSAL §8 risk 2): per palettized
    module with cached indices, low-usage groups over total groups, and
    the layer's collapse verdict — any module with more than
    _CODE_USAGE_GROUP_FRACTION of its groups below 14/16 distinct codes.
    The luts-off run prints the honest n/a (no cached indices)."""
    prefix = f"model.layers.{layer_idx}."
    parts, collapsed = [], []
    for name, mod in pmod.iter_palettized_linears(shell):
        usage = _module_code_usage(mod)
        if usage is None:
            continue
        _bins, n_low, n_groups = usage
        rel = name[len(prefix):] if name.startswith(prefix) else name
        if rel.endswith(".base"):
            rel = rel[:-len(".base")]
        parts.append(f"{rel}={n_low}/{n_groups}")
        if n_groups > 0 and n_low / n_groups > _CODE_USAGE_GROUP_FRACTION:
            collapsed.append(rel)
    if not parts:
        return (f"        code-usage@{step:5d}: n/a (luts off — no "
                f"cached logical indices)")
    verdict = (f"collapse=yes ({','.join(collapsed)})" if collapsed
               else "collapse=no")
    return (f"        code-usage@{step:5d}: " + " ".join(parts)
            + f" | {verdict}")


def _vram_pair_line(step, device):
    """One line per eval (W4-T04): the VRAM pair —
    torch.cuda.max_memory_allocated() / memory_reserved(), both in GiB.
    Reads the trackers WITHOUT resetting them (the T7 watermarks own
    the reset protocol); CPU prints the honest n/a."""
    if not (torch.cuda.is_available()
            and torch.device(device).type == "cuda"):
        return f"        vram-pair@{step:5d}: n/a (CPU box)"
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    reserved = torch.cuda.memory_reserved() / 2 ** 30
    return (f"        vram-pair@{step:5d}: peak={peak:.2f}GiB "
            f"reserved={reserved:.2f}GiB")


def _step0_delta_census_line(census_pre, param_groups, layer_idx):
    """The delta-L2 census after the FIRST applied optimizer step
    (PROPOSAL §2.6 telemetry, W4-T04): per trainable group, the L2 norm
    of the parameter delta — one line, on-device accumulation, one
    float per group. The detailed per-factor record stays behind
    --dump-step0-diff."""
    parts = []
    for g in _TRAIN_GROUPS:
        pre = census_pre.get(g)
        if not pre:
            continue
        now = [p for _, p in param_groups.get(g, [])]
        sq = None
        for p0, p1 in zip(pre, now):
            term = (p1.detach() - p0).float().pow(2).sum()
            sq = term if sq is None else sq + term
        parts.append(f"{g}={float(sq) ** 0.5:.3e}")
    return f"  [L{layer_idx:02d}] step-0 delta-L2: " + " ".join(parts)


def _cuda_watermark_gib(device=None) -> float:
    """W5.T1 (PROPOSAL §3 T7): ONE CUDA watermark sample in GiB —
    torch.cuda.max_memory_allocated() since the last reset — and
    the peak tracker is RESET afterwards (each boundary sample opens a
    fresh window; the running max lives in the caller's watermarks
    dict, so a spike stays attributable to the phase that produced it).

    0.0 on a CPU run — honest zeros: the watermark instrument is
    CUDA-only (no device, no allocator peak to observe)."""
    if not torch.cuda.is_available():
        return 0.0
    if device is not None and torch.device(device).type != "cuda":
        return 0.0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    return peak


def _vram_gib(device=None):
    """W5.T1 (PROPOSAL §3 T7): the step log's vram= pair —
    (current_allocated, total_usable) in GiB on CUDA
    (torch.cuda.memory_allocated() over the device's total_memory);
    (0.0, 0.0) on a CPU run — honest zeros, same CUDA-only instrument
    as the watermarks."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    if device is not None and torch.device(device).type != "cuda":
        return 0.0, 0.0
    cur = torch.cuda.memory_allocated() / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    return cur, total


# --- W5.T1 (PROPOSAL §3 T7): the phase timer ------------------------------
# One instance per layer job; the train loop wraps its EXACT call sites
# (reader_rows, target_fn, the student forward, loss.backward(), the
# lr-update+opt.step() region) and every _evaluate in phase("...")
# contexts. Pure observation: no control flow inside the wrapped regions
# changes — the context manager only clocks them (perf_counter, counted
# even when the region raises).

_PHASE_NAMES = ("reader", "target", "forward", "backward", "optimizer",
                "eval")


class _PhaseTimer:
    """Wall-clock accumulator per train-loop phase (the T7 instrument).

    Phases are the fixed T7 set — reader, target, forward, backward,
    optimizer, eval — validated on every `phase(name)` call (an unknown
    name is a caller bug: ValueError before anything is timed; the
    names are part of the log-line contract, so a typo must be loud).

    Two accumulator sets per phase: TOTALS (wall seconds + call counts
    over the whole job — `means()` is {phase: mean seconds per call})
    and the WINDOW (reset by `tick_window()`, totals preserved —
    `window_means()` is what the --log-every step line prints, so the
    line reports the window just closed, not a job-long average).
    `last(name)` is the most recent single duration (the eval line's
    ms). Every accessor returns 0.0 for a phase with no calls — honest
    zeros, never a division by zero."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero every accumulator (totals, counts, window, last)."""
        self._totals = {p: 0.0 for p in _PHASE_NAMES}
        self._counts = {p: 0 for p in _PHASE_NAMES}
        self._window = {p: 0.0 for p in _PHASE_NAMES}
        self._window_counts = {p: 0 for p in _PHASE_NAMES}
        self._last = {p: 0.0 for p in _PHASE_NAMES}

    def _check(self, name):
        if name not in self._totals:
            raise ValueError(
                f"_PhaseTimer: unknown phase {name!r} — the T7 phase set "
                f"is {_PHASE_NAMES} (the names are part of the log-line "
                f"contract; a typo here is a caller bug)")
        return name

    @contextlib.contextmanager
    def phase(self, name):
        """Context manager timing ONE region of phase `name`
        (perf_counter; accumulated into totals + window even when the
        region raises — the exception itself always propagates)."""
        self._check(name)
        start = time.perf_counter()
        try:
            yield name
        finally:
            dt = time.perf_counter() - start
            self._totals[name] += dt
            self._counts[name] += 1
            self._window[name] += dt
            self._window_counts[name] += 1
            self._last[name] = dt

    @staticmethod
    def _means(totals, counts):
        return {p: (totals[p] / counts[p] if counts[p] > 0 else 0.0)
                for p in _PHASE_NAMES}

    def means(self):
        """{phase: mean seconds per call} over the WHOLE job (the
        totals — what per-layer metrics.json records as mean_ms)."""
        return self._means(self._totals, self._counts)

    def window_means(self):
        """{phase: mean seconds per call} since the last tick_window()
        (read-only; the window keeps accumulating)."""
        return self._means(self._window, self._window_counts)

    def tick_window(self):
        """Close the window: return its means, then zero the window
        accumulators (totals preserved). The --log-every step line
        consumes exactly this — one loud line per window."""
        means = self.window_means()
        for p in _PHASE_NAMES:
            self._window[p] = 0.0
            self._window_counts[p] = 0
        return means

    def last(self, name):
        """Seconds of the most recent `phase(name)` region (0.0 if the
        phase never ran) — the eval line's ms."""
        self._check(name)
        return self._last[name]

    def totals(self):
        """R4 (recovery campaign): {phase: total wall seconds} over the
        whole job — the eval_share denominator's numerator. Read-only."""
        return {p: self._totals[p] for p in _PHASE_NAMES}

    def counts(self):
        """R4: {phase: call count} — e.g. how many evals the job ran."""
        return {p: self._counts[p] for p in _PHASE_NAMES}


def _eval_attn_cap(batch_rows, budget_gib, num_heads, seq):
    """W3.T1 (PROPOSAL T5): the APPLIED holdout-eval batch size for a
    full-attention layer — min(batch_rows, cap) with

        cap = max(1, floor(budget / (num_heads * seq * seq * 4)))

    P7: at head_dim 256 the fast SDPA kernels are out of reach on this
    box's wheel, so an attention forward materializes the fp32 score
    tensor (B, H, S, S) — 16*2048*2048*4 = 0.268 GB/row at the real
    geometry (the PROPOSAL's "0.268 GiB"; the 2.2/9.0 GiB OOM
    signatures). The budget scaling reproduces the PROPOSAL T5
    arithmetic digit-for-digit: its "4 GiB budget caps eval at 14
    rows" divides the budget by the DECIMAL-GB per-row cost, so
    `budget_gib` scales as 1e9 bytes here — 4.0 -> cap 14 (a strict
    2**30 scaling would give 16). A tiny budget floors to 1 row,
    never 0. Linear-attention layers never call this (their path has
    no (B, H, S, S) score tensor); train batches are never capped."""
    per_row = num_heads * seq * seq * 4
    cap = max(1, math.floor((budget_gib * 1e9) / per_row))
    return min(int(batch_rows), cap)


def _new_eval_profile(kind):
    """R4 (recovery campaign): one eval-call timing breakdown record —
    filled in-place by _evaluate when `profile` is this dict. `kind`
    labels the call site ('before' | 'init' | 'step-<n>' | 'final'), the
    phase sums are wall ms (perf_counter, host-side), and `metric_ms`
    brackets the distill_loss + flat-cos accumulation of the batch. The
    record is the G-T5 instrument: one look shows whether an eval's wall
    is reader I/O, target H2D, the student forward, or the metric
    syncs."""
    return {"kind": str(kind), "batches": 0, "rows": 0,
            "reader_ms": 0.0, "target_ms": 0.0, "forward_ms": 0.0,
            "metric_ms": 0.0}


def _evaluate(layer, position_embeddings, reader_rows, target_fn, rows,
              device, batch_rows, mse_weight, model, output_fn=None,
              input_dtype=torch.float32, cos_weight: float = 1.0,
              profile=None, eval_tensors=None):
    """Holdout evaluation: per-token cosine, flattened cosine, relative MSE
    over the given rows. Computed directly from the memmaps.
    `reader_rows(rows, device)` and
    `target_fn(rows, device)` take row lists (shuffled batches are
    non-contiguous — never slice ranges).

    The prediction is the down_proj tap (the LUT path's alignment point)
    unless `output_fn(x, pos) -> pred` is given — the qlora path passes the
    whole layer output, reusing this exact aggregation for its metrics.

    eval_tensors (R10 perf, default None): a device-RESIDENT pair
    (x_dev, t_dev) covering EXACTLY `rows` (both (R, S, H) fp16 on
    `device`, row i <-> rows[i]; built once per layer job by
    _build_eval_tensors). When given, the per-batch reader/target phases
    become device-side slices — zero memmap reads, zero H2D, zero per-
    batch syncs per eval. Values are bit-identical to the streaming path:
    the store/cache serve fp16 either way and the fp32 cast is a
    deterministic elementwise op wherever it runs.

    R10 perf: the metric math (the per-batch 1-cos mean and the flat
    accumulators) is computed ON-DEVICE and materialized ONCE at the end
    — the old path called distill_loss per batch, whose metrics dict did
    FOUR .item()s (four full CUDA-queue flushes) per batch: on a 20-batch
    eval that is 80 forced syncs interleaved with the forwards. The
    arithmetic (means, weights, order of accumulation) is unchanged.

    F16.1 fix: the flattened cosine accumulates num / sq_p / sq_t over
    batches WITHOUT the old per-batch row-count multiplication (double
    weighting whenever the last batch was partial); the tok_cos and
    rel_mse means stay row-weighted (correct for partial batches)."""
    layer.eval()
    R = len(rows)
    tap = {}
    h = None
    if output_fn is None:
        h = layer.mlp.down_proj.register_forward_hook(
            lambda m, i, o: tap.__setitem__(
                "y", (o[0] if isinstance(o, tuple) else o).detach()))
    try:
        # R10: tok_cos accumulates on-device too (the old per-batch
        # Python float came from distill_loss's .item()); the flat
        # accumulators were already on-device (R4).
        dev = torch.device(device)
        flat_num = torch.zeros((), device=dev, dtype=torch.float64)
        flat_sq_p = torch.zeros((), device=dev, dtype=torch.float64)
        flat_sq_t = torch.zeros((), device=dev, dtype=torch.float64)
        resid_sq = torch.zeros((), device=dev, dtype=torch.float64)
        # float64 accumulation, and the per-batch term is (1 - mean) in
        # fp32 THEN cast — exactly distill_loss's old value pipeline
        # (fp32 loss_cos -> .item() float64 -> *n in Python -> float64
        # sum), so the tok_cos digits are unchanged, not just close.
        cos_loss_sum = torch.zeros((), device=dev, dtype=torch.float64)
        x_dev = t_dev = None
        if eval_tensors is not None:
            x_dev, t_dev = eval_tensors
            if int(x_dev.shape[0]) != len(rows) \
                    or int(t_dev.shape[0]) != len(rows):
                raise RuntimeError(
                    f"_evaluate: eval_tensors carry "
                    f"({int(x_dev.shape[0])}, {int(t_dev.shape[0])}) rows "
                    f"but the caller asked for {len(rows)} — the resident "
                    f"pair must cover exactly `rows`")
        with torch.no_grad():
            for i0 in range(0, len(rows), batch_rows):
                idx = rows[i0:i0 + batch_rows]
                _t0 = time.perf_counter() if profile is not None else 0.0
                if x_dev is not None:
                    # R10: device-resident eval inputs/targets — a dim-0
                    # slice view (rows are ascending, so the slice is
                    # contiguous); no memmap read, no H2D, no sync.
                    x = x_dev[i0:i0 + batch_rows]
                else:
                    x = reader_rows(idx, device, input_dtype)
                _t1 = time.perf_counter() if profile is not None else 0.0
                if t_dev is not None:
                    tgt = t_dev[i0:i0 + batch_rows].float()
                else:
                    tgt = target_fn(idx, device)
                _t2 = time.perf_counter() if profile is not None else 0.0
                if output_fn is not None:
                    # position_embeddings=None: the caller's output_fn
                    # needs no positions (the W4-T05 final-norm job — a
                    # bare RMSNorm has no rotary/mask surface)
                    pos = _pos_batch(position_embeddings, x.shape[0]) \
                        if position_embeddings is not None else None
                    pred = output_fn(x, pos)
                else:
                    _forward_layer(layer, x,
                                   _pos_batch(position_embeddings,
                                              x.shape[0]),
                                   model)
                    pred = tap["y"]
                _t3 = time.perf_counter() if profile is not None else 0.0
                # R10: the per-batch metric math inline, on-device, NO
                # .item() — identical arithmetic to distill_loss's
                # weight=None branch: (1 - mean(cos_tok)) * n_b, and the
                # flat sums accumulate in the same order as before.
                pred32 = pred.float()
                cos_tok = F.cosine_similarity(pred32, tgt, dim=-1)  # (B, S)
                # the term: fp32 (1 - mean), cast to float64, *n in
                # float64, accumulated in float64 — distill_loss's old
                # value pipeline verbatim (see the accumulator comment)
                cos_loss_sum += (1.0 - cos_tok.mean()).double() * len(idx)
                # flattened cosine over the CONCATENATION of all rows
                p = pred32.reshape(-1)
                t = tgt.reshape(-1)
                flat_num += (p * t).sum()
                flat_sq_p += p.pow(2).sum()
                flat_sq_t += t.pow(2).sum()
                resid_sq += (p - t).pow(2).sum()
                if profile is not None:
                    _t4 = time.perf_counter()
                    profile["batches"] += 1
                    profile["rows"] += len(idx)
                    profile["reader_ms"] += (_t1 - _t0) * 1000.0
                    profile["target_ms"] += (_t2 - _t1) * 1000.0
                    profile["forward_ms"] += (_t3 - _t2) * 1000.0
                    profile["metric_ms"] += (_t4 - _t3) * 1000.0
        # ONE materialization point for the whole eval (5 scalars, one
        # queue flush — was 4 per batch)
        flat_num_f = float(flat_num.item())
        flat_sq_p_f = float(flat_sq_p.item())
        flat_sq_t_f = float(flat_sq_t.item())
        resid_sq_f = float(resid_sq.item())
        tok_cos_f = 1.0 - float(cos_loss_sum.item()) / R
    finally:
        if h is not None:
            h.remove()
    den = (flat_sq_p_f * flat_sq_t_f) ** 0.5
    return {
        "tok_cos": tok_cos_f,
        # rel_mse as the RATIO OF TOTALS over the whole holdout (the
        # honest definition — the old row-mean of per-batch ratios was
        # biased whenever the per-batch denominators differed; F16)
        "rel_mse": resid_sq_f / max(flat_sq_t_f, 1e-12),
        "flat_cos": flat_num_f / max(den, 1e-12),
    }


# ---------------------------------------------------------------------------
# Student trajectory cache (rolling, resume-friendly) — the L2 escalation
# ---------------------------------------------------------------------------

class XCache:
    """Rolling per-layer student hidden-state cache on disk.

    x_layer{L}.npy (R, S, H) fp16. x_layer0 is materialized from the
    capture's x1. After layer L is finalized, propagate() fills
    x_layer{L+1} and (by default) deletes x_layerL unless it is a
    keep-checkpoint (every `keep_every` layers, for cheap resume replay).

    open_mmap(L) keeps ONE persistent memmap handle per layer (the F16.2
    fix: the old reader did an np.load per ROW; the layer job's reader
    now slices contiguous runs from this handle)."""

    def __init__(self, cache_dir: str, rows: int, seq: int, hidden: int,
                 num_layers: int, keep_every: int = 8):
        self.dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.rows, self.seq, self.hidden = rows, seq, hidden
        self.num_layers = num_layers
        self.keep_every = keep_every
        self._mm = {}

    def path(self, L: int) -> str:
        return os.path.join(self.dir, f"x_layer{L}.npy")

    def exists(self, L: int) -> bool:
        return os.path.exists(self.path(L))

    def open(self, L: int):
        return _memmap_rw(self.path(L), (self.rows, self.seq, self.hidden))

    def open_mmap(self, L: int):
        """A persistent read-only memmap of x_layer{L} (one np.load for
        the whole layer job — F16.2)."""
        if L not in self._mm:
            self._mm[L] = np.load(self.path(L), mmap_mode="r")
        return self._mm[L]

    def seed_from_capture(self, store: CaptureStore):
        if self.exists(0):
            return
        mm = self.open(0)
        CH = 32
        for i0 in range(0, self.rows, CH):
            i1 = min(i0 + CH, self.rows)
            mm[i0:i1] = store.x1(i0, i1, "cpu", torch.float16).numpy()
        mm.flush()

    def read(self, L: int, i0, i1, device, dtype=torch.float32):
        mm = np.load(self.path(L), mmap_mode="r")
        a = np.ascontiguousarray(mm[i0:i1])
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    def maybe_drop(self, L: int):
        keep = (L % self.keep_every == 0) or L == 0
        if not keep and L + 1 <= self.num_layers and \
                self.exists(L + 1) and self.exists(L):
            os.remove(self.path(L))


# F10 divergence tripwire (PROPOSAL §5, pinned; §2.6 re-stated): the
# train-loss EMA is compared against the mean of the first _TRIP_ARM_STEPS
# losses; EMA above _TRIP_FACTOR x that mean for _TRIP_EVALS consecutive
# evals restores the best snapshot, halves the lr scale and resumes —
# TWO firings max; a third firing STOPS the layer (the R15 lesson: the
# engine fired once and resumed into a still-rotting run).
_TRIP_ARM_STEPS = 50
_TRIP_FACTOR = 2.0
_TRIP_EVALS = 2           # PROPOSAL §2.6: 2 consecutive bad evals
_TRIP_MAX_FIRINGS = 2     # ...2 firings max, the 3rd stops

# R2 (recovery campaign): the warm-start red-flag threshold —
# init/before rel_mse ratio above this is the 'catastrophic' verdict (the
# L03 44x class); 1 < ratio <= 2 is 'regression' (visible, anchor-absorbed).
_WARM_START_RED_FLAG = 2.0


class _PreloadedLayerSource:
    """Duck-typed TeacherLayerSource carrying ONE pre-loaded layer state —
    the materialize_student_layer base_stream seam with a SHARED shard
    read (F7/F16.3: the layer's safetensors shard is read ONCE and feeds
    both the teacher build and the student's dense parts)."""

    def __init__(self, sd, layer_idx):
        self.sd, self.layer_idx = sd, layer_idx

    def layer_state(self, i):
        if int(i) != int(self.layer_idx):
            raise RuntimeError(
                f"preloaded layer state is for layer {self.layer_idx}, "
                f"not {i} — a shared read only serves its own layer")
        return self.sd


def _holdout_split(args, rows_total):
    """(eval_rows, train_rows) per the F14 fix: 'random' (default) is a
    seeded permutation held FIXED across layers and runs (the same rows
    serve every layer, exactly like the old contiguous tail, but now an
    iid sample of the capture order); 'tail' reproduces the legacy
    contiguous tail for old-number comparability."""
    hold = int(round(rows_total * args.holdout))
    if hold > 0:
        if args.holdout_split == "random":
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(rows_total)
            eval_rows = sorted(int(i) for i in perm[:hold])
            train_rows = [i for i in range(rows_total) if i not in
                          set(eval_rows)]
        else:
            eval_rows = list(range(rows_total - hold, rows_total))
            train_rows = list(range(0, rows_total - hold))
    else:
        eval_rows = list(range(0, min(4, rows_total)))
        train_rows = list(range(0, rows_total))
    if not train_rows:
        raise SystemExit("holdout consumes all rows; reduce --holdout")
    return eval_rows, train_rows


class _TargetCache:
    """The layer's training/eval targets, computed ONCE by a single
    resident-teacher pass over every row (F12): in captured-input mode
    the target is a pure function of the resident teacher and the frozen
    inputs — recomputing it per step (the old loop, 1000+ teacher
    forwards per layer) and per eval batch (154 rows x 20 evals) was
    pure waste, and the operator's dual-stream copy machinery that
    existed only to hide that waste was dead code (nothing ever called
    it — P6 deleted it, and the cross-stream ordering hazard went with
    it).

    Storage: one fp16 pinned CPU tensor (rows, S, H) — ~12.9 GB host RAM
    on the real 768 x 2048 x 4096 geometry.

    I/O (W3.T2 / PROPOSAL T6): rows() takes a strictly-ascending row
    list, splits it into contiguous runs, and slices the cache per run
    (a contiguous VIEW — no advanced-index gather), one fp16 H2D per
    run — half the bytes of the old fp32 copy — with a single
    on-device fp32 cast after the cat. The build is a single teacher
    pass over every row; there is no second CUDA stream: non_blocking
    is requested only when the run slice is actually pinned (a slice
    shares its source's storage, so a slice of a pinned tensor is
    pinned), and pageable transfers stay synchronous."""

    def __init__(self, rows, seq, hidden, pin, device=None):
        self.t = torch.zeros(rows, seq, hidden, dtype=torch.float16,
                             pin_memory=pin)
        self._device = device

    @classmethod
    def build(cls, teacher, shell, reader_rows, rows_total, seq, hidden,
              device, batch_rows, ctx):
        pin = (torch.cuda.is_available()
               and (torch.device(device).type == "cuda"))
        cache = cls(rows_total, seq, hidden, pin, device=device)
        pos16 = _position_embeddings(
            shell, cache.t.shape[1], torch.float16, device)
        with torch.no_grad():
            for i0 in range(0, rows_total, batch_rows):
                i1 = min(i0 + batch_rows, rows_total)
                idx = list(range(i0, i1))
                x16 = reader_rows(idx, device, torch.float16)
                y = _forward_layer(teacher, x16,
                                   _pos_batch(pos16, x16.shape[0]), shell)
                # R10 (perf): async D2H straight into the PINNED cache
                # slice — copy_ into pinned memory with non_blocking=True
                # returns control to the host immediately instead of the
                # old .cpu() (a pageable D2H that stalled the pipeline on
                # every one of the ~96 build batches). Same stream, so the
                # later rows() H2D reads are ordered after their writes.
                cache.t[i0:i1].copy_(
                    y.detach().to(torch.float16), non_blocking=True)
                del x16, y
                # W3.T1: per-batch empty_cache removed (P5 allocator
                # thrash); hygiene stays phase-boundary only
        if cache.t.is_pinned():
            # belt-and-braces: the pinned writes are stream-ordered, but
            # the build's END is a natural phase boundary — flush once
            # before the cache serves any later host-side consumer.
            torch.cuda.synchronize() if torch.cuda.is_available() \
                else None
        return cache

    def rows(self, idx, device):
        """Gather target rows onto `device` as fp32, in the caller's
        row order.

        Contract (the `_split_runs` one): `idx` is a strictly ascending
        list of row indices in [0, rows) — the trainer sorts every batch
        at the single point it is drawn, so a non-ascending list is a
        caller bug that raises loudly here (silently sorting would
        reorder the caller's rows against their targets). An empty list
        returns an empty (0, S, H) fp32 tensor on `device`.

        Value parity with the old advanced-index gather: slicing the
        SAME fp16 tensor per contiguous run and concatenating in run
        order selects exactly rows `idx` — bit-equal, because the cache
        stores fp16 either way and the fp16->fp32 cast is a
        deterministic elementwise op wherever it runs."""
        idx = [int(r) for r in idx]
        if not idx:
            return torch.empty(0, self.t.shape[1], self.t.shape[2],
                               device=device, dtype=torch.float32)
        # loud bounds check: a slice would silently CLAMP an
        # out-of-range run where the old gather raised IndexError —
        # never a silent no-op
        n = self.t.shape[0]
        if idx[0] < 0 or idx[-1] >= n:
            raise IndexError(
                f"_TargetCache.rows: row indices must be in [0, {n}) — "
                f"got {idx!r}")
        parts = []
        for i0, i1 in _split_runs(idx):
            src = self.t[i0:i1 + 1]        # contiguous view, no gather
            # one fp16 H2D per run — half the bytes of the old fp32
            # transfer. The slice shares the pinned storage (a slice of
            # a pinned tensor is pinned), so it needs no staging copy;
            # non_blocking is requested only when it is actually
            # pinned, keeping pageable transfers synchronous.
            parts.append(src.to(device=device, dtype=torch.float16,
                                non_blocking=src.is_pinned()))
        # one cat, then a single on-device fp32 cast
        return torch.cat(parts, dim=0).float()


def _apply_init_snapshot(shell, layer_idx, acfg, init_dir, ctx):
    """Stage 1.5 (PROPOSAL §3.4): initialize this layer's adapters from a
    completed run's per-layer snapshot (the Stage-1 result — the
    trajectory sweep CONTINUES from it, warm start = Stage-1, not
    re-derived). Geometry is checked against the freshly attached
    config (QKV components compared per component); a layer with no
    snapshot (r=0 / not trained) stays at B=0 and returns False."""
    snap_path = os.path.join(init_dir, "qlora_layers",
                             f"layer_{layer_idx}", "adapter.pt")
    if not os.path.isfile(snap_path):
        return False
    snap = torch.load(snap_path, map_location="cpu")

    def _geom_r(d):
        if not isinstance(d, dict):
            return None
        if "components" in d:
            return {c: (v.get("r") if isinstance(v, dict) else v)
                    for c, v in d["components"].items()}
        return d.get("r")

    for mod, g in snap.get("geometry", {}).items():
        if _geom_r(g) != _geom_r(acfg.tensors.get(mod)):
            raise RuntimeError(
                f"{ctx}: init-from snapshot geometry for {mod} "
                f"({_geom_r(g)}) disagrees with the attached adapter "
                f"({_geom_r(acfg.tensors.get(mod))}) — re-run with the "
                f"SAME rank map as the source run")
    _load_lora_state_dict(shell, snap["lora"], ctx=f"{ctx} init snapshot")
    return True


def _lora_snapshot(shell):
    """R15 / PROPOSAL §2.6: a TRUE in-memory copy of the adapter state
    (for the best banked snapshot). A bare `p.detach().cpu()` is an
    ALIAS, not a copy — on an already-CPU param `.cpu()` returns self
    and `.detach()` shares storage, so an optimizer's in-place add_
    mutates the "snapshot" too (fatal for a snapshot that must survive
    N more optimizer steps; the R15 restore asserts caught exactly
    this: the tripwire/final "restore" silently no-opped). The
    `.clone()` here is the fix — the engine's layer job renamed to this
    convention when the control plane ported (W2-T08).

    Key layout: {<full module path>.lora_A/.lora_B: tensor} — the same
    layout save_qlora writes for a full model."""
    return {n: p.detach().cpu().clone()
            for n, p in shell.named_parameters()
            if n.endswith(".lora_A") or n.endswith(".lora_B")}


def _joint_snapshot(shell):
    """W4-T02 / PROPOSAL §2.6: the true-clone banking lesson applied to
    LUT and norm tensors — a TRUE in-memory copy of the WHOLE trainable
    set (every requires_grad parameter: lora A/B, the fp32 LUT masters,
    the inventory norm gains). The joint channel's banked best / anchor
    / tripwire restores use THIS snapshot: restoring only the lora
    tensors would leave a diverged run's LUT/norm drift in place and
    break the never-finish-worse guarantee.

    Key layout: {<full module path>: tensor} over requires_grad
    parameters — a superset of `_lora_snapshot`'s keys. The adapter-dir
    export still reads the LORA-ONLY subset (`_lora_snapshot`) so the
    qlora_adapters.pt / Stage-1.5 key contract is unchanged."""
    return {n: p.detach().cpu().clone()
            for n, p in shell.named_parameters()
            if p.requires_grad}


def _load_lora_state_dict(shell, sd, ctx):
    """Copy an adapter state dict into the shell's lora parameters (strict:
    missing keys, unknown keys and shape mismatches are loud)."""
    named = dict(shell.named_parameters())
    unknown = [k for k in sd if k not in named]
    if unknown:
        raise RuntimeError(
            f"{ctx}: {len(unknown)}/{len(sd)} adapter keys not found in the "
            f"layer (first: {unknown[0]}) — geometry mismatch, refusing to "
            f"silently skip")
    for k, v in sd.items():
        p = named[k]
        if tuple(v.shape) != tuple(p.shape):
            raise RuntimeError(
                f"{ctx}: adapter {k} shape {tuple(v.shape)} != parameter "
                f"shape {tuple(p.shape)} — rank geometry mismatch")
        p.data.copy_(v.to(p.device, p.dtype))


def _guard_frozen_paths_cuda(root, layer_idx, device):
    """W2.T1 s3 / operator T10 (always-fused): the layer-job startup guard.

    Runs right after _attach_layer_adapters on a CUDA layer job, BEFORE any
    forward (including the baseline eval): every QLoRALinear component in
    the student must sit on `fused-flute` or the EXPLICIT
    `torch-gpu-cached` opt-out (paths are resolved once at attach in
    qlora.py — this catches e.g. a reference-cpu path frozen before a
    later .to("cuda")), AND the fused backward kernel must be available
    unless the FLUTE_FUSED_BWD=0 escape hatch is set explicitly. Any
    violation is a hard startup RuntimeError with the build command and
    the offending module list — a CUDA job never runs degraded and never
    degrades silently. CPU jobs return silently (reference-cpu is the
    legal, default CPU path)."""
    if not (torch.cuda.is_available()
            and torch.device(device).type == "cuda"):
        return
    import qlora_gemm  # local: only CUDA jobs pay the import
    bad = []
    for name, mod in iter_qlora_modules(root):
        if not isinstance(mod, QLoRALinear):
            continue  # a QLoRASplitQKV: its q/k/v QLoRALinear children
                      # are yielded below in their own right
        path = getattr(mod, "_frozen_path", None)
        if path not in ("fused-flute", "torch-gpu-cached"):
            bad.append(f"{name} ({path!r})")
    problems = []
    if bad:
        problems.append(
            "QLoRALinear component(s) not on a CUDA-legal frozen path "
            f"(fused-flute | torch-gpu-cached): {', '.join(bad)}")
    # Mirror qlora_gemm's own escape semantics (FLUTE_FUSED_BWD unset/1 =
    # fused backward required; "0" — and the empty value it also treats
    # as disabled — is the explicit opt-out).
    if os.environ.get("FLUTE_FUSED_BWD", "1") not in ("", "0") \
            and not qlora_gemm.fused_backward_available():
        problems.append(
            "flute_train_kernels fused backward unavailable "
            "(fused_backward_available() is False)")
    if problems:
        raise RuntimeError(
            f"[qlora-guard] layer {layer_idx}: T10 always-fused violated — "
            + "; ".join(problems)
            + ". Build the fused backward kernel: "
              "cd flute_train_kernels && python setup.py build_ext "
              "--inplace (forward: cd flute_extended && python setup.py "
              "build_ext --inplace). Explicit escape hatches: "
              "FLUTE_FROZEN_PATH=torch (frozen-path opt-out), "
              "FLUTE_FUSED_BWD=0 (fused-backward opt-out).")


def _pin_student_attn(student, L, block_type, cfg, device, ctx):
    """W4.T3 (operator T9): pin the STUDENT's full-attention layer to the
    SM86 Triton flash-attention branch (modeling's flute_sm86
    interface branch -> attn_sm86.flute_sm86_attention).

    The only sanctioned way onto the kernel path: every attention
    submodule of the materialized student layer (Qwen3_5DecoderLayer
    exposes `.self_attn`; the walk below is re-nest-proof) gets its OWN
    config copy — `copy.copy(module.config)` with
    `_attn_implementation = "flute_sm86"` — so the MODEL-level cfg object
    is NEVER mutated: the resident teacher layer (built from that same
    object before this call) keeps torch SDPA, and mask construction
    (`_layer_mask` / `_attn_impl` read the model-level config) keeps its
    sdpa semantics (packed rows -> mask=None, exactly the branch's
    contract). Linear-attention layers have no Qwen3_5Attention submodule
    — no pin, no print (the branch only exists in full-attention modules).

    FLUTE_ATTN_IMPL is the ONLY override ("" counts as unset, the
    FLUTE_FROZEN_PATH convention): "flute_sm86" (default) | "sdpa" (the
    reference backend on every box); any other value is a LOUD ValueError
    — never a silent guess.

    `device` is part of the job call frame; the pin itself is
    device-independent (the modeling branch resolves the kernel/reference
    routing per box).

    Returns the RESOLVED implementation string for the W2.T2 path report's
    `attn_impl` field (metrics.json + provenance): "flute_sm86" | "sdpa"
    for full-attention layers, "linear" for linear-attention layers."""
    if block_type != "full_attention":
        return "linear"          # no attention submodule: no pin, no print
    # teacher-safety precondition: the model-level cfg object IS the
    # teacher layer's config object (TeacherLayerSource.load_layer builds
    # from the same object) — it must be sdpa-pinned BEFORE we copy away
    # from it; the caller re-asserts the teacher modules after the pin.
    _ti = getattr(cfg, "_attn_implementation", None)
    assert _ti == "sdpa", (
        f"{ctx}: model-level config._attn_implementation is {_ti!r}, "
        f"expected 'sdpa' — the teacher must keep torch SDPA (the "
        f"flute_sm86 pin is student-only)")
    impl = os.environ.get("FLUTE_ATTN_IMPL") or "flute_sm86"
    if impl not in ("flute_sm86", "sdpa"):
        raise ValueError(
            f"{ctx}: FLUTE_ATTN_IMPL={impl!r} is not one of ('flute_sm86', "
            f"'sdpa') — refusing to guess the student's attention backend "
            f"(unset -> flute_sm86, 'sdpa' -> reference backend)")
    if impl == "sdpa":
        print(f"[L{L:02d}] student attn: sdpa "
              f"(FLUTE_ATTN_IMPL=sdpa override)", flush=True)
        return "sdpa"
    M = _modeling()
    pinned = 0
    for _name, module in student.named_modules():
        if isinstance(module, M.Qwen3_5Attention):
            # shallow copy is enough: only the _attn_implementation
            # attribute diverges, and it lives in the copy's own __dict__
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "flute_sm86"
            pinned += 1
    if pinned == 0:
        raise RuntimeError(
            f"{ctx}: full-attention layer exposes no Qwen3_5Attention "
            f"submodule to pin — the modeling contract changed (the "
            f"flute_sm86 branch lives in Qwen3_5Attention.forward)")
    print(f"[L{L:02d}] student attn pinned: flute_sm86 (kernel path; "
          f"teacher stays sdpa)", flush=True)
    return "flute_sm86"


def _attention_probe(student, L, cfg, device, attn_impl, seq):
    """W5.T1 (PROPOSAL §3 T8): the P7 instrument — ONE probe attention
    call at the layer's EXACT geometry (B=1, H=cfg.num_attention_heads,
    H_kv=cfg.num_key_value_heads, S=`seq` — the job's capture seq, D=the
    student's actual head_dim), fp16, at full-attention layer-job start
    (after the pin, before any training). CUDA runs only.

    Two peak-allocation deltas are measured at that geometry:
      * the kernel probe — attn_sm86.sm86_attention_forward, the SAME
        Triton kernels the student's pinned flute_sm86 branch launches
        (the student's actual attention path; called directly — the
        simplest seam — with no autograd graph);
      * the math/eager reference probe — attn_sm86.reference_attention_
        forward, the SDPA-math numerics (repeat_kv + fp32 softmax) that
        materializes the (B, H, S, S) score tensor at head_dim 256 — the
        P7 class (0.268 GiB/row at the real geometry). A reference delta
        >= 0.20 GiB is reported as the SCORE-MATERIALIZATION SIGNATURE
        explicitly, so the 2.2/9.0 GiB OOM signatures show up here as an
        attributable spike instead of an OOM autopsy (the T5 eval cap is
        what keeps them inside budget).

    The assertion line records the resolved backends — student from the
    pin result (flute_sm86 | sdpa), teacher always torch SDPA (the
    W4.T3 invariant) — and is printed on EVERY box, CPU included. On a
    CPU run the probe itself is skipped with an explicit line (the T8
    instrument is CUDA-only) and None is returned — honest, never a
    silent no-op. Pure observation: every probe tensor is freed before
    returning; nothing about the training that follows changes."""
    H = int(getattr(cfg, "num_attention_heads", 0) or 0)
    H_kv = int(getattr(cfg, "num_key_value_heads", 0) or 0) or H
    # the layer's EXACT geometry: prefer the student's actual attention
    # module (head_dim + scaling exactly as materialized); cfg fallback
    attn_mod = None
    for _name, _m in student.named_modules():
        if isinstance(_m, _modeling().Qwen3_5Attention):
            attn_mod = _m
            break
    D = int(getattr(attn_mod, "head_dim", 0) or 0) if attn_mod else 0
    scale = float(getattr(attn_mod, "scaling", 0.0) or 0.0) \
        if attn_mod else 0.0
    if D <= 0:
        D = int(getattr(cfg, "head_dim", 0) or 0) \
            or (int(cfg.hidden_size) // max(1, H))
    if scale <= 0.0:
        scale = D ** -0.5
    S = int(seq)

    if not (torch.cuda.is_available()
            and torch.device(device).type == "cuda"):
        print(f"[L{L:02d}] attn probe: skipped (CPU box — the T8 "
              f"instrument is CUDA-only)", flush=True)
        print(f"[L{L:02d}] student_attn={attn_impl} teacher_attn=sdpa",
              flush=True)
        return None

    import attn_sm86 as _attn
    q = torch.randn(1, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(1, H_kv, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(1, H_kv, S, D, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated() / 2 ** 30

    # -- probe 1: the kernel path (the student's actual attention) -----
    # D=256 is the SM86 design geometry (attn_sm86's own loud contract);
    # a CUDA triton build is required. A geometry/tooling miss skips the
    # kernel probe with an explicit note — the reference probe below
    # still runs (observation never refuses to observe).
    kernel_note = ""
    ok, why = _attn.kernel_available()
    if D == 256 and ok:
        torch.cuda.reset_peak_memory_stats()
        out = _attn.sm86_attention_forward(q, k, v, scale)
        torch.cuda.synchronize()
        kernel_peak = torch.cuda.max_memory_allocated() / 2 ** 30 - base
        del out
    else:
        kernel_peak = 0.0
        kernel_note = ("; flute_sm86 probe unavailable "
                       + (f"(head_dim {D} != the 256 design geometry)"
                          if D != 256 else f"(no CUDA triton: {why})"))

    # -- probe 2: the math/eager reference (the P7 signature) ----------
    torch.cuda.reset_peak_memory_stats()
    out = _attn.reference_attention_forward(q, k, v, scale)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated() / 2 ** 30 - base
    del out
    del q, k, v
    torch.cuda.empty_cache()

    signature = ref_peak >= 0.20
    note = (("SCORE-MATERIALIZATION SIGNATURE (B,H,S,S) fp32"
             if signature else "no score materialization") + kernel_note)
    print(f"[L{L:02d}] attn probe: flute_sm86 peak=+{kernel_peak:.3f} GiB "
          f"| math/eager reference peak=+{ref_peak:.3f} GiB — {note}",
          flush=True)
    print(f"[L{L:02d}] student_attn={attn_impl} teacher_attn=sdpa",
          flush=True)
    return {"geometry": {"B": 1, "H": H, "H_kv": H_kv, "S": S, "D": D},
            "flute_sm86_peak_gib": round(kernel_peak, 3),
            "reference_peak_gib": round(ref_peak, 3),
            "score_materialization": bool(signature),
            "student_attn": attn_impl, "teacher_attn": "sdpa"}


def _report_dequant_paths(root, layer_idx, attn_impl=None):
    """W2.T2 (P3 fix): the frozen dequant-path fact report — ONE table
    per layer job (printed right after the [qlora-guard] startup guard,
    never per step), plus the compact `dequant_path` record that the
    trainer writes into the layer's metrics.json and the run provenance
    (`_atomic_json_dump` on both).

    Purely observational: nothing is re-resolved, changed or asserted
    here — the guard above already enforces the paths on CUDA; this
    prints what WAS resolved. On a CPU run the table prints
    path=reference-cpu for every module — the legal, expected CPU path,
    not an error.

    W4.T3: `attn_impl` (the student's RESOLVED attention backend — the
    pinned value for full-attention layers, "sdpa" when FLUTE_ATTN_IMPL
    opts out, "linear" for linear-attention layers) is printed in the
    summary line and recorded in the record. When the caller does not
    pass it (the doctor's pinned students) it is read off the attention
    modules themselves — truthful resolution, never re-derived policy.

    Prints (PROPOSAL §6 doctor contract, one line per module):
      [L00] mlp.gate_proj        ( 12288,  4096)  path=fused-flute
    then the summary line:
      [L00] frozen paths: fused-flute: N | torch-gpu-cached: M |
            reference-cpu: K | bwd=fused-kernel|cached-cuBLAS |
            attn=flute_sm86|sdpa|linear

    Returns {"counts": {path: n}, "modules": [[name, path], ...],
    "bwd": "fused-kernel" | "cached-cuBLAS", "attn_impl": str} (JSON-safe:
    counts and per-module (name, path) pairs only — N/K/cached_w_bytes
    live in the walker's own dict for the doctor subcommand)."""
    rep = report_frozen_paths(root)
    for m in rep["modules"]:
        print(f"[L{layer_idx:02d}] {m['name']:<24s} "
              f"({m['N']:>6d},{m['K']:>6d})  path={m['path']}", flush=True)
    if attn_impl is None:
        # W4.T3 auto-resolution (unpinned callers): read the attention
        # modules' own configs; a layer with no full-attention module is
        # "linear" by block type
        _mods = [m for m in root.modules()
                 if isinstance(m, _modeling().Qwen3_5Attention)]
        attn_impl = ((getattr(_mods[0].config, "_attn_implementation",
                              None) or "eager") if _mods else "linear")
    c = rep["counts"]
    print(f"[L{layer_idx:02d}] frozen paths: "
          f"fused-flute: {c['fused-flute']} | "
          f"torch-gpu-cached: {c['torch-gpu-cached']} | "
          f"reference-cpu: {c['reference-cpu']} | bwd={rep['bwd']} | "
          f"attn={attn_impl}",
          flush=True)
    return {"counts": dict(c),
            "modules": [[m["name"], m["path"]] for m in rep["modules"]],
            "bwd": rep["bwd"],
            "attn_impl": attn_impl}


def _run_o1_probe(args, out_dir, layer, device, n_done):
    """The F8 corrective: run the existing O-1 paired probe (a subprocess
    against the CURRENT assembled adapter dir) every --o1-even completed
    layers. Non-fatal (a probe failure is logged loudly, the run
    continues); skipped with a notice when CUDA is unavailable (the CPU
    toy box) — the probe needs the real model. The default probe script
    (o1_baseline_check.py) is not shipped in this repo — supply
    --o1-probe-cmd to wire an external probe."""
    if not torch.cuda.is_available():
        if not getattr(_run_o1_probe, "_noted", False):
            _run_o1_probe._noted = True
            print("  [o1] CUDA unavailable — the O-1 inter-stage probe is "
                  "skipped (run on the GPU box)", flush=True)
        return
    probe_dir = os.path.join(out_dir, "o1_probes")
    os.makedirs(probe_dir, exist_ok=True)
    log_path = os.path.join(probe_dir, f"probe_L{layer:02d}.log")
    default_script = os.path.join(_HERE, "o1_baseline_check.py")
    if not args.o1_probe_cmd and not os.path.isfile(default_script):
        if not getattr(_run_o1_probe, "_no_script", False):
            _run_o1_probe._no_script = True
            print("  [o1] o1_baseline_check.py is not part of this repo — "
                  "the O-1 inter-stage probe is skipped (pass "
                  "--o1-probe-cmd to wire an external probe)", flush=True)
        return
    tmpl = args.o1_probe_cmd or (
        "{python} scripts/o1_baseline_check.py --model {model} "
        "--artifacts-dir {artifacts} --adapters-dir {out} --device "
        "{device} --tag stage1_L{layer}")
    cmd = tmpl.format(python=sys.executable, model=shlex.quote(args.model),
                      artifacts=shlex.quote(args.artifacts_dir),
                      out=shlex.quote(out_dir), device=shlex.quote(device),
                      layer=layer)
    print(f"  [o1] probe after {n_done} layer(s) -> {log_path}", flush=True)
    try:
        with open(log_path, "w") as log:
            log.write(f"# o1 probe cmd: {cmd}\n")
            subprocess.run(shlex.split(cmd), stdout=log, stderr=log,
                           timeout=3600, check=False)
    except Exception as e:      # non-fatal by design, but LOUD
        print(f"  [o1] WARNING: probe subprocess failed ({e}) — the run "
              f"continues; inspect {log_path}", flush=True)


def _build_eval_tensors(reader_rows, target_fn, eval_rows, device, ctx,
                        eval_batch_rows, num_heads, seq, hidden):
    """R10 (perf): device-RESIDENT holdout tensors for one layer job —
    (x_dev, t_dev), both (R, S, H) fp16 on `device`, row i <-> eval_rows[i].

    WHY: the holdout rows are FIXED for the whole layer job (seeded
    random split), yet every one of the ~12 evals re-read them from the
    capture memmap and re-uploaded them: R rows of scattered one-row
    runs = R memmap slices + R pageable H2D transfers + R ascontiguous-
    array copies PER EVAL — ~2.4 GB of I/O for a 154-row holdout,
    repeated per eval, with a cold page cache turning it into seconds
    of disk reads. Residency pays that ONCE per job (2.4 GB fp16 VRAM
    for x + 2.4 GB for t at the real geometry; ~1.2 GB each at the
    default holdout 0.1) and every eval becomes pure device-side
    slicing + forwards.

    Guard: CUDA only, and only when the tensors fit with headroom —
    free VRAM minus the pair must leave room for the working set
    (LoRA/grad/activations ~2.5 GiB) AND one full-attention eval
    batch's fp32 score tensor (B*H*S^2*4 — the P7 class). Otherwise
    the residency is SKIPPED with a loud line and the job keeps the
    streaming reader (identical values, slower evals — never an
    OOM gambit). Returns None on skip; the caller passes None through
    to _evaluate (the streaming path).

    Values are bit-identical to the streaming path: the store/cache
    serve fp16 either way (the target cache stores fp16; casting its
    fp32 return back to fp16 is the identity on values it came from).
    """
    if torch.device(device).type != "cuda" or not eval_rows:
        return None
    need = 2 * len(eval_rows) * int(seq) * int(hidden) * 2  # fp16 pair
    free_b, total_b = torch.cuda.mem_get_info()
    score_b = eval_batch_rows * int(num_heads or 0) * int(seq) ** 2 * 4
    headroom = 2 * (2 ** 30) + max(score_b, 1 << 30)
    if free_b - need < headroom:
        print(f"  [{ctx}] resident-eval SKIPPED: pair ~{need / 2**30:.1f} "
              f"GiB + headroom {headroom / 2**30:.1f} GiB > free "
              f"{free_b / 2**30:.1f} GiB — evals keep the streaming reader "
              f"(identical values, slower)", flush=True)
        return None
    t0 = time.perf_counter()
    x_dev = reader_rows(list(eval_rows), device, torch.float16)
    t_dev = target_fn(list(eval_rows), device).to(torch.float16)
    dt = time.perf_counter() - t0
    print(f"  [{ctx}] resident-eval tensors: x {tuple(x_dev.shape)} + t "
          f"{tuple(t_dev.shape)} fp16 on-device ({need / 2**30:.2f} GiB, "
          f"{dt:.1f}s once — every eval of this job now slices, no I/O)",
          flush=True)
    return x_dev, t_dev


def _report_kernel_status_once():
    """R10 (operator question 'why is the train kernel not used?'): the
    one-per-process kernel truth table printed at the first layer job's
    start — env vars, the verbatim import state of BOTH kernels, and the
    structural note that the torch opt-out path bypasses the fused
    backward BY DESIGN. The old code swallowed the import errors, so a
    box with an unbuilt kernel showed nothing at all."""
    if getattr(_report_kernel_status_once, "_done", False):
        return
    _report_kernel_status_once._done = True
    env_frozen = os.environ.get("FLUTE_FROZEN_PATH") or "fused (unset)"
    env_bwd = os.environ.get("FLUTE_FUSED_BWD") or "1 (unset)"
    try:
        import qlora_gemm as _qg
        ferr = _qg.flute_import_error()
        berr = _qg.backward_import_error()
        fwd_ok = _qg._check_flute_kernel()
        bwd_ok = _qg.fused_backward_available()
    except Exception as e:
        print(f"[kernel-status] qlora_gemm import failed: {e}", flush=True)
        return
    print(f"[kernel-status] FLUTE_FROZEN_PATH={env_frozen} "
          f"FLUTE_FUSED_BWD={env_bwd} | forward kernel "
          f"{'OK' if fwd_ok else 'UNAVAILABLE'} | fused backward "
          f"{'OK' if bwd_ok else 'UNAVAILABLE'}", flush=True)
    if not fwd_ok:
        print(f"[kernel-status]   forward: {ferr}", flush=True)
        print("[kernel-status]     build: cd flute_extended && python "
              "setup.py build_ext --inplace (needs nvcc whose major "
              "matches torch.version.cuda="
              f"{getattr(torch.version, 'cuda', None)})", flush=True)
    if not bwd_ok:
        print(f"[kernel-status]   backward: {berr}", flush=True)
        print("[kernel-status]     build: cd flute_train_kernels && "
              "python setup.py build_ext --inplace", flush=True)
    if "torch" in str(env_frozen):
        print("[kernel-status]   NOTE: under FLUTE_FROZEN_PATH=torch the "
              "fused backward kernel is BYPASSED BY DESIGN — backward "
              "flows through autograd's matmul on the materialized W16. "
              "Dropping the env (after building BOTH kernels) runs the "
              "deployment-faithful fused path: forward "
              "flute_extended.qgemm_per_group_lut + backward "
              "flute_train_kernels.fused_backward_gemm.", flush=True)


def _assemble_qlora_adapters(out_dir, provenance, rank_map, args,
                             partial=False):
    """Final assembly: concatenate the per-layer adapter state dicts (their
    keys are FULL module paths already — no collisions across layers) into
    the standard adapter dir (qlora_adapters.pt + qlora_config.json incl.
    the FULL rank map), consumable by load_qlora(strict=True) and
    load_qlora_model unchanged.

    F6.3 fix: the trainer runs this after EVERY completed layer (atomic
    tmp+rename writes) — a killed run leaves a loadable PARTIAL adapter
    dir instead of nothing (the first run's kill lost the final
    assembly). `partial=True` (the incremental call) tolerates the
    pre-first-adapter state loudly-but-quietly; the end-of-run call
    stays strict."""
    sd = {}
    tensors = {}
    n_layers = 0
    for key, rec in sorted(provenance.get("layers", {}).items(),
                           key=lambda kv: int(kv[0])):
        if not rec.get("done"):
            continue
        if not rec.get("has_adapter"):
            n_layers += 1
            continue
        snap_path = os.path.join(out_dir, rec["snapshot"])
        if not os.path.exists(snap_path):
            raise RuntimeError(
                f"assembly: provenance marks layer {key} done but "
                f"{snap_path} is missing — incomplete run, re-run with "
                f"--resume")
        snap = torch.load(snap_path, map_location="cpu")
        for k in snap["lora"]:
            if k in sd:
                raise RuntimeError(
                    f"assembly: duplicate adapter key {k!r} — two layer "
                    f"snapshots claim the same module path")
            sd[k] = snap["lora"][k]
        tensors.update(snap.get("geometry", {}))
        n_layers += 1
    if not sd:
        if partial:
            print(f"  [assemble] no trained adapters yet — skipping the "
                  f"incremental assembly", flush=True)
            return 0
        raise RuntimeError(
            "assembly: no trained adapters to assemble (every selected "
            "layer is r=0 or none was run) — nothing was written")
    tmp = os.path.join(out_dir, "qlora_adapters.pt.tmp")
    torch.save(sd, tmp)
    os.replace(tmp, os.path.join(out_dir, "qlora_adapters.pt"))
    qlora_cfg = QLoRAConfig(
        r=_QLORA_DEFAULT_RANK, alpha=_QLORA_DEFAULT_ALPHA, dropout=0.0,
        scope="all", include_residual_branch=True,
        init_a="kaiming_uniform", init_b="zero", base_model=args.model,
        artifacts_dir=args.artifacts_dir, rank_map=rank_map,
        alpha_mode="proportional", tensors=tensors)
    qlora_cfg.to_json(os.path.join(out_dir, "qlora_config.json"))
    print(f"  [assemble] {n_layers} layer(s), {len(sd)} adapter tensors "
          f"-> {out_dir}/qlora_adapters.pt + qlora_config.json", flush=True)
    return len(sd)


# ---------------------------------------------------------------------------
# One capture-first layer job (the whole per-layer lifecycle) — the
# control plane per PROPOSAL §2.6, ported from the engine's
# _run_qlora_layer_job (W2-T08). Deltas vs the engine, both per §2.6:
#   * the tripwire fires on 2 consecutive bad evals, 2 firings max, the
#     3rd STOPS (the engine fired once and resumed into a still-rotting
#     run — the R15 lesson);
#   * the banked snapshots are TRUE clones (`_lora_snapshot`, the R15
#     alias-vs-copy lesson) and the optimizer factory takes --adam-eps.
# ---------------------------------------------------------------------------

def _run_qlora_layer_job(L, cfg, teacher_source, store, args, metadata,
                         rank_map, device, xcache=None, init_dir=None,
                         predicted_rel_mse=0.0, *, teacher=None,
                         target_cache=None):
    """One capture-first layer job (the whole per-layer lifecycle) — the
    SAME trainer with the control plane fixed per PROPOSAL §3.3 (every
    fix maps to its flaw):

      1. ONE shared shard read builds teacher + student (F7/F16.3);
      2. holdout split: seeded random by default (F14);
      3. targets computed ONCE by a single teacher pass (F12), the
         teacher then exits the loop (the second stream is retired);
      4. fresh adapters from the rank map — attached BEFORE the baseline
         (W1.T2/PROPOSAL T2: with B=0 the LoRA branch is exactly zero,
         so the baseline VALUE is unchanged but now runs the wrapper
         path — deployment-faithful);
      5. adapter-free holdout baseline (B=0 identity); warm-start (F2/F5)
         or Stage-1 snapshot (--init-from, Stage 1.5);
      6. SGD on rel_mse + 0.05(1-cos) (F1/F11), batch --batch-rows,
         Muon (pilot-gated) or AdamW, warmup+cosine lr (F5/F6);
      7. banking keyed on holdout rel_mse — ANY improvement banks,
         tie-break on cos; the anchor is the better of {B=0 baseline,
         init state} (F9: never-finish-worse);
      8. working stops: target (rel_mse <= max(5e-4, 1.15x predicted)
         AND cos >= 0.9995), patience, max-steps — all exit the epoch
         loop (F6); startup assert in the caller guarantees
         patience*eval_every < max_steps;
      9. divergence tripwire (F10, PROPOSAL §2.6): train-EMA > 2x
         first-50 mean for 2 evals -> restore best, halve lr, resume,
         2 firings max — the 3rd stops;
     10. per-step telemetry: both loss components' window mean +
         grad-norm (F15);
     11. eval: --eval-batch-rows batches, cached targets, fixed
         flat_cos aggregation (F13/F16.1);
    12. W5.T1 (PROPOSAL §3 T7/T8) telemetry, observation ONLY: a
        _PhaseTimer clocks the reader/target/forward/backward/optimizer
        regions and every _evaluate; CUDA watermarks are sampled at the
        phase boundaries; the step log carries the T7 window line
        (reader/target/fwd/bwd/opt/vram/reader_amp); full-attention jobs
        run the T8 attention probe at start. Per-layer metrics.json
        gains phases/watermarks/reader_amp.

    Returns a dict (layer, block_type, before, after, steps, lora state
    dict, geometry, rank slice, skipped flag, wall_s + stop_reason,
    tripwire record, init metrics, phases/watermarks/reader_amp).

    W5.T3 seam (the G1-O pilot harness): the optional KEYWORD-ONLY
    `teacher` / `target_cache` params inject the CALLER's shared objects
    (the pilot loads the teacher layer once per LAYER and builds the
    target cache once, then runs its whole config grid through this
    UNCHANGED job). Default None on both = today's behavior exactly: the
    job builds and manages its own teacher pass and cache."""
    block_type = _layer_type(cfg, L)
    t0 = time.time()

    # R10: the kernel truth table — once per process, first job start.
    _report_kernel_status_once()

    # ---- W5.T1 (PROPOSAL §3 T7/T8): phase telemetry + watermarks -----
    phase_timer = _PhaseTimer()
    _reader_counters_reset()
    watermarks = {"peak_vram_gib": 0.0, "attn_probe": None}

    def _watermark():
        """One T7 boundary sample: fold the CUDA peak since the last
        reset into the job's running max, then reset the tracker."""
        w = _cuda_watermark_gib(device)
        if w > watermarks["peak_vram_gib"]:
            watermarks["peak_vram_gib"] = round(w, 3)
        return w

    def _telemetry():
        """The job's T7/T8 record: phase means (ms, job-long totals),
        the watermarks (running peak + the attn probe), the final
        reader_amp — plus the R4 additions: phase TOTAL seconds (the
        honest denominators: means hide how many evals ran), the eval
        call COUNT, and eval_share = eval wall / job wall (the G-T5
        gate's exact measurement; the target is <= 10%)."""
        tot = phase_timer.totals()
        cnt = phase_timer.counts()
        wall = max(1e-9, time.time() - t0)
        return {"phases": {k: round(v * 1000.0, 3)
                           for k, v in phase_timer.means().items()},
                "phase_totals_s": {k: round(v, 3) for k, v in tot.items()},
                "eval_calls": int(cnt.get("eval", 0)),
                "eval_share": round(tot.get("eval", 0.0) / wall, 4),
                "watermarks": dict(watermarks),
                "reader_amp": round(_reader_amp(), 4)}

    # ---- R4 (recovery campaign): opt-in per-eval timing breakdowns ------
    eval_profiles = []

    def _maybe_profile(kind):
        if int(getattr(args, "profile_eval", 0) or 0) > 0 \
                and len(eval_profiles) < int(args.profile_eval):
            prof = _new_eval_profile(kind)
            eval_profiles.append(prof)
            return prof
        return None

    def _close_profile(prof):
        """Stamp the eval call's whole wall (the phase timer's last
        region) onto its profile record; None-safe."""
        if prof is not None:
            prof["wall_ms"] = round(phase_timer.last("eval") * 1000.0, 3)
            prof["mean_batch_ms"] = round(
                prof["wall_ms"] / max(1, prof["batches"]), 3)

    _watermark()        # T7: job-start boundary

    # ---- W3.T1 (PROPOSAL T5): eval attention budget + train guard -----
    num_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    if block_type == "full_attention" and num_heads > 0:
        effective_eval_rows = _eval_attn_cap(
            args.eval_batch_rows, args.eval_attn_budget_gib,
            num_heads, store.seq)
        eval_attn_cap = {
            "requested": int(args.eval_batch_rows),
            "applied": int(effective_eval_rows),
            "budget_gib": float(args.eval_attn_budget_gib),
            "block_type": block_type,
        }
        if effective_eval_rows < args.eval_batch_rows:
            print(f"  [L{L:02d}] eval attn budget: eval_batch_rows "
                  f"{args.eval_batch_rows} -> {effective_eval_rows} "
                  f"(full-attention, {args.eval_attn_budget_gib} GiB / "
                  f"{num_heads} heads / S={store.seq})", flush=True)
    else:
        effective_eval_rows = args.eval_batch_rows
        eval_attn_cap = None
    # s3 train-side guard (once per job, CUDA runs only): the train
    # batch is NEVER reduced, but a batch whose fp32 score tensor would
    # eat more than half the free VRAM gets ONE loud warning.
    if torch.cuda.is_available() \
            and torch.device(device).type == "cuda" and num_heads > 0:
        free_b, _total = torch.cuda.mem_get_info()
        score_b = args.batch_rows * num_heads * store.seq ** 2 * 4
        if score_b > 0.5 * free_b:
            print(f"  [L{L:02d}] WARNING: train batch {args.batch_rows} "
                  f"rows x fp32 scores ~= {score_b / 2 ** 30:.2f} GiB > "
                  f"50% free VRAM ({free_b / 2 ** 30:.1f} GiB) — expect "
                  f"the P7 allocation class; consider --batch-rows lower",
                  flush=True)

    # ---- F7/F16.3: ONE shared shard read feeds teacher AND student ----
    sd = teacher_source.layer_state(L)
    # W5.T3 seam: an injected teacher (the pilot's shared, resident layer
    # — loaded once per LAYER, shared by every config of the grid) is
    # used as-is, so the shard read above feeds the student alone. The
    # injected teacher's lifecycle stays with the caller (the release
    # block below is skipped).
    teacher_injected = teacher is not None
    if not teacher_injected:
        teacher = teacher_source.load_layer(L, cfg, sd=sd)
    student, n_swapped = materialize_student_layer(
        L, cfg, _PreloadedLayerSource(sd, L), args.artifacts_dir, metadata,
        device=device, dtype=torch.float16)
    shell = StudentLayerShell(student, L, cfg, device=device)

    # ---- W4.T3 (operator T9): pin the student's full-attention blocks --
    # to the SM86 Triton flash-attention branch (per-module config COPY;
    # FLUTE_ATTN_IMPL is the only override). Returns the resolved backend
    # for the W2.T2 path report below.
    attn_impl = _pin_student_attn(student, L, block_type, cfg, device,
                                   ctx=f"L{L}")
    # teacher keeps torch SDPA (the W4.T3 invariant): the teacher layer
    # was built from the SAME model-level cfg object BEFORE the pin, so
    # every teacher attention module — and cfg itself — must still read
    # sdpa now (a copy-pinned student must never leak into the teacher's
    # dense reference numerics)
    for _m in teacher.modules():
        if isinstance(_m, _modeling().Qwen3_5Attention):
            _t_impl = getattr(_m.config, "_attn_implementation", None)
            assert _t_impl == "sdpa", (
                f"L{L}: teacher attention config is {_t_impl!r}, expected "
                f"'sdpa' — the flute_sm86 pin must never reach the teacher")

    # ---- W5.T1 (PROPOSAL §3 T8): the P7 attention probe --------------
    if block_type == "full_attention":
        watermarks["attn_probe"] = _attention_probe(
            student, L, cfg, device, attn_impl, store.seq)

    # the teacher stays PRISTINE (no norm edits); the shell's config/mask
    # surfaces serve both layers (the old code did exactly this)

    rows_total = store.rows
    eval_rows, train_rows = _holdout_split(args, rows_total)

    if xcache is None:
        def reader_rows(rows, dev, dtype=torch.float32):
            if len(rows) == 0:
                return torch.empty(0, store.seq, store.hidden, device=dev,
                                   dtype=dtype)
            # W1.T1: row-EXACT read — the trainer sorts every batch, so
            # store.h_rows moves ONLY the requested rows (one slice + one
            # H2D per contiguous run).
            return store.h_rows(L, rows, dev, dtype)
    else:
        mm = xcache.open_mmap(L)     # F16.2: persistent mmap, block reads

        def reader_rows(rows, dev, dtype=torch.float32):
            rows = list(rows)
            if len(rows) == 0:
                return torch.empty(0, xcache.seq, xcache.hidden,
                                   device=dev, dtype=dtype)
            # W1.T1: the shared run splitter (the trainer sorts every
            # batch, so non-ascending input now fails loudly instead of
            # degrading into single-row runs). Same slicing semantics as
            # before: one mmap slice per run, one cat, one H2D.
            out = [torch.from_numpy(np.ascontiguousarray(mm[i0:i1 + 1]))
                   for i0, i1 in _split_runs(rows)]
            return torch.cat(out, 0).to(device=dev, dtype=dtype)

    pos16 = _position_embeddings(shell, store.seq, torch.float16, device)

    # ---- F12: the layer's targets, computed ONCE -------------------------
    if target_cache is None:
        tgt_cache = _TargetCache.build(teacher, shell, reader_rows,
                                       rows_total, store.seq, store.hidden,
                                       device, effective_eval_rows, f"L{L}")
    else:
        # W5.T3 seam: the CALLER's cache — the pilot builds it ONCE per
        # layer and shares it across the config grid. The build phase is
        # skipped; the rows()/target plumbing below is unchanged; the job
        # never frees it (caller ownership).
        tgt_cache = target_cache
    _watermark()        # T7: after the target-cache build

    def target_fn(rows, dev):
        return tgt_cache.rows(rows, dev)

    # the teacher exits the inner loop for good (captured-input and
    # trajectory modes alike: the inputs are frozen per layer job) —
    # UNLESS the caller injected it (W5.T3 seam).
    if not teacher_injected:
        del teacher
        teacher_source.drop_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def output_fn(x, pos):
        return _forward_layer(student, x, pos, shell)

    # ---- R10 (perf): device-resident holdout tensors ----------------------
    eval_tensors = _build_eval_tensors(
        reader_rows, target_fn, eval_rows, device, f"L{L:02d}",
        effective_eval_rows, num_heads, store.seq, store.hidden)

    # ---- W1.T2 (PROPOSAL T2): ATTACH before the baseline eval ------------
    acfg, rank_slice = _attach_layer_adapters(shell, L, rank_map, args)
    _watermark()        # T7: after attach

    # ---- W4-T01 (PROPOSAL §2.3): the JOINT trainable set at attach -------
    # Enumerated after the attach froze everything but the adapters; the
    # baseline/anchor evals below are value-invariant under it (the fp32
    # master is the frozen buffer's exact cast). The two-channel
    # optimizer (W4-T02, PROPOSAL §2.5) groups these parameters below.
    train_groups = _train_groups_of(args)
    census = _enumerate_joint_trainable(
        shell, L, train_groups,
        lut_path=str(getattr(args, "lut_path", "reference")))

    # ---- W2.T1 s3: T10 always-fused startup guard (CUDA jobs only) ----
    _guard_frozen_paths_cuda(student, L, device)

    # ---- W2.T2 (P3 fix) + W4.T3: print + record the frozen paths -------
    dequant_path = _report_dequant_paths(student, L, attn_impl)

    # --- holdout baseline: adapter-free by construction (B=0 makes the
    # --- LoRA branch exactly zero), through the SAME output_fn machinery
    # --- the training loop uses --------------------------------------------
    _watermark()        # T7: around-eval (before)
    _prof = _maybe_profile("before")
    with phase_timer.phase("eval"):
        before = _evaluate(student, pos16, reader_rows, target_fn, eval_rows,
                           device, effective_eval_rows, args.mse_weight,
                           shell, output_fn=output_fn,
                           input_dtype=torch.float16,
                           cos_weight=args.cos_weight, profile=_prof,
                           eval_tensors=eval_tensors)
    _close_profile(_prof)
    _watermark()        # T7: around-eval (after)

    if acfg is None:
        # every module of this layer is r=0 (already aligned): no adapter
        # is the CORRECT outcome — record it loudly, never silently
        print(f"  [L{L:02d}] {block_type} rank slice is all r=0 — layer "
              f"already aligned, no adapter attached ({n_swapped} modules "
              f"checked)", flush=True)
        eval_tensors = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"layer": L, "block_type": block_type, "before": before,
                "after": dict(before), "steps": 0, "lora": None,
                "joint": None, "geometry": {}, "rank_slice": rank_slice,
                "skipped_all_zero": True, "n_train_rows": len(train_rows),
                "n_holdout_rows": len(eval_rows),
                "eval_attn_cap": eval_attn_cap,
                "dequant_path": dequant_path,
                "stop_reason": "r0-skip", "tripwire": None, "init": None,
                "warm_start_gate": None, "eval_profiles": [],
                "step0_diff": None,
                "wall_s": round(time.time() - t0, 1), **_telemetry()}

    # W4-T02: the trainable UNION (every requires_grad parameter — the
    # clip's, the schedule's and the telemetry's domain). The channel
    # grouping for the optimizer is _joint_param_groups below; a
    # requires_grad parameter no channel claims errors LOUDLY there
    # (the census and the optimizer must agree).
    trainable = [(n, p) for n, p in shell.named_parameters()
                 if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            f"layer {L}: no trainable parameters (ranks "
            f"{rank_slice!r}, --train {sorted(train_groups)}) — "
            f"refusing to train nothing")

    # ---- F9 anchor + initialization (B=0 | warm start | stage-1 snapshot)
    # W4-T02: the anchor/banking snapshot is the JOINT true clone
    # (PROPOSAL §2.6 — the R15 lesson applies to LUT/norm tensors too):
    # a lora-only restore would leave a diverged run's LUT/norm drift
    # in place and break the never-finish-worse guarantee.
    zero_state = _joint_snapshot(shell)
    init_note = "B=0"
    warm_gate = None          # R2: the recorded warm-start quality verdict
    if args.warm_start:
        # R2: --warm-start-skip resolution — block-type pseudo tokens
        # ('full_attention' / 'linear_attention') apply ONLY to this
        # job's own block type; every other token is a module-key suffix
        # pattern.
        skip = []
        for pat in (args.warm_start_skip or "").split(","):
            pat = pat.strip()
            if not pat:
                continue
            if pat in ("full_attention", "linear_attention"):
                if pat == block_type:
                    skip.append("*")
                continue
            skip.append(pat)
        spectrum = _spectrum()  # noqa: E402  (absent in this repo: loud error)
        spectrum._apply_warm_starts(
            shell, args.warm_start, ctx=f"L{L}",
            skip=tuple(skip),
            gauge=getattr(args, "warm_start_gauge", "balanced"))
        init_note = "warm-start"
    elif init_dir is not None:
        if _apply_init_snapshot(shell, L, acfg, init_dir, f"L{L}"):
            init_note = "stage1-init"
    _watermark()        # T7: around-eval (before)
    _prof = _maybe_profile("init")
    with phase_timer.phase("eval"):
        init_eval = _evaluate(student, pos16, reader_rows, target_fn,
                              eval_rows, device, effective_eval_rows,
                              args.mse_weight, shell, output_fn=output_fn,
                              input_dtype=torch.float16,
                              cos_weight=args.cos_weight, profile=_prof,
                              eval_tensors=eval_tensors)
    _close_profile(_prof)
    _watermark()        # T7: around-eval (after)
    # ---- R2 (recovery campaign): warm-start quality GATE -----------------
    # The init-vs-before ratio the halted Stage-1 run lacked: computed for
    # EVERY warm-started job, printed when it is not an improvement, and
    # recorded in metrics.json / pilot summaries. Verdicts: 'improved'
    # (ratio <= 1), 'regression' (1 < ratio <= 2 — the F9 anchor silently
    # absorbs this today, now it is VISIBLE), 'catastrophic' (> 2 — the
    # §7 red flag). The anchor decision below is UNCHANGED — the gate
    # observes, it never steers.
    if args.warm_start and before["rel_mse"] > 0.0:
        ws_ratio = init_eval["rel_mse"] / before["rel_mse"]
        ws_verdict = ("improved" if ws_ratio <= 1.0 else
                      "regression" if ws_ratio <= _WARM_START_RED_FLAG else
                      "catastrophic")
        warm_gate = {"ratio": round(ws_ratio, 4), "verdict": ws_verdict,
                     "red_flag": ws_ratio > _WARM_START_RED_FLAG}
        if ws_verdict != "improved":
            print(f"        [warm-start] GATE: init rel_mse="
                  f"{init_eval['rel_mse']:.4g} vs before {before['rel_mse']:.4g}"
                  f" -> ratio {ws_ratio:.2f}x [{ws_verdict.upper()}]"
                  f"{' — RED FLAG (warm-start): this layer\'s warm-start '
                        f'made the init WORSE; the F9 anchor has restored '
                        f'B=0 for this job — run verify-warm-start '
                        f'--per-module to attribute the module, then '
                        f'--warm-start-skip to exclude it' if warm_gate['red_flag'] else ''}",
                  flush=True)
    if init_eval["rel_mse"] < before["rel_mse"] or (
            init_eval["rel_mse"] == before["rel_mse"]
            and init_eval["tok_cos"] > before["tok_cos"]):
        best = {"rel_mse": init_eval["rel_mse"],
                "tok_cos": init_eval["tok_cos"],
                "state": _joint_snapshot(shell), "step": 0}
    else:
        best = {"rel_mse": before["rel_mse"], "tok_cos": before["tok_cos"],
                "state": zero_state, "step": 0}
        _load_lora_state_dict(shell, zero_state,
                              ctx=f"L{L} B=0 anchor restore")

    # ---- optimizer + schedule (F5/F6; W4-T02: the two-channel joint
    # ---- plan, PROPOSAL §2.5 — lora at args.lr via the
    # ---- --opt-lora kind, codes (LUT masters at --lr-lut, norm gains
    # ---- in their own group at --lr-norm) always AdamW) -------------
    param_groups = _joint_param_groups(shell)
    plans = _make_joint_optimizer(
        param_groups, args.lr,
        float(getattr(args, "lr_lut", 3e-4) or 3e-4),
        float(getattr(args, "lr_norm", 1e-4) or 1e-4),
        opt_lora=args.optimizer,
        adam_eps=float(getattr(args, "adam_eps", 1e-8) or 1e-8))

    # ---- W5.T2 (PROPOSAL §2.9 audit): opt-in per-factor update-RMS ------
    muon_diag = bool(getattr(args, "muon_diagnostics", False))
    diag_every = max(1, int(args.log_every))
    muon_snap = None      # {name: param snapshot} at the window's start

    def _lr_mult(step):
        if args.warmup > 0 and step < args.warmup:
            # Warmup: start at 1e-6/1e-3 = 0.001, ramp to 1.0
            # Using sqrt to ramp from min_ratio to 1.0
            min_ratio = 0.001  # 1e-6 / 1e-3
            progress = (step + 1) / args.warmup
            return min_ratio + (1.0 - min_ratio) * math.sqrt(progress)
        u = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        u = min(1.0, max(0.0, u))
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * u))   # 1.0 -> 0.1

    rng = np.random.default_rng(args.seed + 1000 + L)
    step = 0
    stall = 0
    stop_reason = None
    trip = {"fired": 0, "bad_evals": 0, "lr_scale": 1.0}
    ema_t = None           # R10: float64 DEVICE scalar (None = fresh)
    first_losses = []      # R10: 0-d device tensors, stacked at eval time
    arm_steps = min(_TRIP_ARM_STEPS, max(1, args.max_steps // 2))
    win = collections.deque(maxlen=max(1, int(args.log_every)))

    # ---- R5 (recovery campaign): step-0 weight diff + first-step hold ----
    step0_diff = None
    step0_pre = None
    if int(getattr(args, "dump_step0_diff", 0) or 0):
        step0_pre = {n: p.detach().clone().cpu()
                     for n, p in trainable}
        print(f"        [step0-diff] snapshot taken: {len(step0_pre)} adapter "
              f"factor(s) before the first optimizer step", flush=True)
    # ---- W4-T04 (PROPOSAL §2.6): the step-0 delta-L2 census, ALWAYS ON
    # ---- (one line after the first applied optimizer step; the detailed
    # ---- per-factor record above stays behind --dump-step0-diff) -------
    census_pre = {g: [p.detach().clone() for _, p in param_groups[g]]
                  for g in _TRAIN_GROUPS if param_groups.get(g)}
    hold_steps = max(0, int(getattr(args, "hold_steps", 0) or 0))
    if hold_steps:
        print(f"        [hold-steps] the first {hold_steps} step(s) consume "
              f"batches WITHOUT optimizer updates (warm-start "
              f"preservation)", flush=True)
    print(f"  [L{L:02d}] {block_type} train-target=qlora input-mode="
          f"{args.input_mode} train_rows={len(train_rows)} "
          f"holdout_rows={len(eval_rows)} lora_modules="
          f"{census['lora']['modules']} init={init_note} optimizer="
          f"{args.optimizer} lr={args.lr:.2e} eval_batch_rows="
          f"{effective_eval_rows}", flush=True)
    print(f"        before: tok_cos={before['tok_cos']:.6f} "
          f"flat_cos={before['flat_cos']:.6f} "
          f"rel_mse={before['rel_mse']:.4g}", flush=True)
    print(f"        init : tok_cos={init_eval['tok_cos']:.6f} "
          f"flat_cos={init_eval['flat_cos']:.6f} "
          f"rel_mse={init_eval['rel_mse']:.4g}", flush=True)

    while stop_reason is None:
        order = rng.permutation(train_rows)
        nb = (len(order) + args.batch_rows - 1) // args.batch_rows
        for b in range(nb):
            if step >= args.max_steps:
                stop_reason = "max-steps"
                break
            # rows within a batch are SORTED (contiguous-run reads);
            # the batch COMPOSITION carries the SGD randomness
            idx = sorted(int(i) for i in
                         order[b * args.batch_rows:(b + 1) * args.batch_rows])
            with phase_timer.phase("reader"):
                x16 = reader_rows(idx, device, torch.float16)
            pos = _pos_batch(pos16, x16.shape[0])
            with phase_timer.phase("target"):
                tgt = target_fn(idx, device)
            for _opt, _ in plans:
                _opt.zero_grad(set_to_none=True)
            with phase_timer.phase("forward"):
                y = _forward_layer(student, x16, pos, shell)
            # R10 (perf): sync=False — the metrics stay as 0-d device
            # tensors; the train loop materializes ONE batch of floats at
            # window close.
            loss, met = distill_loss(y.float(), tgt,
                                     mse_weight=args.mse_weight,
                                     cos_weight=args.cos_weight,
                                     sync=False)
            with phase_timer.phase("backward"):
                loss.backward()
            # R10: gnorm stays a 0-d device tensor (clip still scales the
            # grads on-device — the CLIP is unchanged; only the per-step
            # float() sync is gone). Materialized at window close.
            # W4-T02 (§2.5 rule 3): GLOBAL clip — ONE call over the UNION
            # of every group's params (lora + luts + norms).
            if args.clip > 0:
                gnorm_t = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable], args.clip)
            else:
                gnorm_t = (sum(p.grad.detach().float().pow(2).sum()
                               for _, p in trainable
                               if p.grad is not None)) ** 0.5
            if muon_diag and muon_snap is None:
                # W5.T2 window start: snapshot BEFORE this window's first
                # opt.step(), OUTSIDE the timed optimizer phase
                muon_snap = {n: p.detach().clone()
                             for n, p in trainable}
            with phase_timer.phase("optimizer"):
                # W4-T02 (§2.5): every group carries its OWN base LR
                # (lora / lut / norm) times the shared schedule and the
                # tripwire scale; lr_now (the step line's field) is the
                # LORA channel's current lr.
                lr_mult = _lr_mult(step) * trip["lr_scale"]
                for _opt, _bases in plans:
                    for _g, _base in zip(_opt.param_groups, _bases):
                        _g["lr"] = _base * lr_mult
                lr_now = args.lr * lr_mult
                if step >= hold_steps:
                    for _opt, _ in plans:
                        _opt.step()
                    if census_pre is not None:
                        # W4-T04: the delta-L2 census after the FIRST
                        # applied optimizer step (hold_steps-aware)
                        print(_step0_delta_census_line(
                            census_pre, param_groups, L), flush=True)
                        census_pre = None
            step += 1
            if step == 1 and step0_pre is not None:
                # R5: the first optimizer step just landed (or was held) —
                # diff every factor against the pre-step snapshot.
                factors = {}
                for n, p in trainable:
                    d = (p.detach().cpu() - step0_pre[n]).float()
                    pre = step0_pre[n].float()
                    factors[n] = {
                        "pre_l2": round(float(pre.pow(2).sum().sqrt()), 6),
                        "delta_l2": round(float(d.pow(2).sum().sqrt()), 8),
                        "delta_rms": round(
                            float(d.pow(2).mean().sqrt())
                            if d.numel() else 0.0, 8),
                        "delta_max_abs": round(
                            float(d.abs().max()) if d.numel() else 0.0, 8),
                    }
                step0_diff = {
                    "layer": L,
                    "optimizer": args.optimizer,
                    "lr": float(args.lr),
                    "lr_mult_step0": round(float(_lr_mult(0)), 6),
                    "hold_steps": hold_steps,
                    "applied_update": hold_steps == 0,
                    "n_factors": len(factors),
                    "factors": factors,
                }
                step0_pre = None
                top = sorted(factors.items(),
                             key=lambda kv: kv[1]["delta_l2"],
                             reverse=True)[:3]
                print("        [step0-diff] after step 1: " + ", ".join(
                    f"{n} dL2={v['delta_l2']:.3e} (pre L2={v['pre_l2']:.3e})"
                    for n, v in top), flush=True)
            # ---- F15: component telemetry every step (R10: on-device, ----
            # ---- ZERO per-step syncs; ONE tolist() at window close) ------
            lv_t = loss.detach()
            # ema stays a float64 DEVICE scalar — the update arithmetic
            # (0.98*ema + 0.02*lv in binary64) is bit-identical to the old
            # Python-float version, just without the per-step .item().
            ema_t = lv_t.double() if ema_t is None \
                else 0.98 * ema_t + 0.02 * lv_t.double()
            if len(first_losses) < arm_steps:
                first_losses.append(lv_t)
            win.append((lv_t, met["rel_mse"], met["tok_cos"], gnorm_t))
            if step % max(1, int(args.log_every)) == 0 or step == 1:
                # ONE materialization for the whole window: column means
                # + the current step's gnorm + the current EMA (the old
                # per-step float()s are gone; the printed VALUES are the
                # same numbers in the same order).
                stacked = torch.stack([torch.stack(tuple(w)) for w in win])
                means = stacked.mean(dim=0)
                summary = torch.stack((means[0], means[1], means[2],
                                       stacked[-1, 3], ema_t))
                lv_m, wm_rmse, wm_cos, gnorm, ema = summary.tolist()
                wm = [lv_m, wm_rmse, wm_cos]
                # W5.T1 (T7): window phase means (tick_window closes the
                # window — totals preserved), the vram pair, and the
                # cumulative reader_amp, in the exact PROPOSAL §3 T7
                # field order; the F15 fields stay, appended AFTER
                # reader_amp. CPU runs print honest zeros (vram 0.0/0.0).
                pw = phase_timer.tick_window()
                vram_cur, vram_tot = _vram_gib(device)
                print(f"        step {step:5d} loss={wm[0]:.4f} "
                      f"reader={pw['reader'] * 1000:.0f}ms "
                      f"target={pw['target'] * 1000:.0f}ms "
                      f"fwd={pw['forward'] * 1000:.0f}ms "
                      f"bwd={pw['backward'] * 1000:.0f}ms "
                      f"opt={pw['optimizer'] * 1000:.0f}ms "
                      f"vram={vram_cur:.1f}/{vram_tot:.1f}GiB "
                      f"reader_amp={_reader_amp():.2f} "
                      f"rel_mse={wm[1]:.5f} cos={1 - wm[2]:.6f} "
                      f"gnorm={gnorm:.3f} ema={ema:.5f} lr={lr_now:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if muon_diag and muon_snap is not None \
                    and step % diag_every == 0:
                # W5.T2 window close: one update-RMS line per adapter
                # factor (RMS of new_param - old_param over the window).
                # This float() is a device sync BY DESIGN — the reason the
                # flag is opt-in while the phase timer stays no-sync.
                for n, p in trainable:
                    delta = p.detach() - muon_snap[n]
                    update_rms = float(delta.pow(2).mean().sqrt()) \
                        if delta.numel() else 0.0
                    print(f"  [L{L:02d}] muon: {n} "
                          f"update_rms={update_rms:.3e} lr={lr_now:.2e}",
                          flush=True)
                muon_snap = None      # the next window re-snapshots at start
            if step % args.eval_every == 0 or step == args.max_steps:
                _watermark()        # T7: around-eval (before)
                _prof = _maybe_profile(f"step-{step}")
                with phase_timer.phase("eval"):
                    after_eval = _evaluate(
                        student, pos16, reader_rows, target_fn, eval_rows,
                        device, effective_eval_rows, args.mse_weight,
                        shell, output_fn=output_fn,
                        input_dtype=torch.float16,
                        cos_weight=args.cos_weight, profile=_prof,
                        eval_tensors=eval_tensors)
                _close_profile(_prof)
                _watermark()        # T7: around-eval (after)
                # W5.T1 (T7): the eval phase logs its own line
                vram_cur, _ = _vram_gib(device)
                print(f"        eval@{step:5d} "
                      f"tok_cos={after_eval['tok_cos']:.6f} "
                      f"flat_cos={after_eval['flat_cos']:.6f} "
                      f"rel_mse={after_eval['rel_mse']:.4g} "
                      f"eval={phase_timer.last('eval') * 1000:.0f}ms "
                      f"vram={vram_cur:.1f}GiB "
                      f"reader_amp={_reader_amp():.2f}", flush=True)
                # ---- W4-T04 (PROPOSAL §2.6 + §8 risk 2): the per-module
                # ---- code-usage histogram and the VRAM pair, one line
                # ---- each per eval -------------------------------------
                print(_code_usage_line(shell, L, step), flush=True)
                print(_vram_pair_line(step, device), flush=True)
                # ---- F9: bank ANY rel_mse improvement ------------------
                # W4-T02: the banked best is the JOINT true clone (the
                # restore below and the final guard restore the whole
                # trainable set, not just the lora tensors).
                if after_eval["rel_mse"] < best["rel_mse"] or (
                        after_eval["rel_mse"] == best["rel_mse"]
                        and after_eval["tok_cos"] > best["tok_cos"]):
                    best = {"rel_mse": after_eval["rel_mse"],
                            "tok_cos": after_eval["tok_cos"],
                            "state": _joint_snapshot(shell),
                            "step": step}
                    stall = 0
                else:
                    stall += 1
                # ---- F10: divergence tripwire (PROPOSAL §2.6: 2 ------
                # ---- consecutive bad evals, 2 firings, 3rd stops) ------
                if len(first_losses) >= arm_steps:
                    # R10: the arm mean and the comparison run ON DEVICE
                    # (one .item() at this eval boundary, not per step);
                    # the EMA is float64.
                    baseline_t = torch.stack(first_losses).mean()
                    bad_t = (~torch.isfinite(ema_t)) | \
                        (ema_t > _TRIP_FACTOR * baseline_t)
                    if bool(bad_t.item()):
                        trip["bad_evals"] += 1
                    else:
                        trip["bad_evals"] = 0
                    if trip["bad_evals"] >= _TRIP_EVALS:
                        if trip["fired"] >= _TRIP_MAX_FIRINGS:
                            stop_reason = "tripwire"
                            print(f"        [tripwire] train-EMA diverged "
                                  f"again after {trip['fired']} firing(s) "
                                  f"— STOPPING this layer (PROPOSAL §2.6: "
                                  f"2 firings max; the best snapshot is "
                                  f"restored by the final guard)", flush=True)
                        else:
                            trip["fired"] += 1
                            _load_lora_state_dict(shell, best["state"],
                                                  ctx=f"L{L} tripwire "
                                                      f"restore #{trip['fired']}")
                            trip["lr_scale"] *= 0.5
                            ema_t = None
                            trip["bad_evals"] = 0
                            print(f"        [tripwire] train-EMA diverged "
                                  f"(> {_TRIP_FACTOR}x first-{arm_steps} "
                                  f"mean for {_TRIP_EVALS} evals) — restored "
                                  f"best snapshot (step {best['step']}), "
                                  f"halved lr scale to "
                                  f"{trip['lr_scale']:.3f}, resuming",
                                  flush=True)
                # ---- F6: working stops --------------------------------
                target_rel = max(float(args.target_rel_mse),
                                 1.15 * float(predicted_rel_mse))
                if after_eval["rel_mse"] <= target_rel \
                        and after_eval["tok_cos"] >= args.target_cos:
                    stop_reason = "target"
                elif stall >= args.patience:
                    stop_reason = "patience"
        # epoch exhausted: the while loop re-enters only if no stop fired

    # restore the best snapshot (never finish worse than the anchor)
    _load_lora_state_dict(shell, best["state"], ctx=f"L{L} best snapshot")
    _watermark()        # T7: around-eval (before)
    _prof = _maybe_profile("final")
    with phase_timer.phase("eval"):
        after = _evaluate(student, pos16, reader_rows, target_fn, eval_rows,
                          device, effective_eval_rows, args.mse_weight,
                          shell, output_fn=output_fn,
                          input_dtype=torch.float16,
                          cos_weight=args.cos_weight, profile=_prof,
                          eval_tensors=eval_tensors)
    _close_profile(_prof)
    _watermark()        # T7: around-eval (after)
    # R10: the resident eval pair's job is done — release the ~5 GiB back
    # before the caller materializes the next layer.
    eval_tensors = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"layer": L, "block_type": block_type, "before": before,
            "after": after, "steps": step, "lora": _lora_snapshot(shell),
            # W4-T06: the banked-best JOINT state (luts + norms + lora at
            # the best holdout eval) — the export's source of truth; the
            # final guard above restored it into the shell, so the values
            # are best["state"] exactly.
            "joint": best["state"],
            "geometry": acfg.tensors, "rank_slice": rank_slice,
            "skipped_all_zero": False, "n_train_rows": len(train_rows),
            "n_holdout_rows": len(eval_rows),
            "eval_attn_cap": eval_attn_cap,
            "stop_reason": stop_reason,
            "dequant_path": dequant_path,
            "tripwire": {"fired": trip["fired"],
                         "lr_scale": trip["lr_scale"]},
            "init": init_eval, "best_step": best["step"],
            # R2/R4/R5 (recovery campaign): the warm-start verdict, the
            # per-eval timing breakdowns, and the step-0 weight diff
            "warm_start_gate": warm_gate,
            "eval_profiles": list(eval_profiles),
            "step0_diff": step0_diff,
            "wall_s": round(time.time() - t0, 1), **_telemetry()}


# ---------------------------------------------------------------------------
# Trajectory propagation (deploy numerics) + the L2 replay
# ---------------------------------------------------------------------------

def propagate_layer(L, layer, model, xcache, device, use_kernel: bool):
    """x_{L+1} = layer(x_L) over all rows, in fp16 deploy numerics.

    The layer must be frozen + fp16-snapped. With use_kernel (compiled
    flute_extension + CUDA), the modules temporarily switch to the kernel
    path — the exact serving numerics for the trajectory."""
    layer.to(device)
    layer.eval()
    mods = []
    if use_kernel:
        for _, m in pmod.iter_palettized_linears(layer):
            m.reference = False
            mods.append(m)
    try:
        out_mm = xcache.open(L + 1)
        pos = _position_embeddings(model, xcache.seq, torch.float16, device)
        CH = 8
        with torch.no_grad():
            for i0 in range(0, xcache.rows, CH):
                i1 = min(i0 + CH, xcache.rows)
                x = xcache.read(L, i0, i1, device, torch.float16)
                p = _pos_batch(pos, x.shape[0])
                y = _forward_layer(layer, x, p, model)
                out_mm[i0:i1] = y.detach().to(torch.float16).cpu().numpy()
        out_mm.flush()
    finally:
        for m in mods:
            m.reference = True
        layer.to("cpu")
    xcache.maybe_drop(L)


def _replay_to_qlora(xcache, L, cfg, teacher_source, artifacts_dir, metadata,
                     rank_map, snapshots, device, args):
    """Rebuild x_layer{L} (student trajectory, the L2 escalation input) by
    replaying the layer-scoped student through layers 0..L-1 from the
    nearest available cache checkpoint — each replayed layer materializes
    alone (with its trained snapshot's adapters applied when one exists),
    propagates, and is released."""
    start = 0
    for k in range(L - 1, -1, -1):
        if xcache.exists(k):
            start = k
            break
    if not xcache.exists(L):
        print(f"  [replay] x_layer{L} missing — replaying from layer {start}",
              flush=True)
        for k in range(start, L):
            layer, _ = materialize_student_layer(
                k, cfg, teacher_source, artifacts_dir, metadata,
                device=device, dtype=torch.float16)
            shell = StudentLayerShell(layer, k, cfg, device=device)
            # W4.T3: replayed STUDENT layers get the same flute_sm86 pin as
            # the trained ones — the trajectory forward is the same student
            # hot path (un-pinned layers would silently mix backends).
            _pin_student_attn(layer, k, _layer_type(cfg, k), cfg, device,
                              ctx=f"replay L{k}")
            snap_path = snapshots.get(k)
            if snap_path is not None:
                acfg, _ = _attach_layer_adapters(shell, k, rank_map, args)
                if acfg is None:
                    raise RuntimeError(
                        f"replay layer {k}: snapshot exists but the current "
                        f"rank map puts every module at r=0 — geometry "
                        f"mismatch (re-run with the recorded rank map)")
                snap = torch.load(snap_path, map_location="cpu")
                for mod, g in snap.get("geometry", {}).items():
                    cur = acfg.tensors.get(mod)
                    if cur is None or cur.get("r") != g.get("r"):
                        raise RuntimeError(
                            f"replay layer {k}: snapshot geometry for "
                            f"{mod} (r={g.get('r')}) disagrees with the "
                            f"attached adapter (r="
                            f"{None if cur is None else cur.get('r')})")
                _load_lora_state_dict(shell, snap["lora"],
                                      ctx=f"replay L{k} snapshot")
            propagate_layer(k, layer, shell, xcache, device, False)
            del layer, shell
            gc.collect()


# ---------------------------------------------------------------------------
# The W4-T05 loop pieces (PROPOSAL §2.1 P1): the worst-first order from
# the spectrum, and the final L=norm job (the model's final RMSNorm)
# ---------------------------------------------------------------------------

def _worst_first_order(spectrum, wanted, num_layers):
    """`--order worst-first` with a spectrum (PROPOSAL §2.1 P1): the
    sweep order = the given layer selection sorted by the layer's summed
    module err_energy DESCENDING (the largest error reservoirs first).
    A layer absent from the spectrum keeps its given position AFTER the
    ranked ones (with a loud note — the spectrum should cover the
    sweep); without a spectrum the given order is the order (the
    operator's explicit err_energy-descending --layers list)."""
    if not spectrum:
        return list(wanted)
    energy = {}
    for key, m in spectrum.items():
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers" \
                and parts[2].lstrip("-").isdigit():
            L = int(parts[2])
            energy[L] = energy.get(L, 0.0) + float(m.get("err_energy", 0.0))
    ranked = sorted((L for L in wanted if L in energy),
                    key=lambda L: (-energy[L], L))
    rest = [L for L in wanted if L not in energy]
    if rest:
        print(f"  [trainer] worst-first: layer(s) {rest} carry no "
              f"spectrum record — appended after the ranked ones (the "
              f"spectrum should cover the sweep)", flush=True)
    return ranked + rest


def _final_norm_gain(teacher_source, cfg):
    """The student's FINAL RMSNorm gain, loaded from the dense checkpoint
    (the student's dense parts ARE the teacher checkpoint) as an fp32
    MASTER (the dtype ladder — the W4-T04 lesson: an fp16 gain under
    AdamW keeps fp16 moments and explodes)."""
    M = _modeling()
    norm = M.Qwen3_5RMSNorm(int(cfg.hidden_size), eps=cfg.rms_norm_eps)
    w = teacher_source.load_dense_tensor("model.norm.weight")
    if tuple(w.shape) != (int(cfg.hidden_size),):
        raise RuntimeError(
            f"model.norm.weight: shape {tuple(w.shape)} != hidden_size "
            f"({cfg.hidden_size}) — the checkpoint and the config "
            f"disagree")
    norm.weight = nn.Parameter(w.detach().float().clone())
    return norm


def _run_final_norm_job(args, store, device, out_dir, provenance, cfg,
                        teacher_source, metadata, rank_map):
    """The L=norm job (PROPOSAL §2.1 P1's final job, W4-T05): train the
    model's FINAL RMSNorm gain against the boundary store's final_hidden
    — the same loss (distill_loss on the full output), the same control
    plane (holdout split, banking on rel_mse with tok_cos tiebreak, the
    F9 anchor, patience, the §2.6 tripwire, final restore-best).

    The pre-norm input x is computed ONCE by propagating the student's
    LAST layer over every row of h_{num_layers-1} (this run's trained
    snapshot applied when one exists); the layer is FREED before the
    norm training starts (the one-residency rule, §2.2). x and the
    targets live as pinned host caches (the _TargetCache convention).

    Export: the edited gain lands in the shared norm_gain_edits.json
    under out_dir (the 'model.norm.weight' entry — the full edited
    weight, the layer-edit loader's copy semantics).

    Returns the metrics dict (recorded as final_norm/metrics.json and
    provenance['final_norm'])."""
    t0 = time.time()
    rows_total = store.rows
    last = int(store.num_layers) - 1
    holdout_rows, _train_rows = _holdout_split(args, rows_total)

    # ---- ONE pass: the student's last layer -> x (pinned) ------------
    x_cache = _TargetCache(rows_total, store.seq, store.hidden,
                           pin=(torch.cuda.is_available()
                                and torch.device(device).type == "cuda"),
                           device=device)
    layer, _n = materialize_student_layer(
        last, cfg, teacher_source, args.artifacts_dir, metadata,
        device=device, dtype=torch.float16)
    shell = StudentLayerShell(layer, last, cfg, device=device)
    _pin_student_attn(layer, last, _layer_type(cfg, last), cfg, device,
                      ctx=f"norm-prop L{last}")
    snap_path = None
    rec = provenance.get("layers", {}).get(str(last))
    if rec and rec.get("done") and rec.get("has_adapter"):
        snap_path = os.path.join(out_dir, rec["snapshot"])
    if snap_path is not None and os.path.isfile(snap_path):
        acfg, _slice = _attach_layer_adapters(shell, last, rank_map, args)
        if acfg is None:
            raise RuntimeError(
                f"final-norm propagation: layer {last} has a trained "
                f"snapshot but the current rank map puts every module "
                f"at r=0 — geometry mismatch (re-run with the recorded "
                f"rank map)")
        snap = torch.load(snap_path, map_location="cpu")
        for mod, g in snap.get("geometry", {}).items():
            cur = acfg.tensors.get(mod)
            if cur is None or cur.get("r") != g.get("r"):
                raise RuntimeError(
                    f"final-norm propagation: snapshot geometry for "
                    f"{mod} (r={g.get('r')}) disagrees with the attached "
                    f"adapter (r="
                    f"{None if cur is None else cur.get('r')})")
        _load_lora_state_dict(shell, snap["lora"],
                              ctx=f"norm-prop L{last} snapshot")
    else:
        print(f"  [norm] layer {last} carries no trained adapter — the "
              f"propagation uses the plain materialized student",
              flush=True)
    pos16 = _position_embeddings(shell, store.seq, torch.float16, device)
    CH = 8
    with torch.no_grad():
        for i0 in range(0, rows_total, CH):
            i1 = min(i0 + CH, rows_total)
            x16 = store.h(last, i0, i1, device, torch.float16)
            y = _forward_layer(layer, x16,
                               _pos_batch(pos16, x16.shape[0]), shell)
            x_cache.t[i0:i1].copy_(y.detach().to(torch.float16))
    # one-residency: the layer's job is done — free it before the norm
    # training starts
    del layer, shell, pos16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- the target cache: final_hidden (pinned, same convention) ----
    t_cache = _TargetCache(rows_total, store.seq, store.hidden,
                           pin=x_cache.t.is_pinned(), device=device)
    fh = np.load(os.path.join(store.dir, "final_hidden.npy"),
                 mmap_mode="r")
    for i0 in range(0, rows_total, CH):
        i1 = min(i0 + CH, rows_total)
        t_cache.t[i0:i1].copy_(
            torch.from_numpy(np.ascontiguousarray(fh[i0:i1])))

    def reader_rows(rows, dev, dtype=torch.float32):
        return x_cache.rows(rows, dev).to(dtype)

    def target_fn(rows, dev):
        return t_cache.rows(rows, dev)

    # ---- the norm student: ONE fp32-master gain ----------------------
    norm = _final_norm_gain(teacher_source, cfg)
    trainable = [(n, p) for n, p in norm.named_parameters()
                 if p.requires_grad]
    param_groups = {"norms": trainable}
    plans = _make_joint_optimizer(
        param_groups, args.lr, float(getattr(args, "lr_lut", 3e-4)),
        float(getattr(args, "lr_norm", 1e-4)), opt_lora="adamw",
        adam_eps=float(getattr(args, "adam_eps", 1e-8) or 1e-8))

    def output_fn(x, pos):
        return norm(x)

    def _lr_mult(step):
        if args.warmup > 0 and step < args.warmup:
            min_ratio = 0.001
            progress = (step + 1) / args.warmup
            return min_ratio + (1.0 - min_ratio) * math.sqrt(progress)
        u = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        u = min(1.0, max(0.0, u))
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * u))

    print(f"  [norm] final-norm job: gain=({int(cfg.hidden_size)},) "
          f"rows={rows_total} holdout_rows={len(holdout_rows)} "
          f"optimizer=adamw lr={float(args.lr_norm):.2e}", flush=True)
    before = _evaluate(norm, None, reader_rows, target_fn,
                       holdout_rows, device, args.eval_batch_rows,
                       args.mse_weight, norm, output_fn=output_fn,
                       input_dtype=torch.float16,
                       cos_weight=args.cos_weight)
    print(f"        before: tok_cos={before['tok_cos']:.6f} "
          f"flat_cos={before['flat_cos']:.6f} "
          f"rel_mse={before['rel_mse']:.4g}", flush=True)
    best = {"rel_mse": before["rel_mse"], "tok_cos": before["tok_cos"],
            "state": _joint_snapshot(norm), "step": 0}
    rng = np.random.default_rng(args.seed + 1000 + last + 1)
    order_rows = [r for r in range(rows_total)
                  if r not in set(holdout_rows)]
    step = 0
    stall = 0
    stop_reason = None
    trip = {"fired": 0, "bad_evals": 0, "lr_scale": 1.0}
    ema_t = None
    first_losses = []
    arm_steps = min(_TRIP_ARM_STEPS, max(1, args.max_steps // 2))
    census_pre = [p.detach().clone() for _, p in trainable]
    while stop_reason is None:
        order = rng.permutation(order_rows)
        nb = (len(order) + args.batch_rows - 1) // args.batch_rows
        for b in range(nb):
            if step >= args.max_steps:
                stop_reason = "max-steps"
                break
            idx = sorted(int(i) for i in
                         order[b * args.batch_rows:
                               (b + 1) * args.batch_rows])
            x16 = reader_rows(idx, device, torch.float16)
            tgt = target_fn(idx, device)
            for _opt, _ in plans:
                _opt.zero_grad(set_to_none=True)
            y = norm(x16)
            loss, _met = distill_loss(y.float(), tgt,
                                      mse_weight=args.mse_weight,
                                      cos_weight=args.cos_weight,
                                      sync=False)
            loss.backward()
            if args.clip > 0:
                gnorm_t = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable], args.clip)
            else:
                gnorm_t = (sum(p.grad.detach().float().pow(2).sum()
                               for _, p in trainable
                               if p.grad is not None)) ** 0.5
            lr_mult = _lr_mult(step) * trip["lr_scale"]
            for _opt, _bases in plans:
                for _g, _base in zip(_opt.param_groups, _bases):
                    _g["lr"] = _base * lr_mult
            for _opt, _ in plans:
                _opt.step()
            if census_pre is not None:
                sq = sum(((p.detach() - p0).float().pow(2).sum())
                         for (_, p), p0 in zip(trainable, census_pre))
                print(f"  [norm] step-0 delta-L2: "
                      f"norms={float(sq) ** 0.5:.3e}", flush=True)
                census_pre = None
            step += 1
            ema_t = loss.detach().double() if ema_t is None \
                else 0.98 * ema_t + 0.02 * loss.detach().double()
            if len(first_losses) < arm_steps:
                first_losses.append(loss.detach())
            if step % args.eval_every == 0 or step == args.max_steps:
                after_eval = _evaluate(
                    norm, None, reader_rows, target_fn, holdout_rows,
                    device, args.eval_batch_rows, args.mse_weight, norm,
                    output_fn=output_fn, input_dtype=torch.float16,
                    cos_weight=args.cos_weight)
                print(f"        eval@{step:5d} "
                      f"tok_cos={after_eval['tok_cos']:.6f} "
                      f"flat_cos={after_eval['flat_cos']:.6f} "
                      f"rel_mse={after_eval['rel_mse']:.4g}", flush=True)
                print(_vram_pair_line(step, device), flush=True)
                if after_eval["rel_mse"] < best["rel_mse"] or (
                        after_eval["rel_mse"] == best["rel_mse"]
                        and after_eval["tok_cos"] > best["tok_cos"]):
                    best = {"rel_mse": after_eval["rel_mse"],
                            "tok_cos": after_eval["tok_cos"],
                            "state": _joint_snapshot(norm),
                            "step": step}
                    stall = 0
                else:
                    stall += 1
                if len(first_losses) >= arm_steps:
                    baseline_t = torch.stack(first_losses).mean()
                    bad_t = (~torch.isfinite(ema_t)) | \
                        (ema_t > _TRIP_FACTOR * baseline_t)
                    if bool(bad_t.item()):
                        trip["bad_evals"] += 1
                    else:
                        trip["bad_evals"] = 0
                    if trip["bad_evals"] >= _TRIP_EVALS:
                        if trip["fired"] >= _TRIP_MAX_FIRINGS:
                            stop_reason = "tripwire"
                        else:
                            trip["fired"] += 1
                            _load_lora_state_dict(
                                norm, best["state"],
                                ctx=f"norm tripwire restore "
                                    f"#{trip['fired']}")
                            trip["lr_scale"] *= 0.5
                            ema_t = None
                            trip["bad_evals"] = 0
                if stall >= args.patience:
                    stop_reason = "patience"
    _load_lora_state_dict(norm, best["state"], ctx="norm best snapshot")
    after = _evaluate(norm, None, reader_rows, target_fn, holdout_rows,
                      device, args.eval_batch_rows, args.mse_weight, norm,
                      output_fn=output_fn, input_dtype=torch.float16,
                      cos_weight=args.cos_weight)
    # ---- export: the shared norm_gain_edits.json ----------------------
    edits_dir = os.path.join(out_dir, "norm_edits")
    os.makedirs(edits_dir, exist_ok=True)
    arr = norm.weight.detach().cpu().float().numpy()
    npy = os.path.join(edits_dir, "model.norm.weight.npy")
    with open(npy, "wb") as f:
        np.save(f, arr)
    entry = {"file": os.path.relpath(npy, out_dir),
             "sha256": _file_sha256(npy).split(":")[1]}
    norm_edits_path = os.path.join(out_dir, "norm_gain_edits.json")
    doc = {"edits": {"model.norm.weight": entry}}
    if os.path.exists(norm_edits_path):
        with open(norm_edits_path) as f:
            doc = json.load(f)
        doc.setdefault("edits", {})["model.norm.weight"] = entry
    _atomic_json_dump(doc, norm_edits_path)
    print(f"  [norm] after : tok_cos={after['tok_cos']:.6f} "
          f"flat_cos={after['flat_cos']:.6f} "
          f"rel_mse={after['rel_mse']:.4g}  steps={step}  "
          f"stop={stop_reason} -> {norm_edits_path}", flush=True)
    metrics = {"layer": "norm", "train_target": "norm",
               "steps": step, "lr": float(args.lr_norm),
               "optimizer": "adamw", "stop_reason": stop_reason,
               "tripwire": {"fired": trip["fired"],
                            "lr_scale": trip["lr_scale"]},
               "best_step": best["step"], "before": before,
               "after": after, "n_holdout_rows": len(holdout_rows),
               "norm_gain_edit": entry,
               "wall_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(out_dir, "final_norm"), exist_ok=True)
    _atomic_json_dump(metrics, os.path.join(out_dir, "final_norm",
                                            "metrics.json"))
    return metrics


# ---------------------------------------------------------------------------
# The driver's loader/fingerprint family (provenance identity + inputs)
# ---------------------------------------------------------------------------

def _load_pinned_text_config(model_ref):
    """Text config of the checkpoint with a PINNED attention backend —
    config-only load (no weights, no model build). A fresh config silently
    runs eager attention and a direct layer call with a None mask is then
    uncausal (see _attn_impl), so None is never accepted here."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    cfg = _text_config(cfg)
    if getattr(cfg, "_attn_implementation", None) is None:
        cfg._attn_implementation = "sdpa"
        print("  [trainer] config._attn_implementation was None — pinned "
              "to 'sdpa' (direct layer calls must not run uncausal eager "
              "attention)", flush=True)
    if not int(getattr(cfg, "num_hidden_layers", 0)):
        raise SystemExit(
            f"{model_ref}: the text config carries no num_hidden_layers — "
            f"cannot determine the layer count without loading the model")
    return cfg


def _parse_layers_ordered(spec, num_layers):
    """--layers for the qlora path: same grammar as _parse_layers but the
    GIVEN order is preserved (worst-first scheduling) and duplicates keep
    their first position."""
    if spec is None:
        return list(range(num_layers))
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [L for L in out if not (0 <= L < num_layers)]
    if bad:
        raise SystemExit(f"layer indices out of range: {bad} "
                         f"(model has {num_layers} layers)")
    seen = set()
    ordered = []
    for L in out:
        if L not in seen:
            seen.add(L)
            ordered.append(L)
    return ordered


def _file_sha256(path):
    """sha256 of a file (None-safe); resolves a dir to warm_starts.pt when
    it looks like one (the --warm-start fingerprint)."""
    if os.path.isdir(path):
        cand = os.path.join(path, "warm_starts.pt")
        if os.path.isfile(cand):
            path = cand
    if not os.path.isfile(path):
        raise SystemExit(f"cannot fingerprint {path!r} — not a file")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _dir_fingerprint(path):
    """A stable fingerprint of a completed adapter dir (the --init-from
    resume identity): sha256 over qlora_adapters.pt + the canonical rank
    map of qlora_config.json."""
    if not os.path.isdir(path):
        raise SystemExit(
            f"--init-from {path!r}: not a directory — pass the Stage-1 "
            f"run's output dir (the one with qlora_adapters.pt)")
    for req in ("qlora_adapters.pt", "qlora_config.json"):
        if not os.path.isfile(os.path.join(path, req)):
            raise SystemExit(
                f"--init-from {path!r}: missing {req} — not a completed "
                f"adapter dir")
    parts = [_file_sha256(os.path.join(path, "qlora_adapters.pt"))]
    with open(os.path.join(path, "qlora_config.json")) as f:
        cfg = json.load(f)
    rm = cfg.get("rank_map")
    if rm is not None:
        blob = json.dumps(rm, sort_keys=True, separators=(",", ":"))
        parts.append("sha256:" + hashlib.sha256(
            blob.encode("utf-8")).hexdigest())
    return "+".join(parts)


def _iter_init_snapshots(init_dir):
    """{layer: snapshot path} of a completed run's per-layer adapter
    snapshots (the Stage-1.5 trajectory-replay source)."""
    out = {}
    root = os.path.join(init_dir, "qlora_layers")
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if not name.startswith("layer_"):
            continue
        try:
            L = int(name[len("layer_"):])
        except ValueError:
            continue
        p = os.path.join(root, name, "adapter.pt")
        if os.path.isfile(p):
            out[L] = p
    return out


def _load_spectrum_predictions(path):
    """--spectrum: the per-module predicted post-fit rel_mse mapping
    (the G1a instrument) + per-layer energy-composed predictions for the
    stop target. Validated loudly (the spectrum schema, every rank a
    valid mult-4)."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"--spectrum {path}: cannot read ({e})")
    if not isinstance(doc, dict) \
            or doc.get("schema") != "functional_spectrum_v1" \
            or not isinstance(doc.get("modules"), dict):
        raise SystemExit(
            f"--spectrum {path}: unknown schema — expected the spectrum "
            f"subcommand's functional_spectrum.json")
    out = {}
    for k, m in doc["modules"].items():
        try:
            r = int(m.get("r", 0))
        except (TypeError, ValueError):
            raise SystemExit(f"--spectrum {path}: bad rank for {k!r}")
        if r < 0 or (r != 0 and r % 4 != 0):
            raise SystemExit(
                f"--spectrum {path}: module {k!r} rank {r} violates the "
                f"mult-4 attach contract")
        out[k] = {
            "r": r,
            "rel_mse": float(m.get("rel_mse", 0.0)),
            "predicted_post_rel_mse": float(
                m.get("predicted_post_rel_mse", 0.0)),
            "r_reason": str(m.get("r_reason", "")),
            "teacher_energy": float(m.get("teacher_energy", 0.0)),
            # W4-T05: the worst-first sweep ranking's input (the
            # spectrum's per-module error energy; summed per layer)
            "err_energy": float(m.get("err_energy", 0.0)),
        }
    if not out:
        raise SystemExit(f"--spectrum {path}: no module records")
    return out


def _layer_predicted_rel_mse(spectrum, L):
    """Energy-composed per-layer predicted post-fit rel_mse (the stop
    target's 1.15x basis): sum(unabsorbed error) / sum(teacher output
    energy) over the layer's modules — an ESTIMATE (module outputs are
    not strictly additive at the layer output; the G1a per-module
    comparison is the precise instrument). 0.0 when no spectrum given."""
    if not spectrum:
        return 0.0
    num = den = 0.0
    prefix = f"model.layers.{L}."
    for k, m in spectrum.items():
        if not k.startswith(prefix):
            continue
        te = float(m.get("teacher_energy", 0.0))
        if te <= 0.0:
            continue
        num += float(m["predicted_post_rel_mse"]) * te
        den += te
    return num / den if den > 0 else 0.0


def _load_pilot_json(path):
    """--pilot-json: the G1-O Muon-vs-AdamW decision record, copied into
    the run provenance (the swept numbers + the adoption call)."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"--pilot-json {path}: cannot read ({e})")
    if not isinstance(doc, dict):
        raise SystemExit(f"--pilot-json {path}: not a JSON object")
    return doc


def _load_rank_map(path, num_layers):
    """Parse a rank_map.json of the distill_rank_alloc schema and validate
    that every key names a layer of THIS model — a typo'd or foreign key
    fails loudly here, before any training (never a silent uniform
    fallback)."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except OSError as e:
        raise SystemExit(f"--rank-map {path}: cannot read ({e})")
    except json.JSONDecodeError as e:
        raise SystemExit(f"--rank-map {path}: not valid JSON ({e})")
    if not isinstance(doc, dict) or not isinstance(doc.get("rank_map"), dict):
        raise SystemExit(
            f"--rank-map {path}: unknown schema — expected the "
            f"distill_rank_alloc.py output (a top-level 'rank_map' mapping "
            f"of module path -> rank)")
    bad = []
    for key in doc["rank_map"]:
        parts = key.split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers" \
                or not parts[2].lstrip("-").isdigit() \
                or not (0 <= int(parts[2]) < num_layers):
            bad.append(key)
    if bad:
        shown = ", ".join(sorted(bad)[:8]) + \
            (", ..." if len(bad) > 8 else "")
        raise SystemExit(
            f"--rank-map {path}: {len(bad)} key(s) do not name a module of "
            f"model.layers.0..{num_layers - 1} (got: {shown}) — refusing to "
            f"silently drop them")
    return doc["rank_map"]


def _rank_map_fingerprint(rank_map):
    """Resume identity of the rank decision: sha256 of the canonical JSON
    (sorted keys), or the recorded uniform default when no map was used."""
    if rank_map is None:
        return f"uniform-r{_QLORA_DEFAULT_RANK}-alpha{_QLORA_DEFAULT_ALPHA}"
    blob = json.dumps(rank_map, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _layer_fingerprint(capture_id, rank_map, layer_idx, train_groups,
                        warm_start_fp, init_from_fp, seed):
    """The per-layer job identity (PROPOSAL §2.6 / W4-T08): capture id,
    rank-map id, scope (the layer index + the trainable-set selection),
    init identity (the warm-start sha / the init-from sha / cold), seed —
    everything a done layer's training depended on. A resumed layer whose
    recorded fingerprint differs from the current configuration is NOT
    the same job and the resume refuses (naming both identities)."""
    return {
        "capture": capture_id,
        "rank_map": _rank_map_fingerprint(rank_map),
        "scope": {"layer": int(layer_idx),
                  "train": sorted(train_groups)},
        "init": (warm_start_fp if warm_start_fp
                 else init_from_fp if init_from_fp else "cold"),
        "seed": int(seed),
    }


def _load_weights_alignment(path):
    """Parse a weights_alignment.json (the `weights` subcommand output);
    provenance only — per-module cos/rel_mse recorded into metrics."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except OSError as e:
        raise SystemExit(f"--weights-alignment {path}: cannot read ({e})")
    except json.JSONDecodeError as e:
        raise SystemExit(f"--weights-alignment {path}: not valid JSON ({e})")
    modules = doc.get("modules") if isinstance(doc, dict) else None
    if not isinstance(modules, dict) or not modules:
        raise SystemExit(
            f"--weights-alignment {path}: unknown schema — expected a "
            f"'modules' mapping as written by 'report.py weights'")
    return {k: {"cos": float(v["cos"]), "rel_mse": float(v["rel_mse"])}
            for k, v in modules.items()}


# ---------------------------------------------------------------------------
# The export machinery (PROPOSAL §2.8, P2) — the ground-truth artifact
# writer path for the LUT-escalation export; the adapter-dir export is
# _assemble_qlora_adapters above (the qlora channel's own export).
# ---------------------------------------------------------------------------

def _palettizer():
    """The ground-truth producer module (the canonical artifact writer).

    INTENTIONALLY ABSENT in this repo: the palettized model + heads
    artifacts exist pre-built and are deployed as-is — re-palettization
    is out of scope. The canonical artifact export (write_layer_artifacts
    / the `export` subcommand) is therefore unavailable here; serve
    fine-tuned weights through the adapter channel instead
    (qlora_adapters.pt + qlora_config.json ->
    eval_common.load_quant_model(qlora_adapters=...))."""
    raise SystemExit(
        "trainer: the canonical palettizer (palettize_qwen3_5_9b.py) is "
        "intentionally absent from this repo — the palettized model and "
        "heads artifacts are provided pre-built. The canonical artifact "
        "export path is unavailable; use the QLoRA adapter channel "
        "(qlora_adapters.pt + qlora_config.json) and load it with "
        "eval_common.load_quant_model(qlora_adapters=...).")


def _spectrum():
    """spectrum.py (warm-start factors + the fold-scale reader) — a
    sibling module of the source repo, intentionally not shipped here.
    The --warm-start path and the canonical export's fold-ratio step
    require it; default runs never touch this path."""
    raise SystemExit(
        "trainer: spectrum.py was not carried into this repo — the "
        "--warm-start path and the canonical export's fold-ratio step "
        "are unavailable. Run without --warm-start and serve the "
        "fine-tune through the QLoRA adapter channel instead.")


def _module_logical_indices(mod):
    """(N, K) uint8 torch indices reconstructed from the frozen module's
    idx4 blob (unpack->repack is the identity; idx4.self_test proves it)."""
    return torch.from_numpy(mod._logical_indices_numpy().astype(np.int64))


def write_layer_artifacts(out_dir, metadata, model, layer_idx, prov):
    """Rewrite every artifact of `layer_idx` from the (frozen, fp16-snapped)
    modules of the resident student model, using the ground-truth writer.
    Merges metadata in place: core fields refreshed, provenance fields
    ('assignment', 'gs_decision', 'awq', residual entries) preserved,
    'finetune' block added. Returns {tensor_name: sha256_lut}."""
    pal = _palettizer()
    shas = {}
    for tensor_name, old_meta in list(metadata["tensors"].items()):
        parts = tensor_name.split(".")
        if int(parts[2]) != layer_idx:
            continue
        parent, leaf = pmod.resolve_module(model, tensor_name)
        mod = getattr(parent, leaf)
        cos = float(prov["after"]["tok_cos"])
        _unlink_tensor_files(out_dir, tensor_name, pal)
        if isinstance(mod, pmod.SplitQKV):
            qkv_split = (mod.q_proj.N, mod.k_proj.N, mod.v_proj.N)
            split_results = []
            for comp_name, cm in (("Q", mod.q_proj), ("K", mod.k_proj),
                                  ("V", mod.v_proj)):
                split_results.append({
                    "part": comp_name,
                    "indices": _module_logical_indices(cm),
                    "lut": cm.lut,
                    "n_groups": int(cm.lut.shape[0]),
                    "group_size": int(cm.group_size),
                    "bitwidth": int(cm.bitwidth),
                    "cos": cos,
                })
            fresh = pal.write_palettized_tensor(
                out_dir, tensor_name,
                None, None, [sum(qkv_split), mod.q_proj.K], cos,
                is_qkv_split=True, qkv_split=qkv_split,
                split_results=split_results)
            # preserve per-component provenance fields + top-level marker
            for comp in ("Q", "K", "V"):
                oldc = old_meta.get("components", {}).get(comp, {})
                newc = fresh["components"][comp]
                merged = dict(oldc)
                merged.update(newc)
                merged["finetune"] = _fin_meta(prov)
                fresh["components"][comp] = merged
                shas[tensor_name + ":" + comp] = newc["sha256_lut"]
            fresh["finetune"] = _fin_meta(prov)
        else:
            fresh = pal.write_palettized_tensor(
                out_dir, tensor_name,
                _module_logical_indices(mod), mod.lut,
                [mod.N, mod.K], cos,
                actual_group_size=mod.group_size)
            merged = dict(old_meta)
            merged.update(fresh)
            merged["finetune"] = _fin_meta(prov)
            fresh = merged
            shas[tensor_name] = fresh["sha256_lut"]
        metadata["tensors"][tensor_name] = fresh
    return shas


def _fin_meta(prov):
    return {
        "layer": prov["layer"],
        "steps": prov["steps"],
        "scope": prov["scope"],
        "target_mode": prov["target_mode"],
        "tok_cos_before": prov["before"]["tok_cos"],
        "tok_cos_after": prov["after"]["tok_cos"],
        "flat_cos_before": prov["before"]["flat_cos"],
        "flat_cos_after": prov["after"]["flat_cos"],
        "rel_mse_before": prov["before"]["rel_mse"],
        "rel_mse_after": prov["after"]["rel_mse"],
    }


def _verify_layer_roundtrip(out_dir, metadata, model, layer_idx,
                           awq_scales=None):
    """Bit-exact deployment round-trip check: reload every artifact of this
    layer from disk (fresh modules, full sha256 validation) and compare the
    dequantized weights against the resident (frozen) modules. Catches any
    write/read drift before it ships.

    W14: `awq_scales` (the map the resident modules were compensated
    with) must be handed to the fresh load — the comparison runs through
    _quantized_weight, whose un-rotation sheds the AWQ scale on the
    module's own awq_scale. Without the map the fresh module un-rotates
    the PLAIN fold and every compensated resident falsely mismatches."""
    checked = 0
    for tensor_name, tmeta in metadata["tensors"].items():
        if int(tensor_name.split(".")[2]) != layer_idx:
            continue
        parent, leaf = pmod.resolve_module(model, tensor_name)
        mod = getattr(parent, leaf)
        fresh = pmod.load_palettized_weight(tmeta, out_dir,
                                            awq_scales=awq_scales)
        if isinstance(mod, pmod.SplitQKV):
            for comp, fm, cm in (("Q", fresh.q_proj, mod.q_proj),
                                 ("K", fresh.k_proj, mod.k_proj),
                                 ("V", fresh.v_proj, mod.v_proj)):
                _assert_dequant_equal(fm, cm, f"{tensor_name}:{comp}")
                checked += 1
        else:
            _assert_dequant_equal(fresh, mod, tensor_name)
            checked += 1
    return checked


def _assert_dequant_equal(a, b, name):
    Wa = a._quantized_weight().float()
    Wb = b._quantized_weight().float()
    # Move to CPU for comparison (layer may be on GPU)
    Wa = Wa.cpu()
    Wb = Wb.cpu()
    if not torch.equal(Wa, Wb):
        raise AssertionError(
            f"{name}: deployment round-trip mismatch after finetune "
            f"(max|dW|={float((Wa - Wb).abs().max()):.3e})")


def _materialize_output_dir(artifacts_dir: str, out_dir: str):
    """Self-contained deployment-ready copy of the artifacts dir.

    Unchanged files are hard-linked (zero extra space on the same volume).
    Trained tensors are unlinked before rewrite (see _unlink_tensor_files)
    so the ground-truth writer's open(..., 'wb') never writes through a
    link shared with the source — the source directory stays pristine."""
    if not os.path.exists(os.path.join(artifacts_dir, "metadata.json")):
        raise SystemExit(f"no metadata.json under {artifacts_dir}")
    os.makedirs(out_dir, exist_ok=True)
    linked = 0
    for f in os.listdir(artifacts_dir):
        src = os.path.join(artifacts_dir, f)
        dst = os.path.join(out_dir, f)
        if os.path.exists(dst) or not os.path.isfile(src):
            continue
        os.link(src, dst)
        linked += 1
    src_ne = os.path.join(artifacts_dir, "norm_edits")
    if os.path.isdir(src_ne):
        os.makedirs(os.path.join(out_dir, "norm_edits"), exist_ok=True)
        for f in os.listdir(src_ne):
            dstf = os.path.join(out_dir, "norm_edits", f)
            if not os.path.exists(dstf):
                os.link(os.path.join(src_ne, f), dstf)
                linked += 1
    return linked


def _unlink_tensor_files(out_dir: str, tensor_name: str, pal) -> None:
    """Remove the destination artifact files of `tensor_name` BEFORE the
    writer rewrites them (they may be hard links into the source artifacts
    directory; writing through them would corrupt the source). Idempotent.

    W4-T06 scope: ONLY the idx4/lut files the writer rewrites. The resA/
    resB files are FROZEN through training and stay hard-linked in the
    export copy — their metadata entries (preserved by write_layer_
    artifacts' old-then-new merge) keep pointing at valid, sha-matching
    files. Unlinking them would leave the export copy referencing files
    it no longer carries (the residual artifacts ARE part of the deployed
    weight, R1)."""
    san = pal.sanitize_name(tensor_name)
    names = [f"{san}.idx4", f"{san}.lut_scalar"]
    for comp in ("Q", "K", "V"):
        names += [f"{san}_{comp}.idx4", f"{san}_{comp}.lut_scalar"]
    for n in names:
        p = os.path.join(out_dir, n)
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# W4-T06 (PROPOSAL §2.8, P2 / §8 risk 1+3 / §2.9 G-J3): the export path.
# Per done layer, from the run's banked-best JOINT snapshot: the LUT
# masters are frozen (fp16 snap) and polished on the exact quadratic where
# the persisted calibration Gram exists, the canonical idx4/lut pair is
# written into a COPY of the artifacts dir (the source stays read-only),
# the reload is verified bit-exact, the trained norm gains land in the
# export's norm_gain_edits.json (the fold ratios read back through the
# ONE fold-aware helper, spectrum._fold_scales), and the reloaded module
# is evaluated on the training holdout (G-J3: rel_mse(snap) <= 1.05 x
# rel_mse(train)). The adapter dir is verified against every snapshot; the
# deployment merge hook is recorded (the box step).
# ---------------------------------------------------------------------------

def _export_joint_subset(joint):
    """The write-phase subset of a JOINT snapshot: the LUT masters and the
    norm gains with the adapter keys dropped (the adapter dir is their
    export), re-keyed from the WRAPPED shell's parameter names to the
    plain (unwrapped) shell's. Two renamings undo the wrappers: every
    QLoRALinear inserts one '.base.' segment (dropped), and the
    QLoRASplitQKV names its components q/k/v while the plain SplitQKV
    registers q_proj/k_proj/v_proj (no parameter path of the modeling
    contains a bare q/k/v segment outside that wrapper, so the rewrite is
    unambiguous)."""
    out = {}
    for k, v in joint.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue
        segs = [s for s in k.split(".") if s != "base"]
        fixed = [s + "_proj" if s in ("q", "k", "v") else s
                 for s in segs]
        out[".".join(fixed)] = v
    return out


def _export_resolve_gram(pm, artifacts_dir, san):
    """Path of a tensor's persisted calibration Gram, if any (the
    qlora_merge convention: the metadata's gram_file, then the
    grams/<san>.gram.npy fallback). None = no polish for this tensor."""
    rel = pm.get("gram_file")
    if rel:
        path = os.path.join(artifacts_dir, rel)
        if os.path.exists(path):
            return path
    fallback = os.path.join(artifacts_dir, "grams", f"{san}.gram.npy")
    return fallback if os.path.exists(fallback) else None


def _fold_polish_frame(cm, W_dense, H):
    """(W_target, H_target) for the codebook polish, in the module's OWN
    fold frame (W14).

    The polish quadratic must be computed in the frame the codebook
    actually lives in. For an unrotated module that is the identity
    (the pre-W14 behavior — target the dense teacher weight under the
    captured Gram). For a rotated/AWQ module the deployed computation
    is y = fold(h) @ (dequant(lut) + resA@resB)^T, so:

      legacy order  (W' = W @ T @ D):  target = (W @ T) @ D
                                       Gram  = D^-1 (T^T H T) D^-1
      W13 order     (W' = (W @ D) @ T): target = (W @ D) @ T
                                       Gram  = T^T (D^-1 H D^-1) T

    (derived from the module error e = z @ (Wq - W'_ideal)^T with the
    fold input z = X T D^-1 resp. X D^-1 T and H = X^T X the captured
    PRISTINE-input Gram; at load time the norm edit supplies the 1/s,
    which is the O(1) part of the input ratio — the trained-gain drift
    is second-order and ignored here, same approximation class as the
    polish itself.)

    The stored residual is FOLD-space, so it is subtracted from the
    target AFTER the fold (the codebook alone must approximate
    W'_ideal - resA@resB — the pre-W14 polish targeted the full weight
    and overshot by the residual on every residual-carrying artifact,
    rotated or not).

    `cm` is the resident PalettizedLinear (rotation records + AWQ
    scale + residual factors all live on it); `W_dense` the teacher's
    original-space (N, K) weight; `H` the captured (K, K) Gram. Both
    returns fp32 CPU."""
    W = W_dense.float().cpu()
    Hd = H.float().cpu()
    fht = pmod._get_fht()
    signs = cm.rot_signs
    if signs is None:
        target, gram = W, Hd
    else:
        signs = signs.to(W.device, torch.float32)
        if cm.awq_scale is not None and cm.fold_order == "rotate_then_awq":
            s = cm.awq_scale.to(W.device, torch.float32)
            target = fht.fht_apply(W, signs) * s.view(1, -1)
        elif cm.awq_scale is not None:      # awq_then_rotate (W13 order)
            s = cm.awq_scale.to(W.device, torch.float32)
            target = fht.fht_apply(W * s.view(1, -1), signs)
        else:                               # rotation only
            target = fht.fht_apply(W, signs)
        gram = pmod.fold_input_gram(signs, cm.awq_scale, cm.fold_order, Hd)
    if cm.resA is not None and cm.resB is not None:
        target = target - (cm.resA.float().cpu()
                           @ cm.resB.float().cpu()).to(target.dtype)
    return target, gram


def _polish_lut_grid(lut, W, idx, group_size, H):
    """The export's train/deploy gap minimizer (PROPOSAL §2.8 step 2, the
    §8 risk 1 mitigation): polish_lut_fp16 on the exact per-group
    quadratic of the damped Gram.

    The quadratic is computed in the STORED frame — the original column
    order of the persisted idx4/lut artifacts — with the palettizer's own
    damping constant. The act-order permutation the palettizer's engines
    used internally is a similarity on this quadratic (columns, indices
    and Gram permuted together relabel the same sums), so the un-permuted
    computation IS the palettizer's objective for the stored assignment.

    Returns fp32 values exactly on the fp16 grid (the later storage cast
    is a no-op)."""
    pal = _palettizer()
    W = W.float()
    Hd = pal._engine_gram(H.float().to(W.device), W, pal.V2_DAMP_PCT,
                          act_order=False)[0]
    M, b, _const = pal._group_quadratic(W, idx.to(W.device), lut,
                                        group_size, Hd)
    return pal.polish_lut_fp16(lut, M, b, _const)


def _norm_consumers_for(var_names, norm_param, layer_idx):
    """The fold consumers of one norm-gain edit (the palettizer's own AWQ
    structural convention, PROPOSAL §8 risk 3): the layer's palettized VAR
    paths whose module input is (structurally) this RMSNorm's output.

    input_layernorm feeds the attention branch's projections (q/k/v_proj,
    in_proj_qkv/z/a/b); post_attention_layernorm feeds the MLP's gate/up;
    the per-head q/k norms and the linear-attention gated norm have no
    direct palettized consumer ([] — nothing to claim). The consumers make
    spectrum._fold_scales see the student-input/teacher-input ratio the
    trained gain implies on the NEXT sweep over the exported dir."""
    toks = norm_param.split(".")
    if toks and toks[0] == "model":
        toks = toks[1:]
    if len(toks) >= 2 and toks[0] == "language_model":
        toks = toks[1:]
    attr = ".".join(toks[2:-1])
    prefix = f"model.layers.{layer_idx}."
    out = []
    for var in var_names:
        if not var.startswith(prefix):
            continue
        rel = var[len(prefix):-len(".weight")]
        leaf = rel.split(".")[-1]
        if attr == "input_layernorm":
            if rel.split(".")[0] in ("self_attn", "linear_attn") \
                    and leaf in ("q_proj", "k_proj", "v_proj", "in_proj_qkv",
                                 "in_proj_z", "in_proj_a", "in_proj_b"):
                out.append(var)
        elif attr == "post_attention_layernorm":
            if rel.startswith("mlp.") and leaf in ("gate_proj", "up_proj"):
                out.append(var)
    return out


def _resident_awq_scale_map(shell, layer_idx, metadata):
    """{consumer tensor name: the resident module's FROZEN awq_scale} for
    one layer (W14): the EXACT scales the resident modules were
    compensated with — the roundtrip's fresh-load input and the export
    pin's source of truth. SplitQKV components share the fused tensor's
    scale (one fold, one s); the map is keyed by the FUSED var name the
    loader resolves."""
    out = {}
    for tensor_name, pm in metadata["tensors"].items():
        if int(tensor_name.split(".")[2]) != layer_idx:
            continue
        parent, leaf = pmod.resolve_module(shell, tensor_name)
        mod = getattr(parent, leaf)
        mods = (mod.q_proj, mod.k_proj, mod.v_proj) \
            if isinstance(mod, pmod.SplitQKV) else (mod,)
        s = next((m.awq_scale for m in mods if m.awq_scale is not None),
                 None)
        if s is not None:
            out[tensor_name] = s
    return out


def _pin_awq_scale_file(shell, consumers, key, out_dir, pal, entry):
    """W14: pin the FROZEN compensation into a norm-edit entry.

    The trained-gain write-back replaces the .npy the diff-based
    recovery reads ((1+w_orig)/(1+w_edit)); without a recorded scale
    the NEXT load would infer a DRIFTED s and the deployed student
    would silently diverge from the trained one (the norm delta is
    exactly canceled by the renormalized h*s product). This writes the
    resident modules' awq_scale (the value the training actually used,
    bit-exact) to awq_scales/<san>.npy and records
    entry["awq_scale_file"/"awq_scale_sha256"] — pmod._recover_awq_scales
    prefers the record over the diff.

    Returns True when a pin landed (a non-empty consumer list whose
    resident modules carry no awq_scale means an unrotated/un-AWQ toy —
    no pin, the legacy entry shape stays valid). Refuses (loudly) a
    consumer set whose modules disagree on the scale."""
    s_pin = None
    for cname in consumers:
        try:
            parent, leaf = pmod.resolve_module(shell, cname)
        except Exception:
            continue        # a consumer outside this shell is not pin-able here
        mod = getattr(parent, leaf)
        mods = (mod.q_proj, mod.k_proj, mod.v_proj) \
            if isinstance(mod, pmod.SplitQKV) else (mod,)
        for m in mods:
            if m.awq_scale is None:
                continue
            if s_pin is None:
                s_pin = m.awq_scale
            elif not torch.equal(s_pin, m.awq_scale):
                raise RuntimeError(
                    f"_pin_awq_scale_file: {key}: consumers disagree on "
                    f"the frozen AWQ scale ({cname}) — one norm edit feeds "
                    f"ONE scale; corrupted artifacts")
    if s_pin is None:
        return False
    asan = pal.sanitize_name(key)
    adir = os.path.join(out_dir, "awq_scales")
    os.makedirs(adir, exist_ok=True)      # a NEW dir: no hard links
    apath = os.path.join(adir, f"{asan}.npy")
    np.save(apath, s_pin.detach().cpu().float().numpy())
    entry["awq_scale_file"] = os.path.relpath(apath, out_dir)
    entry["awq_scale_sha256"] = _file_sha256(apath).split(":")[1]
    entry["awq_scale_note"] = (
        "W14 pin: the FROZEN compensation the training used — the "
        "(1+w_orig)/(1+w_edit) diff moved when the trained gain "
        "replaced the edit; loaders MUST prefer this record "
        "(pmod._recover_awq_scales does)")
    return True


def _export_norm_edits(shell, layer_idx, subset, out_dir, norm_edits_doc,
                       layer_vars, pal):
    """(W4-T06 step 3 + the W14 pin) write one layer's trained norm
    gains into the export copy's norm_gain_edits.json.

    Every trained norm gain lands as the COMPLETE edited parameter
    (the loader's copy_ semantics), consumers/alpha preserved from the
    source entry, and — when the consumers' resident modules carry an
    AWQ compensation — the FROZEN scale is pinned alongside
    (_pin_awq_scale_file) so the reload reproduces the TRAINED
    deployment exactly. Updates `norm_edits_doc` in place and dumps it
    atomically; returns {key: entry}."""
    norm_edits_dir = os.path.join(out_dir, "norm_edits")
    os.makedirs(norm_edits_dir, exist_ok=True)
    norm_entries = {}
    for key, w_train in subset.items():
        if not any(key.endswith(s + ".weight")
                   for s in _NORM_GAIN_SUFFIXES):
            continue
        san = pal.sanitize_name(key)
        npy = os.path.join(norm_edits_dir, f"{san}.npy")
        if os.path.exists(npy):
            os.unlink(npy)      # never write through a hard link
        arr = w_train.detach().cpu().float().numpy()
        np.save(npy, arr)
        entry = {
            "file": os.path.relpath(npy, out_dir),
            "dtype": "float32",
            "shape": list(arr.shape),
            "consumers": _norm_consumers_for(layer_vars, key, layer_idx),
            "sha256": _file_sha256(npy).split(":")[1],
            "note": "joint-trainer gain: the COMPLETE edited parameter "
                    "(the loader's copy_ semantics); the fold ratio is "
                    "(1+w')/(1+w), never w'/w — spectrum._fold_scales is "
                    "the one reader of the convention",
        }
        old = norm_edits_doc.get("edits", {}).get(key, {})
        if old.get("consumers"):
            entry["consumers"] = list(old["consumers"])
        if "alpha" in old:
            entry["alpha"] = old["alpha"]
        # W14: pin the frozen compensation BEFORE the entry lands (the
        # consumers come from the old entry when it has them)
        _pin_awq_scale_file(shell, entry["consumers"], key, out_dir, pal,
                            entry)
        norm_edits_doc.setdefault("edits", {})[key] = entry
        norm_entries[key] = entry
    _atomic_json_dump(norm_edits_doc,
                      os.path.join(out_dir, "norm_gain_edits.json"))
    return norm_entries


def _export_layer_artifacts(L, cfg, teacher_source, snap, met, rank_map,
                            prov_model, store, holdout_rows, artifacts_dir,
                            out_dir, metadata, device, norm_edits_doc,
                            pal):
    """One layer's export pass (W4-T06 steps 1-3 + the reload check).

    `prov_model` is the run's checkpoint ref, `store` its capture store
    and `holdout_rows` its holdout row list (the G-J3 harness re-evaluates
    on the TRAINING's split — the seeded split is part of the run's
    identity). Returns the layer's export record ({shas, roundtrip_modules,
    polished, polish_skipped, norm_edits, folds, gj3}) — everything the
    export report records per layer. Raises on any inconsistency (a bad
    export never ships silently)."""
    sd = teacher_source.layer_state(L)

    # -- the plain (unwrapped) shell: the write/verify machinery reads
    # -- module attributes (mod.lut, mod.q_proj...) directly — the
    # -- QLoRALinear wrappers would shadow them
    layer, _n = materialize_student_layer(
        L, cfg, _PreloadedLayerSource(sd, L), artifacts_dir, metadata,
        device=device, dtype=torch.float16)
    shell = StudentLayerShell(layer, L, cfg, device=device)

    # the JOINT restore (luts + norms; the adapter keys are not the
    # write phase's business): promote every LUT, copy the banked-best
    # values in through the strict loader (a key the plain shell cannot
    # resolve is a loud geometry mismatch), then freeze (step 1: the
    # fp16 snap from the banked best).
    for _name, mod in pmod.iter_palettized_linears(shell):
        mod.make_trainable()
    subset = _export_joint_subset(snap["joint"])
    _load_lora_state_dict(shell, subset,
                          ctx=f"L{L} export joint restore")
    for _name, mod in pmod.iter_palettized_linears(shell):
        mod.freeze_lut(snap_fp16=True)

    # -- step 2: the polish grid re-snap where the Gram exists ----------
    polished, polish_skipped = [], []
    layer_vars = []
    for tensor_name, pm in metadata["tensors"].items():
        if int(tensor_name.split(".")[2]) != L:
            continue
        layer_vars.append(tensor_name)
        san = pal.sanitize_name(tensor_name)
        gram_path = _export_resolve_gram(pm, artifacts_dir, san)
        parent, leaf = pmod.resolve_module(shell, tensor_name)
        mod = getattr(parent, leaf)
        W = teacher_source.load_dense_tensor(tensor_name).float()
        if isinstance(mod, pmod.SplitQKV):
            comps, Ws, off = (mod.q_proj, mod.k_proj, mod.v_proj), [], 0
            for cm in comps:
                Ws.append(W[off:off + cm.N])
                off += cm.N
        else:
            comps, Ws = (mod,), (W,)
        for cm, Wc, ctag in zip(
                comps, Ws, ("Q", "K", "V") if isinstance(mod, pmod.SplitQKV)
                else (None,)):
            tag = f"{tensor_name}:{ctag}" if ctag else tensor_name
            if gram_path is None:
                polish_skipped.append(tag)
                continue
            H = torch.from_numpy(
                np.load(gram_path).astype(np.float32))
            # W14: the polish runs in the module's fold frame — the
            # teacher weight is folded (and the stored residual
            # subtracted) so the quadratic the codebook minimizes is
            # the one the DEPLOYED module actually serves.
            W_target, H_frame = _fold_polish_frame(cm, Wc, H)
            idx = _module_logical_indices(cm)
            lut16 = _polish_lut_grid(cm.lut, W_target, idx, cm.group_size,
                                     H_frame)
            cm.lut = lut16.to(torch.float16).contiguous()
            polished.append(tag)

    # -- step 1's write: the canonical writer into the COPY -------------
    fin_prov = {"layer": L, "steps": met["steps"], "scope": "all",
                "target_mode": met.get("input_mode", "captured"),
                "before": met["before"], "after": met["after"]}
    shas = write_layer_artifacts(out_dir, metadata, shell, L, fin_prov)
    # the merged metadata lands atomically (tmp+rename — never written
    # through the hard link into the source dir)
    _atomic_json_dump(metadata, os.path.join(out_dir, "metadata.json"))
    # W14: the roundtrip's fresh load must be compensated with the SAME
    # scales the resident modules carry (the comparison runs through
    # _quantized_weight, whose un-rotation sheds the module's awq_scale)
    rt_awq = _resident_awq_scale_map(shell, L, metadata)
    n_checked = _verify_layer_roundtrip(out_dir, metadata, shell, L,
                                        awq_scales=rt_awq)

    # -- step 3: the norm-gain edits (the fold-aware ratio + W14 pin) --
    norm_entries = _export_norm_edits(shell, L, subset, out_dir,
                                      norm_edits_doc, layer_vars, pal)

    # the fold ratios THROUGH the one helper (spectrum's own reader —
    # the next sweep over the exported dir sees exactly these scales)
    spectrum = _spectrum()  # noqa: E402  (absent in this repo: loud error)
    folds = {}
    for var, rec_f in spectrum._fold_scales(L, sd, out_dir,
                                            layer_vars).items():
        if rec_f is None:
            continue
        s = rec_f["s"]
        folds[rec_f["norm"]] = {"s_min": float(s.min()),
                                "s_max": float(s.max())}

    del layer, shell
    gc.collect()

    # -- step 5: the G-J3 reload (the deployment state, re-evaluated) ---
    out_metadata = pmod.load_metadata(out_dir)
    gj3_layer, _m = materialize_student_layer(
        L, cfg, _PreloadedLayerSource(sd, L), out_dir, out_metadata,
        device=device, dtype=torch.float16)
    gj3_shell = StudentLayerShell(gj3_layer, L, cfg, device=device)
    args_ns = types.SimpleNamespace(model=prov_model,
                                    artifacts_dir=out_dir)
    acfg, _slice = _attach_layer_adapters(gj3_shell, L, rank_map, args_ns)
    if acfg is None:
        raise RuntimeError(
            f"export L{L}: the run's rank map puts every module at r=0 "
            f"but the snapshot carries adapters — geometry mismatch")
    _load_lora_state_dict(gj3_shell, snap["lora"],
                          ctx=f"L{L} G-J3 reload")
    teacher = teacher_source.load_layer(L, cfg, sd=sd)
    pos16 = _position_embeddings(gj3_shell, store.seq, torch.float16,
                                 device)

    def reader_rows(rows, dev, dtype=torch.float32):
        if len(rows) == 0:
            return torch.empty(0, store.seq, store.hidden, device=dev,
                               dtype=dtype)
        return store.h_rows(L, rows, dev, dtype)

    def target_fn(rows, dev):
        x16 = reader_rows(list(rows), dev, torch.float16)
        y16 = _forward_layer(teacher, x16,
                             _pos_batch(pos16, x16.shape[0]), gj3_shell)
        return y16.detach().float()

    def output_fn(x, pos):
        return _forward_layer(gj3_layer, x, pos, gj3_shell)

    snap_eval = _evaluate(gj3_layer, pos16, reader_rows, target_fn,
                          holdout_rows, device, 8, 1.0, gj3_shell,
                          output_fn=output_fn,
                          input_dtype=torch.float16, cos_weight=0.05)
    del gj3_layer, gj3_shell, teacher
    teacher_source.drop_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rel_train = float(met["after"]["rel_mse"])
    rel_snap = float(snap_eval["rel_mse"])
    if not (rel_snap <= 1.05 * rel_train):
        ratio = rel_snap / rel_train if rel_train > 0 else float("inf")
        raise RuntimeError(
            f"[export] L{L:02d}: G-J3 FAILED — rel_mse(snap)={rel_snap:.6g} "
            f"> 1.05 x rel_mse(train)={rel_train:.6g} (ratio {ratio:.3f}x). "
            f"The export copy's train/deploy gap exceeds the PROPOSAL "
            f"§2.9 bound — do not deploy this artifact (WP5, the dL/dLUT "
            f"kernel, is the documented escalation).")
    print(f"  [L{L:02d}] G-J3: rel_mse(snap)={rel_snap:.4g} <= 1.05 x "
          f"rel_mse(train)={rel_train:.4g} [PASS]", flush=True)

    return {"shas": shas, "roundtrip_modules": int(n_checked),
            "polished": polished, "polish_skipped": polish_skipped,
            "norm_edits": norm_entries, "folds": folds,
            "rel_mse_train": rel_train, "rel_mse_snap": rel_snap,
            "n_polished": len(polished),
            "n_polish_skipped": len(polish_skipped)}


# ---------------------------------------------------------------------------
# W4-T07 (PROPOSAL §7.1): the 2-config pilot — the codes-LR decision.
# Two configurations run through the SAME layer job per pilot layer (the
# W5.T3 seam: the teacher and the target cache are built ONCE per layer
# and shared by the grid); the decision rule — the arm with the largest
# banked first-layer holdout rel_mse drop among the arms whose tripwire
# never fired — lands in <out>/pilot_joint.json.
# ---------------------------------------------------------------------------

_PILOT_ALT_KNOBS = {"lr-lut": "lr_lut", "lr-norm": "lr_norm",
                    "lr-lora": "lr"}


def _pilot_decide(arms, decision_layer):
    """The §7.1 decision rule, pure: given {arm: {"layers": {L: rec}}},
    adopt the arm with the largest banked `decision_layer` holdout
    rel_mse drop (before - banked_best) among the arms whose tripwire
    never fired on that layer. Both arms tripped -> chosen None (the
    operator re-runs with lower LRs — an honest non-decision, never a
    silent pick). Returns (chosen, record)."""
    key = str(decision_layer)
    rec = {"decision_layer": int(decision_layer), "eligible": {},
           "drops": {}}
    for name, arm in arms.items():
        lr = arm["layers"][key]
        fired = int(lr.get("tripwire_fired", 0))
        drop = float(lr["drop"])
        rec["drops"][name] = drop
        rec["eligible"][name] = fired == 0
    eligible = [n for n in rec["eligible"] if rec["eligible"][n]]
    if not eligible:
        rec["chosen"] = None
        rec["reason"] = ("every arm fired the tripwire on layer "
                         f"{decision_layer} — no adoptable configuration; "
                         "re-run the pilot with lower LRs")
        return None, rec
    if len(eligible) == 1:
        chosen = eligible[0]
    else:
        chosen = max(eligible, key=lambda n: (rec["drops"][n], n))
    rec["chosen"] = chosen
    rec["reason"] = (
        f"largest banked layer-{decision_layer} holdout rel_mse drop "
        f"without a tripwire firing: "
        + ", ".join(f"{n} {rec['drops'][n]:.6g}"
                    for n in sorted(rec["drops"])))
    return chosen, rec


def cmd_pilot(args):
    """trainer.py pilot (W4-T07, PROPOSAL §7.1): the one-command LR
    decision. Two configs on --layers (the base and the --alt override,
    both carrying the same --lr-lora/--lr-norm and control plane) run
    through the SAME per-layer jobs — one teacher + one target cache per
    layer shared by the grid (the W5.T3 seam). The decision rule (the
    largest banked first-layer holdout rel_mse drop without a tripwire
    firing) and both arms' numbers (banked best, before, tripwire count,
    steps run) land in <out>/pilot_joint.json."""
    out_dir = os.path.abspath(args.out)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.patience * args.eval_every >= args.steps:
        raise SystemExit(
            f"early stop is arithmetically unreachable: patience "
            f"({args.patience}) x eval-every ({args.eval_every}) = "
            f"{args.patience * args.eval_every} >= steps ({args.steps})")
    if not args.capture:
        raise SystemExit("pilot requires --capture (the boundary-state "
                         "store)")
    if not os.path.isdir(args.model):
        raise SystemExit("pilot requires --model to be a LOCAL checkpoint "
                         "directory")
    if args.alt is None:
        raise SystemExit(
            "pilot is the 2-config decision — pass --alt KNOB VALUE (e.g. "
            "--alt lr-lut 1e-4; overridable knobs: lr-lut, lr-norm, "
            "lr-lora)")
    knob, value_s = args.alt
    if knob not in _PILOT_ALT_KNOBS:
        raise SystemExit(
            f"--alt {knob!r}: unknown knob — legal knobs: "
            f"{', '.join(sorted(_PILOT_ALT_KNOBS))}")
    try:
        alt_value = float(value_s)
    except ValueError:
        raise SystemExit(f"--alt {knob} {value_s!r}: not a number")
    if alt_value <= 0:
        raise SystemExit(f"--alt {knob} {value_s!r}: must be positive")
    if args.warm_start and not args.rank_map:
        raise SystemExit("--warm-start requires --rank-map (the spectrum's "
                         "rank map — a uniform default has no matching "
                         "warm-start geometry)")

    base_cfg = {"lr_lut": float(args.lr_lut),
                "lr_norm": float(args.lr_norm),
                "lr": float(args.lr_lora)}
    alt_cfg = dict(base_cfg)
    alt_cfg[_PILOT_ALT_KNOBS[knob]] = alt_value

    cfg = _load_pinned_text_config(args.model)
    store = CaptureStore(capture_dir=args.capture)
    if store.kind != "boundary":
        raise SystemExit(
            f"pilot requires a BOUNDARY store (format 3); {args.capture} "
            f"is a {store.kind!r} store")
    if store.num_layers != int(cfg.num_hidden_layers):
        raise SystemExit(
            f"layer count mismatch: config {cfg.num_hidden_layers} vs "
            f"capture {store.num_layers}")
    metadata = pmod.load_metadata(args.artifacts)
    rank_map = _load_rank_map(args.rank_map, int(cfg.num_hidden_layers)) \
        if args.rank_map else None
    wanted = _parse_layers_ordered(args.layers, int(cfg.num_hidden_layers))
    if len(wanted) < 1:
        raise SystemExit("--layers selects nothing")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("Joint pilot (2-config, PROPOSAL §7.1) — the codes-LR decision")
    print("=" * 78)
    print(f"  model      : {args.model}")
    print(f"  artifacts  : {args.artifacts}")
    print(f"  capture    : {store.kind} ({store.rows} rows x {store.seq} x "
          f"{store.hidden})")
    print(f"  layers     : {wanted}")
    print(f"  grid       : base (lr-lut {base_cfg['lr_lut']:.3g}) vs alt "
          f"({knob} {alt_value:.3g})")
    print(f"  steps      : {args.steps} eval-every {args.eval_every} "
          f"patience {args.patience}", flush=True)

    torch.manual_seed(args.seed)
    teacher_source = TeacherLayerSource(args.model, device=device)

    def _ns(lr_cfg):
        return types.SimpleNamespace(
            adam_eps=1e-8,
            train_groups=frozenset(("luts", "norms", "lora")),
            lr_lut=lr_cfg["lr_lut"], lr_norm=lr_cfg["lr_norm"],
            lr=lr_cfg["lr"], optimizer=args.opt_lora,
            model=args.model, artifacts_dir=args.artifacts,
            capture_dir=args.capture, output_dir=out_dir,
            rank_map=args.rank_map, warm_start=args.warm_start,
            layers=args.layers, input_mode="captured",
            max_steps=args.steps, warmup=0, eval_every=args.eval_every,
            patience=args.patience, batch_rows=args.rows_batch,
            eval_batch_rows=8, eval_attn_budget_gib=4.0,
            holdout=args.holdout, holdout_split="random", seed=args.seed,
            clip=1.0, cos_weight=0.05, mse_weight=1.0,
            attn_tap_weight=0.0, train_target="qlora", device=device,
            target_cos=0.9995, target_rel_mse=5e-4, log_every=25,
            target_cache=True, o1_every=0, o1_probe_cmd=None,
            pilot_json=None, init_from=None, spectrum=None,
            weights_alignment=None, propagate_kernel=False, tf32=True,
            muon_diagnostics=False, profile_eval=0, dump_step0_diff=False,
            hold_steps=0, warm_start_gauge="balanced",
            warm_start_skip="", legacy_hidden_state_dir=None,
            legacy_activations_dir=None, legacy_include_padding=False,
            cache_subdir="student_cache", cache_keep_every=8,
            scope="all", target_mode="captured", resume=False)

    grid = [("base", _ns(base_cfg)), ("alt", _ns(alt_cfg))]
    arms = {
        "base": {"config": dict(base_cfg, opt_lora=args.opt_lora),
                 "override": None, "layers": {}},
        "alt": {"config": dict(alt_cfg, opt_lora=args.opt_lora),
                "override": {"knob": knob, "value": alt_value},
                "layers": {}},
    }

    # one teacher + one target cache per layer, shared by the grid (the
    # W5.T3 seam — the jobs' own readers re-derive per config; the target
    # cache is the shared expensive part)
    for L in wanted:
        sd = teacher_source.layer_state(L)
        teacher = teacher_source.load_layer(L, cfg, sd=sd)
        shell = StudentLayerShell(teacher, L, cfg, device=device)
        pos16 = _position_embeddings(shell, store.seq, torch.float16,
                                     device)

        def _reader(rows, dev, dtype=torch.float32):
            if len(rows) == 0:
                return torch.empty(0, store.seq, store.hidden,
                                   device=dev, dtype=dtype)
            return store.h_rows(L, rows, dev, dtype)

        tgt_cache = _TargetCache.build(teacher, shell, _reader, store.rows,
                                       store.seq, store.hidden, device, 8,
                                       f"pilot L{L}")
        print(f"  [L{L:02d}] shared teacher + target cache built "
              f"({len(grid)} config(s) ahead)", flush=True)
        for name, ns in grid:
            res = _run_qlora_layer_job(
                L, cfg, teacher_source, store, ns, metadata, rank_map,
                device, teacher=teacher, target_cache=tgt_cache)
            before = float(res["before"]["rel_mse"])
            banked = float(res["after"]["rel_mse"])
            rec = {"before": before, "banked_best": banked,
                   "drop": before - banked,
                   "tripwire_fired": int(res["tripwire"]["fired"]),
                   "steps": int(res["steps"]),
                   "best_step": int(res.get("best_step", 0)),
                   "stop_reason": res.get("stop_reason")}
            arms[name]["layers"][str(L)] = rec
            print(f"  [L{L:02d}] {name:4s}: before={before:.6g} "
                  f"banked={banked:.6g} drop={rec['drop']:.6g} "
                  f"tripwire={rec['tripwire_fired']} steps={rec['steps']}",
                  flush=True)
        del teacher, shell, tgt_cache, pos16
        teacher_source.drop_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    chosen, decision = _pilot_decide(arms, wanted[0])
    doc = {
        "schema": "pilot_joint_v1",
        "rule": ("adopt the arm with the largest banked first-layer "
                 "holdout rel_mse drop without a tripwire firing "
                 "(PROPOSAL §7.1)"),
        "decision_layer": int(wanted[0]),
        "arms": arms,
        "decision": decision,
    }
    path = os.path.join(out_dir, "pilot_joint.json")
    _atomic_json_dump(doc, path)
    if chosen is None:
        print(f"  [pilot] decision: NONE — {decision['reason']}",
              flush=True)
    else:
        print(f"  [pilot] decision: {chosen} — {decision['reason']}",
              flush=True)
    print(f"  [pilot] {path}", flush=True)


def cmd_export(args):
    """trainer.py export (W4-T06, PROPOSAL §2.8/§7.3): build the
    deployment artifacts COPY from a completed joint run.

    Per done layer, from the banked-best JOINT snapshot: freeze_lut
    (snap_fp16=True), the polish grid re-snap where the persisted
    calibration Gram exists, the canonical idx4/lut write (the input
    artifacts dir stays READ-ONLY — hard-link copy, unlink-before-
    rewrite), the bit-exact roundtrip, the trained norm gains into the
    export's norm_gain_edits.json (fold ratios verified through the ONE
    helper), and the G-J3 reload check. The adapter dir is verified
    against every layer snapshot; the deployment merge hook is printed
    and recorded (the box step). The report lands in
    <run>/export_report.json."""
    run_dir = os.path.abspath(args.run)
    out_dir = os.path.abspath(args.out)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- the run record (the identity of everything exported) ----------
    prov_path = os.path.join(run_dir, "finetune_provenance.json")
    if not os.path.isfile(prov_path):
        raise SystemExit(f"export: {prov_path} not found — --run must be "
                         f"a trainer train output dir")
    with open(prov_path) as f:
        prov = json.load(f)
    hp = prov.get("hyperparameters", {})
    artifacts_dir = prov["artifacts_dir"]
    if not os.path.isdir(artifacts_dir):
        raise SystemExit(
            f"export: the run's artifacts dir {artifacts_dir!r} does not "
            f"exist on this box — the export needs the read-only source")
    capture_dir = prov["capture"]["dir"]
    if not os.path.isdir(capture_dir):
        raise SystemExit(
            f"export: the run's capture dir {capture_dir!r} does not "
            f"exist on this box — the G-J3 harness reads the holdout rows")
    for req in ("qlora_adapters.pt", "qlora_config.json"):
        if not os.path.isfile(os.path.join(run_dir, req)):
            raise SystemExit(f"export: {run_dir} carries no {req} — the "
                             f"adapter dir is incomplete; re-run train")
    cfg = _load_pinned_text_config(prov["model"])
    store = CaptureStore(capture_dir=capture_dir)
    cap_id = prov["capture"]
    live_id = {"dir": cap_id["dir"], "kind": store.kind, "rows": store.rows,
               "seq_length": store.seq, "hidden_size": store.hidden,
               "num_layers": store.num_layers}
    if live_id != cap_id:
        raise RuntimeError(
            f"export: the capture at {capture_dir!r} no longer matches the "
            f"run's record ({cap_id!r} vs {live_id!r}) — not the same run")
    rank_map = prov.get("rank_map")
    metadata = pmod.load_metadata(artifacts_dir)
    done = {int(k): rec for k, rec in prov["layers"].items()
            if rec.get("done")}
    order = [L for L in (prov.get("order") or sorted(done))
             if L in done]

    # the G-J3 harness must see the TRAINING's holdout rows (the seeded
    # split is part of the run's identity)
    split_ns = types.SimpleNamespace(
        holdout=hp.get("holdout", 0.1),
        holdout_split=hp.get("holdout_split", "random"),
        seed=hp.get("seed", 42))
    holdout_rows, _train_rows = _holdout_split(split_ns, store.rows)

    print("=" * 78)
    print("Joint export (P2) — the deployment artifacts copy")
    print("=" * 78)
    print(f"  run        : {run_dir} ({len(done)} layer(s) done)")
    print(f"  artifacts  : {artifacts_dir} (READ-ONLY — the export writes "
          f"a copy)", flush=True)
    print(f"  out        : {out_dir}", flush=True)

    # ---- the COPY (hard links; rewritten files unlinked first) ---------
    _materialize_output_dir(artifacts_dir, out_dir)
    # the norm-edits doc starts from the INPUT's AWQ entries (re-read
    # from the source; never through the hard-linked copy)
    norm_edits_doc = {"edits": {}}
    ne_src = os.path.join(artifacts_dir, "norm_gain_edits.json")
    if os.path.exists(ne_src):
        with open(ne_src) as f:
            norm_edits_doc = json.load(f)

    pal = _palettizer()
    teacher_source = TeacherLayerSource(prov["model"], device=device)

    report_layers = {}
    lora_states = {}
    for L in order:
        rec = done[L]
        if not rec.get("has_adapter"):
            print(f"  [L{L:02d}] r=0-skip layer — the artifacts copy "
                  f"carries it byte-identical (nothing was trained)",
                  flush=True)
            continue
        snap_path = os.path.join(run_dir, rec["snapshot"])
        snap = torch.load(snap_path, map_location="cpu")
        if not snap.get("joint"):
            raise SystemExit(
                f"export: {snap_path} carries no 'joint' state — the run "
                f"predates W4-T06 (the joint snapshot landed with the "
                f"export path); re-run train into a fresh --out")
        with open(os.path.join(run_dir, rec["metrics"])) as f:
            met = json.load(f)
        lora_states[L] = snap["lora"]
        n_mods = sum(len(pm["components"]) if pm.get("is_qkv_split") else 1
                     for _tname, pm in metadata["tensors"].items()
                     if int(_tname.split(".")[2]) == L)
        print(f"  [L{L:02d}] export: {n_mods} module(s) "
              f"({met['steps']} steps, stop={met.get('stop_reason')})",
              flush=True)
        report_layers[str(L)] = _export_layer_artifacts(
            L, cfg, teacher_source, snap, met, rank_map, prov["model"],
            store, holdout_rows, artifacts_dir, out_dir, metadata, device,
            norm_edits_doc, pal)
        rep = report_layers[str(L)]
        note = ""
        if rep["n_polish_skipped"]:
            note = (f", {rep['n_polish_skipped']} plain-snap only (no "
                    f"persisted Gram)")
        print(f"  [L{L:02d}] wrote {len(rep['shas'])} artifact(s), "
              f"{rep['n_polished']} polished{note}, "
              f"{len(rep['norm_edits'])} norm edit(s), roundtrip "
              f"{rep['roundtrip_modules']} module(s) bit-exact", flush=True)

    # ---- the final-norm entry (the W4-T05 job's export) -----------------
    final_norm_copied = False
    run_ne = os.path.join(run_dir, "norm_gain_edits.json")
    if os.path.isfile(run_ne):
        with open(run_ne) as f:
            entry = json.load(f).get("edits", {}).get(
                "model.norm.weight")
        if entry is not None:
            src_npy = os.path.join(run_dir, entry["file"])
            dst_npy = os.path.join(out_dir, "norm_edits",
                                   "model.norm.weight.npy")
            if os.path.exists(dst_npy):
                os.unlink(dst_npy)
            with open(src_npy, "rb") as fi, open(dst_npy, "wb") as fo:
                fo.write(fi.read())
            new_entry = dict(entry)
            new_entry["file"] = os.path.relpath(dst_npy, out_dir)
            new_entry["sha256"] = _file_sha256(dst_npy).split(":")[1]
            norm_edits_doc.setdefault("edits", {})[
                "model.norm.weight"] = new_entry
            _atomic_json_dump(norm_edits_doc,
                              os.path.join(out_dir,
                                           "norm_gain_edits.json"))
            final_norm_copied = True
            print("  [norm] final-norm entry copied (model.norm.weight)",
                  flush=True)
    if not final_norm_copied:
        print("  [norm] no final-norm entry in the run — the model-level "
              "gain was not trained", flush=True)

    # ---- the adapter dir: the union of the snapshots, bit-equal --------
    sd_all = torch.load(os.path.join(run_dir, "qlora_adapters.pt"),
                        map_location="cpu")
    n_tensors = 0
    for L, lora_sd in sorted(lora_states.items()):
        for k, v in lora_sd.items():
            if k not in sd_all:
                raise RuntimeError(
                    f"export: adapter key {k!r} (layer {L}) is missing "
                    f"from the assembled qlora_adapters.pt — the run's "
                    f"assembly is incomplete; re-run train")
            if not torch.equal(sd_all[k], v):
                raise RuntimeError(
                    f"export: adapter tensor {k!r} (layer {L}) differs "
                    f"from the assembled qlora_adapters.pt")
            n_tensors += 1
    print(f"  [adapters] strict-subset check: {n_tensors} tensor(s), all "
          f"bit-equal to the assembled dir", flush=True)

    # ---- the deployment merge hook (the box step) ----------------------
    merge_cmd = (f"{sys.executable} "
                 f"{shlex.quote(os.path.join(_HERE, 'qlora_merge.py'))} "
                 f"--artifacts-dir {shlex.quote(out_dir)} "
                 f"--adapters-dir {shlex.quote(run_dir)} "
                 f"--output-dir {shlex.quote(os.path.join(out_dir, 'merged'))} "
                 f"--assign {args.merge_assign}")
    print(f"  [merge] box step: {merge_cmd}", flush=True)

    report = {
        "run": run_dir, "out": out_dir,
        "layers": report_layers,
        "final_norm_entry": final_norm_copied,
        "adapter_dir": {"tensors_checked": n_tensors},
        "merge_hook": merge_cmd,
        "gj3_all_pass": True,
    }
    _atomic_json_dump(report, os.path.join(run_dir, "export_report.json"))
    print(f"  export report: {os.path.join(run_dir, 'export_report.json')}",
          flush=True)

# ---------------------------------------------------------------------------
# The `report` subcommand — the box run's evidence digest (the intake
# protocol's tool: PROPOSAL 2.9's G-J gates from the run's own files)
# ---------------------------------------------------------------------------

def _report_load_json(path, what):
    if not os.path.isfile(path):
        raise SystemExit(f"report: {what} {path!r} does not exist")
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"report: {what} {path!r} is malformed JSON: {e}") from e


def _report_layer_lines(run_dir, prov):
    """Per-layer G-J2 lines from the provenance's done layers: the
    before/banked/after rel_mse + the improvement ratio."""
    lines = []
    for key in sorted(prov["layers"], key=lambda k: int(k)):
        rec = prov["layers"][key]
        if not rec.get("done"):
            continue
        met = _report_load_json(
            os.path.join(run_dir, rec["metrics"]),
            f"layer {key} metrics")
        try:
            before = float(met["before"]["rel_mse"])
            after = float(met["after"]["rel_mse"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(
                f"report: layer {key} metrics carries no numeric "
                f"before/after rel_mse ({e})")
        improvement = ((before - after) / before) if before > 0 else None
        lines.append({
            "layer": int(key), "block_type": met.get("block_type"),
            "before_rel_mse": before, "after_rel_mse": after,
            "improvement": improvement,
            "banked_step": met.get("best_step"),
            "steps": met.get("steps"),
            "stop_reason": met.get("stop_reason"),
        })
    if not lines:
        raise SystemExit(
            f"report: {run_dir!r} trains no done layer — nothing to "
            f"report")
    return lines


def _report_gj2(lines):
    """G-J2: the worst quartile of layers (by improvement, ascending)
    must improve >= 10% in the mean (PROPOSAL 2.9). A layer whose
    before rel_mse is <= 0 has no defined ratio and is excluded from
    the quartile (counted in the note)."""
    usable = [l for l in lines if l["improvement"] is not None]
    n_excluded = len(lines) - len(usable)
    if not usable:
        return {"verdict": "unknown",
                "reason": "no layer carries a positive before rel_mse — "
                          "the improvement ratio is undefined"}
    n_q = max(1, (len(usable) + 3) // 4)
    quartile = sorted(usable, key=lambda l: l["improvement"])[:n_q]
    q_mean = sum(l["improvement"] for l in quartile) / n_q
    return {
        "verdict": "pass" if q_mean >= 0.10 else "fail",
        "worst_quartile_mean_improvement": q_mean,
        "threshold": 0.10, "n_layers": len(usable),
        "quartile_layers": n_q,
        "quartile_layer_ids": [l["layer"] for l in quartile],
        "n_ratio_undefined": n_excluded,
    }


def _report_gj3(run_dir):
    """G-J3: every exported layer's roundtrip ratio
    rel_mse(snap)/rel_mse(train) <= 1.05 (PROPOSAL 2.9; the export
    itself refuses a violation — a False here means tampering)."""
    path = os.path.join(run_dir, "export_report.json")
    if not os.path.isfile(path):
        return {"verdict": "unknown",
                "reason": "no export_report.json in the run dir — the "
                          "export step has not run"}
    rep = _report_load_json(path, "export report")
    layers = rep.get("layers")
    if not isinstance(layers, dict) or not layers:
        return {"verdict": "unknown",
                "reason": "export_report.json carries no layer records"}
    ratios = {}
    for key, entry in layers.items():
        try:
            train = float(entry["rel_mse_train"])
            snap = float(entry["rel_mse_snap"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(
                f"report: export_report.json layers[{key!r}] carries no "
                f"numeric rel_mse_train/rel_mse_snap ({e})")
        ratios[key] = (snap / train) if train > 0 else None
    ok = [r for r in ratios.values() if r is not None]
    if not ok:
        return {"verdict": "unknown",
                "reason": "no exported layer carries a positive "
                          "rel_mse_train — the ratio is undefined"}
    worst = max(ok)
    return {"verdict": "pass" if all(r <= 1.05 for r in ok) else "fail",
            "max_ratio": worst, "threshold": 1.05,
            "n_layers": len(ok), "ratios": ratios,
            "export_report": path}


def _report_gj4(path):
    """G-J4: the O-1 paired gap after export (dense vs base+adapters)
    <= 0.04 nats/doc, computed from the probe report's per_doc vectors
    with the shared paired_mean semantics."""
    rep = _report_load_json(path, "o1 report")
    results = rep.get("results", {})
    per = {}
    for stage in ("dense_fp16", "base_adapterN"):
        st = results.get(stage) or {}
        vec = st.get("per_doc")
        if not isinstance(vec, list) or not vec:
            return {"verdict": "unknown",
                    "reason": f"the o1 report carries no "
                              f"{stage}.per_doc vector — the paired gap "
                              f"needs the dense and base+adapters stages"}
        per[stage] = [float(x) for x in vec]
    if len(per["dense_fp16"]) != len(per["base_adapterN"]):
        raise SystemExit(
            "report: the o1 report's dense and base+adapters per_doc "
            "vectors differ in length — the paired mean only holds "
            "doc-for-doc")
    gap = paired_mean(per["dense_fp16"], per["base_adapterN"])
    return {"verdict": "pass" if gap <= 0.04 else "fail",
            "paired_gap_nats_per_doc": gap, "threshold": 0.04,
            "n_docs": len(per["dense_fp16"]), "o1_report": path}


def _report_gj5(path):
    """G-J5: greedy equivalence — mean first-divergence >= 25 AND the
    exact-match fraction >= 0.15 (PROPOSAL 2.9)."""
    rep = _report_load_json(path, "greedy report")
    agg = rep.get("aggregate") or {}
    mfd = agg.get("mean_first_divergence")
    frac = agg.get("exact_match_fraction")
    out = {"greedy_report": path,
           "mean_first_divergence": mfd,
           "exact_match_fraction": frac,
           "thresholds": {"mean_first_divergence": 25,
                          "exact_match_fraction": 0.15}}
    if not isinstance(mfd, (int, float)) or \
            not isinstance(frac, (int, float)):
        out.update({"verdict": "unknown",
                    "reason": "the greedy report's aggregate carries no "
                              "numeric mean_first_divergence / "
                              "exact_match_fraction"})
        return out
    ok = (float(mfd) >= 25.0) and (float(frac) >= 0.15)
    out["verdict"] = "pass" if ok else "fail"
    return out


def cmd_report(args):
    """`trainer report --run <dir>`: the box run's evidence digest.

    Reads the run manifest (finetune_provenance.json), the per-layer
    banked lines (qlora_layers/layer_<L>/metrics.json), the export
    roundtrip lines (export_report.json), and — when given — the pilot
    decision (pilot_joint.json) and the eval JSONs (the O-1 probe
    report, the greedy report). Emits reports/joint_<tag>.json plus
    the verdict table: G-J2 (worst-quartile banked improvement),
    G-J3 (roundtrip ratios), G-J4 (paired gap) and G-J5 (greedy
    equivalence) — every verdict from the run's own files, never a
    remembered number (PROPOSAL 2.9; the PROMPT 9 intake protocol).
    """
    run_dir = os.path.abspath(args.run)
    prov = _report_load_json(
        os.path.join(run_dir, "finetune_provenance.json"),
        "run provenance")
    if not isinstance(prov.get("layers"), dict):
        raise SystemExit(
            "report: the provenance carries no 'layers' map — not a "
            "trainer train output dir")
    tag = args.tag or os.path.basename(run_dir)
    out_path = args.out or os.path.join("reports", f"joint_{tag}.json")

    lines = _report_layer_lines(run_dir, prov)
    gj2 = _report_gj2(lines)
    gj3 = _report_gj3(run_dir)
    gj4 = (_report_gj4(args.o1_report) if args.o1_report else
           {"verdict": "unknown",
            "reason": "no --o1-report given — the paired gap needs the "
                      "probe report with the dense and base+adapters "
                      "stages"})
    gj5 = (_report_gj5(args.greedy_report) if args.greedy_report else
           {"verdict": "unknown",
            "reason": "no --greedy-report given"})
    pilot = None
    if args.pilot:
        pilot = _report_load_json(args.pilot, "pilot decision")

    doc = {
        "schema": "joint_report_v1", "tag": tag,
        "run": run_dir,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gj2": {**gj2, "per_layer": lines},
        "gj3": gj3, "gj4": gj4, "gj5": gj5,
        "pilot": pilot,
        "thresholds_source": "PROPOSAL 2.9",
        "note": "every number reads from the run's own files at report "
                "time; the intake protocol files this JSON and the run "
                "dir's reports (never the weights) under reports/",
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _atomic_json_dump(doc, out_path)

    print("=" * 78, flush=True)
    print(f"joint report — {tag} (run {run_dir})", flush=True)
    print("=" * 78, flush=True)
    if gj2["verdict"] != "unknown":
        print(f"  G-J2 worst-quartile banked improvement: "
              f"{gj2['worst_quartile_mean_improvement']:.4f} "
              f"(target >= 0.10 over {gj2['quartile_layers']} of "
              f"{gj2['n_layers']} layers)  "
              f"{gj2['verdict'].upper()}", flush=True)
    else:
        print(f"  G-J2: UNKNOWN — {gj2['reason']}", flush=True)
    if gj3["verdict"] != "unknown":
        print(f"  G-J3 export roundtrip: max ratio "
              f"{gj3['max_ratio']:.4f} (<= 1.05 over "
              f"{gj3['n_layers']} layer(s))  {gj3['verdict'].upper()}",
              flush=True)
    else:
        print(f"  G-J3: UNKNOWN — {gj3['reason']}", flush=True)
    if gj4["verdict"] != "unknown":
        print(f"  G-J4 O-1 paired gap: "
              f"{gj4['paired_gap_nats_per_doc']:.4f} nats/doc "
              f"(<= 0.04 over {gj4['n_docs']} docs)  "
              f"{gj4['verdict'].upper()}", flush=True)
    else:
        print(f"  G-J4: UNKNOWN — {gj4['reason']}", flush=True)
    if gj5["verdict"] != "unknown":
        print(f"  G-J5 greedy equivalence: mean first-div "
              f"{gj5['mean_first_divergence']:.1f} (>= 25), exact "
              f"{gj5['exact_match_fraction']:.4f} (>= 0.15)  "
              f"{gj5['verdict'].upper()}", flush=True)
    else:
        print(f"  G-J5: UNKNOWN — {gj5['reason']}", flush=True)
    if pilot:
        dec = pilot.get("decision") or {}
        print(f"  pilot decision: {dec.get('choice', '?')} — "
              f"{dec.get('reason', '?')}", flush=True)
    print(f"  per-layer lines: {len(lines)}  (in {out_path})", flush=True)
    print(f"wrote {out_path}", flush=True)
    return doc


# ---------------------------------------------------------------------------
# The `train` subcommand — the adapter-only channel driver (ported from
# the engine's _cmd_finetune_qlora; W2-T07). The per-layer job (the
# control plane: banking/patience/tripwire/anchor/eval/export) is the
# trainer's own since W2-T08 (PROPOSAL §2.6, _run_qlora_layer_job above).
# ---------------------------------------------------------------------------

def _to_engine_args(args):
    """Map the trainer's 25-flag surface onto the layer-job namespace
    (the engine's finetune attribute names — the job's arg contract).
    The flags the trainer does not expose are pinned to the engine's
    finetune defaults (parity: `trainer.py train` at its defaults
    behaves exactly like `finetune --train-target qlora` at its
    defaults, modulo the PROPOSAL §2.6 tripwire semantics). --adam-eps
    is LIVE since W2-T08 (the ported optimizer factory takes it; the
    engine's factory pinned 1e-8, which stays the default here).
    --train is LIVE since W4-T01 (the joint trainable-set selection;
    parsed here so a bad value refuses at CLI entry, not mid-job),
    and --lr-lut/--lr-norm since W4-T02 (the codes channel's split
    AdamW LRs, PROPOSAL §2.5)."""
    if args.attn_tap_weight > 0:
        raise SystemExit(
            "--attn-tap-weight is the LUT escalation path's knob (it taps "
            "module outputs); the qlora path trains on the whole layer "
            "output against the resident teacher layer")
    if float(args.adam_eps) <= 0:
        raise SystemExit("--adam-eps must be positive")
    if float(args.lr_lut) <= 0 or float(args.lr_norm) <= 0:
        raise SystemExit("--lr-lut and --lr-norm must be positive")
    return types.SimpleNamespace(
        adam_eps=float(args.adam_eps),
        lut_path=str(getattr(args, "lut_path", "reference")),
        train_groups=_parse_train_groups(args.train),
        lr_lut=float(args.lr_lut),
        lr_norm=float(args.lr_norm),
        # -- the trainer's exposed surface, engine names --
        model=args.model,
        artifacts_dir=args.artifacts,
        capture_dir=args.capture,
        output_dir=args.out,
        rank_map=args.rank_map,
        warm_start=args.warm_start,
        layers=args.layers,
        input_mode=args.input_mode,
        lr=args.lr_lora,
        optimizer=args.opt_lora,
        max_steps=args.steps,
        warmup=args.warmup,
        eval_every=args.eval_every,
        patience=args.patience,
        batch_rows=args.rows_batch,
        holdout=args.holdout,
        seed=args.seed,
        clip=args.clip,
        cos_weight=args.cos_weight,
        attn_tap_weight=args.attn_tap_weight,
        train_target="qlora",
        # -- engine finetune defaults for the unexposed surface (parity) --
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        eval_batch_rows=8,
        eval_attn_budget_gib=4.0,
        holdout_split="random",
        target_cos=0.9995,
        target_rel_mse=5e-4,
        mse_weight=1.0,
        log_every=25,
        target_cache=True,
        o1_every=8,
        o1_probe_cmd=None,
        pilot_json=None,
        init_from=None,
        spectrum=args.spectrum,
        weights_alignment=None,
        propagate_kernel=False,
        tf32=True,
        muon_diagnostics=False,
        profile_eval=0,
        dump_step0_diff=False,
        hold_steps=0,
        warm_start_gauge="balanced",
        warm_start_skip="",
        legacy_hidden_state_dir=None,
        legacy_activations_dir=None,
        legacy_include_padding=False,
        cache_subdir="student_cache",
        cache_keep_every=8,
        scope="all",
        target_mode="captured",
        resume=bool(getattr(args, "resume", False)),
    )


def cmd_train(args):
    """trainer.py train: the capture-first layerwise adapter distillation
    (PROPOSAL §4 Stage 1 — parity with the engine's `finetune
    --train-target qlora`). Writes NO artifact files — the base artifacts
    dir stays byte-identical; the output is per-layer adapter snapshots +
    provenance + the assembled adapter dir."""
    ns = _to_engine_args(args)
    if ns.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = ns.device
    # resolve the mode-dependent defaults exactly like the engine's
    # cmd_finetune qlora branch (parity incl. the loud notes)
    if ns.optimizer is None:
        ns.optimizer = "muon"      # PROPOSAL §3.3: pilot-gated default
    if ns.lr is None:
        if ns.optimizer == "muon":
            ns.lr = 3e-3            # mid sweep point {1e-3,3e-3,1e-2,3e-2}
            print("  [trainer] NOTE: Muon lr is NOT transferable from "
                  "AdamW (different step calibration) — defaulting to "
                  "the sweep midpoint 3e-3; run the G1-O pilot "
                  "(L00+L03, 200 steps) and pass the selected --lr-lora",
                  flush=True)
        else:
            ns.lr = 1e-4
    if ns.optimizer == "adamw":
        print("  [trainer] AdamW selected (the G1-O fallback) — Muon "
              "is one --opt-lora flag away", flush=True)
    if ns.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if ns.batch_rows <= 0:
        raise SystemExit("--rows-batch must be positive")
    # ---- F6 startup assert: the stop conditions must be REACHABLE ------
    if ns.patience * ns.eval_every >= ns.max_steps:
        raise SystemExit(
            f"early stop is arithmetically unreachable: patience "
            f"({ns.patience}) x eval-every ({ns.eval_every}) = "
            f"{ns.patience * ns.eval_every} >= steps "
            f"({ns.max_steps}) — lower patience/eval-every or raise "
            f"steps (the first run ground 1000 steps with patience "
            f"30 x 50 = 1500)")
    # Reproducibility: lora_A's kaiming init consumes the global torch RNG —
    # seed it so --seed alone reproduces a run across processes (cmd_capture
    # does the same; caught by the replay checks).
    torch.manual_seed(ns.seed)
    if not ns.capture_dir:
        raise SystemExit(
            "trainer train requires --capture (the boundary-state store "
            "from the capture subcommand)")
    store = CaptureStore(capture_dir=ns.capture_dir,
                         legacy_hidden_dir=ns.legacy_hidden_state_dir,
                         legacy_activations_dir=ns.legacy_activations_dir,
                         legacy_include_padding=ns.legacy_include_padding)
    if ns.input_mode == "captured" and store.kind != "boundary":
        raise SystemExit(
            f"trainer train --input-mode captured requires a BOUNDARY "
            f"store (format 3: h_0..h_{store.num_layers - 1}); "
            f"{ns.capture_dir} is a {store.kind!r} store with no per-layer"
            f" input hidden states — re-run `capture.py capture` into a "
            f"fresh dir (or use --input-mode student-trajectory)")
    if not os.path.isdir(ns.model):
        raise SystemExit(
            "trainer train requires --model to be a LOCAL checkpoint "
            "directory (one teacher layer + the student's dense parts are "
            "streamed from its safetensors shards)")

    cfg = _load_pinned_text_config(ns.model)
    num_layers = int(cfg.num_hidden_layers)
    if num_layers != store.num_layers:
        raise SystemExit(
            f"layer count mismatch: config {num_layers} vs capture "
            f"{store.num_layers}")

    metadata = pmod.load_metadata(ns.artifacts_dir)
    rank_map = _load_rank_map(ns.rank_map, num_layers) \
        if ns.rank_map else None
    weights_modules = _load_weights_alignment(ns.weights_alignment) \
        if ns.weights_alignment else None
    # --spectrum: per-module predictions (G1a) + the per-layer stop target
    spectrum = _load_spectrum_predictions(ns.spectrum) \
        if ns.spectrum else None
    warm_start_fp = _file_sha256(ns.warm_start) if ns.warm_start else None
    init_from_fp = _dir_fingerprint(ns.init_from) if ns.init_from else None
    if ns.warm_start and ns.init_from:
        raise SystemExit(
            "--warm-start (Stage-R factors) and --init-from (a completed "
            "run's snapshots) are competing initializations — pass exactly "
            "one (Stage 1 uses the warm start; Stage 1.5 uses --init-from)")
    if rank_map is None:
        print(f"  [trainer] no --rank-map given — attaching uniform "
              f"default r={_QLORA_DEFAULT_RANK}/"
              f"alpha={_QLORA_DEFAULT_ALPHA} (kappa 0.25); recorded in the "
              f"config", flush=True)
    if ns.warm_start and rank_map is None:
        raise SystemExit(
            "--warm-start requires --rank-map (the spectrum's rank map — "
            "a uniform default has no matching warm-start geometry)")

    out_dir = ns.output_dir
    os.makedirs(out_dir, exist_ok=True)
    prov_path = os.path.join(out_dir, "finetune_provenance.json")
    if os.path.exists(os.path.join(out_dir, "qlora_adapters.pt")) and \
            not ns.resume:
        raise SystemExit(
            f"{out_dir}/qlora_adapters.pt already exists (a completed run) "
            f"— resume is the W2-T08 control-plane port; choose a fresh "
            f"--out")
    provenance = {}
    if os.path.exists(prov_path):
        if not ns.resume:
            raise SystemExit(
                f"{prov_path} already exists — resume is the W2-T08 "
                f"control-plane port; choose a fresh --out")
        with open(prov_path) as f:
            provenance = json.load(f)
    elif ns.resume:
        raise SystemExit(
            f"resume given but {prov_path} does not exist — nothing to "
            f"resume from; start a fresh run")

    # --- run identity (resume validation: mismatch is loud) ---------------
    capture_id = {"dir": os.path.abspath(ns.capture_dir),
                  "kind": store.kind, "rows": int(store.rows),
                  "seq_length": int(store.seq),
                  "hidden_size": int(store.hidden),
                  "num_layers": int(store.num_layers)}
    if provenance:
        recorded = provenance.get("capture", {})
        if recorded != capture_id:
            raise RuntimeError(
                f"resume provenance mismatch: capture was recorded as "
                f"{recorded!r} but this run reads {capture_id!r} — a "
                f"different capture is not the same run")
        checks = {
            "train_target": ("qlora", ns.train_target),
            "input_mode": (provenance.get("input_mode"), ns.input_mode),
            "artifacts_dir": (provenance.get("artifacts_dir"),
                              os.path.abspath(ns.artifacts_dir)),
            "rank_map_fingerprint": (
                provenance.get("rank_map_fingerprint"),
                _rank_map_fingerprint(rank_map)),
            "warm_start_fingerprint": (
                provenance.get("hyperparameters", {}).get(
                    "warm_start_fingerprint"), warm_start_fp),
            "init_from_fingerprint": (
                provenance.get("hyperparameters", {}).get(
                    "init_from_fingerprint"), init_from_fp),
            "holdout_split": (
                provenance.get("hyperparameters", {}).get("holdout_split"),
                ns.holdout_split),
            # W4-T08: the scope's trainable-set selection and the seed —
            # a different trainable set or seed is a different job
            "train_groups": (
                provenance.get("train_groups"), sorted(ns.train_groups)),
            "seed": (
                provenance.get("hyperparameters", {}).get("seed"),
                int(ns.seed)),
        }
        for name, (was, now) in checks.items():
            if was != now:
                raise RuntimeError(
                    f"resume provenance mismatch: {name} was recorded as "
                    f"{was!r} but this run uses {now!r} — re-run the "
                    f"affected layers in a fresh --out instead")
    provenance = {
        "train_target": "qlora",
        "train_groups": sorted(ns.train_groups),
        "input_mode": ns.input_mode,
        "artifacts_dir": os.path.abspath(ns.artifacts_dir),
        "model": ns.model,
        "capture": capture_id,
        "rank_map_fingerprint": _rank_map_fingerprint(rank_map),
        "rank_map": rank_map,
        "uniform_default": {"r": _QLORA_DEFAULT_RANK,
                            "alpha": _QLORA_DEFAULT_ALPHA,
                            "alpha_mode": "proportional"},
        "hyperparameters": {
            "lr": ns.lr, "optimizer": ns.optimizer,
            "max_steps": ns.max_steps, "batch_rows": ns.batch_rows,
            "eval_batch_rows": ns.eval_batch_rows,
            "holdout": ns.holdout,
            "holdout_split": ns.holdout_split,
            "target_cos": ns.target_cos,
            "target_rel_mse": ns.target_rel_mse,
            "patience": ns.patience, "eval_every": ns.eval_every,
            "mse_weight": ns.mse_weight, "cos_weight": ns.cos_weight,
            "clip": ns.clip, "warmup": ns.warmup,
            "target_cache": bool(ns.target_cache),
            "seed": ns.seed,
            "warm_start": ns.warm_start,
            "warm_start_fingerprint": warm_start_fp,
            "init_from": ns.init_from,
            "init_from_fingerprint": init_from_fp,
            "spectrum": ns.spectrum,
            "o1_every": int(ns.o1_every),
            "pilot": _load_pilot_json(ns.pilot_json)
            if ns.pilot_json else None,
        },
        "layers": provenance.get("layers", {}) if provenance else {},
    }

    print("=" * 78)
    print("Capture-first layerwise adapter distillation (qlora)")
    print("=" * 78)
    print(f"  model      : {ns.model} (streamed per layer)")
    print(f"  artifacts  : {ns.artifacts_dir} (READ-ONLY — never written)")
    print(f"  capture    : {store.kind} ({store.rows} rows x {store.seq} x "
          f"{store.hidden}, {store.num_layers} layers)")
    print(f"  output     : {out_dir} (adapter dir)")
    print(f"  input-mode : {ns.input_mode}  rank-map: "
          f"{ns.rank_map or 'none (uniform r=%d)' % _QLORA_DEFAULT_RANK}"
          f"  weights-alignment: "
          f"{ns.weights_alignment or 'none'}", flush=True)

    teacher_source = TeacherLayerSource(ns.model, device=device)
    wanted = _parse_layers_ordered(ns.layers, num_layers)
    if args.order == "in-order":
        # worst-first (the default) honors the --layers order exactly as
        # the engine does (the operator's spectrum-err_energy descending
        # schedule IS the given order); in-order forces ascending
        wanted = sorted(wanted)
    elif args.order == "worst-first" and spectrum:
        # W4-T05 (PROPOSAL §2.1 P1): with a spectrum, worst-first RANKS
        # the sweep itself — the layers by summed err_energy descending
        # (the largest error reservoirs first)
        wanted = _worst_first_order(spectrum, wanted, num_layers)
    # W4-T05: the executed order is part of the run manifest (the
    # provenance's identity — resume re-orders must match)
    provenance["order"] = list(wanted)
    xcache = None
    if ns.input_mode == "student-trajectory":
        xcache = XCache(os.path.join(out_dir, ns.cache_subdir),
                               store.rows, store.seq, store.hidden, num_layers,
                               keep_every=ns.cache_keep_every)
        xcache.seed_from_capture(store)
        # replay applies the adapters of every already-finalized layer:
        # THIS run's provenance first (re-trained layers update it as the
        # sweep progresses), then --init-from's Stage-1 snapshots for the
        # not-yet-retrained layers (the F4-correct trajectory source)
        snapshots = {int(k): os.path.join(out_dir, v["snapshot"])
                     for k, v in provenance["layers"].items()
                     if v.get("has_adapter") and v.get("done")}
        if ns.init_from:
            for k, rec in _iter_init_snapshots(ns.init_from).items():
                snapshots.setdefault(int(k), rec)
        _replay_to_qlora(xcache, wanted[0], cfg, teacher_source,
                                ns.artifacts_dir, metadata, rank_map,
                                snapshots, device, ns)

    t_start = time.time()
    summary = []
    n_done = 0
    for L in wanted:
        rec = provenance["layers"].get(str(L))
        if ns.resume and rec and rec.get("done"):
            snap = rec.get("snapshot")
            if rec.get("has_adapter") and \
                    not os.path.exists(os.path.join(out_dir, snap)):
                raise RuntimeError(
                    f"resume: layer {L} is marked done but its snapshot "
                    f"{snap} is missing — the output dir is incomplete; "
                    f"re-run the layer in a fresh --out")
            # W4-T08: the per-layer fingerprint — the recorded identity
            # vs this run's. A mismatch (or a record predating the
            # fingerprint contract) REFUSES the resume, loudly, naming
            # both identities.
            recorded_fp = rec.get("fingerprint")
            now_fp = _layer_fingerprint(
                capture_id, rank_map, L, ns.train_groups, warm_start_fp,
                init_from_fp, ns.seed)
            if recorded_fp is None:
                raise RuntimeError(
                    f"resume: layer {L}'s provenance record carries no "
                    f"fingerprint — it predates the W4-T08 identity "
                    f"contract and cannot be verified; re-run the layer "
                    f"in a fresh --out")
            if recorded_fp != now_fp:
                raise RuntimeError(
                    f"resume fingerprint mismatch: layer {L} was trained "
                    f"with {recorded_fp!r} but this resume is {now_fp!r} "
                    f"— a different job is not the same layer; re-run in "
                    f"a fresh --out")
            print(f"  [L{L:02d}] already trained (provenance, fingerprint "
                  f"verified) — skipping", flush=True)
            summary.append((L, rec.get("block_type"),
                            rec.get("before", {}).get("tok_cos"),
                            rec.get("after", {}).get("tok_cos")))
            continue
        if ns.input_mode == "student-trajectory" and not xcache.exists(L):
            _replay_to_qlora(xcache, L, cfg, teacher_source,
                                    ns.artifacts_dir, metadata, rank_map,
                                    snapshots, device, ns)
        predicted_layer = _layer_predicted_rel_mse(spectrum, L)
        res = _run_qlora_layer_job(
            L, cfg, teacher_source, store, ns, metadata, rank_map, device,
            xcache=xcache if ns.input_mode == "student-trajectory"
            else None,
            init_dir=ns.init_from,
            predicted_rel_mse=predicted_layer)
        layer_dir = os.path.join(out_dir, "qlora_layers", f"layer_{L}")
        os.makedirs(layer_dir, exist_ok=True)
        has_adapter = res["lora"] is not None
        if has_adapter:
            tmp_snap = os.path.join(layer_dir, "adapter.pt.tmp")
            # W4-T06: snap["joint"] carries the banked-best JOINT state
            # (luts + norms + lora) — the export's source; snap["lora"]
            # stays the adapter-dir contract (assembly/replay/init-readers
            # read it only).
            torch.save({"layer": L, "lora": res["lora"],
                        "joint": res["joint"],
                        "geometry": res["geometry"]}, tmp_snap)
            os.replace(tmp_snap, os.path.join(layer_dir, "adapter.pt"))
        metrics = {
            "layer": L, "block_type": res["block_type"],
            "train_target": "qlora", "input_mode": ns.input_mode,
            "dequant_path": res.get("dequant_path"),
            "steps": res["steps"], "lr": ns.lr,
            "optimizer": ns.optimizer,
            "stop_reason": res.get("stop_reason"),
            "tripwire": res.get("tripwire"),
            "best_step": res.get("best_step", 0),
            "init": res.get("init"),
            "before": res["before"], "after": res["after"],
            "n_train_rows": res["n_train_rows"],
            "n_holdout_rows": res["n_holdout_rows"],
            "eval_attn_cap": res.get("eval_attn_cap"),
            "ranks": {k: (v.get("r") if isinstance(v, dict) and "r" in v
                          else (v.get("components") if isinstance(v, dict)
                                else v))
                      for k, v in res["geometry"].items()},
            "weights_alignment": (
                {k: v for k, v in weights_modules.items()
                 if k.startswith(f"model.layers.{L}.")}
                if weights_modules else None),
            "spectrum_predictions": (
                {k: v for k, v in (spectrum or {}).items()
                 if k.startswith(f"model.layers.{L}.")}
                if spectrum else None),
            "predicted_layer_rel_mse": predicted_layer,
            # R2/R4/R5 (recovery campaign): the warm-start verdict
            # (init/before ratio + the §7 red flag), the per-eval timing
            # breakdowns (--profile-eval), and eval_share (G-T5's exact
            # measurement); step0_diff ALSO lands as its own report-back
            # file next to metrics.json when --dump-step0-diff is on.
            "warm_start_gate": res.get("warm_start_gate"),
            "eval_profiles": res.get("eval_profiles"),
            "eval_share": res.get("eval_share"),
            "phase_totals_s": res.get("phase_totals_s"),
            "eval_calls": res.get("eval_calls"),
            "wall_s": res["wall_s"],
            # W5.T1 (PROPOSAL §3 T7/T8): phase means (ms), watermarks
            # (running peak + the attn probe record), final reader_amp
            "phases": res.get("phases"),
            "watermarks": res.get("watermarks"),
            "reader_amp": res.get("reader_amp"),
        }
        _atomic_json_dump(metrics, os.path.join(layer_dir, "metrics.json"))
        if res.get("step0_diff") is not None:
            _atomic_json_dump(res["step0_diff"],
                              os.path.join(layer_dir, "step0_diff.json"))
        provenance["layers"][str(L)] = {
            "done": True, "has_adapter": has_adapter,
            "fingerprint": _layer_fingerprint(
                capture_id, rank_map, L, ns.train_groups, warm_start_fp,
                init_from_fp, ns.seed),
            "steps": res["steps"], "block_type": res["block_type"],
            "stop_reason": res.get("stop_reason"),
            "dequant_path": res.get("dequant_path"),
            "snapshot": os.path.join("qlora_layers", f"layer_{L}",
                                     "adapter.pt"),
            "metrics": os.path.join("qlora_layers", f"layer_{L}",
                                    "metrics.json"),
        }
        _atomic_json_dump(provenance, prov_path)
        print(f"        after : tok_cos={res['after']['tok_cos']:.6f} "
              f"flat_cos={res['after']['flat_cos']:.6f} "
              f"rel_mse={res['after']['rel_mse']:.4g}  steps={res['steps']}"
              f"  stop={res.get('stop_reason')}"
              f"{'  (all r=0 — no adapter)' if res['skipped_all_zero'] else ''}",
              flush=True)
        summary.append((L, res["block_type"], res["before"]["tok_cos"],
                        res["after"]["tok_cos"]))
        # ---- F6.3/F8: INCREMENTAL assembly + the O-1 probe hook ----------
        n_done += 1
        _assemble_qlora_adapters(out_dir, provenance, rank_map, ns,
                                        partial=True)
        if ns.o1_every > 0 and n_done % ns.o1_every == 0:
            _run_o1_probe(ns, out_dir, L, device, n_done)
        # RELEASE: the job frame already dropped the teacher, the student
        # and the shell (with the optimizer); free the stream cache too
        teacher_source.drop_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _assemble_qlora_adapters(out_dir, provenance, rank_map, ns)

    # ---- W4-T05 (PROPOSAL §2.1 P1): the final L=norm job — the model's
    # ---- final RMSNorm gain against the boundary store's final_hidden,
    # ---- after the whole layer sweep (its input is the student's last
    # ---- layer, snapshot applied)
    norm_metrics = _run_final_norm_job(
        ns, store, device, out_dir, provenance, cfg, teacher_source,
        metadata, rank_map)
    provenance["final_norm"] = {
        "done": True, "steps": norm_metrics["steps"],
        "stop_reason": norm_metrics["stop_reason"],
        "metrics": "final_norm/metrics.json",
        "norm_gain_edits": "norm_gain_edits.json",
    }
    _atomic_json_dump(provenance, prov_path)
    summary.append(("norm", "final_norm", norm_metrics["before"]["tok_cos"],
                    norm_metrics["after"]["tok_cos"]))

    print("\n" + "=" * 78)
    print("Layer summary (holdout per-token cosine of the layer output)")
    print("=" * 78)
    for L, bt, c0, c1 in summary:
        if c0 is None or c1 is None:
            continue
        # the final-norm job's row carries the 'norm' label (W4-T05)
        label = L if isinstance(L, str) else f"L{L:02d}"
        print(f"  {label:>4s} [{str(bt):16s}]  {c0:.6f} -> {c1:.6f}   "
              f"({'+' if c1 >= c0 else ''}{(c1 - c0) * 1e4:.1f}e-4)")
    print(f"\nAdapter dir (serve via eval_common.load_quant_model"
          f"(qlora_adapters=...)): {out_dir}")
    print(f"Provenance: {prov_path}")
    print(f"Total wall: {(time.time() - t_start) / 60:.1f} min")


# ---------------------------------------------------------------------------
# CLI (PROPOSAL §7.2's final form; this task implements `train` —
# pilot/export/report arrive with the W4/W5 wave tasks)
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="The layerwise distillation trainer (the engine's "
                    "ported home; adapter-only channel in W2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser(
        "pilot",
        help="the 2-config codes-LR decision (PROPOSAL §7.1)",
        description=(
            "trainer.py pilot (W4-T07, PROPOSAL §7.1): two configurations "
            "on --layers (default 0,3) — the base and the --alt override, "
            "both with the same --lr-lora/--lr-norm and control plane — "
            "run through the SAME per-layer jobs (one shared teacher + "
            "target cache per layer). The decision rule (the largest "
            "banked first-layer holdout rel_mse drop without a tripwire "
            "firing) and both arms' numbers land in "
            "<out>/pilot_joint.json."))
    pp.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="LOCAL checkpoint dir (the teacher layers are "
                         "streamed from its safetensors shards)")
    pp.add_argument("--artifacts", required=True,
                    help="original palettization output (metadata.json) — "
                         "READ-ONLY (never written)")
    pp.add_argument("--capture", default=None,
                    help="teacher capture dir — boundary store (format 3)")
    pp.add_argument("--rank-map", default=None,
                    help="rank_map.json of the distill_rank_alloc.py schema "
                         "(per-module adapter ranks; required with "
                         "--warm-start)")
    pp.add_argument("--warm-start", default=None,
                    help="warm_starts.pt (or its dir) from the spectrum "
                         "subcommand — the Stage-R analytic initialization")
    pp.add_argument("--layers", default="0,3",
                    help="the pilot layers (default '0,3' — the §7.1 "
                         "geometry; the decision is made on the FIRST "
                         "layer of the set)")
    pp.add_argument("--lr-lora", type=float, default=3e-4,
                    help="adapter learning rate for BOTH arms (the §7.1 "
                         "value 3e-4)")
    pp.add_argument("--lr-lut", type=float, default=3e-4,
                    help="LUT-codebook learning rate of the BASE arm (the "
                         "§7.1 value 3e-4; v1 default, pilot-confirmed)")
    pp.add_argument("--lr-norm", type=float, default=1e-4,
                    help="norm-gain learning rate for BOTH arms (the "
                         "§7.1 value 1e-4)")
    pp.add_argument("--alt", nargs=2, metavar=("KNOB", "VALUE"),
                    default=None,
                    help="the ALT arm's override — one knob-value pair, "
                         "e.g. --alt lr-lut 1e-4 (legal knobs: lr-lut, "
                         "lr-norm, lr-lora); everything else matches the "
                         "base arm")
    pp.add_argument("--steps", type=int, default=200,
                    help="max steps per layer per arm (§7.1: 200)")
    pp.add_argument("--eval-every", type=int, default=25,
                    help="holdout-eval interval in steps (§7.1: 25)")
    pp.add_argument("--patience", type=int, default=4,
                    help="early-stop after N evals without a banked "
                         "improvement (§7.1: 4)")
    pp.add_argument("--rows-batch", type=int, default=8,
                    help="packed rows per training step")
    pp.add_argument("--holdout", type=float, default=0.1,
                    help="held-out row fraction (the banking metric)")
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--opt-lora", default="adamw",
                    choices=["adamw", "muon"],
                    help="adapter optimizer for BOTH arms (the codes "
                         "channel is always AdamW; §7.1 passes the "
                         "AdamW-scale --lr-lora 3e-4)")
    pp.add_argument("--out", required=True,
                    help="the pilot output dir (pilot_joint.json)")
    pp.set_defaults(func=cmd_pilot)

    e = sub.add_parser(
        "export",
        help="build the deployment artifacts COPY from a completed joint "
             "run (PROPOSAL §2.8, P2)",
        description=(
            "trainer.py export (W4-T06, PROPOSAL §2.8/§7.3): per done "
            "layer, from the banked-best JOINT snapshot — freeze_lut "
            "(fp16 snap), the polish_lut_fp16 grid re-snap where the "
            "persisted calibration Gram exists, the canonical idx4/lut "
            "write into a COPY of the read-only artifacts dir, the "
            "bit-exact roundtrip, the trained norm gains into "
            "norm_gain_edits.json (the fold-aware (1+w')/(1+w) ratio "
            "convention), and the G-J3 reload check: rel_mse(snap) <= "
            "1.05 x rel_mse(train). The adapter dir is verified; the "
            "deployment merge hook (qlora_merge) is recorded."))
    e.add_argument("--run", required=True,
                   help="the train run's output dir (provenance + "
                        "per-layer joint snapshots + the adapter dir)")
    e.add_argument("--out", required=True,
                   help="the deployment artifacts dir (a copy of the "
                        "input artifacts with the trained LUTs, norm "
                        "edits and merged metadata)")
    e.add_argument("--merge-assign", default="gptvq",
                   choices=["lloyd", "gptq", "gptvq"],
                   help="the recorded merge hook's re-palettization "
                        "engine (PROPOSAL §2.8 step 3b; gptq/gptvq "
                        "require the persisted calibration Grams)")
    e.set_defaults(func=cmd_export)

    r = sub.add_parser(
        "report",
        help="the box run's evidence digest — the G-J verdicts from the "
             "run's own files (the intake protocol's tool)",
        description=(
            "trainer.py report: reads the run manifest "
            "(finetune_provenance.json), the per-layer banked lines "
            "(before/banked/after rel_mse), the export roundtrip lines "
            "(export_report.json), and — when given — the pilot "
            "decision (pilot_joint.json) and the eval JSONs (the O-1 "
            "probe report with the dense and base+adapters stages; the "
            "greedy report). Writes reports/joint_<tag>.json and "
            "prints the verdict table: G-J2 (the worst quartile of "
            "layers' banked improvement, >= 10%), G-J3 (the roundtrip "
            "ratios, <= 1.05), G-J4 (the paired gap, <= 0.04 nats/doc), "
            "G-J5 (greedy equivalence: mean first-divergence >= 25, "
            "exact >= 15%). Every verdict reads from files at report "
            "time (PROPOSAL 2.9; never a remembered number)."))
    r.add_argument("--run", required=True,
                   help="the trainer train output dir (the run "
                        "manifest + per-layer metrics + export report)")
    r.add_argument("--tag", default=None,
                   help="the report tag (default: the run dir's name) — "
                        "names reports/joint_<tag>.json")
    r.add_argument("--pilot", default=None,
                   help="pilot_joint.json — the pilot decision record, "
                        "embedded as provenance context")
    r.add_argument("--o1-report", default=None,
                   help="the O-1 probe's JSON report carrying the "
                             "dense and base+adapters per_doc vectors "
                             "(G-J4's producer)")
    r.add_argument("--greedy-report", default=None,
                   help="the greedy matcher's JSON report (G-J5's "
                        "producer)")
    r.add_argument("--out", default=None,
                   help="output JSON (default: reports/joint_<tag>.json)")
    r.set_defaults(func=cmd_report)

    t = sub.add_parser(
        "train",
        help="capture-first layerwise ADAPTER distillation (parity with "
             "the engine's finetune --train-target qlora)",
        description=(
            "trainer.py train (W2-T07): the capture-first layerwise "
            "adapter distillation — per layer, ONE teacher layer + ONE "
            "layer-scoped student layer resident, inputs = the captured "
            "boundary state h_i, targets computed on the fly by the "
            "resident teacher layer, adapters from --rank-map (B=0 "
            "identity init), the artifacts dir NEVER written; the output "
            "is the assembled adapter dir."))
    t.add_argument("--lut-path", default="reference",
                   choices=["reference", "kernel"],
                   help="the trainable-LUT forward/backward route: "
                        "'reference' (the default — the fp32 gather "
                        "path; the only CPU-legal route and the default "
                        "until the box G-B5 verdict) or 'kernel' (W5: "
                        "the FLUTE forward + lut_grad_scatter for "
                        "dL/dLUT; requires CUDA and both kernels — "
                        "the enumeration refuses unservable modules)")
    t.add_argument("--train", default="luts,norms,lora", metavar="GROUPS",
                   help="the joint trainable-set selection: a comma list "
                        "of luts, norms, lora (default: all three; the "
                        "empty selection and unknown tokens are refused; "
                        "W4-T01 enumerates the set and prints the step-0 "
                        "census — the joint optimizer arrives with "
                        "W4-T02)")
    t.add_argument("--model", default="Qwen/Qwen3.5-9B",
                   help="LOCAL checkpoint dir (required: one teacher "
                        "layer + the student's dense parts are streamed "
                        "from its safetensors shards)")
    t.add_argument("--artifacts", required=True,
                   help="original palettization output (metadata.json) — "
                        "READ-ONLY (never written)")
    t.add_argument("--capture", default=None,
                   help="teacher capture dir — boundary store (format 3) "
                        "from the capture subcommand")
    t.add_argument("--rank-map", default=None,
                   help="rank_map.json of the distill_rank_alloc.py schema "
                        "(per-module adapter ranks; r=0 leaves a module "
                        "unwrapped; alpha_i = r_i/4); absent => uniform "
                        "default r=64/alpha=16 (recorded)")
    t.add_argument("--spectrum", default=None,
                   help="functional_spectrum.json (the spectrum "
                        "subcommand's output) — per-module predictions "
                        "(the stop target) AND, with --order worst-first "
                        "(the default), the sweep ranking: layers by "
                        "summed err_energy descending (W4-T05)")
    t.add_argument("--warm-start", default=None,
                   help="warm_starts.pt (or its dir) from the spectrum "
                        "subcommand — the Stage-R analytic initialization")
    t.add_argument("--layers", default=None,
                   help="layer subset, e.g. '0-7,12' (default: all)")
    t.add_argument("--order", default="worst-first",
                   choices=["in-order", "worst-first"],
                   help="layer sweep order: 'worst-first' (the engine's "
                        "convention — the --layers order is honored as "
                        "given, the operator's spectrum err_energy "
                        "descending schedule) or 'in-order' (always "
                        "ascending)")
    t.add_argument("--input-mode", default="captured",
                   choices=["captured", "student-trajectory"],
                   help="input source: 'captured' (default) = the "
                        "boundary store's h_i; 'student-trajectory' = the "
                        "L2 escalation (rolling student hidden states, "
                        "targets still computed by the resident teacher "
                        "layer)")
    t.add_argument("--steps", type=int, default=600,
                   help="max steps per layer")
    t.add_argument("--warmup", type=int, default=50,
                   help="linear lr warmup steps, then cosine decay to "
                        "10%% of peak per layer")
    t.add_argument("--eval-every", type=int, default=25,
                   help="holdout-eval interval (steps)")
    t.add_argument("--patience", type=int, default=8,
                   help="early-stop after N evals without a banked "
                        "improvement (startup asserts patience x "
                        "eval-every < steps)")
    t.add_argument("--rows-batch", type=int, default=8,
                   help="packed rows per training step (x seq_length)")
    t.add_argument("--holdout", type=float, default=0.1,
                   help="held-out row fraction (the holdout is the "
                        "banking metric, never trained on)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--lr-lora", type=float, default=None,
                   help="adapter learning rate (default: 3e-3 for "
                        "--opt-lora muon — NOT transferable from AdamW, "
                        "the G1-O pilot selects; 1e-4 for adamw)")
    t.add_argument("--lr-lut", type=float, default=3e-4,
                   help="LUT-codebook channel learning rate (the codes "
                        "group is always AdamW; v1 value 3e-4, "
                        "pilot-confirmed per PROPOSAL §2.5/§7.2)")
    t.add_argument("--lr-norm", type=float, default=1e-4,
                   help="norm-gain channel learning rate (its own third "
                        "AdamW group; conservative v1 value 1e-4 per "
                        "PROPOSAL §2.5 — gains couple into every "
                        "downstream module)")
    t.add_argument("--opt-lora", default=None, choices=["adamw", "muon"],
                   help="adapter optimizer: 'muon' (default — vendored "
                        "NS5, pilot-gated) or 'adamw' (the fallback)")
    t.add_argument("--adam-eps", type=float, default=1e-8,
                   help="AdamW eps (LIVE since W2-T08: the ported "
                        "optimizer factory takes it; default 1e-8 = the "
                        "engine's pin, the R13 noise-floor value for the "
                        "real run is 1e-15)")
    t.add_argument("--clip", type=float, default=1.0,
                   help="grad-norm clip (0 disables)")
    t.add_argument("--cos-weight", type=float, default=0.05,
                   help="(1-cos) regularizer weight — rel_mse is the "
                        "trained objective")
    t.add_argument("--attn-tap-weight", type=float, default=0.0,
                   help="REFUSED on the qlora channel (the LUT "
                        "escalation path's knob — it taps module outputs; "
                        "this path trains on the whole layer output)")
    t.add_argument("--out", required=True,
                   help="the ADAPTER dir (qlora_adapters.pt + "
                        "qlora_config.json + per-layer snapshots)")
    t.add_argument("--resume", action="store_true",
                   help="resume the run in --out: the run identity "
                        "(capture, rank map, trainable set, seed, init) "
                        "is verified against the provenance and every "
                        "done layer is skipped bit-identically "
                        "(W4-T08, PROPOSAL §2.6); a mismatching "
                        "fingerprint refuses loudly")
    t.set_defaults(func=cmd_train)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
