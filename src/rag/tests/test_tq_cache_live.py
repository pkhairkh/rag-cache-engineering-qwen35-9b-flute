"""test_tq_cache_live.py — W2.3: the TQ cache wrapper against the REAL
transformers cache machinery.

W2.2 tested the stub route (`TQCache(layer_types=[...])`). This file builds
the production-side objects for real and verifies the wrapper integrates:

  * A REAL `Qwen3_5TextConfig` (the model class the vendored
    `src/scripts/modeling.py` implements — checked importable in
    transformers 5.19.0 as `transformers.Qwen3_5TextConfig`) with a tiny
    4-layer hybrid plan [linear_attention, full_attention, linear_attention,
    full_attention] and small head dims. `TQCache(config=...)` then runs the
    REAL `DynamicCache.__init__(config=...)` path — `get_layer_types_and_
    kwargs` + `DYNAMIC_LAYER_TYPE_MAPPING` — and `_wrap_linear_layers`
    swaps the built `LinearAttentionLayer`s for TQ layers while leaving the
    full-attention `DynamicLayer`s untouched. NO divergence from the W2.3
    brief's primary path: the config class is importable and accepts tiny
    shapes, so the documented fallback (layer_types-only + bare isinstance
    checks) was not needed — those isinstance assertions are included here
    anyway, as the brief asked.
  * The FULL inherited Cache API through the config-built cache:
    `update_conv_state` / `update_recurrent_state` / `has_previous_state`
    (the transformers dispatch — isinstance(layer,
    LinearAttentionCacheLayerMixin) — is what routes these calls to the TQ
    layer; exercised end-to-end, including the ValueError contracts).
  * A differential against an UNWRAPPED real `DynamicCache(config=same
    config)`: same generator seeds, same call sequence (prefill, S write,
    single-token decode) — identical shapes at every step, value agreement
    within the quant budget.
  * A 3-step decode loop in the exact call order `modeling.py`'s
    Qwen3_5GatedDeltaNet.forward makes on single-token steps (read
    conv_states[0] -> causal_conv1d_update territory -> update_conv_state
    -> update_recurrent_state -> read recurrent_states[0]).

Conv/S units stay the W2.2 stub sizes ((1, 32, 4) and (1, 8, 16) — 128
power-of-two dims each): the FHT single-block contract (PROPOSAL D2)
requires power-of-two units, and the live checks here are about the CACHE
machinery, not the model's production tensor shapes.

Determinism: fixed torch.Generator seeds; every rel-MSE number quoted in
comments was measured on this box with exactly those seeds. Gates: single
quantization round < 0.06 (the W2.2/W2.3 budget); the 3-step decode-loop
conv compounding (one extra round per step — PROPOSAL's recurrent-RMW
compounding, the Phase-2/D4 decision, NOT a single-round contract) is
gated at 0.10 = ~3 compounded rounds.

NOTE (compounding, documented not pinned): at test scale d=128 the 3-step
compounded window rel-MSE reaches 0.047-0.081 over random seeds (seed 322
measures 0.081) — a fat-tail artifact of the small d (single-round p99 is
already ~0.05); the loop below pins seed 321 (measured 0.047) at the
documented 3-round gate. Model-level compounding through the real gated-
delta-net recurrence is W4's differential job, not the cache contract's.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import (
    DynamicCache,
    DynamicLayer,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
    get_layer_types_and_kwargs,
)

from tq_cache import TQCache, TQLinearAttentionLayer

CONV_KERNEL = 4                       # the linear-attention conv kernel (Qwen3.5: 4)
CONV_UNIT = (1, 32, 4)                # 128 dims — the stored window shape
S_UNIT = (1, 8, 16)                   # 128 dims — the recurrent state shape
M_UNIT = (1, 128)                     # 128 dims — M1/M2 global memories

REL_MSE_GATE = 0.06                   # single quantization round
COMPOUNDED_GATE = 0.10                # ~3 compounded rounds (decode loop)

LAYER_PLAN = ["linear_attention", "full_attention",
              "linear_attention", "full_attention"]


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-30).item()


def _tiny_config() -> Qwen3_5TextConfig:
    """A REAL Qwen3.5 text config, shrunk to a tiny 4-layer hybrid plan."""
    return Qwen3_5TextConfig(
        num_hidden_layers=4,
        layer_types=list(LAYER_PLAN),
    )


# --- the config route: real machinery builds, the wrapper wraps -------------- #
def test_real_config_builds_and_wraps_layers():
    cfg = _tiny_config()

    # the REAL transformers machinery resolves the layer plan from the config
    # (this is exactly the path DynamicCache.__init__(config=...) takes)
    layer_types, per_layer_kwargs = get_layer_types_and_kwargs(
        cfg.get_text_config(decoder=True))
    assert layer_types == LAYER_PLAN
    # transformers API changed: per_layer_kwargs is now a dict shared across layers
    assert per_layer_kwargs == {"number_of_states": 1}

    cache = TQCache(config=cfg)
    assert isinstance(cache, DynamicCache)             # a real transformers Cache
    assert len(cache.layers) == 4

    # linear layers wrapped, full-attention layers untouched
    for i in (0, 2):
        layer = cache.layers[i]
        assert isinstance(layer, TQLinearAttentionLayer)
        # the dispatch relationships the Cache machinery relies on
        assert isinstance(layer, LinearAttentionLayer)
        assert isinstance(layer, LinearAttentionCacheLayerMixin)
        assert layer.number_of_states == 1
        assert layer.record_past is False
        assert layer.online is True and layer.bits == 3.5
    for i in (1, 3):
        assert isinstance(cache.layers[i], DynamicLayer)
        assert not isinstance(cache.layers[i], LinearAttentionCacheLayerMixin)

    assert cache.linear_layer_indices() == [0, 2]


# --- the full inherited Cache API on the config-built cache ------------------- #
def test_config_route_full_cache_api_roundtrip():
    cfg = _tiny_config()
    cache = TQCache(config=cfg)
    layer = cache.layers[0]

    # the inherited has_previous_state routes through the TQ layer's dicts
    assert cache.has_previous_state(0) is False

    g = torch.Generator().manual_seed(123)
    x = torch.randn(1, 32, 7, generator=g, dtype=torch.float16)   # prefill
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16)

    # cache-level dispatch (Cache.update_conv_state isinstance-checks the
    # layer, then delegates) — the exact calls modeling.py makes
    full = cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    out = cache.update_recurrent_state(s, 0)
    assert tuple(full.shape) == (1, 32, 7)
    assert torch.equal(full, x)                       # prefill return is exact
    assert tuple(out.shape) == S_UNIT

    assert cache.has_previous_state(0) is True

    # reads: dequantized transients, within the quant budget
    w = layer.conv_states[0]
    assert tuple(w.shape) == CONV_UNIT and w.dtype == torch.float16
    assert _rel_mse(w, x[..., -CONV_KERNEL:]) < REL_MSE_GATE    # 0.018
    r = layer.recurrent_states[0]
    assert tuple(r.shape) == S_UNIT
    assert _rel_mse(r, s) < REL_MSE_GATE                         # 0.021

    # the no-fp16 invariant + uint8 codes hold on the config route too
    assert dict.get(layer.conv_states, 0) is None
    assert dict.get(layer.recurrent_states, 0) is None
    assert layer.s_codes.idx_lo.dtype == np.uint8

    # the inherited transformers error contracts
    with pytest.raises(ValueError):
        cache.has_previous_state(1)                   # a full-attention layer
    with pytest.raises(ValueError):
        cache.update_conv_state(                       # a non-linear-attention layer
            torch.zeros(1, 32, 1, dtype=torch.float16), 1)
    with pytest.raises(ValueError):
        cache.update_recurrent_state(torch.zeros(*S_UNIT), 1)

    # M1/M2 are cache-level state on the config route as well
    m1 = torch.randn(*M_UNIT, generator=g, dtype=torch.float16)
    m2 = torch.randn(*M_UNIT, generator=g, dtype=torch.float16)
    assert _rel_mse(cache.update_m1(m1), m1) < REL_MSE_GATE     # 0.020
    assert _rel_mse(cache.update_m2(m2), m2) < REL_MSE_GATE     # 0.018
    assert cache.m1_codes.seed == 303 and cache.m2_codes.seed == 404


def test_second_linear_layer_is_isolated():
    cfg = _tiny_config()
    cache = TQCache(config=cfg)
    g = torch.Generator().manual_seed(126)
    x = torch.randn(*CONV_UNIT, generator=g, dtype=torch.float16)
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16)

    # write through the cache-level API at the SECOND linear layer
    cache.update_conv_state(x, 2, conv_kernel_size=CONV_KERNEL)
    cache.update_recurrent_state(s, 2)

    # layer 0 was wrapped too but holds nothing; no state leaked across
    assert cache.layers[0].conv_states[0] is None
    assert cache.layers[0].recurrent_states[0] is None
    assert cache.has_previous_state(0) is False
    assert cache.has_previous_state(2) is True

    assert cache.s_codes == {0: None, 2: cache.layers[2].s_codes}
    assert cache.conv_codes[2] is not None
    assert cache.conv_codes[2].seed == 202 and cache.s_codes[2].seed == 101

    r = cache.layers[2].recurrent_states[0]
    assert _rel_mse(r, s) < REL_MSE_GATE                         # 0.023
    w = cache.layers[2].conv_states[0]
    assert _rel_mse(w, x) < REL_MSE_GATE                         # 0.022


# --- differential vs the UNWRAPPED real DynamicCache(config=...) -------------- #
@pytest.mark.parametrize("seed", [312, 314])
def test_live_differential_vs_real_dynamic_cache(seed):
    cfg = _tiny_config()
    g = torch.Generator().manual_seed(seed)
    prefill = torch.randn(1, 32, 7, generator=g, dtype=torch.float16)
    token = torch.randn(1, 32, 1, generator=g, dtype=torch.float16)
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16)

    real = DynamicCache(config=cfg)    # the REAL machinery, unwrapped
    tqc = TQCache(config=cfg)          # the wrapped cache

    # prefill: identical shape (B, D, T), exact values
    f_real = real.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)
    f_tq = tqc.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(f_tq.shape) == (1, 32, 7) == tuple(f_real.shape)
    assert torch.equal(f_tq, f_real)

    # S write: identical shape, within one quant round
    s_real = real.update_recurrent_state(s, 0)
    s_tq = tqc.update_recurrent_state(s, 0)
    assert tuple(s_tq.shape) == S_UNIT == tuple(s_real.shape)
    assert _rel_mse(s_tq, s_real) < REL_MSE_GATE                 # 0.020

    # single-token decode: identical shape (B, D, k+1)
    d_real = real.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    d_tq = tqc.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(d_tq.shape) == (1, 32, 5) == tuple(d_real.shape)
    assert _rel_mse(d_tq, d_real) < REL_MSE_GATE                 # 0.014-0.015

    # stored windows: identical shape (B, D, k), two quant rounds in TQ
    w_real = real.layers[0].conv_states[0]
    w_tq = tqc.layers[0].conv_states[0]
    assert tuple(w_tq.shape) == CONV_UNIT == tuple(w_real.shape)
    assert _rel_mse(w_tq, w_real) < REL_MSE_GATE                 # 0.027-0.034

    # the real layer keeps exact windowing; TQ's window is the last k of
    # its own decode return within one round
    assert torch.equal(w_real, d_real[..., -CONV_KERNEL:])
    assert _rel_mse(w_tq, d_tq[..., -CONV_KERNEL:]) < REL_MSE_GATE


# --- a 3-step decode loop in the modeling.py call order ----------------------- #
def test_config_cache_decode_loop():
    cfg = _tiny_config()
    g = torch.Generator().manual_seed(321)
    prefill = torch.randn(1, 32, 7, generator=g, dtype=torch.float16)
    real = DynamicCache(config=cfg)
    tqc = TQCache(config=cfg)
    real.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)
    tqc.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)

    for step in range(3):
        token = torch.randn(1, 32, 1, generator=g, dtype=torch.float16)
        s_new = torch.randn(*S_UNIT, generator=g, dtype=torch.float16)

        # the single-token step of modeling.py::Qwen3_5GatedDeltaNet.forward:
        # read the conv state (handed to causal_conv1d_update), then the two
        # writes, then the recurrent read
        _ = real.layers[0].conv_states[0]
        _ = tqc.layers[0].conv_states[0]
        d_real = real.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
        d_tq = tqc.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
        s_real = real.update_recurrent_state(s_new, 0)
        s_tq = tqc.update_recurrent_state(s_new, 0)

        # shapes stable at every step, both caches
        assert tuple(d_tq.shape) == (1, 32, 5) == tuple(d_real.shape)
        assert tuple(s_tq.shape) == S_UNIT == tuple(s_real.shape)
        w_real = real.layers[0].conv_states[0]
        w_tq = tqc.layers[0].conv_states[0]
        assert tuple(w_tq.shape) == CONV_UNIT == tuple(w_real.shape)

        # S is replaced fresh each step: single-round fidelity
        assert _rel_mse(s_tq, s_real) < REL_MSE_GATE             # worst 0.045

        # the conv path COMPOUNDS one quantization round per decode step
        # (see the module docstring note): 3 steps -> ~3 rounds -> the
        # documented compounded gate, not the single-round budget
        assert _rel_mse(d_tq, d_real) < COMPOUNDED_GATE          # worst 0.028
        assert _rel_mse(w_tq, w_real) < COMPOUNDED_GATE          # worst 0.047

    # after the whole loop: codes kept, the dict still never held fp16
    assert dict.get(tqc.layers[0].conv_states, 0) is None
    assert dict.get(tqc.layers[0].recurrent_states, 0) is None
    assert tqc.layers[0].conv_codes is not None
    assert tqc.layers[0].s_codes is not None
    assert tqc.has_previous_state(0) is True
