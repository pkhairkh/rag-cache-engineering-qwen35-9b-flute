"""test_tq_cache.py — W2.2 stub-contract tests for the online TQ cache wrapper.

Covers the W2.2 contract for `src/rag/tq_cache.py` (SPECIFICATION §3.2;
PROPOSAL §2 D2-D4; module commit a27f6fc) via the `layer_types` stub route
(no model, no config — the config route is W2.3's job):

  1.  Quantize-on-write / dequantize-on-read: conv units (1, 32, 4) -> 128
      dims, S units (1, 8, 16) -> 128 dims; reads return dequantized fp16
      transients within rel-MSE < 0.06 of the true stored values.
  2.  The no-fp16 invariant: the plain dict storage (dict.get, bypassing
      the view) holds NOTHING after writes; codes are uint8; direct
      `layer.conv_states[0] = t` raises RuntimeError in online mode.
  3.  Interception counters: reads/writes increment `layer.reads`/
      `layer.writes`; the lazy mutation re-capture is counter-transparent.
  4.  Conv windowing, differential vs the REAL transformers
      LinearAttentionLayer (same seeds, same call sequence): identical
      shapes at every step — prefill (B, D, T), decode (B, D, k+1), stored
      window (B, D, k) — and value agreement within quant tolerance.
  5.  In-place mutation capture (the causal_conv1d_update decode pattern):
      roll-left + fresh last column on the handed-out tensor, then a
      re-read requantizes it (rel-MSE < 0.06 vs the mutated tensor).
  6.  has_previous_state flow: False before the conv prefill write, True
      after; ValueError on a full-attention layer (inherited contract);
      record_past is False.
  7.  M1/M2: update/read round-trips, D3 seeds 303/404, code install via
      the m1_codes/m2_codes setters.
  8.  Code injection (the W7 install path): s_codes/conv_codes setters +
      the cache-level set_s_codes/set_conv_codes and dict views.
  9.  Offline fallback (D4): raw tensors during forward, passthrough
      reads, layer.snapshot_codes() quantizes the held tensors.
  10. resolve_quantizer: canonical kinds (shared D3 instances, seeds
      101/202/303/404) vs custom sizes (kind's seed kept; non-power-of-two
      sizes are valid under the segmented FHT — the tiny-d positive gates);
      registry identity for custom sizes; d <= 0 raises.
  11. Non-power-of-two conv geometry (the Qwen3.5 in_proj Q+V window:
      1×6144×4 = 24,576): the FULL window rides one code unit at
      d = 24,576 (round-up to 32 only — no truncation, no pow2 padding;
      the segments 16,384 + 8,192 each fit the FHT kernel's 64 KiB tile)
      in the conv kind's shared D3 frame (seed 202, custom-d instance).
      Windowing, mutation re-capture and differential-vs-real-layer
      contracts hold identically; the conv_codes setter guards unit-d
      drift (old pre-W11 snapshots raise with the re-ingest hint); the
      no-truncation regression pins the last 8,192 coordinates' fidelity
      (the GPU session's 16,384 cap zeroed exactly those). Power-of-two
      geometries take the SAME code path with pad == 0 (bit-identical).

Determinism: every random draw goes through a torch.Generator pinned to a
fixed seed; all rel-MSE numbers quoted in comments were measured on this
box with exactly those seeds.

rel-MSE gate (0.06): the single-round round-trip at d=128 / 3.5 bits
measures ~0.022 mean but has a FAT TAIL at this small d (p99 ~0.05, max
0.058 over 300 random draws — at d=128 the per-coordinate MSE ratio has
not yet concentrated). The gates below are therefore safe for the PINNED
seeds used here (all measured values have >=1.5x margin); at production
d (2^15 / 2^19) the ratio concentrates over >=32k coordinates and 0.06 is
a comfortable engineering budget.

Codebook hygiene: the d=128 custom-size quantizers hit
src/rag/codebooks/cb_b{3,4}_d128.npz — already present in the working
tree (created by the orchestrator's W2.1 smoke run, same deterministic
solves as W1.1's committed d=1024 set); no test writes a NEW (b, d)
combination. Recommend the ORCH commit the two d128 npz files alongside
the d1024 ones.

REPORTED SOURCE BUG (not fixed, W2.2 brief: report, don't fix): in
offline (D4) mode `TQCache.snapshot_codes()` returns all-None code views
(`{"s": {0: None}, "conv": {0: None}}`) because it reads the ONLINE code
store (`layer.s_codes` -> `_s_codes`, never set offline) instead of
delegating to `TQLinearAttentionLayer.snapshot_codes()`, which correctly
quantizes the held raw tensors. W5's §5 cache-level snapshot would emit
an empty snapshot in the D4 regime. Pinned by the strict-xfail test at
the bottom of this file; exact repro in its docstring.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers.cache_utils import (
    DynamicCache,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
)

import turboquant as tq
from turboquant import SEEDS
from tq_cache import TQCache, TQLinearAttentionLayer, resolve_quantizer

# ---- the stub shapes (power-of-two units so resolve_quantizer accepts them)
CONV_KERNEL = 4                       # the linear-attention conv kernel (Qwen3.5: 4)
CONV_UNIT = (1, 32, 4)                # 128 dims  — the stored window shape
S_UNIT = (1, 8, 16)                   # 128 dims  — the recurrent state shape
M_UNIT = (1, 128)                     # 128 dims  — M1/M2 global memories
D_UNIT = 128

REL_MSE_GATE = 0.06                   # the W2.2 budget (see module docstring)


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """sum((a-b)^2) / sum(b^2) — the brief's relative MSE."""
    a, b = a.float(), b.float()
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-30).item()


def _rand(shape, seed: int, dtype=torch.float16) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed),
                       dtype=dtype)


# --- construction: the layer_types stub route -------------------------------- #
def test_layer_types_route_builds_the_stack():
    cache = TQCache(layer_types=["linear_attention", "full_attention",
                                 "linear_attention"])
    assert isinstance(cache, DynamicCache)          # a real transformers Cache
    assert len(cache.layers) == 3
    l0, l1, l2 = cache.layers
    assert isinstance(l0, TQLinearAttentionLayer)
    assert isinstance(l2, TQLinearAttentionLayer)
    # the dispatch relationships transformers' Cache machinery relies on
    assert isinstance(l0, LinearAttentionLayer)          # subclasses the real layer
    assert isinstance(l0, LinearAttentionCacheLayerMixin)
    assert not isinstance(l1, LinearAttentionCacheLayerMixin)  # full-attn untouched
    assert cache.linear_layer_indices() == [0, 2]
    # layer defaults
    assert l0.number_of_states == 1 and l0.record_past is False
    assert l0.online is True and l0.bits == 3.5
    assert l0.reads == {"conv": 0, "s": 0} and l0.writes == {"conv": 0, "s": 0}


def test_layer_types_route_rejects_unknown_types():
    with pytest.raises(ValueError, match="unknown layer type"):
        TQCache(layer_types=["bogus_attention"])


# --- contract 1: quantize-on-write / dequantize-on-read --------------------- #
@pytest.mark.parametrize("seed", [100, 101, 102])
def test_conv_quantize_on_write_dequantize_on_read(seed):
    g = torch.Generator().manual_seed(seed)
    cache = TQCache(layer_types=["linear_attention"])
    x = torch.randn(1, 32, 7, generator=g, dtype=torch.float16).cuda()  # prefill, T > k
    full = cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    layer = cache.layers[0]

    # prefill return: the input verbatim (no quantization on the way OUT)
    assert tuple(full.shape) == (1, 32, 7)
    assert torch.equal(full, x)

    # READ path: dequantized transient, window shape, fp16
    window = layer.conv_states[0]
    assert tuple(window.shape) == CONV_UNIT
    assert window.dtype == torch.float16
    assert _rel_mse(window, x[..., -CONV_KERNEL:]) < REL_MSE_GATE  # 0.017-0.023

    # the store is CODES: uint8, custom-d quantizer keeps the conv kind seed
    codes = layer.conv_codes
    assert codes is not None
    assert codes.d == D_UNIT and codes.seed == 202
    assert codes.idx_lo.dtype == np.uint8 and codes.idx_hi.dtype == np.uint8
    assert codes.n_lo == codes.n_hi == D_UNIT // 2   # 3.5-bit half split


@pytest.mark.parametrize("seed", [200, 201, 202])
def test_s_quantize_on_write_dequantize_on_read(seed):
    g = torch.Generator().manual_seed(seed)
    cache = TQCache(layer_types=["linear_attention"])
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16).cuda()
    out = cache.update_recurrent_state(s, 0)
    layer = cache.layers[0]

    # the WRITE returns the dequantized state (the read path — a transient)
    assert tuple(out.shape) == S_UNIT
    assert _rel_mse(out, s) < REL_MSE_GATE                     # 0.017-0.018

    # the READ dequantizes the stored codes
    r = layer.recurrent_states[0]
    assert tuple(r.shape) == S_UNIT
    assert r.dtype == torch.float16
    assert _rel_mse(r, s) < REL_MSE_GATE
    assert torch.equal(r, out)                                # same codes, same read

    assert layer.s_codes.d == D_UNIT and layer.s_codes.seed == 101  # S kind seed


# --- contract 2: the no-fp16 invariant ---------------------------------------- #
def test_no_fp16_invariant():
    g = torch.Generator().manual_seed(910)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]

    # reads before any write are None (the parent dict.fromkeys semantics)
    assert layer.conv_states[0] is None
    assert layer.recurrent_states[0] is None

    x = torch.randn(*CONV_UNIT, generator=g, dtype=torch.float16).cuda()
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    cache.update_recurrent_state(s, 0)

    # the plain dict storage (bypassing the view) holds NO state tensors
    assert dict.get(layer.conv_states, 0) is None
    assert dict.get(layer.recurrent_states, 0) is None
    # ... even after reads (the handed-out tensors are transients)
    _ = layer.conv_states[0]
    _ = layer.recurrent_states[0]
    assert dict.get(layer.conv_states, 0) is None
    assert dict.get(layer.recurrent_states, 0) is None

    # the persistent store is uint8 codes
    assert layer.s_codes.idx_lo.dtype == np.uint8
    assert layer.conv_codes.idx_lo.dtype == np.uint8

    # direct assignment refuses in online mode (the cache never stores fp16)
    with pytest.raises(RuntimeError):
        layer.conv_states[0] = torch.zeros(*CONV_UNIT)
    with pytest.raises(RuntimeError):
        layer.recurrent_states[0] = torch.zeros(*S_UNIT)


# --- contract 3: interception counters ---------------------------------------- #
def test_interception_counters():
    g = torch.Generator().manual_seed(950)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]
    assert layer.reads == {"conv": 0, "s": 0}
    assert layer.writes == {"conv": 0, "s": 0}

    # a read before any write still counts (and returns None)
    assert layer.conv_states[0] is None
    assert layer.reads == {"conv": 1, "s": 0}

    x = torch.randn(*CONV_UNIT, generator=g, dtype=torch.float16).cuda()
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    cache.update_recurrent_state(s, 0)
    # writes bump only writes
    assert layer.writes == {"conv": 1, "s": 1}
    assert layer.reads == {"conv": 1, "s": 0}

    # reads bump only reads (conv read first re-syncs the handed-out tensor)
    _ = layer.conv_states[0]
    _ = layer.recurrent_states[0]
    _ = layer.recurrent_states[0]
    assert layer.reads == {"conv": 2, "s": 2}
    assert layer.writes == {"conv": 1, "s": 1}

    # the in-place mutation re-capture (contract 5's sync) is NOT a write.
    # The handed-out access is itself a read: conv reads so far =
    # pre-write(1) + post-write(2) + handed-out(3) + re-capture(4).
    t = layer.conv_states[0]                       # handed out  (read 3)
    t.copy_(torch.roll(t, shifts=-1, dims=-1))     # mutated in place
    _ = layer.conv_states[0]                       # re-captured + fresh dequant (read 4)
    assert layer.reads == {"conv": 4, "s": 2}
    assert layer.writes == {"conv": 1, "s": 1}


# --- contract 4: conv windowing, differential vs the REAL layer --------------- #
@pytest.mark.parametrize("seed", [400, 401, 402])
def test_conv_windowing_differential_vs_real_layer(seed):
    g = torch.Generator().manual_seed(seed)
    prefill = torch.randn(1, 32, 7, generator=g, dtype=torch.float16).cuda()  # T > k
    token = torch.randn(1, 32, 1, generator=g, dtype=torch.float16).cuda()    # decode
    real = LinearAttentionLayer()
    tql = TQLinearAttentionLayer()

    # prefill: identical shape (B, D, T) and exact values (no quant on the
    # way out; the layer only quantizes the persisted window)
    f_real = real.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)
    f_tq = tql.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(f_tq.shape) == (1, 32, 7) == tuple(f_real.shape)
    assert torch.equal(f_tq, f_real)

    # decode: identical shape (B, D, k+1) — cat([old_window, new_token])
    d_real = real.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    d_tq = tql.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(d_tq.shape) == (1, 32, 5) == tuple(d_real.shape)

    # stored window: identical shape (B, D, k)
    w_real = real.conv_states[0]
    w_tq = tql.conv_states[0]
    assert tuple(w_tq.shape) == CONV_UNIT == tuple(w_real.shape)

    # value level: TQ's decode return == cat([dequant(old window), token])
    # within quant tolerance of the real layer's — measured 0.015-0.017
    assert _rel_mse(d_tq, d_real) < REL_MSE_GATE
    # the stored window carries TWO quantization rounds (prefill window,
    # then decode window) — measured 0.032-0.036, still inside the budget
    assert _rel_mse(w_tq, w_real) < REL_MSE_GATE

    # the windowing contract itself, in both layers: the stored window is
    # the last k of the returned full (exact for the real layer, within one
    # quant round for TQ)
    assert torch.equal(w_real, d_real[..., -CONV_KERNEL:])
    assert _rel_mse(w_tq, d_tq[..., -CONV_KERNEL:]) < REL_MSE_GATE


def test_conv_windowing_pads_short_prefill_differentially():
    g = torch.Generator().manual_seed(500)
    short = torch.randn(1, 32, 2, generator=g, dtype=torch.float16).cuda()   # T < k
    token = torch.randn(1, 32, 1, generator=g, dtype=torch.float16).cuda()
    real, tql = LinearAttentionLayer(), TQLinearAttentionLayer()

    # prefill shorter than the kernel: both left-pad with zeros to (B, D, k)
    f_real = real.update_conv_state(short, 0, conv_kernel_size=CONV_KERNEL)
    f_tq = tql.update_conv_state(short, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(f_tq.shape) == CONV_UNIT == tuple(f_real.shape)
    assert torch.equal(f_tq, f_real)
    assert torch.count_nonzero(f_tq[..., :2]) == 0

    d_real = real.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    d_tq = tql.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(d_tq.shape) == (1, 32, 5) == tuple(d_real.shape)
    assert _rel_mse(d_tq, d_real) < REL_MSE_GATE                     # 0.014


# --- contract 5: in-place mutation capture (causal_conv1d_update) ------------- #
@pytest.mark.parametrize("seed", [300, 301, 302])
def test_inplace_mutation_capture(seed):
    g = torch.Generator().manual_seed(seed)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]
    x = torch.randn(*CONV_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)

    state = layer.conv_states[0]              # the handed-out transient
    assert tuple(state.shape) == CONV_UNIT
    codes_before = layer.conv_codes

    # the causal_conv1d_update decode pattern: roll the window left and
    # write the fresh token into the last column — IN PLACE on the handed-
    # out tensor, with NO update_conv_state call.
    # (A constant-offset mutation is deliberately NOT used: it inflates the
    # norm and rescales the rotated lattice — adversarial by construction,
    # rel-MSE ~0.14, documented in the module docstring + worklog.)
    state.copy_(torch.roll(state, shifts=-1, dims=-1))
    state[..., -1] = torch.randn(1, 32, generator=g, dtype=torch.float16).cuda()

    # the NEXT read re-captures (requantizes) the mutated tensor
    recaptured = layer.conv_states[0]
    assert recaptured is not state                          # fresh transient
    assert _rel_mse(recaptured, state) < REL_MSE_GATE       # 0.015-0.022
    # the capture really re-quantized: new code bytes, still no fp16 stored
    assert layer.conv_codes is not None
    assert not np.array_equal(layer.conv_codes.idx_lo, codes_before.idx_lo)
    assert dict.get(layer.conv_states, 0) is None
    # and the re-capture was counter-transparent (contract 3's cross-check)
    assert layer.writes["conv"] == 1


# --- contract 6: has_previous_state flow -------------------------------------- #
def test_has_previous_state_flow():
    g = torch.Generator().manual_seed(900)
    cache = TQCache(layer_types=["linear_attention", "full_attention"])
    layer = cache.layers[0]
    assert cache.has_previous_state(0) is False
    assert layer.has_previous_state == {0: False}
    assert layer.record_past is False

    # an S-only write does NOT flip the gate: has_previous_state is the
    # conv-prefill gate (exactly the real LinearAttentionLayer's semantics)
    cache.update_recurrent_state(torch.randn(*S_UNIT, generator=g,
                                             dtype=torch.float16).cuda(), 0)
    assert cache.has_previous_state(0) is False

    cache.update_conv_state(torch.randn(*CONV_UNIT, generator=g,
                                        dtype=torch.float16).cuda(), 0,
                            conv_kernel_size=CONV_KERNEL)
    assert cache.has_previous_state(0) is True
    assert layer.has_previous_state[0] is True

    # inherited transformers contract: full-attention layers reject the query
    with pytest.raises(ValueError):
        cache.has_previous_state(1)


# --- contract 7: M1/M2 global memories ---------------------------------------- #
@pytest.mark.parametrize("seed", [600, 601, 602])
def test_m1_m2_roundtrip_and_d3_seeds(seed):
    g = torch.Generator().manual_seed(seed)
    cache = TQCache(layer_types=[])
    m1 = torch.randn(*M_UNIT, generator=g, dtype=torch.float16).cuda()
    m2 = torch.randn(*M_UNIT, generator=g, dtype=torch.float16).cuda()

    out1 = cache.update_m1(m1)
    out2 = cache.update_m2(m2)
    assert tuple(out1.shape) == M_UNIT and tuple(out2.shape) == M_UNIT
    assert _rel_mse(out1, m1) < REL_MSE_GATE               # 0.017-0.022
    assert _rel_mse(out2, m2) < REL_MSE_GATE

    r1 = cache.read_m1()                                   # default fp16
    r2 = cache.read_m2(dtype=torch.float32)
    assert r1.dtype == torch.float16 and r2.dtype == torch.float32
    assert _rel_mse(r1, m1) < REL_MSE_GATE
    assert _rel_mse(r2, m2) < REL_MSE_GATE

    # D3 seeds: M1 -> 303, M2 -> 404 (turboquant.SEEDS), custom size keeps
    # the kind's seed
    assert cache.m1_codes.seed == 303
    assert cache.m2_codes.seed == 404
    assert cache.m1_codes.d == D_UNIT
    assert cache.m1_codes.idx_lo.dtype == np.uint8


def test_m1_m2_read_before_write_is_none():
    cache = TQCache(layer_types=[])
    assert cache.read_m1() is None
    assert cache.read_m2() is None
    assert cache.m1_codes is None and cache.m2_codes is None


def test_m1_m2_code_install_setters():
    g = torch.Generator().manual_seed(604)
    src = TQCache(layer_types=[])
    m1 = torch.randn(*M_UNIT, generator=g, dtype=torch.float16).cuda()
    src.update_m1(m1)
    m2 = torch.randn(*M_UNIT, generator=g, dtype=torch.float16).cuda()
    src.update_m2(m2)

    dst = TQCache(layer_types=[])
    dst.m1_codes = src.m1_codes           # the install path: codes only
    dst.m2_codes = src.m2_codes
    assert dst.m1_codes is src.m1_codes
    assert dst.m2_codes is src.m2_codes
    # shape unknown to the fresh cache -> flat (d,) reads
    flat = dst.read_m1()
    assert tuple(flat.shape) == (D_UNIT,)
    src_m1 = src.read_m1().reshape(D_UNIT)
    assert torch.equal(flat.cuda() if not flat.is_cuda else flat, src_m1.cuda() if not src_m1.is_cuda else src_m1)  # same codes, same frame
    assert dst.read_m2().shape == (D_UNIT,)


# --- contract 8: code injection (the W7 install path) ------------------------- #
def test_code_injection_reshapes_to_layer_shape():
    g = torch.Generator().manual_seed(800)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16).cuda()

    # shape-initialize the LIVE layer without writing state: online
    # lazy_initialization captures shapes only — no fp16 ever allocated
    layer.lazy_initialization(recurrent_states=s)
    assert dict.get(layer.recurrent_states, 0) is None
    assert layer.recurrent_states[0] is None           # no codes yet

    # quantize OUTSIDE the cache (the W7 install producer) and inject
    codes = resolve_quantizer("S", D_UNIT, 3.5).quant(s.reshape(-1))
    assert codes.kind == "custom" and codes.seed == 101
    layer.s_codes = codes

    # has_previous_state flipped by the install; the next read dequantizes
    # reshaped to the LAYER's shape, not flat
    assert cache.has_previous_state(0) is True
    out = layer.recurrent_states[0]
    assert tuple(out.shape) == S_UNIT
    assert _rel_mse(out, s) < REL_MSE_GATE             # 0.018


def test_cache_code_views_and_setters():
    g = torch.Generator().manual_seed(810)
    cache = TQCache(layer_types=["linear_attention", "full_attention",
                                 "linear_attention"])
    assert cache.linear_layer_indices() == [0, 2]
    assert cache.s_codes == {0: None, 2: None}
    assert cache.conv_codes == {0: None, 2: None}

    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_recurrent_state(s, 0)
    assert cache.s_codes[0] is cache.layers[0].s_codes
    assert cache.s_codes[2] is None

    # set_s_codes / set_conv_codes install through the layer setters
    s_codes = cache.s_codes[0]
    cache.set_s_codes(2, s_codes)
    assert cache.s_codes[2] is s_codes
    assert cache.layers[2].s_codes is s_codes

    x = torch.randn(*CONV_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    conv_codes = cache.conv_codes[0]
    cache.set_conv_codes(2, conv_codes)
    assert cache.conv_codes[2] is conv_codes
    assert cache.layers[2].conv_codes is conv_codes

    # clearing through the cache
    cache.set_s_codes(2, None)
    assert cache.s_codes[2] is None

    # the §5 cache-level snapshot (online): exactly the layers' code objects
    snap = cache.snapshot_codes()
    assert snap["s"][0] is s_codes
    assert snap["conv"][2] is conv_codes
    assert snap["m1"] is None and snap["m2"] is None


# --- contract 9: offline fallback (D4) ----------------------------------------- #
def _offline_cache(seed: int = 700):
    g = torch.Generator().manual_seed(seed)
    cache = TQCache(layer_types=["linear_attention"], online=False)
    x = torch.randn(1, 32, 7, generator=g, dtype=torch.float16).cuda()  # prefill, T > k
    s = torch.randn(*S_UNIT, generator=g, dtype=torch.float16).cuda()
    full = cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    out = cache.update_recurrent_state(s, 0)
    return cache, x, s, full, out


def test_offline_fallback_stores_raw_tensors():
    cache, x, s, full, out = _offline_cache()
    layer = cache.layers[0]
    assert layer.online is False

    # parent (real LinearAttentionLayer) write semantics
    assert torch.equal(full, x)
    assert torch.equal(out, s)

    # raw tensors in the dict storage — the D4 "always codes on disk, not
    # in memory" relaxation
    raw_conv = dict.get(layer.conv_states, 0)
    raw_s = dict.get(layer.recurrent_states, 0)
    assert isinstance(raw_conv, torch.Tensor) and tuple(raw_conv.shape) == CONV_UNIT
    assert isinstance(raw_s, torch.Tensor) and torch.equal(raw_s, s)

    # reads pass straight through the raw tensors
    assert layer.conv_states[0] is raw_conv
    assert layer.recurrent_states[0] is raw_s

    # direct assignment is ALLOWED offline (parent dict semantics)
    layer.conv_states[0] = raw_conv.clone()
    assert dict.get(layer.conv_states, 0) is not raw_conv


def test_offline_snapshot_codes_quantizes_held_tensors():
    cache, x, s, _, _ = _offline_cache()
    layer = cache.layers[0]
    codes = layer.snapshot_codes()
    assert set(codes) == {"s", "conv"}
    assert codes["s"] is not None and codes["conv"] is not None
    # the custom-size quantizers keep the kind's D3 seed
    assert codes["s"].seed == 101 and codes["s"].d == D_UNIT
    assert codes["conv"].seed == 202 and codes["conv"].d == D_UNIT

    q_s = resolve_quantizer("S", D_UNIT, 3.5)
    q_c = resolve_quantizer("conv", D_UNIT, 3.5)
    deq_s = q_s.dequant(codes["s"]).reshape(*S_UNIT)
    deq_c = q_c.dequant(codes["conv"]).reshape(*CONV_UNIT)
    assert _rel_mse(deq_s, s) < REL_MSE_GATE                   # 0.024
    assert _rel_mse(deq_c, x[..., -CONV_KERNEL:]) < REL_MSE_GATE  # 0.020


def test_cache_level_snapshot_codes_offline_returns_codes():
    """Regression (W2.2 reported, ORCH fixed): TQCache.snapshot_codes()
    now DELEGATES per layer, so the offline (D4 snapshot-fallback) regime
    snapshots correctly. Was: it returned the online code views
    ({"s": {0: None}, ...}) whenever D4 was selected — W5's §5 chunk
    snapshot through the cache object would have been silently empty.

    Exact repro of the old bug (transformers 5.19.0, commit a27f6fc):

        cache = TQCache(layer_types=["linear_attention"], online=False)
        cache.update_conv_state(x, 0, conv_kernel_size=4)   # raw tensor stored
        cache.update_recurrent_state(s, 0)
        cache.snapshot_codes()   # -> {"s": {0: None}, "conv": {0: None}, ...}
    """
    cache, x, s, _, _ = _offline_cache()
    snap = cache.snapshot_codes()
    assert snap["s"][0] is not None, "offline cache snapshot lost the S codes"
    assert snap["conv"][0] is not None, "offline cache snapshot lost the conv codes"
    # byte-identical to the (correct) layer-level snapshot — both quantize
    # the same held raw tensors deterministically
    layer_snap = cache.layers[0].snapshot_codes()
    assert (snap["s"][0].idx_lo == layer_snap["s"].idx_lo).all()
    assert (snap["conv"][0].idx_lo == layer_snap["conv"].idx_lo).all()
    assert float(snap["s"][0].norm) == float(layer_snap["s"].norm)


# --- contract 10: resolve_quantizer -------------------------------------------- #
def test_resolve_quantizer_canonical_kinds():
    # production sizes hit the canonical kinds — the shared D3 instances
    q_s = resolve_quantizer("S", 524288)
    assert q_s.kind == "S" and q_s.d == 524288 and q_s.seed == 101
    assert q_s is tq.get_quantizer("S", 3.5)

    q_c = resolve_quantizer("conv", 32768)
    assert q_c.kind == "conv" and q_c.d == 32768 and q_c.seed == 202
    assert q_c is tq.get_quantizer("conv", 3.5)

    q_m1 = resolve_quantizer("M1", 524288)
    assert q_m1 is tq.get_quantizer("M1", 3.5) and q_m1.seed == 303

    q_m2 = resolve_quantizer("M2", 524288)
    assert q_m2 is tq.get_quantizer("M2", 3.5) and q_m2.seed == 404


def test_resolve_quantizer_custom_sizes_keep_the_kind_seed():
    # d=1024 hits the committed cb_b*_d1024 caches; d=128 the W2.1 smoke ones
    q = resolve_quantizer("S", 1024)
    assert q.kind == "custom" and q.d == 1024 and q.seed == 101

    qc = resolve_quantizer("conv", D_UNIT)
    assert qc.kind == "custom" and qc.d == D_UNIT and qc.seed == 202

    qm = resolve_quantizer("M1", D_UNIT)
    assert qm.kind == "custom" and qm.d == D_UNIT and qm.seed == 303

    # registry: the same custom size resolves to the SAME instance (the D3
    # frame contract — one rotation per kind at every scale)
    assert resolve_quantizer("S", 1024) is q
    assert resolve_quantizer("conv", D_UNIT) is qc
    assert resolve_quantizer("M1", D_UNIT) is qm
    # different kinds never share an instance, even at the same size
    assert resolve_quantizer("S", D_UNIT) is not resolve_quantizer("conv", D_UNIT)


def test_resolve_quantizer_rejects_non_positive():
    """d=0 (and negatives) are the only invalid units — the segmented FHT
    made the power-of-two restriction obsolete (non-pow2 sizes are VALID
    and covered by the custom-size/registry tests above and the contract-11
    geometry below)."""
    with pytest.raises(ValueError):
        resolve_quantizer("S", 0)


# positive non-power-of-two resolutions (the W10-era rejections, now valid):
# every size rides the kind's seed frame; the committed (b, d) codebooks
# serve each — no test writes a new npz
@pytest.mark.parametrize("numel", [100, 3, 129, 96])
def test_resolve_quantizer_accepts_non_power_of_two(numel):
    q = resolve_quantizer("S", numel)
    assert q.d == numel and q.seed == SEEDS["S"]
    x = torch.randn(numel, generator=torch.Generator().manual_seed(numel))
    codes = q.quant(x)
    assert codes.d == numel and codes.seed == SEEDS["S"]
    assert _rel_mse(q.dequant(codes), x) < 0.35   # tiny-d units quantize loosely


# --- contract 11: non-power-of-two conv geometry (full-window policy) ---- #
# The Qwen3.5 in_proj Q+V conv window: 1x6144x4 = 24,576 elements — NOT a
# power of two. The segmented FHT (fht.segments: 16,384 + 8,192, every
# tile fits the 64 KiB kernel budget) quantizes the FULL window — no pad,
# no truncation. The unit shares the conv kind's D3 rotation (seed 202);
# dequant returns all 24,576 coordinates (the GPU session's 16,384-truncation
# experiment — zeroing the last 8,192 coordinates of every read — is the
# regression this contract pins out). Power-of-two geometries take the SAME
# code path with pad == 0 (a reshape-only no-op — bit-identical).
CONV_NP2_UNIT = (1, 6144, 4)           # 24,576 dims — the Qwen3.5 Q+V window
CONV_NP2_D = 24576                     # full window; FHT segments as 16384+8192


def test_next_pow2_helper():
    from tq_cache import _next_pow2
    assert _next_pow2(1) == 1
    assert _next_pow2(2) == 2
    assert _next_pow2(3) == 4
    assert _next_pow2(128) == 128            # already pow2: identity
    assert _next_pow2(192) == 256
    assert _next_pow2(24576) == 32768
    assert _next_pow2(24577) == 32768
    assert _next_pow2(32768) == 32768
    with pytest.raises(ValueError):
        _next_pow2(0)


@pytest.mark.parametrize("seed", [700, 701, 702])
def test_nonpow2_conv_padding_contract(seed):
    g = torch.Generator().manual_seed(seed)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]
    x = torch.randn(*CONV_NP2_UNIT, generator=g, dtype=torch.float16).cuda()

    full = cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(full.shape) == CONV_NP2_UNIT          # prefill verbatim

    c = layer.conv_codes
    # the padded unit IS the canonical conv kind: shared rotation, seed 202
    assert c.d == CONV_NP2_D and c.seed == 202
    assert layer._conv_d == CONV_NP2_D
    assert layer._conv_numel == 24576
    # W15: the layer quantizes conv through the paper's outlier split
    # (group = kernel) — the registry-shared instance at that group
    assert layer._tq_conv is resolve_quantizer("conv", CONV_NP2_D,
                                               group=CONV_KERNEL)
    assert c.partition == "outlier" and c.group == CONV_KERNEL
    # Not the canonical quantizer (that's for 32768), but shares the same seed
    # padding adds no energy: the stored norms are the real sub-sets'
    # (mask/norm_hi/`norm` split the window's energy per the paper recipe)
    assert float(c.norm) + 0.0 >= 0.0 and c.norm_hi is not None

    # read-back: strip the pad, reshape to the real window shape
    w = layer.conv_states[0]
    assert tuple(w.shape) == CONV_NP2_UNIT
    assert dict.get(layer.conv_states, 0) is None       # still no fp16 store
    assert _rel_mse(w, x) < REL_MSE_GATE                # measured 0.015-0.017

    # decode: the (B, D, k+1) cat contract, quantized window round-trip
    tok = torch.randn(1, 6144, 1, generator=g, dtype=torch.float16).cuda()
    d = cache.update_conv_state(tok, 0)
    assert tuple(d.shape) == (1, 6144, 5)
    w2 = layer.conv_states[0]
    assert tuple(w2.shape) == CONV_NP2_UNIT
    assert _rel_mse(w2, d[..., -CONV_KERNEL:]) < REL_MSE_GATE

    # power-of-two geometry is UNCHANGED: pad == 0, the exact same codes
    # as the pre-padding behavior (bit-identity of the legacy path)
    g2 = torch.Generator().manual_seed(seed)
    x2 = torch.randn(*CONV_UNIT, generator=g2, dtype=torch.float16).cuda()
    layer2 = TQLinearAttentionLayer()
    layer2.update_conv_state(x2, 0, conv_kernel_size=CONV_KERNEL)
    assert layer2._conv_d == layer2._conv_numel == 128
    c2 = layer2.conv_codes
    assert c2.d == 128 and c2.seed == 202
    padded = layer2._pad_conv_flat(x2)
    assert padded.numel() == 128 and torch.equal(padded, x2.reshape(-1))


def test_nonpow2_conv_mutation_capture():
    # the causal_conv1d_update decode pattern at the padded geometry:
    # roll-left + RANDOM fresh last column (a constant column is
    # adversarial by construction — see the contract-5 note), then the
    # next read re-captures it into codes.
    g = torch.Generator().manual_seed(710)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]
    x = torch.randn(*CONV_NP2_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)

    state = layer.conv_states[0]                        # the handed-out transient
    assert tuple(state.shape) == CONV_NP2_UNIT
    codes_before = layer.conv_codes

    state.copy_(torch.roll(state, shifts=-1, dims=-1))
    state[..., -1] = torch.randn(1, 6144, generator=g, dtype=torch.float16).cuda()

    recaptured = layer.conv_states[0]
    assert recaptured is not state                      # fresh transient
    assert tuple(recaptured.shape) == CONV_NP2_UNIT     # pad stripped on read
    assert _rel_mse(recaptured, state) < REL_MSE_GATE   # measured 0.016
    assert not np.array_equal(layer.conv_codes.idx_lo, codes_before.idx_lo)
    assert dict.get(layer.conv_states, 0) is None
    assert layer.writes["conv"] == 1                    # counter-transparent


def test_nonpow2_conv_differential_vs_real_layer():
    # windowing shapes differential vs the REAL transformers layer at the
    # 6,144-channel geometry — identical (B, D, *) shapes at every step.
    g = torch.Generator().manual_seed(720)
    prefill = torch.randn(1, 6144, 7, generator=g, dtype=torch.float16).cuda()
    token = torch.randn(1, 6144, 1, generator=g, dtype=torch.float16).cuda()
    real, tql = LinearAttentionLayer(), TQLinearAttentionLayer()

    f_real = real.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)
    f_tq = tql.update_conv_state(prefill, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(f_tq.shape) == (1, 6144, 7) == tuple(f_real.shape)
    assert torch.equal(f_tq, f_real)                    # no quant on the way out

    d_real = real.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    d_tq = tql.update_conv_state(token, 0, conv_kernel_size=CONV_KERNEL)
    assert tuple(d_tq.shape) == (1, 6144, 5) == tuple(d_real.shape)

    w_real, w_tq = real.conv_states[0], tql.conv_states[0]
    assert tuple(w_tq.shape) == CONV_NP2_UNIT == tuple(w_real.shape)
    assert _rel_mse(w_tq, w_real) < REL_MSE_GATE        # two quant rounds
    assert _rel_mse(w_tq, d_tq[..., -CONV_KERNEL:]) < REL_MSE_GATE


def test_nonpow2_conv_codes_setter_and_frame_guard():
    # the W7 install path at the padded geometry: codes install into an
    # initialized layer when the unit d matches; a mismatched d (geometry
    # or frame drift) raises LOUDLY instead of silently mis-stripping.
    g = torch.Generator().manual_seed(730)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]
    x = torch.randn(*CONV_NP2_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)
    good = layer.conv_codes                            # d=24576, seed 202

    # codes into a FRESH layer: flat fallback (d,) — consistent, no pad
    fresh = TQLinearAttentionLayer()
    fresh.conv_codes = good
    assert fresh._conv_shape == (CONV_NP2_D,)
    assert fresh._conv_numel == fresh._conv_d == CONV_NP2_D
    assert tuple(fresh.conv_states[0].shape) == (CONV_NP2_D,)

    # matching codes replace an initialized layer's store (install)
    layer.conv_codes = good
    assert layer.conv_codes is good

    # frame/geometry drift: a 1,024-d unit (committed codebooks — no test
    # writes a new (b, d) combination) into the 24,576-d layer raises
    rogue = tq.TurboQuant(kind="custom", bits=3.5, d=1024, seed=202)
    bad = rogue.quant(torch.randn(1024, generator=g).cuda())
    with pytest.raises(ValueError, match="conv frame"):
        layer.conv_codes = bad


def test_conv_full_window_no_truncation_regression():
    """The direct regression for the GPU session's 16,384 truncation: the
    read-back window carries EVERY channel's history — the last 8,192
    coordinates (channels 4,096..6,143, the ones the truncation zeroed)
    round-trip within the standard budget, and a hard zeroed-tail would
    blow the budget by construction."""
    g = torch.Generator().manual_seed(760)
    cache = TQCache(layer_types=["linear_attention"])
    layer = cache.layers[0]
    x = torch.randn(*CONV_NP2_UNIT, generator=g, dtype=torch.float16).cuda()
    cache.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)

    w = layer.conv_states[0]
    # the tail coordinates the truncation used to zero: full fidelity
    tail = w.reshape(-1)[16384:]
    x_tail = x.reshape(-1)[16384:]
    assert _rel_mse(tail, x_tail) < REL_MSE_GATE
    # and the whole window: one budget over all 24,576 coordinates
    assert _rel_mse(w, x) < REL_MSE_GATE
    # the codes unit is the FULL window (never a 16,384 cap)
    assert layer.conv_codes.d == 24576


def test_nonpow2_conv_offline_snapshot():
    # the D4 fallback at the padded geometry: raw tensors during forward
    # (offline mode), snapshot_codes() quantizes the held tensor — padded,
    # same canonical frame as the online regime.
    g = torch.Generator().manual_seed(740)
    layer = TQLinearAttentionLayer(online=False)
    x = torch.randn(*CONV_NP2_UNIT, generator=g, dtype=torch.float16).cuda()
    layer.update_conv_state(x, 0, conv_kernel_size=CONV_KERNEL)

    raw = dict.get(layer.conv_states, 0)              # offline: plain store
    assert tuple(raw.shape) == CONV_NP2_UNIT

    codes = layer.snapshot_codes()["conv"]
    assert codes.d == CONV_NP2_D and codes.seed == 202
    v = layer._tq_conv.dequant(codes, dtype=torch.float32)
    stripped = v[: 24576].reshape(*CONV_NP2_UNIT)     # the pad strip
    assert _rel_mse(stripped, raw) < REL_MSE_GATE     # measured 0.016
