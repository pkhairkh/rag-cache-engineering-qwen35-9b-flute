"""test_hooks.py — W4.2: acceptance gates for src/rag/hooks.py (W4.1).

Gates (SPECIFICATION §1's enumerated hook contract + the §5 snapshot):
  1. hook_map(SPEC_LAYER_TYPES) reproduces the enumerated table EXACTLY
     (9 points; union of linear_layers == the 24 linear indices; no
     duplicates; after_layers = [0, 3, 7, ..., 31]).
  2. the general rule on smaller stacks, + the loud refusals.
  3. stub 32-layer stack: all 9 points fire, exactly 24 S codes + 24 conv
     codes captured, each dequantizing within the Wave-1/2 rel-MSE budget
     (< 0.06) of the layer's true final state.
  4. the round-trip budget on 3 sampled layers (one per hook region).
  5. capture_vector(): 1-D fp32, §4 ordering (S layers ascending, then
     M1, then M2) — pinned by a bit-identical manual concat + segment
     boundaries.
  6. M1/M2 in the snapshot (read at capture time) + their segments.
  7. incomplete capture raises: partial stack (hooks never fired) and a
     silent linear layer (mapped codes missing).
  8. idempotent re-fire: a second full pass refreshes every captured code
     (latest wins — the decode-loop contract).
  9. real-model integration: a tiny real Qwen3_5TextModel + TQCache, the
     hooks attached to m.layers, one prefill forward -> full capture.

DECISIONS pinned here (see hooks.py's module docstring):
  * EMPTY CAPTURE POINTS are kept, not skipped: a full-attention boundary
    with no uncaptured linear layers behind it still fires (a pure
    boundary witness). The production map is unchanged under this
    decision — hook_map(SPEC_LAYER_TYPES) has NO empty points.
  * DIAGRAM ARITHMETIC: spec §1's inline "1 + 8×3 = 24" line miscounts
    its own diagram (hook 0 takes S_0 out of the first group). The true
    total is 1 + 2 + 7×3 = 24: group sizes [1, 2, 3, 3, 3, 3, 3, 3, 3].
  * [L] alone is VALID (hook 0 fully captures it); the all-linear refusal
    is exactly the total-mismatch case (linear layers behind the last
    boundary — e.g. [L, L, ...], or trailing L's after the last F).

Determinism: every random tensor comes from a pinned torch.Generator
seed; every rel-MSE number quoted in comments was measured on this box
with exactly those seeds (all-24 pass-1 max 0.0324, pass-2 max 0.0436,
M1/M2 0.0218/0.0198 — gates hold at the house budget 0.06).
"""
from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import Qwen3_5TextConfig

from hooks import (
    SPEC_LAYER_TYPES,
    CacheSnapshot,
    CaptureHooks,
    CapturePoint,
    hook_map,
)
from tq_cache import TQCache, resolve_quantizer
from turboquant import TQCodes

LIN, FULL = "linear_attention", "full_attention"

# ------------------------------------------------------------- geometry ---
# Stub sizes: power-of-two FHT units (PROPOSAL D2) at the committed
# codebook scales (d128). S (1, 8, 16) = 128 dims; the conv input is
# prefill-shaped (1, 32, 6) so the windowing contract is exercised (the
# stored window is its last 4 columns); M1/M2 (1, 128) = 128 dims.
S_UNIT = (1, 8, 16)
CONV_IN_UNIT = (1, 32, 6)
CONV_UNIT = (1, 32, 4)
M_UNIT = (1, 128)
CONV_KERNEL = 4
S_D = 128
M_D = 128

REL_MSE_GATE = 0.06          # the house single-quant-round budget

LINEAR_IDX = [i for i, lt in enumerate(SPEC_LAYER_TYPES) if lt == LIN]
assert len(LINEAR_IDX) == 24

# THE enumerated contract (the authority; hooks.py's docstring cites it)
SPEC_MAP_EXPECTED = [
    (0, 0, (0,)),
    (1, 3, (1, 2)),
    (2, 7, (4, 5, 6)),
    (3, 11, (8, 9, 10)),
    (4, 15, (12, 13, 14)),
    (5, 19, (16, 17, 18)),
    (6, 23, (20, 21, 22)),
    (7, 27, (24, 25, 26)),
    (8, 31, (28, 29, 30)),
]


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-30).item()


# ------------------------------------------------------------ stub stack ---
def _stub_s(layer_idx: int, pass_idx: int) -> torch.Tensor:
    """Deterministic per-(layer, pass) recurrent state."""
    g = torch.Generator().manual_seed(5000 + 101 * pass_idx + layer_idx)
    return torch.randn(*S_UNIT, generator=g, dtype=torch.float32)


def _stub_conv_in(layer_idx: int, pass_idx: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(9000 + 101 * pass_idx + layer_idx)
    return torch.randn(*CONV_IN_UNIT, generator=g, dtype=torch.float32)


class _StubLinear(nn.Module):
    """A linear-attention stand-in: writes deterministic S + conv codes
    into the bound TQCache exactly the way the real GatedDeltaNet forward
    does (update_conv_state, then update_recurrent_state), and remembers
    the true final states for the round-trip gates. `write=False` makes
    it a SILENT layer (runs, writes nothing — the missing-codes gate)."""

    def __init__(self, layer_idx: int, cache: TQCache, write: bool = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache = cache
        self.write = write
        self.last_s: torch.Tensor | None = None
        self.last_conv: torch.Tensor | None = None

    def forward(self, x, pass_idx: int = 0):
        conv_in = _stub_conv_in(self.layer_idx, pass_idx)
        s = _stub_s(self.layer_idx, pass_idx)
        if self.write:
            self.cache.update_conv_state(
                conv_in, self.layer_idx, conv_kernel_size=CONV_KERNEL)
            self.cache.update_recurrent_state(s, self.layer_idx)
            self.last_conv = conv_in[..., -CONV_KERNEL:].clone()
            self.last_s = s.clone()
        return x


class _StubFull(nn.Module):
    """A full-attention stand-in: identity no-op (spec §2.4)."""

    def forward(self, x, pass_idx: int = 0):
        return x


class _StackModel(nn.Module):
    """A model-shaped wrapper (the `.layers` attach route)."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, pass_idx: int = 0):
        for layer in self.layers:
            x = layer(x, pass_idx)
        return x


def _build_stub(layer_types=None, silent=()):
    """(layers ModuleList, bound TQCache) for a layer plan."""
    types = list(layer_types if layer_types is not None else SPEC_LAYER_TYPES)
    cache = TQCache(layer_types=types)
    layers = nn.ModuleList([
        _StubLinear(i, cache, write=(i not in silent)) if lt == LIN
        else _StubFull()
        for i, lt in enumerate(types)])
    return layers, cache


def _run(layers, pass_idx: int = 0, upto: int | None = None) -> None:
    """Run the stack's layers in order (hooks fire per layer)."""
    n = len(layers) if upto is None else upto
    x = torch.zeros(1)
    for layer in list(layers)[:n]:
        x = layer(x, pass_idx)


# =============================================== 1. the enumerated contract ==
def test_hook_map_matches_enumerated_contract():
    pts = hook_map(SPEC_LAYER_TYPES)
    assert len(pts) == 9
    assert [(p.hook_idx, p.after_layer, p.linear_layers) for p in pts] \
        == SPEC_MAP_EXPECTED
    assert all(isinstance(p, CapturePoint) for p in pts)

    # hook indices sequential 0..8; boundaries exactly [0, 3, ..., 31]
    assert [p.hook_idx for p in pts] == list(range(9))
    assert [p.after_layer for p in pts] == [0, 3, 7, 11, 15, 19, 23, 27, 31]

    # the union of captured layers is EXACTLY the 24 linear indices,
    # each captured exactly once (no duplicates)
    captured = [L for p in pts for L in p.linear_layers]
    assert len(captured) == 24
    assert sorted(captured) == LINEAR_IDX
    assert len(set(captured)) == 24

    # diagram-arithmetic note: 1 + 2 + 7×3 = 24 (NOT the spec's inline
    # "1 + 8×3" — hook 0 takes S_0 out of the first group)
    assert [len(p.linear_layers) for p in pts] == [1, 2, 3, 3, 3, 3, 3, 3, 3]

    # SPEC_LAYER_TYPES is the 32-layer [L,L,L,F]×8 plan
    assert len(SPEC_LAYER_TYPES) == 32
    assert SPEC_LAYER_TYPES[:4] == [LIN, LIN, LIN, FULL]
    assert SPEC_LAYER_TYPES[28:] == [LIN, LIN, LIN, FULL]
    assert SPEC_LAYER_TYPES == ([LIN, LIN, LIN, FULL] * 8)


# =============================================== 2. the general rule ========
def test_general_rule_smaller_stacks():
    # [L, L, L, F]: hook 0 captures (0,), hook 1 (after the F) captures (1, 2)
    assert [(p.hook_idx, p.after_layer, p.linear_layers)
            for p in hook_map([LIN, LIN, LIN, FULL])] \
        == [(0, 0, (0,)), (1, 3, (1, 2))]

    # THE DECISION — a full-attention boundary with no uncaptured linear
    # layers behind it stays as an EMPTY capture point (pure boundary
    # witness): [L, F, L, L, F] -> hook 1 after layer 1 captures ()
    assert [(p.hook_idx, p.after_layer, p.linear_layers)
            for p in hook_map([LIN, FULL, LIN, LIN, FULL])] \
        == [(0, 0, (0,)), (1, 1, ()), (2, 4, (2, 3))]

    # [L, F]: the trailing boundary is empty too
    assert [(p.hook_idx, p.after_layer, p.linear_layers)
            for p in hook_map([LIN, FULL])] == [(0, 0, (0,)), (1, 1, ())]

    # layer 0 full-attention: hook 0 fires after it but captures nothing
    assert [(p.hook_idx, p.after_layer, p.linear_layers)
            for p in hook_map([FULL, LIN, LIN, FULL])] \
        == [(0, 0, ()), (1, 3, (1, 2))]

    # consecutive full-attention boundaries: the middle one is empty
    assert [(p.hook_idx, p.after_layer, p.linear_layers)
            for p in hook_map([LIN, LIN, FULL, FULL])] \
        == [(0, 0, (0,)), (1, 2, (1,)), (2, 3, ())]

    # a 4-layer capture group (uneven groups are fine)
    assert [(p.hook_idx, p.after_layer, p.linear_layers)
            for p in hook_map([LIN, LIN, LIN, FULL, LIN, LIN, LIN, LIN, FULL])] \
        == [(0, 0, (0,)), (1, 3, (1, 2)), (2, 8, (4, 5, 6, 7))]

    # [L] alone: hook 0 fully captures it (no uncaptured layers remain)
    assert [(p.hook_idx, p.after_layer, p.linear_layers)
            for p in hook_map([LIN])] == [(0, 0, (0,))]

    # totals == the linear count, in EVERY valid case
    valid_plans = [
        SPEC_LAYER_TYPES,
        [LIN, LIN, LIN, FULL],
        [LIN, FULL, LIN, LIN, FULL],
        [LIN, FULL],
        [FULL, LIN, LIN, FULL],
        [LIN, LIN, FULL, FULL],
        [LIN, LIN, LIN, FULL, LIN, LIN, LIN, LIN, FULL],
        [LIN],
        [LIN, LIN, LIN, FULL] * 3 + [LIN, LIN, LIN, LIN, FULL],
    ]
    for types in valid_plans:
        pts = hook_map(types)
        captured = [L for p in pts for L in p.linear_layers]
        lin = [i for i, lt in enumerate(types) if lt == LIN]
        assert sorted(captured) == lin, f"plan {types}: {captured} vs {lin}"
        assert [p.hook_idx for p in pts] == list(range(len(pts)))
        assert [p.after_layer for p in pts] == [0] + [
            i for i, lt in enumerate(types) if i > 0 and lt == FULL]

    # the spec map is UNCHANGED under the empty-point decision: the
    # production pattern yields no empty capture points
    assert all(p.linear_layers for p in hook_map(SPEC_LAYER_TYPES))


def test_general_rule_loud_refusals():
    # all-linear stack: only hook 0 fires; layers 1..4 are NEVER captured
    with pytest.raises(ValueError, match=r"\[1, 2, 3, 4\]"):
        hook_map([LIN] * 5)
    # trailing linear layer after the last full-attention boundary
    with pytest.raises(ValueError, match=r"\[5\]"):
        hook_map([LIN, LIN, FULL, LIN, FULL, LIN])
    with pytest.raises(ValueError, match=r"\[2\]"):
        hook_map([LIN, FULL, LIN])
    # empty plan / unknown layer type
    with pytest.raises(ValueError, match="empty"):
        hook_map([])
    with pytest.raises(ValueError, match="sliding_attention"):
        hook_map([LIN, "sliding_attention"])
    # the construction-time refusal surfaces through CaptureHooks too
    with pytest.raises(ValueError):
        CaptureHooks([LIN] * 5)


# =============================================== 3. stub-stack 24/24 ========
def test_stub_stack_captures_all_24():
    layers, cache = _build_stub()
    hooks = CaptureHooks()          # default: SPEC_LAYER_TYPES
    hooks.bind(cache)
    hooks.attach(layers)
    _run(layers)

    # all 9 points fired exactly once
    assert hooks.fired_counts == {i: 1 for i in range(9)}

    snap = hooks.capture()
    assert isinstance(snap, CacheSnapshot)

    # exactly the 24 mapped S codes + 24 conv codes, keyed by layer idx
    assert sorted(snap.s_codes) == LINEAR_IDX
    assert sorted(snap.conv_codes) == LINEAR_IDX
    for L in LINEAR_IDX:
        assert isinstance(snap.s_codes[L], TQCodes)
        assert isinstance(snap.conv_codes[L], TQCodes)
        # the D3 frame contract: S seed 101, conv seed 202, d = 128
        assert snap.s_codes[L].seed == 101 and snap.s_codes[L].d == S_D
        assert snap.conv_codes[L].seed == 202 and snap.conv_codes[L].d == S_D

    # identity: the captured codes ARE the cache's code objects (the
    # hooks copy references — codes only, never fp tensors)
    for L in (0, 5, 17, 30):
        assert snap.s_codes[L] is cache.s_codes[L]
        assert snap.conv_codes[L] is cache.conv_codes[L]

    # every captured S code dequantizes within the budget of the layer's
    # true final state (measured max over the 24: 0.0324)
    q_s = resolve_quantizer("S", S_D)
    for L in LINEAR_IDX:
        rel = _rel_mse(q_s.dequant(snap.s_codes[L]), layers[L].last_s)
        assert rel < REL_MSE_GATE, f"layer {L}: rel-MSE {rel:.4f}"

    # M1/M2 untouched by the stub forward: absent, and allowed to be
    assert snap.m1_codes is None and snap.m2_codes is None


# =============================================== 4. round-trip budget =======
def test_round_trip_budget_three_sampled_layers():
    layers, cache = _build_stub()
    hooks = CaptureHooks()
    hooks.bind(cache)
    hooks.attach(layers)
    _run(layers)
    snap = hooks.capture()

    q_s = resolve_quantizer("S", S_D)
    # one layer per hook region: hooks 1, 4 and 8's capture groups
    for L in (2, 13, 29):
        dq = q_s.dequant(snap.s_codes[L])
        assert dq.dtype == torch.float32 and dq.numel() == S_D
        rel = _rel_mse(dq, layers[L].last_s)
        assert rel < REL_MSE_GATE, f"layer {L}: rel-MSE {rel:.4f}"
        # and the codes ARE single-round: equal to quantizing the true
        # state again, byte-for-byte
        recode = q_s.quant(layers[L].last_s.reshape(-1))
        assert torch.equal(dq, q_s.dequant(recode))


# =============================================== 5. capture_vector ==========
def test_capture_vector_layout():
    layers, cache = _build_stub()
    hooks = CaptureHooks()
    hooks.bind(cache)
    hooks.attach(layers)
    _run(layers)
    g = torch.Generator().manual_seed(777)
    cache.update_m1(torch.randn(*M_UNIT, generator=g, dtype=torch.float32))
    cache.update_m2(torch.randn(*M_UNIT, generator=g, dtype=torch.float32))
    snap = hooks.capture()

    v = snap.capture_vector()
    assert v.dim() == 1
    assert v.dtype == torch.float32
    # length = sum of captured S dims + M1 dim + M2 dim
    assert v.numel() == 24 * S_D + M_D + M_D == 3328

    # §4 ordering, bit-identical: S layers ascending, then M1, then M2
    q_s = resolve_quantizer("S", S_D)
    q_m1 = resolve_quantizer("M1", M_D)
    q_m2 = resolve_quantizer("M2", M_D)
    expected = torch.cat(
        [q_s.dequant(snap.s_codes[L]) for L in LINEAR_IDX]
        + [q_m1.dequant(snap.m1_codes), q_m2.dequant(snap.m2_codes)])
    assert torch.equal(v, expected)

    # segment boundaries: each S segment is exactly its layer's piece
    for i, L in enumerate(LINEAR_IDX[:3]):
        assert torch.equal(v[i * S_D:(i + 1) * S_D],
                           q_s.dequant(snap.s_codes[L]))
    assert torch.equal(v[-M_D:], q_m2.dequant(snap.m2_codes))
    assert torch.equal(v[-2 * M_D:-M_D], q_m1.dequant(snap.m1_codes))

    # the ordering check is non-vacuous: distinct states per layer
    # (independent generators) -> distinct segments
    assert not torch.equal(v[:S_D], v[S_D:2 * S_D])
    assert not torch.equal(v[:S_D], v[-2 * M_D:-M_D])
    assert not torch.equal(v[-2 * M_D:-M_D], v[-M_D:])


# =============================================== 6. M1/M2 in the snapshot ===
def test_m1_m2_read_at_capture_time():
    layers, cache = _build_stub()
    hooks = CaptureHooks()
    hooks.bind(cache)
    hooks.attach(layers)
    _run(layers)

    # before any M1/M2 write: absent (None), capture still succeeds
    snap0 = hooks.capture()
    assert snap0.m1_codes is None and snap0.m2_codes is None
    assert snap0.capture_vector().numel() == 24 * S_D

    g = torch.Generator().manual_seed(778)
    m1 = torch.randn(*M_UNIT, generator=g, dtype=torch.float32)
    m2 = torch.randn(*M_UNIT, generator=g, dtype=torch.float32)
    cache.update_m1(m1)
    cache.update_m2(m2)
    snap = hooks.capture()

    # the snapshot carries the codes (read at capture time)
    assert snap.m1_codes is cache.m1_codes
    assert snap.m2_codes is cache.m2_codes
    q_m1 = resolve_quantizer("M1", M_D)
    q_m2 = resolve_quantizer("M2", M_D)
    assert _rel_mse(q_m1.dequant(snap.m1_codes), m1) < REL_MSE_GATE  # 0.022
    assert _rel_mse(q_m2.dequant(snap.m2_codes), m2) < REL_MSE_GATE  # 0.020

    # capture_vector includes their segments
    v = snap.capture_vector()
    assert v.numel() == 24 * S_D + 2 * M_D
    assert torch.equal(v[-M_D:], q_m2.dequant(snap.m2_codes))
    assert torch.equal(v[-2 * M_D:-M_D], q_m1.dequant(snap.m1_codes))

    # read at CAPTURE time: an m1 rewrite between forwards is picked up
    # by the next capture without any hook re-firing
    g2 = torch.Generator().manual_seed(779)
    cache.update_m1(torch.randn(*M_UNIT, generator=g2, dtype=torch.float32))
    snap2 = hooks.capture()
    assert snap2.m1_codes is cache.m1_codes
    assert snap2.m1_codes is not snap.m1_codes
    assert _rel_mse(q_m1.dequant(snap2.m1_codes), m1) > 0.5  # stale ref gone


# =============================================== 7. incomplete capture ======
def test_partial_stack_raises():
    layers, cache = _build_stub()
    hooks = CaptureHooks()
    hooks.bind(cache)
    hooks.attach(layers)
    _run(layers, upto=10)      # hooks 3..8 (after layers 11..31) never fire
    with pytest.raises(RuntimeError, match="never fired"):
        hooks.capture()


def test_missing_codes_raise():
    # layer 4 runs but writes nothing: every hook fires, hook 2 copies
    # None for layer 4 -> capture raises on the missing codes
    layers, cache = _build_stub(silent={4})
    hooks = CaptureHooks()
    hooks.bind(cache)
    hooks.attach(layers)
    _run(layers)
    assert hooks.fired_counts == {i: 1 for i in range(9)}  # pattern ran
    with pytest.raises(RuntimeError, match="no codes"):
        hooks.capture()


def test_bind_and_capture_loud_validation():
    hooks = CaptureHooks()

    # capture without a bound cache
    with pytest.raises(RuntimeError, match="bind"):
        hooks.capture()

    # attach: stack length mismatch
    with pytest.raises(ValueError, match="mismatch"):
        hooks.attach(nn.ModuleList([_StubFull(), _StubFull()]))

    # attach: not a stack at all
    with pytest.raises(TypeError):
        hooks.attach(torch.zeros(3))

    # bind: cache plan disagrees with the hook map
    cache4 = TQCache(layer_types=[LIN, FULL, LIN, FULL])
    with pytest.raises(ValueError, match="disagrees"):
        hooks.bind(cache4)

    # bind: not a TQCache-like object
    with pytest.raises(TypeError):
        hooks.bind(object())

    # hook firing with no cache bound: loud AT THE SOURCE
    layers, _ = _build_stub()
    hooks2 = CaptureHooks()
    hooks2.attach(layers)
    with pytest.raises(RuntimeError, match="no cache bound"):
        _run(layers, upto=2)


# =============================================== 8. idempotent re-fire ======
def test_attach_idempotent_no_duplicate_hooks():
    # the `.layers` model route, attached TWICE: old handles come off
    # first — each hook fires exactly once per forward
    layers, cache = _build_stub()
    model = _StackModel(layers)
    hooks = CaptureHooks()
    hooks.bind(cache)
    hooks.attach(model)
    hooks.attach(model)
    with torch.no_grad():
        model(torch.zeros(1))
    assert hooks.fired_counts == {i: 1 for i in range(9)}
    snap = hooks.capture()
    assert sorted(snap.s_codes) == LINEAR_IDX


def test_refire_latest_wins():
    layers, cache = _build_stub()
    hooks = CaptureHooks()
    hooks.bind(cache)
    hooks.attach(layers)

    _run(layers, pass_idx=0)               # pass 1
    snap1 = hooks.capture()
    _run(layers, pass_idx=1)               # pass 2 = decode-step refresh
    snap2 = hooks.capture()

    # every hook fired on every pass
    assert hooks.fired_counts == {i: 2 for i in range(9)}

    q_s = resolve_quantizer("S", S_D)
    for L in (0, 8, 22):
        # snap2 reflects the LATEST codes: pass-2 states (measured max
        # over all 24: 0.0436 — S is re-quantized fresh every pass)
        dq2 = q_s.dequant(snap2.s_codes[L])
        assert _rel_mse(dq2, layers[L].last_s) < REL_MSE_GATE
        # snap1's codes are the pass-1 objects: distinct codes, stale
        # against the pass-2 state (independent randoms -> rel-MSE ~ 2)
        dq1 = q_s.dequant(snap1.s_codes[L])
        assert snap1.s_codes[L] is not snap2.s_codes[L]
        assert not torch.equal(dq1, dq2)
        assert _rel_mse(dq1, layers[L].last_s) > 0.5


# =============================================== 9. real-model integration ==
def test_real_model_integration():
    import modeling  # the vendored Qwen3.5 (src/scripts, conftest-anchored)

    plan = [LIN, FULL, LIN, FULL]
    cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=4, layer_types=plan,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, use_m1m2=True, m1m2_mem_size=8)
    torch.manual_seed(21)
    m = modeling.Qwen3_5TextModel(cfg).eval()

    cache = TQCache(config=cfg)
    hooks = CaptureHooks(plan)
    hooks.attach(m.layers)
    hooks.bind(cache)

    ids = torch.randint(0, 128, (1, 6),
                        generator=torch.Generator().manual_seed(22))
    with torch.no_grad():
        out = m(input_ids=ids, past_key_values=cache, use_cache=True)
    assert tuple(out.last_hidden_state.shape) == (1, 6, 64)

    # the 4-layer pattern's map: hook 0 -> [0]; hook 1 after the layer-1
    # full-attention -> [] (the documented EMPTY point); hook 2 -> [2]
    assert hooks.points == [
        CapturePoint(0, 0, (0,)), CapturePoint(1, 1, ()),
        CapturePoint(2, 3, (2,))]
    assert hooks.fired_counts == {0: 1, 1: 1, 2: 1}

    snap = hooks.capture()
    assert sorted(snap.s_codes) == [0, 2]        # the 2 linear layers
    assert sorted(snap.conv_codes) == [0, 2]
    assert snap.s_codes[0] is cache.s_codes[0]   # codes-only reference copy
    assert snap.s_codes[2] is cache.s_codes[2]

    # the tiny geometry's units: S (1, 4, 16, 16) -> 1024 dims; conv
    # (1, 128, 4) -> 512; M1/M2 (4, 8, 16) -> 512
    assert snap.s_codes[0].d == 1024 and snap.s_codes[2].d == 1024
    assert snap.conv_codes[0].d == 512 and snap.conv_codes[2].d == 512
    assert snap.m1_codes is not None and snap.m1_codes.d == 512
    assert snap.m2_codes is not None and snap.m2_codes.d == 512
    # zero write gates (P3): the memories round-trip as codes of zero
    assert float(snap.m1_codes.norm) == 0.0
    assert float(snap.m2_codes.norm) == 0.0

    # the §4 vector through the real codes: 2 S + M1 + M2 segments
    v = snap.capture_vector()
    assert v.dim() == 1 and v.dtype == torch.float32
    assert v.numel() == 2 * 1024 + 512 + 512 == 3072
    assert torch.equal(v[-512:], torch.zeros(512))       # M2 zero-norm
    assert torch.equal(v[-1024:-512], torch.zeros(512))  # M1 zero-norm
    q_s = resolve_quantizer("S", 1024)
    assert torch.equal(v[:1024], q_s.dequant(snap.s_codes[0]))
    assert torch.equal(v[1024:2048], q_s.dequant(snap.s_codes[2]))
    assert torch.isfinite(v).all()
