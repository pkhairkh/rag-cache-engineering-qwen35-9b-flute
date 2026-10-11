"""test_w19_finetune.py — the W19 saddle + capture-pool gates.

THE ROOT CAUSE PINNED (the box's flat run): at the P3 zero-init write
gates the §7 InfoNCE has an EXACT saddle — every pair-side replays to
the SAME state (m_init + 0), the states are the zero vector (the system
reset point is built at P3 gates), every cosine is 0, every logit equal,
loss = ln(B) (the observed 2.0794 at B=8), and the gate gradient is
IDENTICALLY zero (zero states: dsim/dstate = 0 exactly — the clamped
norm Jacobian multiplies a zero vector; identical nonzero states: cos at
its maximum AND the CE row-gradient sums cancel exactly over the shared
direction). The local write Jacobian dm/dg = Delta is nonzero (P3's
branch-free add) — but dL/dg = <dL/dm, Delta> needs dL/dm, and dL/dm is
the zero that kills it. Gates here:

  1. THE SADDLE: zero-init + zero m_init -> grads EXACTLY zero
     (torch.equal against zeros), loss EXACTLY ln(B); 3 AdamW steps
     leave the gates at exactly 0.0 — the box's 50-step flat line in
     miniature. The nonzero-m_init regime (open-gate system state) is
     the same cancellation (<= 1e-5 residual at the stub scale).
  2. THE ESCAPE: GatesTrainConfig.gate_init (0.1 default) fills both
     write gates before step 0 — the escaped point has LIVE gradients
     (the W19 history field > 0) and one step MOVES the gates.
  3. LEARNING: a controlled rig — layer 0 carries the pair signal, layer
     1 a SHARED common component (the W16 collapse, call-borne so the
     delta centering does not remove it), layer 2 small idiosyncratic
     noise. The loop must descend the InfoNCE and put layer 0's write
     weight above the common layer's.
  4. DELTA LOSS: the default scores (m - m_init) — the W16 centered
     retrieval frame's geometry; the trainer's history loss equals the
     manual delta-InfoNCE, and delta_loss=False recovers the absolute
     form (the A/B).
  5. THE CAPTURE POOL: refresh_every=2 halves the prefill count
     (cache_factory invocations) over 4 steps; refresh_every=1 is the
     exact W17 regime; pool reuse replays bit-identically (the Function
     is pure in its inputs).
  6. INSTRUMENTATION: every history entry carries the gate-grad absmax
     (the box's explicit ask) + the capture/replay/backward phase times.
"""
from __future__ import annotations

import math

import pytest
import torch

import m1m2_finetune as ft
from m1m2 import M1M2
from tq_cache import TQCache

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H, D, MEM, NLAY = 4, 16, 8, 3
LAYER_TYPES = ["linear_attention", "full_attention"] * 2
BITS = 3.5


class _ModelShim:
    """The `model` train_gates needs: .m1m2 + .parameters() — a bare
    M1M2 module wrapped (the replay needs no transformer)."""

    def __init__(self, module: M1M2):
        self.m1m2 = module

    def parameters(self, recurse=True):
        return self.m1m2.parameters(recurse)


def _synth_calls(gen, n_pairs: int, signal_layer: int = 0,
                 common_layer: int = 1, common_scale: float = 3.0,
                 noise_scale: float = 0.1, per_pair: int = 2):
    """Controlled-signal capture stand-ins (B=1 ranks):
      * signal_layer: k IDENTICAL within a pair (+ 2% jitter on side b),
        different across pairs — the retrievable pair signal;
      * common_layer: the SAME k for every side (call-borne common —
        the W16 collapse mode; the delta centering does NOT remove it);
      * every other layer: small idiosyncratic noise.
    v = 0.5 * k everywhere. Returns sides[pair][side] = call list."""
    base_k = torch.randn(n_pairs, H, 6, D, generator=gen).to(DEVICE)
    common_k = common_scale * torch.randn(1, H, 6, D, generator=gen).to(DEVICE)
    sides = []
    for p in range(n_pairs):
        pair_sides = []
        for side in range(per_pair):
            calls = []
            for lay in range(NLAY):
                if lay == signal_layer:
                    k = base_k[p].unsqueeze(0).clone()
                    if side:
                        k = k + 0.02 * torch.randn(
                            1, H, 6, D, generator=gen).to(DEVICE)
                elif lay == common_layer:
                    k = common_k.clone()
                else:
                    k = noise_scale * torch.randn(
                        1, H, 6, D, generator=gen).to(DEVICE)
                v = 0.5 * k
                calls.append((k.half(), v.half(), lay, None))
            pair_sides.append(calls)
        sides.append(pair_sides)
    return sides


def _stacked_states(module, sides, m1_i, m2_i, delta: bool, grad: bool):
    """Replay every side (the plain chain) and stack the four InfoNCE
    operands; `grad` keeps the graph (the saddle/escape gates need it)."""
    with torch.set_grad_enabled(grad):
        out = []
        for pair_sides in sides:
            ps = []
            for calls in pair_sides:
                m1, m2 = ft.replay_states(module, calls, m1_i, m2_i,
                                          memopt=False)
                if delta:
                    m1, m2 = m1 - m1_i, m2 - m2_i
                ps.append((m1, m2))
            out.append(ps)
    return (torch.stack([s[0][0] for s in out]),
            torch.stack([s[0][1] for s in out]),
            torch.stack([s[1][0] for s in out]),
            torch.stack([s[1][1] for s in out]))


def _trainable_gates(module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.write_gate_k.requires_grad_(True)
    module.write_gate_v.requires_grad_(True)


def _patched_train(module, sides, m1_i, m2_i, **cfg_kw):
    """train_gates against the SYNTHETIC calls (prefill_capture_calls
    patched to hand out the recorded sides verbatim)."""
    shim = _ModelShim(module)
    _trainable_gates(module)
    flat = [s for pair in sides for s in pair]
    idx = {"i": 0}

    def _fake_capture(model, ids, cache, system):
        s = flat[idx["i"] % len(flat)]
        idx["i"] += 1
        return s, m1_i, m2_i

    def _batches():
        while True:
            yield [(None, None)] * 4

    orig = ft.prefill_capture_calls
    ft.prefill_capture_calls = _fake_capture
    try:
        cfg = ft.GatesTrainConfig(log_every=0, **cfg_kw)
        return ft.train_gates(shim, _batches(), system=None,
                              cache_factory=lambda: None, cfg=cfg)
    finally:
        ft.prefill_capture_calls = orig


# ================= 1. the saddle: exact zero gradient ==================
def test_w19f_1_p3_zero_init_is_an_exact_saddle():
    """THE box bug reproduced in miniature: P3 zero gates + the zero
    reset point -> loss EXACTLY ln(B), grads EXACTLY zero, 3 AdamW steps
    leave the gates at exactly 0.0. Not a weak signal — exact arithmetic."""
    gen = torch.Generator().manual_seed(1)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    sides = _synth_calls(gen, n_pairs=4)
    m1_i = module.init_state(dtype=torch.float32, device=DEVICE)   # zeros
    m2_i = module.init_state(dtype=torch.float32, device=DEVICE)
    _trainable_gates(module)

    m1_a, m2_a, m1_b, m2_b = _stacked_states(module, sides, m1_i, m2_i,
                                              delta=True, grad=True)
    loss = ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b, tau=0.07)
    assert float(loss.detach()) == pytest.approx(math.log(4), abs=1e-6), \
        "the saddle loss must be exactly ln(B)"
    loss.backward()
    assert torch.equal(module.write_gate_k.grad,
                       torch.zeros_like(module.write_gate_k)), \
        "zero states: the gate gradient is EXACTLY zero"
    assert torch.equal(module.write_gate_v.grad,
                       torch.zeros_like(module.write_gate_v))

    opt = torch.optim.AdamW([module.write_gate_k, module.write_gate_v],
                            lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        m1_a, m2_a, m1_b, m2_b = _stacked_states(module, sides, m1_i, m2_i,
                                                 delta=True, grad=True)
        ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b, tau=0.07).backward()
        opt.step()
    assert float(module.write_gate_k.detach().abs().max()) == 0.0 and \
        float(module.write_gate_v.detach().abs().max()) == 0.0, \
        "3 optimizer steps at the exact saddle must not move the gates"


def test_w19f_1b_open_gate_system_state_is_the_same_cancellation():
    """The OTHER regime (a system reset point built at open gates, m_init
    nonzero): all pair-states identical -> every cosine AT its maximum
    and the CE row-gradient sums cancel over the shared direction -> the
    gate grads are numerically dead (<= 1e-5 at the stub scale; AdamW
    would amplify any residual into noise-steering, hence gate_init)."""
    gen = torch.Generator().manual_seed(2)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    sides = _synth_calls(gen, n_pairs=4)
    with torch.no_grad():
        m1_i = 0.37 * torch.randn(
            H, MEM, D, generator=torch.Generator().manual_seed(3)).to(DEVICE)
        m2_i = -m1_i
    _trainable_gates(module)

    # ABSOLUTE states (the pre-W19 loss form): identical across pairs
    m1_a, m2_a, m1_b, m2_b = _stacked_states(module, sides, m1_i, m2_i,
                                             delta=False, grad=True)
    loss = ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b, tau=0.07)
    assert float(loss.detach()) == pytest.approx(math.log(4), abs=1e-4)
    loss.backward()
    scale = float(m1_i.abs().max())
    assert float(module.write_gate_k.grad.abs().max()) <= 1e-5 * scale, \
        "identical nonzero states: the saddle residual must be dead"
    assert float(module.write_gate_v.grad.abs().max()) <= 1e-5 * scale


# ======================== 2. the escape =================================
def test_w19f_2_gate_init_escapes_the_saddle():
    """GatesTrainConfig.gate_init (default 0.1) fills both write gates
    before step 0; the escaped point has LIVE gradients (the W19 history
    field > 0) and one step MOVES the gates off the fill value."""
    gen = torch.Generator().manual_seed(4)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    sides = _synth_calls(gen, n_pairs=4)
    m1_i = module.init_state(dtype=torch.float32, device=DEVICE)
    m2_i = module.init_state(dtype=torch.float32, device=DEVICE)
    assert ft.GatesTrainConfig().gate_init == 0.1  # the escape default

    cfg = _patched_train(module, sides, m1_i, m2_i,
                         max_steps=1, batch_pairs=4, lr=5e-2,
                         warmup_steps=1, refresh_every=1, delta_loss=True)
    gk = module.write_gate_k.detach()
    moved = (gk - 0.1).abs().max()
    assert float(moved) > 0.005, "the escape must let one step move the gates"
    assert cfg.history[0]["gk_absmax"] > 0.0, \
        "the grad instrumentation must report a LIVE path"
    assert cfg.history[0]["gv_absmax"] > 0.0


def test_w19f_3_the_loop_descends_the_objective():
    """The controlled rig (layer 0 = the pair signal, layer 1 = the
    shared common component, layer 2 = noise): 30 steps descend the
    InfoNCE well below ln(4) and the signal layer's write weight ends
    above the common layer's — the gates learn, given a nonzero start."""
    gen = torch.Generator().manual_seed(5)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    sides = _synth_calls(gen, n_pairs=4)
    m1_i = module.init_state(dtype=torch.float32, device=DEVICE)
    m2_i = module.init_state(dtype=torch.float32, device=DEVICE)

    cfg = _patched_train(module, sides, m1_i, m2_i,
                         max_steps=30, batch_pairs=4, lr=5e-2,
                         warmup_steps=2, tau=0.07,
                         refresh_every=4, delta_loss=True)
    losses = [h["loss"] for h in cfg.history]
    assert len(losses) == 30
    assert losses[-1] < math.log(4) - 0.5, \
        f"the loop must descend the InfoNCE (last {losses[-1]:.4f} " \
        f"vs ln(4)={math.log(4):.4f})"
    assert losses[-1] < losses[0] - 0.3
    gk = module.write_gate_k.detach()
    assert float(gk[0].abs()) > float(gk[1].abs()), \
        "the pair signal (layer 0) must out-weight the common layer"


# ================= 4. the delta-loss wiring =============================
def test_w19f_4_delta_loss_scores_the_centered_frame():
    """delta_loss=True (default) feeds the InfoNCE the DELTAS
    (m - m_init): the history loss equals the manual delta-InfoNCE;
    delta_loss=False recovers the absolute form (the A/B)."""
    gen = torch.Generator().manual_seed(6)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    with torch.no_grad():
        module.write_gate_k.fill_(0.3)
        module.write_gate_v.fill_(0.3)
    sides = _synth_calls(gen, n_pairs=4)
    with torch.no_grad():
        m1_i = 0.2 * torch.randn(
            H, MEM, D, generator=torch.Generator().manual_seed(7)).to(DEVICE)
        m2_i = torch.zeros_like(m1_i)

    for delta in (True, False):
        cfg = _patched_train(module, sides, m1_i, m2_i,
                             max_steps=1, batch_pairs=4, lr=0.0,
                             refresh_every=1, delta_loss=delta,
                             gate_init=None)
        m1_a, m2_a, m1_b, m2_b = _stacked_states(
            module, sides, m1_i, m2_i, delta=delta, grad=False)
        manual = float(ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b,
                                             tau=cfg.tau))
        assert cfg.history[0]["loss"] == pytest.approx(manual, abs=1e-6), \
            f"delta={delta}: the trainer's loss must be the {'delta' if delta else 'absolute'} form"


# ================= 5. the capture pool ==================================
def test_w19f_5_capture_pool_amortizes_the_prefills():
    """refresh_every=2 halves the prefills (cache_factory calls) over 4
    steps: 4 steps x 4 pairs x 2 sides = 32 at refresh=1 vs 16 at
    refresh=2 — the 22 s/step capture cost amortizes exactly as
    designed, and the W18 replay contract is untouched."""
    from transformers import Qwen3_5TextConfig
    import modeling

    torch.manual_seed(2100)
    cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=4,
        layer_types=list(LAYER_TYPES),
        linear_num_value_heads=H, linear_num_key_heads=2,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128,
        use_m1m2=True, m1m2_mem_size=MEM)
    model = modeling.Qwen3_5TextModel(cfg).eval().to(DEVICE)

    from ingest import prefill_system
    sys_cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    g = torch.Generator().manual_seed(11)
    system = prefill_system(model, torch.randint(
        0, 128, (1, 4), generator=g).to(DEVICE), sys_cache,
        system_ref="w19ft", bits=BITS)

    pairs = [(torch.randint(0, 128, (1, 10), generator=g).to(DEVICE),
              torch.randint(0, 128, (1, 10), generator=g).to(DEVICE))
             for _ in range(16)]

    for refresh, want in ((1, 32), (2, 16)):
        count = {"n": 0}

        def factory():
            count["n"] += 1
            return TQCache(layer_types=LAYER_TYPES, bits=BITS)

        def _batches():
            i = 0
            while True:
                yield [pairs[(i + j) % len(pairs)] for j in range(4)]
                i = (i + 4) % len(pairs)

        c = ft.GatesTrainConfig(max_steps=4, batch_pairs=4, lr=1e-3,
                                log_every=0, refresh_every=refresh)
        ft.train_gates(model, _batches(), system, factory, cfg=c)
        assert count["n"] == want, \
            f"refresh_every={refresh}: {count['n']} prefills, want {want}"
    model.zero_grad(set_to_none=True)


def test_w19f_5b_pool_reuse_is_replay_identical():
    """The same captured calls replay to BIT-IDENTICAL states however
    many times they are reused — the pool's premise (the Function is
    deterministic in (calls, m_init, gates))."""
    gen = torch.Generator().manual_seed(12)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    with torch.no_grad():
        module.write_gate_k.fill_(0.23)
        module.write_gate_v.fill_(-0.11)
    sides = _synth_calls(gen, n_pairs=2)
    m1_i = module.init_state(dtype=torch.float32, device=DEVICE)
    m2_i = module.init_state(dtype=torch.float32, device=DEVICE)
    calls = sides[0][0]
    outs = []
    for _ in range(3):
        m1, m2 = ft.replay_states(module, calls, m1_i, m2_i, memopt=True)
        outs.append((m1.detach().clone(), m2.detach().clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and \
        torch.equal(outs[1][0], outs[2][0])
    assert torch.equal(outs[0][1], outs[1][1]) and \
        torch.equal(outs[1][1], outs[2][1])


# ============== 6. the instrumentation ==================================
def test_w19f_6_history_carries_grad_and_phase_fields():
    """Every history entry: gk_absmax/gv_absmax (the saddle detector the
    box asked for) + capture_s/replay_s/backward_s (the phase timers that
    settle where the 22 s/step went) — and the escaped real-model run
    reports LIVE gradients."""
    from transformers import Qwen3_5TextConfig
    import modeling

    torch.manual_seed(2101)
    cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=4,
        layer_types=list(LAYER_TYPES),
        linear_num_value_heads=H, linear_num_key_heads=2,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128,
        use_m1m2=True, m1m2_mem_size=MEM)
    model = modeling.Qwen3_5TextModel(cfg).eval().to(DEVICE)
    from ingest import prefill_system
    sys_cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    g = torch.Generator().manual_seed(13)
    system = prefill_system(model, torch.randint(
        0, 128, (1, 4), generator=g).to(DEVICE), sys_cache,
        system_ref="w19ft", bits=BITS)
    pairs = [(torch.randint(0, 128, (1, 10), generator=g).to(DEVICE),
              torch.randint(0, 128, (1, 10), generator=g).to(DEVICE))
             for _ in range(4)]

    def _batches():
        for _ in range(2):
            yield pairs

    c = ft.GatesTrainConfig(max_steps=2, batch_pairs=4, lr=1e-3,
                            log_every=0, refresh_every=2)
    c = ft.train_gates(
        model, _batches(), system,
        cache_factory=lambda: TQCache(layer_types=LAYER_TYPES, bits=BITS),
        cfg=c)
    assert len(c.history) == 2
    for h in c.history:
        for key in ("gk_absmax", "gv_absmax", "capture_s",
                    "replay_s", "backward_s"):
            assert key in h and isinstance(h[key], float)
        assert h["gk_absmax"] > 0.0, "the escaped run must report grads"
        assert h["capture_s"] >= 0.0 and h["replay_s"] >= 0.0
        assert h["backward_s"] >= 0.0
    model.zero_grad(set_to_none=True)
