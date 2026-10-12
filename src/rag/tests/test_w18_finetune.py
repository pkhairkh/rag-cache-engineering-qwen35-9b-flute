"""test_w18_finetune.py — the W18 memory-safe replay gates.

THE CONTRACT (the A10G OOM fix, done INSIDE the architecture): the §7
gate trainer's replay re-runs the recorded wiring calls under grad. The
W17 paths (plain autograd chain; torch.utils.checkpoint per call) both
retain a boundary state PER CALL — 24 x 2 sides x batch 8 live fp32
(32, mem, 128) tensors at the production geometries (tens of GiB at
mem_size=4096 — the measured OOM). The additive write (spec §2.2) is
LINEAR in the gates, so the chain's exact Jacobian is identity-plus-
deltas; `_GatedMemoryReplay` (a custom autograd.Function) exploits it:
forward = the module.write chain under no_grad (bit-identical to the
plain chain — same ops, same order), backward recomputes each call's
delta from the recorded k/v ONE AT A TIME (module.write_deltas — the
SAME _scatter_delta the write itself uses) and accumulates the exact
per-layer gate gradients.

Gates:
  1. write_deltas == the write path's own deltas (bit-exact, via a
     gates=1 write onto zero states);
  2. memopt forward BIT-IDENTICAL to the plain chain (torch.equal);
  3. memopt gradients EXACTLY the plain chain's (torch.equal on both
     gate vectors after the same backward);
  4. non-fp32 states -> loud ValueError (the exactness contract);
  5. multi-call layers accumulate gate grads across calls (the
     chain's linearity, not a last-call-only approximation).
"""
from __future__ import annotations

import pytest
import torch
from transformers import Qwen3_5TextConfig

import modeling

import m1m2_finetune as ft
from m1m2 import M1M2
from tq_cache import TQCache
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H, D, MEM = 4, 16, 8
LAYER_TYPES = ["linear_attention", "full_attention"] * 2
BITS = 3.5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(2100)
    m = modeling.Qwen3_5TextModel(Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=4, layer_types=list(LAYER_TYPES),
        linear_num_value_heads=H, linear_num_key_heads=2,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128,
        use_m1m2=True, m1m2_mem_size=MEM)).eval().to(DEVICE)
    with torch.no_grad():
        m.m1m2.write_gate_k.fill_(0.37)
        m.m1m2.write_gate_v.fill_(-0.21)
    return m


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 128, (1, n), generator=g).to(DEVICE)


def _system(model):
    from ingest import prefill_system
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS, s_flat=True)
    return prefill_system(model, _ids(4, 1), cache,
                          system_ref="w18ft", bits=BITS)


def _capture(model, n, seed):
    system = _system(model)
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS, s_flat=True)
    calls, m1_i, m2_i = ft.prefill_capture_calls(model, _ids(n, seed),
                                                 cache, system)
    return calls, m1_i, m2_i


# ============================== 1. write_deltas ============================
def test_w18f_1_write_deltas_match_the_write_path(model):
    """write_deltas(k, v, L) == the delta the write itself adds: a
    gates=1 write onto ZERO states returns exactly (Δ1, Δ2) — compared
    bit-exact (the same _scatter_delta, shared code)."""
    g = torch.Generator().manual_seed(88)
    k = torch.randn(1, H, 9, D, generator=g).to(DEVICE).half()
    v = torch.randn(1, H, 9, D, generator=g).to(DEVICE).half()
    pos = torch.arange(2, 11)
    scratch = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                   num_linear_layers=2).to(DEVICE)
    with torch.no_grad():
        scratch.write_gate_k.fill_(1.0)
        scratch.write_gate_v.fill_(1.0)
        m1_zero = scratch.init_state(dtype=torch.float32)
        m2_zero = scratch.init_state(dtype=torch.float32)
        via_write, _ = scratch.write(k, v, m1_zero, m2_zero, 1,
                                     positions=pos)
        d1, d2 = scratch.write_deltas(k, v, 1, positions=pos,
                                      state=m1_zero)
    # m_new = 0 + 1.0 * delta -> via_write IS the delta (promoted fp32)
    assert torch.equal(via_write, d1)
    assert d1.dtype == torch.float32 and d2.dtype == torch.float32


# ===================== 2. forward parity: bit-identical ====================
def test_w18f_2_memopt_forward_bit_identical(model):
    """The memory-safe replay's forward == the plain autograd chain,
    BIT-EXACT (the same module.write ops in the same order — the W18
    Function changes WHAT IS RETAINED, never the arithmetic)."""
    calls, m1_i, m2_i = _capture(model, 14, 5)
    m1_plain, m2_plain = ft.replay_states(model.m1m2, calls, m1_i, m2_i,
                                          checkpoint=False)
    m1_opt, m2_opt = ft.replay_states(model.m1m2, calls, m1_i, m2_i,
                                      memopt=True)
    assert torch.equal(m1_plain.detach(), m1_opt.detach())
    assert torch.equal(m2_plain.detach(), m2_opt.detach())
    # and the default (unadorned) call IS the memopt path
    m1_def, _ = ft.replay_states(model.m1m2, calls, m1_i, m2_i)
    assert torch.equal(m1_def.detach(), m1_opt.detach())


# ===================== 3. gradient parity: exact ===========================
def test_w18f_3_memopt_gradients_exact(model):
    """Same setup, two paths, one backward each: the gate gradients are
    EXACTLY equal (the custom backward computes the chain's true
    Jacobian — identity to m_init, per-call inner products to the
    gates), and no other parameter receives gradient."""
    def _run_path(memopt):
        calls, m1_i, m2_i = _capture(model, 12, 7)
        model.zero_grad(set_to_none=True)
        m1, m2 = ft.replay_states(model.m1m2, calls, m1_i, m2_i,
                                  memopt=memopt)
        (m1.sum() + m2.sum()).backward()
        gk = model.m1m2.write_gate_k.grad.clone()
        gv = model.m1m2.write_gate_v.grad.clone()
        n_grad = sum(1 for p in model.parameters()
                     if p.grad is not None
                     and float(p.grad.abs().sum()) > 0)
        model.zero_grad(set_to_none=True)
        return gk, gv, n_grad

    gk_plain, gv_plain, n_plain = _run_path(False)
    gk_opt, gv_opt, n_opt = _run_path(True)
    assert torch.equal(gk_plain, gk_opt), \
        f"gate_k grads differ: max {float((gk_plain - gk_opt).abs().max())}"
    assert torch.equal(gv_plain, gv_opt)
    assert n_plain == n_opt == 2  # ONLY the two write-gate vectors


# ===================== 4. the exactness contract ===========================
def test_w18f_4_memopt_refuses_fp16_states(model):
    """Non-fp32 initial states -> a LOUD ValueError (the per-layer fp16
    cast-backs would break the identity Jacobian — the exactness
    contract), never silently inexact gradients."""
    calls, _, _ = _capture(model, 10, 9)
    m1_16 = model.m1m2.init_state(dtype=torch.float16)
    m2_16 = model.m1m2.init_state(dtype=torch.float16)
    with pytest.raises(ValueError, match="fp32 regime"):
        ft.replay_states(model.m1m2, calls, m1_16, m2_16, memopt=True)
    # the plain chain stays available for the reference semantics
    m1, m2 = ft.replay_states(model.m1m2, calls, m1_16, m2_16,
                              memopt=False)
    assert m1.dtype == torch.float16


# ===================== 5. multi-call layer accumulation ====================
def test_w18f_5_gate_grads_accumulate_across_calls():
    """Two recorded calls at the SAME layer ordinal: the gate's gradient
    is the SUM over both calls' inner products (the chain's linearity),
    not the last call's — the property the 24-layer, multi-pass capture
    relies on (e.g. decode steps re-firing a layer)."""
    torch.manual_seed(55)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=1).to(DEVICE)
    with torch.no_grad():
        module.write_gate_k.fill_(0.5)
        module.write_gate_v.fill_(0.5)
    g = torch.Generator().manual_seed(56)
    k1 = torch.randn(1, H, 5, D, generator=g).to(DEVICE)
    v1 = torch.randn(1, H, 5, D, generator=g).to(DEVICE)
    k2 = torch.randn(1, H, 5, D, generator=g).to(DEVICE)
    v2 = torch.randn(1, H, 5, D, generator=g).to(DEVICE)
    calls = [(k1, v1, 0, None), (k2, v2, 0, None)]
    m1_i = module.init_state(dtype=torch.float32)
    m2_i = module.init_state(dtype=torch.float32)

    model_grads = {}
    module.write_gate_k.register_hook(
        lambda grad: model_grads.setdefault("k", grad))
    m1, m2 = ft.replay_states(module, calls, m1_i, m2_i, memopt=True)
    (m1.sum() + m2.sum()).backward()
    grad_memopt = model_grads["k"].clone()

    # the manual exact answer: sum over BOTH calls' deltas, PER HEAD
    # (W21: the gates are (L, H) — the Jacobian row is one inner
    # product per head: <ones[h], Delta[h]> summed over slots+dims)
    module.write_gate_k.grad = None
    with torch.no_grad():
        d1_1, _ = module.write_deltas(k1, v1, 0, state=m1_i)
        d1_2, _ = module.write_deltas(k2, v2, 0, state=m1_i)
        want = d1_1.sum(dim=(1, 2)) + d1_2.sum(dim=(1, 2))
    # grad wrt g[L, h] = <dL/dm1[h], Delta[h]> with dL/dm1 = ones ->
    # the per-head sums over both calls' deltas
    assert torch.allclose(grad_memopt, want.view(1, -1), atol=0, rtol=0)
    # and equal to the plain chain's gradient on the same setup
    module.write_gate_k.grad = None
    m1p, m2p = ft.replay_states(module, calls, m1_i, m2_i,
                                memopt=False)
    (m1p.sum() + m2p.sum()).backward()
    assert torch.equal(module.write_gate_k.grad, grad_memopt)
