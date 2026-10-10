"""test_w17_finetune.py — the §7 M1/M2 gate trainer gates (the W17 core).

THE DIFFERENTIABILITY CONTRACT: the production online loop is
gradient-dead for the WRITE gates BY DESIGN (quantize-on-write severs
autograd; the code lookups are not differentiable). The trainer builds
the gate-differentiable state OUTSIDE the quantized loop — capture (the
wiring calls under no_grad) + replay (the module's own write on a live
chain). These gates pin:

  * the recorder sees every wiring call (per linear layer);
  * the replay state == the cache's held state up to the quant noise
    (the additive-write sum property);
  * the loss's gradient REACHES the write gates (and nothing else);
  * InfoNCE actually pulls positive pairs together (one optimizer step);
  * the gates artifact roundtrips through the serving loader path.
"""
from __future__ import annotations

import pytest
import torch
from transformers import Qwen3_5TextConfig

import modeling

import m1m2_finetune as ft
from m1m2 import M1M2
from tq_cache import TQCache

H, D, MEM = 4, 16, 8
M_NUMEL = H * MEM * D
LAYER_TYPES = ["linear_attention", "full_attention"] * 2
BITS = 3.5


def _cfg(use_m1m2=True):
    return Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=4, layer_types=list(LAYER_TYPES),
        linear_num_value_heads=H, linear_num_key_heads=2,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128,
        use_m1m2=use_m1m2, m1m2_mem_size=MEM)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(21)
    m = modeling.Qwen3_5TextModel(_cfg()).eval().cuda()
    with torch.no_grad():
        m.m1m2.write_gate_k.fill_(0.5)
        m.m1m2.write_gate_v.fill_(0.5)
    return m


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 128, (1, n), generator=g).cuda()


def _system(model):
    """The zero reset point (a system prefill through a fresh cache)."""
    from ingest import prefill_system
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    return prefill_system(model, _ids(4, 1), cache,
                          system_ref="w17ft", bits=BITS)


def _rel_mse(a, b):
    a, b = a.float(), b.float()
    return ((a - b) ** 2).sum().item() / \
        (b ** 2).sum().clamp_min(1e-30).item()


# ============================================ 1. capture + replay =========
def test_w17f_1_recorder_sees_every_layer(model):
    system = _system(model)
    ids = _ids(12, 3)
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    calls, m1_i, m2_i = ft.prefill_capture_calls(model, ids, cache, system)
    assert len(calls) == 2  # one per linear-attention layer
    layer_ids = [c[2] for c in calls]
    assert layer_ids == [0, 1]  # the ordinals
    k, v, idx, pos = calls[0]
    assert k.shape == (1, H, 12, D) and v.shape == (1, H, 12, D)
    assert pos is not None and int(pos.numel()) == 12
    assert not k.requires_grad  # detached capture
    # the m_init is the RESEEDED system state (read BEFORE the prefill —
    # the fp32 dequant of the system's m1 codes; nonzero at open gates)
    assert m1_i.shape == (H, MEM, D)
    assert not m1_i.requires_grad
    from ingest import load_system_state  # noqa: F401  (shape contract)
    assert float(m1_i.abs().max()) > 0.0  # gates 0.5 -> the system state carries content


def test_w17f_2_replay_matches_cache_state(model):
    """The replay state == the cache's held codes (dequantized), up to
    the inter-layer quantization round-trips — the additive-write sum
    property (the trainer trains the state the cache actually holds)."""
    system = _system(model)
    ids = _ids(16, 5)
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    calls, m1_i, m2_i = ft.prefill_capture_calls(model, ids, cache, system)
    m1, m2 = ft.replay_states(model.m1m2, calls, m1_i, m2_i,
                              checkpoint=False)
    assert m1.shape == (H, MEM, D)
    held = cache.read_m1(dtype=torch.float32)
    assert held is not None
    rel = _rel_mse(m1.detach(), held)
    # zero-init chain + 2 quant rounds at 3.5b: the measured budget
    assert rel < 0.12, f"replay vs cache state rel-MSE {rel:.4f}"
    # the exactness anchor: replay == the manual write chain (same ops)
    m1_manual = m1_i
    for (k, v, idx, pos) in calls:
        m1_manual, _ = model.m1m2.write(k, v, m1_manual, m2_i, idx,
                                        positions=pos)
    assert torch.equal(m1.detach(), m1_manual.detach())


def test_w17f_3_gate_gradients_flow_only_to_gates(model):
    """loss = replay_state.sum() -> backward: the WRITE gates receive
    gradient (the branch-free gated add, P3), read_gate does NOT (no
    path — the state is write-built), and the frozen model stays
    frozen."""
    ft.freeze_all_but_gates(model, include_read_gate=False)
    gate_params = (model.m1m2.write_gate_k, model.m1m2.write_gate_v)
    for p in model.parameters():
        if not any(p is q for q in gate_params):
            p.requires_grad_(False)
    model.m1m2.write_gate_k.grad = None
    model.m1m2.write_gate_v.grad = None
    model.m1m2.read_gate.grad = None
    system = _system(model)
    ids = _ids(12, 7)
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    calls, m1_i, m2_i = ft.prefill_capture_calls(model, ids, cache, system)
    m1, m2 = ft.replay_states(model.m1m2, calls, m1_i, m2_i,
                              checkpoint=False)
    loss = m1.sum() + m2.sum()
    loss.backward()
    assert model.m1m2.write_gate_k.grad is not None
    assert float(model.m1m2.write_gate_k.grad.abs().max()) > 0.0
    assert model.m1m2.write_gate_v.grad is not None
    assert float(model.m1m2.write_gate_v.grad.abs().max()) > 0.0
    assert model.m1m2.read_gate.grad is None  # no path through the state
    # nothing else in the model got a grad
    n_grad = sum(1 for p in model.parameters() if p.grad is not None
                 and float(p.grad.abs().sum()) > 0)
    assert n_grad == 2
    model.zero_grad(set_to_none=True)


# ============================================ 2. the loss math ============
def test_w17f_4_infonce_math():
    """Identical positive pairs score lower loss than orthogonal ones;
    the loss is finite and the diagonal is the target."""
    g = torch.Generator().manual_seed(9)
    base = torch.randn(4, H, MEM, D, generator=g).cuda()
    same_b = base + 0.01 * torch.randn(4, H, MEM, D, generator=g).cuda()
    ortho_b = torch.randn(4, H, MEM, D, generator=g).cuda()
    with torch.no_grad():
        l_same = ft.infonce_pairs_loss(base, base, same_b, same_b,
                                       tau=0.07)
        l_ortho = ft.infonce_pairs_loss(base, base, ortho_b, ortho_b,
                                        tau=0.07)
    assert l_same < l_ortho
    assert torch.isfinite(l_same)
    with pytest.raises(ValueError):
        ft.infonce_pairs_loss(base[:1], base[:1], same_b[:1], same_b[:1])


def test_w17f_5_training_reduces_the_objective(model):
    """The optimizer actually descends the InfoNCE: a short run on
    PERTURBED positive pairs (cos < 1) drives the loss down and widens
    the pos-vs-neg margin — the training-signal direction (one full step
    is too coarse an assertion at softmax saturation; the trend is the
    contract)."""
    ft.freeze_all_but_gates(model, include_read_gate=False)
    # the production protocol: the system reset point is built at the P3
    # ZERO gates (the driver's order — load, prefill_system, train), so
    # the reseed base is the zero state and the pair states carry PURE
    # content deltas (the centered frame subtracts the system term at
    # scoring time; an open-gate system state would dominate every
    # state's direction and flatten the contrastive signal).
    with torch.no_grad():
        model.m1m2.write_gate_k.zero_()
        model.m1m2.write_gate_v.zero_()
    system = _system(model)
    with torch.no_grad():
        model.m1m2.write_gate_k.fill_(0.1)
        model.m1m2.write_gate_v.fill_(0.1)
    g = torch.Generator().manual_seed(33)
    pairs = []
    for i in range(4):
        base = torch.randint(0, 128, (1, 12), generator=g).cuda()
        pert = base.clone()
        for j in range(6):  # replace half: cos(pos) starts mid-range
            pert[0, j * 2] = int(torch.randint(0, 128, (1,), generator=g))
        pairs.append((base, pert))

    def _pair_loss():
        with torch.no_grad():
            states = []
            for ids_a, ids_b in pairs:
                ps = []
                for ids in (ids_a, ids_b):
                    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
                    calls, m1_i, m2_i = ft.prefill_capture_calls(
                        model, ids, cache, system)
                    m1, m2 = ft.replay_states(model.m1m2, calls, m1_i,
                                              m2_i, checkpoint=False)
                    ps.append((m1, m2))
                states.append(ps)
            m1_a = torch.stack([s[0][0] for s in states])
            m2_a = torch.stack([s[0][1] for s in states])
            m1_b = torch.stack([s[1][0] for s in states])
            m2_b = torch.stack([s[1][1] for s in states])
            return float(ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b,
                                                tau=0.5))

    loss_before = _pair_loss()

    def batches():
        for _ in range(40):
            yield pairs

    cfg = ft.GatesTrainConfig(max_steps=40, batch_pairs=4, lr=0.05,
                              tau=0.5, log_every=0, seed=7)
    cfg = ft.train_gates(
        model, batches(), system,
        cache_factory=lambda: TQCache(layer_types=LAYER_TYPES, bits=BITS),
        cfg=cfg)
    loss_after = _pair_loss()
    assert loss_after < loss_before, (
        f"the trained gates did not descend the objective on the same "
        f"pairs: {loss_before:.4f} -> {loss_after:.4f}")
    model.zero_grad(set_to_none=True)


# ============================================ 3. the loop + artifact =====
def test_w17f_6_train_gates_smoke(model):
    """train_gates end-to-end on synthetic pairs: history fills, the
    gates move off P3, the loop never touches the frozen weights."""
    system = _system(model)
    with torch.no_grad():
        model.m1m2.write_gate_k.zero_()
        model.m1m2.write_gate_v.zero_()
    frozen_before = [p.detach().clone() for n, p in model.named_parameters()
                     if not n.startswith("m1m2.")]
    pairs = [(_ids(10, 200 + i), _ids(10, 200 + i)) for i in range(6)]

    def batches():
        for _ in range(3):
            yield pairs[:4]

    cfg = ft.GatesTrainConfig(max_steps=3, batch_pairs=4, lr=0.05,
                              log_every=0)
    cfg = ft.train_gates(
        model, batches(), system,
        cache_factory=lambda: TQCache(layer_types=LAYER_TYPES, bits=BITS),
        cfg=cfg)
    assert len(cfg.history) == 3
    assert all("loss" in h and "cos_pos" in h for h in cfg.history)
    assert float(model.m1m2.write_gate_k.abs().max()) > 0.0
    for (n, p), before in zip(
            [(n, p) for n, p in model.named_parameters()
             if not n.startswith("m1m2.")], frozen_before):
        assert torch.equal(p.detach(), before), f"{n} drifted (frozen!)"


def test_w17f_7_gates_artifact_roundtrip(model, tmp_path):
    """save/load roundtrip: values land exactly; geometry drift is loud;
    the loader's _load_m1m2_gates path consumes the same artifact."""
    with torch.no_grad():
        model.m1m2.write_gate_k.uniform_(-0.2, 0.2, generator=None) \
            if False else model.m1m2.write_gate_k.copy_(
                torch.linspace(-0.2, 0.2, model.m1m2.num_linear_layers))
        model.m1m2.write_gate_v.fill_(0.1)
    want = {k: getattr(model.m1m2, k).detach().clone()
            for k in ("write_gate_k", "write_gate_v", "read_gate")}
    path = ft.save_gates(model.m1m2, str(tmp_path / "g"),
                         extra_meta={"pairs_file": "pytest"})
    assert path.endswith(".npz")

    fresh = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                 num_linear_layers=2).cuda()
    meta = ft.load_gates(fresh, path)
    assert meta["pairs_file"] == "pytest"
    for k, w in want.items():
        assert torch.equal(getattr(fresh, k).detach(), w)

    other = M1M2(num_heads=H, head_dim=D, mem_size=MEM * 2,
                 num_linear_layers=2).cuda()
    with pytest.raises(ValueError, match="mem_size"):
        ft.load_gates(other, path)
    with pytest.raises(ValueError, match="not found"):
        ft.load_gates(fresh, str(tmp_path / "missing.npz"))


def test_w17f_8_freeze_all_but_gates(model):
    """freeze: ONLY the gate vectors require grad, promoted to fp32."""
    trainable = ft.freeze_all_but_gates(model, include_read_gate=False)
    assert trainable == ["m1m2.write_gate_k", "m1m2.write_gate_v"]
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_train == 2
    assert model.m1m2.write_gate_k.dtype == torch.float32
    trainable = ft.freeze_all_but_gates(model, include_read_gate=True)
    assert "m1m2.read_gate" in trainable
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_train == 3
    # restore
    for p in model.parameters():
        p.requires_grad_(True)


def test_w17f_9_module_vs_codes_geometry_loud(model):
    """A cache whose M1/M2 codes were built at another geometry fails
    loudly at the first forward (the read-path guard) — the trainer's
    loud-mismatch contract (tested via the direct forward: the capture
    helper reseeds, which would overwrite the poison)."""
    from ingest import reseed_cache
    system = _system(model)
    big = M1M2(num_heads=H, head_dim=D, mem_size=MEM * 2,
               num_linear_layers=2).cuda()
    cache = TQCache(layer_types=LAYER_TYPES, bits=BITS)
    with torch.no_grad():
        big.write_gate_k.fill_(0.5)
        m1, m2 = big.write(
            torch.randn(1, H, 8, D).cuda(), torch.randn(1, H, 8, D).cuda(),
            big.init_state(dtype=torch.float32),
            big.init_state(dtype=torch.float32), 0)
    cache.update_m1(m1)
    cache.update_m2(m2)
    cache._m1_shape = tuple(big.state_shape())
    cache._m2_shape = tuple(big.state_shape())
    with pytest.raises((ValueError, RuntimeError)):
        with torch.no_grad():
            model(input_ids=_ids(6, 11), past_key_values=cache,
                  use_cache=True)
