"""test_w22_hardneg_topk.py — W22: the sparse read + the curated hard
negatives (the two proven-memory adoptions of the wave).

CONTEXT (the box's W21.6 run): the content-addressed write finally
trained REAL discrimination signal (spread +0.20..0.35, no plateau) —
but the held-out ladder OOM'd on every row and the script crashed with
UnboundLocalError after it (`del val_group` unbinding the name the
trailing report reads), so generalization was never MEASURED; and the
B=8 InfoNCE swung its loss 0.28..1.64 on seven random in-batch draws
while 17.5k curated label-0 non-pairs sat unused.

The two module-side adoptions this wave pins (both from the reference
key-value-memory lineage the W21 wave already cites):

  * read_topk (m1m2.M1M2): the softmax reads the TOP-k M1 slots only —
    Product-Key Memory's top-32 (Lample et al. 2019, arXiv:1907.05242),
    Memorizing Transformers' exact kNN (Wu et al. 2022, arXiv:2203.08913),
    RETRO's top-k chunks (Borgeaud et al. 2022, arXiv:2112.04426). A
    dense softmax over an accumulating memory is a near-uniform average;
    every proven design retrieves sparsely. None (the default) keeps
    the spec §2.2 dense contract bit-identically.
  * the curated hard-negative columns (m1m2_finetune): the loss logits
    gain N shared negative columns sim(a_i, n_l) that are never targets
    — DPR's in-batch + hard negatives (Karpukhin et al. 2020,
    arXiv:2004.04906), MoCo's queue (He et al. 2020, arXiv:1911.05722),
    Robinson et al. 2021 (arXiv:2011.12031) for the small-batch
    shortcut analysis. neg_pairs=0 must be the EXACT W21 regime.

Gates (deterministic, pinned seeds, fp32, device-agnostic — CUDA when
available else CPU; no network, no transformers):

  A. read_topk DEFAULT IS THE SPEC CONTRACT: a freshly built module has
     read_topk None and its read() == the manual dense reference
     (softmax(q @ M1^T / sqrt(D)) @ M2 * gate) exactly.
  B. k = mem_size == dense (up to softmax summation order); the top-k
     probs live ONLY on the top-k slots (EXACT zeros elsewhere — probed
     through a one-hot M2, which turns the read output INTO the prob
     row), each row sums to 1.
  C. k = 1 returns EXACTLY the argmax slot's M2 row (the one-hot
     multiply is exact in fp32).
  D. P3 at any k: zero memories read exact zeros.
  E. LOUD GUARDS: construction and post-init assignment both validate
     (bool/0/negative/> mem/non-int raise; None and [1, mem] pass).
  F. THE HARD-NEGATIVE LOSS, OFF: bit-identical to the W21 loss (the
     4-arg call == the explicit None call), and equal to a manual
     cross-entropy over pair_sim_matrix / tau.
  G. THE HARD-NEGATIVE LOSS, ON: a HARD negative (near-duplicate of an
     anchor's pair) raises the loss above an EASY one; gradients reach
     the negative states (m1_n.grad finite/non-None) and one SGD step
     on the anchors+negatives PUSHES APART (the mean sim(a, n)
     strictly decreases).
  H. THE LOSS'S OWN GUARDS: m1_n without m2_n raises; N=0 raises;
     mismatched row counts raise.
  I. THE METRIC: _pair_sim_stats reports cos_hneg == the mean of the
     shared pair_sim_matrix(a, neg) when negatives ride, NaN when not.
  J. THE TRAINER (the W19 shim harness, extended): a negatives feed of
     single-side batches trains — history carries finite cos_hneg, the
     capture log names the neg sides; iterator exhaustion mid-run warns
     once and falls back to the W21 regime (NaN cos_hneg afterwards);
     and cfg.neg_pairs=0 with a live iterator is bit-identical to
     negatives=None (the W21 regime preserved by construction).
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch

import m1m2_finetune as ft
from m1m2 import M1M2

# Device for all tests (the GPU box runs CUDA; CPU only if unavailable)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- the read-test geometry: D == MEM so a one-hot M2 turns the read
# output INTO the probability row (out[b, h, t, s] = probs[..., s])
RH, RD, RMEM, RNLAY = 2, 8, 8, 2
RT = 5

# ---- the trainer-test geometry (the W19 harness's)
H, D, MEM, NLAY = 4, 16, 8, 3
B = 4


def _read_fixture(seed: int = 7):
    """A module + (q, m1, m2) with D == MEM == 8: m2 is the identity
    (one-hot rows), so read() == the probability row itself."""
    gen = torch.Generator().manual_seed(seed)
    mod = M1M2(num_heads=RH, head_dim=RD, mem_size=RMEM,
               num_linear_layers=RNLAY).to(DEVICE)
    q = torch.randn(1, RH, RT, RD, generator=gen).to(DEVICE)
    m1 = torch.randn(RH, RMEM, RD, generator=gen).to(DEVICE)
    m2 = torch.eye(RMEM, device=DEVICE).unsqueeze(0).expand(RH, -1, -1)
    return mod, q, m1, m2.contiguous()


def _dense_reference(mod, q, m1, m2, layer_idx: int = 0):
    """The spec §2.2 read, written out by hand (fp32, /sqrt(D), the
    per-layer gate multiply, cast back to q.dtype)."""
    scores = torch.matmul(q.to(torch.float32),
                          m1.to(torch.float32).transpose(-1, -2))
    scores = scores / math.sqrt(mod.head_dim)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, m2.to(torch.float32))
    out = out * mod.read_gate[layer_idx]
    return out.to(q.dtype)


# ================= A. the default is the spec §2.2 contract ==========
def test_w22a_default_read_is_the_dense_spec_contract():
    mod, q, m1, m2 = _read_fixture()
    assert mod.read_topk is None, "the default must be the dense contract"
    out = mod.read(q, m1, m2, layer_idx=0)
    ref = _dense_reference(mod, q, m1, m2)
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=0.0, rtol=0.0), \
        "the None path must be bit-identical to the manual dense read"
    assert "read_topk=None" in repr(mod), \
        "extra_repr must name the knob (the box's regime reporting)"


# ================= B. k = mem == dense; exact zeros off the top-k ====
def test_w22b_topk_full_and_support():
    mod, q, m1, m2 = _read_fixture()
    mod.read_topk = RMEM                       # k = mem
    out = mod.read(q, m1, m2, layer_idx=0)
    ref = _dense_reference(mod, q, m1, m2)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6), \
        "k = mem_size is the dense softmax up to summation order"

    # k = 2: with the one-hot m2 the OUTPUT IS the prob row — exactly
    # two nonzero entries, summing to 1, sitting on the top-2 slots
    mod.read_topk = 2
    out2 = mod.read(q, m1, m2, layer_idx=0).to(torch.float32)
    nz = (out2 != 0).sum(dim=-1)
    assert torch.equal(nz, torch.full_like(nz, 2)), \
        "the top-k read must put EXACT zeros on the unselected slots"
    sums = out2.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        "the top-k softmax still normalizes"
    scores = torch.matmul(q.to(torch.float32),
                          m1.to(torch.float32).transpose(-1, -2))
    scores = scores / math.sqrt(mod.head_dim)
    want_idx = torch.topk(scores, 2, dim=-1).indices
    got_idx = out2.argmax(dim=-1)  # the largest prob slot
    assert (got_idx == want_idx[..., 0]).all(), \
        "the top-1 of the sparse read must be the score argmax"


# ================= C. k = 1 is the exact argmax retrieval ============
def test_w22c_topk1_is_exact_argmax_row():
    mod, q, m1, m2 = _read_fixture()
    # rig m2 with DISTINCT rows (not one-hot) so the retrieved row is
    # identifiable by value; k=1 -> probs one-hot -> out == m2[argmax]
    gen = torch.Generator().manual_seed(11)
    m2 = torch.randn(RH, RMEM, RD, generator=gen).to(DEVICE)
    mod.read_topk = 1
    out = mod.read(q, m1, m2, layer_idx=0)
    scores = torch.matmul(q.to(torch.float32),
                          m1.to(torch.float32).transpose(-1, -2))
    scores = scores / math.sqrt(mod.head_dim)
    am = scores.argmax(dim=-1)                       # (B, H, T)
    rows = torch.gather(
        m2.to(torch.float32).unsqueeze(0).unsqueeze(2).expand(
            1, RH, RT, RMEM, RD), 3, am[..., None, None].expand(
            1, RH, RT, 1, RD)).squeeze(3)
    assert torch.equal(out.to(torch.float32), rows), \
        "k=1 must return the argmax slot's M2 row EXACTLY (one-hot matmul)"


# ================= D. P3: zero memories read exact zeros =============
def test_w22d_zero_memories_topk_noop():
    mod, q, _, _ = _read_fixture()
    mod.read_topk = 3
    z = torch.zeros_like(mod.init_state(dtype=torch.float32,
                                        device=DEVICE))
    out = mod.read(q, z, z.clone(), layer_idx=1)
    assert torch.equal(out, torch.zeros_like(out)), \
        "P3: reads of zero memories are exact zeros at ANY k"


# ================= E. loud guards (init + post-init) =================
def test_w22e_read_topk_loud_guards():
    def build(k):
        return M1M2(num_heads=RH, head_dim=RD, mem_size=RMEM,
                    num_linear_layers=RNLAY, read_topk=k)
    with pytest.raises(TypeError):
        build(True)                    # bool is not an int knob
    with pytest.raises(ValueError):
        build(0)                       # below 1
    with pytest.raises(ValueError):
        build(RMEM + 1)                # above mem_size
    with pytest.raises(TypeError):
        build("32")                    # a string is not a python int
    mod = build(4)                     # a legal k
    assert mod.read_topk == 4
    mod.read_topk = None               # post-init reset to dense
    assert mod.read_topk is None
    # post-init BAD values raise AT READ TIME (the loudness contract
    # the write knobs get at _addr_geometry)
    mod.read_topk = RMEM + 5
    _, q, m1, m2 = _read_fixture()
    with pytest.raises(ValueError):
        mod.read(q, m1, m2, layer_idx=0)
    mod.read_topk = 1.5                # float, not int
    with pytest.raises(TypeError):
        mod.read(q, m1, m2, layer_idx=0)
    mod.read_topk = 2                  # restored: reads fine
    mod.read(q, m1, m2, layer_idx=0)


# ================= F. the loss OFF is the exact W21 loss =============
def _loss_states(seed: int = 21, n: int = B):
    """B (pair-of-two-states) operands as flat differentiable leaves:
    (m1_a, m2_a, m1_b, m2_b) each (n, 3*7) — flat_unit works on any
    (n, ...) fp32 leaves."""
    gen = torch.Generator().manual_seed(seed)
    def _lv():
        return torch.randn(n, 3, 7, generator=gen).to(DEVICE)
    return _lv(), _lv(), _lv(), _lv()


def test_w22f_loss_off_is_bit_identical():
    m1_a, m2_a, m1_b, m2_b = _loss_states()
    m1_n, m2_n = _loss_states(seed=22)
    plain = ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b, tau=0.07)
    explicit = ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b, tau=0.07,
                                     m1_n=None, m2_n=None)
    assert torch.equal(plain, explicit), \
        "None negatives must be the bit-identical W21 code path"
    sim = ft.pair_sim_matrix(m1_a, m2_a, m1_b, m2_b)
    ref = torch.nn.functional.cross_entropy(
        sim / 0.07, torch.arange(B, device=sim.device))
    assert torch.equal(plain, ref), \
        "the 4-arg call is the manual cross-entropy over the shared sim"


# ================= G. hard negatives bite (loss + gradient) ==========
def test_w22g_hard_negative_raises_and_pushes_apart():
    m1_a, m2_a, m1_b, m2_b = _loss_states()
    # an EASY negative: independent noise; a HARD negative: a near-copy
    # of anchor 0's b-side (cos ~ 1 against the pair it tries to beat)
    easy_m1, easy_m2 = _loss_states(seed=23)
    hard_m1 = m1_b[0:1].detach().clone().requires_grad_(True)
    hard_m2 = m2_b[0:1].detach().clone().requires_grad_(True)
    easy_loss = ft.infonce_pairs_loss(
        m1_a, m2_a, m1_b, m2_b, tau=0.07,
        m1_n=easy_m1.detach(), m2_n=easy_m2.detach())
    hard_loss = ft.infonce_pairs_loss(
        m1_a, m2_a, m1_b.detach(), m2_b.detach(), tau=0.07,
        m1_n=hard_m1, m2_n=hard_m2)
    assert float(hard_loss) > float(easy_loss), \
        "a near-duplicate negative must be HARDER (higher loss)"

    # gradients reach the negative states (the push-away path)
    hard_loss.backward()
    assert hard_m1.grad is not None and hard_m2.grad is not None
    assert torch.isfinite(hard_m1.grad).all()

    # one SGD step on the anchors + the negative PUSHES APART: the mean
    # sim(a, n) strictly decreases after the step
    leaves = [t.detach().clone().requires_grad_(True)
              for t in (m1_a, m2_a)]
    neg = [hard_m1.detach().clone().requires_grad_(True),
           hard_m2.detach().clone().requires_grad_(True)]
    before = float(ft.pair_sim_matrix(leaves[0], leaves[1],
                                      neg[0], neg[1]).mean())
    for _ in range(20):
        loss = ft.infonce_pairs_loss(
            leaves[0], leaves[1], m1_b.detach(), m2_b.detach(),
            tau=0.07, m1_n=neg[0], m2_n=neg[1])
        for t in leaves + neg:
            if t.grad is not None:
                t.grad = None
        loss.backward()
        with torch.no_grad():
            for t in leaves + neg:
                t.add_(t.grad, alpha=-0.5)
    after = float(ft.pair_sim_matrix(leaves[0], leaves[1],
                                     neg[0], neg[1]).mean())
    assert after < before - 1e-3, \
        f"descending must push anchors and curated negatives apart " \
        f"({before:+.4f} -> {after:+.4f})"


# ================= H. the loss's own guards ==========================
def test_w22h_loss_negative_guards():
    m1_a, m2_a, m1_b, m2_b = _loss_states()
    m1_n, _m2_n = _loss_states(seed=24)
    with pytest.raises(ValueError):
        ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b,
                              m1_n=m1_n, m2_n=None)      # half a pair
    with pytest.raises(ValueError):
        ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b,
                              m1_n=m1_n[:0], m2_n=_m2_n[:0])  # N = 0
    with pytest.raises(ValueError):
        ft.infonce_pairs_loss(m1_a, m2_a, m1_b, m2_b,
                              m1_n=m1_n, m2_n=_m2_n[:2])  # N mismatch


# ================= I. the metric's curated column ====================
def test_w22i_stats_carry_cos_hneg():
    m1_a, m2_a, m1_b, m2_b = _loss_states()
    m1_n, m2_n = _loss_states(seed=25)
    st_off = ft._pair_sim_stats(m1_a, m2_a, m1_b, m2_b)
    assert math.isnan(st_off["cos_hneg"]), \
        "no negatives -> cos_hneg is NaN (the W21 regime's marker)"
    st_on = ft._pair_sim_stats(m1_a, m2_a, m1_b, m2_b,
                               m1_n=m1_n, m2_n=m2_n)
    want = float(ft.pair_sim_matrix(m1_a, m2_a, m1_n, m2_n).mean())
    assert st_on["cos_hneg"] == pytest.approx(want, abs=1e-6), \
        "cos_hneg is the shared-site mean over the negative columns"
    for k in ("cos_pos", "cos_neg", "spread", "off_max", "diag_min"):
        assert st_on[k] == pytest.approx(st_off[k], abs=1e-7), \
        f"{k} must not move when negatives ride (the metric is additive)"


# ================= J. the trainer (the W19 shim, extended) ===========
class _ModelShim:
    """The `model` train_gates needs: .m1m2 + .parameters()."""

    def __init__(self, module: M1M2):
        self.m1m2 = module

    def parameters(self, recurse=True):
        return self.m1m2.parameters(recurse)


def _synth_sides(gen, n_sides: int, common: float = 3.0):
    """Capture stand-ins (B=1 ranks, per-call (k, v, layer, None)):
    side i's signal layer key = base_i + 2% jitter (pair-mateable),
    layer 1 the SAME common key for every side (the W16 pedestal),
    layer 2 small noise. Mirrors test_w19's _synth_calls generator."""
    base = torch.randn(n_sides, H, 6, D, generator=gen)
    common_k = common * torch.randn(1, H, 6, D, generator=gen)
    sides = []
    for i in range(n_sides):
        calls = []
        for lay in range(NLAY):
            if lay == 0:
                k = base[i].unsqueeze(0).clone()
            elif lay == 1:
                k = common_k.clone()
            else:
                k = 0.1 * torch.randn(1, H, 6, D, generator=gen)
            v = 0.5 * k
            calls.append((k.half(), v.half(), lay, None))
        sides.append(calls)
    return sides


def _patched_train(module, pos_sides, neg_sides, n_neg: int,
                   m1_i, m2_i, **cfg_kw):
    """train_gates against SYNTHETIC sides: prefill_capture_calls is
    patched to hand out the recorded sides in the trainer's exact
    consumption order (2*batch_pairs pos sides then n_neg neg sides
    per pool — the pool block's order), cycling. `neg_batches` yields
    the negatives feed (batches of single ids; the fake capture ignores
    ids). Returns (cfg, train_gates' captured stdout helper state)."""
    shim = _ModelShim(module)
    for p in module.parameters():
        p.requires_grad_(False)
    module.write_gate_k.requires_grad_(True)
    module.write_gate_v.requires_grad_(True)
    m1_i = m1_i.to(torch.float32)
    m2_i = m2_i.to(torch.float32)
    supply = itertools.cycle(list(pos_sides) + list(neg_sides))
    state = {"pops": 0}

    def _fake_capture(model, ids, cache, system):
        state["pops"] += 1
        return next(supply), m1_i, m2_i

    def _batches():
        while True:
            yield [(None, None)] * 4

    def _neg_batches():
        while True:
            yield [None] * n_neg

    orig = ft.prefill_capture_calls
    ft.prefill_capture_calls = _fake_capture
    try:
        cfg = ft.GatesTrainConfig(log_every=1, neg_pairs=n_neg, **cfg_kw)
        cfg = ft.train_gates(shim, _batches(), system=None,
                             cache_factory=lambda: None, cfg=cfg,
                             negatives=(_neg_batches()
                                        if n_neg else None))
        return cfg, state
    finally:
        ft.prefill_capture_calls = orig


def test_w22j_trainer_runs_with_curated_negatives():
    gen = torch.Generator().manual_seed(31)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    m1_i = torch.zeros(H, MEM, D, device=DEVICE)
    m2_i = torch.zeros(H, MEM, D, device=DEVICE)
    pos = _synth_sides(gen, 8)                 # 4 pairs x 2 sides
    neg = _synth_sides(gen, 4)                 # 4 curated sides
    cfg, state = _patched_train(
        module, pos, neg, n_neg=4, m1_i=m1_i, m2_i=m2_i,
        max_steps=3, refresh_every=1, gate_init=0.1)
    assert len(cfg.history) == 3
    for h in cfg.history:
        assert math.isfinite(h["cos_hneg"]), \
            "every step with negatives in the pool carries a finite cos_hneg"
    # 3 steps at refresh 1 = 3 pools = 3 x (8 pos + 4 neg) fake pops
    assert state["pops"] == 3 * (8 + 4)


def test_w22j2_negative_exhaustion_falls_back_loudly(capsys):
    gen = torch.Generator().manual_seed(32)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    m1_i = torch.zeros(H, MEM, D, device=DEVICE)
    m2_i = torch.zeros(H, MEM, D, device=DEVICE)
    pos = _synth_sides(gen, 8)
    neg = _synth_sides(gen, 4)

    shim = _ModelShim(module)
    for p in module.parameters():
        p.requires_grad_(False)
    module.write_gate_k.requires_grad_(True)
    module.write_gate_v.requires_grad_(True)
    supply = itertools.cycle(list(pos) + list(neg))

    def _fake_capture(model, ids, cache, system):
        return next(supply), m1_i, m2_i

    def _batches():
        while True:
            yield [(None, None)] * 4

    def _one_neg_batch():
        yield [None] * 4
        # then StopIteration forever after

    orig = ft.prefill_capture_calls
    ft.prefill_capture_calls = _fake_capture
    try:
        cfg = ft.GatesTrainConfig(log_every=0, max_steps=4,
                                  refresh_every=1, gate_init=0.1,
                                  neg_pairs=4)
        cfg = ft.train_gates(shim, _batches(), system=None,
                             cache_factory=lambda: None, cfg=cfg,
                             negatives=_one_neg_batch())
    finally:
        ft.prefill_capture_calls = orig
    out = capsys.readouterr().out
    assert "the negatives iterator is exhausted" in out, \
        "exhaustion warns exactly once, loudly"
    assert len(cfg.history) == 4
    assert math.isfinite(cfg.history[0]["cos_hneg"]), \
        "pool 1 carried the curated columns"
    assert math.isnan(cfg.history[1]["cos_hneg"]), \
        "pools after exhaustion fall back to the W21 regime"


def test_w22j3_neg_pairs_zero_is_the_w21_regime_bit_identical():
    gen = torch.Generator().manual_seed(33)
    mods = []
    for _ in range(2):
        m = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                 num_linear_layers=NLAY).to(DEVICE)
        with torch.no_grad():
            m.write_gate_k.fill_(0.1)
            m.write_gate_v.fill_(0.1)
        mods.append(m)
    m1_i = torch.zeros(H, MEM, D, device=DEVICE)
    m2_i = torch.zeros(H, MEM, D, device=DEVICE)
    pos = _synth_sides(gen, 8)
    neg = _synth_sides(gen, 4)

    # run A: no negatives at all; run B: a LIVE iterator but
    # neg_pairs=0 — the documented bit-identical contract
    cfg_a, _ = _patched_train(mods[0], pos, neg, n_neg=0,
                              m1_i=m1_i, m2_i=m2_i,
                              max_steps=4, refresh_every=1, gate_init=None)

    def _live():
        while True:
            yield [None] * 4

    shim = _ModelShim(mods[1])
    for p in mods[1].parameters():
        p.requires_grad_(False)
    mods[1].write_gate_k.requires_grad_(True)
    mods[1].write_gate_v.requires_grad_(True)
    supply = itertools.cycle(list(pos) + list(neg))

    def _fake_capture(model, ids, cache, system):
        return next(supply), m1_i, m2_i

    def _batches():
        while True:
            yield [(None, None)] * 4

    orig = ft.prefill_capture_calls
    ft.prefill_capture_calls = _fake_capture
    try:
        cfg_b = ft.GatesTrainConfig(log_every=0, max_steps=4,
                                    refresh_every=1, gate_init=None,
                                    neg_pairs=0)
        cfg_b = ft.train_gates(shim, _batches(), system=None,
                               cache_factory=lambda: None, cfg=cfg_b,
                               negatives=_live())
    finally:
        ft.prefill_capture_calls = orig
    assert torch.equal(mods[0].write_gate_k, mods[1].write_gate_k) \
        and torch.equal(mods[0].write_gate_v, mods[1].write_gate_v), \
        "neg_pairs=0 with a live iterator must train bit-identically"
    assert all(math.isnan(h["cos_hneg"]) for h in cfg_a.history) \
        and all(math.isnan(h["cos_hneg"]) for h in cfg_b.history)
