"""test_w20_15_regime.py — the W20.15 fine-tune-regime gates (box-run).

THE RUN THIS WAVE ANSWERS (the box's post-W20.14 300-step re-run): the
W20.14 metric fix held — cos(neg) finally differed from cos(pos) — and
what it saw was a SAWTOOTH. The loss block means ranged 0.40..2.98
(chance ln 8 = 2.079) with the BETWEEN-block variance 7,595x the
WITHIN-block variance: the block difficulty IS the capture batch (its
8 pairs' own cross-side alignment — the in-batch-negative background
cos_neg is stable at ~0.35 while cos_pos swings 0.35..0.74), the
pre-clip |gv| swung 160x with the batch, only 600 of 14,469 train
pairs were consumed (4.1%), and the v1 `_cycle_batches` iterator FIXED
every pair's in-batch negatives to its file-layout neighbors forever.
Gates here:

  1. EPOCH_BATCHES — the per-epoch re-batching: epoch 2 draws fresh
     batches (not epoch 1's), the order is deterministic per seed,
     a different seed re-orders, every batch is full-size (the wrap
     tail), every batch holds DISTINCT pairs (the wrap's
     disjointness — no pair is its own in-batch negative), and one
     epoch covers the whole pair set.
  2. THE CONTRACTS — batch < 2 / bool batch / a single pair refuse
     loudly (the InfoNCE's own B >= 2 law); n <= batch yields one
     unduplicated batch.
  3. LEXICAL_OVERLAP — the surface-alignment containment over
     lowercased word MULTISETS: |A∩B| / min(|A|, |B|).
  4. BUCKETED_SPREAD — the held-out report's core: per-bucket
     diagonal means against the overall off-diagonal background; a
     rigged rig (half the pairs high-similarity, half independent)
     must split the buckets apart while the overall row carries the
     W20.14 fields verbatim; a label-count mismatch refuses loudly.
  5. THE HISTORY FIELDS — every entry carries the capture-batch
     ordinal (the block id the loss sawtooths on) and the gate VALUE
     stats (gk/gv min/max/absmean — WHERE the gates are; the box's
     300-step run logged 300 gradient magnitudes and zero parameter
     locations); the final entry's values equal the module's actual
     gates.

GPU-ONLY note (the standing law): AUTHORED on the dev box, PENDING on
the A10G box — this file imports repo modules and executes torch, so
it runs on the box, never on the dev box.
"""
from __future__ import annotations

from itertools import islice

import pytest
import torch

import m1m2_finetune as ft
from m1m2 import M1M2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H, D, MEM, NLAY = 4, 16, 8, 3


# ---------------------------------------------- epoch_batches contracts --
def _take_epochs(pairs, batch, seed, epochs):
    """The first `epochs` epochs of ft.epoch_batches as lists of
    batches (a fresh generator per call — determinism checks compare
    two INDEPENDENT generators)."""
    out = []
    gen = ft.epoch_batches(pairs, batch, seed=seed)
    per_epoch = (len(pairs) + batch - 1) // batch if pairs else 0
    for _ in range(epochs):
        out.append([tuple(p) for p in islice(gen, per_epoch)])
    return out


def test_w20_15a_epoch_batches_rebatch_per_epoch():
    """The v1 defect, pinned: one fixed order forever meant a pair's
    in-batch negatives were its file-layout neighbors for the whole
    run. epoch_batches must draw FRESH compositions per epoch —
    deterministic per seed, re-ordered by a different seed."""
    pairs = [(i, -i) for i in range(20)]
    e1, e2 = _take_epochs(pairs, 5, seed=1234, epochs=2)
    assert len(e1) == 4 and all(len(b) == 5 for b in e1), \
        "one epoch = ceil(20/5) full batches"
    assert e1 != e2, "epoch 2 must re-batch (the v1 cycle repeated " \
                     "the same neighbor sets forever)"
    assert sorted(x for b in e1 for x in b) == pairs, \
        "one epoch covers every pair exactly once (no wrap at n%b==0)"

    # determinism: a fresh generator with the same seed replays exactly
    assert _take_epochs(pairs, 5, seed=1234, epochs=2) == [e1, e2]
    # a different seed re-orders
    assert _take_epochs(pairs, 5, seed=7, epochs=1)[0] != e1


def test_w20_15b_epoch_batches_contracts():
    """The loud refusals + the wrap's disjointness: every batch
    full-size, every batch of DISTINCT pairs (no pair is its own
    in-batch negative — the wrap takes indices disjoint from the tail
    slice), full coverage with exactly batch-1 duplicated members
    across the epoch boundary, n <= batch as ONE batch."""
    pairs = [(i, 1000 + i) for i in range(10)]
    (e1,) = _take_epochs(pairs, 4, seed=1234, epochs=1)
    assert len(e1) == 3 and all(len(b) == 4 for b in e1), \
        "ceil(10/4)=3 batches, the wrap tail fills the last to 4"
    for b in e1:
        assert len(set(b)) == len(b) == 4, \
            "no pair twice in ONE batch (the wrap is disjoint)"
    flat = [x for b in e1 for x in b]
    assert set(flat) == set(pairs), "full coverage"
    assert len(flat) - len(set(flat)) == 4 - (10 - 8), \
        "exactly batch - tail_len duplicated members straddle the " \
        "epoch boundary (harmless: their batches' OTHER members differ)"

    # n <= batch: the whole set as ONE batch, no wrap duplication
    (small,) = _take_epochs([(0, 1), (2, 3), (4, 5)], 8, seed=5,
                            epochs=1)
    assert len(small) == 1 and len(set(small[0])) == 3

    # the loud refusals (the InfoNCE's own B >= 2 law)
    with pytest.raises(ValueError):
        ft.epoch_batches(pairs, 1)
    with pytest.raises(ValueError):
        ft.epoch_batches(pairs, True)     # a bool is not a batch size
    with pytest.raises(ValueError):
        ft.epoch_batches([(0, 1)], 4)     # one pair cannot form a batch


# ------------------------------------------- lexical_overlap contracts ---
def test_w20_15c_lexical_overlap_math():
    """The bucketing key: word-MULTISET containment |A∩B|/min(|A|,|B|)
    over lowercased whitespace words — 1.0 when the shorter side is
    lexically contained, 0.0 for disjoint vocabularies, case-blind,
    multiset-aware (repeated words count once each)."""
    assert ft.lexical_overlap("a b c", "a b c") == pytest.approx(1.0)
    assert ft.lexical_overlap("a b", "c d e f") == pytest.approx(0.0)
    assert ft.lexical_overlap("A B", "a b c d") == pytest.approx(1.0), \
        "containment + case-insensitive"
    assert ft.lexical_overlap("a a b", "a b b") == pytest.approx(2 / 3), \
        "multiset intersection: min-count per word, over min length"
    assert ft.lexical_overlap("", "") == 0.0
    with pytest.raises(TypeError):
        ft.lexical_overlap("a b", None)


# -------------------------------------------- bucketed_spread rig -------
def test_w20_15d_bucketed_spread_splits_the_rigged_rig():
    """The position-lock instrument: the SAME B x B matrix the loss
    and cosine_spread read, but the DIAGONAL reported per bucket
    against the overall off-diagonal background. A rigged rig — half
    the pairs share their direction (high sim), half independent —
    must split the buckets apart while the overall row keeps the
    W20.14 fields verbatim."""
    gen = torch.Generator().manual_seed(31)
    b, d = 6, 64
    base = torch.randn(b, d, generator=gen).to(DEVICE)
    noise = 0.02 * torch.randn(b, d, generator=gen).to(DEVICE)
    m1_a = base.clone()
    m1_b = base + noise                  # pairs 0..5 near-aligned
    m1_b[3:] = torch.randn(3, d, generator=gen).to(DEVICE)  # 3..5 random
    m2_a, m2_b = -m1_a.clone(), -m1_b.clone()   # sign flip: sim = cos
    buckets = ["high", "high", "high", "low", "low", "low"]

    rep = ft.bucketed_spread(m1_a, m2_a, m1_b, m2_b, buckets)
    for key in ("cos_pos", "cos_neg", "spread", "off_max", "diag_min"):
        assert key in rep["overall"], f"overall keeps the W20.14 {key}"
    hi, lo = rep["buckets"]["high"], rep["buckets"]["low"]
    assert hi["n"] == 3 and lo["n"] == 3
    assert hi["cos_pos"] > 0.9, "the aligned half must score ~1"
    assert lo["cos_pos"] < 0.5, "the independent half must NOT score"
    assert hi["gap"] - lo["gap"] > 0.4, \
        "the bucket split IS the diagnostic (the position-lock read)"
    for bb in (hi, lo):
        assert bb["gap"] == pytest.approx(
            bb["cos_pos"] - rep["overall"]["cos_neg"]), \
            "gap = bucket diagonal - the overall off-diagonal mean"

    # the labels must arrive one per pair — a mismatch is a loud error
    with pytest.raises(ValueError):
        ft.bucketed_spread(m1_a, m2_a, m1_b, m2_b, ["high"] * 5)


# ------------------------------------- the trainer's history fields ------
class _ModelShim:
    """The `model` train_gates needs: .m1m2 + .parameters()."""

    def __init__(self, module: M1M2):
        self.m1m2 = module

    def parameters(self, recurse=True):
        return self.m1m2.parameters(recurse)


def _synth_calls(gen, n_pairs: int, per_pair: int = 2):
    """Capture stand-ins (B=1 ranks): a per-pair base key with 2% side-b
    jitter (the pair signal) + per-layer idiosyncratic noise; v = k/2."""
    base_k = torch.randn(n_pairs, H, 6, D, generator=gen).to(DEVICE)
    sides = []
    for p in range(n_pairs):
        pair_sides = []
        for side in range(per_pair):
            calls = []
            for lay in range(NLAY):
                k = base_k[p].unsqueeze(0).clone()
                if side:
                    k = k + 0.02 * torch.randn(
                        1, H, 6, D, generator=gen).to(DEVICE)
                calls.append((k.half(), (0.5 * k).half(), lay, None))
            pair_sides.append(calls)
        sides.append(pair_sides)
    return sides


def _patched_train(module, sides, m1_i, m2_i, **cfg_kw):
    """train_gates against the SYNTHETIC calls (the capture patched to
    hand out the recorded sides verbatim — the w19f rig, trimmed)."""
    shim = _ModelShim(module)
    for p in module.parameters():
        p.requires_grad_(False)
    module.write_gate_k.requires_grad_(True)
    module.write_gate_v.requires_grad_(True)
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


def test_w20_15e_history_carries_batch_ordinal_and_gate_values():
    """Every history entry: the capture-batch ORDINAL (the block id the
    loss sawtooths on — refresh_every=2 over 5 steps walks 1,1,2,2,3)
    and the gate VALUE stats (gk/gv min/max/absmean, read post-update:
    WHERE the gates are). The final entry's values must equal the
    module's actual gates — 300 steps of |g| absmax without a single
    parameter location was the observability hole this closes."""
    gen = torch.Generator().manual_seed(32)
    module = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                  num_linear_layers=NLAY).to(DEVICE)
    m1_i = module.init_state(dtype=torch.float32, device=DEVICE)
    m2_i = module.init_state(dtype=torch.float32, device=DEVICE)

    sides = _synth_calls(gen, n_pairs=4)
    cfg = _patched_train(module, sides, m1_i, m2_i,
                         max_steps=5, batch_pairs=4, lr=1e-2,
                         refresh_every=2, delta_loss=True)
    assert len(cfg.history) == 5
    assert [h["batch"] for h in cfg.history] == [1, 1, 2, 2, 3], \
        "batch = the capture ordinal (a new block every refresh)"

    for h in cfg.history:
        for key in ("gk_min", "gk_max", "gk_absmean",
                    "gv_min", "gv_max", "gv_absmean"):
            assert key in h and isinstance(h[key], float), \
                f"history misses the W20.15 gate-value field {key}"
        assert h["gk_min"] <= h["gk_absmean"] <= h["gk_max"] and \
            h["gv_min"] <= h["gv_absmean"] <= h["gv_max"], \
            "absmean sits inside the min/max bracket"

    last = cfg.history[-1]
    gk = module.write_gate_k.detach()
    gv = module.write_gate_v.detach()
    assert last["gk_min"] == pytest.approx(float(gk.min()), abs=1e-6)
    assert last["gk_max"] == pytest.approx(float(gk.max()), abs=1e-6)
    assert last["gv_min"] == pytest.approx(float(gv.min()), abs=1e-6)
    assert last["gv_max"] == pytest.approx(float(gv.max()), abs=1e-6)
    assert float((gk - 0.1).abs().max()) > 1e-4, \
        "the gates actually moved off the gate_init fill (live path)"
    assert last["gk_absmean"] > 0.0
