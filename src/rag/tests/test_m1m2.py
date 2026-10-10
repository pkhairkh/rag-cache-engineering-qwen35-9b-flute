"""test_m1m2.py — W3.3: parity + reference tests for the M1/M2 addition.

The object under test is the W3.1 module `src/rag/m1m2.py` (the two global
memories, SPECIFICATION.md §2.2) plus its W3.2 wiring in
`src/scripts/modeling.py` (the M1/M2 block in
`Qwen3_5GatedDeltaNet.forward`, gated by `config.use_m1m2`, default OFF).

Gates (all deterministic — pinned torch seeds, fp32 for bit-identical
asserts, CUDA-only, no network):

  A. LAYER PARITY (torch.equal — the first forward is bit-exact through
     every cache route):
       A.1 flag OFF, cache-free vs a configured plain `DynamicCache`
           (an empty cache only affects FUTURE forwards).
       A.2 m1m2 attached (zero gates) + `TQCache` vs the A.1 plain-cache
           baseline — the quantized store's FIRST forward is exact
           (update_conv_state returns the prefill verbatim; the recurrent
           state starts at None), and the M1/M2 block provably RAN (codes
           exist in the cache) while being a bit-identical no-op.
       A.3 m1m2 attached + plain `DynamicCache` (no `read_m1` -> the wiring
           guard skips the block entirely) vs A.1.
  B. FULL-MODEL PARITY: `Qwen3_5TextModel(use_m1m2=False)` vs
     `(use_m1m2=True)` (state_dict loaded strict=False — the 3 gate
     vectors are the only extra params): bit-identical last_hidden_state
     on the same input_ids; `m1m2` registered EXACTLY ONCE
     (named_modules identity count == 1) yet shared by every linear layer
     (plain-object refs); write gates all 0, read gates all 1.
  C. GATE-OPENING SEMANTICS (the online loop: chunk, chunk, decode,
     decode through a TQCache):
       zero gates -> EVERY forward bit-identical to the no-m1m2 TQ
       baseline; open gates (0.5/0.7) -> the FIRST forward is still
       bit-identical (the read saw the zero memories), the SECOND forward
       diverges (macroscopic max|diff| ~0.5); M1/M2 codes exist after the
       first forward (codes of the zero state carry norm == 0.0 — the
       round-trip of an all-zero unit is exactly zero, WHICH is why the
       no-op survives the quantized store); `s_codes[0]` works alongside.
  D. REFERENCES (module-level, H=2, D=3, mem=4):
       D.4 read vs an explicit numpy softmax loop (fp64) within 1e-5
           (measured 1.6e-7); read_gate scaling is exact; zero memories
           read back exact zeros.
       D.5 write vs a numpy scatter loop (1e-5; measured 1.3e-7); the
           two-token slot collision is bit-exactly g*(k0+k1) (sum-then-
           gate, the W3.1 deviation note — gates 0.37/-0.61 are NOT
           powers of two); sequential [0,1]+[2,3] == single [0,1,2,3]
           bit-identically; zero gates leave a NONZERO state bit-identical
           and never mutate the inputs; forward == read+write.
       D.6 the §2.2 "summing is lossless" claim THROUGH the quantized
           store (the D4 delta protocol at the wiring level): dequant(A)
           + dequant(B) == dequant(sequential A-then-B) within the house
           quant tolerance rel-MSE < 0.06 (measured 0.031 / 0.024) —
           via the `m1m2_from_cache` / `push_to_cache` §3.2 glue.  E. LOUD GUARDS: read/write shape mismatches, k/v disagreement, bad
     positions, non-tensors, bool layer_idx and layer_idx >=
     num_linear_layers (read, write AND forward) all raise.

CAVEAT discovered in A.1 (documented, not a divergence): the FIRST
forward with an empty cache is bit-identical to cache-free on every
route (plain DynamicCache, TQCache) — the cached conv/recurrent state
only influences FUTURE forwards. From the SECOND forward on, cache-free
and cached are semantically DIFFERENT computations (warm recurrent state
+ conv window), and TQ-vs-plain diverge by one quant round (measured
max|diff| 0.113 at this geometry) — so gate C's two-forward baseline is
apples-to-apples: the no-m1m2 layer through the SAME TQCache, not the
cache-free path. A.2/A.3 compare single forwards where all routes agree
bit-exactly, so the plain-cache baseline from A.1 is the honest
reference there.

WIRING NOTE (reported to the W3.2 owner, not pinned here): the modeling
block calls `_m1m2(_q, _k, _v, ...)`` WITHOUT `positions`, so every call
scatters into slots arange(T) mod mem — decode steps (T=1) all write
slot 0 and a second chunk re-uses slots 0..T-1 instead of T..2T-1.
Invisible at zero gates (writes are no-ops) and irrelevant to every
parity gate here (the baselines carry no m1m2 at all), but per SPEC
§2.2 ("slot = position mod mem_size") the production open-gate path
should pass past_seen_tokens + arange(T). Do not "fix" it by editing
this file's expectations when the wiring is corrected — delete this
note instead.

Codebook note: the tiny geometries here flatten to power-of-two units
already carrying solved codebooks in the tree (M1/M2/conv at d=512,
S at d=1024). A fresh clone re-solves the d=512 set deterministically
(see the W2.2 worklog note about the untracked npz caches).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache

import modeling
from m1m2 import M1M2, m1m2_from_cache, push_to_cache
from tq_cache import TQCache

# Device for all tests - CUDA required for causal_conv1d
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------ geometries ---
# The wiring-level geometry (matches the shared W3.2/W3.3 context kit):
# v-heads 4, v-head-dim 16, mem 8 -> M1/M2 numel 512 = the d512 quant unit.
WI_H, WI_D, WI_MEM = 4, 16, 8
WI_M_NUMEL = WI_H * WI_MEM * WI_D                 # 512 (power of two)

# The numpy-reference geometry (small, non-power-of-two on purpose — the
# module itself has NO power-of-two constraint, only the quantized store).
REF_H, REF_D, REF_MEM = 2, 3, 4

REL_MSE_GATE = 0.06        # house single-quant-round budget (W1.4/W2.2/W2.3)
LOOP_FORWARD_MAXDIFF = 1e-3  # open gates: the 2nd forward must differ LOUDLY


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-30).item()


def _tiny_config() -> Qwen3_5TextConfig:
    """The context-kit tiny config: 1 linear + 1 full-attention layer."""
    return Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=WI_H, linear_num_key_heads=2,
        linear_key_head_dim=WI_D, linear_value_head_dim=WI_D,
        linear_conv_kernel_dim=4,
    )


def _tiny_layer() -> modeling.Qwen3_5GatedDeltaNet:
    """A fresh tiny GatedDeltaNet (deterministic weights; layer_idx 0)."""
    torch.manual_seed(11)
    layer = modeling.Qwen3_5GatedDeltaNet(_tiny_config(), 0)
    layer = layer.to(DEVICE)
    layer.eval()
    return layer


def _attach_m1m2(layer, gates=None) -> M1M2:
    """Attach a shared M1M2 the way `Qwen3_5TextModel.__init__` does:
    plain-object ref (NOT a submodule registration) + linear ordinal."""
    m = M1M2(num_heads=WI_H, head_dim=WI_D, mem_size=WI_MEM,
             num_linear_layers=1)
    m = m.to(DEVICE)
    if gates is not None:
        with torch.no_grad():
            m.write_gate_k.fill_(gates[0])
            m.write_gate_v.fill_(gates[1])
    object.__setattr__(layer, "m1m2", m)
    layer.m1m2_linear_ordinal = 0
    return m


def _loop_inputs():
    """The online-loop inputs: 2 chunks of 6 tokens + 2 single-token
    decode steps (one shared generator -> identical across runs)."""
    g = torch.Generator(device=DEVICE).manual_seed(12)
    return [torch.randn(1, T, 64, generator=g, device=DEVICE)
            for T in (6, 6, 1, 1)]


def _run_online_loop(layer, cache) -> list:
    """prefill chunk -> second chunk -> decode -> decode (modeling.py's
    exact cache call order; every forward in eval/no_grad)."""
    outs = []
    with torch.no_grad():
        for x in _loop_inputs():
            outs.append(layer(hidden_states=x, cache_params=cache))
    return outs


# ============================================================ A. parity === #
def test_a1_flag_off_cache_free_vs_plain_dynamic_cache():
    """A.1: with NO m1m2 attached, a configured plain DynamicCache and the
    cache-free route agree bit-identically on the first forward (an empty
    cache only affects future forwards; the conv/recurrent paths compute
    the same kernels on the same tensors)."""
    layer = _tiny_layer()
    xs = _loop_inputs()
    with torch.no_grad():
        out_free = layer(hidden_states=xs[0], cache_params=None)
        out_plain = layer(hidden_states=xs[0],
                          cache_params=DynamicCache(config=_tiny_config()))
    assert torch.equal(out_free, out_plain)
    assert out_free.shape == (1, 6, 64)
    # the parity baseline object for A.2/A.3 is this plain-cache output
    # (same input, fresh cache — rebuilt deterministically below).
    layer2 = _tiny_layer()
    with torch.no_grad():
        out_plain2 = layer2(hidden_states=xs[0],
                            cache_params=DynamicCache(config=_tiny_config()))
    assert torch.equal(out_plain, out_plain2)


def test_a2_zero_gate_tqcache_vs_plain_cache_baseline():
    """A.2: m1m2 attached with ZERO gates + TQCache vs the A.1 plain-cache
    baseline — bit-identical, and the M1/M2 block provably executed (the
    cache holds M1/M2 codes afterwards, carrying norm 0.0: TurboQuant maps
    the all-zero unit to codes of zero, so the zero state survives the
    quantize/dequantize round-trip exactly — the P3 no-op through the
    §3.2 online store)."""
    layer = _tiny_layer()
    _attach_m1m2(layer)                      # zero gates (P3 init)
    xs = _loop_inputs()
    cache = TQCache(layer_types=["linear_attention"])
    with torch.no_grad():
        out_tq = layer(hidden_states=xs[0], cache_params=cache)
        out_plain = _tiny_layer()(
            hidden_states=xs[0],
            cache_params=DynamicCache(config=_tiny_config()))
    assert torch.equal(out_tq, out_plain)
    # the block RAN (codes exist) while changing nothing:
    assert cache.m1_codes is not None and cache.m2_codes is not None
    assert float(cache.m1_codes.norm) == 0.0
    assert float(cache.m2_codes.norm) == 0.0
    assert cache.m1_codes.d == WI_M_NUMEL
    assert cache.m1_codes.seed == 303 and cache.m2_codes.seed == 404


def test_a3_m1m2_attached_plain_cache_skips_block():
    """A.3: m1m2 attached but a PLAIN DynamicCache (no `read_m1`) — the
    W3.2 wiring guard skips the M1/M2 block entirely; output bit-identical
    to the A.1 cache-free/plain routes."""
    layer = _tiny_layer()
    _attach_m1m2(layer)                      # would be a no-op anyway
    cache = DynamicCache(config=_tiny_config())
    assert not hasattr(cache, "read_m1")     # the wiring's exact guard
    xs = _loop_inputs()
    with torch.no_grad():
        out = layer(hidden_states=xs[0], cache_params=cache)
        out_free = _tiny_layer()(hidden_states=xs[0], cache_params=None)
    assert torch.equal(out, out_free)


# ==================================================== B. full-model parity == #
def test_b_full_model_flag_parity_and_wiring_structure():
    """B: use_m1m2 ON vs OFF — bit-identical last_hidden_state (the gates
    are zero/one-initialized extras loaded via strict=False), `m1m2`
    registered exactly once and shared by both linear layers."""
    def cfg(use_m1m2: bool) -> Qwen3_5TextConfig:
        kw = dict(
            hidden_size=64, num_hidden_layers=4,
            layer_types=["linear_attention", "full_attention",
                         "linear_attention", "full_attention"],
            linear_num_value_heads=WI_H, linear_num_key_heads=2,
            linear_key_head_dim=WI_D, linear_value_head_dim=WI_D,
            linear_conv_kernel_dim=4,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
        if use_m1m2:
            kw["use_m1m2"] = True
            kw["m1m2_mem_size"] = WI_MEM
        return Qwen3_5TextConfig(**kw)

    torch.manual_seed(21)
    model_off = modeling.Qwen3_5TextModel(cfg(False)); model_off.eval()
    model_off = model_off.to(DEVICE)
    torch.manual_seed(21)
    model_on = modeling.Qwen3_5TextModel(cfg(True)); model_on.eval()
    model_on = model_on.to(DEVICE)

    # the OFF model carries no M1/M2 anywhere (default: zero code-path)
    assert getattr(model_off, "m1m2", None) is None
    assert getattr(model_off.layers[0].linear_attn, "m1m2", None) is None

    # strict=False load: the 3 gate vectors are the ONLY extra params
    missing, unexpected = model_on.load_state_dict(
        model_off.state_dict(), strict=False)
    assert unexpected == []
    assert set(missing) == {"m1m2.write_gate_k", "m1m2.write_gate_v",
                            "m1m2.read_gate"}
    # P3 init survives the partial load:
    assert bool((model_on.m1m2.write_gate_k == 0).all())
    assert bool((model_on.m1m2.write_gate_v == 0).all())
    assert bool((model_on.m1m2.read_gate == 1).all())

    # registered ONCE, shared by every linear layer (plain-object refs):
    assert sum(1 for _n, m in model_on.named_modules()
               if m is model_on.m1m2) == 1
    lin0, lin2 = model_on.layers[0].linear_attn, model_on.layers[2].linear_attn
    assert lin0.m1m2 is model_on.m1m2 and lin2.m1m2 is model_on.m1m2
    assert lin0.m1m2_linear_ordinal == 0 and lin2.m1m2_linear_ordinal == 1
    assert model_on.m1m2.num_linear_layers == 2
    assert tuple(model_on.m1m2.state_shape()) == (WI_H, WI_MEM, WI_D)

    # bit-identical outputs, cache route (default use_cache=True — the
    # internal plain DynamicCache has no read_m1, so the block is skipped)
    # and cache-free alike:
    ids = torch.randint(0, 128, (2, 6), device=DEVICE)
    with torch.no_grad():
        out_off = model_off(input_ids=ids).last_hidden_state
        out_on = model_on(input_ids=ids).last_hidden_state
        out_off_nc = model_off(input_ids=ids, use_cache=False).last_hidden_state
        out_on_nc = model_on(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(out_off, out_on)
    assert torch.equal(out_off_nc, out_on_nc)


# ================================================= C. gate-opening semantic == #
def test_c_zero_gates_two_forwards_bit_identical_to_baseline():
    """C (zero gates): the full online loop — chunk, chunk, decode, decode
    through a TQCache — is bit-identical at EVERY step to the no-m1m2
    layer through the SAME cache (the apples-to-apples baseline: from the
    second forward on, TQ-vs-plain diverge by one quant round, so the
    plain-cache route can no longer serve as the bit-identity reference).
    M1/M2 codes exist after the FIRST forward (norm 0.0 — the zero state
    round-trips exactly) and the S path works alongside."""
    # baseline: no m1m2, TQCache
    base_outs = _run_online_loop(_tiny_layer(),
                                 TQCache(layer_types=["linear_attention"]))
    base_cache = TQCache(layer_types=["linear_attention"])
    _run_online_loop(_tiny_layer(), base_cache)
    assert base_cache.s_codes[0] is not None

    # zero gates + TQCache: bit-identical everywhere, codes live after fwd 1
    layer = _tiny_layer()
    _attach_m1m2(layer)                      # P3 zero gates
    cache = TQCache(layer_types=["linear_attention"])
    outs = []
    xs = _loop_inputs()
    with torch.no_grad():
        for i, x in enumerate(xs):
            outs.append(layer(hidden_states=x, cache_params=cache))
            if i == 0:                       # codes exist after the FIRST fwd
                assert cache.m1_codes is not None
                assert cache.m2_codes is not None
                assert float(cache.m1_codes.norm) == 0.0
                assert float(cache.m2_codes.norm) == 0.0
                assert cache.s_codes[0] is not None
    for i, (a, b) in enumerate(zip(outs, base_outs)):
        assert torch.equal(a, b), f"zero-gate forward {i + 1} diverged"
    # the loop itself genuinely warmed the cache (fwd2 != fwd1 in the
    # baseline — otherwise the parity above would be vacuous):
    assert not torch.equal(base_outs[0], base_outs[1])


def test_c_open_gates_first_forward_identical_second_differs():
    """C (open gates 0.5/0.7): the FIRST forward is STILL bit-identical to
    the baseline — the read saw the (still zero) memories, and this
    chunk's writes only land in the state the NEXT forward reads (the
    causal M1/M2 contract). The SECOND forward (and the decode steps
    after it) diverge macroscopically."""
    base_outs = _run_online_loop(_tiny_layer(),
                                 TQCache(layer_types=["linear_attention"]))

    layer = _tiny_layer()
    _attach_m1m2(layer, gates=(0.5, 0.7))
    cache = TQCache(layer_types=["linear_attention"])
    outs = _run_online_loop(layer, cache)

    assert torch.equal(outs[0], base_outs[0])          # read saw zero M1/M2
    for i in (1, 2, 3):                                # chunk 2 + decode steps
        assert not torch.equal(outs[i], base_outs[i])
        assert (outs[i] - base_outs[i]).abs().max().item() \
            > LOOP_FORWARD_MAXDIFF                     # measured ~0.55
    # the memories truly opened (nonzero round-trip state in the cache):
    m1 = cache.read_m1(dtype=torch.float32)
    m2 = cache.read_m2(dtype=torch.float32)
    assert m1 is not None and float(m1.norm().item()) > 0.0
    assert m2 is not None and float(m2.norm().item()) > 0.0


# ======================================================= D. reference checks == #
def test_d4_read_reference_numpy():
    """D.4: `read` vs an explicit numpy softmax loop (fp64 reference) at
    H=2, D=3, mem=4 — within 1e-5 (measured 1.6e-7); the read gate is an
    exact scalar scaling; zero memories read back exact zeros."""
    torch.manual_seed(31)
    mod = M1M2(num_heads=REF_H, head_dim=REF_D, mem_size=REF_MEM,
               num_linear_layers=2)
    mod = mod.to(DEVICE)
    B, H, T, D, M = 2, REF_H, 5, REF_D, REF_MEM
    q = torch.randn(B, H, T, D, device=DEVICE)
    m1 = torch.randn(H, M, D, device=DEVICE)
    m2 = torch.randn(H, M, D, device=DEVICE)
    with torch.no_grad():
        out = mod.read(q, m1, m2, layer_idx=1)     # read_gate[1] == 1.0
        ref = np.zeros((B, H, T, D), dtype=np.float64)
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    scores = m1[h].cpu().numpy().astype(np.float64) \
                        @ q[b, h, t].cpu().numpy().astype(np.float64)
                    scores = scores / np.sqrt(D)
                    e = np.exp(scores - scores.max())
                    probs = e / e.sum()
                    ref[b, h, t] = probs @ m2[h].cpu().numpy().astype(np.float64)
        assert np.abs(out.cpu().numpy() - ref).max() < 1e-5

        # per-layer read_gate scaling is exact (out * gate, fp32):
        mod.read_gate[1] = 2.5
        out_g25 = mod.read(q, m1, m2, layer_idx=1)
        mod.read_gate[1] = 1.0
        out_g1 = mod.read(q, m1, m2, layer_idx=1)
        assert torch.equal(out_g25, 2.5 * out_g1)

        # zero memories -> exact zeros (softmax(uniform) @ 0):
        z = torch.zeros(H, M, D, device=DEVICE)
        assert torch.equal(mod.read(q, z, z, layer_idx=0),
                           torch.zeros(B, H, T, D, device=DEVICE))


def test_d5_write_reference_numpy():
    """D.5: `write` vs a numpy scatter loop; the collision contract
    m1[h, 0] == g*(k0 + k1) holds bit-exactly (sum-then-gate — gates
    0.37/-0.61 are not powers of two); sequential [0,1]+[2,3] equals the
    single [0,1,2,3] call bit-identically; zero gates leave a nonzero
    state untouched and nothing is mutated in place; forward composes."""
    H, D, M = REF_H, REF_D, REF_MEM
    torch.manual_seed(32)
    mod = M1M2(num_heads=H, head_dim=D, mem_size=M, num_linear_layers=2)
    mod = mod.to(DEVICE)
    with torch.no_grad():
        mod.write_gate_k[0] = 0.37
        mod.write_gate_v[0] = -0.61
    gk, gv = 0.37, -0.61
    z = torch.zeros(H, M, D, device=DEVICE)

    # -- numpy scatter reference (default positions: slot = t mod mem) --
    torch.manual_seed(34)
    k = torch.randn(1, H, 4, D, device=DEVICE)
    v = torch.randn(1, H, 4, D, device=DEVICE)
    with torch.no_grad():
        m1n, m2n = mod.write(k, v, z, z, layer_idx=0)
        ref1 = np.zeros((H, M, D), dtype=np.float64)
        ref2 = np.zeros((H, M, D), dtype=np.float64)
        for t in range(4):
            slot = t % M
            ref1[:, slot] += gk * k[0, :, t].cpu().numpy().astype(np.float64)
            ref2[:, slot] += gv * v[0, :, t].cpu().numpy().astype(np.float64)
        assert np.abs(m1n.cpu().numpy() - ref1).max() < 1e-5     # measured 1.3e-7
        assert np.abs(m2n.cpu().numpy() - ref2).max() < 1e-5

    # -- the two-token slot collision: g*(k0 + k1) EXACTLY --
    torch.manual_seed(33)
    kc = torch.randn(1, H, 2, D, device=DEVICE)
    vc = torch.randn(1, H, 2, D, device=DEVICE)
    with torch.no_grad():
        m1c, m2c = mod.write(kc, vc, z, z, layer_idx=0,
                             positions=torch.tensor([0, M], device=DEVICE))  # both -> slot 0
        exp_m1 = torch.zeros(H, M, D, device=DEVICE)
        exp_m2 = torch.zeros(H, M, D, device=DEVICE)
        exp_m1[:, 0] = gk * (kc[0, :, 0, :] + kc[0, :, 1, :])
        exp_m2[:, 0] = gv * (vc[0, :, 0, :] + vc[0, :, 1, :])
        assert torch.equal(m1c, exp_m1)
        assert torch.equal(m2c, exp_m2)

    # -- sequential [0,1] then [2,3] == single [0,1,2,3], bit-identical --
    torch.manual_seed(35)
    kA = torch.randn(1, H, 2, D, device=DEVICE); vA = torch.randn(1, H, 2, D, device=DEVICE)
    kB = torch.randn(1, H, 2, D, device=DEVICE); vB = torch.randn(1, H, 2, D, device=DEVICE)
    with torch.no_grad():
        s1 = mod.write(kA, vA, z, z, layer_idx=0,
                       positions=torch.tensor([0, 1], device=DEVICE))
        s2 = mod.write(kB, vB, s1[0], s1[1], layer_idx=0,
                       positions=torch.tensor([2, 3], device=DEVICE))
        single = mod.write(torch.cat([kA, kB], dim=2),
                           torch.cat([vA, vB], dim=2), z, z, layer_idx=0,
                           positions=torch.tensor([0, 1, 2, 3], device=DEVICE))
        assert torch.equal(s2[0], single[0])
        assert torch.equal(s2[1], single[1])

    # -- zero gates: a NONZERO state comes back bit-identical; no mutation --
    torch.manual_seed(36)
    mod_zero = M1M2(num_heads=H, head_dim=D, mem_size=M,
                    num_linear_layers=2)
    mod_zero = mod_zero.to(DEVICE)
    st1 = torch.randn(H, M, D, device=DEVICE); st2 = torch.randn(H, M, D, device=DEVICE)
    st1c, st2c = st1.clone(), st2.clone()
    q = torch.randn(1, H, 5, D, device=DEVICE)
    with torch.no_grad():
        n1, n2 = mod_zero.write(k, v, st1, st2, layer_idx=0,
                                positions=torch.tensor([0, M, 2, 3], device=DEVICE))
        assert torch.equal(n1, st1) and torch.equal(n2, st2)
        assert torch.equal(st1, st1c) and torch.equal(st2, st2c)
        # forward == read (pre-write state) + write composition:
        f_out, f1, f2 = mod_zero.forward(q, k, v, st1, st2, layer_idx=1,
                                         positions=torch.tensor([0, 1, 2, 3], device=DEVICE))
        r_out = mod_zero.read(q, st1, st2, layer_idx=1)
        w1, w2 = mod_zero.write(k, v, st1, st2, layer_idx=1,
                                positions=torch.tensor([0, 1, 2, 3], device=DEVICE))
        assert torch.equal(f_out, r_out)
        assert torch.equal(f1, w1) and torch.equal(f2, w2)


def test_d6_composability_through_the_quantized_store():
    """D.6: SPEC §2.2's "summing is lossless" through the QUANTIZED store
    (the D4 delta protocol at the wiring level): writing tokens A into a
    fresh cache, tokens B into another fresh cache, and A-then-B
    sequentially (the second write landing on the DEQUANTIZED round-trip
    state, exactly as the §3.2 online loop serves it) satisfies

        dequant(codes_A) + dequant(codes_B) ~= dequant(codes_AB)

    within the house quant tolerance rel-MSE < 0.06 (measured 0.031 for
    M1, 0.024 for M2). Run through the `m1m2_from_cache` / `push_to_cache`
    glue (the exact §3.2 read/write sides). NOTE the denominator: the
    delta form rel-MSE(dequant(AB) - dequant(A), dequant(B)) measures
    0.065 at this geometry — its denominator ||delta_B|| is half the sum
    norm, so the SAME absolute error sits 2x outside the gate; the sum
    form (the task's formulation, denominator ||state_AB||) is the one
    that carries the honest single-round budget."""
    H, D, MEM = WI_H, WI_D, WI_MEM
    torch.manual_seed(41)
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM, num_linear_layers=1)
    mod = mod.to(DEVICE)
    with torch.no_grad():
        mod.write_gate_k.fill_(0.5)
        mod.write_gate_v.fill_(0.7)
    g = torch.Generator(device=DEVICE).manual_seed(777)
    kA = torch.randn(1, H, 5, D, generator=g, device=DEVICE); vA = torch.randn(1, H, 5, D, generator=g, device=DEVICE)
    kB = torch.randn(1, H, 5, D, generator=g, device=DEVICE); vB = torch.randn(1, H, 5, D, generator=g, device=DEVICE)

    # the §3.2 first forward: a cache with no codes reads back ZEROS
    fresh = TQCache(layer_types=[])
    m1z, m2z = m1m2_from_cache(fresh, mod, torch.float32)
    # fresh cache returns CPU tensors; move to device for comparison
    m1z = m1z.to(DEVICE)
    m2z = m2z.to(DEVICE)
    assert torch.equal(m1z, torch.zeros(H, MEM, D, device=DEVICE))
    assert torch.equal(m2z, torch.zeros(H, MEM, D, device=DEVICE))

    with torch.no_grad():
        # run A: fresh cache, write tokens A
        cA = TQCache(layer_types=[])
        a1, a2 = m1m2_from_cache(cA, mod, torch.float32)
        a1, a2 = a1.to(DEVICE), a2.to(DEVICE)
        dA = push_to_cache(cA, *mod.write(kA, vA, a1, a2, layer_idx=0))
        # run B: ANOTHER fresh cache, write tokens B (the standalone delta)
        cB = TQCache(layer_types=[])
        b1, b2 = m1m2_from_cache(cB, mod, torch.float32)
        b1, b2 = b1.to(DEVICE), b2.to(DEVICE)
        dB = push_to_cache(cB, *mod.write(kB, vB, b1, b2, layer_idx=0))
        # run AB: the online loop — write A, then the next forward READS
        # the round-trip and writes B on top of it
        cAB = TQCache(layer_types=[])
        f1, f2 = m1m2_from_cache(cAB, mod, torch.float32)
        f1, f2 = f1.to(DEVICE), f2.to(DEVICE)
        push_to_cache(cAB, *mod.write(kA, vA, f1, f2, layer_idx=0))
        r1, r2 = m1m2_from_cache(cAB, mod, torch.float32)
        assert torch.equal(r1, dA[0]) and torch.equal(r2, dA[1])  # what you
        # wrote is what you read back
        dAB = push_to_cache(cAB, *mod.write(kB, vB, r1, r2, layer_idx=0))

    # the additive-delta property through the quantized store:
    assert _rel_mse(dA[0] + dB[0], dAB[0]) < REL_MSE_GATE    # 0.031
    assert _rel_mse(dA[1] + dB[1], dAB[1]) < REL_MSE_GATE    # 0.024
    # M1/M2 codes keep their D3 seeds at the custom (test) size:
    assert cAB.m1_codes.d == WI_M_NUMEL
    assert cAB.m1_codes.seed == 303 and cAB.m2_codes.seed == 404


# =========================================================== E. loud guards == #
@pytest.mark.parametrize("bad_call,exc", [
    # read: q head-count mismatch vs the module geometry
    (lambda m, q, k, v, m1, m2: m.read(torch.randn(1, 3, 5, 3, device=DEVICE), m1, m2, 0),
     ValueError),
    # read: m1 shape mismatch (state_shape contract)
    (lambda m, q, k, v, m1, m2: m.read(q, torch.randn(2, 5, 3, device=DEVICE), m2, 0),
     ValueError),
    # write: k/v must share (B, H, T, D)
    (lambda m, q, k, v, m1, m2: m.write(k, torch.randn(1, 2, 4, 3, device=DEVICE), m1, m2, 0),
     ValueError),
    # write: state shape mismatch
    (lambda m, q, k, v, m1, m2: m.write(k, v, torch.randn(2, 5, 3, device=DEVICE), m2, 0),
     ValueError),
    # positions: wrong length / negative / float dtype
    (lambda m, q, k, v, m1, m2: m.write(k, v, m1, m2, 0,
                                        positions=torch.tensor([0, 1], device=DEVICE)),
     ValueError),
    (lambda m, q, k, v, m1, m2: m.write(k, v, m1, m2, 0,
                                        positions=torch.tensor([-1, 0, 1, 2, 3], device=DEVICE)),
     ValueError),
    (lambda m, q, k, v, m1, m2: m.write(k, v, m1, m2, 0,
                                        positions=torch.tensor([0.5, 1, 2, 3, 4], device=DEVICE)),
     TypeError),
    # non-tensor q; bool layer_idx
    (lambda m, q, k, v, m1, m2: m.read("nope", m1, m2, 0), TypeError),
    (lambda m, q, k, v, m1, m2: m.write(k, v, m1, m2, True), TypeError),
    # layer_idx out of range: read / write / forward
    (lambda m, q, k, v, m1, m2: m.read(q, m1, m2, 2), ValueError),
    (lambda m, q, k, v, m1, m2: m.write(k, v, m1, m2, 2), ValueError),
    (lambda m, q, k, v, m1, m2: m.forward(q, k, v, m1, m2, 5), ValueError),
    (lambda m, q, k, v, m1, m2: m.read(q, m1, m2, -1), ValueError),
])
def test_e_loud_guards(bad_call, exc):
    """E: shape mismatches, k/v disagreement, malformed positions,
    non-tensors, bool layer_idx, and layer_idx >= num_linear_layers
    (read, write AND forward) raise loudly — never a silent reshape."""
    H, D, M = REF_H, REF_D, REF_MEM
    mod = M1M2(num_heads=H, head_dim=D, mem_size=M, num_linear_layers=2)
    mod = mod.to(DEVICE)
    q = torch.randn(1, H, 5, D, device=DEVICE)
    k = torch.randn(1, H, 5, D, device=DEVICE)
    v = torch.randn(1, H, 5, D, device=DEVICE)
    m1 = torch.randn(H, M, D, device=DEVICE)
    m2 = torch.randn(H, M, D, device=DEVICE)
    with pytest.raises(exc):
        bad_call(mod, q, k, v, m1, m2)
