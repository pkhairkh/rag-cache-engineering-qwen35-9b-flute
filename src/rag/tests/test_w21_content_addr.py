"""test_w21_content_addr.py — W21: the content-addressed global cache.

The object under test is the W21 `src/rag/m1m2.py` write: the slot
ADDRESS moved from position (slot = p mod mem_size — the v1
position-lock the box measured: spread plateau ~0.13, cos_neg parked
at ~+0.36) to CONTENT (simhash of the per-(b,h) mean-centered key over
a fixed Sylvester-Hadamard mixing, kappa independent sign-blocks), and
the write gates moved from per-layer scalars to per-(layer, head)
rows. The reference lineage: Product-Key Memory (Lample et al. 2019,
arXiv:1907.05242) and Memorizing Transformers (Wu et al. 2022,
arXiv:2203.08913) address by key CONTENT; Su et al. 2021 is the
centering fix for the anisotropy pedestal; Charikar 2002 is simhash.

Gates (all deterministic — pinned torch seeds, fp32, device-agnostic
(CUDA when available, else CPU), no network, no transformers import —
module-level contracts only; the model-level integration rides the
existing W17/W18/W19 suites, whose module constructions now run the
content default):

  A. ORDER INVARIANCE (the position-lock, gone): permuting a call's
     token order leaves the content-mode state ~unchanged (the same
     multiset lands in the same slots; only fp re-association moves)
     while the positional v1 state flips completely. This is THE
     property the §7 objective needed: content matches anywhere in
     the two texts, not at aligned positions.
  B. THE KERNEL (content matches, position doesn't): two sides built
     from the SAME token multiset in different orders -> pair-sim
     ~1; two independent sides -> pair-sim ~0 — under CONTENT
     addressing. Under POSITIONAL addressing the permuted pair scores
     ~0 (misaligned slots) — the A/B contrast that names the failure.
  C. THE PEDESTAL (centering, measured at module level): two
     independent sides sharing a huge common direction (k = eps +
     8*mu) score ~1 uncentered (the v1 +0.36 world, amplified) and
     ~0 centered — the Su et al. fix, inside the write, train and
     serve consistent.
  D. SIMHASH QUALITY: keys at cos ~0.95 share >= 1 block slot with
     high probability; independent keys almost never — the collision
     monotonicity the kernel depends on (direct _content_slots probe).
  E. PER-HEAD GATES: shapes (L, H); a one-hot gate row writes ONLY
     that head's delta; zero gates leave a nonzero state bit-identical
     (P3, content edition); a uniform row == the v1 scalar behavior
     on the positional path.
  F. DECODE (T=1): the centered write is exactly zero — the state
     comes back bit-identical (the memory freezes at prefill end).
  G. W18 EXACTNESS (content edition): write_deltas == the write's own
     delta bit-exact (gates=1 onto zero states); the replay Function's
     per-head gradients == the plain autograd chain's, bit-exact.
  H. DETERMINISM: two fresh modules of the same geometry derive the
     SAME slots (the Hadamard is mathematically pinned, no RNG, no
     version drift); repeated writes of the same k are bit-identical.
  I. LOUD GUARDS: bogus write_addr / addr_blocks (0, >8, blocks >
     mixing width) raise at construction or write; load_gates REFUSES
     a v1-shaped (L,) artifact loudly (shape mismatch — the addressing
     changed semantics; v1 gate files must be retrained).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import m1m2_finetune as ft
from m1m2 import M1M2

# Device for all tests (the GPU box runs CUDA; CPU only if unavailable)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The test geometry: enough slots that the simhash resolution is real
# (mem 256, kappa 4 -> stride 64, b = 6 sign bits per block).
H, D, MEM, NLAY = 4, 16, 256, 2
T = 32


def _keys(seed: int, t: int = T, common: float = 0.0,
          mu_seed: int = 99) -> torch.Tensor:
    """(1, H, t, D) fp32 keys; `common` adds `common * mu` (a fixed
    unit direction, seeded independently) to every key — the anisotropy
    pedestal knob."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, H, t, D, generator=g)
    if common:
        gm = torch.Generator().manual_seed(mu_seed)
        mu = torch.randn(D, generator=gm)
        mu = mu / mu.norm().clamp_min(1e-12)
        k = k + common * mu.view(1, 1, 1, D)
    return k.to(DEVICE)


def _state_cos(mod, ka, kb):
    """The pair-similarity of two single-side writes onto zero states,
    via the trainer's own pair_sim_matrix (the matrix the loss trains
    on — the metric and the kernel under test are the same object)."""
    z = mod.init_state(dtype=torch.float32)
    with torch.no_grad():
        m1a, m2a = mod.write(ka, ka, z, z, 0)
        m1b, m2b = mod.write(kb, kb, z, z, 0)
        sim = ft.pair_sim_matrix(m1a - z, m2a - z, m1b - z, m2b - z)
    return float(sim[0, 0])


# ============================================== A. order invariance ========
def test_w21a_order_invariance():
    """Permuting the token order leaves the content-mode state
    ~unchanged (the same multiset -> the same slots -> the same
    per-slot sums, up to fp re-association); the positional v1 state
    flips completely. THE position-lock contract."""
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
               num_linear_layers=NLAY).to(DEVICE)
    v1 = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
              num_linear_layers=NLAY,
              write_addr="positional", center_write=False).to(DEVICE)
    with torch.no_grad():
        mod.write_gate_k.fill_(1.0)
        mod.write_gate_v.fill_(1.0)
        v1.write_gate_k.fill_(1.0)
        v1.write_gate_v.fill_(1.0)
    k = _keys(11)
    g = torch.Generator().manual_seed(12)
    perm = torch.randperm(T, generator=g).to(DEVICE)
    kperm = k[:, :, perm, :]
    z = mod.init_state(dtype=torch.float32)
    with torch.no_grad():
        m1, _ = mod.write(k, k, z, z, 0)
        m1p, _ = mod.write(kperm, kperm, z, z, 0)
        rel = ((m1 - m1p) ** 2).sum() / m1.pow(2).sum().clamp_min(1e-30)
        assert float(rel) < 1e-5          # the same bag, re-associated
        # the v1 path: the same permutation is a DIFFERENT state
        s1, _ = v1.write(k, k, z, z, 0)
        s1p, _ = v1.write(kperm, kperm, z, z, 0)
        rel_v1 = ((s1 - s1p) ** 2).sum() / s1.pow(2).sum().clamp_min(1e-30)
        assert float(rel_v1) > 1e-3       # positions moved -> state moved


# ================================================ B. the kernel ============
def test_w21b_kernel_same_content_high_disjoint_low():
    """The content kernel: the SAME multiset (permuted) scores ~1;
    independent keys score ~0 — content matches anywhere; the
    positional mode cannot see the permuted pair at all (its slot
    contents misalign) — the A/B contrast that names the v1 ceiling."""
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
               num_linear_layers=NLAY).to(DEVICE)
    with torch.no_grad():
        mod.write_gate_k.fill_(1.0)
        mod.write_gate_v.fill_(1.0)
    k = _keys(13)
    g = torch.Generator().manual_seed(14)
    perm = torch.randperm(T, generator=g).to(DEVICE)
    kperm = k[:, :, perm, :]
    kind = _keys(15)                      # independent keys
    assert _state_cos(mod, k, kperm) > 0.95    # content match, moved
    assert abs(_state_cos(mod, k, kind)) < 0.30  # independent ~ 0
    v1 = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
              num_linear_layers=NLAY,
              write_addr="positional", center_write=False).to(DEVICE)
    with torch.no_grad():
        v1.write_gate_k.fill_(1.0)
        v1.write_gate_v.fill_(1.0)
    assert _state_cos(v1, k, kperm) < 0.30    # position-locked: blind


# ================================================ C. the pedestal ==========
def test_w21c_centering_removes_the_common_direction():
    """Two INDEPENDENT sides sharing a huge common direction (k = eps +
    8*mu): uncentered they score ~1 (the +0.36 box world, amplified);
    centered they score ~0 — the pedestal is removed INSIDE the write
    (train and serve see the same arithmetic)."""
    mod_c = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                 num_linear_layers=NLAY,
                 write_addr="content", center_write=True).to(DEVICE)
    mod_u = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
                 num_linear_layers=NLAY,
                 write_addr="content", center_write=False).to(DEVICE)
    for m in (mod_c, mod_u):
        with torch.no_grad():
            m.write_gate_k.fill_(1.0)
            m.write_gate_v.fill_(1.0)
    ka = _keys(16, common=8.0)            # eps + 8*mu
    kb = _keys(17, common=8.0)            # independent eps, SAME mu
    assert _state_cos(mod_u, ka, kb) > 0.80    # the pedestal dominates
    assert abs(_state_cos(mod_c, ka, kb)) < 0.30   # centered away


# =========================================== D. simhash quality ============
def test_w21d_simhash_collision_monotone():
    """Keys at cos ~0.95 share >= 1 block slot far more often than
    independent keys (the collision monotonicity the kernel rides;
    direct _content_slots probe at mem 256: P_share(same) ~0.9 vs
    ~0.06 per block for independent). W21.6: the keys broadcast across
    heads (a (D,) key views to (1, 1, 1, D) — the original .view(1, H,
    1, D) crashed: 64 elements requested from 16) and the independent
    bound is the ANY-of-kappa statistic: 1 - (1 - 2^-b)^kappa =
    1 - (1 - 1/16)^4 ~= 0.23 expected hits (the docstring's ~0.06 is
    the PER-BLOCK share) — the bound 0.35 sits ~3 sigma above the mean
    at 60 trials (sigma ~= 3.3), a real ceiling, not a coin flip."""
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
               num_linear_layers=NLAY).to(DEVICE)
    g = torch.Generator().manual_seed(18)
    hits_sim, hits_ind = 0, 0
    trials = 60
    for _ in range(trials):
        base = torch.randn(D, generator=g).to(DEVICE)
        noise = 0.05 * torch.randn(D, generator=g).to(DEVICE)
        other = torch.randn(D, generator=g).to(DEVICE)
        ka = (base + noise).view(1, 1, 1, D).expand(1, H, 1, D)
        kb_sim = base.view(1, 1, 1, D) + \
            0.05 * torch.randn(1, H, 1, D, generator=g).to(DEVICE)
        kb_ind = other.view(1, 1, 1, D) + \
            0.05 * torch.randn(1, H, 1, D, generator=g).to(DEVICE)
        with torch.no_grad():
            sa = mod._content_slots(ka)[0, 0, 0]    # (K,) per head 0
            sb_s = mod._content_slots(kb_sim)[0, 0, 0]
            sb_i = mod._content_slots(kb_ind)[0, 0, 0]
        if bool((sa == sb_s).any()):
            hits_sim += 1
        if bool((sa == sb_i).any()):
            hits_ind += 1
    assert hits_sim >= trials * 0.6       # similar keys collide (a lot)
    # independent keys rarely do (0.23 expected: 1 - (1 - 2^-4)^4;
    # ~3 sigma headroom at 60 trials)
    assert hits_ind <= trials * 0.35


# ============================================= E. per-head gates ===========
def test_w21e_per_head_gate_rows():
    """The gates are (L, H): a one-hot row writes ONLY that head's
    delta; zero gates leave a nonzero state bit-identical (P3, content
    edition); a uniform row reproduces the v1 per-layer scalar on the
    positional path (the D.5 numpy reference already pins that path —
    here the shape contract)."""
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
               num_linear_layers=NLAY).to(DEVICE)
    assert tuple(mod.write_gate_k.shape) == (NLAY, H)
    assert tuple(mod.write_gate_v.shape) == (NLAY, H)
    assert tuple(mod.read_gate.shape) == (NLAY,)
    k = _keys(19)
    z = mod.init_state(dtype=torch.float32)
    st = torch.randn(H, MEM, D, device=DEVICE)
    # zero gates: P3 no-op under content addressing (bit-identical)
    with torch.no_grad():
        n1, n2 = mod.write(k, k, st, st.clone(), 0)
        assert torch.equal(n1, st) and torch.equal(n2, st)
        # one-hot row: ONLY head 1's rows move
        mod.write_gate_k[0] = 0.0
        mod.write_gate_k[0, 1] = 1.0
        m1, _ = mod.write(k, k, z, z, 0)
        moved = (m1.abs().sum(dim=(1, 2)) > 0)
        assert bool(moved[1]) and not bool(moved[0]) \
            and not bool(moved[2]) and not bool(moved[3])


# ================================================== F. decode (T=1) ========
def test_w21f_decode_write_is_zero():
    """A T=1 call centers to the zero vector and writes NOTHING — the
    state comes back bit-identical (the memory freezes at prefill end;
    the v1 counter hack only approximated this)."""
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
               num_linear_layers=NLAY).to(DEVICE)
    with torch.no_grad():
        mod.write_gate_k.fill_(1.0)
        mod.write_gate_v.fill_(1.0)
    k1 = _keys(20, t=1)
    st = torch.randn(H, MEM, D, device=DEVICE)
    with torch.no_grad():
        n1, n2 = mod.write(k1, k1, st, st.clone(), 0)
        assert torch.equal(n1, st) and torch.equal(n2, st)
        d1, d2 = mod.write_deltas(k1, k1, 0, state=st)
        assert float(d1.abs().max()) == 0.0
        assert float(d2.abs().max()) == 0.0


# ============================================ G. W18 exactness =============
def test_w21g_write_deltas_match_and_grads_exact():
    """write_deltas == the write's own delta (gates=1 onto zero states,
    bit-exact — the shared scatter code), and the W18 replay
    Function's per-HEAD gate gradients == the plain autograd chain's,
    bit-exact, under the content path (the exactness contract carries
    over: the slots are a deterministic function of the captured k)."""
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
               num_linear_layers=NLAY).to(DEVICE)
    with torch.no_grad():
        mod.write_gate_k.fill_(1.0)
        mod.write_gate_v.fill_(1.0)
    k = _keys(21)
    v = _keys(22)
    z = mod.init_state(dtype=torch.float32)
    with torch.no_grad():
        via_write, _ = mod.write(k, v, z, z, 0)
        d1, d2 = mod.write_deltas(k, v, 0, state=z)
    assert torch.equal(via_write, d1)      # gates=1: the write IS delta
    assert not torch.equal(d1, d2)         # k != v -> distinct deltas

    # the replay gradient parity (Function vs plain chain), per head
    mod.write_gate_k.requires_grad_(True)
    mod.write_gate_v.requires_grad_(True)
    calls = [(k, v, 0, None), (_keys(23), _keys(24), 1, None)]
    m_i = mod.init_state(dtype=torch.float32)
    m2_i = mod.init_state(dtype=torch.float32)

    def _run(memopt):
        mod.zero_grad(set_to_none=True)
        m1, m2 = ft.replay_states(mod, calls, m_i, m2_i, memopt=memopt)
        (m1.sum() + m2.sum()).backward()
        gk = mod.write_gate_k.grad.clone()
        gv = mod.write_gate_v.grad.clone()
        mod.zero_grad(set_to_none=True)
        return gk, gv

    gk_f, gv_f = _run(True)
    gk_p, gv_p = _run(False)
    assert torch.equal(gk_f, gk_p)
    assert torch.equal(gv_f, gv_p)
    # per-head rows, nonzero, and the manual per-head answer per ROW:
    # g[L, h] = <ones[h], Delta_L[h]> summed over slots+dims (the
    # layer-1 call lands in row 1, not summed into row 0)
    with torch.no_grad():
        d1_1, _ = mod.write_deltas(k, v, 0, state=m_i)
        d1_2, _ = mod.write_deltas(calls[1][0], calls[1][1], 1,
                                   state=m_i)
    assert torch.allclose(gk_f[0], d1_1.sum(dim=(1, 2)), atol=0, rtol=0)
    assert torch.allclose(gk_f[1], d1_2.sum(dim=(1, 2)), atol=0, rtol=0)
    assert float(gk_f.abs().max()) > 0.0


# ================================================ H. determinism ===========
def test_w21h_two_fresh_modules_same_slots():
    """Two fresh modules of the same geometry derive the SAME mixing
    and the SAME slots (the Sylvester Hadamard is mathematically
    pinned — no RNG, no seed, no version drift), and repeated writes
    of the same k are bit-identical. The ingest box and the query box
    address identically without shipping any artifact."""
    m1 = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
              num_linear_layers=NLAY).to(DEVICE)
    m2 = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
              num_linear_layers=NLAY).to(DEVICE)
    k = _keys(25)
    z = m1.init_state(dtype=torch.float32)
    with torch.no_grad():
        m1.write_gate_k.fill_(0.7)
        m2.write_gate_k.fill_(0.7)
        m1.write_gate_v.fill_(-0.4)
        m2.write_gate_v.fill_(-0.4)
        a1, a2 = m1.write(k, k, z, z, 0)
        b1, b2 = m2.write(k, k, z, z, 0)
        assert torch.equal(a1, b1) and torch.equal(a2, b2)
        # repeated write of the same k: bit-identical
        c1, _ = m1.write(k, k, z, z, 0)
        assert torch.equal(a1, c1)
        # the mixing master is NOT a buffer/parameter: the state_dict
        # carries EXACTLY the 3 gate vectors (the checkpoint contract —
        # the load report's MISSING set stays the 3 gate names)
        assert len(list(m1.named_buffers())) == 0
        assert set(m1.state_dict()) == {
            "write_gate_k", "write_gate_v", "read_gate"}


# ================================================== I. loud guards =========
def test_w21i_loud_guards(tmp_path):
    """Bogus write_addr / addr_blocks raise at construction (or at
    write for post-init attribute sets); load_gates REFUSES a
    v1-shaped (L,) artifact loudly (the shape mismatch — v1 gate files
    are meaningless under content addressing; retrain)."""
    with pytest.raises(ValueError, match="write_addr"):
        M1M2(num_heads=H, head_dim=D, mem_size=MEM,
             num_linear_layers=NLAY, write_addr="bogus")
    with pytest.raises(ValueError, match="addr_blocks"):
        M1M2(num_heads=H, head_dim=D, mem_size=MEM,
             num_linear_layers=NLAY, addr_blocks=0)
    with pytest.raises(ValueError, match="addr_blocks"):
        M1M2(num_heads=H, head_dim=D, mem_size=MEM,
             num_linear_layers=NLAY, addr_blocks=9)
    # blocks > the mixing width: impossible at head_dim=1 (dp=1 < 2)
    with pytest.raises(ValueError, match="addr_blocks"):
        M1M2(num_heads=2, head_dim=1, mem_size=8,
             num_linear_layers=1, addr_blocks=2)
    mod = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
               num_linear_layers=NLAY).to(DEVICE)
    k = _keys(26, t=4)
    z = mod.init_state(dtype=torch.float32)
    mod.write_addr = "bogus"               # post-init attribute set
    with pytest.raises(ValueError, match="write_addr"):
        mod.write(k, k, z, z, 0)
    mod.write_addr = "content"

    # a v1-shaped (L,) gates artifact is refused loudly
    v1_style = {
        "write_gate_k": np.zeros((NLAY,), dtype=np.float64),
        "write_gate_v": np.zeros((NLAY,), dtype=np.float64),
        "read_gate": np.ones((NLAY,), dtype=np.float64),
        "meta_num_heads": H, "meta_head_dim": D, "meta_mem_size": MEM,
        "meta_num_linear_layers": NLAY,
    }
    path = str(tmp_path / "w21_v1_style_gates.npz")
    np.savez(path, **v1_style)
    with pytest.raises(ValueError, match="shape"):
        ft.load_gates(mod, path)
    # the W21-shaped artifact round-trips exactly
    with torch.no_grad():
        mod.write_gate_k.copy_(
            torch.linspace(-0.3, 0.3, NLAY * H).view(NLAY, H))
        mod.write_gate_v.fill_(0.2)
    w21_path = str(tmp_path / "w21_gates.npz")
    ft.save_gates(mod, w21_path, extra_meta={"write_addr": "content"})
    meta = ft.load_gates(mod, w21_path)
    assert int(meta["num_linear_layers"]) == NLAY
    assert torch.equal(
        mod.write_gate_k.detach(),
        torch.linspace(-0.3, 0.3, NLAY * H).view(NLAY, H).to(DEVICE))
    # the regime guard: a content-trained artifact into a POSITIONAL
    # module is a loud refusal (the kernel would silently mismatch)
    pos = M1M2(num_heads=H, head_dim=D, mem_size=MEM,
              num_linear_layers=NLAY,
              write_addr="positional", center_write=False).to(DEVICE)
    with pytest.raises(ValueError, match="write_addr"):
        ft.load_gates(pos, w21_path)
