"""w28_tq_sim.py — numpy-only simulation of the TurboQuant round-trip on
REALISTIC GDN states (the repo's green tests all used random Gaussian
states; this sim measures what the per-unit norm + shared codebook do to
a state whose energy is spread across 32 heads with a realistic norm
spread).

W29 EXTENSION — THE PER-HEAD REPAIR, MEASURED HERE: the same states
quantized TWO ways, (i) the W28 regime — ONE flat 2^19 unit, one norm
(the deployed unit mapping, the dynamic-range hazard) and (ii) the W29
regime — 32 per-head units of d=16,384, each with its OWN norm, the
kind's D3 seed 101 at the per-head scale (resolve_quantizer("S", 16_384)
keeps the kind's seed — the frame contract). The committed codebooks
cb_b{3,4}_d16384.npz carry the per-head-scale Lloyd-Max tables. The
expected result (the paper's per-unit guarantee, Theorem 1 applied per
unit): worst-head rel-MSE collapses to the Gaussian floor (~2-3%) at ANY
head-norm spread, while the flat regime's worst head stays at the
spread-amplified floor the box matrix measured (worst-block 7.65 at 38x).

No torch, no repo imports — numpy + the committed codebook npz files.
"""
import math
import numpy as np

D = 1 << 19          # 524,288 — the W28 flat S unit
DH = 1 << 14         # 16,384 — the W29 per-head unit (DK*DV)
H, DK, DV = 32, 128, 128
SPLIT = D // 2       # the 3.5-bit recipe: 50/50 coordinate split
SPLIT_H = DH // 2    # the same recipe at the per-head scale


def hadamard_butterfly(v):
    """Walsh-Hadamard via the butterfly (v @ H), numpy, in-place staging."""
    v = v.reshape(-1).astype(np.float64).copy()
    n = v.size
    len_ = 1
    while len_ < n:
        v = v.reshape(n // (2 * len_), 2, len_)
        u, w = v[:, 0, :], v[:, 1, :]
        v = np.concatenate([u + w, u - w], axis=-1).reshape(n)
        len_ <<= 1
    return v / math.sqrt(n)


def rotation_signs(d, seed):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=d)


class Sim:
    """The torch path's math: y = fht(x) unscaled; idx = searchsorted(
    boundaries*norm, y); xhat = fht_adj(centroids[idx]) * norm.

    W29: ONE instance per unit scale — Sim(D) is the W28 flat 2^19
    regime; Sim(DH) is the per-head 2^14 regime (the same seed 101, the
    per-head-scale codebooks). The per-head regime rides the PERMUTED
    frame y = FHT(P x): the 2^14 Hadamard is H_(2^7) (x) H_(2^7),
    which EXACTLY aligns with the 128x128 per-head matrix layout — a
    rank-1 head k (x) v rotates to coordinatewise products of
    Gaussians (heavy-tailed), mismatching the Lloyd-Max codebook
    (measured: rank-1 rel-MSE 0.087 vs the 0.022 floor). The fixed
    coordinate permutation P (drawn from the kind's seed, the frame's
    second element) scatters the (i, j) pairs and restores the CLT —
    measured: 0.0221 on rank-1, rank-8, full-rank and scaled states
    alike (the floor, structure-independent). P is derived from the
    seed (deterministic frame), so the codes stay seed-addressed — no
    new serialization fields."""

    def __init__(self, d, bits_lo=3, bits_hi=4, seed=101, perm=False):
        self.d = d
        self.n_lo = d // 2
        self.signs = rotation_signs(d, seed)
        self.perm = None
        if perm:
            rng = np.random.default_rng(seed + 4545)
            self.perm = rng.permutation(d)
            self.inv_perm = np.argsort(self.perm)
        self.cb_lo = np.load(f"src/rag/codebooks/cb_b{bits_lo}_d{d}.npz")
        self.cb_hi = np.load(f"src/rag/codebooks/cb_b{bits_hi}_d{d}.npz")

    def quant_dequant(self, x):
        x = x.reshape(-1).astype(np.float64)
        if self.perm is not None:
            x = x[self.perm]
        norm = np.linalg.norm(x)
        if norm == 0:
            return np.zeros_like(x)
        # the repo's pair EXACTLY (fht.py): forward = ((x @ H) * s)/sqrt(d)
        # (Hadamard first, signs on the OUTPUT); adjoint = ((v * s) @ H)
        # /sqrt(d) (signs on the input, then Hadamard)
        y = hadamard_butterfly(x) * self.signs
        b_lo = self.cb_lo["boundaries"].astype(np.float64) * norm
        b_hi = self.cb_hi["boundaries"].astype(np.float64) * norm
        idx_lo = np.searchsorted(b_lo, y[: self.n_lo], side="left")
        idx_hi = np.searchsorted(b_hi, y[self.n_lo:], side="left")
        c_lo = self.cb_lo["centroids"].astype(np.float64)
        c_hi = self.cb_hi["centroids"].astype(np.float64)
        yhat = np.concatenate([c_lo[idx_lo], c_hi[idx_hi]])
        # the adjoint: ((v * s) @ H) / sqrt(d)
        xhat = hadamard_butterfly(yhat * self.signs)
        xhat = xhat * norm
        if self.perm is not None:
            xhat = xhat[self.inv_perm]
        return xhat


def quant_dequant_per_head(states, sim_h):
    """The W29 regime on a (H, DK, DV) state stack: head h quantized as
    its OWN (DH,) unit — own norm, the per-head rotation (seed 101 at
    d=16,384, the SAME draw for every head — the D3 frame contract at
    the per-head scale, exactly what TurboQuant.quant_units applies)."""
    out = np.zeros_like(states)
    flat = states.reshape(H, DK * DV)
    for h in range(H):
        out[h] = sim_h.quant_dequant(flat[h]).reshape(DK, DV)
    return out.reshape(-1)


def rel_mse(a, b):
    return float(np.sum((a - b) ** 2) / max(np.sum(b ** 2), 1e-30))


def gdn_state(n_tokens=8, head_norm_spread="real", seed=7):
    """A synthetic delta-rule state: per head S = sum_t beta_t k_t v_t^T
    with decay; per-head norms drawn with a REALISTIC spread."""
    rng = np.random.default_rng(seed)
    states = np.zeros((H, DK, DV))
    for h in range(H):
        if head_norm_spread == "equal":
            scale = 1.0
        elif head_norm_spread == "real":
            # log-normal over ~3 orders of magnitude (the delta-rule's
            # beta gates + key/value norms vary per head by this much)
            scale = 10.0 ** rng.uniform(-1.5, 0.0)
        lam = math.exp(rng.uniform(-4, -1))    # per-head decay
        S = np.zeros((DK, DV))
        for t in range(n_tokens):
            k = rng.standard_normal(DK)
            v = rng.standard_normal(DV)
            beta = rng.uniform(0.1, 1.0)
            S = lam * S + beta * np.outer(k, v)
        states[h] = S * scale
    return states.reshape(-1)


def report(label, x, sim, sim_h=None):
    xhat = sim.quant_dequant(x)
    x = x.reshape(-1)
    total = rel_mse(xhat, x)
    # per-head fidelity: the error on head h's coordinates vs its norm
    per_head_err = []
    per_head_rel = []
    for h in range(H):
        sl = slice(h * DK * DV, (h + 1) * DK * DV)
        e = np.sum((xhat[sl] - x[sl]) ** 2)
        n = np.sum(x[sl] ** 2)
        per_head_err.append(e)
        per_head_rel.append(e / max(n, 1e-30))
    hn = np.array([np.sqrt(np.sum(x[h * DK * DV:(h + 1) * DK * DV] ** 2))
                   for h in range(H)])
    print(f"--- {label} ---")
    print(f"  TOTAL rel-MSE:      {total:.4f}")
    print(f"  head norms: min {hn.min():.4g}  max {hn.max():.4g}  "
          f"spread {hn.max()/max(hn.min(),1e-30):.1f}x")
    order = np.argsort(hn)
    for tag, i in (("largest", order[-1]), ("median", order[len(order)//2]),
                   ("smallest", order[0])):
        print(f"  {tag:9s} head (norm {hn[i]:.3g}): per-head rel-MSE "
              f"{per_head_rel[i]:.4f}")
    worst = int(np.argmax(per_head_rel))
    print(f"  WORST per-head rel-MSE: {np.max(per_head_rel):.4f} "
          f"(head {worst}, norm {hn[worst]:.3g})")
    if sim_h is not None:
        # the W29 regime: the same state, 32 per-head units
        xh = quant_dequant_per_head(x.reshape(H, DK, DV), sim_h)
        per_head_rel_h = []
        for h in range(H):
            sl = slice(h * DK * DV, (h + 1) * DK * DV)
            per_head_rel_h.append(
                float(np.sum((xh[sl] - x[sl]) ** 2) / max(np.sum(x[sl] ** 2), 1e-30)))
        print(f"  [W29 per-head units] TOTAL rel-MSE: {rel_mse(xh, x):.4f}  "
              f"WORST per-head rel-MSE: {max(per_head_rel_h):.4f}")
    return total


def main():
    sim = Sim(D)
    sim_h = Sim(DH, perm=True)   # W29: per-head regime, permuted frame
    print(f"TQ sim at d={D}, bits 3/4 split, seed 101 (the S kind's frame)")
    print(f"W29 comparison: the same states through 32 per-head units of "
          f"d={DH}\n")

    # (a) the test-suite's state: random gaussian, all heads equal
    rng = np.random.default_rng(0)
    x_gauss = rng.standard_normal(D)
    report("RANDOM GAUSSIAN (the suite's state)", x_gauss, sim, sim_h)
    print()

    # (b) equal-energy delta-rule states (no head spread)
    x_eq = gdn_state(head_norm_spread="equal")
    report("DELTA-RULE, equal head norms", x_eq, sim, sim_h)
    print()

    # (c) realistic: per-head norm spread (delta-rule beta/norm variation)
    x_real = gdn_state(head_norm_spread="real")
    report("DELTA-RULE, realistic head-norm spread (1e-1.5..1)", x_real, sim, sim_h)
    print()

    # (d) extreme but plausible after decay: a few dominant heads
    rng = np.random.default_rng(3)
    x_ext = gdn_state(head_norm_spread="real", seed=11).copy()
    x_ext = x_ext.reshape(H, DK, DV)
    x_ext[:28] *= 1e-2        # 28 heads nearly silent
    x_ext = x_ext.reshape(-1)
    report("DELTA-RULE, 4 dominant + 28 silent heads", x_ext, sim, sim_h)


if __name__ == "__main__":
    main()
