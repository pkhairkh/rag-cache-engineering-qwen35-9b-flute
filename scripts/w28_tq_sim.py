"""w28_tq_sim.py — numpy-only simulation of the TurboQuant round-trip on
REALISTIC GDN states (the repo's green tests all used random Gaussian
states; this sim measures what the per-unit norm + shared codebook do to
a state whose energy is spread across 32 heads with a realistic norm
spread).

No torch, no repo imports — numpy + the committed codebook npz files.
"""
import math
import numpy as np

D = 1 << 19          # 524,288 — the S unit
H, DK, DV = 32, 128, 128
SPLIT = D // 2       # the 3.5-bit recipe: 50/50 coordinate split


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
    boundaries*norm, y); xhat = fht_adj(centroids[idx]) * norm."""

    def __init__(self, d, bits_lo=3, bits_hi=4, seed=101):
        self.d = d
        self.n_lo = d // 2
        self.signs = rotation_signs(d, seed)
        self.cb_lo = np.load(f"src/rag/codebooks/cb_b{bits_lo}_d{d}.npz")
        self.cb_hi = np.load(f"src/rag/codebooks/cb_b{bits_hi}_d{d}.npz")

    def quant_dequant(self, x):
        x = x.reshape(-1).astype(np.float64)
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
        return xhat * norm


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


def report(label, x, sim):
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
    return total


def main():
    sim = Sim(D)
    print(f"TQ sim at d={D}, bits 3/4 split, seed 101 (the S kind's frame)\n")

    # (a) the test-suite's state: random gaussian, all heads equal
    rng = np.random.default_rng(0)
    x_gauss = rng.standard_normal(D)
    report("RANDOM GAUSSIAN (the suite's state)", x_gauss, sim)
    print()

    # (b) equal-energy delta-rule states (no head spread)
    x_eq = gdn_state(head_norm_spread="equal")
    report("DELTA-RULE, equal head norms", x_eq, sim)
    print()

    # (c) realistic: per-head norm spread (delta-rule beta/norm variation)
    x_real = gdn_state(head_norm_spread="real")
    report("DELTA-RULE, realistic head-norm spread (1e-1.5..1)", x_real, sim)
    print()

    # (d) extreme but plausible after decay: a few dominant heads
    rng = np.random.default_rng(3)
    x_ext = gdn_state(head_norm_spread="real", seed=11).copy()
    x_ext = x_ext.reshape(H, DK, DV)
    x_ext[:28] *= 1e-2        # 28 heads nearly silent
    x_ext = x_ext.reshape(-1)
    report("DELTA-RULE, 4 dominant + 28 silent heads", x_ext, sim)


if __name__ == "__main__":
    main()
