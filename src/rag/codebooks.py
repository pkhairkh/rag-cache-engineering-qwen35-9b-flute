"""codebooks.py — Lloyd-Max scalar codebooks on the sphere-coordinate density.

References: SPECIFICATION.md §3.1 (TurboQuant per-coordinate quantization),
PROPOSAL.md §1.1 (paper fidelity: the Beta coordinate density), Phase 1 (P1
build items) and mapping decision D2 (units of 2^15 / 2^19; data-oblivious
codebooks solved ONCE per (b, d) and cached on disk — the "no calibration"
property of TurboQuant, arXiv:2504.19874).

MATH
----
A random rotation (Hadamard/FHT) of a d-dim unit vector makes every coordinate
marginally distributed as a uniform point on S^(d-1):

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2),
    x in [-1, 1]          (a Beta((d-1)/2, (d-1)/2) law on [-1, 1])

with Var(x) = 1/d exactly. We solve the optimal b-bit scalar quantizer
(Lloyd-Max = continuous 1-D k-means) for this density. Because f is symmetric
and 2^b is even, the optimal codebook is symmetric: levels +/-c_1.. +/-c_K with
K = 2^(b-1) positive centroids, boundaries at 0, +/-b_j, b_j = (c_j+c_{j+1})/2.

NUMERICS (all in scaled coordinates u = x*sqrt(d))
--------------------------------------------------
The scaled density p(u) = f(u/sqrt(d)) / sqrt(d) integrates to 1, has
Var(u) = 1 EXACTLY, and tends to N(0,1); the 1/sqrt(d) scaling law of the
codebook is exact by construction. Working in u avoids the underflow of the
O(sqrt(d))-tall f in x-units. Details:

  * p is evaluated in log space (math.lgamma for the Gamma ratio) and the tail
    is truncated at u = min(sqrt(d), 12) sigma (missing mass < 1e-30).
  * warm start: equal-mass Voronoi cells of the half-density g = 2p on
    [0, u_hi] (cumulative-trapezoid CDF on a dense grid), centroids
    initialized at the conditional cell means.
  * Lloyd update per cell: c_j = E[u | u in cell j] via Gauss-Legendre
    quadrature (numpy.polynomial.legendre.leggauss), vectorized as a
    (cells x nodes) matrix; the plain 1/D integrals have no closed form
    (incomplete beta), so quadrature is the pragmatic choice.
  * convergence is linear (rate ~0.97 for K=8), so a vector Aitken
    extrapolation of the dominant mode is applied every _AITKEN_PERIOD
    iterations (guarded: positivity, ordering, gain cap; the returned point is
    always a genuine Lloyd iterate verified to move < tol).
  * mse_per_variance = E[(u - c(u))^2] / Var(u) = E[(u - c(u))^2] since
    Var(u) = 1 exactly; equivalently d * E[(x - c(x))^2] in x-units.

The solver is pure numpy + math + stdlib (no torch) and fully deterministic
(no RNG), so the cached .npz artifacts are reproducible bit-for-bit and are
reused verbatim on the GPU box.

Cache layout: <this dir>/codebooks/cb_b{bits}_d{d}.npz with keys
bits, d, centroids, boundaries, mse_per_variance.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Codebook",
    "solve_codebook",
    "get_codebook",
    "load_codebook",
    "CACHE_DIR",
    "PRODUCTION_DIMS",
]

# Directory holding the committed per-(b, d) codebook caches (relative to this file).
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codebooks")

# Production unit dims (PROPOSAL.md D2: 2^15 / 2^19 single-segment FHT). Both
# powers of two; the solver itself accepts ANY d >= 3 (d = 3 is the uniform
# density on [-sqrt(3), sqrt(3)]).
PRODUCTION_DIMS = (2 ** 15, 2 ** 19)

_QUAD_NODES = 1024        # Gauss-Legendre nodes per Voronoi cell (>= 512, cheap)
_INIT_GRID = 20001        # dense grid for the equal-mass warm start
_TAIL_SIGMAS = 12.0       # support truncation in sigma=1 units (tail mass < 1e-30)
_AITKEN_PERIOD = 20       # Aitken extrapolation cadence (iterations)
_AITKEN_GAIN_CAP = 100.0  # trust cap on the extrapolation gain lam/(1-lam)
_MASS_RTOL = 1e-6         # self-check: total mass of the discretized density


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _validate_bits_d(bits: int, d: int) -> tuple[int, int]:
    """Loud validation shared by every public entry point."""
    for name, val in (("bits", bits), ("d", d)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise TypeError(f"{name} must be an int, got {type(val).__name__}: {val!r}")
    bits, d = int(bits), int(d)
    if not 1 <= bits <= 4:
        raise ValueError(
            f"bits must be in 1..4 (got {bits}): TurboQuant uses b in 1..4 and the "
            "symmetric +/- level layout requires an even level count"
        )
    # The symmetric layout (no zero level, boundary at 0) needs 2**bits even.
    # This holds for every b >= 1; the assert pins the invariant loud and early.
    assert 2 ** bits % 2 == 0, "symmetric codebook layout requires an even level count"
    if d < 3:
        raise ValueError(
            f"d must be >= 3 (got {d}): the density exponent (d-3)/2 and Var(x)=1/d "
            "are defined for the sphere S^(d-1) with d >= 3"
        )
    return bits, d


# --------------------------------------------------------------------------- #
# The scaled coordinate density
# --------------------------------------------------------------------------- #
def _log_norm(d: int) -> float:
    """log C(d), the normalizer of p(u) = C(d) * (1 - u^2/d)^((d-3)/2)."""
    return (
        -0.5 * math.log(d)
        + math.lgamma(d / 2.0)
        - math.lgamma((d - 1) / 2.0)
        - 0.5 * math.log(math.pi)
    )


def _scaled_density(u: np.ndarray, d: int) -> np.ndarray:
    """p(u) = f(u/sqrt(d)) / sqrt(d): the coordinate density in sigma=1 units.

    Integrates to 1 over [-sqrt(d), sqrt(d)] and Var(u) = 1 exactly
    (E[x^2] = 1/d for a uniform point on S^(d-1)). Zero outside the support.
    Evaluated in log space so huge d never over/underflows.
    """
    a = 0.5 * (d - 3.0)
    s = (u * u) / d
    with np.errstate(divide="ignore", invalid="ignore"):  # s == 1 edge is masked below
        logf = _log_norm(d) + a * np.log1p(-np.minimum(s, 1.0))
        out = np.exp(logf)
    return np.where(s < 1.0, out, 0.0)


# --------------------------------------------------------------------------- #
# Lloyd-Max machinery (positive half [0, u_hi], half-density g = 2p)
# --------------------------------------------------------------------------- #
def _warm_start_centroids(K: int, d: int, u_hi: float) -> np.ndarray:
    """Equal-mass cells of g on [0, u_hi]; centroids = conditional cell means.

    Uses a dense grid + cumulative trapezoid CDF (init only — Lloyd then
    iterates with exact Gauss-Legendre cell moments, so grid error only costs
    iterations, never accuracy).
    """
    u = np.linspace(0.0, u_hi, _INIT_GRID)
    w = 2.0 * _scaled_density(u, d)
    h = 0.5 * (u[1] - u[0])
    cw = np.empty_like(u)     # cumulative mass of g
    cw[0] = 0.0
    np.cumsum((w[1:] + w[:-1]) * h, out=cw[1:])
    cuw = np.empty_like(u)    # cumulative first moment (u * g)
    cuw[0] = 0.0
    np.cumsum((u[1:] * w[1:] + u[:-1] * w[:-1]) * h, out=cuw[1:])
    if K == 1:
        edges = np.array([0.0, u_hi])
    else:
        targets = np.arange(1, K) * (cw[-1] / K)
        edges = np.concatenate(([0.0], np.interp(targets, cw, u), [u_hi]))
    m0 = np.interp(edges, u, cw)
    m1 = np.interp(edges, u, cuw)
    c = (m1[1:] - m1[:-1]) / (m0[1:] - m0[:-1])
    if not (np.all(np.diff(c) > 0) and np.all(c > 0) and c[-1] < u_hi):
        raise RuntimeError(f"warm start produced an invalid centroid set: {c!r}")
    return c


def _cell_edges(centroids: np.ndarray, u_hi: float) -> np.ndarray:
    """Voronoi edges of the positive half: [0, b_1, ..., b_{K-1}, u_hi]."""
    b = 0.5 * (centroids[:-1] + centroids[1:])
    return np.concatenate(([0.0], b, [u_hi]))


def _cell_moments(
    edges: np.ndarray, d: int, X: np.ndarray, Wgl: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre moments over each cell: returns (U, F*quad, mass, first).

    U is the (cells x nodes) node matrix, `F*quad` the weighted density, and
    mass/first the per-cell zeroth/first moments of the half-density g.
    """
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    U = mid[:, None] + half[:, None] * X[None, :]
    F = _scaled_density(U, d)
    quad = Wgl[None, :] * half[:, None]
    wq = F * quad
    mass = wq.sum(axis=1)
    first = (wq * U).sum(axis=1)
    return U, wq, mass, first


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Codebook:
    """A solved per-(b, d) Lloyd-Max codebook for one scalar coordinate.

    Attributes:
        bits:             b in {1, 2, 3, 4} (2**b levels, all even -> symmetric).
        d:                dimension of the density (2**15 / 2**19 in production;
                          any d >= 3 is solvable).
        centroids:        float32, shape (2**bits,), ASCENDING, symmetric +/-.
        boundaries:       float32, shape (2**bits - 1,), ascending Voronoi edges
                          (boundaries[i] = (centroids[i] + centroids[i+1]) / 2).
        mse_per_variance: E[(X - c(X))^2] / Var(X); Var(X) = 1/d, so this is
                          dimensionless (~0.3634 / 0.1175 / 0.03454 / 0.009497
                          for b = 1..4 — the paper's D_mse constants).

    encode maps x to the index of its Voronoi cell (np.searchsorted with
    side='left': cell i = (boundaries[i-1], boundaries[i]]), decode maps an
    index back to its centroid.
    """

    bits: int
    d: int
    centroids: np.ndarray
    boundaries: np.ndarray
    mse_per_variance: float

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Vector -> uint8 cell indices (np.searchsorted(boundaries, x))."""
        idx = np.searchsorted(self.boundaries, np.asarray(x))
        return idx.astype(np.uint8)

    def decode(self, idx: np.ndarray) -> np.ndarray:
        """uint8 cell indices -> centroid values (float32)."""
        return self.centroids[np.asarray(idx)]


def solve_codebook(bits: int, d: int, iters: int = 500, tol: float = 1e-12) -> Codebook:
    """Solve the Lloyd-Max quantizer of the sphere-coordinate density.

    Deterministic (no RNG). Raises RuntimeError if the fixed point is not
    reached within `iters` (a successful return IS the convergence flag), or
    on any internal self-check failure (empty cell, density normalization).

    Args:
        bits: b in {1, 2, 3, 4}.
        d:    density dimension, any int >= 3 (production: 2**15, 2**19).
        iters: Lloyd iteration cap (default 500; b=4 converges in ~200 thanks
            to Aitken acceleration, ~670 without).
        tol:  fixed-point tolerance on max |centroid change| per iteration,
            measured in sigma=1 (scaled) units.
    """
    bits, d = _validate_bits_d(bits, d)
    if isinstance(iters, bool) or not isinstance(iters, (int, np.integer)) or int(iters) < 1:
        raise ValueError(f"iters must be a positive int, got {iters!r}")
    if not (isinstance(tol, (float, int, np.floating)) and tol > 0):
        raise ValueError(f"tol must be > 0, got {tol!r}")
    iters, tol = int(iters), float(tol)

    K = 2 ** (bits - 1)                     # positive half of the level set
    u_hi = min(math.sqrt(d), _TAIL_SIGMAS)  # 12 sigma; tail mass < 1e-30
    X, Wgl = np.polynomial.legendre.leggauss(_QUAD_NODES)

    c = _warm_start_centroids(K, d, u_hi)
    prev_delta: np.ndarray | None = None
    converged = False
    step = math.inf
    for it in range(1, iters + 1):
        edges = _cell_edges(c, u_hi)
        U, wq, mass, first = _cell_moments(edges, d, X, Wgl)
        if not np.all(mass > 0.0):
            raise RuntimeError(
                f"empty Voronoi cell at iteration {it} (bits={bits}, d={d})"
            )
        c_new = first / mass
        delta = c_new - c
        step = float(np.max(np.abs(delta)))
        c = c_new
        if step < tol:
            converged = True
            break
        # Vector Aitken extrapolation of the dominant (slow, rate ~0.97 for
        # K=8) linear mode. Guarded: only applied while it keeps the centroid
        # set valid; the next Lloyd step re-verifies it, so a bad jump can
        # only cost iterations, never accuracy.
        if prev_delta is not None and it % _AITKEN_PERIOD == 0:
            denom = float(prev_delta @ prev_delta)
            if denom > 0.0:
                lam = float(delta @ prev_delta) / denom
                if 0.0 < lam < 1.0:
                    gain = min(lam / (1.0 - lam), _AITKEN_GAIN_CAP)
                    c_ext = c + gain * delta
                    if (
                        np.all(np.diff(c_ext) > 0)
                        and np.all(c_ext > 0)
                        and c_ext[-1] < u_hi
                    ):
                        c = c_ext
        prev_delta = delta
    if not converged:
        raise RuntimeError(
            f"Lloyd iteration did not reach tol={tol:g} within iters={iters} "
            f"(bits={bits}, d={d}; last step {step:.3e})"
        )

    # Final partition: recompute moments once and run the self-checks.
    edges = _cell_edges(c, u_hi)
    U, wq, mass, first = _cell_moments(edges, d, X, Wgl)
    total_mass = 2.0 * float(mass.sum())
    if abs(total_mass - 1.0) > _MASS_RTOL:
        raise RuntimeError(
            f"density normalization self-check failed (mass={total_mass!r}, "
            f"bits={bits}, d={d})"
        )
    residual = float(np.max(np.abs(first / mass - c)))
    if residual > 100.0 * tol:
        raise RuntimeError(
            f"fixed-point self-check failed (residual={residual:.3e}, "
            f"bits={bits}, d={d})"
        )

    # Normalized distortion: E[(u - c(u))^2] with Var(u) = 1 exactly, mirrored
    # over both halves. In x-units this is d * E[(x - c(x))^2] = D_mse / Var.
    mse = 2.0 * float((((U - c[:, None]) ** 2) * wq).sum())

    # Symmetric assembly: centroids -c_K..-c_1, c_1..c_K; boundaries
    # -b_{K-1}..-b_1, 0, b_1..b_{K-1}  (2**bits - 1 edges in total).
    b_half = 0.5 * (c[:-1] + c[1:])
    centroids_u = np.concatenate((-c[::-1], c))
    boundaries_u = np.concatenate((-b_half[::-1], [0.0], b_half))

    s = math.sqrt(d)  # the 1/sqrt(d) scaling law, made explicit
    return Codebook(
        bits=bits,
        d=d,
        centroids=(centroids_u / s).astype(np.float32),
        boundaries=(boundaries_u / s).astype(np.float32),
        mse_per_variance=float(mse),
    )


def get_codebook(bits: int, d: int, cache_dir: str | None = None) -> Codebook:
    """Return the (bits, d) codebook, solving and caching it if needed.

    `cache_dir` defaults to `<this module's dir>/codebooks` (the committed
    artifact directory). Cache files are cb_b{bits}_d{d}.npz with keys
    bits, d, centroids, boundaries, mse_per_variance. Deterministic: a cache
    hit returns exactly what solve_codebook would produce.
    """
    bits, d = _validate_bits_d(bits, d)
    cdir = str(CACHE_DIR if cache_dir is None else os.fspath(cache_dir))
    path = os.path.join(cdir, f"cb_b{bits}_d{d}.npz")
    if os.path.isfile(path):
        cb = load_codebook(path)
        if cb.bits != bits or cb.d != d:
            raise RuntimeError(
                f"cache file {path} holds (bits={cb.bits}, d={cb.d}) but "
                f"(bits={bits}, d={d}) was requested"
            )
        return cb
    cb = solve_codebook(bits, d)
    os.makedirs(cdir, exist_ok=True)
    np.savez(
        path,
        bits=np.int64(cb.bits),
        d=np.int64(cb.d),
        centroids=cb.centroids,
        boundaries=cb.boundaries,
        mse_per_variance=np.float64(cb.mse_per_variance),
    )
    return cb


def load_codebook(path: str) -> Codebook:
    """Load a codebook from an .npz cache file (loudly validated)."""
    with np.load(os.fspath(path)) as z:
        required = ("bits", "d", "centroids", "boundaries", "mse_per_variance")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"{path}: npz is missing keys {missing}")
        bits = int(z["bits"])
        d = int(z["d"])
        centroids = np.asarray(z["centroids"])
        boundaries = np.asarray(z["boundaries"])
        mse = float(z["mse_per_variance"])
    bits, d = _validate_bits_d(bits, d)

    if centroids.dtype != np.float32 or centroids.shape != (2 ** bits,):
        raise ValueError(
            f"{path}: centroids must be float32 of shape ({2 ** bits},), "
            f"got {centroids.dtype} {centroids.shape}"
        )
    if boundaries.dtype != np.float32 or boundaries.shape != (2 ** bits - 1,):
        raise ValueError(
            f"{path}: boundaries must be float32 of shape ({2 ** bits - 1},), "
            f"got {boundaries.dtype} {boundaries.shape}"
        )
    if not np.all(np.diff(centroids) > 0):
        raise ValueError(f"{path}: centroids are not strictly ascending")
    if not np.array_equal(centroids, -centroids[::-1]):
        raise ValueError(f"{path}: centroids are not symmetric (c = -flip(c))")
    if boundaries.size and not np.all(np.diff(boundaries) > 0):
        raise ValueError(f"{path}: boundaries are not strictly ascending")
    if not np.array_equal(boundaries, -boundaries[::-1]):
        raise ValueError(f"{path}: boundaries are not symmetric (b = -flip(b))")
    if not np.allclose(
        boundaries, 0.5 * (centroids[:-1] + centroids[1:]), rtol=1e-5, atol=1e-12
    ):
        raise ValueError(f"{path}: boundaries are not the Voronoi edges of the centroids")
    if not (math.isfinite(mse) and 0.0 < mse < 1.0):
        raise ValueError(f"{path}: mse_per_variance out of range: {mse!r}")
    return Codebook(
        bits=bits, d=d, centroids=centroids, boundaries=boundaries, mse_per_variance=mse
    )
