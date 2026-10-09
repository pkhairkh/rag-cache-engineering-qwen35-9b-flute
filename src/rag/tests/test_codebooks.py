"""test_codebooks.py — paper-constant gates for the Lloyd-Max codebook solver.

Covers the W1.1/W1.2 acceptance gates (TurboQuant, arXiv:2504.19874 §3.1
constants; SPECIFICATION.md §3.1; PROPOSAL.md §1.1 / Phase 1 / D2):

  1. b=1 centroids within 0.5% of ±sqrt(2/pi)/sqrt(d)        (d = 1024, 32768)
  2. b=2 centroids within 1% of ±{0.4528, 1.5104}/sqrt(d)
  3. mse_per_variance in the paper ranges for b = 1..4
     (b=2 additionally within 5% of 0.117)
  4. structure: level/edge counts, ordering, exact symmetry, float32
  5. scaling law: c(d=32768) ≈ c(d=524288) * sqrt(524288/32768) within 0.1%
  6. encode/decode Voronoi self-consistency
  7. disk cache: solve -> save -> load bit-identical; get_codebook hits it
  8. fixed-point convergence (no exception) for every (b, d) pair

Deterministic end to end: the solver has no RNG, so plain asserts suffice.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

import codebooks
from codebooks import get_codebook, load_codebook, solve_codebook

BITS = (1, 2, 3, 4)
D_TEST = 1024                      # test-convenience dim (committed cache too)
D_PROD = (32768, 524288)           # PROPOSAL.md D2 production dims
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
# TurboQuant paper's rounded 4-level constants, sigma-normalized.
B2_LEVELS = (0.4528, 1.5104)


@pytest.fixture(scope="module")
def cb_1024() -> dict[int, codebooks.Codebook]:
    """Fresh solves at the test-convenience dim (one per bit width)."""
    return {b: solve_codebook(b, D_TEST) for b in BITS}


@pytest.fixture(scope="module")
def cb_prod() -> dict[tuple[int, int], codebooks.Codebook]:
    """Production codebooks via the default (committed) cache directory."""
    return {(b, d): get_codebook(b, d) for d in D_PROD for b in BITS}


# --- gate 1: b=1 centroid = E[|X|] -> sqrt(2/pi)/sqrt(d) --------------------- #
def test_b1_centroids_match_sqrt_two_over_pi(cb_1024, cb_prod):
    for cb in (cb_1024[1], cb_prod[(1, 32768)]):
        expected = SQRT_2_OVER_PI / math.sqrt(cb.d)
        assert cb.centroids.shape == (2,)
        rel = np.abs(cb.centroids) / expected - 1.0
        assert np.all(np.abs(rel) < 0.005), f"d={cb.d}: {cb.centroids}"


# --- gate 2: b=2 centroids vs the paper's rounded constants ------------------ #
def test_b2_centroids_match_paper_constants(cb_1024, cb_prod):
    for cb in (cb_1024[2], cb_prod[(2, 32768)]):
        pos = np.abs(cb.centroids[2:]) * math.sqrt(cb.d)  # sigma-normalized
        assert pos.shape == (2,)
        for got, want in zip(pos, B2_LEVELS):
            assert abs(got / want - 1.0) < 0.01, f"d={cb.d}: {got} vs {want}"


# --- gate 3: normalized distortion ranges ------------------------------------ #
def test_mse_per_variance_ranges(cb_prod):
    ranges = {
        1: (0.34, 0.39),
        2: (0.112, 0.123),
        3: (0.028, 0.043),
        4: (0.0075, 0.0107),
    }
    for (b, d), cb in cb_prod.items():
        lo, hi = ranges[b]
        assert lo <= cb.mse_per_variance <= hi, f"b={b} d={d}: {cb.mse_per_variance}"
    # b=2 additionally within 5% of the paper constant 0.117.
    for d in D_PROD:
        m = cb_prod[(2, d)].mse_per_variance
        assert abs(m / 0.117 - 1.0) < 0.05, f"d={d}: {m}"


# --- gate 4: structure, ordering, symmetry, dtypes --------------------------- #
def test_structure_symmetry_and_ordering(cb_1024, cb_prod):
    for cb in list(cb_1024.values()) + list(cb_prod.values()):
        n = 2 ** cb.bits
        assert cb.centroids.dtype == np.float32
        assert cb.centroids.shape == (n,)
        assert cb.boundaries.dtype == np.float32
        assert cb.boundaries.shape == (n - 1,)  # boundary count = 2^b - 1
        assert np.all(np.diff(cb.centroids) > 0)
        if cb.boundaries.size:
            assert np.all(np.diff(cb.boundaries) > 0)
        # exact mirror symmetry by construction (c = -flip(c), b = -flip(b))
        assert np.array_equal(cb.centroids, -cb.centroids[::-1])
        assert np.array_equal(cb.boundaries, -cb.boundaries[::-1])
        # boundaries are the Voronoi midpoints of adjacent centroids
        mid = 0.5 * (cb.centroids[:-1] + cb.centroids[1:])
        assert np.allclose(cb.boundaries, mid, rtol=1e-5, atol=1e-12)
        assert np.all(np.isfinite(cb.centroids))
        assert np.all(np.isfinite(cb.boundaries))


# --- gate 5: the 1/sqrt(d) scaling law --------------------------------------- #
def test_scaling_law(cb_prod):
    factor = math.sqrt(D_PROD[1] / D_PROD[0])  # = 4
    for b in BITS:
        c_small = cb_prod[(b, D_PROD[0])].centroids.astype(np.float64)
        c_big = cb_prod[(b, D_PROD[1])].centroids.astype(np.float64)
        rel = np.abs(c_big * factor / c_small - 1.0)
        assert np.all(rel < 0.001), f"b={b}: {rel}"


# --- gate 6: encode/decode Voronoi consistency ------------------------------- #
def test_encode_decode_voronoi_consistency(cb_1024):
    for b in BITS:
        cb = cb_1024[b]
        c = cb.centroids
        n = 2 ** b

        # every centroid maps to its own index and decodes to itself
        idx = cb.encode(c)
        assert idx.dtype == np.uint8
        assert np.array_equal(idx, np.arange(n, dtype=np.uint8))
        assert np.array_equal(cb.decode(idx), c)

        # points strictly inside a cell map to that cell
        for i in range(n - 1):
            span = c[i + 1] - c[i]
            assert int(cb.encode(c[i] + np.float32(0.4) * span)) == i
            assert int(cb.encode(c[i + 1] - np.float32(0.4) * span)) == i + 1
        # outside the extreme centroids: clipped to the outer cells
        assert int(cb.encode(np.float32(3.0) * c[0])) == 0
        assert int(cb.encode(np.float32(3.0) * c[-1])) == n - 1

        # decode(encode(x)) is the nearest centroid (Voronoi property)
        xs = np.linspace(3.0 * float(c[0]), 3.0 * float(c[-1]), 97)
        out = cb.decode(cb.encode(xs)).astype(np.float64)
        dist = np.abs(xs - out)
        dist_min = np.min(np.abs(xs[:, None] - c[None, :].astype(np.float64)), axis=1)
        assert np.all(dist <= dist_min + 1e-9)


# --- gate 7: disk cache round-trip + cache hit ------------------------------- #
def test_cache_roundtrip_and_hit(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = codebooks.solve_codebook

    def counting(bits, d, iters=500, tol=1e-12):
        calls["n"] += 1
        return orig(bits, d, iters, tol)

    monkeypatch.setattr(codebooks, "solve_codebook", counting)
    cdir = str(tmp_path)
    path = os.path.join(cdir, "cb_b3_d1024.npz")
    assert not os.path.exists(path)

    cb1 = get_codebook(3, D_TEST, cache_dir=cdir)      # solves + saves
    assert calls["n"] == 1
    assert os.path.isfile(path)

    cb2 = get_codebook(3, D_TEST, cache_dir=cdir)      # cache hit: no re-solve
    assert calls["n"] == 1

    # solve -> save -> load is bit-identical
    cb3 = load_codebook(path)
    assert cb3.bits == cb1.bits == 3
    assert cb3.d == cb1.d == D_TEST
    assert np.array_equal(cb3.centroids, cb1.centroids)
    assert np.array_equal(cb3.boundaries, cb1.boundaries)
    assert cb3.mse_per_variance == cb1.mse_per_variance
    # the cache-hit object is bit-identical too
    assert np.array_equal(cb2.centroids, cb1.centroids)
    assert cb2.mse_per_variance == cb1.mse_per_variance


def test_committed_production_caches_are_current(monkeypatch):
    # the 8 production artifacts ship with the repo and are used as-is
    for d in D_PROD:
        for b in BITS:
            path = os.path.join(codebooks.CACHE_DIR, f"cb_b{b}_d{d}.npz")
            assert os.path.isfile(path), f"missing committed cache: {path}"

    calls = {"n": 0}
    orig = codebooks.solve_codebook

    def counting(bits, d, iters=500, tol=1e-12):
        calls["n"] += 1
        return orig(bits, d, iters, tol)

    monkeypatch.setattr(codebooks, "solve_codebook", counting)
    for d in D_PROD:
        for b in BITS:
            get_codebook(b, d)                          # default cache dir
    assert calls["n"] == 0, "committed caches were not hit"

    # and they match a fresh deterministic solve (cache freshness)
    for d in D_PROD:
        for b in BITS:
            cached = get_codebook(b, d)
            fresh = orig(b, d)
            assert np.allclose(cached.centroids, fresh.centroids, rtol=1e-6, atol=0.0)
            assert np.allclose(cached.boundaries, fresh.boundaries, rtol=1e-6, atol=0.0)
            assert abs(cached.mse_per_variance - fresh.mse_per_variance) < 1e-9


# --- gate 8: fixed-point convergence for every (b, d) ------------------------ #
def test_solver_converges_for_all_pairs():
    for d in (D_TEST, *D_PROD):
        for b in BITS:
            cb = solve_codebook(b, d)  # raises RuntimeError if not converged
            assert cb.bits == b and cb.d == d
            assert np.all(np.isfinite(cb.centroids))


# --- loud validation ---------------------------------------------------------- #
def test_validation_is_loud(tmp_path):
    for bad_bits in (0, 5, -1):
        with pytest.raises(ValueError):
            solve_codebook(bad_bits, 32768)
    for bad_d in (2, 1, -8):
        with pytest.raises(ValueError):
            solve_codebook(2, bad_d)
    with pytest.raises(TypeError):
        solve_codebook(2.0, 32768)
    with pytest.raises(TypeError):
        solve_codebook(2, "1024")
    with pytest.raises(ValueError):
        get_codebook(0, 1024)
    with pytest.raises(RuntimeError):
        solve_codebook(4, 1024, iters=5)  # cannot reach the fixed point in 5 iters
    # npz with missing keys is rejected loudly
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, bits=np.int64(2), d=np.int64(1024))
    with pytest.raises(ValueError):
        load_codebook(bad)
