"""Test that Newton optimization matches grid search minimum."""

import math
import random

import numpy as np
from scipy.optimize import minimize_scalar

from hbw import kde_bandwidth, lscv, nw_bandwidth
from hbw.kde import lscv_score
from hbw.nw import loocv_mse_score


def mixture_sample(n: int, rng: random.Random) -> np.ndarray:
    """Generate samples from a Gaussian mixture."""
    samples = []
    for _ in range(n):
        if rng.random() < 0.5:
            samples.append(rng.gauss(-2, 0.5))
        else:
            samples.append(rng.gauss(2, 1.0))
    return np.array(samples)


def test_newton_matches_grid_search() -> None:
    """Verify Newton finds approximately the same optimum as grid search."""
    rng = random.Random(1)
    x = mixture_sample(30, rng)
    grid = [0.1 + i * (2.0 - 0.1) / 39 for i in range(40)]
    for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
        scores = [lscv(x, h, kernel)[0] for h in grid]
        h_grid = grid[scores.index(min(scores))]

        h_newton = kde_bandwidth(x, kernel=kernel, h0=h_grid * 1.1, max_n=None)
        assert math.isclose(h_newton, h_grid, rel_tol=0.1, abs_tol=0.05)


def test_gaussian_selectors_match_scipy_bounded_minimization() -> None:
    """Gaussian selectors must match an independent derivative-free optimizer."""
    rng = np.random.default_rng(2026)
    x = rng.normal(size=300)
    y = np.sin(x) + 0.2 * rng.normal(size=300)
    scale = np.std(x, ddof=1)
    bounds = (math.log(0.01 * scale), math.log(3.0 * scale))

    kde_reference = minimize_scalar(
        lambda log_h: lscv_score(x, math.exp(log_h)),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-11},
    )
    assert kde_reference.success
    h_kde = kde_bandwidth(x, max_n=None)
    assert math.isclose(h_kde, math.exp(kde_reference.x), rel_tol=1e-3)
    assert lscv_score(x, h_kde) <= kde_reference.fun + 1e-9

    nw_reference = minimize_scalar(
        lambda log_h: loocv_mse_score(x, y, math.exp(log_h)),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-11},
    )
    assert nw_reference.success
    h_nw = nw_bandwidth(x, y, max_n=None)
    assert math.isclose(h_nw, math.exp(nw_reference.x), rel_tol=1e-3)
    assert loocv_mse_score(x, y, h_nw) <= nw_reference.fun + 1e-9


def test_selectors_match_dense_log_grid_objectives() -> None:
    """Optimized scores must agree with a slow dense-grid oracle across kernels."""
    rng = np.random.default_rng(31)
    x = np.concatenate((rng.normal(-1.0, 0.45, 40), rng.normal(1.0, 0.7, 40)))
    y = np.sin(x) + 0.2 * rng.normal(size=len(x))
    grid = np.geomspace(0.02, 3.0, 800)

    for kernel in ALL_KERNELS:
        h = kde_bandwidth(x, kernel=kernel, max_n=None)
        selected = lscv_score(x, h, kernel)
        grid_min = min(lscv_score(x, candidate, kernel) for candidate in grid)
        assert selected - grid_min <= 1e-3 * abs(grid_min), kernel

    for kernel in ("gauss", "epan", "biweight", "triweight", "cosine"):
        h = nw_bandwidth(x, y, kernel=kernel, max_n=None)
        selected = loocv_mse_score(x, y, h, kernel)
        grid_min = min(loocv_mse_score(x, y, candidate, kernel) for candidate in grid)
        assert selected - grid_min <= 1e-3 * grid_min, kernel


def test_kde_bandwidth_basic() -> None:
    """Test basic functionality of kde_bandwidth."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    h = kde_bandwidth(x)
    assert 0.1 < h < 2.0


def test_kde_bandwidth_subsampling() -> None:
    """Test that subsampling works for large data."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10000)
    h = kde_bandwidth(x, max_n=500, seed=42)
    assert 0.1 < h < 2.0


ALL_KERNELS = ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]


def test_lscv_objective_is_scale_equivariant() -> None:
    """c * LSCV(c*x, c*h) == LSCV(x, h): the identity the selector must inherit."""
    from hbw.kde import lscv_score

    x = np.random.default_rng(11).normal(size=200)
    for kernel in ALL_KERNELS:
        base = lscv_score(x, 0.35, kernel)
        for c in (1e-3, 10.0, 1e4):
            scaled = c * lscv_score(c * x, c * 0.35, kernel)
            assert math.isclose(scaled, base, rel_tol=1e-12)


def test_kde_bandwidth_is_scale_equivariant() -> None:
    """kde_bandwidth(c*x) must equal c * kde_bandwidth(x).

    The LSCV gradient scales like 1/c**2, so an absolute gradient tolerance made
    large-unit data converge immediately and return Silverman's rule untouched.
    """
    from hbw import kde_bandwidth

    x = np.random.default_rng(11).normal(size=400)
    for kernel in ALL_KERNELS:
        h1 = kde_bandwidth(x, kernel=kernel)
        for c in (1e-6, 1e-4, 1e-2, 100.0, 1000.0, 1e5):
            hc = kde_bandwidth(c * x, kernel=kernel)
            assert math.isclose(hc / c, h1, rel_tol=1e-6), (
                f"{kernel} c={c}: {hc / c} vs {h1}"
            )


def test_nw_bandwidth_is_scale_equivariant() -> None:
    """nw_bandwidth(c*x, y) must equal c * nw_bandwidth(x, y)."""
    from hbw import nw_bandwidth

    rng = np.random.default_rng(11)
    x = rng.normal(size=400)
    y = np.sin(x) + 0.2 * rng.normal(size=400)
    for kernel in ALL_KERNELS:
        h1 = nw_bandwidth(x, y, kernel=kernel)
        for c in (1e-6, 1e-4, 1e-2, 100.0, 1000.0, 1e5):
            hc = nw_bandwidth(c * x, y, kernel=kernel)
            assert math.isclose(hc / c, h1, rel_tol=1e-6), (
                f"{kernel} c={c}: {hc / c} vs {h1}"
            )


def test_numba_loocv_score_matches_numpy_when_weights_underflow() -> None:
    """A bandwidth so small that every weight underflows must not score as zero."""
    from hbw._numba_nw import loocv_numba_gauss, loocv_score_numba_gauss
    from hbw.nw import loocv_mse_score

    rng = np.random.default_rng(42)
    x = np.linspace(-2, 2, 200)
    y = np.sin(x) + 0.1 * rng.standard_normal(200)
    for h in (1e-6, 1e-4):
        expected = loocv_mse_score(x, y, h, "gauss")
        assert expected > 0.1
        assert math.isclose(loocv_score_numba_gauss(x, y, h), expected, rel_tol=1e-12)
        assert math.isclose(loocv_numba_gauss(x, y, h)[0], expected, rel_tol=1e-12)


def test_grad_converged_handles_a_criterion_passing_through_zero():
    """A zero criterion value used to make convergence unreachable.

    _grad_converged scaled the gradient test by abs(f) and returned False
    outright when that scale was zero, so an iterate sitting at f == 0 with a
    negligible gradient could never converge and burned the whole iteration
    budget. It now tests the relative Newton step at that single point.
    """
    from hbw._optim import _grad_converged

    assert _grad_converged(g=1e-12, h=1.0, f=0.0, hess=1.0, tol=1e-5)
    assert not _grad_converged(g=1.0, h=1.0, f=0.0, hess=1.0, tol=1e-5)
    assert _grad_converged(g=0.0, h=1.0, f=0.0, hess=0.0, tol=1e-5)
    assert not _grad_converged(g=1e-12, h=1.0, f=0.0, hess=-1.0, tol=1e-5)
    # A non-zero scale is unaffected.
    assert _grad_converged(g=1e-12, h=1.0, f=2.0, hess=1.0, tol=1e-5)
    assert not _grad_converged(g=1e-4, h=1.0, f=1.0, hess=1.0, tol=1e-5)


def test_grad_converged_at_zero_is_scale_equivariant() -> None:
    """Equivalent LSCV states must make the same convergence decision at f=0."""
    from hbw._optim import _grad_converged

    for g, expected in ((1e-6, True), (1e-4, False)):
        base = _grad_converged(g=g, h=1.0, f=0.0, hess=1.0, tol=1e-5)
        assert base is expected
        for c in (1e-6, 1e6):
            scaled = _grad_converged(
                g=g / c**2,
                h=c,
                f=0.0,
                hess=1.0 / c**3,
                tol=1e-5,
            )
            assert scaled is expected


def test_kde_bandwidth_mv_is_scale_equivariant_without_standardization() -> None:
    """The multivariate optimizer floor must scale with the input units."""
    from hbw import kde_bandwidth_mv

    data = np.random.default_rng(11).normal(size=(200, 2))
    h1 = kde_bandwidth_mv(data, standardize=False)
    for c in (1e-6, 1e6):
        hc = kde_bandwidth_mv(c * data, standardize=False)
        assert math.isclose(hc / c, h1, rel_tol=1e-6)
