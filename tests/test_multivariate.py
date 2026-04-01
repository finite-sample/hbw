"""Test multivariate KDE bandwidth selection."""

import math

import numpy as np

from hbw import kde_bandwidth_mv, lscv_mv


def test_lscv_mv_gradient_accuracy() -> None:
    """Verify multivariate LSCV analytic gradient matches finite-difference."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 2))

    for kernel in ["gauss", "epan"]:
        for h in [0.3, 0.5, 0.8]:
            _, grad_a, _ = lscv_mv(X, h, kernel)
            eps = 1e-5
            grad_n = (lscv_mv(X, h + eps, kernel)[0] - lscv_mv(X, h - eps, kernel)[0]) / (2 * eps)
            assert math.isclose(grad_a, grad_n, rel_tol=1e-4)


def test_kde_bandwidth_mv_2d() -> None:
    """Test 2D bandwidth selection matches grid search."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 2))

    for kernel in ["gauss", "epan"]:
        h_newton = kde_bandwidth_mv(X, kernel=kernel, max_n=None)

        grid = np.linspace(0.2, 2.0, 50)
        scores = [lscv_mv(X, h, kernel)[0] for h in grid]
        h_grid = grid[np.argmin(scores)]

        assert math.isclose(h_newton, h_grid, rel_tol=0.1, abs_tol=0.1)


def test_kde_bandwidth_mv_3d() -> None:
    """Test 3D bandwidth selection."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 3))

    h = kde_bandwidth_mv(X, max_n=None)
    assert 0.1 < h < 2.0


def test_kde_bandwidth_mv_subsampling() -> None:
    """Test that subsampling works for large data."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (5000, 2))

    h = kde_bandwidth_mv(X, max_n=500, seed=42)
    assert 0.1 < h < 2.0
