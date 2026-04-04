"""Test multivariate KDE and NW bandwidth selection."""

import math

import numpy as np

from hbw import kde_bandwidth_mv, loocv_mse_mv, lscv_mv, nw_bandwidth_mv


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


def test_loocv_mse_mv_gradient_accuracy() -> None:
    """Verify multivariate LOOCV-MSE analytic gradient matches finite-difference."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * rng.normal(0, 1, 100)

    for h in [0.3, 0.5, 0.8]:
        _, grad_a, _ = loocv_mse_mv(X, y, h, "gauss")
        eps = 1e-5
        grad_n = (loocv_mse_mv(X, y, h + eps, "gauss")[0] - loocv_mse_mv(X, y, h - eps, "gauss")[0]) / (2 * eps)
        assert math.isclose(grad_a, grad_n, rel_tol=1e-4)


def test_nw_bandwidth_mv_2d() -> None:
    """Test 2D NW bandwidth selection matches grid search."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.3 * rng.normal(0, 1, 300)

    h_newton = nw_bandwidth_mv(X, y, kernel="gauss", max_n=None)

    grid = np.linspace(0.2, 2.0, 50)
    scores = [loocv_mse_mv(X, y, h, "gauss")[0] for h in grid]
    h_grid = grid[np.argmin(scores)]

    assert math.isclose(h_newton, h_grid, rel_tol=0.15, abs_tol=0.15)


def test_nw_bandwidth_mv_3d() -> None:
    """Test 3D NW bandwidth selection."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.3 * rng.normal(0, 1, 300)

    h = nw_bandwidth_mv(X, y, max_n=None)
    assert 0.1 < h < 2.0


def test_nw_bandwidth_mv_subsampling() -> None:
    """Test that subsampling works for large data."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (5000, 2))
    y = np.sin(X[:, 0]) + 0.3 * rng.normal(0, 1, 5000)

    h = nw_bandwidth_mv(X, y, max_n=500, seed=42)
    assert 0.1 < h < 2.0


def test_loocv_mse_mv_numba_matches_numpy() -> None:
    """Verify that Numba implementation matches NumPy for various bandwidths."""
    from hbw._numba_nw import loocv_mv_numba_gauss, loocv_score_mv_numba_gauss

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * rng.normal(0, 1, 100)

    for h in [0.3, 0.5, 0.8, 1.0, 1.5]:
        loss_np, grad_np, hess_np = loocv_mse_mv(X, y, h, "gauss")
        loss_nb, grad_nb, hess_nb = loocv_mv_numba_gauss(X, y, h)
        score_nb = loocv_score_mv_numba_gauss(X, y, h)

        assert math.isclose(loss_np, loss_nb, rel_tol=1e-10), f"Loss mismatch at h={h}"
        assert math.isclose(grad_np, grad_nb, rel_tol=1e-10), f"Gradient mismatch at h={h}"
        assert math.isclose(hess_np, hess_nb, rel_tol=1e-10), f"Hessian mismatch at h={h}"
        assert math.isclose(loss_np, score_nb, rel_tol=1e-10), f"Score-only mismatch at h={h}"


def test_nw_bandwidth_mv_matches_grid_search() -> None:
    """Verify Newton optimization matches or beats grid search on multiple DGPs."""
    for seed in [42, 123, 456]:
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (300, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.3 * rng.normal(0, 1, 300)

        h_newton = nw_bandwidth_mv(X, y, kernel="gauss", max_n=None, standardize=False)

        grid = np.linspace(0.1, 2.5, 200)
        scores = [loocv_mse_mv(X, y, h, "gauss")[0] for h in grid]
        h_grid = grid[np.argmin(scores)]
        min_score_grid = min(scores)

        score_newton = loocv_mse_mv(X, y, h_newton, "gauss")[0]

        assert score_newton <= min_score_grid + 1e-10, f"Newton should find <= grid score at seed {seed}"
        assert abs(h_newton - h_grid) / h_grid < 0.15, f"Bandwidth diff > 15% at seed {seed}"


def test_nw_bandwidth_mv_newton_vs_numba_consistency() -> None:
    """Verify Newton optimization with numpy vs numba gives consistent results."""
    from hbw.nw import _newton_armijo_mv_nw, _newton_armijo_mv_nw_numba, _scott_h_mv_nw

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.3 * rng.normal(0, 1, 200)

    h0 = _scott_h_mv_nw(X)
    h_numpy = _newton_armijo_mv_nw(X, y, h0, "gauss")
    h_numba = _newton_armijo_mv_nw_numba(X, y, h0)

    assert math.isclose(h_numpy, h_numba, rel_tol=0.01)
