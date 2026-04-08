"""Tests for KDE evaluation functions."""

import numpy as np
import pytest

from hbw import kde_bandwidth, kde_bandwidth_mv, kde_evaluate, kde_evaluate_mv


class TestKDEEvaluate:
    def test_kde_evaluate_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)

        h = kde_bandwidth(x, seed=123)
        x_grid = np.linspace(-3, 3, 100)
        density = kde_evaluate(x, x_grid, h)

        assert density.shape == (100,)
        assert np.all(density >= 0)
        assert np.trapezoid(density, x_grid) > 0.9

    def test_kde_evaluate_normal_shape(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)

        h = kde_bandwidth(x, seed=123)
        x_grid = np.linspace(-4, 4, 200)
        density = kde_evaluate(x, x_grid, h)

        max_idx = np.argmax(density)
        assert -0.5 < x_grid[max_idx] < 0.5

    def test_kde_evaluate_all_kernels(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)

        for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
            h = kde_bandwidth(x, kernel=kernel, seed=123)
            x_grid = np.linspace(-3, 3, 100)
            density = kde_evaluate(x, x_grid, h, kernel=kernel)

            assert density.shape == (100,)
            assert np.all(density >= 0), f"kernel {kernel} produced negative density"

    def test_kde_evaluate_invalid_kernel(self):
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="kernel must be one of"):
            kde_evaluate(x, x, 0.5, kernel="invalid")

    def test_kde_evaluate_integrates_to_one(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)

        h = kde_bandwidth(x, seed=123)
        x_grid = np.linspace(-6, 6, 500)
        density = kde_evaluate(x, x_grid, h)

        integral = np.trapezoid(density, x_grid)
        assert 0.95 < integral < 1.05

    def test_kde_evaluate_same_points(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)

        h = kde_bandwidth(x, seed=123)
        density = kde_evaluate(x, x, h)

        assert density.shape == x.shape
        assert np.all(density > 0)


class TestKDEEvaluateMV:
    def test_kde_evaluate_mv_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 2))

        h = kde_bandwidth_mv(X, seed=123)
        X_grid = rng.standard_normal((50, 2))
        density = kde_evaluate_mv(X, X_grid, h)

        assert density.shape == (50,)
        assert np.all(density >= 0)

    def test_kde_evaluate_mv_same_points(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        h = kde_bandwidth_mv(X, seed=123)
        density = kde_evaluate_mv(X, X, h)

        assert density.shape == (100,)
        assert np.all(density > 0)

    def test_kde_evaluate_mv_all_kernels(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 2))

        for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
            h = kde_bandwidth_mv(X, kernel=kernel, seed=123)
            density = kde_evaluate_mv(X, X[:10], h, kernel=kernel)

            assert density.shape == (10,)
            assert np.all(density >= 0), f"kernel {kernel} produced negative density"

    def test_kde_evaluate_mv_invalid_kernel(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="kernel must be one of"):
            kde_evaluate_mv(X, X, 0.5, kernel="invalid")

    def test_kde_evaluate_mv_dimension_mismatch(self):
        data_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_eval = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="same number of dimensions"):
            kde_evaluate_mv(data_train, data_eval, 0.5)

    def test_kde_evaluate_mv_1d_input(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)

        h = kde_bandwidth_mv(x.reshape(-1, 1), seed=123)
        density = kde_evaluate_mv(x, x, h)
        assert density.shape == (100,)

    def test_kde_evaluate_mv_3d(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 3))

        h = kde_bandwidth_mv(X, seed=123)
        density = kde_evaluate_mv(X, X[:10], h)

        assert density.shape == (10,)
        assert np.all(density >= 0)

    def test_kde_evaluate_mv_peak_at_mode(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 2))

        h = kde_bandwidth_mv(X, seed=123)

        origin = np.array([[0.0, 0.0]])
        far_point = np.array([[5.0, 5.0]])
        density_origin = kde_evaluate_mv(X, origin, h)[0]
        density_far = kde_evaluate_mv(X, far_point, h)[0]

        assert density_origin > density_far
