"""Tests for Nadaraya-Watson prediction functions."""

import numpy as np
import pytest

from hbw import nw_bandwidth, nw_bandwidth_mv, nw_predict, nw_predict_mv


class TestNWPredict:
    def test_nw_predict_basic(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)

        h = nw_bandwidth(x, y, seed=123)
        y_pred = nw_predict(x, y, x, h)

        assert y_pred.shape == y.shape
        mse = np.mean((y_pred - np.sin(x)) ** 2)
        assert mse < 0.05

    def test_nw_predict_out_of_sample(self):
        rng = np.random.default_rng(42)
        x_train = np.linspace(-2, 2, 200)
        y_train = np.sin(x_train) + 0.1 * rng.standard_normal(200)
        x_test = np.array([-1.5, 0.0, 1.5])

        h = nw_bandwidth(x_train, y_train, seed=123)
        y_pred = nw_predict(x_train, y_train, x_test, h)

        assert y_pred.shape == (3,)
        for i, x_val in enumerate(x_test):
            assert abs(y_pred[i] - np.sin(x_val)) < 0.3

    def test_nw_predict_all_kernels(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)

        for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
            h = nw_bandwidth(x, y, kernel=kernel, seed=123)
            y_pred = nw_predict(x, y, x, h, kernel=kernel)
            assert y_pred.shape == y.shape
            mse = np.mean((y_pred - np.sin(x)) ** 2)
            assert mse < 0.1, f"kernel {kernel} failed with MSE {mse}"

    def test_nw_predict_invalid_kernel(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="kernel must be one of"):
            nw_predict(x, y, x, 0.5, kernel="invalid")

    def test_nw_predict_mismatched_lengths(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            nw_predict(x, y, x, 0.5)

    def test_nw_predict_zero_weights_fallback(self):
        x_train = np.array([0.0, 1.0, 2.0])
        y_train = np.array([1.0, 2.0, 3.0])
        x_test = np.array([100.0])
        y_pred = nw_predict(x_train, y_train, x_test, 0.1, kernel="epan")
        assert np.isclose(y_pred[0], np.mean(y_train))


class TestNWPredictMV:
    def test_nw_predict_mv_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(300)

        h = nw_bandwidth_mv(X, y, seed=123)
        y_pred = nw_predict_mv(X, y, X, h)

        assert y_pred.shape == y.shape
        corr = np.corrcoef(y_pred, y)[0, 1]
        assert corr > 0.8

    def test_nw_predict_mv_out_of_sample(self):
        rng = np.random.default_rng(42)
        data_train = rng.standard_normal((300, 2))
        y_train = np.sin(data_train[:, 0]) + 0.5 * data_train[:, 1] + 0.1 * rng.standard_normal(300)
        data_test = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])

        h = nw_bandwidth_mv(data_train, y_train, seed=123)
        y_pred = nw_predict_mv(data_train, y_train, data_test, h)

        assert y_pred.shape == (3,)

    def test_nw_predict_mv_all_kernels(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(200)

        for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
            h = nw_bandwidth_mv(X, y, kernel=kernel, seed=123)
            y_pred = nw_predict_mv(X, y, X, h, kernel=kernel)
            assert y_pred.shape == y.shape

    def test_nw_predict_mv_invalid_kernel(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="kernel must be one of"):
            nw_predict_mv(X, y, X, 0.5, kernel="invalid")

    def test_nw_predict_mv_mismatched_lengths(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            nw_predict_mv(X, y, X, 0.5)

    def test_nw_predict_mv_dimension_mismatch(self):
        data_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_train = np.array([1.0, 2.0])
        data_test = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="same number of dimensions"):
            nw_predict_mv(data_train, y_train, data_test, 0.5)

    def test_nw_predict_mv_1d_input(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-2, 2, 100)
        y = np.sin(x) + 0.1 * rng.standard_normal(100)

        h = nw_bandwidth_mv(x.reshape(-1, 1), y, seed=123)
        y_pred = nw_predict_mv(x, y, x, h)
        assert y_pred.shape == y.shape

    def test_nw_predict_mv_zero_weights_fallback(self):
        data_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        data_test = np.array([[100.0, 100.0]])
        y_pred = nw_predict_mv(data_train, y_train, data_test, 0.1, kernel="epan")
        assert np.isclose(y_pred[0], np.mean(y_train))
