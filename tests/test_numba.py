"""Tests for Numba-accelerated implementations vs NumPy reference."""

import numpy as np

from hbw._numba_kde import (
    lscv_mv_numba_gauss,
    lscv_numba_biweight,
    lscv_numba_cosine,
    lscv_numba_epan,
    lscv_numba_gauss,
    lscv_numba_triweight,
    lscv_numba_unif,
    lscv_score_numba_biweight,
    lscv_score_numba_cosine,
    lscv_score_numba_epan,
    lscv_score_numba_gauss,
    lscv_score_numba_triweight,
    lscv_score_numba_unif,
)
from hbw._numba_nw import (
    loocv_numba_biweight,
    loocv_numba_cosine,
    loocv_numba_epan,
    loocv_numba_gauss,
    loocv_numba_triweight,
    loocv_numba_unif,
    loocv_score_numba_biweight,
    loocv_score_numba_cosine,
    loocv_score_numba_epan,
    loocv_score_numba_gauss,
    loocv_score_numba_triweight,
    loocv_score_numba_unif,
)
from hbw.kde import kde_bandwidth, kde_bandwidth_mv, lscv, lscv_mv
from hbw.nw import loocv_mse, nw_bandwidth


class TestNumbaKDE:
    def test_lscv_gauss_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np, grad_np, hess_np = lscv(x, h, kernel="gauss")
        score_nb, grad_nb, hess_nb = lscv_numba_gauss(x, h)

        assert abs(score_np - score_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_lscv_epan_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np, grad_np, hess_np = lscv(x, h, kernel="epan")
        score_nb, grad_nb, hess_nb = lscv_numba_epan(x, h)

        assert abs(score_np - score_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_lscv_unif_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np, grad_np, hess_np = lscv(x, h, kernel="unif")
        score_nb, grad_nb, hess_nb = lscv_numba_unif(x, h)

        assert abs(score_np - score_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_lscv_biweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np, grad_np, hess_np = lscv(x, h, kernel="biweight")
        score_nb, grad_nb, hess_nb = lscv_numba_biweight(x, h)

        assert abs(score_np - score_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_lscv_triweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np, grad_np, hess_np = lscv(x, h, kernel="triweight")
        score_nb, grad_nb, hess_nb = lscv_numba_triweight(x, h)

        assert abs(score_np - score_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_lscv_cosine_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np, grad_np, hess_np = lscv(x, h, kernel="cosine")
        score_nb, grad_nb, hess_nb = lscv_numba_cosine(x, h)

        assert abs(score_np - score_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_lscv_score_gauss_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np = lscv(x, h, kernel="gauss")[0]
        score_nb = lscv_score_numba_gauss(x, h)

        assert abs(score_np - score_nb) < 1e-10

    def test_lscv_score_epan_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np = lscv(x, h, kernel="epan")[0]
        score_nb = lscv_score_numba_epan(x, h)

        assert abs(score_np - score_nb) < 1e-10

    def test_lscv_score_unif_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np = lscv(x, h, kernel="unif")[0]
        score_nb = lscv_score_numba_unif(x, h)

        assert abs(score_np - score_nb) < 1e-10

    def test_lscv_score_biweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np = lscv(x, h, kernel="biweight")[0]
        score_nb = lscv_score_numba_biweight(x, h)

        assert abs(score_np - score_nb) < 1e-10

    def test_lscv_score_triweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np = lscv(x, h, kernel="triweight")[0]
        score_nb = lscv_score_numba_triweight(x, h)

        assert abs(score_np - score_nb) < 1e-10

    def test_lscv_score_cosine_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        h = 0.5

        score_np = lscv(x, h, kernel="cosine")[0]
        score_nb = lscv_score_numba_cosine(x, h)

        assert abs(score_np - score_nb) < 1e-10

    def test_kde_bandwidth_runs(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(300)

        h = kde_bandwidth(x, kernel="gauss", seed=123)
        assert h > 0

    def test_kde_bandwidth_all_kernels(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(300)

        for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
            h = kde_bandwidth(x, kernel=kernel, seed=123)
            assert h > 0, f"kernel {kernel} failed"

    def test_lscv_mv_gauss_matches_numpy(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 2))
        h = 0.5

        score_np, grad_np, hess_np = lscv_mv(data, h, kernel="gauss")
        score_nb, grad_nb, hess_nb = lscv_mv_numba_gauss(data, h)

        assert abs(score_np - score_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_kde_bandwidth_mv_runs(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 2))

        h = kde_bandwidth_mv(data, kernel="gauss", seed=123)
        assert h > 0


class TestNumbaNW:
    def test_loocv_gauss_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np, grad_np, hess_np = loocv_mse(x, y, h, kernel="gauss")
        loss_nb, grad_nb, hess_nb = loocv_numba_gauss(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_loocv_epan_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np, grad_np, hess_np = loocv_mse(x, y, h, kernel="epan")
        loss_nb, grad_nb, hess_nb = loocv_numba_epan(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_loocv_unif_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np, grad_np, hess_np = loocv_mse(x, y, h, kernel="unif")
        loss_nb, grad_nb, hess_nb = loocv_numba_unif(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_loocv_biweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np, grad_np, hess_np = loocv_mse(x, y, h, kernel="biweight")
        loss_nb, grad_nb, hess_nb = loocv_numba_biweight(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_loocv_triweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np, grad_np, hess_np = loocv_mse(x, y, h, kernel="triweight")
        loss_nb, grad_nb, hess_nb = loocv_numba_triweight(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_loocv_cosine_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np, grad_np, hess_np = loocv_mse(x, y, h, kernel="cosine")
        loss_nb, grad_nb, hess_nb = loocv_numba_cosine(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10
        assert abs(grad_np - grad_nb) < 1e-10
        assert abs(hess_np - hess_nb) < 1e-10

    def test_loocv_score_gauss_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np = loocv_mse(x, y, h, kernel="gauss")[0]
        loss_nb = loocv_score_numba_gauss(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10

    def test_loocv_score_epan_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np = loocv_mse(x, y, h, kernel="epan")[0]
        loss_nb = loocv_score_numba_epan(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10

    def test_loocv_score_unif_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np = loocv_mse(x, y, h, kernel="unif")[0]
        loss_nb = loocv_score_numba_unif(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10

    def test_loocv_score_biweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np = loocv_mse(x, y, h, kernel="biweight")[0]
        loss_nb = loocv_score_numba_biweight(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10

    def test_loocv_score_triweight_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np = loocv_mse(x, y, h, kernel="triweight")[0]
        loss_nb = loocv_score_numba_triweight(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10

    def test_loocv_score_cosine_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 200)
        y = np.sin(x) + 0.1 * rng.standard_normal(200)
        h = 0.5

        loss_np = loocv_mse(x, y, h, kernel="cosine")[0]
        loss_nb = loocv_score_numba_cosine(x, y, h)

        assert abs(loss_np - loss_nb) < 1e-10

    def test_nw_bandwidth_runs(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 300)
        y = np.sin(x) + 0.1 * rng.standard_normal(300)

        h = nw_bandwidth(x, y, kernel="gauss", seed=123)
        assert h > 0

    def test_nw_bandwidth_all_kernels(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, 300)
        y = np.sin(x) + 0.1 * rng.standard_normal(300)

        for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
            h = nw_bandwidth(x, y, kernel=kernel, seed=123)
            assert h > 0, f"kernel {kernel} failed"
