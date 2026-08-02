"""Test that analytic gradients match finite-difference approximations."""

import math
import random

import numpy as np

from hbw import lscv
from hbw.kde import lscv_grad
from hbw.nw import loocv_mse, loocv_mse_grad


def finite_diff(f, h: float, eps: float = 1e-5) -> tuple[float, float]:
    """Compute finite-difference gradient and Hessian."""
    f_plus = f(h + eps)
    f_minus = f(h - eps)
    grad = (f_plus - f_minus) / (2 * eps)
    hess = (f_plus - 2 * f(h) + f_minus) / (eps**2)
    return grad, hess


def test_lscv_derivatives_against_finite_diff() -> None:
    """Verify LSCV analytic gradient matches finite-difference."""
    rng = random.Random(0)
    x = np.array([rng.gauss(0, 1) for _ in range(15)])
    for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
        for h in [0.5, 1.0, 1.5]:
            _, grad, _ = lscv(x, h, kernel)
            k = kernel
            num_grad, _ = finite_diff(lambda hh, k=k: lscv(x, hh, k)[0], h)
            assert math.isclose(grad, num_grad, rel_tol=1e-4, abs_tol=1e-5)


def test_lscv_grad_matches_lscv() -> None:
    """Verify lscv_grad returns same score and gradient as lscv."""
    rng = random.Random(42)
    x = np.array([rng.gauss(0, 1) for _ in range(20)])
    for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
        for h in [0.3, 0.7, 1.2]:
            score_full, grad_full, _ = lscv(x, h, kernel)
            score_grad, grad_grad = lscv_grad(x, h, kernel)
            assert math.isclose(score_full, score_grad, rel_tol=1e-10)
            assert math.isclose(grad_full, grad_grad, rel_tol=1e-10)


def test_loocv_mse_grad_matches_loocv_mse() -> None:
    """Verify loocv_mse_grad returns same loss and gradient as loocv_mse."""
    rng = random.Random(42)
    x = np.array([rng.gauss(0, 1) for _ in range(20)])
    y = np.sin(x) + np.array([rng.gauss(0, 0.1) for _ in range(20)])
    for kernel in ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]:
        for h in [0.3, 0.7, 1.2]:
            loss_full, grad_full, _ = loocv_mse(x, y, h, kernel)
            loss_grad, grad_grad = loocv_mse_grad(x, y, h, kernel)
            assert math.isclose(loss_full, loss_grad, rel_tol=1e-10)
            assert math.isclose(grad_full, grad_grad, rel_tol=1e-10)


KERNELS = ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]


def fd_hessian(f, h: float, eps: float) -> float:
    """Central second difference of f at h."""
    return (f(h + eps) - 2 * f(h) + f(h - eps)) / eps**2


def fd_gradient(f, h: float, eps: float) -> float:
    """Central first difference of f at h."""
    return (f(h + eps) - f(h - eps)) / (2 * eps)


def _sample(seed: int, n: int = 60):
    """Draw a reproducible (x, y) pair for derivative checks."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.sin(x) + 0.2 * rng.normal(size=n)
    return x, y


def test_lscv_hessian_against_finite_diff() -> None:
    """The LSCV Hessian must be the second derivative of the LSCV score.

    The published Hessian used to collapse to exactly -2 * grad / h, because the
    correction term summed a function odd in u over a symmetric matrix and was
    therefore identically zero.
    """
    x, _ = _sample(7)
    for kernel in KERNELS:
        for h in (0.45, 0.8):
            _, _, hess = lscv(x, h, kernel)
            fd = fd_hessian(lambda hh, k=kernel: lscv(x, hh, k)[0], h, 1e-4)
            assert math.isclose(hess, fd, rel_tol=1e-3, abs_tol=1e-6), (
                f"{kernel} h={h}: analytic {hess} vs finite difference {fd}"
            )


def test_lscv_hessian_is_not_proportional_to_gradient() -> None:
    """Guard the specific degeneracy: hess == -2 * grad / h for every kernel."""
    x, _ = _sample(7)
    for kernel in KERNELS:
        _, grad, hess = lscv(x, 0.45, kernel)
        assert not math.isclose(hess, -2 * grad / 0.45, rel_tol=1e-6)


def test_lscv_hessian_converges_quadratically() -> None:
    """Halving the step must cut the finite-difference error roughly fourfold."""
    x, _ = _sample(7)
    _, _, hess = lscv(x, 0.45, "gauss")
    errs = [
        abs(fd_hessian(lambda hh: lscv(x, hh, "gauss")[0], 0.45, eps) - hess)
        for eps in (1e-2, 1e-3)
    ]
    assert errs[1] < errs[0] / 50


def test_loocv_mse_gradient_against_finite_diff() -> None:
    """The LOOCV-MSE gradient must match finite differences for every kernel.

    The cosine weight derivative used to carry a spurious (pi*u/2)**2 * cos term
    and the wrong sign on the sine term.
    """
    x, y = _sample(7)
    for kernel in KERNELS:
        for h in (0.45, 0.8):
            _, grad, _ = loocv_mse(x, y, h, kernel)
            fd = fd_gradient(lambda hh, k=kernel: loocv_mse(x, y, hh, k)[0], h, 1e-4)
            assert math.isclose(grad, fd, rel_tol=1e-3, abs_tol=1e-9), (
                f"{kernel} h={h}: analytic {grad} vs finite difference {fd}"
            )


def test_loocv_mse_hessian_against_finite_diff() -> None:
    """The LOOCV-MSE Hessian must be the second derivative of the loss."""
    x, y = _sample(7)
    for kernel in KERNELS:
        if kernel == "unif":
            continue  # loss is locally constant in h; both sides are ~0
        for h in (0.45, 0.8):
            _, _, hess = loocv_mse(x, y, h, kernel)
            fd = fd_hessian(lambda hh, k=kernel: loocv_mse(x, y, hh, k)[0], h, 1e-4)
            assert math.isclose(hess, fd, rel_tol=1e-3, abs_tol=1e-8), (
                f"{kernel} h={h}: analytic {hess} vs finite difference {fd}"
            )


def test_lscv_mv_hessian_against_finite_diff() -> None:
    """The multivariate LSCV Hessian must match finite differences."""
    from hbw.kde import lscv_mv

    rng = np.random.default_rng(7)
    data = rng.normal(size=(50, 2))
    for h in (0.6, 0.9):
        _, _, hess = lscv_mv(data, h, "gauss")
        fd = fd_hessian(lambda hh: lscv_mv(data, hh, "gauss")[0], h, 1e-4)
        assert math.isclose(hess, fd, rel_tol=1e-4)


def test_loocv_mse_mv_hessian_against_finite_diff() -> None:
    """The multivariate LOOCV-MSE Hessian must match finite differences."""
    from hbw.nw import loocv_mse_mv

    rng = np.random.default_rng(7)
    data = rng.normal(size=(50, 2))
    y = np.sin(data[:, 0]) + 0.2 * rng.normal(size=50)
    for h in (0.6, 0.9):
        _, _, hess = loocv_mse_mv(data, y, h, "gauss")
        fd = fd_hessian(lambda hh: loocv_mse_mv(data, y, hh, "gauss")[0], h, 1e-4)
        assert math.isclose(hess, fd, rel_tol=1e-4)


def test_nw_weight_derivatives_against_finite_diff() -> None:
    """w' and w'' returned by _nw_weights must be derivatives of w in h."""
    from hbw.nw import _nw_weights

    delta = np.array([0.0, 0.05, 0.12, -0.2, 0.31, -0.45, 0.5])
    h, eps = 0.7, 1e-4

    def w_at(hh: float) -> np.ndarray:
        return _nw_weights(delta / hh, hh, kernel)[0]

    for kernel in KERNELS:
        _, w1, w2 = _nw_weights(delta / h, h, kernel)
        fd1 = (w_at(h + eps) - w_at(h - eps)) / (2 * eps)
        fd2 = (w_at(h + eps) - 2 * w_at(h) + w_at(h - eps)) / eps**2
        assert np.max(np.abs(fd1 - w1)) < 1e-5, kernel
        assert np.max(np.abs(fd2 - w2)) < 1e-4, kernel
