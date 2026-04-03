"""NW bandwidth selection via LOOCV-MSE minimization."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import _KERNELS, _SQRT_2PI
from ._numba_nw import (
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
from ._optim import _newton_armijo, _silverman_h, _subsample


def _nw_weights(
    u: NDArray[Any],
    h: float,
    kernel: str,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return weights w, w', w'' for Nadaraya-Watson."""
    if kernel == "gauss":
        base = np.exp(-0.5 * u * u) / (h * _SQRT_2PI)
        w1 = base * (u * u - 1) / h
        w2 = base * (u**4 - 3 * u * u + 1) / (h * h)
        return base, w1, w2
    elif kernel == "epan":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        w2 = np.zeros_like(u, dtype=float)
        uu = u * u
        w[mask] = 0.75 * (1 - uu[mask]) / h
        w1[mask] = 0.75 * (-1 + 3 * uu[mask]) / (h * h)
        w2[mask] = 1.5 * (1 - 6 * uu[mask]) / (h**3)
        return w, w1, w2
    elif kernel == "unif":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w[mask] = 0.5 / h
        w1 = np.zeros_like(u, dtype=float)
        w2 = np.zeros_like(u, dtype=float)
        return w, w1, w2
    elif kernel == "biweight":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        w2 = np.zeros_like(u, dtype=float)
        uu = u * u
        one_minus_uu = 1 - uu
        w[mask] = (15 / 16) * one_minus_uu[mask] ** 2 / h
        w1[mask] = (15 / 16) * one_minus_uu[mask] * (5 * uu[mask] - 1) / (h * h)
        w2[mask] = (15 / 8) * (1 - 12 * uu[mask] + 20 * uu[mask] ** 2 - 5 * uu[mask] ** 3) / (h**3)
        return w, w1, w2
    elif kernel == "triweight":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        w2 = np.zeros_like(u, dtype=float)
        uu = u * u
        one_minus_uu = 1 - uu
        w[mask] = (35 / 32) * one_minus_uu[mask] ** 3 / h
        w1[mask] = (35 / 32) * one_minus_uu[mask] ** 2 * (7 * uu[mask] - 1) / (h * h)
        w2[mask] = (
            (35 / 16) * one_minus_uu[mask] * (1 - 20 * uu[mask] + 35 * uu[mask] ** 2) / (h**3)
        )
        return w, w1, w2
    elif kernel == "cosine":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        w2 = np.zeros_like(u, dtype=float)
        uu = u * u
        cos_val = np.cos(np.pi * u / 2)
        sin_val = np.sin(np.pi * u / 2)
        pi = np.pi
        w[mask] = (pi / 4) * cos_val[mask] / h
        w1[mask] = (
            (pi / 4)
            * ((pi**2 * uu[mask] / 4 - 1) * cos_val[mask] - (pi * u[mask] / 2) * sin_val[mask])
            / (h * h)
        )
        w2[mask] = (
            (pi / 4)
            * (
                (2 - 3 * pi**2 * uu[mask] / 2 + pi**4 * uu[mask] ** 2 / 16) * cos_val[mask]
                + (3 * pi * u[mask] / 2 - pi**3 * uu[mask] * u[mask] / 8) * sin_val[mask]
            )
            / (h**3)
        )
        return w, w1, w2
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def loocv_mse_score(
    x: NDArray[Any],
    y: NDArray[Any],
    h: float,
    kernel: str = "gauss",
) -> float:
    """Compute only the LOOCV MSE (no gradient/Hessian).

    This is more efficient for grid search where only the score is needed.

    Parameters
    ----------
    x
        Predictor values (1D array).
    y
        Response values (1D array).
    h
        Bandwidth.
    kernel
        Kernel name: "gauss", "epan", or "unif".

    Returns
    -------
    float
        LOOCV MSE score.
    """
    if kernel == "gauss":
        base = np.exp(-0.5 * ((x[:, None] - x[None, :]) / h) ** 2) / (h * _SQRT_2PI)
    elif kernel == "epan":
        u = (x[:, None] - x[None, :]) / h
        mask = np.abs(u) <= 1
        base = np.zeros_like(u, dtype=float)
        base[mask] = 0.75 * (1 - u[mask] ** 2) / h
    elif kernel == "unif":
        u = (x[:, None] - x[None, :]) / h
        mask = np.abs(u) <= 1
        base = np.zeros_like(u, dtype=float)
        base[mask] = 0.5 / h
    elif kernel == "biweight":
        u = (x[:, None] - x[None, :]) / h
        mask = np.abs(u) <= 1
        base = np.zeros_like(u, dtype=float)
        base[mask] = (15 / 16) * (1 - u[mask] ** 2) ** 2 / h
    elif kernel == "triweight":
        u = (x[:, None] - x[None, :]) / h
        mask = np.abs(u) <= 1
        base = np.zeros_like(u, dtype=float)
        base[mask] = (35 / 32) * (1 - u[mask] ** 2) ** 3 / h
    elif kernel == "cosine":
        u = (x[:, None] - x[None, :]) / h
        mask = np.abs(u) <= 1
        base = np.zeros_like(u, dtype=float)
        base[mask] = (np.pi / 4) * np.cos(np.pi * u[mask] / 2) / h
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    np.fill_diagonal(base, 0.0)
    num = base @ y
    den = base.sum(axis=1)
    den_safe = np.where(den == 0, np.finfo(float).eps, den)
    m = num / den_safe
    return float(np.mean((y - m) ** 2))


def _nw_weights_grad(
    u: NDArray[Any],
    h: float,
    kernel: str,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return weights w, w' for Nadaraya-Watson (no second derivative)."""
    if kernel == "gauss":
        base = np.exp(-0.5 * u * u) / (h * _SQRT_2PI)
        w1 = base * (u * u - 1) / h
        return base, w1
    elif kernel == "epan":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        uu = u * u
        w[mask] = 0.75 * (1 - uu[mask]) / h
        w1[mask] = 0.75 * (-1 + 3 * uu[mask]) / (h * h)
        return w, w1
    elif kernel == "unif":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w[mask] = 0.5 / h
        w1 = np.zeros_like(u, dtype=float)
        return w, w1
    elif kernel == "biweight":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        uu = u * u
        one_minus_uu = 1 - uu
        w[mask] = (15 / 16) * one_minus_uu[mask] ** 2 / h
        w1[mask] = (15 / 16) * one_minus_uu[mask] * (5 * uu[mask] - 1) / (h * h)
        return w, w1
    elif kernel == "triweight":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        uu = u * u
        one_minus_uu = 1 - uu
        w[mask] = (35 / 32) * one_minus_uu[mask] ** 3 / h
        w1[mask] = (35 / 32) * one_minus_uu[mask] ** 2 * (7 * uu[mask] - 1) / (h * h)
        return w, w1
    elif kernel == "cosine":
        mask = np.abs(u) <= 1
        w = np.zeros_like(u, dtype=float)
        w1 = np.zeros_like(u, dtype=float)
        uu = u * u
        cos_val = np.cos(np.pi * u / 2)
        sin_val = np.sin(np.pi * u / 2)
        pi = np.pi
        w[mask] = (pi / 4) * cos_val[mask] / h
        w1[mask] = (
            (pi / 4)
            * ((pi**2 * uu[mask] / 4 - 1) * cos_val[mask] - (pi * u[mask] / 2) * sin_val[mask])
            / (h * h)
        )
        return w, w1
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def loocv_mse_grad(
    x: NDArray[Any],
    y: NDArray[Any],
    h: float,
    kernel: str = "gauss",
) -> tuple[float, float]:
    """Compute LOOCV MSE and gradient (no Hessian) for NW bandwidth selection.

    This is more efficient than loocv_mse() when only loss and gradient are needed,
    as it skips the w'' computation required for the Hessian.

    Parameters
    ----------
    x
        Predictor values (1D array).
    y
        Response values (1D array).
    h
        Bandwidth.
    kernel
        Kernel name: "gauss", "epan", or "unif".

    Returns
    -------
    tuple[float, float]
        (loss, gradient) of the LOOCV MSE objective.
    """
    n = len(x)
    u = (x[:, None] - x[None, :]) / h
    w, w1 = _nw_weights_grad(u, h, kernel)
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(w1, 0.0)

    num = w @ y
    den = w.sum(axis=1)
    den_safe = np.where(den == 0, np.finfo(float).eps, den)
    m = num / den_safe

    num1 = w1 @ y
    den1 = w1.sum(axis=1)
    m1 = (num1 * den_safe - num * den1) / (den_safe**2)

    resid = y - m
    loss = float(np.mean(resid**2))
    grad = float((-2.0 / n) * np.sum(resid * m1))
    return loss, grad


def loocv_mse(
    x: NDArray[Any],
    y: NDArray[Any],
    h: float,
    kernel: str = "gauss",
) -> tuple[float, float, float]:
    """Compute LOOCV MSE, gradient, and Hessian for NW bandwidth selection.

    Parameters
    ----------
    x
        Predictor values (1D array).
    y
        Response values (1D array).
    h
        Bandwidth.
    kernel
        Kernel name: "gauss" or "epan".

    Returns
    -------
    tuple[float, float, float]
        (loss, gradient, hessian) of the LOOCV MSE objective.
    """
    n = len(x)
    u = (x[:, None] - x[None, :]) / h
    w, w1, w2 = _nw_weights(u, h, kernel)
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(w1, 0.0)
    np.fill_diagonal(w2, 0.0)

    num = w @ y
    den = w.sum(axis=1)
    den_safe = np.where(den == 0, np.finfo(float).eps, den)
    m = num / den_safe

    num1 = w1 @ y
    den1 = w1.sum(axis=1)
    m1 = (num1 * den_safe - num * den1) / (den_safe**2)

    num2 = w2 @ y
    den2 = w2.sum(axis=1)
    m2 = (num2 * den_safe - num * den2) / (den_safe**2) - 2 * m1 * den1 / den_safe

    resid = y - m
    loss = float(np.mean(resid**2))
    grad = float((-2.0 / n) * np.sum(resid * m1))
    hess = float((2.0 / n) * np.sum(m1 * m1 - resid * m2))
    return loss, grad, hess


def _loocv_numba_wrapper(
    x: NDArray[Any], y: NDArray[Any], h: float, kernel: str
) -> tuple[float, float, float]:
    """Wrap numba functions to match numpy API signature."""
    if kernel == "gauss":
        return loocv_numba_gauss(x, y, h)
    elif kernel == "epan":
        return loocv_numba_epan(x, y, h)
    elif kernel == "unif":
        return loocv_numba_unif(x, y, h)
    elif kernel == "biweight":
        return loocv_numba_biweight(x, y, h)
    elif kernel == "triweight":
        return loocv_numba_triweight(x, y, h)
    elif kernel == "cosine":
        return loocv_numba_cosine(x, y, h)
    raise ValueError(f"Numba not available for kernel {kernel!r}")


def _loocv_score_numba_wrapper(x: NDArray[Any], y: NDArray[Any], h: float, kernel: str) -> float:
    """Wrap numba score functions to match numpy API signature."""
    if kernel == "gauss":
        return loocv_score_numba_gauss(x, y, h)
    elif kernel == "epan":
        return loocv_score_numba_epan(x, y, h)
    elif kernel == "unif":
        return loocv_score_numba_unif(x, y, h)
    elif kernel == "biweight":
        return loocv_score_numba_biweight(x, y, h)
    elif kernel == "triweight":
        return loocv_score_numba_triweight(x, y, h)
    elif kernel == "cosine":
        return loocv_score_numba_cosine(x, y, h)
    raise ValueError(f"Numba not available for kernel {kernel!r}")


def nw_bandwidth(
    x: ArrayLike,
    y: ArrayLike,
    kernel: str = "gauss",
    h0: float | None = None,
    max_n: int | None = 5000,
    seed: int | None = None,
) -> float:
    """Select optimal Nadaraya-Watson bandwidth via Newton-Armijo on LOOCV-MSE.

    Uses analytic gradients and Hessians for fast convergence (6-12 evaluations
    vs 50-100 for grid search).

    Parameters
    ----------
    x
        Predictor values (1D array-like).
    y
        Response values (1D array-like).
    kernel
        Kernel function: "gauss" (Gaussian) or "epan" (Epanechnikov).
    h0
        Initial bandwidth guess. If None, uses Silverman's rule on x.
    max_n
        Maximum sample size for optimization. If len(x) > max_n, a random
        subsample is used. Set to None to disable subsampling.
    seed
        Random seed for reproducible subsampling.

    Returns
    -------
    float
        Optimal bandwidth that minimizes the LOOCV MSE criterion.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-2, 2, 200)
    >>> y = np.sin(x) + 0.1 * np.random.randn(len(x))
    >>> h = nw_bandwidth(x, y)
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if len(x_arr) != len(y_arr):
        raise ValueError(f"x and y must have same length, got {len(x_arr)} and {len(y_arr)}")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")

    rng = np.random.default_rng(seed)
    x_opt, y_opt = _subsample(x_arr, y_arr, max_n, rng)

    if h0 is None:
        h0 = _silverman_h(x_opt, kernel)

    return _newton_armijo(
        _loocv_numba_wrapper,
        x_opt,
        y_opt,
        h0,
        kernel,
        score_only=_loocv_score_numba_wrapper,
    )
