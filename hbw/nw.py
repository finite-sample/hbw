"""NW bandwidth selection via LOOCV-MSE minimization."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import _KERNELS, _SQRT_2PI
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
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


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
        raise ValueError(f"kernel must be 'gauss' or 'epan', got {kernel!r}")

    rng = np.random.default_rng(seed)
    x_opt, y_opt = _subsample(x_arr, y_arr, max_n, rng)

    if h0 is None:
        h0 = _silverman_h(x_opt)

    return _newton_armijo(loocv_mse, x_opt, y_opt, h0, kernel)
