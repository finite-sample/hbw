"""Fast kernel bandwidth selection via analytic Hessian Newton optimization.

This module provides optimal bandwidth selection for:
- Univariate kernel density estimation (KDE) via LSCV minimization
- Nadaraya-Watson regression via LOOCV-MSE minimization

The key innovation is using closed-form analytic gradients and Hessians
of the cross-validation objectives, enabling Newton optimization that
converges in 6-12 evaluations vs 50-100 for grid search.
"""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["kde_bandwidth", "nw_bandwidth", "lscv", "loocv_mse"]
__version__ = "0.1.0"

_SQRT_2PI = sqrt(2 * np.pi)
_SQRT_4PI = sqrt(4 * np.pi)


def _silverman_h(x: NDArray[Any]) -> float:
    """Silverman's rule of thumb for initial bandwidth."""
    n = len(x)
    std = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * scale * n ** (-0.2)


def _subsample(
    x: NDArray[Any],
    y: NDArray[Any] | None,
    max_n: int | None,
    rng: np.random.Generator,
) -> tuple[NDArray[Any], NDArray[Any] | None]:
    """Subsample data if n > max_n."""
    if max_n is None or len(x) <= max_n:
        return x, y
    idx = rng.choice(len(x), size=max_n, replace=False)
    x_sub = x[idx]
    y_sub = y[idx] if y is not None else None
    return x_sub, y_sub


def _gauss(u: NDArray[Any]) -> NDArray[Any]:
    """Gaussian kernel K(u)."""
    return np.exp(-0.5 * u * u) / _SQRT_2PI


def _gauss_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Gaussian kernel."""
    return -u * _gauss(u)


def _gauss_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Gaussian kernel."""
    return (u * u - 1.0) * _gauss(u)


def _gauss_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of Gaussian kernel."""
    return np.exp(-0.25 * u * u) / _SQRT_4PI


def _gauss_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Gaussian kernel convolution."""
    return -0.5 * u * _gauss_conv(u)


def _gauss_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Gaussian kernel convolution."""
    return (0.25 * u * u - 0.5) * _gauss_conv(u)


def _poly_mask(u: NDArray[Any], mask: NDArray[Any], expr: NDArray[Any]) -> NDArray[Any]:
    """Return piecewise polynomial values for Epanechnikov kernels."""
    out = np.zeros_like(u, dtype=float)
    out[mask] = expr[mask]
    return out


def _epan(u: NDArray[Any]) -> NDArray[Any]:
    """Epanechnikov kernel K(u)."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, 0.75 * (1 - u * u))


def _epan_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Epanechnikov kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, -1.5 * u)


def _epan_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Epanechnikov kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, np.full_like(u, -1.5))


def _epan_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of Epanechnikov kernel (valid for |u|<=2)."""
    absu = np.abs(u)
    mask = absu <= 2
    poly = 0.6 - 0.75 * absu**2 + 0.375 * absu**3 - 0.01875 * absu**5
    return _poly_mask(u, mask, poly)


def _epan_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Epanechnikov kernel convolution."""
    absu = np.abs(u)
    mask = absu <= 2
    poly = np.sign(u) * (-0.09375 * absu**4 + 1.125 * absu**2 - 1.5 * absu)
    return _poly_mask(u, mask, poly)


def _epan_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Epanechnikov kernel convolution."""
    absu = np.abs(u)
    mask = absu <= 2
    poly = -0.375 * absu**3 + 2.25 * absu - 1.5
    return _poly_mask(u, mask, poly)


_KERNELS = {
    "gauss": (_gauss, _gauss_p, _gauss_pp, _gauss_conv, _gauss_conv_p, _gauss_conv_pp),
    "epan": (_epan, _epan_p, _epan_pp, _epan_conv, _epan_conv_p, _epan_conv_pp),
}


def lscv(x: NDArray[Any], h: float, kernel: str = "gauss") -> tuple[float, float, float]:
    """Compute LSCV score, gradient, and Hessian for KDE bandwidth selection.

    Parameters
    ----------
    x
        Sample data (1D array).
    h
        Bandwidth.
    kernel
        Kernel name: "gauss" or "epan".

    Returns
    -------
    tuple[float, float, float]
        (score, gradient, hessian) of the LSCV objective.
    """
    K, Kp, Kpp, K2, K2p, K2pp = _KERNELS[kernel]
    n = len(x)
    u = (x[:, None] - x[None, :]) / h

    term1 = K2(u).sum() / (n**2 * h)
    Ku = K(u)
    term2 = (Ku.sum() - np.trace(Ku)) / (n * (n - 1) * h)
    score = float(term1 - 2 * term2)

    S_F = (K2(u) + u * K2p(u)).sum()
    S_K_matrix = Ku + u * Kp(u)
    S_K = S_K_matrix.sum() - np.trace(S_K_matrix)
    grad = float(-S_F / (n**2 * h**2) + 2 * S_K / (n * (n - 1) * h**2))

    S_F2 = (2 * K2p(u) + u * K2pp(u)).sum()
    Kp_u = Kp(u)
    Kpp_u = Kpp(u)
    S_K2_matrix = 2 * Kp_u + u * Kpp_u
    S_K2 = S_K2_matrix.sum() - np.trace(S_K2_matrix)
    hess = 2 * S_F / (n**2 * h**3) - S_F2 / (n**2 * h**2)
    hess += -4 * S_K / (n * (n - 1) * h**3) + 2 * S_K2 / (n * (n - 1) * h**2)
    return score, grad, float(hess)


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


def _newton_armijo(
    objective: Callable[..., tuple[float, float, float]],
    x: NDArray[Any],
    y: NDArray[Any] | None,
    h0: float,
    kernel: str,
    tol: float = 1e-5,
    max_iter: int = 12,
) -> float:
    """Run Newton-Armijo optimization for bandwidth selection."""
    h = h0
    for _ in range(max_iter):
        if y is None:
            f, g, H = objective(x, h, kernel)
        else:
            f, g, H = objective(x, y, h, kernel)
        if abs(g) < tol:
            break
        step = -g / H if (H > 0 and np.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            if y is None:
                f_new = objective(x, h_new, kernel)[0]
            else:
                f_new = objective(x, y, h_new, kernel)[0]
            if f_new < f:
                h = h_new
                break
            step *= 0.5
    return h


def kde_bandwidth(
    x: ArrayLike,
    kernel: str = "gauss",
    h0: float | None = None,
    max_n: int | None = 5000,
    seed: int | None = None,
) -> float:
    """Select optimal KDE bandwidth via Newton-Armijo on LSCV.

    Uses analytic gradients and Hessians for fast convergence (6-12 evaluations
    vs 50-100 for grid search).

    Parameters
    ----------
    x
        Sample data (1D array-like).
    kernel
        Kernel function: "gauss" (Gaussian) or "epan" (Epanechnikov).
    h0
        Initial bandwidth guess. If None, uses Silverman's rule.
    max_n
        Maximum sample size for optimization. If len(x) > max_n, a random
        subsample is used. Set to None to disable subsampling.
    seed
        Random seed for reproducible subsampling.

    Returns
    -------
    float
        Optimal bandwidth that minimizes the LSCV criterion.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> h = kde_bandwidth(x)
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be 'gauss' or 'epan', got {kernel!r}")

    rng = np.random.default_rng(seed)
    x_opt, _ = _subsample(x_arr, None, max_n, rng)

    if h0 is None:
        h0 = _silverman_h(x_opt)

    return _newton_armijo(lscv, x_opt, None, h0, kernel)


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
