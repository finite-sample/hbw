"""NW bandwidth selection via LOOCV-MSE minimization."""

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import _KERNELS, _SQRT_2PI
from ._numba_nw import (
    loocv_mv_numba_gauss,
    loocv_numba_biweight,
    loocv_numba_cosine,
    loocv_numba_epan,
    loocv_numba_gauss,
    loocv_numba_triweight,
    loocv_numba_unif,
    loocv_score_mv_numba_gauss,
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


def loocv_mse_mv(
    data: NDArray[Any],
    y: NDArray[Any],
    h: float,
    kernel: str = "gauss",
) -> tuple[float, float, float]:
    """Compute LOOCV MSE, gradient, and Hessian for multivariate NW bandwidth selection.

    Uses product kernel with isotropic bandwidth h across all dimensions.

    Parameters
    ----------
    data
        Predictor values, shape (n, d) where n is samples and d is dimension.
    y
        Response values (1D array of length n).
    h
        Bandwidth (scalar, applied to all dimensions).
    kernel
        Kernel name: "gauss".

    Returns
    -------
    tuple[float, float, float]
        (loss, gradient, hessian) of the LOOCV MSE objective.
    """
    K, Kp, Kpp, _, _, _ = _KERNELS[kernel]
    n, d = data.shape

    U = (data[:, None, :] - data[None, :, :]) / h

    K_vals = K(U)
    K_prod = np.prod(K_vals, axis=2) / h**d

    np.fill_diagonal(K_prod, 0.0)
    num = K_prod @ y
    den = K_prod.sum(axis=1)
    den_safe = np.where(den == 0, np.finfo(float).eps, den)
    m = num / den_safe

    Kp_vals = Kp(U)
    sum_ratio = np.zeros((n, n))
    for k in range(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(K_vals[:, :, k] != 0, Kp_vals[:, :, k] / K_vals[:, :, k], 0)
        sum_ratio += U[:, :, k] * ratio

    w1 = -(K_prod / h) * (d + sum_ratio)
    np.fill_diagonal(w1, 0.0)
    num1 = w1 @ y
    den1 = w1.sum(axis=1)
    m1 = (num1 * den_safe - num * den1) / (den_safe**2)

    Kpp_vals = Kpp(U)
    sum_d2 = np.zeros((n, n))
    for k in range(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(K_vals[:, :, k] != 0, Kp_vals[:, :, k] / K_vals[:, :, k], 0)
            r2 = np.where(K_vals[:, :, k] != 0, Kpp_vals[:, :, k] / K_vals[:, :, k], 0)
        sum_d2 += 2.0 * U[:, :, k] * r + U[:, :, k] ** 2 * (r2 - r * r)

    w2 = (K_prod / (h * h)) * ((d + 1) * d + 2.0 * (d + 1) * sum_ratio + sum_d2)
    np.fill_diagonal(w2, 0.0)
    num2 = w2 @ y
    den2 = w2.sum(axis=1)
    m2 = (num2 * den_safe - num * den2) / (den_safe**2) - 2 * m1 * den1 / den_safe

    resid = y - m
    loss = float(np.mean(resid**2))
    grad = float((-2.0 / n) * np.sum(resid * m1))
    hess = float((2.0 / n) * np.sum(m1 * m1 - resid * m2))
    return loss, grad, hess


def _scott_h_mv_nw(data: NDArray[Any]) -> float:
    """Scott's rule of thumb for multivariate NW initial bandwidth."""
    n, d = data.shape
    std_avg = float(np.mean(np.std(data, axis=0, ddof=1)))
    return std_avg * n ** (-1.0 / (d + 4))


def _newton_armijo_mv_nw(
    data: NDArray[Any],
    y: NDArray[Any],
    h0: float,
    kernel: str,
    tol: float = 1e-5,
    max_iter: int = 15,
) -> float:
    """Run Newton-Armijo optimization for multivariate NW bandwidth selection."""
    h = h0
    f_prev = float("inf")
    for _ in range(max_iter):
        f, g, hess = loocv_mse_mv(data, y, h, kernel)
        if abs(g) < tol:
            break
        if abs(f - f_prev) < 1e-8 * abs(f):
            break
        f_prev = f

        if hess > 0 and np.isfinite(hess):
            step = -g / hess
            step = np.clip(step, -0.5 * h, 0.5 * h)
        else:
            step = 0.1 * h * np.sign(-g) if g != 0 else 0.0
        if abs(step) / h < 1e-3:
            break

        h_new = max(h + step, 0.01 * h0)
        f_new = loocv_mse_mv(data, y, h_new, kernel)[0]

        if f_new < f:
            h = h_new
            continue

        for _ in range(4):
            step *= 0.5
            h_new = max(h + step, 0.01 * h0)
            f_new = loocv_mse_mv(data, y, h_new, kernel)[0]
            if f_new < f:
                h = h_new
                break
    return h


def _newton_armijo_mv_nw_numba(
    data: NDArray[Any],
    y: NDArray[Any],
    h0: float,
    tol: float = 1e-5,
    max_iter: int = 15,
) -> float:
    """Run Newton-Armijo optimization for multivariate NW bandwidth selection with Numba."""
    h = h0
    f_prev = float("inf")
    for _ in range(max_iter):
        f, g, hess = loocv_mv_numba_gauss(data, y, h)
        if abs(g) < tol:
            break
        if abs(f - f_prev) < 1e-8 * abs(f):
            break
        f_prev = f

        if hess > 0 and np.isfinite(hess):
            step = -g / hess
            step = np.clip(step, -0.5 * h, 0.5 * h)
        else:
            step = 0.1 * h * np.sign(-g) if g != 0 else 0.0
        if abs(step) / h < 1e-3:
            break

        h_new = max(h + step, 0.01 * h0)
        f_new = loocv_score_mv_numba_gauss(data, y, h_new)

        if f_new < f:
            h = h_new
            continue

        for _ in range(4):
            step *= 0.5
            h_new = max(h + step, 0.01 * h0)
            f_new = loocv_score_mv_numba_gauss(data, y, h_new)
            if f_new < f:
                h = h_new
                break
    return h


def nw_predict(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    h: float,
    kernel: str = "gauss",
) -> NDArray[Any]:
    """Nadaraya-Watson kernel regression predictions.

    Parameters
    ----------
    x_train
        Training predictor values (1D array-like).
    y_train
        Training response values (1D array-like).
    x_test
        Test predictor values where predictions are desired (1D array-like).
    h
        Bandwidth (obtained from nw_bandwidth).
    kernel
        Kernel function: "gauss", "epan", "unif", "biweight", "triweight", or "cosine".

    Returns
    -------
    NDArray
        Predicted values at x_test locations.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-2, 2, 200)
    >>> y = np.sin(x) + 0.1 * np.random.randn(len(x))
    >>> h = nw_bandwidth(x, y)
    >>> y_pred = nw_predict(x, y, x, h)  # in-sample predictions

    Notes
    -----
    This function returns point estimates only. No confidence intervals or
    standard errors are provided. Key assumptions: smooth regression function,
    IID observations, design density bounded away from zero in region of interest.

    For inference, consider bootstrap resampling or see statsmodels.nonparametric.
    """
    x_tr = np.asarray(x_train, dtype=float).ravel()
    y_tr = np.asarray(y_train, dtype=float).ravel()
    x_te = np.asarray(x_test, dtype=float).ravel()

    if len(x_tr) != len(y_tr):
        raise ValueError(f"x_train and y_train must have same length, got {len(x_tr)} and {len(y_tr)}")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")

    K, _, _, _, _, _ = _KERNELS[kernel]

    u = (x_te[:, None] - x_tr[None, :]) / h
    w = K(u)

    w_sum = w.sum(axis=1)
    w_sum_safe = np.where(w_sum == 0, np.finfo(float).eps, w_sum)
    y_pred = (w @ y_tr) / w_sum_safe

    zero_weight_mask = w_sum == 0
    if np.any(zero_weight_mask):
        y_pred[zero_weight_mask] = np.mean(y_tr)

    return y_pred


def nw_predict_mv(
    data_train: ArrayLike,
    y_train: ArrayLike,
    data_test: ArrayLike,
    h: float,
    kernel: str = "gauss",
) -> NDArray[Any]:
    """Multivariate Nadaraya-Watson kernel regression predictions using product kernel.

    Parameters
    ----------
    data_train
        Training predictor values, shape (n_train, d).
    y_train
        Training response values (1D array of length n_train).
    data_test
        Test predictor values where predictions are desired, shape (n_test, d).
    h
        Bandwidth (scalar, applied to all dimensions; obtained from nw_bandwidth_mv).
    kernel
        Kernel function: "gauss", "epan", "unif", "biweight", "triweight", or "cosine".

    Returns
    -------
    NDArray
        Predicted values at data_test locations.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(500, 2)
    >>> y = np.sin(data[:, 0]) + 0.5 * data[:, 1] + 0.3 * np.random.randn(500)
    >>> h = nw_bandwidth_mv(data, y)
    >>> y_pred = nw_predict_mv(data, y, data, h)  # in-sample predictions

    Notes
    -----
    This function returns point estimates only. No confidence intervals or
    standard errors are provided. Key assumptions: smooth regression function,
    IID observations, design density bounded away from zero in region of interest.
    Uses product kernel with isotropic bandwidth; data should be standardized
    for best results.

    For inference, consider bootstrap resampling or see statsmodels.nonparametric.
    """
    data_tr = np.asarray(data_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=float).ravel()
    data_te = np.asarray(data_test, dtype=float)

    if data_tr.ndim == 1:
        data_tr = data_tr.reshape(-1, 1)
    if data_te.ndim == 1:
        data_te = data_te.reshape(-1, 1)

    if data_tr.ndim != 2:
        raise ValueError(f"data_train must be 2D array, got shape {data_tr.shape}")
    if data_te.ndim != 2:
        raise ValueError(f"data_test must be 2D array, got shape {data_te.shape}")
    if len(data_tr) != len(y_tr):
        raise ValueError(f"data_train and y_train must have same length, got {len(data_tr)} and {len(y_tr)}")
    if data_tr.shape[1] != data_te.shape[1]:
        raise ValueError(f"data_train and data_test must have same number of dimensions, got {data_tr.shape[1]} and {data_te.shape[1]}")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")

    K, _, _, _, _, _ = _KERNELS[kernel]

    U = (data_te[:, None, :] - data_tr[None, :, :]) / h
    K_vals = K(U)
    w = np.prod(K_vals, axis=2)

    w_sum = w.sum(axis=1)
    w_sum_safe = np.where(w_sum == 0, np.finfo(float).eps, w_sum)
    y_pred = (w @ y_tr) / w_sum_safe

    zero_weight_mask = w_sum == 0
    if np.any(zero_weight_mask):
        y_pred[zero_weight_mask] = np.mean(y_tr)

    return y_pred


def nw_bandwidth_mv(
    data: ArrayLike,
    y: ArrayLike,
    kernel: str = "gauss",
    h0: float | None = None,
    max_n: int | None = 3000,
    seed: int | None = None,
    standardize: bool = True,
) -> float:
    """Select optimal multivariate NW bandwidth via Newton-Armijo on LOOCV-MSE.

    Uses product kernel with isotropic bandwidth across all dimensions.
    For best results, predictors should be standardized (each dimension scaled
    to similar variance).

    Parameters
    ----------
    data
        Predictor values, shape (n, d) where n is samples and d is dimension.
    y
        Response values (1D array of length n).
    kernel
        Kernel function: "gauss", "epan", or "unif".
    h0
        Initial bandwidth guess. If None, uses Scott's rule.
    max_n
        Maximum sample size for optimization. If n > max_n, a random
        subsample is used. Set to None to disable subsampling.
    seed
        Random seed for reproducible subsampling.
    standardize
        If True, standardize each predictor dimension to unit variance before
        bandwidth selection.

    Returns
    -------
    float
        Optimal bandwidth that minimizes the LOOCV MSE criterion.
        If standardize=True, this is the bandwidth for the standardized data.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(500, 2)
    >>> y = np.sin(data[:, 0]) + 0.5 * data[:, 1] + 0.3 * np.random.randn(500)
    >>> h = nw_bandwidth_mv(data, y)
    """
    data_arr = np.asarray(data, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(-1, 1)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {data_arr.shape}")
    if len(data_arr) != len(y_arr):
        raise ValueError(f"data and y must have same length, got {len(data_arr)} and {len(y_arr)}")

    n, d = data_arr.shape
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")
    if d > 4:
        warnings.warn(
            f"Dimension d={d} is high; NW regression becomes unreliable due to curse of dimensionality",
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    if max_n is not None and n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        data_opt = data_arr[idx]
        y_opt = y_arr[idx]
    else:
        data_opt = data_arr
        y_opt = y_arr

    if standardize:
        stds = np.std(data_opt, axis=0, ddof=1)
        stds = np.where(stds > 0, stds, 1.0)
        data_opt = data_opt / stds

    if h0 is None:
        h0 = _scott_h_mv_nw(data_opt)

    if kernel == "gauss":
        return _newton_armijo_mv_nw_numba(data_opt, y_opt, h0)
    return _newton_armijo_mv_nw(data_opt, y_opt, h0, kernel)
