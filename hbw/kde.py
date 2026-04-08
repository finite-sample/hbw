"""KDE bandwidth selection via LSCV minimization."""

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import _KERNELS
from ._numba_kde import (
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
from ._optim import _newton_armijo, _silverman_h, _subsample


def lscv_score(x: NDArray[Any], h: float, kernel: str = "gauss") -> float:
    """Compute only the LSCV score (no gradient/Hessian).

    This is more efficient for grid search where only the score is needed.

    Parameters
    ----------
    x
        Sample data (1D array).
    h
        Bandwidth.
    kernel
        Kernel name: "gauss", "epan", or "unif".

    Returns
    -------
    float
        LSCV score.
    """
    K, _, _, K2, _, _ = _KERNELS[kernel]
    n = len(x)
    u = (x[:, None] - x[None, :]) / h

    term1 = K2(u).sum() / (n**2 * h)
    K_u = K(u)
    term2 = (K_u.sum() - np.trace(K_u)) / (n * (n - 1) * h)
    return float(term1 - 2 * term2)


def lscv_grad(x: NDArray[Any], h: float, kernel: str = "gauss") -> tuple[float, float]:
    """Compute LSCV score and gradient (no Hessian) for KDE bandwidth selection.

    This is more efficient than lscv() when only score and gradient are needed,
    as it skips the K'' and (K*K)'' computations required for the Hessian.

    Parameters
    ----------
    x
        Sample data (1D array).
    h
        Bandwidth.
    kernel
        Kernel name: "gauss", "epan", or "unif".

    Returns
    -------
    tuple[float, float]
        (score, gradient) of the LSCV objective.
    """
    K, Kp, _, K2, K2p, _ = _KERNELS[kernel]
    n = len(x)
    u = (x[:, None] - x[None, :]) / h

    K2_u = K2(u)
    K2p_u = K2p(u)
    K_u = K(u)
    Kp_u = Kp(u)

    term1 = K2_u.sum() / (n**2 * h)
    term2 = (K_u.sum() - np.trace(K_u)) / (n * (n - 1) * h)
    score = float(term1 - 2 * term2)

    S_F = (K2_u + u * K2p_u).sum()
    S_K_matrix = K_u + u * Kp_u
    S_K = S_K_matrix.sum() - np.trace(S_K_matrix)
    grad = float(-S_F / (n**2 * h**2) + 2 * S_K / (n * (n - 1) * h**2))

    return score, grad


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

    K2_u = K2(u)
    K2p_u = K2p(u)
    K2pp_u = K2pp(u)
    K_u = K(u)
    Kp_u = Kp(u)
    Kpp_u = Kpp(u)

    term1 = K2_u.sum() / (n**2 * h)
    term2 = (K_u.sum() - np.trace(K_u)) / (n * (n - 1) * h)
    score = float(term1 - 2 * term2)

    S_F = (K2_u + u * K2p_u).sum()
    S_K_matrix = K_u + u * Kp_u
    S_K = S_K_matrix.sum() - np.trace(S_K_matrix)
    grad = float(-S_F / (n**2 * h**2) + 2 * S_K / (n * (n - 1) * h**2))

    S_F2 = (2 * K2p_u + u * K2pp_u).sum()
    S_K2_matrix = 2 * Kp_u + u * Kpp_u
    S_K2 = S_K2_matrix.sum() - np.trace(S_K2_matrix)
    hess = 2 * S_F / (n**2 * h**3) - S_F2 / (n**2 * h**2)
    hess += -4 * S_K / (n * (n - 1) * h**3) + 2 * S_K2 / (n * (n - 1) * h**2)
    return score, grad, float(hess)


def _lscv_numba_wrapper(x: NDArray[Any], h: float, kernel: str) -> tuple[float, float, float]:
    """Wrap numba functions to match numpy API signature."""
    if kernel == "gauss":
        return lscv_numba_gauss(x, h)
    elif kernel == "epan":
        return lscv_numba_epan(x, h)
    elif kernel == "unif":
        return lscv_numba_unif(x, h)
    elif kernel == "biweight":
        return lscv_numba_biweight(x, h)
    elif kernel == "triweight":
        return lscv_numba_triweight(x, h)
    elif kernel == "cosine":
        return lscv_numba_cosine(x, h)
    raise ValueError(f"Numba not available for kernel {kernel!r}")


def _lscv_score_numba_wrapper(x: NDArray[Any], h: float, kernel: str) -> float:
    """Wrap numba score functions to match numpy API signature."""
    if kernel == "gauss":
        return lscv_score_numba_gauss(x, h)
    elif kernel == "epan":
        return lscv_score_numba_epan(x, h)
    elif kernel == "unif":
        return lscv_score_numba_unif(x, h)
    elif kernel == "biweight":
        return lscv_score_numba_biweight(x, h)
    elif kernel == "triweight":
        return lscv_score_numba_triweight(x, h)
    elif kernel == "cosine":
        return lscv_score_numba_cosine(x, h)
    raise ValueError(f"Numba not available for kernel {kernel!r}")


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
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")

    rng = np.random.default_rng(seed)
    x_opt, _ = _subsample(x_arr, None, max_n, rng)

    if h0 is None:
        h0 = _silverman_h(x_opt, kernel)

    return _newton_armijo(
        _lscv_numba_wrapper,
        x_opt,
        None,
        h0,
        kernel,
        score_only=_lscv_score_numba_wrapper,
    )


def lscv_mv(data: NDArray[Any], h: float, kernel: str = "gauss") -> tuple[float, float, float]:
    """Compute LSCV score, gradient, and Hessian for multivariate KDE.

    Uses product kernel with isotropic bandwidth h across all dimensions.

    Parameters
    ----------
    data
        Sample data, shape (n, d) where n is number of samples and d is dimension.
    h
        Bandwidth (scalar, applied to all dimensions).
    kernel
        Kernel name: "gauss", "epan", or "unif".

    Returns
    -------
    tuple[float, float, float]
        (score, gradient, hessian) of the LSCV objective.
    """
    K, Kp, Kpp, K2, K2p, K2pp = _KERNELS[kernel]
    n, d = data.shape

    U = (data[:, None, :] - data[None, :, :]) / h

    K2_prod = np.prod(K2(U), axis=2)
    term1 = K2_prod.sum() / (n**2 * h**d)

    K_prod = np.prod(K(U), axis=2)
    term2 = (K_prod.sum() - np.trace(K_prod)) / (n * (n - 1) * h**d)
    score = float(term1 - 2 * term2)

    K2_vals = K2(U)
    K2p_vals = K2p(U)
    sum_K2_ratio = np.zeros((n, n))
    for k in range(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(K2_vals[:, :, k] != 0, K2p_vals[:, :, k] / K2_vals[:, :, k], 0)
        sum_K2_ratio += U[:, :, k] * ratio
    S_F = (K2_prod * (d + sum_K2_ratio)).sum()

    K_vals = K(U)
    Kp_vals = Kp(U)
    sum_K_ratio = np.zeros((n, n))
    for k in range(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(K_vals[:, :, k] != 0, Kp_vals[:, :, k] / K_vals[:, :, k], 0)
        sum_K_ratio += U[:, :, k] * ratio
    S_K_matrix = K_prod * (d + sum_K_ratio)
    S_K = S_K_matrix.sum() - np.trace(S_K_matrix)

    grad = float(-S_F / (n**2 * h ** (d + 1)) + 2 * S_K / (n * (n - 1) * h ** (d + 1)))

    Kpp_vals = Kpp(U)
    K2pp_vals = K2pp(U)

    d2_K2 = np.zeros((n, n))
    for k in range(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = np.where(K2_vals[:, :, k] != 0, K2p_vals[:, :, k] / K2_vals[:, :, k], 0)
            r2 = np.where(K2_vals[:, :, k] != 0, K2pp_vals[:, :, k] / K2_vals[:, :, k], 0)
        d2_K2 += 2 * U[:, :, k] * r1 + U[:, :, k] ** 2 * r2
    S_F2 = (K2_prod * ((d + 1) * d + 2 * (d + 1) * sum_K2_ratio + d2_K2)).sum()

    d2_K = np.zeros((n, n))
    for k in range(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = np.where(K_vals[:, :, k] != 0, Kp_vals[:, :, k] / K_vals[:, :, k], 0)
            r2 = np.where(K_vals[:, :, k] != 0, Kpp_vals[:, :, k] / K_vals[:, :, k], 0)
        d2_K += 2 * U[:, :, k] * r1 + U[:, :, k] ** 2 * r2
    S_K2_matrix = K_prod * ((d + 1) * d + 2 * (d + 1) * sum_K_ratio + d2_K)
    S_K2 = S_K2_matrix.sum() - np.trace(S_K2_matrix)

    hess = (d + 1) * S_F / (n**2 * h ** (d + 2)) - S_F2 / (n**2 * h ** (d + 1))
    hess += -2 * (d + 1) * S_K / (n * (n - 1) * h ** (d + 2)) + 2 * S_K2 / (
        n * (n - 1) * h ** (d + 1)
    )

    return score, grad, float(hess)


def _silverman_h_mv(data: NDArray[Any]) -> float:
    """Scott's rule of thumb for multivariate initial bandwidth."""
    n, d = data.shape
    std_avg = float(np.mean(np.std(data, axis=0, ddof=1)))
    return std_avg * n ** (-1.0 / (d + 4))


def _newton_armijo_mv(
    data: NDArray[Any],
    h0: float,
    kernel: str,
    tol: float = 1e-5,
    max_iter: int = 15,
) -> float:
    """Run Newton-Armijo optimization for multivariate bandwidth selection."""
    h = h0
    for _ in range(max_iter):
        f, g, hess = lscv_mv(data, h, kernel)
        if abs(g) < tol:
            break
        step = (
            -g / hess
            if hess > 0 and np.isfinite(hess)
            else (0.25 * h * np.sign(-g) if g != 0 else 0.0)
        )
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            f_new = lscv_mv(data, h_new, kernel)[0]
            if f_new < f:
                h = h_new
                break
            step *= 0.5
    return h


def _newton_armijo_mv_numba(
    data: NDArray[Any],
    h0: float,
    tol: float = 1e-5,
    max_iter: int = 15,
) -> float:
    """Run Newton-Armijo optimization for multivariate bandwidth selection with Numba."""
    h = h0
    for _ in range(max_iter):
        f, g, hess = lscv_mv_numba_gauss(data, h)
        if abs(g) < tol:
            break
        step = (
            -g / hess
            if hess > 0 and np.isfinite(hess)
            else (0.25 * h * np.sign(-g) if g != 0 else 0.0)
        )
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            f_new = lscv_mv_numba_gauss(data, h_new)[0]
            if f_new < f:
                h = h_new
                break
            step *= 0.5
    return h


def kde_evaluate(
    x_train: ArrayLike,
    x_eval: ArrayLike,
    h: float,
    kernel: str = "gauss",
) -> NDArray[Any]:
    """Evaluate kernel density estimate at given points.

    Parameters
    ----------
    x_train
        Training sample data (1D array-like).
    x_eval
        Points at which to evaluate the density (1D array-like).
    h
        Bandwidth (obtained from kde_bandwidth).
    kernel
        Kernel function: "gauss", "epan", "unif", "biweight", "triweight", or "cosine".

    Returns
    -------
    Estimated density values at x_eval locations.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> h = kde_bandwidth(x)
    >>> x_grid = np.linspace(-3, 3, 100)
    >>> density = kde_evaluate(x, x_grid, h)

    Notes
    -----
    This function returns point estimates only. No confidence intervals or
    standard errors are provided. Key assumptions: smooth underlying density,
    IID observations, continuous density (no point masses).

    For inference, consider bootstrap resampling or see statsmodels.nonparametric.
    """
    x_tr = np.asarray(x_train, dtype=float).ravel()
    x_ev = np.asarray(x_eval, dtype=float).ravel()

    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")

    K, _, _, _, _, _ = _KERNELS[kernel]
    n = len(x_tr)

    u = (x_ev[:, None] - x_tr[None, :]) / h
    density = K(u).sum(axis=1) / (n * h)

    return density


def kde_evaluate_mv(
    data_train: ArrayLike,
    data_eval: ArrayLike,
    h: float,
    kernel: str = "gauss",
) -> NDArray[Any]:
    """Evaluate multivariate kernel density estimate at given points using product kernel.

    Parameters
    ----------
    data_train
        Training sample data, shape (n_train, d).
    data_eval
        Points at which to evaluate the density, shape (n_eval, d).
    h
        Bandwidth (scalar, applied to all dimensions; obtained from kde_bandwidth_mv).
    kernel
        Kernel function: "gauss", "epan", "unif", "biweight", "triweight", or "cosine".

    Returns
    -------
    Estimated density values at data_eval locations.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(500, 2)
    >>> h = kde_bandwidth_mv(data)
    >>> data_grid = np.column_stack([np.linspace(-3, 3, 50), np.linspace(-3, 3, 50)])
    >>> density = kde_evaluate_mv(data, data_grid, h)

    Notes
    -----
    This function returns point estimates only. No confidence intervals or
    standard errors are provided. Key assumptions: smooth underlying density,
    IID observations, continuous density (no point masses). Uses product kernel
    with isotropic bandwidth; data should be standardized for best results.

    For inference, consider bootstrap resampling or see statsmodels.nonparametric.
    """
    data_tr = np.asarray(data_train, dtype=float)
    data_ev = np.asarray(data_eval, dtype=float)

    if data_tr.ndim == 1:
        data_tr = data_tr.reshape(-1, 1)
    if data_ev.ndim == 1:
        data_ev = data_ev.reshape(-1, 1)

    if data_tr.ndim != 2:
        raise ValueError(f"data_train must be 2D array, got shape {data_tr.shape}")
    if data_ev.ndim != 2:
        raise ValueError(f"data_eval must be 2D array, got shape {data_ev.shape}")
    if data_tr.shape[1] != data_ev.shape[1]:
        raise ValueError(f"data_train and data_eval must have same number of dimensions, got {data_tr.shape[1]} and {data_ev.shape[1]}")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")

    K, _, _, _, _, _ = _KERNELS[kernel]
    n = len(data_tr)
    d = data_tr.shape[1]

    U = (data_ev[:, None, :] - data_tr[None, :, :]) / h
    K_vals = K(U)
    K_prod = np.prod(K_vals, axis=2)

    density = K_prod.sum(axis=1) / (n * h**d)

    return density


def kde_bandwidth_mv(
    data: ArrayLike,
    kernel: str = "gauss",
    h0: float | None = None,
    max_n: int | None = 3000,
    seed: int | None = None,
    standardize: bool = True,
) -> float:
    """Select optimal multivariate KDE bandwidth via Newton-Armijo on LSCV.

    Uses product kernel with isotropic bandwidth across all dimensions.
    For best results, data should be standardized (each dimension scaled to
    similar variance).

    Parameters
    ----------
    data
        Sample data, shape (n, d) where n is samples and d is dimension.
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
        If True, standardize each dimension to unit variance before
        bandwidth selection.

    Returns
    -------
    float
        Optimal bandwidth that minimizes the LSCV criterion.
        If standardize=True, this is the bandwidth for the standardized data.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(500, 2)
    >>> h = kde_bandwidth_mv(data)
    """
    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(-1, 1)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {data_arr.shape}")

    n, d = data_arr.shape
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got {kernel!r}")
    if d > 4:
        warnings.warn(
            f"Dimension d={d} is high; KDE becomes unreliable due to curse of dimensionality",
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    if max_n is not None and n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        data_opt = data_arr[idx]
    else:
        data_opt = data_arr

    if standardize:
        stds = np.std(data_opt, axis=0, ddof=1)
        stds = np.where(stds > 0, stds, 1.0)
        data_opt = data_opt / stds

    if h0 is None:
        h0 = _silverman_h_mv(data_opt)

    if kernel == "gauss":
        return _newton_armijo_mv_numba(data_opt, h0)
    return _newton_armijo_mv(data_opt, h0, kernel)
