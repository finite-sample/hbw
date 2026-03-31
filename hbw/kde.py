"""KDE bandwidth selection via LSCV minimization."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._kernels import _KERNELS
from ._optim import _newton_armijo, _silverman_h, _subsample


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
