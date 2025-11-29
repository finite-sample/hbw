"""Test utilities providing scalar reference implementations.

These scalar (non-vectorized) implementations serve as independent references
for verifying the vectorized numpy implementations in src/. They use the same
mathematical formulas but with explicit Python loops instead of matrix operations.
"""

import math
import sys
import os

# Add src/ to path so we can import the production kernel functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from derivatives import KERNELS


def _eval_scalar(func, u: float) -> float:
    """Evaluate a numpy kernel function at a scalar point."""
    import numpy as np
    return float(func(np.array([u]))[0])


def lscv_generic(x, h, kernel):
    """Scalar reference implementation of LSCV score, gradient, and Hessian.

    This implementation uses explicit loops rather than vectorized operations,
    serving as an independent verification of the numpy implementation.

    Parameters
    ----------
    x : list or array-like
        Sample data points.
    h : float
        Bandwidth parameter.
    kernel : str
        Kernel name: "gauss" or "epan".

    Returns
    -------
    tuple
        (score, gradient, hessian) at the given bandwidth.
    """
    K, Kp, Kpp, K2, K2p, K2pp = KERNELS[kernel]
    n = len(x)
    score_term1 = 0.0
    score_term2 = 0.0
    S_F = 0.0
    S_K = 0.0
    S_F2 = 0.0
    S_K2 = 0.0

    for i in range(n):
        for j in range(n):
            u = (x[i] - x[j]) / h
            k2 = _eval_scalar(K2, u)
            k2p = _eval_scalar(K2p, u)
            k2pp = _eval_scalar(K2pp, u)
            score_term1 += k2
            S_F += k2 + u * k2p
            S_F2 += 2 * k2p + u * k2pp
            if i != j:
                k = _eval_scalar(K, u)
                kp = _eval_scalar(Kp, u)
                kpp = _eval_scalar(Kpp, u)
                score_term2 += k
                S_K += k + u * kp
                S_K2 += 2 * kp + u * kpp

    score = score_term1 / (n ** 2 * h) - 2 * (score_term2 / (n * (n - 1) * h))
    grad = -S_F / (n ** 2 * h ** 2) + 2 * S_K / (n * (n - 1) * h ** 2)
    hess = 2 * S_F / (n ** 2 * h ** 3) - S_F2 / (n ** 2 * h ** 2)
    hess += -4 * S_K / (n * (n - 1) * h ** 3) + 2 * S_K2 / (n * (n - 1) * h ** 2)
    return score, grad, hess


def newton_opt(x, h0, score_grad_hess, tol=1e-5, max_iter=12):
    """Newton-Armijo optimisation using a numeric Hessian estimate.

    This reference implementation computes the Hessian via finite differences,
    useful for comparing against the analytic Hessian implementation.

    Parameters
    ----------
    x : list or array-like
        Sample data.
    h0 : float
        Initial bandwidth guess.
    score_grad_hess : callable
        Function returning (score, gradient, hessian) for given (x, h).
    tol : float
        Convergence tolerance for gradient magnitude.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    tuple
        (optimal_h, num_evaluations)
    """
    h, evals = h0, 0
    for _ in range(max_iter):
        f, g, _ = score_grad_hess(x, h)
        evals += 1
        if abs(g) < tol:
            break
        # Numeric Hessian via finite difference
        eps = max(1e-4 * h, 1e-6)
        _, g_plus, _ = score_grad_hess(x, h + eps)
        H = (g_plus - g) / eps
        step = -g / H if (H > 0 and math.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        # Armijo backtracking
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            if score_grad_hess(x, h_new)[0] < f:
                h = h_new
                break
            step *= 0.5
    return h, evals
