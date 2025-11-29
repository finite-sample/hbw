"""Optimisation utilities for bandwidth selection.

This module provides the Newton-Armijo optimizer used for bandwidth selection
in kernel density estimation and Nadaraya-Watson regression.
"""

from typing import Callable, Tuple

import numpy as np

# Algorithm constants with rationale:
# - TOL_DEFAULT: 1e-5 absolute gradient tolerance, empirically sufficient for
#   bandwidth selection where typical gradients are O(1) to O(0.01)
# - MAX_ITER_DEFAULT: 12 iterations typically sufficient for Newton convergence
#   (quadratic convergence means ~6 iterations for 12 digits of precision)
# - ARMIJO_MAX_BACKTRACKS: 10 halvings reduce step by factor of 1024, enough
#   to find descent direction in practice
# - REL_STEP_TOL: 1e-3 relative step size below which we consider converged
# - H_MIN: 1e-6 minimum bandwidth to avoid numerical underflow
# - FALLBACK_STEP_FACTOR: 0.25 conservative gradient descent when Hessian invalid

TOL_DEFAULT: float = 1e-5
MAX_ITER_DEFAULT: int = 12
ARMIJO_MAX_BACKTRACKS: int = 10
REL_STEP_TOL: float = 1e-3
H_MIN: float = 1e-6
FALLBACK_STEP_FACTOR: float = 0.25


def newton_armijo(
    objective: Callable[[np.ndarray, float, str], Tuple[float, float, float]],
    x: np.ndarray,
    h0: float,
    kernel: str = "gauss",
    tol: float = TOL_DEFAULT,
    max_iter: int = MAX_ITER_DEFAULT,
) -> Tuple[float, int]:
    """Run Newton–Armijo iterations for a generic objective.

    Uses Newton's method with Armijo backtracking line search to find the
    bandwidth that minimizes the given objective function. Falls back to
    gradient descent when the Hessian is non-positive.

    Parameters
    ----------
    objective : Callable
        Callable returning ``(score, grad, hess)`` for given ``(x, h, kernel)``.
    x : np.ndarray
        Sample locations passed to ``objective``.
    h0 : float
        Initial bandwidth guess (must be positive).
    kernel : str
        Kernel name forwarded to ``objective``.
    tol : float
        Tolerance for gradient magnitude to stop optimisation.
    max_iter : int
        Maximum number of Newton updates.

    Returns
    -------
    Tuple[float, int]
        Optimised bandwidth and number of objective evaluations.

    Raises
    ------
    ValueError
        If h0 is not positive.
    """
    if h0 <= 0:
        raise ValueError(f"Initial bandwidth h0 must be positive, got {h0}")

    h: float = float(h0)
    evals: int = 0

    for _ in range(max_iter):
        f, g, H = objective(x, h, kernel)
        evals += 1

        # Check gradient convergence
        if abs(g) < tol:
            break

        # Newton step with fallback to gradient descent if Hessian invalid
        if H > 0 and np.isfinite(H):
            step = -g / H
        else:
            step = -FALLBACK_STEP_FACTOR * g

        # Check relative step convergence
        if abs(step) / h < REL_STEP_TOL:
            break

        # Armijo backtracking line search
        for _ in range(ARMIJO_MAX_BACKTRACKS):
            h_new = max(h + step, H_MIN)
            if objective(x, h_new, kernel)[0] < f:
                h = h_new
                break
            step *= 0.5

    return h, evals
