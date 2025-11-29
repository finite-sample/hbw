"""Newton–Armijo bandwidth selection for Nadaraya–Watson regression.

This module implements Leave-One-Out Cross-Validation (LOOCV) MSE for
Nadaraya-Watson kernel regression with analytic gradient and Hessian computation.
"""

import argparse
from collections.abc import Callable

import numpy as np

from derivatives import NW_WEIGHTS
from optim import newton_armijo


def loocv_mse(
    x: np.ndarray, y: np.ndarray, h: float, kernel: str
) -> tuple[float, float, float]:
    """Return LOOCV MSE, gradient and Hessian for bandwidth ``h``.

    The LOOCV MSE estimates the prediction error of the Nadaraya-Watson
    estimator by leaving out each observation in turn.

    Parameters
    ----------
    x : np.ndarray
        Predictor values (1D array).
    y : np.ndarray
        Response values (1D array, same length as x).
    h : float
        Bandwidth parameter (must be positive).
    kernel : str
        Kernel name: "gauss" or "epan".

    Returns
    -------
    tuple[float, float, float]
        (loss, gradient, hessian) of the LOOCV MSE w.r.t. h.

    Raises
    ------
    ValueError
        If h is not positive or kernel name is not recognized.
    """
    if h <= 0:
        raise ValueError(f"Bandwidth h must be positive, got {h}")
    if kernel not in NW_WEIGHTS:
        raise ValueError(f"Unknown kernel '{kernel}'. Available: {list(NW_WEIGHTS)}")

    n = len(x)
    u = (x[:, None] - x[None, :]) / h
    weight_func = NW_WEIGHTS[kernel]
    w, w1, w2 = weight_func(u, h)
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(w1, 0.0)
    np.fill_diagonal(w2, 0.0)

    num = w @ y
    den = w.sum(axis=1)
    eps = np.finfo(float).eps
    den_safe = np.where(den == 0, eps, den)  # Guard against division by zero
    m = num / den_safe

    num1 = w1 @ y
    den1 = w1.sum(axis=1)
    m1 = (num1 * den_safe - num * den1) / (den_safe**2)

    num2 = w2 @ y
    den2 = w2.sum(axis=1)
    m2 = (num2 * den_safe - num * den2) / (den_safe**2) - 2 * m1 * den1 / den_safe

    resid = y - m
    loss = np.mean(resid**2)
    grad = (-2.0 / n) * np.sum(resid * m1)
    hess = (2.0 / n) * np.sum(m1 * m1 - resid * m2)
    return loss, grad, hess


def make_nw_objective(
    y: np.ndarray,
) -> Callable[[np.ndarray, float, str], tuple[float, float, float]]:
    """Create a NW LOOCV objective function with fixed response values.

    Parameters
    ----------
    y : np.ndarray
        Response values to use in the objective.

    Returns
    -------
    Callable
        Objective function with signature (x, h, kernel) -> (loss, grad, hess).
    """

    def objective(
        x: np.ndarray, h: float, kernel: str
    ) -> tuple[float, float, float]:
        return loocv_mse(x, y, h, kernel)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analytic-Hessian NW bandwidth selection"
    )
    parser.add_argument("data", nargs="?", help="Path to data with two columns x,y")
    parser.add_argument("--kernel", choices=["gauss", "epan"], default="gauss")
    parser.add_argument("--h0", type=float, default=1.0, help="Initial bandwidth guess")
    args = parser.parse_args()

    if args.data:
        arr = np.loadtxt(args.data)
        x, y = arr[:, 0], arr[:, 1]
    else:
        x = np.linspace(-2, 2, 200)
        y = np.sin(x) + 0.1 * np.random.randn(len(x))

    objective = make_nw_objective(y)
    h, evals = newton_armijo(objective, x, args.h0, kernel=args.kernel)
    print(f"Optimal h={h:.5f} after {evals} evaluations")


if __name__ == "__main__":
    main()
