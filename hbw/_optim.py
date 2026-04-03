"""Newton-Armijo optimizer for bandwidth selection."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _silverman_h(x: NDArray[Any], kernel: str = "gauss") -> float:
    """Silverman's rule of thumb for initial bandwidth."""
    n = len(x)
    std = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    h = 0.9 * scale * n ** (-0.2)
    if kernel == "unif":
        h *= 3.0
    return h


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


def _newton_armijo(
    objective: Callable[..., tuple[float, float, float]],
    x: NDArray[Any],
    y: NDArray[Any] | None,
    h0: float,
    kernel: str,
    tol: float = 1e-5,
    max_iter: int = 12,
    score_only: Callable[..., float] | None = None,
) -> float:
    """Run Newton-Armijo optimization for bandwidth selection.

    Parameters
    ----------
    objective
        Function returning (score, gradient, hessian).
    x
        Data array.
    y
        Optional response array (for NW regression).
    h0
        Initial bandwidth.
    kernel
        Kernel name.
    tol
        Convergence tolerance on gradient.
    max_iter
        Maximum Newton iterations.
    score_only
        Optional score-only function for efficient backtracking.
        If None, uses objective(...)[0].

    Returns
    -------
    float
        Optimal bandwidth.
    """

    def _eval_score(h: float) -> float:
        if score_only is not None:
            return score_only(x, h, kernel) if y is None else score_only(x, y, h, kernel)
        return objective(x, h, kernel)[0] if y is None else objective(x, y, h, kernel)[0]

    def _eval_full(h: float) -> tuple[float, float, float]:
        return objective(x, h, kernel) if y is None else objective(x, y, h, kernel)

    def _run_from(h_start: float) -> tuple[float, float]:
        h = h_start
        f, g, H = _eval_full(h)
        f_prev = float("inf")

        for _ in range(max_iter):
            if abs(g) < tol:
                break
            if abs(f - f_prev) < 1e-8 * abs(f):
                break
            f_prev = f

            if H > 0 and np.isfinite(H):
                step = -g / H
                step = np.clip(step, -0.5 * h, 0.5 * h)
            else:
                step = 0.1 * h * np.sign(-g) if g != 0 else 0.0
            if abs(step) / h < 1e-3:
                break

            h_new = max(h + step, 1e-6)
            f_new = _eval_score(h_new)

            if f_new < f:
                h = h_new
                f, g, H = _eval_full(h)
                continue

            for _ in range(4):
                step *= 0.5
                h_new = max(h + step, 1e-6)
                f_new = _eval_score(h_new)
                if f_new < f:
                    h = h_new
                    f, g, H = _eval_full(h)
                    break
            else:
                break

        return h, f

    h_best, f_best = _run_from(h0)

    if kernel == "unif":
        std = float(np.std(x, ddof=1))
        for mult in [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 3.0]:
            h_cand, f_cand = _run_from(mult * std)
            if f_cand < f_best:
                h_best, f_best = h_cand, f_cand

    return h_best
