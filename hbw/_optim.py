"""Newton-Armijo optimizer for bandwidth selection."""

import math
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


def _grad_converged(g: float, h: float, f: float, hess: float, tol: float) -> bool:
    """Test gradient convergence on a scale-free footing.

    An absolute test ``|g| < tol`` is not usable here because the criteria are
    only scale-*equivariant*, not scale-invariant: replacing ``x`` by ``c * x``
    sends the LSCV minimiser to ``c * h`` while its gradient shrinks like
    ``1 / c**2`` (LOOCV-MSE: like ``1 / c``). With a fixed absolute tolerance,
    data in large units looks converged at the starting guess and the optimiser
    returns Silverman's rule untouched. The dimensionless ratio ``|g| * h / |f|``
    is invariant under ``x -> c * x`` for both criteria, so test that instead.

    Args:
        g: Gradient of the criterion at ``h``.
        h: Current bandwidth.
        f: Value of the criterion at ``h``.
        hess: Hessian of the criterion at ``h``.
        tol: Relative tolerance.

    Returns:
        bool: True when the scaled gradient is below ``tol``.
    """
    scale = abs(f)
    if not np.isfinite(scale):
        return False
    if scale == 0.0:
        if g == 0.0:
            return True
        if hess > 0.0 and np.isfinite(hess):
            return abs(g / hess) < tol * h
        return False
    return abs(g) * h < tol * scale


def _bandwidth_floor(h_reference: float, ratio: float = 1e-6) -> float:
    """Return a positive floor that scales with the bandwidth problem."""
    if not np.isfinite(h_reference) or h_reference <= 0.0:
        raise ValueError(f"initial bandwidth must be finite and positive, got {h_reference!r}")
    return max(ratio * h_reference, np.nextafter(0.0, 1.0))


def _newton_step(g: float, hess: float, h: float) -> float:
    """Return a trial step: the Newton step, capped above and floored below.

    Args:
        g: Gradient at ``h``.
        hess: Hessian at ``h``.
        h: Current bandwidth.

    Returns:
        float: Trial step, zero only when the gradient vanishes.
    """
    if g == 0.0:
        return 0.0
    if hess > 0 and np.isfinite(hess):
        step = float(np.clip(-g / hess, -0.5 * h, 0.5 * h))
    else:
        step = math.copysign(0.1 * h, -g)
    if abs(step) < 1e-3 * h:
        step = math.copysign(1e-3 * h, step if step != 0.0 else -g)
    return step


def _line_search(
    score_at: Callable[[float], float],
    h: float,
    step: float,
    f: float,
    h_floor: float,
    n_back: int = 4,
    n_forward: int = 12,
) -> tuple[float, float] | None:
    """Armijo backtracking with forward-tracking, along a descent direction.

    Backtracking alone is not enough here. The compactly supported kernels are
    only C0, so every pair entering or leaving the kernel support puts a kink in
    the criterion. Local curvature is then dominated by those kinks and grossly
    overstates the curvature of the criterion's global shape, which makes the
    honest Newton step far too short: the iterate crawls and the iteration cap
    binds long before the minimum. Doubling while the score keeps falling
    escapes such a region. Near a genuine quadratic minimum the first doubling
    is rejected immediately, so Newton's fast local convergence is preserved.

    Known limitation: the accepted point is the last one that improved, which
    can lie past the nearest minimum -- on a multi-modal criterion that is
    enough to settle in a neighbouring basin. This is why 1-D cosine can land
    on a local minimum 0.03% worse in objective than a dense grid. Refining the
    bracket the overshoot leaves behind was tried and rejected: every variant
    needed an absolute floor or tolerance somewhere, which reintroduced the
    scale dependence this module exists to remove. If it is revisited, the
    invariant to hold is h(c*x) == c*h(x) to 1e-6, per test_newton.py.

    Args:
        score_at: Function returning the criterion at a bandwidth.
        h: Current bandwidth.
        step: Trial step (must point downhill).
        f: Criterion value at ``h``.
        h_floor: Smallest admissible bandwidth.
        n_back: Maximum number of step halvings.
        n_forward: Maximum number of step doublings.

    Returns:
        tuple[float, float] | None: The accepted ``(bandwidth, score)``, or None if no decrease was found.
    """
    h_new = max(h + step, h_floor)
    f_new = score_at(h_new)
    if not f_new < f:
        for _ in range(n_back):
            step *= 0.5
            h_new = max(h + step, h_floor)
            f_new = score_at(h_new)
            if f_new < f:
                break
        else:
            return None

    for _ in range(n_forward):
        step_try = 2.0 * step
        h_try = max(h + step_try, h_floor)
        if h_try == h_new:
            break
        f_try = score_at(h_try)
        if not f_try < f_new:
            break
        step, h_new, f_new = step_try, h_try, f_try

    return h_new, f_new


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

    Args:
        objective: Function returning (score, gradient, hessian).
        x: Data array.
        y: Optional response array (for NW regression).
        h0: Initial bandwidth.
        kernel: Kernel name.
        tol: Relative convergence tolerance on the scaled gradient ``|g| * h / |f|``.
        max_iter: Maximum Newton iterations.
        score_only: Optional score-only function for efficient backtracking. If None, uses objective(...)[0].

    Returns:
        float: Optimal bandwidth.
    """

    def _eval_score(h: float) -> float:
        if score_only is not None:
            return score_only(x, h, kernel) if y is None else score_only(x, y, h, kernel)
        return objective(x, h, kernel)[0] if y is None else objective(x, y, h, kernel)[0]

    def _eval_full(h: float) -> tuple[float, float, float]:
        return objective(x, h, kernel) if y is None else objective(x, y, h, kernel)

    def _run_from(h_start: float) -> tuple[float, float]:
        h = h_start
        h_floor = _bandwidth_floor(h_start)
        f, g, H = _eval_full(h)
        f_prev = float("inf")

        for _ in range(max_iter):
            if _grad_converged(g, h, f, H, tol):
                break
            if abs(f - f_prev) < 1e-8 * abs(f):
                break
            f_prev = f

            step = _newton_step(g, H, h)
            if step == 0.0:
                break

            accepted = _line_search(_eval_score, h, step, f, h_floor)
            if accepted is None:
                break
            h = accepted[0]
            f, g, H = _eval_full(h)

        return h, f

    h_best, f_best = _run_from(h0)

    if kernel == "unif":
        std = float(np.std(x, ddof=1))
        for mult in [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 3.0]:
            h_cand, f_cand = _run_from(mult * std)
            if f_cand < f_best:
                h_best, f_best = h_cand, f_cand

    return h_best
