"""One-dimensional box-kernel regression bandwidth selection."""

from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from ._optim import _UNIFORM_START_SCALES, _silverman_h


@numba.njit
def _prefix_sum(values: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Keep Neumaier compensation so subtracting two prefixes retains small terms."""
    high = np.zeros(len(values) + 1)
    low = np.zeros(len(values) + 1)
    for i, value in enumerate(values):
        total = high[i] + value
        if abs(high[i]) >= abs(value):
            error = (high[i] - total) + value
        else:
            error = (value - total) + high[i]
        high[i + 1] = total
        low[i + 1] = low[i] + error
    return high, low


@numba.njit
def _window_loss(
    x: NDArray[Any],
    y: NDArray[Any],
    center: float,
    high: NDArray[Any],
    low: NDArray[Any],
    h: float,
) -> float:
    """Evaluate the sum of leave-one-out squared errors on sorted predictors."""
    left = right = 0
    loss = 0.0
    for i in range(len(x)):
        while x[i] - x[left] > h:
            left += 1
        while right < len(x) and x[right] - x[i] <= h:
            right += 1
        count = right - left - 1
        if count:
            total = ((high[right] - high[left]) - y[i]) + (low[right] - low[left])
            residual = y[i] - total / count
        else:
            # Existing LOOCV convention predicts zero for an isolated point.
            residual = y[i] + center
        loss += residual * residual
    return loss


def _uniform_nw_bandwidth(x: NDArray[Any], y: NDArray[Any], h0: float | None) -> float:
    """Score each small-sample neighborhood or a bounded large-sample grid."""
    if len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("uniform regression needs at least two finite observations")
    if h0 is not None and (not np.isfinite(h0) or h0 <= 0):
        raise ValueError("initial bandwidth must be finite and positive")
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    if x[0] == x[-1]:
        return 1.0 if h0 is None else h0

    # Estimate scales in power-of-two units. Squaring the raw predictors for
    # their standard deviation can overflow or underflow even for a finite h.
    x_exponent = int(np.frexp(np.max(np.abs(x)))[1])
    scaled_x = np.ldexp(x, -x_exponent)
    scaled_x = scaled_x - scaled_x[len(x) // 2]
    default = _silverman_h(scaled_x, "unif")
    tiny = np.finfo(float).smallest_subnormal
    limit = np.finfo(float).max
    with np.errstate(over="ignore", under="ignore"):
        initial = float(np.clip(np.ldexp(default, x_exponent), tiny, limit))
        if h0 is not None:
            initial = h0
        if len(x) <= 64:
            # A finite radius cannot reach a pair whose distance overflows.
            i, j = np.triu_indices(len(x), k=1)
            distances = x[j] - x[i]
            breaks = np.unique(np.append(0.0, distances[np.isfinite(distances)]))
            mids = breaks[:-1] + np.diff(breaks) / 2
            # If there is no interior float, the lower endpoint attains the
            # neighborhood. Rounding up would skip it.
            mids = np.where(mids < breaks[1:], mids, breaks[:-1])
            candidates = np.append(mids, min(np.nextafter(breaks[-1], np.inf), limit))
        else:
            grid = default * np.geomspace(0.1, 3.0, 50)
            starts = np.std(scaled_x, ddof=1) * np.asarray(_UNIFORM_START_SCALES)
            candidates = np.ldexp(np.r_[grid, starts], x_exponent)
            gaps = np.diff(x)
            candidates = np.r_[candidates, gaps[gaps > 0].min() / 2, x[-1] - x[0]]
            candidates = np.clip(candidates, tiny, limit)

    # Scaling by a power of two preserves the ordering of losses and keeps
    # their squares finite. Center same-sign responses before accumulating
    # so a large common offset does not swamp differences within a window.
    exponent = np.frexp(np.max(np.abs(y)))[1]
    y = np.ldexp(y, -int(exponent))
    center = max(0.0, float(y.min())) + min(0.0, float(y.max()))
    y = y - center
    high, low = _prefix_sum(y)
    best = initial
    best_loss = _window_loss(x, y, center, high, low, best)
    for h in candidates:
        if h > 0 and np.isfinite(h):
            loss = _window_loss(x, y, center, high, low, h)
            if loss < best_loss:
                best, best_loss = float(h), loss
    return float(best)
