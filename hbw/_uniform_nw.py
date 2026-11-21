"""One-dimensional box-kernel regression bandwidth selection."""

from fractions import Fraction
from itertools import accumulate
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
def _bounds(x: NDArray[Any], h: float) -> tuple[NDArray[Any], NDArray[Any]]:
    """Locate inclusive neighborhoods by their half-open array slices."""
    lower = np.empty(len(x), dtype=np.int64)
    upper = np.empty(len(x), dtype=np.int64)
    left = right = 0
    for i in range(len(x)):
        while x[i] - x[left] > h:
            left += 1
        while right < len(x) and x[right] - x[i] <= h:
            right += 1
        lower[i], upper[i] = left, right
    return lower, upper


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
    left, right = _bounds(x, h)
    loss = 0.0
    for i in range(len(x)):
        a, b = left[i], right[i]
        count = b - a - 1
        if count:
            total = ((high[b] - high[a]) - y[i]) + (low[b] - low[a])
            residual = y[i] - total / count
        else:
            # Existing LOOCV convention predicts zero for an isolated point.
            residual = y[i] + center
        loss += residual * residual
    return loss


def _exact_minimum(x: NDArray[Any], y: NDArray[Any], candidates: NDArray[Any]) -> float:
    """Compare rational losses without losing tiny differences alongside large terms."""
    # Every binary float has a power-of-two denominator. Converting to common
    # units once makes window sums exact; the common squared unit cancels.
    ratios = [float(value).as_integer_ratio() for value in y]
    unit = max(denominator for _, denominator in ratios)
    values = [num * (unit // den) for num, den in ratios]
    prefix = list(accumulate(values, initial=0))

    def loss(h: float) -> Fraction:
        left, right = _bounds(x, h)
        by_count: dict[int, int] = {}
        for i, (a, b) in enumerate(zip(left.tolist(), right.tolist(), strict=True)):
            count = b - a - 1
            if count:
                numerator = (count + 1) * values[i] - (prefix[b] - prefix[a])
            else:
                count, numerator = 1, values[i]
            by_count[count] = by_count.get(count, 0) + numerator * numerator
        return sum((Fraction(ss, k * k) for k, ss in by_count.items()), Fraction())

    return float(min(candidates, key=loss))


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

    candidates = np.r_[initial, candidates[(candidates > 0) & np.isfinite(candidates)]]
    if len(x) <= 64:
        return _exact_minimum(x, y, candidates)

    # If the response range or individual nonzero magnitudes are small against
    # a common level, squaring scaled residuals can erase meaningful differences.
    nonzero = np.abs(y[y != 0])
    if not len(nonzero):
        return initial
    min_exp = int(np.frexp(nonzero.min())[1])
    exponent = int(np.frexp(nonzero.max())[1])
    with np.errstate(over="ignore"):
        span = y.max() - y.min()
    if span > 0 and np.isfinite(span):
        min_exp = min(min_exp, int(np.frexp(span)[1]))
    if exponent - min_exp > 26:  # squaring doubles the exponent spread
        return _exact_minimum(x, y, candidates)

    y = np.ldexp(y, -exponent)
    center = max(0.0, float(y.min())) + min(0.0, float(y.max()))
    y = y - center
    high, low = _prefix_sum(y)
    losses = np.array([_window_loss(x, y, center, high, low, h) for h in candidates])
    return float(candidates[losses.argmin()])
