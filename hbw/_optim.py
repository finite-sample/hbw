"""Newton-Armijo optimizer for bandwidth selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _silverman_h(x: NDArray[Any]) -> float:
    """Silverman's rule of thumb for initial bandwidth."""
    n = len(x)
    std = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * scale * n ** (-0.2)


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
) -> float:
    """Run Newton-Armijo optimization for bandwidth selection."""
    h = h0
    for _ in range(max_iter):
        if y is None:
            f, g, H = objective(x, h, kernel)
        else:
            f, g, H = objective(x, y, h, kernel)
        if abs(g) < tol:
            break
        step = -g / H if (H > 0 and np.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            if y is None:
                f_new = objective(x, h_new, kernel)[0]
            else:
                f_new = objective(x, y, h_new, kernel)[0]
            if f_new < f:
                h = h_new
                break
            step *= 0.5
    return h
