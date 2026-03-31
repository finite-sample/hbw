"""Test that Newton optimization matches grid search minimum."""

import math
import random

import numpy as np

from hbw import kde_bandwidth, lscv


def mixture_sample(n: int, rng: random.Random) -> np.ndarray:
    """Generate samples from a Gaussian mixture."""
    samples = []
    for _ in range(n):
        if rng.random() < 0.5:
            samples.append(rng.gauss(-2, 0.5))
        else:
            samples.append(rng.gauss(2, 1.0))
    return np.array(samples)


def test_newton_matches_grid_search() -> None:
    """Verify Newton finds approximately the same optimum as grid search."""
    rng = random.Random(1)
    x = mixture_sample(30, rng)
    grid = [0.1 + i * (2.0 - 0.1) / 39 for i in range(40)]
    for kernel in ["gauss", "epan"]:
        scores = [lscv(x, h, kernel)[0] for h in grid]
        h_grid = grid[scores.index(min(scores))]

        h_newton = kde_bandwidth(x, kernel=kernel, h0=h_grid * 1.1, max_n=None)
        assert math.isclose(h_newton, h_grid, rel_tol=0.1, abs_tol=0.05)


def test_kde_bandwidth_basic() -> None:
    """Test basic functionality of kde_bandwidth."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    h = kde_bandwidth(x)
    assert 0.1 < h < 2.0


def test_kde_bandwidth_subsampling() -> None:
    """Test that subsampling works for large data."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10000)
    h = kde_bandwidth(x, max_n=500, seed=42)
    assert 0.1 < h < 2.0
