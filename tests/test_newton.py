import math
import random

from .utils import lscv_generic, newton_opt


def mixture_sample(n, rng):
    samples = []
    for _ in range(n):
        if rng.random() < 0.5:
            samples.append(rng.gauss(-2, 0.5))
        else:
            samples.append(rng.gauss(2, 1.0))
    return samples


def test_newton_matches_grid_search():
    rng = random.Random(1)
    x = mixture_sample(30, rng)
    grid = [0.1 + i * (2.0 - 0.1) / 39 for i in range(40)]
    for kernel in ["gauss", "epan"]:
        scores = [lscv_generic(x, h, kernel)[0] for h in grid]
        h_grid = grid[scores.index(min(scores))]

        h_start = h_grid * 1.1
        h_newton, _ = newton_opt(x, h_start, lambda data, h: lscv_generic(data, h, kernel))
        assert math.isclose(h_newton, h_grid, rel_tol=0.1, abs_tol=0.05)
