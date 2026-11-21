"""Benchmark Newton vs Grid Search implementations across sample sizes and kernels."""

import time
from collections.abc import Callable
from functools import partial

import numpy as np

from hbw._numba_kde import (
    lscv_mv_numba_gauss,
    lscv_score_numba_biweight,
    lscv_score_numba_cosine,
    lscv_score_numba_epan,
    lscv_score_numba_gauss,
    lscv_score_numba_triweight,
    lscv_score_numba_unif,
)
from hbw._numba_kde import warmup as warmup_kde
from hbw._numba_nw import (
    loocv_score_mv_numba_gauss,
    loocv_score_numba_biweight,
    loocv_score_numba_cosine,
    loocv_score_numba_epan,
    loocv_score_numba_gauss,
    loocv_score_numba_triweight,
    loocv_score_numba_unif,
)
from hbw._numba_nw import warmup as warmup_nw
from hbw._optim import _silverman_h
from hbw.kde import kde_bandwidth, kde_bandwidth_mv, lscv_mv
from hbw.nw import loocv_mse_mv, nw_bandwidth, nw_bandwidth_mv

KERNELS = ["gauss", "epan", "unif", "biweight", "triweight", "cosine"]

HEADER = (
    "Kernel       | Grid (50 pts) | Newton     | Speedup | "
    "Grid h | Newton h | Grid loss | Newton loss"
)

KDE_SCORE_FUNCS = {
    "gauss": lscv_score_numba_gauss,
    "epan": lscv_score_numba_epan,
    "unif": lscv_score_numba_unif,
    "biweight": lscv_score_numba_biweight,
    "triweight": lscv_score_numba_triweight,
    "cosine": lscv_score_numba_cosine,
}

NW_SCORE_FUNCS = {
    "gauss": loocv_score_numba_gauss,
    "epan": loocv_score_numba_epan,
    "unif": loocv_score_numba_unif,
    "biweight": loocv_score_numba_biweight,
    "triweight": loocv_score_numba_triweight,
    "cosine": loocv_score_numba_cosine,
}


def grid_search_kde(x: np.ndarray, kernel: str, n_grid: int = 50) -> float:
    """Grid search using Numba-accelerated score function."""
    h0 = _silverman_h(x, kernel)
    h_grid = np.logspace(np.log10(h0 * 0.1), np.log10(h0 * 3), n_grid)
    score_fn = KDE_SCORE_FUNCS[kernel]
    scores = [score_fn(x, h) for h in h_grid]
    return h_grid[np.argmin(scores)]


def grid_search_nw(
    x: np.ndarray, y: np.ndarray, kernel: str, n_grid: int = 50
) -> float:
    """Grid search using Numba-accelerated score function."""
    h0 = _silverman_h(x, kernel)
    h_grid = np.logspace(np.log10(h0 * 0.1), np.log10(h0 * 3), n_grid)
    score_fn = NW_SCORE_FUNCS[kernel]
    scores = [score_fn(x, y, h) for h in h_grid]
    return h_grid[np.argmin(scores)]


def report_comparison(
    kernel: str,
    grid: Callable[[], float],
    newton: Callable[[], float],
    loss: Callable[[float], float],
) -> None:
    """Time both selectors and score their selected bandwidths outside the timer."""
    selectors = (("grid", grid), ("newton", newton))
    samples: dict[str, list[float]] = {"grid": [], "newton": []}
    bandwidths = {}
    for block in range(4):
        order = selectors if block % 2 == 0 else reversed(selectors)
        for name, select in order:
            start = time.perf_counter()
            h = select()
            elapsed = time.perf_counter() - start
            if block:  # First fit warms up each selector and its compiled kernels.
                samples[name].append(elapsed * 1000)
            bandwidths[name] = h
    grid_ms = float(np.median(samples["grid"]))
    newton_ms = float(np.median(samples["newton"]))
    grid_h, newton_h = bandwidths["grid"], bandwidths["newton"]
    print(
        f"{kernel:<12} | {grid_ms:>9.2f} ms | {newton_ms:>7.2f} ms | "
        f"{grid_ms / newton_ms:>6.2f}x | {grid_h:.6g} | {newton_h:.6g} | "
        f"{loss(grid_h):.9g} | {loss(newton_h):.9g}"
    )


def run_kde_benchmarks(rng: np.random.Generator) -> None:
    """Run KDE benchmarks comparing Newton vs Grid Search across kernels."""
    print("\n" + "=" * 80)
    print("KDE BANDWIDTH SELECTION: NEWTON vs GRID SEARCH")
    print("=" * 80)
    print("\nCompares Newton optimization vs Grid Search (50 points).")
    print("Both methods use Numba-accelerated score functions.\n")

    sample_sizes = [1000, 2000, 5000]

    for n in sample_sizes:
        print(f"\n--- n = {n} ---")
        print(HEADER)
        print("-" * len(HEADER))

        x = rng.standard_normal(n)

        for kernel in KERNELS:
            report_comparison(
                kernel,
                partial(grid_search_kde, x, kernel),
                partial(kde_bandwidth, x, kernel=kernel, max_n=None),
                partial(KDE_SCORE_FUNCS[kernel], x),
            )


def run_nw_benchmarks(rng: np.random.Generator) -> None:
    """Run NW benchmarks comparing Newton vs Grid Search across kernels."""
    print("\n" + "=" * 80)
    print("NW REGRESSION BANDWIDTH SELECTION: NEWTON vs GRID SEARCH")
    print("=" * 80)
    print("\nCompares Newton optimization vs Grid Search (50 points).")
    print("Both methods use Numba-accelerated score functions.\n")

    sample_sizes = [1000, 2000, 5000]

    for n in sample_sizes:
        print(f"\n--- n = {n} ---")
        print(HEADER)
        print("-" * len(HEADER))

        x = rng.uniform(-3, 3, n)
        y = np.sin(x) + 0.3 * rng.standard_normal(n)

        for kernel in KERNELS:
            report_comparison(
                kernel,
                partial(grid_search_nw, x, y, kernel),
                partial(nw_bandwidth, x, y, kernel=kernel, max_n=None),
                partial(NW_SCORE_FUNCS[kernel], x, y),
            )


def gaussian_kde_loss_mv(data: np.ndarray, h: float) -> float:
    return lscv_mv_numba_gauss(data, h)[0]


def grid_search_kde_mv(data: np.ndarray, kernel: str, n_grid: int = 50) -> float:
    """Grid search a multivariate KDE bandwidth with the Numba score function."""
    n, d = data.shape
    std_avg = float(np.mean(np.std(data, axis=0, ddof=1)))
    h_init = std_avg * n ** (-1.0 / (d + 4))
    h_grid = np.logspace(np.log10(h_init * 0.1), np.log10(h_init * 3), n_grid)
    if kernel == "gauss":
        scores = [gaussian_kde_loss_mv(data, h) for h in h_grid]
    else:
        scores = [lscv_mv(data, h, kernel)[0] for h in h_grid]
    return h_grid[np.argmin(scores)]


def grid_search_nw_mv(
    data: np.ndarray, y: np.ndarray, kernel: str, n_grid: int = 50
) -> float:
    """Grid search a multivariate NW bandwidth with the Numba score function."""
    n, d = data.shape
    std_avg = float(np.mean(np.std(data, axis=0, ddof=1)))
    h_init = std_avg * n ** (-1.0 / (d + 4))
    h_grid = np.logspace(np.log10(h_init * 0.1), np.log10(h_init * 3), n_grid)
    if kernel == "gauss":
        scores = [loocv_score_mv_numba_gauss(data, y, h) for h in h_grid]
    else:
        scores = [loocv_mse_mv(data, y, h, kernel)[0] for h in h_grid]
    return h_grid[np.argmin(scores)]


def run_kde_mv_benchmarks(rng: np.random.Generator) -> None:
    """Run multivariate KDE benchmarks comparing Newton vs Grid Search."""
    print("\n" + "=" * 80)
    print("MULTIVARIATE KDE BANDWIDTH SELECTION: NEWTON vs GRID SEARCH")
    print("=" * 80)
    print("\nCompares Newton optimization vs Grid Search (50 points).")
    print("Gaussian kernel only for multivariate.\n")

    sample_sizes = [500, 1000]
    dims = [2, 3]

    for n in sample_sizes:
        for d in dims:
            print(f"\n--- n = {n}, d = {d} ---")
            print(HEADER)
            print("-" * len(HEADER))

            data = rng.standard_normal((n, d))

            report_comparison(
                "gauss",
                partial(grid_search_kde_mv, data, "gauss"),
                partial(
                    kde_bandwidth_mv,
                    data,
                    kernel="gauss",
                    max_n=None,
                    standardize=False,
                ),
                partial(gaussian_kde_loss_mv, data),
            )


def run_nw_mv_benchmarks(rng: np.random.Generator) -> None:
    """Run multivariate NW benchmarks comparing Newton vs Grid Search."""
    print("\n" + "=" * 80)
    print("MULTIVARIATE NW REGRESSION BANDWIDTH SELECTION: NEWTON vs GRID SEARCH")
    print("=" * 80)
    print("\nCompares Newton optimization vs Grid Search (50 points).")
    print("Gaussian kernel only for multivariate.\n")

    sample_sizes = [500, 1000]
    dims = [2, 3]

    for n in sample_sizes:
        for d in dims:
            print(f"\n--- n = {n}, d = {d} ---")
            print(HEADER)
            print("-" * len(HEADER))

            data = rng.standard_normal((n, d))
            y = np.sin(data[:, 0]) + 0.5 * data[:, 1] + 0.3 * rng.standard_normal(n)

            report_comparison(
                "gauss",
                partial(grid_search_nw_mv, data, y, "gauss"),
                partial(
                    nw_bandwidth_mv,
                    data,
                    y,
                    kernel="gauss",
                    max_n=None,
                    standardize=False,
                ),
                partial(loocv_score_mv_numba_gauss, data, y),
            )


def run_benchmarks() -> None:
    """Run all benchmarks."""
    print("Warming up Numba JIT compilation...")
    warmup_kde()
    warmup_nw()
    print("Warmup complete.")

    rng = np.random.default_rng(42)

    print("Lower loss is better. Multivariate bandwidths use raw input units.")
    run_kde_benchmarks(rng)
    run_nw_benchmarks(rng)
    run_kde_mv_benchmarks(rng)
    run_nw_mv_benchmarks(rng)


if __name__ == "__main__":
    run_benchmarks()
