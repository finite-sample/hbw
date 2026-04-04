"""Generate publication-quality figures for the hbw paper."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.stats import t as t_dist

from hbw import (_KERNELS, kde_bandwidth, kde_bandwidth_mv, loocv_mse,
                 loocv_mse_score, lscv, lscv_mv, lscv_score,
                 nw_bandwidth, nw_bandwidth_mv)
from hbw._numba_kde import lscv_mv_numba_gauss
from hbw._numba_nw import loocv_score_mv_numba_gauss

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# DGP Functions
# =============================================================================


def sample_bimodal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bimodal: 50-50 mixture of N(-2, 0.5) and N(2, 1)."""
    return np.where(
        rng.random(n) < 0.5,
        rng.normal(-2, 0.5, n),
        rng.normal(2, 1.0, n),
    )


def pdf_bimodal(z: np.ndarray) -> np.ndarray:
    """True density for bimodal mixture."""
    return 0.5 * norm.pdf(z, -2, 0.5) + 0.5 * norm.pdf(z, 2, 1.0)


def sample_unimodal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unimodal: standard normal."""
    return rng.normal(0, 1, n)


def pdf_unimodal(z: np.ndarray) -> np.ndarray:
    """True density for unimodal."""
    return norm.pdf(z, 0, 1)


def sample_skewed(n: int, rng: np.random.Generator) -> np.ndarray:
    """Skewed: log-normal with mu=0, sigma=0.5."""
    return rng.lognormal(0, 0.5, n)


def pdf_skewed(z: np.ndarray) -> np.ndarray:
    """True density for log-normal(0, 0.5)."""
    sigma = 0.5
    pdf = np.zeros_like(z)
    mask = z > 0
    pdf[mask] = np.exp(-0.5 * (np.log(z[mask]) / sigma) ** 2) / (
        z[mask] * sigma * np.sqrt(2 * np.pi)
    )
    return pdf


def sample_heavy_tailed(n: int, rng: np.random.Generator) -> np.ndarray:
    """Heavy-tailed: t-distribution with df=3."""
    return rng.standard_t(df=3, size=n)


def pdf_heavy_tailed(z: np.ndarray) -> np.ndarray:
    """True density for t(df=3)."""
    return t_dist.pdf(z, df=3)


DGPS = {
    "bimodal": (sample_bimodal, pdf_bimodal, (-8, 8)),
    "unimodal": (sample_unimodal, pdf_unimodal, (-5, 5)),
    "skewed": (sample_skewed, pdf_skewed, (0.01, 6)),
    "heavy_tailed": (sample_heavy_tailed, pdf_heavy_tailed, (-8, 8)),
}


def sample_bimodal_mv(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Multivariate bimodal: mixture of two Gaussians centered at different locations."""
    data = np.zeros((n, d))
    mask = rng.random(n) < 0.5
    mean1 = np.ones(d) * -1.5
    mean2 = np.ones(d) * 1.5
    data[mask] = rng.normal(0, 0.7, (mask.sum(), d)) + mean1
    data[~mask] = rng.normal(0, 1.0, ((~mask).sum(), d)) + mean2
    return data


def pdf_bimodal_mv(z: np.ndarray) -> np.ndarray:
    """True density for multivariate bimodal mixture (product of marginals)."""
    d = z.shape[-1] if z.ndim > 1 else 1
    pdf = np.ones(z.shape[0])
    for k in range(d):
        zk = z[:, k] if z.ndim > 1 else z
        pdf *= 0.5 * norm.pdf(zk, -1.5, 0.7) + 0.5 * norm.pdf(zk, 1.5, 1.0)
    return pdf


# =============================================================================
# Bandwidth Selection Methods
# =============================================================================


def silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule of thumb."""
    n = len(x)
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * scale * n ** (-0.2)


def timed_grid_search_kde(
    x: np.ndarray, kernel: str, n_grid: int = 50
) -> tuple[float, int, float]:
    """Grid search for KDE bandwidth with timing."""
    h_silv = silverman_bandwidth(x)
    grid = np.logspace(np.log10(h_silv * 0.1), np.log10(h_silv * 5), n_grid)
    t0 = time.perf_counter()
    scores = [lscv_score(x, h, kernel) for h in grid]
    elapsed = time.perf_counter() - t0
    return float(grid[np.argmin(scores)]), n_grid, elapsed


def timed_golden_section_kde(x: np.ndarray, kernel: str) -> tuple[float, int, float]:
    """Bounded scalar search for KDE bandwidth with timing."""
    evals = [0]
    h_silv = silverman_bandwidth(x)

    def obj(h: float) -> float:
        evals[0] += 1
        return lscv(x, h, kernel)[0]

    t0 = time.perf_counter()
    res = minimize_scalar(
        obj,
        bounds=(h_silv * 0.1, h_silv * 5),
        method="bounded",
        options={"xatol": 1e-3},
    )
    elapsed = time.perf_counter() - t0
    return float(res.x), evals[0], elapsed


def timed_newton_kde(x: np.ndarray, kernel: str) -> tuple[float, int, float]:
    """Newton for KDE bandwidth with timing."""
    t0 = time.perf_counter()
    h_opt = kde_bandwidth(x, kernel=kernel, max_n=None)
    elapsed = time.perf_counter() - t0
    return h_opt, 8, elapsed


def timed_silverman(x: np.ndarray) -> tuple[float, int, float]:
    """Silverman's rule with timing."""
    t0 = time.perf_counter()
    h_opt = silverman_bandwidth(x)
    elapsed = time.perf_counter() - t0
    return h_opt, 1, elapsed


def timed_grid_search_kde_mv(
    data: np.ndarray, kernel: str, n_grid: int = 50
) -> tuple[float, int, float]:
    """Grid search for multivariate KDE bandwidth with timing."""
    n, d = data.shape
    std_avg = float(np.mean(np.std(data, axis=0, ddof=1)))
    h_init = std_avg * n ** (-1.0 / (d + 4))
    grid = np.logspace(np.log10(h_init * 0.1), np.log10(h_init * 5), n_grid)
    t0 = time.perf_counter()
    if kernel == "gauss":
        scores = [lscv_mv_numba_gauss(data, h)[0] for h in grid]
    else:
        scores = [lscv_mv(data, h, kernel)[0] for h in grid]
    elapsed = time.perf_counter() - t0
    return float(grid[np.argmin(scores)]), n_grid, elapsed


def timed_newton_kde_mv(data: np.ndarray, kernel: str) -> tuple[float, int, float]:
    """Newton for multivariate KDE bandwidth with timing."""
    t0 = time.perf_counter()
    h_opt = kde_bandwidth_mv(data, kernel=kernel, max_n=None, standardize=False)
    elapsed = time.perf_counter() - t0
    return h_opt, 8, elapsed


def timed_grid_search_nw(
    x: np.ndarray, y: np.ndarray, kernel: str, n_grid: int = 50
) -> tuple[float, int, float]:
    """Grid search for NW bandwidth with timing."""
    h_silv = silverman_bandwidth(x)
    grid = np.logspace(np.log10(h_silv * 0.1), np.log10(h_silv * 5), n_grid)
    t0 = time.perf_counter()
    scores = [loocv_mse_score(x, y, h, kernel) for h in grid]
    elapsed = time.perf_counter() - t0
    return float(grid[np.argmin(scores)]), n_grid, elapsed


def timed_golden_section_nw(
    x: np.ndarray, y: np.ndarray, kernel: str
) -> tuple[float, int, float]:
    """Bounded scalar search for NW bandwidth with timing."""
    evals = [0]
    h_silv = silverman_bandwidth(x)

    def obj(h: float) -> float:
        evals[0] += 1
        return loocv_mse(x, y, h, kernel)[0]

    t0 = time.perf_counter()
    res = minimize_scalar(
        obj,
        bounds=(h_silv * 0.1, h_silv * 5),
        method="bounded",
        options={"xatol": 1e-3},
    )
    elapsed = time.perf_counter() - t0
    return float(res.x), evals[0], elapsed


def timed_newton_nw(
    x: np.ndarray, y: np.ndarray, kernel: str
) -> tuple[float, int, float]:
    """Newton for NW bandwidth with timing."""
    t0 = time.perf_counter()
    h_opt = nw_bandwidth(x, y, kernel=kernel, max_n=None)
    elapsed = time.perf_counter() - t0
    return h_opt, 8, elapsed


# =============================================================================
# ISE/MSE Evaluation
# =============================================================================


def ise_kde(
    x: np.ndarray, h: float, kernel: str, pdf_func, bounds: tuple[float, float]
) -> float:
    """Integrated squared error for KDE."""
    K = _KERNELS[kernel][0]
    zz = np.linspace(bounds[0], bounds[1], 601)
    est = np.array([K((z - x) / h).mean() / h for z in zz])
    return float(np.trapezoid((est - pdf_func(zz)) ** 2, zz))


def mse_nw(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    h: float,
    kernel: str,
) -> float:
    """Test MSE for NW regression."""
    K = _KERNELS[kernel][0]
    preds = []
    for xt in x_test:
        u = (xt - x_train) / h
        w = K(u)
        if w.sum() > 0:
            preds.append(np.sum(w * y_train) / w.sum())
        else:
            preds.append(0.0)
    return float(np.mean((np.array(preds) - y_test) ** 2))


def ise_kde_mv(
    data: np.ndarray,
    h: float,
    kernel: str,
    n_grid: int = 30,
) -> float:
    """Integrated squared error for multivariate KDE via Monte Carlo integration."""
    K = _KERNELS[kernel][0]
    n, d = data.shape

    grid_1d = np.linspace(-5, 5, n_grid)
    grids = np.meshgrid(*[grid_1d] * d, indexing="ij")
    grid_points = np.stack([g.ravel() for g in grids], axis=1)

    est = np.zeros(len(grid_points))
    for i, z in enumerate(grid_points):
        u = (z - data) / h
        est[i] = np.prod(K(u), axis=1).mean() / (h**d)

    true_pdf = pdf_bimodal_mv(grid_points)

    vol = (10.0**d) / len(grid_points)
    return float(np.sum((est - true_pdf) ** 2) * vol)


# =============================================================================
# Simulation Functions
# =============================================================================


def run_kde_simulations(
    ns: list[int],
    kernels: list[str],
    dgp_names: list[str],
    n_reps: int = 20,
) -> pd.DataFrame:
    """Run KDE simulation study with timing."""
    results = []
    for n in ns:
        for dgp_name in dgp_names:
            sample_func, pdf_func, bounds = DGPS[dgp_name]
            for kernel in kernels:
                for rep in range(n_reps):
                    rng = np.random.default_rng(n * 1000 + hash(dgp_name) % 1000 + rep)
                    x = sample_func(n, rng)

                    h_grid, ev_grid, t_grid = timed_grid_search_kde(x, kernel)
                    ise_grid = ise_kde(x, h_grid, kernel, pdf_func, bounds)

                    h_gold, ev_gold, t_gold = timed_golden_section_kde(x, kernel)
                    ise_gold = ise_kde(x, h_gold, kernel, pdf_func, bounds)

                    h_newt, ev_newt, t_newt = timed_newton_kde(x, kernel)
                    ise_newt = ise_kde(x, h_newt, kernel, pdf_func, bounds)

                    h_silv, ev_silv, t_silv = timed_silverman(x)
                    ise_silv = ise_kde(x, h_silv, kernel, pdf_func, bounds)

                    for method, h, ise, evals, t in [
                        ("Grid", h_grid, ise_grid, ev_grid, t_grid),
                        ("Golden", h_gold, ise_gold, ev_gold, t_gold),
                        ("Newton", h_newt, ise_newt, ev_newt, t_newt),
                        ("Silverman", h_silv, ise_silv, ev_silv, t_silv),
                    ]:
                        results.append(
                            {
                                "n": n,
                                "dgp": dgp_name,
                                "kernel": kernel,
                                "method": method,
                                "h": h,
                                "ISE": ise,
                                "evals": evals,
                                "time_ms": t * 1000,
                                "rep": rep,
                            }
                        )
                    print(f"KDE: n={n}, dgp={dgp_name}, kernel={kernel}, rep={rep}")
    return pd.DataFrame(results)


def run_nw_simulations(
    ns: list[int],
    noises: list[float],
    kernels: list[str],
    n_reps: int = 20,
) -> pd.DataFrame:
    """Run NW regression simulation study with timing."""
    results = []
    for n in ns:
        for noise in noises:
            for kernel in kernels:
                for rep in range(n_reps):
                    rng = np.random.default_rng(
                        n * 1000 + int(noise * 100) + rep + 5000
                    )
                    x = rng.uniform(-2, 2, n)
                    y = np.sin(2 * x) + rng.normal(0, noise, n)

                    x_test = np.linspace(-2, 2, 1000)
                    y_test = np.sin(2 * x_test)

                    h_grid, ev_grid, t_grid = timed_grid_search_nw(x, y, kernel)
                    mse_grid = mse_nw(x, y, x_test, y_test, h_grid, kernel)

                    h_gold, ev_gold, t_gold = timed_golden_section_nw(x, y, kernel)
                    mse_gold = mse_nw(x, y, x_test, y_test, h_gold, kernel)

                    h_newt, ev_newt, t_newt = timed_newton_nw(x, y, kernel)
                    mse_newt = mse_nw(x, y, x_test, y_test, h_newt, kernel)

                    h_silv, ev_silv, t_silv = timed_silverman(x)
                    mse_silv = mse_nw(x, y, x_test, y_test, h_silv, kernel)

                    for method, h, mse, evals, t in [
                        ("Grid", h_grid, mse_grid, ev_grid, t_grid),
                        ("Golden", h_gold, mse_gold, ev_gold, t_gold),
                        ("Newton", h_newt, mse_newt, ev_newt, t_newt),
                        ("Silverman", h_silv, mse_silv, ev_silv, t_silv),
                    ]:
                        results.append(
                            {
                                "n": n,
                                "noise": noise,
                                "kernel": kernel,
                                "method": method,
                                "h": h,
                                "MSE": mse,
                                "evals": evals,
                                "time_ms": t * 1000,
                                "rep": rep,
                            }
                        )
                    print(f"NW: n={n}, noise={noise}, kernel={kernel}, rep={rep}")
    return pd.DataFrame(results)


def run_bootstrap_timing(
    ns: list[int],
    n_bootstrap: int = 200,
    kernel: str = "gauss",
) -> pd.DataFrame:
    """Run bootstrap timing study for KDE."""
    results = []

    warmup_rng = np.random.default_rng(999)
    x_warmup = sample_bimodal(50, warmup_rng)
    timed_grid_search_kde(x_warmup, kernel)
    timed_golden_section_kde(x_warmup, kernel)
    timed_newton_kde(x_warmup, kernel)

    for n in ns:
        rng = np.random.default_rng(42 + n)
        x_orig = sample_bimodal(n, rng)

        for method in ["Grid", "Golden", "Newton"]:
            t0 = time.perf_counter()
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                x_boot = x_orig[idx]
                if method == "Grid":
                    timed_grid_search_kde(x_boot, kernel)
                elif method == "Golden":
                    timed_golden_section_kde(x_boot, kernel)
                else:
                    timed_newton_kde(x_boot, kernel)
            total_time = time.perf_counter() - t0

            results.append(
                {
                    "n": n,
                    "method": method,
                    "n_bootstrap": n_bootstrap,
                    "total_time_s": total_time,
                    "time_per_boot_ms": total_time / n_bootstrap * 1000,
                }
            )
            print(f"Bootstrap KDE: n={n}, method={method}, total={total_time:.2f}s")

    return pd.DataFrame(results)


def run_bootstrap_timing_nw(
    ns: list[int],
    n_bootstrap: int = 200,
    kernel: str = "gauss",
    noise: float = 1.0,
) -> pd.DataFrame:
    """Run bootstrap timing study for NW regression."""
    results = []

    warmup_rng = np.random.default_rng(999)
    x_warmup = warmup_rng.uniform(-2, 2, 50)
    y_warmup = np.sin(2 * x_warmup) + warmup_rng.normal(0, noise, 50)
    timed_grid_search_nw(x_warmup, y_warmup, kernel)
    timed_golden_section_nw(x_warmup, y_warmup, kernel)
    timed_newton_nw(x_warmup, y_warmup, kernel)

    for n in ns:
        rng = np.random.default_rng(42 + n)
        x_orig = rng.uniform(-2, 2, n)
        y_orig = np.sin(2 * x_orig) + rng.normal(0, noise, n)

        for method in ["Grid", "Golden", "Newton"]:
            t0 = time.perf_counter()
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                x_boot = x_orig[idx]
                y_boot = y_orig[idx]
                if method == "Grid":
                    timed_grid_search_nw(x_boot, y_boot, kernel)
                elif method == "Golden":
                    timed_golden_section_nw(x_boot, y_boot, kernel)
                else:
                    timed_newton_nw(x_boot, y_boot, kernel)
            total_time = time.perf_counter() - t0

            results.append(
                {
                    "n": n,
                    "method": method,
                    "n_bootstrap": n_bootstrap,
                    "total_time_s": total_time,
                    "time_per_boot_ms": total_time / n_bootstrap * 1000,
                }
            )
            print(f"Bootstrap NW: n={n}, method={method}, total={total_time:.2f}s")

    return pd.DataFrame(results)


def run_mv_kde_simulations(
    ns: list[int],
    dims: list[int],
    kernels: list[str] | None = None,
    n_reps: int = 20,
) -> pd.DataFrame:
    """Run multivariate KDE simulation study."""
    if kernels is None:
        kernels = ["gauss"]
    results = []
    for n in ns:
        for d in dims:
            for kernel in kernels:
                for rep in range(n_reps):
                    rng = np.random.default_rng(n * 1000 + d * 100 + rep + 9000)
                    data = sample_bimodal_mv(n, d, rng)

                    h_grid, ev_grid, t_grid = timed_grid_search_kde_mv(data, kernel)
                    ise_grid = ise_kde_mv(data, h_grid, kernel)

                    h_newt, ev_newt, t_newt = timed_newton_kde_mv(data, kernel)
                    ise_newt = ise_kde_mv(data, h_newt, kernel)

                    for method, h, ise, evals, t in [
                        ("Grid", h_grid, ise_grid, ev_grid, t_grid),
                        ("Newton", h_newt, ise_newt, ev_newt, t_newt),
                    ]:
                        results.append(
                            {
                                "n": n,
                                "d": d,
                                "kernel": kernel,
                                "method": method,
                                "h": h,
                                "ISE": ise,
                                "evals": evals,
                                "time_ms": t * 1000,
                                "rep": rep,
                            }
                        )
                    print(f"MV KDE: n={n}, d={d}, kernel={kernel}, rep={rep}")
    return pd.DataFrame(results)


def sample_nw_mv(n: int, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Multivariate NW: y = sin(x1) + 0.5*x2 + noise."""
    X = rng.normal(0, 1, (n, d))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    y += 0.3 * rng.normal(0, 1, n)
    return X, y


def timed_grid_search_nw_mv(
    data: np.ndarray, y: np.ndarray, kernel: str, n_grid: int = 50
) -> tuple[float, int, float]:
    """Grid search for multivariate NW bandwidth with timing."""
    if kernel != "gauss":
        raise ValueError("Only 'gauss' kernel supported for multivariate NW")
    n, d = data.shape
    std_avg = float(np.mean(np.std(data, axis=0, ddof=1)))
    h_init = std_avg * n ** (-1.0 / (d + 4))
    grid = np.logspace(np.log10(h_init * 0.1), np.log10(h_init * 5), n_grid)
    t0 = time.perf_counter()
    scores = [loocv_score_mv_numba_gauss(data, y, h) for h in grid]
    elapsed = time.perf_counter() - t0
    return float(grid[np.argmin(scores)]), n_grid, elapsed


def timed_newton_nw_mv(
    data: np.ndarray, y: np.ndarray, kernel: str
) -> tuple[float, int, float]:
    """Newton for multivariate NW bandwidth with timing."""
    t0 = time.perf_counter()
    h_opt = nw_bandwidth_mv(data, y, kernel=kernel, max_n=None)
    elapsed = time.perf_counter() - t0
    return h_opt, 8, elapsed


def mse_nw_mv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    h: float,
    kernel: str,
) -> float:
    """Test MSE for multivariate NW regression."""
    K = _KERNELS[kernel][0]
    n_test = X_test.shape[0]
    d = X_train.shape[1]
    preds = []
    for i in range(n_test):
        u = (X_test[i] - X_train) / h
        w = np.prod(K(u), axis=1)
        if w.sum() > 0:
            preds.append(np.sum(w * y_train) / w.sum())
        else:
            preds.append(0.0)
    return float(np.mean((np.array(preds) - y_test) ** 2))


def run_mv_nw_simulations(
    ns: list[int],
    dims: list[int],
    kernels: list[str] | None = None,
    n_reps: int = 20,
) -> pd.DataFrame:
    """Run multivariate NW regression simulation study."""
    if kernels is None:
        kernels = ["gauss"]
    results = []
    for n in ns:
        for d in dims:
            for kernel in kernels:
                for rep in range(n_reps):
                    rng = np.random.default_rng(n * 1000 + d * 100 + rep + 20000)
                    X_train, y_train = sample_nw_mv(n, d, rng)

                    rng_test = np.random.default_rng(rep + 30000)
                    X_test, y_test_noisy = sample_nw_mv(500, d, rng_test)
                    y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1]

                    h_grid, ev_grid, t_grid = timed_grid_search_nw_mv(X_train, y_train, kernel)
                    mse_grid = mse_nw_mv(X_train, y_train, X_test, y_test, h_grid, kernel)

                    h_newt, ev_newt, t_newt = timed_newton_nw_mv(X_train, y_train, kernel)
                    mse_newt = mse_nw_mv(X_train, y_train, X_test, y_test, h_newt, kernel)

                    for method, h, mse, evals, t in [
                        ("Grid", h_grid, mse_grid, ev_grid, t_grid),
                        ("Newton", h_newt, mse_newt, ev_newt, t_newt),
                    ]:
                        results.append(
                            {
                                "n": n,
                                "d": d,
                                "kernel": kernel,
                                "method": method,
                                "h": h,
                                "MSE": mse,
                                "evals": evals,
                                "time_ms": t * 1000,
                                "rep": rep,
                            }
                        )
                    print(f"MV NW: n={n}, d={d}, kernel={kernel}, rep={rep}")
    return pd.DataFrame(results)


def plot_mv_nw_comparison(mv_nw_df: pd.DataFrame) -> None:
    """Plot multivariate NW comparison: MSE and timing by dimension (Gaussian kernel)."""
    df = mv_nw_df[mv_nw_df["kernel"] == "gauss"] if "kernel" in mv_nw_df.columns else mv_nw_df
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    dims = sorted(df["d"].unique())
    methods = ["Grid", "Newton"]
    colors = {"Grid": "C0", "Newton": "C2"}

    ax1 = axes[0]
    width = 0.35
    x = np.arange(len(dims))
    for i, method in enumerate(methods):
        mses = [
            df[(df["d"] == d) & (df["method"] == method)]["MSE"].mean() for d in dims
        ]
        ax1.bar(
            x + i * width, mses, width, label=method, color=colors[method], alpha=0.7
        )

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels([str(d) for d in dims])
    ax1.set_xlabel("Dimension (d)")
    ax1.set_ylabel("Mean Test MSE")
    ax1.set_title("Estimation Accuracy")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    for i, method in enumerate(methods):
        times = [
            df[(df["d"] == d) & (df["method"] == method)]["time_ms"].mean()
            for d in dims
        ]
        ax2.bar(
            x + i * width, times, width, label=method, color=colors[method], alpha=0.7
        )

    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels([str(d) for d in dims])
    ax2.set_xlabel("Dimension (d)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Computation Time")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Multivariate NW: Newton vs Grid Search (n=300, Gaussian)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mv_nw_comparison.pdf")
    plt.close()


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_kde_ise(df: pd.DataFrame) -> None:
    """Plot KDE ISE comparison."""
    kernels = ["gauss", "epan", "biweight", "triweight", "cosine"]
    kernel_names = {
        "gauss": "Gaussian",
        "epan": "Epanechnikov",
        "biweight": "Biweight",
        "triweight": "Triweight",
        "cosine": "Cosine",
    }
    n_kernels = len(kernels)
    ncols = min(3, n_kernels)
    nrows = (n_kernels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, kernel in zip(axes[:n_kernels], kernels, strict=False):
        data = df[(df["kernel"] == kernel) & (df["dgp"] == "bimodal")]
        methods = ["Grid", "Golden", "Newton", "Silverman"]
        for i, method in enumerate(methods):
            d = data[data["method"] == method]["ISE"]
            bp = ax.boxplot(
                [d],
                positions=[i],
                widths=0.6,
                patch_artist=True,
            )
            color = {"Grid": "C0", "Golden": "C1", "Newton": "C2", "Silverman": "C3"}[
                method
            ]
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(f"{kernel_names[kernel]} Kernel")
        ax.set_ylabel("ISE")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[n_kernels:]:
        ax.set_visible(False)

    fig.suptitle("KDE Integrated Squared Error by Method", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kde_ise_comparison.pdf")
    plt.close()


def plot_nw_mse(df: pd.DataFrame) -> None:
    """Plot NW MSE comparison."""
    kernels = ["gauss", "epan", "biweight", "triweight", "cosine"]
    kernel_names = {
        "gauss": "Gaussian",
        "epan": "Epanechnikov",
        "biweight": "Biweight",
        "triweight": "Triweight",
        "cosine": "Cosine",
    }
    n_kernels = len(kernels)
    ncols = min(3, n_kernels)
    nrows = (n_kernels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, kernel in zip(axes[:n_kernels], kernels, strict=False):
        data = df[df["kernel"] == kernel]
        methods = ["Grid", "Golden", "Newton", "Silverman"]
        for i, method in enumerate(methods):
            d = data[data["method"] == method]["MSE"]
            bp = ax.boxplot(
                [d],
                positions=[i],
                widths=0.6,
                patch_artist=True,
            )
            color = {"Grid": "C0", "Golden": "C1", "Newton": "C2", "Silverman": "C3"}[
                method
            ]
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(f"{kernel_names[kernel]} Kernel")
        ax.set_ylabel("Test MSE")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[n_kernels:]:
        ax.set_visible(False)

    fig.suptitle("NW Regression Test MSE by Method", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nw_mse_comparison.pdf")
    plt.close()


def plot_eval_counts(kde_df: pd.DataFrame, nw_df: pd.DataFrame) -> None:
    """Plot evaluation count comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = ["Grid", "Golden", "Newton", "Silverman"]
    kde_means = [kde_df[kde_df["method"] == m]["evals"].mean() for m in methods]
    nw_means = [nw_df[nw_df["method"] == m]["evals"].mean() for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, kde_means, width, label="KDE", color="C0", alpha=0.7)
    ax.bar(x + width / 2, nw_means, width, label="NW", color="C1", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean Evaluations")
    ax.set_title("Objective Evaluations by Method")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eval_count_comparison.pdf")
    plt.close()


def plot_timing_comparison(kde_df: pd.DataFrame, nw_df: pd.DataFrame) -> None:
    """Plot wall-clock timing comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    methods = ["Grid", "Golden", "Newton", "Silverman"]
    colors = {"Grid": "C0", "Golden": "C1", "Newton": "C2", "Silverman": "C3"}

    for ax, (df, title, metric) in zip(
        axes,
        [(kde_df, "KDE", "time_ms"), (nw_df, "NW Regression", "time_ms")],
        strict=True,
    ):
        ns = sorted(df["n"].unique())
        width = 0.2
        x = np.arange(len(ns))

        for i, method in enumerate(methods):
            means = [
                df[(df["n"] == n) & (df["method"] == method)][metric].mean() for n in ns
            ]
            ax.bar(
                x + i * width,
                means,
                width,
                label=method,
                color=colors[method],
                alpha=0.7,
            )

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{title} Bandwidth Selection")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Wall-Clock Time by Method and Sample Size", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "timing_comparison.pdf")
    plt.close()


def plot_dgp_comparison(kde_df: pd.DataFrame) -> None:
    """Plot ISE across different DGPs."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    dgp_names = ["bimodal", "unimodal", "skewed", "heavy_tailed"]
    dgp_titles = [
        "Bimodal Mixture",
        "Unimodal (Normal)",
        "Skewed (Log-normal)",
        "Heavy-tailed (t₃)",
    ]

    methods = ["Grid", "Golden", "Newton", "Silverman"]
    colors = {"Grid": "C0", "Golden": "C1", "Newton": "C2", "Silverman": "C3"}

    for ax, dgp, title in zip(axes, dgp_names, dgp_titles, strict=True):
        data = kde_df[(kde_df["dgp"] == dgp) & (kde_df["kernel"] == "gauss")]
        for i, method in enumerate(methods):
            d = data[data["method"] == method]["ISE"]
            bp = ax.boxplot(
                [d],
                positions=[i],
                widths=0.6,
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor(colors[method])
            bp["boxes"][0].set_alpha(0.7)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("ISE")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "KDE Performance Across Data Generating Processes (Gaussian kernel)", y=1.02
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dgp_comparison.pdf")
    plt.close()


def plot_bootstrap_speedup(boot_df: pd.DataFrame) -> None:
    """Plot bootstrap timing speedup."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = ["Grid", "Golden", "Newton"]
    colors = {"Grid": "C0", "Golden": "C1", "Newton": "C2"}
    ns = sorted(boot_df["n"].unique())

    ax1 = axes[0]
    width = 0.25
    x = np.arange(len(ns))
    for i, method in enumerate(methods):
        times = [
            boot_df[(boot_df["n"] == n) & (boot_df["method"] == method)][
                "total_time_s"
            ].values[0]
            for n in ns
        ]
        ax1.bar(
            x + i * width, times, width, label=method, color=colors[method], alpha=0.7
        )

    ax1.set_xticks(x + width)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_xlabel("Sample Size (n)")
    ax1.set_ylabel("Total Time (seconds)")
    ax1.set_title("Time for 200 Bootstrap Resamples")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    for method in methods:
        times = [
            boot_df[(boot_df["n"] == n) & (boot_df["method"] == method)][
                "total_time_s"
            ].values[0]
            for n in ns
        ]
        grid_times = [
            boot_df[(boot_df["n"] == n) & (boot_df["method"] == "Grid")][
                "total_time_s"
            ].values[0]
            for n in ns
        ]
        speedup = [g / t for g, t in zip(grid_times, times, strict=True)]
        ax2.plot(ns, speedup, "o-", label=method, color=colors[method], markersize=8)

    ax2.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Sample Size (n)")
    ax2.set_ylabel("Speedup vs Grid Search")
    ax2.set_title("Speedup Factor (Grid / Method)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Bootstrap Bandwidth Selection Performance", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bootstrap_speedup.pdf")
    plt.close()


def plot_convergence_trace() -> None:
    """Plot Newton convergence trace."""
    rng = np.random.default_rng(42)
    x = sample_bimodal(200, rng)

    h = 2.0
    trace = []
    for _ in range(12):
        f, g, H = lscv(x, h, "gauss")
        trace.append((h, f))
        if abs(g) < 1e-5:
            break
        step = -g / H if (H > 0 and np.isfinite(H)) else -0.25 * g
        if abs(step) / h < 1e-3:
            break
        for _ in range(10):
            h_new = max(h + step, 1e-6)
            if lscv(x, h_new, "gauss")[0] < f:
                h = h_new
                break
            step *= 0.5
    trace.append((h, lscv(x, h, "gauss")[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    hs = [t[0] for t in trace]
    ax1.plot(range(len(hs)), hs, "o-", color="C0", markersize=8)
    ax1.axhline(
        hs[-1], color="C2", linestyle="--", alpha=0.7, label=f"$h^* = {hs[-1]:.3f}$"
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Bandwidth $h$")
    ax1.set_title("Bandwidth Convergence")
    ax1.legend()
    ax1.grid(alpha=0.3)

    fs = [t[1] for t in trace]
    ax2.plot(range(len(fs)), fs, "o-", color="C1", markersize=8)
    ax2.axhline(
        fs[-1], color="C2", linestyle="--", alpha=0.7, label=f"LSCV$^* = {fs[-1]:.4f}$"
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("LSCV Score")
    ax2.set_title("Objective Convergence")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Newton-Armijo Convergence (n=200, Gaussian kernel)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "convergence_trace.pdf")
    plt.close()


def plot_mv_kde_comparison(mv_df: pd.DataFrame) -> None:
    """Plot multivariate KDE comparison: ISE and timing by dimension (Gaussian kernel)."""
    df = mv_df[mv_df["kernel"] == "gauss"] if "kernel" in mv_df.columns else mv_df
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    dims = sorted(df["d"].unique())
    methods = ["Grid", "Newton"]
    colors = {"Grid": "C0", "Newton": "C2"}

    ax1 = axes[0]
    width = 0.35
    x = np.arange(len(dims))
    for i, method in enumerate(methods):
        ises = [
            df[(df["d"] == d) & (df["method"] == method)]["ISE"].mean() for d in dims
        ]
        ax1.bar(
            x + i * width, ises, width, label=method, color=colors[method], alpha=0.7
        )

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels([str(d) for d in dims])
    ax1.set_xlabel("Dimension (d)")
    ax1.set_ylabel("Mean ISE")
    ax1.set_title("Estimation Accuracy")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    for i, method in enumerate(methods):
        times = [
            df[(df["d"] == d) & (df["method"] == method)]["time_ms"].mean()
            for d in dims
        ]
        ax2.bar(
            x + i * width, times, width, label=method, color=colors[method], alpha=0.7
        )

    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels([str(d) for d in dims])
    ax2.set_xlabel("Dimension (d)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Computation Time")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Multivariate KDE: Newton vs Grid Search (n=300, Gaussian)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mv_kde_comparison.pdf")
    plt.close()


def print_summary_tables(
    kde_df: pd.DataFrame, nw_df: pd.DataFrame, boot_df: pd.DataFrame
) -> None:
    """Print summary tables for the paper."""
    print("\n" + "=" * 60)
    print("KDE Results Summary (Bimodal, Gaussian kernel)")
    print("=" * 60)
    bimodal = kde_df[(kde_df["dgp"] == "bimodal") & (kde_df["kernel"] == "gauss")]
    summary = (
        bimodal.groupby("method")
        .agg(
            {
                "ISE": ["mean", "std"],
                "evals": "mean",
                "time_ms": ["mean", "std"],
            }
        )
        .round(4)
    )
    print(summary)

    print("\n" + "=" * 60)
    print("NW Results Summary (Gaussian kernel)")
    print("=" * 60)
    nw_gauss = nw_df[nw_df["kernel"] == "gauss"]
    summary = (
        nw_gauss.groupby("method")
        .agg(
            {
                "MSE": ["mean", "std"],
                "evals": "mean",
                "time_ms": ["mean", "std"],
            }
        )
        .round(4)
    )
    print(summary)

    print("\n" + "=" * 60)
    print("Bootstrap Speedup Summary (200 resamples)")
    print("=" * 60)
    print(boot_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Grid/Newton Speedup Ratios")
    print("=" * 60)
    for n in sorted(boot_df["n"].unique()):
        grid_time = boot_df[(boot_df["n"] == n) & (boot_df["method"] == "Grid")][
            "total_time_s"
        ].values[0]
        newton_time = boot_df[(boot_df["n"] == n) & (boot_df["method"] == "Newton")][
            "total_time_s"
        ].values[0]
        print(
            f"n={n}: Grid={grid_time:.2f}s, Newton={newton_time:.2f}s, Speedup={grid_time/newton_time:.1f}x"
        )


def main() -> None:
    """Generate all figures."""
    print("Running KDE simulations (with multiple DGPs)...")
    kde_df = run_kde_simulations(
        ns=[100, 200, 500],
        kernels=["gauss", "epan", "biweight", "triweight", "cosine"],
        dgp_names=["bimodal", "unimodal", "skewed", "heavy_tailed"],
        n_reps=20,
    )
    kde_df.to_csv(FIGURES_DIR / "kde_results.csv", index=False)

    print("\nRunning NW simulations...")
    nw_df = run_nw_simulations(
        ns=[100, 200, 500],
        noises=[0.5, 1.0, 2.0],
        kernels=["gauss", "epan", "biweight", "triweight", "cosine"],
        n_reps=20,
    )
    nw_df.to_csv(FIGURES_DIR / "nw_results.csv", index=False)

    print("\nRunning bootstrap timing study (KDE)...")
    boot_df = run_bootstrap_timing(
        ns=[100, 200, 500],
        n_bootstrap=200,
        kernel="gauss",
    )
    boot_df.to_csv(FIGURES_DIR / "bootstrap_results.csv", index=False)

    print("\nRunning bootstrap timing study (NW)...")
    boot_nw_df = run_bootstrap_timing_nw(
        ns=[100, 200, 500],
        n_bootstrap=200,
        kernel="gauss",
    )
    boot_nw_df.to_csv(FIGURES_DIR / "bootstrap_nw_results.csv", index=False)

    print("\nRunning multivariate KDE simulations...")
    mv_df = run_mv_kde_simulations(
        ns=[300],
        dims=[2, 3],
        kernels=["gauss", "epan", "biweight", "triweight", "cosine"],
        n_reps=20,
    )
    mv_df.to_csv(FIGURES_DIR / "mv_kde_results.csv", index=False)

    print("\nRunning multivariate NW simulations...")
    mv_nw_df = run_mv_nw_simulations(
        ns=[300],
        dims=[2, 3],
        kernels=["gauss"],
        n_reps=20,
    )
    mv_nw_df.to_csv(FIGURES_DIR / "mv_nw_results.csv", index=False)

    print("\nGenerating figures...")
    plot_kde_ise(kde_df)
    print("  - kde_ise_comparison.pdf")

    plot_nw_mse(nw_df)
    print("  - nw_mse_comparison.pdf")

    plot_eval_counts(kde_df, nw_df)
    print("  - eval_count_comparison.pdf")

    plot_timing_comparison(kde_df, nw_df)
    print("  - timing_comparison.pdf")

    plot_dgp_comparison(kde_df)
    print("  - dgp_comparison.pdf")

    plot_bootstrap_speedup(boot_df)
    print("  - bootstrap_speedup.pdf")

    plot_convergence_trace()
    print("  - convergence_trace.pdf")

    plot_mv_kde_comparison(mv_df)
    print("  - mv_kde_comparison.pdf")

    plot_mv_nw_comparison(mv_nw_df)
    print("  - mv_nw_comparison.pdf")

    print_summary_tables(kde_df, nw_df, boot_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
