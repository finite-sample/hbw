"""Generate LaTeX tables for the hbw paper from simulation CSV data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).parent / "figures"


def load_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Load all CSV data files."""
    kde_df = pd.read_csv(FIGURES_DIR / "kde_results.csv")
    nw_df = pd.read_csv(FIGURES_DIR / "nw_results.csv")
    boot_df = pd.read_csv(FIGURES_DIR / "bootstrap_results.csv")
    mv_df = pd.read_csv(FIGURES_DIR / "mv_kde_results.csv")
    boot_nw_path = FIGURES_DIR / "bootstrap_nw_results.csv"
    if boot_nw_path.exists():
        boot_nw_df = pd.read_csv(boot_nw_path)
    else:
        boot_nw_df = pd.DataFrame()
    mv_nw_path = FIGURES_DIR / "mv_nw_results.csv"
    if mv_nw_path.exists():
        mv_nw_df = pd.read_csv(mv_nw_path)
    else:
        mv_nw_df = pd.DataFrame()
    return kde_df, nw_df, boot_df, mv_df, boot_nw_df, mv_nw_df


def table1_main_results(kde_df: pd.DataFrame, nw_df: pd.DataFrame) -> str:
    """
    Table 1: Bandwidth Selection Performance (Main Results)
    Shows both accuracy and speed for KDE and NW.
    """
    kde_bimodal = kde_df[
        (kde_df["dgp"] == "bimodal")
        & (kde_df["kernel"] == "gauss")
        & (kde_df["n"] == 200)
    ]

    nw_gauss = nw_df[
        (nw_df["kernel"] == "gauss") & (nw_df["noise"] == 1.0) & (nw_df["n"] == 200)
    ]

    methods = ["Grid", "Golden", "Newton", "Silverman"]
    rows = []

    for method in methods:
        kde_m = kde_bimodal[kde_bimodal["method"] == method]
        nw_m = nw_gauss[nw_gauss["method"] == method]

        kde_ise_mean = kde_m["ISE"].mean() * 1000
        kde_ise_se = kde_m["ISE"].std() / np.sqrt(len(kde_m)) * 1000
        kde_time = kde_m["time_ms"].mean()
        evals = int(kde_m["evals"].mean())

        nw_mse_mean = nw_m["MSE"].mean()
        nw_mse_se = nw_m["MSE"].std() / np.sqrt(len(nw_m))
        nw_time = nw_m["time_ms"].mean()

        if method == "Silverman":
            evals_str = "1"
        elif method == "Golden":
            evals_str = f"$\\sim${evals}"
        else:
            evals_str = str(evals)

        rows.append(
            {
                "method": method,
                "evals": evals_str,
                "kde_ise": f"{kde_ise_mean:.2f} ({kde_ise_se:.2f})",
                "kde_time": f"{kde_time:.1f}",
                "nw_mse": f"{nw_mse_mean:.3f} ({nw_mse_se:.3f})",
                "nw_time": f"{nw_time:.1f}",
            }
        )

    latex = r"""\begin{table}[t]
\centering
\caption{Bandwidth selection performance: accuracy and speed for KDE and NW regression.}
\label{tab:main-results}
\begin{tabular}{lcrrcrr}
\toprule
 & & \multicolumn{2}{c}{KDE} & & \multicolumn{2}{c}{NW Regression} \\
\cmidrule(lr){3-4} \cmidrule(lr){6-7}
Method & Evals & ISE ($\times 10^{-3}$) & Time (ms) & & MSE & Time (ms) \\
\midrule
"""
    for r in rows:
        latex += f"{r['method']} & {r['evals']} & {r['kde_ise']} & {r['kde_time']} & & {r['nw_mse']} & {r['nw_time']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
Values are means (standard errors) over 20 replicates. KDE: bimodal DGP, Gaussian kernel, $n=200$. NW: $\sigma=1.0$, Gaussian kernel, $n=200$.
\end{table}
"""
    return latex


def table2_scalability(kde_df: pd.DataFrame, nw_df: pd.DataFrame) -> str:
    """
    Table 2: Scalability with Sample Size
    Shows speedup increases with n.
    """
    ns = [100, 200, 500]
    rows = []

    for n in ns:
        kde_n = kde_df[
            (kde_df["dgp"] == "bimodal")
            & (kde_df["kernel"] == "gauss")
            & (kde_df["n"] == n)
        ]
        nw_n = nw_df[
            (nw_df["kernel"] == "gauss") & (nw_df["noise"] == 1.0) & (nw_df["n"] == n)
        ]

        kde_grid_time = kde_n[kde_n["method"] == "Grid"]["time_ms"].mean()
        kde_newton_time = kde_n[kde_n["method"] == "Newton"]["time_ms"].mean()
        kde_speedup = kde_grid_time / kde_newton_time

        nw_grid_time = nw_n[nw_n["method"] == "Grid"]["time_ms"].mean()
        nw_newton_time = nw_n[nw_n["method"] == "Newton"]["time_ms"].mean()
        nw_speedup = nw_grid_time / nw_newton_time

        rows.append(
            {
                "n": n,
                "kde_grid": f"{kde_grid_time:.1f}",
                "kde_newton": f"{kde_newton_time:.1f}",
                "kde_speedup": f"{kde_speedup:.2f}$\\times$",
                "nw_grid": f"{nw_grid_time:.1f}",
                "nw_newton": f"{nw_newton_time:.1f}",
                "nw_speedup": f"{nw_speedup:.2f}$\\times$",
            }
        )

    latex = r"""\begin{table}[t]
\centering
\caption{Scalability: wall-clock time (ms) and speedup by sample size.}
\label{tab:scalability}
\begin{tabular}{rrrcrrc}
\toprule
 & \multicolumn{3}{c}{KDE} & \multicolumn{3}{c}{NW Regression} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
$n$ & Grid & Newton & Speedup & Grid & Newton & Speedup \\
\midrule
"""
    for r in rows:
        latex += f"{r['n']} & {r['kde_grid']} & {r['kde_newton']} & {r['kde_speedup']} & {r['nw_grid']} & {r['nw_newton']} & {r['nw_speedup']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
Gaussian kernel. KDE: bimodal DGP. NW: $\sigma=1.0$.
\end{table}
"""
    return latex


def table3_bootstrap(boot_df: pd.DataFrame, boot_nw_df: pd.DataFrame) -> str:
    """
    Table 3: Bootstrap Application (200 resamples)
    Shows practical value in repeated estimation for both KDE and NW.
    Side-by-side layout matching the scalability table.
    """
    ns = [100, 200, 500]
    rows = []

    for n in ns:
        kde_grid_time = boot_df[(boot_df["n"] == n) & (boot_df["method"] == "Grid")][
            "total_time_s"
        ].values[0]
        kde_newton_time = boot_df[
            (boot_df["n"] == n) & (boot_df["method"] == "Newton")
        ]["total_time_s"].values[0]
        kde_speedup = kde_grid_time / kde_newton_time

        if not boot_nw_df.empty:
            nw_grid_time = boot_nw_df[
                (boot_nw_df["n"] == n) & (boot_nw_df["method"] == "Grid")
            ]["total_time_s"].values[0]
            nw_newton_time = boot_nw_df[
                (boot_nw_df["n"] == n) & (boot_nw_df["method"] == "Newton")
            ]["total_time_s"].values[0]
            nw_speedup = nw_grid_time / nw_newton_time
        else:
            nw_grid_time = nw_newton_time = nw_speedup = 0.0

        rows.append(
            {
                "n": n,
                "kde_grid": f"{kde_grid_time:.2f}",
                "kde_newton": f"{kde_newton_time:.2f}",
                "kde_speedup": f"{kde_speedup:.2f}$\\times$",
                "nw_grid": f"{nw_grid_time:.2f}",
                "nw_newton": f"{nw_newton_time:.2f}",
                "nw_speedup": f"{nw_speedup:.2f}$\\times$",
            }
        )

    latex = r"""\begin{table}[t]
\centering
\caption{Bootstrap bandwidth selection: total time (seconds) for 200 resamples.}
\label{tab:bootstrap}
\begin{tabular}{rrrcrrc}
\toprule
 & \multicolumn{3}{c}{KDE} & \multicolumn{3}{c}{NW Regression} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
$n$ & Grid & Newton & Speedup & Grid & Newton & Speedup \\
\midrule
"""
    for r in rows:
        latex += f"{r['n']} & {r['kde_grid']} & {r['kde_newton']} & {r['kde_speedup']} & {r['nw_grid']} & {r['nw_newton']} & {r['nw_speedup']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
Gaussian kernel. KDE: bimodal DGP. NW: $\sigma=1.0$. Times in seconds.
\end{table}
"""
    return latex


def table_mv_nw(mv_nw_df: pd.DataFrame) -> str:
    """Table: Multivariate NW Performance by dimension."""
    if mv_nw_df.empty:
        return "% MV NW table: No data available\n"

    dims = sorted(mv_nw_df["d"].unique())
    rows = []

    for d in dims:
        d_data = mv_nw_df[mv_nw_df["d"] == d]

        grid_data = d_data[d_data["method"] == "Grid"]
        newton_data = d_data[d_data["method"] == "Newton"]

        grid_mse = grid_data["MSE"].mean()
        newton_mse = newton_data["MSE"].mean()

        grid_time = grid_data["time_ms"].mean()
        newton_time = newton_data["time_ms"].mean()
        speedup = grid_time / newton_time

        rows.append(
            {
                "d": d,
                "grid_mse": f"{grid_mse:.4f}",
                "newton_mse": f"{newton_mse:.4f}",
                "grid_time": f"{grid_time:.1f}",
                "newton_time": f"{newton_time:.1f}",
                "speedup": f"{speedup:.2f}$\\times$",
            }
        )

    latex = r"""\begin{table}[t]
\centering
\caption{Multivariate NW regression: Newton vs Grid search by dimension ($n=300$, Gaussian kernel).}
\label{tab:mv-nw}
\begin{tabular}{rrrrrr}
\toprule
 & \multicolumn{2}{c}{Test MSE} & \multicolumn{3}{c}{Time (ms)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6}
$d$ & Grid & Newton & Grid & Newton & Speedup \\
\midrule
"""
    for r in rows:
        latex += f"{r['d']} & {r['grid_mse']} & {r['newton_mse']} & {r['grid_time']} & {r['newton_time']} & {r['speedup']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
Regression function: $m(\mathbf{x}) = \sin(x_1) + 0.5 x_2$. Values are means over 20 replicates.
\end{table}
"""
    return latex


def robustness_summary(kde_df: pd.DataFrame, nw_df: pd.DataFrame) -> str:
    """Generate robustness summary statistics for in-text reporting."""
    kernels = ["gauss", "epan", "biweight", "triweight", "cosine"]
    dgps = ["bimodal", "unimodal", "skewed", "heavy_tailed"]

    ratios = []
    speedups = []

    for kernel in kernels:
        for dgp in dgps:
            subset = kde_df[(kde_df["kernel"] == kernel) & (kde_df["dgp"] == dgp)]
            if len(subset) == 0:
                continue

            grid_ise = subset[subset["method"] == "Grid"]["ISE"].mean()
            newton_ise = subset[subset["method"] == "Newton"]["ISE"].mean()

            grid_time = subset[subset["method"] == "Grid"]["time_ms"].mean()
            newton_time = subset[subset["method"] == "Newton"]["time_ms"].mean()

            ratio = newton_ise / grid_ise
            speedup = grid_time / newton_time

            ratios.append(ratio)
            speedups.append(speedup)

    nw_ratios = []
    nw_speedups = []

    for kernel in kernels:
        for noise in [0.5, 1.0, 2.0]:
            subset = nw_df[(nw_df["kernel"] == kernel) & (nw_df["noise"] == noise)]
            if len(subset) == 0:
                continue

            grid_mse = subset[subset["method"] == "Grid"]["MSE"].mean()
            newton_mse = subset[subset["method"] == "Newton"]["MSE"].mean()

            grid_time = subset[subset["method"] == "Grid"]["time_ms"].mean()
            newton_time = subset[subset["method"] == "Newton"]["time_ms"].mean()

            ratio = newton_mse / grid_mse
            speedup = grid_time / newton_time

            nw_ratios.append(ratio)
            nw_speedups.append(speedup)

    summary = f"""
Robustness Summary
==================

KDE (across {len(kernels)} kernels and {len(dgps)} DGPs):
  - Newton/Grid ISE ratio: mean={np.mean(ratios):.3f}, range=[{min(ratios):.3f}, {max(ratios):.3f}]
  - Grid/Newton speedup: mean={np.mean(speedups):.2f}x, range=[{min(speedups):.2f}x, {max(speedups):.2f}x]

NW Regression (across {len(kernels)} kernels and 3 noise levels):
  - Newton/Grid MSE ratio: mean={np.mean(nw_ratios):.3f}, range=[{min(nw_ratios):.3f}, {max(nw_ratios):.3f}]
  - Grid/Newton speedup: mean={np.mean(nw_speedups):.2f}x, range=[{min(nw_speedups):.2f}x, {max(nw_speedups):.2f}x]

Text for paper:
"Results were consistent across all 5 kernel functions (Gaussian, Epanechnikov,
Biweight, Triweight, Cosine) and 4 data generating processes (bimodal, unimodal,
skewed, heavy-tailed). Newton achieved ISE/MSE within {(max(max(ratios), max(nw_ratios)) - 1) * 100:.0f}% of Grid search
across all configurations. For NW regression, Newton provided {np.mean(nw_speedups):.1f}x average speedup."
"""
    return summary


def print_plain_text_tables(
    kde_df: pd.DataFrame,
    nw_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    boot_nw_df: pd.DataFrame,
) -> None:
    """Print plain-text versions of tables for inspection."""
    print("\n" + "=" * 70)
    print("TABLE 1: Main Results (n=200, Gaussian kernel)")
    print("=" * 70)

    kde_bimodal = kde_df[
        (kde_df["dgp"] == "bimodal")
        & (kde_df["kernel"] == "gauss")
        & (kde_df["n"] == 200)
    ]
    nw_gauss = nw_df[
        (nw_df["kernel"] == "gauss") & (nw_df["noise"] == 1.0) & (nw_df["n"] == 200)
    ]

    print(
        f"{'Method':<12} {'Evals':>6} {'KDE ISE (×10⁻³)':>18} {'KDE Time':>10} {'NW MSE':>14} {'NW Time':>10}"
    )
    print("-" * 70)

    for method in ["Grid", "Golden", "Newton", "Silverman"]:
        kde_m = kde_bimodal[kde_bimodal["method"] == method]
        nw_m = nw_gauss[nw_gauss["method"] == method]

        kde_ise_mean = kde_m["ISE"].mean() * 1000
        kde_ise_se = kde_m["ISE"].std() / np.sqrt(len(kde_m)) * 1000
        kde_time = kde_m["time_ms"].mean()
        evals = int(kde_m["evals"].mean())

        nw_mse_mean = nw_m["MSE"].mean()
        nw_mse_se = nw_m["MSE"].std() / np.sqrt(len(nw_m))
        nw_time = nw_m["time_ms"].mean()

        print(
            f"{method:<12} {evals:>6} {kde_ise_mean:>8.2f} ({kde_ise_se:.2f}) {kde_time:>10.1f} {nw_mse_mean:>7.3f} ({nw_mse_se:.3f}) {nw_time:>10.1f}"
        )

    print("\n" + "=" * 70)
    print("TABLE 2: Scalability with Sample Size")
    print("=" * 70)
    print(
        f"{'n':>5} {'KDE Grid':>10} {'KDE Newton':>12} {'Speedup':>10} {'NW Grid':>10} {'NW Newton':>12} {'Speedup':>10}"
    )
    print("-" * 70)

    for n in [100, 200, 500]:
        kde_n = kde_df[
            (kde_df["dgp"] == "bimodal")
            & (kde_df["kernel"] == "gauss")
            & (kde_df["n"] == n)
        ]
        nw_n = nw_df[
            (nw_df["kernel"] == "gauss") & (nw_df["noise"] == 1.0) & (nw_df["n"] == n)
        ]

        kde_grid = kde_n[kde_n["method"] == "Grid"]["time_ms"].mean()
        kde_newton = kde_n[kde_n["method"] == "Newton"]["time_ms"].mean()
        nw_grid = nw_n[nw_n["method"] == "Grid"]["time_ms"].mean()
        nw_newton = nw_n[nw_n["method"] == "Newton"]["time_ms"].mean()

        print(
            f"{n:>5} {kde_grid:>10.1f} {kde_newton:>12.1f} {kde_grid/kde_newton:>9.2f}x {nw_grid:>10.1f} {nw_newton:>12.1f} {nw_grid/nw_newton:>9.2f}x"
        )

    print("\n" + "=" * 90)
    print("TABLE 3: Bootstrap (200 resamples)")
    print("=" * 90)
    print(
        f"{'n':>5} {'KDE Grid':>10} {'KDE Newton':>12} {'Speedup':>10} {'NW Grid':>10} {'NW Newton':>12} {'Speedup':>10}"
    )
    print("-" * 90)

    for n in [100, 200, 500]:
        kde_grid = boot_df[(boot_df["n"] == n) & (boot_df["method"] == "Grid")][
            "total_time_s"
        ].values[0]
        kde_newton = boot_df[(boot_df["n"] == n) & (boot_df["method"] == "Newton")][
            "total_time_s"
        ].values[0]
        if not boot_nw_df.empty:
            nw_grid = boot_nw_df[
                (boot_nw_df["n"] == n) & (boot_nw_df["method"] == "Grid")
            ]["total_time_s"].values[0]
            nw_newton = boot_nw_df[
                (boot_nw_df["n"] == n) & (boot_nw_df["method"] == "Newton")
            ]["total_time_s"].values[0]
        else:
            nw_grid = nw_newton = 0.0
        print(
            f"{n:>5} {kde_grid:>10.2f} {kde_newton:>12.2f} {kde_grid/kde_newton:>9.2f}x {nw_grid:>10.2f} {nw_newton:>12.2f} {nw_grid/nw_newton if nw_newton else 0:>9.2f}x"
        )


def main() -> None:
    """Generate all tables."""
    print("Loading data...")
    kde_df, nw_df, boot_df, mv_df, boot_nw_df, mv_nw_df = load_data()

    print("\n" + "=" * 70)
    print("LATEX TABLES")
    print("=" * 70)

    print("\n--- Table 1: Main Results ---")
    print(table1_main_results(kde_df, nw_df))

    print("\n--- Table 2: Scalability ---")
    print(table2_scalability(kde_df, nw_df))

    print("\n--- Table 3: Bootstrap ---")
    print(table3_bootstrap(boot_df, boot_nw_df))

    print("\n--- Table: Multivariate NW ---")
    print(table_mv_nw(mv_nw_df))

    print_plain_text_tables(kde_df, nw_df, boot_df, boot_nw_df)

    print(robustness_summary(kde_df, nw_df))

    print("\nDone!")


if __name__ == "__main__":
    main()
