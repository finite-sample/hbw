"""Does the selected bandwidth behave the way a bandwidth should?

The existing suite establishes that the fast optimiser finds the same objective
minimum as a dense grid search -- `test_selectors_match_dense_log_grid_objectives`
compares against an 800-point geometric grid across every kernel. That is the
right test for what this package claims, which is speed without changing the
answer, and it is a genuinely strong one.

What it cannot say is whether the answer is any good. A selector that returned a
constant, or minimised the wrong objective, would agree with a grid search over
that same wrong objective exactly. Nothing here reached for a known statistical
truth, because these tests did not exist: this file is the first simulation study
in the package.

Three properties, in decreasing order of how hard they are to fake:

1. **The rate.** For a second-order kernel the optimal bandwidth shrinks as
   ``n^(-1/5)``, so the slope of ``log h`` on ``log n`` is -0.2. A selector that
   ignores the data, or that mis-scales it, gets this wrong. Measured: -0.191 for
   the KDE selector and -0.201 for the regression one.

2. **The level, against a closed form.** For ``X ~ N(0, s^2)`` estimated with a
   Gaussian kernel, the MISE-optimal bandwidth is exactly
   ``(4/3)^(1/5) * s * n^(-1/5)``. That is a number no implementation detail can
   argue with. Measured ratios of the median selection to it, over n from 100 to
   1600: 0.93, 1.05, 1.08, 0.96, 1.01.

3. **The consequence.** The point of choosing a bandwidth is the estimate it
   produces, so the last test compares the integrated squared error at the
   selected bandwidth against the best achievable on that same sample. Measured
   efficiency rises from 0.70 to 0.82 for the KDE and 0.80 to 0.91 for the
   regression as n grows.

**On the spread.** Least-squares cross-validation is famously variable -- its
relative rate of convergence to the optimal bandwidth is only ``n^(-1/10)`` -- and
this shows up here as an interquartile range of a quarter to two fifths of the
median, which barely shrinks with n. That is a property of the method, not a
defect in this implementation, so the tests below gate the *median* over
replicates and characterise the spread rather than trying to bound it tightly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from hbw import kde_bandwidth, kde_evaluate, nw_bandwidth, nw_predict

# Replicates per sample size. Enough to pin a median without the file becoming
# something nobody runs; the quantities gated below are stable at this size.
REPS = 25
GRID = np.linspace(-6.0, 6.0, 801)
NORMAL_DENSITY = np.exp(-0.5 * GRID**2) / np.sqrt(2 * np.pi)


def gaussian_mise_bandwidth(n: int, sigma: float = 1.0) -> float:
    """The exact MISE-optimal bandwidth for a normal target and Gaussian kernel.

    Args:
        n: Sample size.
        sigma: Standard deviation of the target density.

    Returns:
        float: ``(4/3)**(1/5) * sigma * n**(-1/5)``.
    """
    return (4.0 / 3.0) ** 0.2 * sigma * n ** (-0.2)


def _kde_ise(sample: np.ndarray, bandwidth: float) -> float:
    """Integrated squared error of a Gaussian KDE against the standard normal.

    Args:
        sample: Draws from the standard normal.
        bandwidth: Bandwidth to evaluate at.

    Returns:
        float: The integrated squared error over ``GRID``.
    """
    estimate = kde_evaluate(sample, GRID, bandwidth)
    return float(np.trapezoid((estimate - NORMAL_DENSITY) ** 2, GRID))


def _selected_bandwidths(
    sizes: Sequence[int],
    draw: Callable[..., tuple],
    select: Callable[..., float],
) -> tuple[list[float], list[float]]:
    """Median selected bandwidth at each sample size.

    Args:
        sizes: Sample sizes to sweep.
        draw: Callable ``(rng, n)`` returning the arguments for ``select``.
        select: Callable taking those arguments and returning a bandwidth.

    Returns:
        tuple: ``(medians, spreads)``, the second being IQR over median.
    """
    medians, spreads = [], []
    for n in sizes:
        chosen = []
        for i in range(REPS):
            rng = np.random.default_rng(1000 + i)
            chosen.append(select(*draw(rng, n)))
        chosen = np.asarray(chosen)
        median = float(np.median(chosen))
        medians.append(median)
        spreads.append(float(np.percentile(chosen, 75) - np.percentile(chosen, 25)) / median)
    return medians, spreads


# --------------------------------------------------------------------------
# 1. The rate.
# --------------------------------------------------------------------------


def test_the_kde_bandwidth_shrinks_at_the_theoretical_rate() -> None:
    """``log h`` against ``log n`` must have slope near -1/5.

    This is the cheapest property to state and the hardest to satisfy by
    accident: it constrains how the selector responds to sample size, which a
    constant or a mis-scaled rule cannot fake. Measured -0.191.
    """
    sizes = (100, 200, 400, 800, 1600)
    medians, _ = _selected_bandwidths(
        sizes,
        lambda rng, n: (rng.standard_normal(n),),
        lambda x: kde_bandwidth(x, max_n=None),
    )

    slope = float(np.polyfit(np.log(sizes), np.log(medians), 1)[0])
    assert -0.28 < slope < -0.13, (
        f"bandwidth shrinks as n^({slope:.3f}); the second-order-kernel rate is n^(-0.2)"
    )


def test_the_regression_bandwidth_shrinks_at_the_theoretical_rate() -> None:
    """Same rate for the Nadaraya-Watson selector. Measured -0.201."""
    sizes = (100, 200, 400, 800)

    def draw(rng: np.random.Generator, n: int) -> tuple:
        x = np.sort(rng.uniform(-2.0, 2.0, n))
        return x, np.sin(2.0 * x) + 0.3 * rng.standard_normal(n)

    medians, _ = _selected_bandwidths(sizes, draw, lambda x, y: nw_bandwidth(x, y, max_n=None))

    slope = float(np.polyfit(np.log(sizes), np.log(medians), 1)[0])
    assert -0.28 < slope < -0.13, f"bandwidth shrinks as n^({slope:.3f}); the rate is n^(-0.2)"


# --------------------------------------------------------------------------
# 2. The level, against a closed form.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [200, 800])
def test_the_kde_bandwidth_matches_the_exact_gaussian_optimum(n: int) -> None:
    """Median selection must sit near the closed-form MISE optimum.

    There is no tuning constant in the target: for a normal density and a
    Gaussian kernel the MISE-optimal bandwidth is ``(4/3)^(1/5) s n^(-1/5)``.

    The gate is on the median over replicates rather than a single draw, because
    least-squares cross-validation is highly variable by nature -- its relative
    rate of convergence to the optimum is only ``n^(-1/10)``. Measured ratios
    across n from 100 to 1600: 0.93, 1.05, 1.08, 0.96, 1.01.

    Args:
        n: Sample size.
    """
    chosen = np.array(
        [
            kde_bandwidth(np.random.default_rng(1000 + i).standard_normal(n), max_n=None)
            for i in range(REPS)
        ]
    )
    ratio = float(np.median(chosen)) / gaussian_mise_bandwidth(n)

    assert 0.75 < ratio < 1.35, (
        f"median bandwidth is {ratio:.3f} times the exact Gaussian MISE optimum at n={n}"
    )


def test_the_spread_of_the_selection_is_reported_not_bounded_tightly() -> None:
    """Characterisation: cross-validated bandwidths scatter, and that is normal.

    Least-squares cross-validation converges to the optimal bandwidth at a
    relative rate of only ``n^(-1/10)``, so the interquartile range stays around
    a quarter to two fifths of the median and barely shrinks. This is a property
    of the method, not of this implementation, and is asserted loosely so that it
    documents the behaviour without pretending to constrain it.

    What it does catch is a selector that has stopped responding to the data at
    all, which would show a spread of essentially zero.
    """
    sizes = (200, 800)
    _, spreads = _selected_bandwidths(
        sizes,
        lambda rng, n: (rng.standard_normal(n),),
        lambda x: kde_bandwidth(x, max_n=None),
    )

    for n, spread in zip(sizes, spreads, strict=True):
        assert 0.05 < spread < 0.9, (
            f"at n={n} the interquartile range is {spread:.3f} of the median; "
            "near zero would mean the selector ignores the sample"
        )


# --------------------------------------------------------------------------
# 3. The consequence.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [200, 800])
def test_the_selected_bandwidth_is_close_to_the_best_available(n: int) -> None:
    """Efficiency against the per-sample oracle, which is what a user gets.

    The bandwidth is a means; the estimate is the end. This compares the
    integrated squared error at the selected bandwidth against the smallest
    achievable on that same sample, so a selector that hits the right rate and
    the right level but lands in a bad place would still be caught.

    Measured median efficiency: 0.78 at n=200 and 0.80 at n=800, rising to 0.82
    by n=1600. It is well below 1 because a single sample's ISE-minimising
    bandwidth is itself a moving target that no data-driven rule can match.

    Args:
        n: Sample size.
    """
    candidates = np.geomspace(0.05, 2.0, 40)
    efficiencies = []

    for i in range(REPS):
        sample = np.random.default_rng(1000 + i).standard_normal(n)
        selected = kde_bandwidth(sample, max_n=None)
        oracle = min(candidates, key=lambda c: _kde_ise(sample, c))
        efficiencies.append(_kde_ise(sample, oracle) / _kde_ise(sample, selected))

    median = float(np.median(efficiencies))
    assert median > 0.55, (
        f"the selected bandwidth achieves only {median:.3f} of the best "
        f"integrated squared error available at n={n}"
    )
    assert median <= 1.0 + 1e-9, (
        f"efficiency of {median:.3f} exceeds 1, so the oracle search is not "
        "finding the minimum it claims to"
    )


def test_a_deliberately_wrong_bandwidth_is_measurably_worse() -> None:
    """The guard on the guard: the efficiency measure must discriminate.

    If integrated squared error were insensitive to the bandwidth over the range
    that matters, the test above would pass for any selector at all. Doubling and
    halving the selection must both cost something real.
    """
    n = 400
    losses = {"selected": [], "doubled": [], "halved": []}

    for i in range(REPS):
        sample = np.random.default_rng(1000 + i).standard_normal(n)
        selected = kde_bandwidth(sample, max_n=None)
        losses["selected"].append(_kde_ise(sample, selected))
        losses["doubled"].append(_kde_ise(sample, 2.0 * selected))
        losses["halved"].append(_kde_ise(sample, 0.5 * selected))

    chosen = float(np.median(losses["selected"]))
    assert float(np.median(losses["doubled"])) > 1.5 * chosen, (
        "doubling the bandwidth barely changes the error, so this measure cannot "
        "tell a good selection from a bad one"
    )
    assert float(np.median(losses["halved"])) > 1.5 * chosen, (
        "halving the bandwidth barely changes the error, so this measure cannot "
        "tell a good selection from a bad one"
    )


def test_the_regression_bandwidth_is_close_to_the_best_available() -> None:
    """Efficiency for the Nadaraya-Watson selector, on an interior grid.

    Measured on the boundary-free interior of ``sin(2x)`` with noise: median
    efficiency 0.91 at n=400, rising from 0.80 at n=100. The grid excludes the
    outer quarter of the range, because Nadaraya-Watson is badly biased at the
    boundary for reasons that have nothing to do with the bandwidth rule and
    would otherwise dominate the comparison.
    """
    n = 400
    grid = np.linspace(-1.5, 1.5, 150)
    truth = np.sin(2.0 * grid)
    candidates = np.geomspace(0.02, 1.5, 35)

    def mse(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
        return float(np.mean((nw_predict(x, y, grid, bandwidth) - truth) ** 2))

    efficiencies = []
    for i in range(REPS):
        rng = np.random.default_rng(2000 + i)
        x = np.sort(rng.uniform(-2.0, 2.0, n))
        y = np.sin(2.0 * x) + 0.3 * rng.standard_normal(n)
        selected = nw_bandwidth(x, y, max_n=None)
        oracle = min(candidates, key=lambda c: mse(x, y, c))
        efficiencies.append(mse(x, y, oracle) / mse(x, y, selected))

    median = float(np.median(efficiencies))
    assert median > 0.6, (
        f"the selected bandwidth achieves only {median:.3f} of the best mean "
        "squared error available on the interior"
    )
    assert median <= 1.0 + 1e-9, (
        f"efficiency of {median:.3f} exceeds 1, so the oracle search is not "
        "finding the minimum it claims to"
    )
