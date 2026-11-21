"""Independent leave-one-out checks for the uniform regression selector."""

import math

import numpy as np
import pytest

from hbw import nw_bandwidth


def direct_loss(x, y, h):
    residuals = []
    for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
        other = (np.abs(x - xi) <= h) & (np.arange(len(x)) != i)
        residual = (
            math.fsum(yi - value for value in y[other]) / other.sum()
            if other.any()
            else yi
        )
        residuals.append(residual**2)
    return math.fsum(residuals) / len(x)


def test_uniform_nw_finds_zero_error_plateau():
    x = np.array([0, 0.125, 1, 1.125, 10, 10.125, 11, 11.125])
    y = np.tile([-1.0, -1.0, 1.0, 1.0], 2)
    for scale in (1e-6, 1.0, 1e5):
        h = nw_bandwidth(scale * x, y, kernel="unif", h0=20 * scale)
        assert direct_loss(scale * x, y, h) == 0


@pytest.mark.parametrize("offset", [0, 2**47])
def test_uniform_nw_matches_all_attainable_losses(offset):
    rng = np.random.default_rng(321)
    for _ in range(20):
        x = rng.integers(-12, 12, size=13).astype(float)
        y = rng.integers(-40, 40, size=len(x)).astype(float) + offset
        distances = np.unique(np.abs(x[:, None] - x))
        expected = min(direct_loss(x, y, h) for h in np.r_[0.25, distances[1:]])
        order = rng.permutation(len(x))
        h = nw_bandwidth(x[order], y[order], kernel="unif")
        assert direct_loss(x, y, h) == pytest.approx(expected, rel=1e-12)


def test_uniform_nw_large_sample_includes_start_and_grid():
    rng = np.random.default_rng(108)
    x = rng.uniform(-3, 3, size=140)
    y = np.sin(x) + 0.15 * rng.normal(size=len(x))
    start = 0.02
    h = nw_bandwidth(x, y, kernel="unif", h0=start)
    assert direct_loss(x, y, h) <= direct_loss(x, y, start)
    std = np.std(x, ddof=1)
    candidates = [0.3 * std, std, 2.5 * std]
    assert direct_loss(x, y, h) <= min(direct_loss(x, y, c) for c in candidates)


def test_uniform_nw_handles_identical_predictors_and_invalid_start():
    x = np.ones(8)
    y = np.arange(8, dtype=float)
    assert nw_bandwidth(x, y, kernel="unif", h0=3.0) == 3.0
    assert nw_bandwidth(x, y, kernel="unif") > 0.0
    for h in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="bandwidth"):
            nw_bandwidth(x, y, kernel="unif", h0=h)


def test_adjacent_distance_breaks_can_select_lower_endpoint():
    low = np.nextafter(1.0, 2.0)
    high = np.nextafter(low, 2.0)
    x = np.array([0, low, high])
    y = np.array([-2.0, -2.0, -1.0])
    assert low + (high - low) / 2 == high
    assert nw_bandwidth(x, y, kernel="unif", h0=2.0) == low


@pytest.mark.parametrize("n", [8, 80])
@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_finite_positive_bandwidth_across_predictor_scales(n, scale):
    x = np.resize([0, 0.125, 1, 1.125, 10, 10.125, 11, 11.125], n)
    y = np.resize([-1.0, -1.0, 1.0, 1.0], n)
    h = nw_bandwidth(scale * x, y, kernel="unif")
    assert np.isfinite(h)
    assert h > 0
    assert direct_loss(scale * x, y, h) == 0


def test_isolated_points_use_their_squared_responses():
    x = np.array([0.0, 1.0])
    y = np.array([3.0, -3.0])
    h = nw_bandwidth(x, y, kernel="unif")
    assert 0 < h < 1
    assert direct_loss(x, y, h) == 9.0


def test_signed_cancellation_and_subsampling():
    rng = np.random.default_rng(832)
    x = np.arange(90, dtype=float)
    y = np.resize([2.0**50, 9.0, -(2.0**50), -13.0, 4.0], len(x))
    idx = rng.choice(len(x), size=32, replace=False)
    h = nw_bandwidth(x, y, kernel="unif", max_n=32, seed=832)
    spans = np.unique(np.abs(x[idx, None] - x[idx]))
    optimum = min(direct_loss(x[idx], y[idx], c) for c in np.r_[0.25, spans[1:]])
    assert direct_loss(x[idx], y[idx], h) == pytest.approx(optimum, rel=1e-13)


@pytest.mark.parametrize("small", [1.0, 1e-150])
def test_repeated_huge_responses_do_not_hide_zero_error(small):
    x = np.array([100.0, 100.0, 0.0, 0.25, 2.0, 2.25])
    y = np.array([1e200, 1e200, small, small, 2 * small, 2 * small])
    h = nw_bandwidth(x, y, kernel="unif", h0=0.01)
    assert 0.25 <= h < 1.75


@pytest.mark.parametrize("small", [1.0, 1e-150])
def test_large_sample_extreme_response_grid_retains_better_start(small):
    x = np.r_[np.repeat(100.0, 64), [0.0, 0.25, 2.0, 2.25]]
    y = np.r_[np.repeat(1e200, 64), [small, small, 2 * small, 2 * small]]
    h = nw_bandwidth(x, y, kernel="unif", h0=0.5)
    assert 0.25 <= h < 1.75
