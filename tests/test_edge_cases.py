"""Edge case tests for bandwidth selection algorithms."""

import math
import random
import sys
import os

import numpy as np
import pytest

# Add src/ to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kde_analytic_hessian import lscv_generic as kde_lscv
from nw_analytic_hessian import loocv_mse as nw_loocv
from optim import newton_armijo


class TestSmallSampleSizes:
    """Tests for small sample sizes (n=2, 3, 5)."""

    def test_kde_n2(self):
        """KDE with n=2 should return finite values."""
        x = np.array([0.0, 1.0])
        for kernel in ["gauss", "epan"]:
            score, grad, hess = kde_lscv(x, h=0.5, kernel=kernel)
            assert np.isfinite(score), f"Non-finite score for {kernel}"
            assert np.isfinite(grad), f"Non-finite gradient for {kernel}"
            assert np.isfinite(hess), f"Non-finite Hessian for {kernel}"

    def test_kde_n3(self):
        """KDE with n=3 should converge."""
        x = np.array([-1.0, 0.0, 1.0])
        for kernel in ["gauss", "epan"]:
            h_opt, evals = newton_armijo(kde_lscv, x, h0=0.5, kernel=kernel)
            assert h_opt > 0, f"Negative bandwidth for {kernel}"
            assert evals <= 12, f"Too many evaluations for {kernel}"

    def test_nw_n3(self):
        """NW with n=3 should return finite values."""
        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([0.0, 1.0, 0.0])
        for kernel in ["gauss", "epan"]:
            loss, grad, hess = nw_loocv(x, y, h=0.5, kernel=kernel)
            assert np.isfinite(loss), f"Non-finite loss for {kernel}"
            assert np.isfinite(grad), f"Non-finite gradient for {kernel}"
            assert np.isfinite(hess), f"Non-finite Hessian for {kernel}"


class TestExtremeBandwidths:
    """Tests for very small and very large bandwidth values."""

    def test_kde_tiny_bandwidth(self):
        """KDE with very small bandwidth should not crash."""
        rng = random.Random(0)
        x = np.array([rng.gauss(0, 1) for _ in range(10)])
        for kernel in ["gauss", "epan"]:
            score, grad, hess = kde_lscv(x, h=1e-4, kernel=kernel)
            assert np.isfinite(score), f"Non-finite score for {kernel} at tiny h"

    def test_kde_large_bandwidth(self):
        """KDE with very large bandwidth should return finite values."""
        rng = random.Random(0)
        x = np.array([rng.gauss(0, 1) for _ in range(10)])
        for kernel in ["gauss", "epan"]:
            score, grad, hess = kde_lscv(x, h=100.0, kernel=kernel)
            assert np.isfinite(score), f"Non-finite score for {kernel} at large h"
            assert np.isfinite(grad), f"Non-finite gradient for {kernel} at large h"

    def test_nw_tiny_bandwidth(self):
        """NW with very small bandwidth should handle zero-weight sums."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.0, 1.0])
        # With Epanechnikov and tiny h, most weights become zero
        loss, grad, hess = nw_loocv(x, y, h=0.01, kernel="epan")
        assert np.isfinite(loss), "Non-finite loss for tiny bandwidth NW"


class TestKernelValidation:
    """Tests for kernel name validation."""

    def test_kde_invalid_kernel(self):
        """KDE should raise KeyError for invalid kernel names."""
        x = np.array([0.0, 1.0, 2.0])
        with pytest.raises(KeyError):
            kde_lscv(x, h=0.5, kernel="invalid_kernel")

    def test_nw_invalid_kernel(self):
        """NW should raise ValueError for invalid kernel names."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="Unknown kernel"):
            nw_loocv(x, y, h=0.5, kernel="invalid_kernel")


class TestDifferentDistributions:
    """Tests with various data distributions."""

    def test_kde_bimodal(self):
        """KDE should converge on bimodal data."""
        rng = random.Random(456)
        x = np.array(
            [rng.gauss(-2, 0.5) for _ in range(15)] +
            [rng.gauss(2, 0.5) for _ in range(15)]
        )
        for kernel in ["gauss", "epan"]:
            h_opt, _ = newton_armijo(kde_lscv, x, h0=1.0, kernel=kernel)
            assert 0.1 < h_opt < 5.0, f"Unreasonable bandwidth for bimodal data: {h_opt}"

    def test_kde_uniform(self):
        """KDE should handle uniformly distributed data."""
        x = np.linspace(-2, 2, 20)
        for kernel in ["gauss", "epan"]:
            h_opt, _ = newton_armijo(kde_lscv, x, h0=0.5, kernel=kernel)
            assert h_opt > 0, f"Non-positive bandwidth for uniform data"

    def test_nw_linear_function(self):
        """NW should work with linear response."""
        x = np.linspace(-1, 1, 25)
        y = 2 * x + 1
        for kernel in ["gauss", "epan"]:
            loss, _, _ = nw_loocv(x, y, h=0.3, kernel=kernel)
            # Linear function should have low LOOCV error with reasonable bandwidth
            assert loss < 1.0, f"High loss for linear function with {kernel}"


class TestConvergence:
    """Tests for optimization convergence properties."""

    def test_newton_reduces_score_from_different_starts(self):
        """Newton should reduce the score regardless of starting point."""
        rng = random.Random(789)
        x = np.array([rng.gauss(0, 1) for _ in range(30)])

        for kernel in ["gauss", "epan"]:
            for h0 in [0.2, 0.5, 1.0, 2.0]:
                score_init, _, _ = kde_lscv(x, h0, kernel)
                h_opt, _ = newton_armijo(kde_lscv, x, h0=h0, kernel=kernel)
                score_opt, _, _ = kde_lscv(x, h_opt, kernel)

                # Optimization should never increase score
                assert score_opt <= score_init + 1e-10, (
                    f"Score increased for {kernel} from h0={h0}: {score_init} -> {score_opt}"
                )

    def test_optimization_reduces_score(self):
        """Optimization should reduce the objective score."""
        rng = random.Random(101)
        x = np.array([rng.gauss(0, 1) for _ in range(25)])
        h0 = 2.0

        for kernel in ["gauss", "epan"]:
            score_init, _, _ = kde_lscv(x, h0, kernel)
            h_opt, _ = newton_armijo(kde_lscv, x, h0=h0, kernel=kernel)
            score_opt, _, _ = kde_lscv(x, h_opt, kernel)

            assert score_opt <= score_init, (
                f"Optimization increased score for {kernel}: {score_init} -> {score_opt}"
            )
