"""Test that analytic gradients match finite-difference approximations."""

import math
import random

import numpy as np

from hbw import lscv


def finite_diff(f, h: float, eps: float = 1e-5) -> tuple[float, float]:
    """Compute finite-difference gradient and Hessian."""
    f_plus = f(h + eps)
    f_minus = f(h - eps)
    grad = (f_plus - f_minus) / (2 * eps)
    hess = (f_plus - 2 * f(h) + f_minus) / (eps**2)
    return grad, hess


def test_lscv_derivatives_against_finite_diff() -> None:
    """Verify LSCV analytic gradient matches finite-difference."""
    rng = random.Random(0)
    x = np.array([rng.gauss(0, 1) for _ in range(15)])
    for kernel in ["gauss", "epan"]:
        for h in [0.5, 1.0, 1.5]:
            _, grad, _ = lscv(x, h, kernel)
            k = kernel
            num_grad, _ = finite_diff(lambda hh, k=k: lscv(x, hh, k)[0], h)
            assert math.isclose(grad, num_grad, rel_tol=1e-4, abs_tol=1e-5)
