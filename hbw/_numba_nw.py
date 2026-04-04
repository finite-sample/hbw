"""Numba-accelerated NW regression bandwidth selection via fused LOOCV computation."""

import math

import numba
import numpy as np
from numpy.typing import NDArray

_SQRT_2PI = math.sqrt(2 * math.pi)
_PI = math.pi
_PI_4 = _PI / 4
_PI_2 = _PI / 2


@numba.njit(fastmath=True, parallel=True)
def loocv_numba_gauss(x: NDArray, y: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LOOCV-MSE score, gradient, Hessian for Gaussian kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    h2 = h * h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)
    local_grad = np.zeros(n_threads)
    local_hess = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0
        num_p = 0.0
        den_p = 0.0
        num_pp = 0.0
        den_pp = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            u2 = u * u

            exp_val = math.exp(-0.5 * u2)
            w = exp_val / (_SQRT_2PI * h)
            wp = w * (u2 - 1.0) / h
            wpp = w * (u2 * u2 - 3.0 * u2 + 1.0) / h2

            yi = y[i]
            num += w * yi
            den += w
            num_p += wp * yi
            den_p += wp
            num_pp += wpp * yi
            den_pp += wpp

        if den > 0:
            m = num / den
            den2 = den * den

            mp = (num_p * den - num * den_p) / den2

            mpp_num = (num_pp * den - num * den_pp) * den - 2.0 * (
                num_p * den - num * den_p
            ) * den_p
            mpp = mpp_num / (den2 * den)

            resid = y[j] - m
            local_loss[tid] += resid * resid
            local_grad[tid] += -2.0 * resid * mp
            local_hess[tid] += 2.0 * (mp * mp - resid * mpp)

    inv_n = 1.0 / n
    return local_loss.sum() * inv_n, local_grad.sum() * inv_n, local_hess.sum() * inv_n


@numba.njit(fastmath=True, parallel=True)
def loocv_score_numba_gauss(x: NDArray, y: NDArray, h: float) -> float:
    """Compute LOOCV-MSE score only (for Armijo backtracking) - Gaussian kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            exp_val = math.exp(-0.5 * u * u)
            w = exp_val / (_SQRT_2PI * h)

            num += w * y[i]
            den += w

        if den > 0:
            m = num / den
            resid = y[j] - m
            local_loss[tid] += resid * resid

    return local_loss.sum() / n


@numba.njit(fastmath=True, parallel=True)
def loocv_numba_epan(x: NDArray, y: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LOOCV-MSE score, gradient, Hessian for Epanechnikov kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    h2 = h * h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)
    local_grad = np.zeros(n_threads)
    local_hess = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0
        num_p = 0.0
        den_p = 0.0
        num_pp = 0.0
        den_pp = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            absu = abs(u)

            if absu <= 1.0:
                u2 = u * u
                k = 0.75 * (1.0 - u2)
                kp = -1.5 * u
                kpp = -1.5

                w = k / h
                wp = -(u * kp + k) / h2
                wpp = (u2 * kpp + 4.0 * u * kp + 2.0 * k) / (h2 * h)

                yi = y[i]
                num += w * yi
                den += w
                num_p += wp * yi
                den_p += wp
                num_pp += wpp * yi
                den_pp += wpp

        if den > 0:
            m = num / den
            den2 = den * den

            mp = (num_p * den - num * den_p) / den2

            mpp_num = (num_pp * den - num * den_pp) * den - 2.0 * (
                num_p * den - num * den_p
            ) * den_p
            mpp = mpp_num / (den2 * den)

            resid = y[j] - m
            local_loss[tid] += resid * resid
            local_grad[tid] += -2.0 * resid * mp
            local_hess[tid] += 2.0 * (mp * mp - resid * mpp)

    inv_n = 1.0 / n
    return local_loss.sum() * inv_n, local_grad.sum() * inv_n, local_hess.sum() * inv_n


@numba.njit(fastmath=True, parallel=True)
def loocv_score_numba_epan(x: NDArray, y: NDArray, h: float) -> float:
    """Compute LOOCV-MSE score only (for Armijo backtracking) - Epanechnikov kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            if abs(u) <= 1.0:
                k = 0.75 * (1.0 - u * u)
                w = k / h
                num += w * y[i]
                den += w

        if den > 0:
            m = num / den
            resid = y[j] - m
            local_loss[tid] += resid * resid

    return local_loss.sum() / n


@numba.njit(fastmath=True, parallel=True)
def loocv_numba_unif(x: NDArray, y: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LOOCV-MSE score, gradient, Hessian for uniform kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    h2 = h * h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)
    local_grad = np.zeros(n_threads)
    local_hess = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0
        num_p = 0.0
        den_p = 0.0
        num_pp = 0.0
        den_pp = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            absu = abs(u)

            if absu <= 1.0:
                k = 0.5
                w = k / h
                wp = -k / h2
                wpp = 2.0 * k / (h2 * h)

                yi = y[i]
                num += w * yi
                den += w
                num_p += wp * yi
                den_p += wp
                num_pp += wpp * yi
                den_pp += wpp

        if den > 0:
            m = num / den
            den2 = den * den

            mp = (num_p * den - num * den_p) / den2

            mpp_num = (num_pp * den - num * den_pp) * den - 2.0 * (
                num_p * den - num * den_p
            ) * den_p
            mpp = mpp_num / (den2 * den)

            resid = y[j] - m
            local_loss[tid] += resid * resid
            local_grad[tid] += -2.0 * resid * mp
            local_hess[tid] += 2.0 * (mp * mp - resid * mpp)

    inv_n = 1.0 / n
    return local_loss.sum() * inv_n, local_grad.sum() * inv_n, local_hess.sum() * inv_n


@numba.njit(fastmath=True, parallel=True)
def loocv_score_numba_unif(x: NDArray, y: NDArray, h: float) -> float:
    """Compute LOOCV-MSE score only (for Armijo backtracking) - uniform kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            if abs(u) <= 1.0:
                w = 0.5 / h
                num += w * y[i]
                den += w

        if den > 0:
            m = num / den
            resid = y[j] - m
            local_loss[tid] += resid * resid

    return local_loss.sum() / n


@numba.njit(fastmath=True, parallel=True)
def loocv_numba_biweight(x: NDArray, y: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LOOCV-MSE score, gradient, Hessian for biweight kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    h2 = h * h
    h3 = h2 * h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)
    local_grad = np.zeros(n_threads)
    local_hess = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0
        num_p = 0.0
        den_p = 0.0
        num_pp = 0.0
        den_pp = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            absu = abs(u)

            if absu <= 1.0:
                u2 = u * u
                one_minus_u2 = 1.0 - u2

                k = (15.0 / 16.0) * one_minus_u2 * one_minus_u2
                w = k / h

                wp = (15.0 / 16.0) * one_minus_u2 * (5.0 * u2 - 1.0) / h2

                wpp = (15.0 / 8.0) * (1.0 - 12.0 * u2 + 20.0 * u2 * u2 - 5.0 * u2 * u2 * u2) / h3

                yi = y[i]
                num += w * yi
                den += w
                num_p += wp * yi
                den_p += wp
                num_pp += wpp * yi
                den_pp += wpp

        if den > 0:
            m = num / den
            den2 = den * den

            mp = (num_p * den - num * den_p) / den2

            mpp_num = (num_pp * den - num * den_pp) * den - 2.0 * (
                num_p * den - num * den_p
            ) * den_p
            mpp = mpp_num / (den2 * den)

            resid = y[j] - m
            local_loss[tid] += resid * resid
            local_grad[tid] += -2.0 * resid * mp
            local_hess[tid] += 2.0 * (mp * mp - resid * mpp)

    inv_n = 1.0 / n
    return local_loss.sum() * inv_n, local_grad.sum() * inv_n, local_hess.sum() * inv_n


@numba.njit(fastmath=True, parallel=True)
def loocv_score_numba_biweight(x: NDArray, y: NDArray, h: float) -> float:
    """Compute LOOCV-MSE score only (for Armijo backtracking) - biweight kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            if abs(u) <= 1.0:
                u2 = u * u
                one_minus_u2 = 1.0 - u2
                k = (15.0 / 16.0) * one_minus_u2 * one_minus_u2
                w = k / h
                num += w * y[i]
                den += w

        if den > 0:
            m = num / den
            resid = y[j] - m
            local_loss[tid] += resid * resid

    return local_loss.sum() / n


@numba.njit(fastmath=True, parallel=True)
def loocv_numba_triweight(x: NDArray, y: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LOOCV-MSE score, gradient, Hessian for triweight kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    h2 = h * h
    h3 = h2 * h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)
    local_grad = np.zeros(n_threads)
    local_hess = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0
        num_p = 0.0
        den_p = 0.0
        num_pp = 0.0
        den_pp = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            absu = abs(u)

            if absu <= 1.0:
                u2 = u * u
                one_minus_u2 = 1.0 - u2
                one_minus_u2_sq = one_minus_u2 * one_minus_u2

                k = (35.0 / 32.0) * one_minus_u2_sq * one_minus_u2
                w = k / h

                wp = (35.0 / 32.0) * one_minus_u2_sq * (7.0 * u2 - 1.0) / h2

                wpp = (
                    (35.0 / 16.0)
                    * one_minus_u2
                    * (1.0 - 20.0 * u2 + 35.0 * u2 * u2)
                    / h3
                )

                yi = y[i]
                num += w * yi
                den += w
                num_p += wp * yi
                den_p += wp
                num_pp += wpp * yi
                den_pp += wpp

        if den > 0:
            m = num / den
            den2 = den * den

            mp = (num_p * den - num * den_p) / den2

            mpp_num = (num_pp * den - num * den_pp) * den - 2.0 * (
                num_p * den - num * den_p
            ) * den_p
            mpp = mpp_num / (den2 * den)

            resid = y[j] - m
            local_loss[tid] += resid * resid
            local_grad[tid] += -2.0 * resid * mp
            local_hess[tid] += 2.0 * (mp * mp - resid * mpp)

    inv_n = 1.0 / n
    return local_loss.sum() * inv_n, local_grad.sum() * inv_n, local_hess.sum() * inv_n


@numba.njit(fastmath=True, parallel=True)
def loocv_score_numba_triweight(x: NDArray, y: NDArray, h: float) -> float:
    """Compute LOOCV-MSE score only (for Armijo backtracking) - triweight kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            if abs(u) <= 1.0:
                u2 = u * u
                one_minus_u2 = 1.0 - u2
                k = (35.0 / 32.0) * one_minus_u2 * one_minus_u2 * one_minus_u2
                w = k / h
                num += w * y[i]
                den += w

        if den > 0:
            m = num / den
            resid = y[j] - m
            local_loss[tid] += resid * resid

    return local_loss.sum() / n


@numba.njit(fastmath=True, parallel=True)
def loocv_numba_cosine(x: NDArray, y: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LOOCV-MSE score, gradient, Hessian for cosine kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    h2 = h * h
    h3 = h2 * h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)
    local_grad = np.zeros(n_threads)
    local_hess = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0
        num_p = 0.0
        den_p = 0.0
        num_pp = 0.0
        den_pp = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            absu = abs(u)

            if absu <= 1.0:
                u2 = u * u
                cos_val = math.cos(_PI_2 * u)
                sin_val = math.sin(_PI_2 * u)

                k = _PI_4 * cos_val
                w = k / h

                wp = (
                    _PI_4
                    * ((_PI * _PI * u2 / 4.0 - 1.0) * cos_val - (_PI_2 * u) * sin_val)
                    / h2
                )

                wpp = (
                    _PI_4
                    * (
                        (2.0 - 3.0 * _PI * _PI * u2 / 2.0 + _PI**4 * u2 * u2 / 16.0)
                        * cos_val
                        + (3.0 * _PI_2 * u - _PI**3 * u2 * u / 8.0) * sin_val
                    )
                    / h3
                )

                yi = y[i]
                num += w * yi
                den += w
                num_p += wp * yi
                den_p += wp
                num_pp += wpp * yi
                den_pp += wpp

        if den > 0:
            m = num / den
            den2 = den * den

            mp = (num_p * den - num * den_p) / den2

            mpp_num = (num_pp * den - num * den_pp) * den - 2.0 * (
                num_p * den - num * den_p
            ) * den_p
            mpp = mpp_num / (den2 * den)

            resid = y[j] - m
            local_loss[tid] += resid * resid
            local_grad[tid] += -2.0 * resid * mp
            local_hess[tid] += 2.0 * (mp * mp - resid * mpp)

    inv_n = 1.0 / n
    return local_loss.sum() * inv_n, local_grad.sum() * inv_n, local_hess.sum() * inv_n


@numba.njit(fastmath=True, parallel=True)
def loocv_score_numba_cosine(x: NDArray, y: NDArray, h: float) -> float:
    """Compute LOOCV-MSE score only (for Armijo backtracking) - cosine kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0

        for i in range(n):
            if i == j:
                continue

            u = (x[j] - x[i]) * inv_h
            if abs(u) <= 1.0:
                k = _PI_4 * math.cos(_PI_2 * u)
                w = k / h
                num += w * y[i]
                den += w

        if den > 0:
            m = num / den
            resid = y[j] - m
            local_loss[tid] += resid * resid

    return local_loss.sum() / n


@numba.njit(fastmath=True, parallel=True)
def loocv_mv_numba_gauss(data: NDArray, y: NDArray, h: float) -> tuple[float, float, float]:
    """Compute multivariate LOOCV-MSE for Gaussian product kernel using fused loop."""
    n, d = data.shape
    inv_h = 1.0 / h
    h_d = h**d
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)
    local_grad = np.zeros(n_threads)
    local_hess = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0
        num_p = 0.0
        den_p = 0.0
        num_pp = 0.0
        den_pp = 0.0

        for i in range(n):
            if i == j:
                continue

            prod_k = 1.0
            sum_ratio = 0.0
            sum_d2 = 0.0

            for dim in range(d):
                u = (data[j, dim] - data[i, dim]) * inv_h
                u2 = u * u
                k = math.exp(-0.5 * u2) / _SQRT_2PI
                prod_k *= k

                if k > 0:
                    r = -u
                    sum_ratio += u * r
                    sum_d2 += 2.0 * u * r + u2 * (u2 - 1.0 - r * r)

            w = prod_k / h_d
            wp = -(w / h) * (d + sum_ratio)
            wpp = (w / (h * h)) * ((d + 1) * d + 2.0 * (d + 1) * sum_ratio + sum_d2)

            yi = y[i]
            num += w * yi
            den += w
            num_p += wp * yi
            den_p += wp
            num_pp += wpp * yi
            den_pp += wpp

        if den > 0:
            m = num / den
            den2 = den * den

            mp = (num_p * den - num * den_p) / den2

            mpp_num = (num_pp * den - num * den_pp) * den - 2.0 * (
                num_p * den - num * den_p
            ) * den_p
            mpp = mpp_num / (den2 * den)

            resid = y[j] - m
            local_loss[tid] += resid * resid
            local_grad[tid] += -2.0 * resid * mp
            local_hess[tid] += 2.0 * (mp * mp - resid * mpp)

    inv_n = 1.0 / n
    return local_loss.sum() * inv_n, local_grad.sum() * inv_n, local_hess.sum() * inv_n


@numba.njit(fastmath=True, parallel=True)
def loocv_score_mv_numba_gauss(data: NDArray, y: NDArray, h: float) -> float:
    """Compute multivariate LOOCV-MSE score only (for Armijo backtracking) - Gaussian kernel."""
    n, d = data.shape
    inv_h = 1.0 / h
    h_d = h**d
    n_threads = numba.get_num_threads()

    local_loss = np.zeros(n_threads)

    for j in numba.prange(n):
        tid = numba.get_thread_id()
        num = 0.0
        den = 0.0

        for i in range(n):
            if i == j:
                continue

            prod_k = 1.0
            for dim in range(d):
                u = (data[j, dim] - data[i, dim]) * inv_h
                k = math.exp(-0.5 * u * u) / _SQRT_2PI
                prod_k *= k

            w = prod_k / h_d
            num += w * y[i]
            den += w

        if den > 0:
            m = num / den
            resid = y[j] - m
            local_loss[tid] += resid * resid

    return local_loss.sum() / n


def warmup() -> None:
    """Trigger JIT compilation with small arrays."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.5])
    X_mv = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    loocv_numba_gauss(x, y, 1.0)
    loocv_score_numba_gauss(x, y, 1.0)
    loocv_numba_epan(x, y, 1.0)
    loocv_score_numba_epan(x, y, 1.0)
    loocv_numba_unif(x, y, 1.0)
    loocv_score_numba_unif(x, y, 1.0)
    loocv_numba_biweight(x, y, 1.0)
    loocv_score_numba_biweight(x, y, 1.0)
    loocv_numba_triweight(x, y, 1.0)
    loocv_score_numba_triweight(x, y, 1.0)
    loocv_numba_cosine(x, y, 1.0)
    loocv_score_numba_cosine(x, y, 1.0)
    loocv_mv_numba_gauss(X_mv, y, 1.0)
    loocv_score_mv_numba_gauss(X_mv, y, 1.0)
