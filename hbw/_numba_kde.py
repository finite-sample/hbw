"""Numba-accelerated KDE bandwidth selection via fused LSCV computation."""

import math

import numba
import numpy as np
from numpy.typing import NDArray

_SQRT_2PI = math.sqrt(2 * math.pi)
_SQRT_4PI = math.sqrt(4 * math.pi)
_PI = math.pi
_PI_4 = _PI / 4
_PI_2 = _PI / 2


@numba.njit(fastmath=True)
def _gauss_all(u: float) -> tuple[float, float, float, float, float, float]:
    """Compute Gaussian kernel, derivatives, convolution and its derivatives."""
    exp_half = math.exp(-0.5 * u * u)
    k = exp_half / _SQRT_2PI
    kp = -u * k
    kpp = (u * u - 1.0) * k

    exp_quarter = math.exp(-0.25 * u * u)
    k2 = exp_quarter / _SQRT_4PI
    k2p = -0.5 * u * k2
    k2pp = (0.25 * u * u - 0.5) * k2

    return k, kp, kpp, k2, k2p, k2pp


@numba.njit(fastmath=True)
def _epan_all(u: float) -> tuple[float, float, float, float, float, float]:
    """Compute Epanechnikov kernel, derivatives, convolution and its derivatives."""
    absu = abs(u)

    if absu <= 1.0:
        k = 0.75 * (1.0 - u * u)
        kp = -1.5 * u
        kpp = -1.5
    else:
        k = 0.0
        kp = 0.0
        kpp = 0.0

    if absu <= 2.0:
        k2 = 0.6 - 0.75 * absu * absu + 0.375 * absu**3 - 0.01875 * absu**5
        sign_u = 1.0 if u >= 0 else -1.0
        k2p = sign_u * (-0.09375 * absu**4 + 1.125 * absu**2 - 1.5 * absu)
        k2pp = -0.375 * absu**3 + 2.25 * absu - 1.5
    else:
        k2 = 0.0
        k2p = 0.0
        k2pp = 0.0

    return k, kp, kpp, k2, k2p, k2pp


@numba.njit(fastmath=True)
def _unif_all(u: float) -> tuple[float, float, float, float, float, float]:
    """Compute uniform kernel, derivatives, convolution and its derivatives."""
    absu = abs(u)

    k = 0.5 if absu <= 1.0 else 0.0
    kp = 0.0
    kpp = 0.0

    if absu <= 2.0:
        k2 = 0.5 * (1.0 - 0.5 * absu)
        if u > 0:
            sign_u = 1.0
        elif u < 0:
            sign_u = -1.0
        else:
            sign_u = 0.0
        k2p = -0.25 * sign_u
        k2pp = 0.0
    else:
        k2 = 0.0
        k2p = 0.0
        k2pp = 0.0

    return k, kp, kpp, k2, k2p, k2pp


@numba.njit(fastmath=True)
def _biweight_all(u: float) -> tuple[float, float, float, float, float, float]:
    """Compute biweight kernel, derivatives, convolution and its derivatives."""
    absu = abs(u)
    uu = u * u

    if absu <= 1.0:
        one_minus_uu = 1.0 - uu
        k = (15.0 / 16.0) * one_minus_uu * one_minus_uu
        kp = -(15.0 / 4.0) * u * one_minus_uu
        kpp = (15.0 / 4.0) * (3.0 * uu - 1.0)
    else:
        k = 0.0
        kp = 0.0
        kpp = 0.0

    if absu <= 2.0:
        k2 = (
            5.0 / 7.0
            - 15.0 * uu / 14.0
            + 15.0 * uu * uu / 16.0
            - 15.0 * absu**5 / 32.0
            + 15.0 * absu**7 / 448.0
            - 5.0 * absu**9 / 3584.0
        )
        k2p = (
            15.0
            * u
            * (-3.0 * absu**7 + 56.0 * absu**5 - 560.0 * absu**3 + 896.0 * uu - 512.0)
            / 3584.0
        )
        k2pp = (
            -45.0 * absu**7 / 448.0
            + 45.0 * absu**5 / 32.0
            - 75.0 * absu**3 / 8.0
            + 45.0 * uu / 4.0
            - 15.0 / 7.0
        )
    else:
        k2 = 0.0
        k2p = 0.0
        k2pp = 0.0

    return k, kp, kpp, k2, k2p, k2pp


@numba.njit(fastmath=True)
def _triweight_all(u: float) -> tuple[float, float, float, float, float, float]:
    """Compute triweight kernel, derivatives, convolution and its derivatives."""
    absu = abs(u)
    uu = u * u

    if absu <= 1.0:
        one_minus_uu = 1.0 - uu
        k = (35.0 / 32.0) * one_minus_uu * one_minus_uu * one_minus_uu
        kp = -(105.0 / 16.0) * u * one_minus_uu * one_minus_uu
        kpp = -(105.0 / 16.0) * (1.0 - 6.0 * uu + 5.0 * uu * uu)
    else:
        k = 0.0
        kp = 0.0
        kpp = 0.0

    if absu <= 2.0:
        k2 = (
            350.0 / 429.0
            - 35.0 * uu / 22.0
            + 35.0 * uu * uu / 24.0
            - 35.0 * absu**6 / 32.0
            + 35.0 * absu**7 / 64.0
            - 35.0 * absu**9 / 768.0
            + 35.0 * absu**11 / 11264.0
            - 175.0 * absu**13 / 1757184.0
        )
        k2p = (
            35.0
            * u
            * (
                -5.0 * absu**11
                + 132.0 * absu**9
                - 1584.0 * absu**7
                + 14784.0 * absu**5
                - 25344.0 * uu * uu
                + 22528.0 * uu
                - 12288.0
            )
            / 135168.0
        )
        k2pp = (
            -175.0 * absu**11 / 11264.0
            + 175.0 * absu**9 / 512.0
            - 105.0 * absu**7 / 32.0
            + 735.0 * absu**5 / 32.0
            - 525.0 * uu * uu / 16.0
            + 35.0 * uu / 2.0
            - 35.0 / 11.0
        )
    else:
        k2 = 0.0
        k2p = 0.0
        k2pp = 0.0

    return k, kp, kpp, k2, k2p, k2pp


@numba.njit(fastmath=True)
def _cosine_all(u: float) -> tuple[float, float, float, float, float, float]:
    """Compute cosine kernel, derivatives, convolution and its derivatives."""
    absu = abs(u)

    if absu <= 1.0:
        cos_val = math.cos(_PI_2 * u)
        sin_val = math.sin(_PI_2 * u)
        k = _PI_4 * cos_val
        kp = -(_PI * _PI / 8.0) * sin_val
        kpp = -(_PI * _PI * _PI / 16.0) * cos_val
    else:
        k = 0.0
        kp = 0.0
        kpp = 0.0

    if absu <= 2.0:
        cos_half = math.cos(_PI_2 * absu)
        sin_half = math.sin(_PI_2 * absu)
        k2 = _PI / 32.0 * (_PI * (2.0 - absu) * cos_half + 2.0 * sin_half)
        sign_u = 1.0 if u >= 0 else -1.0
        k2p = sign_u * (-_PI**3 / 64.0) * sin_half * (2.0 - absu)
        k2pp = _PI**3 / 128.0 * (_PI * (absu - 2.0) * cos_half + 2.0 * sin_half)
    else:
        k2 = 0.0
        k2p = 0.0
        k2pp = 0.0

    return k, kp, kpp, k2, k2p, k2pp


@numba.njit(fastmath=True, parallel=True)
def lscv_numba_gauss(x: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LSCV score, gradient, Hessian for Gaussian kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 6))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            k, kp, kpp, k2, k2p, k2pp = _gauss_all(u)

            local_sums[tid, 0] += k2
            local_sums[tid, 1] += k2 + u * k2p
            local_sums[tid, 2] += 2.0 * k2p + u * k2pp

            if i != j:
                local_sums[tid, 3] += k
                local_sums[tid, 4] += k + u * kp
                local_sums[tid, 5] += 2.0 * kp + u * kpp

    sum_K2 = local_sums[:, 0].sum()
    sum_SF = local_sums[:, 1].sum()
    sum_SF2 = local_sums[:, 2].sum()
    sum_K = local_sums[:, 3].sum()
    sum_SK = local_sums[:, 4].sum()
    sum_SK2 = local_sums[:, 5].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    h2 = h * h
    h3 = h2 * h

    score = sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)
    grad = -sum_SF / (n2 * h2) + 2.0 * sum_SK / (nn1 * h2)
    hess = (
        2.0 * sum_SF / (n2 * h3)
        - sum_SF2 / (n2 * h2)
        - 4.0 * sum_SK / (nn1 * h3)
        + 2.0 * sum_SK2 / (nn1 * h2)
    )

    return score, grad, hess


@numba.njit(fastmath=True, parallel=True)
def lscv_score_numba_gauss(x: NDArray, h: float) -> float:
    """Compute LSCV score only (for Armijo backtracking) - Gaussian kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 2))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h

            exp_quarter = math.exp(-0.25 * u * u)
            k2 = exp_quarter / _SQRT_4PI
            local_sums[tid, 0] += k2

            if i != j:
                exp_half = math.exp(-0.5 * u * u)
                k = exp_half / _SQRT_2PI
                local_sums[tid, 1] += k

    sum_K2 = local_sums[:, 0].sum()
    sum_K = local_sums[:, 1].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    return sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)


@numba.njit(fastmath=True, parallel=True)
def lscv_numba_epan(x: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LSCV score, gradient, Hessian for Epanechnikov kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 6))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            k, kp, kpp, k2, k2p, k2pp = _epan_all(u)

            local_sums[tid, 0] += k2
            local_sums[tid, 1] += k2 + u * k2p
            local_sums[tid, 2] += 2.0 * k2p + u * k2pp

            if i != j:
                local_sums[tid, 3] += k
                local_sums[tid, 4] += k + u * kp
                local_sums[tid, 5] += 2.0 * kp + u * kpp

    sum_K2 = local_sums[:, 0].sum()
    sum_SF = local_sums[:, 1].sum()
    sum_SF2 = local_sums[:, 2].sum()
    sum_K = local_sums[:, 3].sum()
    sum_SK = local_sums[:, 4].sum()
    sum_SK2 = local_sums[:, 5].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    h2 = h * h
    h3 = h2 * h

    score = sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)
    grad = -sum_SF / (n2 * h2) + 2.0 * sum_SK / (nn1 * h2)
    hess = (
        2.0 * sum_SF / (n2 * h3)
        - sum_SF2 / (n2 * h2)
        - 4.0 * sum_SK / (nn1 * h3)
        + 2.0 * sum_SK2 / (nn1 * h2)
    )

    return score, grad, hess


@numba.njit(fastmath=True, parallel=True)
def lscv_score_numba_epan(x: NDArray, h: float) -> float:
    """Compute LSCV score only (for Armijo backtracking) - Epanechnikov kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 2))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            absu = abs(u)

            if absu <= 2.0:
                k2 = 0.6 - 0.75 * absu * absu + 0.375 * absu**3 - 0.01875 * absu**5
                local_sums[tid, 0] += k2

            if i != j and absu <= 1.0:
                k = 0.75 * (1.0 - u * u)
                local_sums[tid, 1] += k

    sum_K2 = local_sums[:, 0].sum()
    sum_K = local_sums[:, 1].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    return sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)


@numba.njit(fastmath=True, parallel=True)
def lscv_numba_unif(x: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LSCV score, gradient, Hessian for uniform kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 6))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            k, kp, kpp, k2, k2p, k2pp = _unif_all(u)

            local_sums[tid, 0] += k2
            local_sums[tid, 1] += k2 + u * k2p
            local_sums[tid, 2] += 2.0 * k2p + u * k2pp

            if i != j:
                local_sums[tid, 3] += k
                local_sums[tid, 4] += k + u * kp
                local_sums[tid, 5] += 2.0 * kp + u * kpp

    sum_K2 = local_sums[:, 0].sum()
    sum_SF = local_sums[:, 1].sum()
    sum_SF2 = local_sums[:, 2].sum()
    sum_K = local_sums[:, 3].sum()
    sum_SK = local_sums[:, 4].sum()
    sum_SK2 = local_sums[:, 5].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    h2 = h * h
    h3 = h2 * h

    score = sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)
    grad = -sum_SF / (n2 * h2) + 2.0 * sum_SK / (nn1 * h2)
    hess = (
        2.0 * sum_SF / (n2 * h3)
        - sum_SF2 / (n2 * h2)
        - 4.0 * sum_SK / (nn1 * h3)
        + 2.0 * sum_SK2 / (nn1 * h2)
    )

    return score, grad, hess


@numba.njit(fastmath=True, parallel=True)
def lscv_score_numba_unif(x: NDArray, h: float) -> float:
    """Compute LSCV score only (for Armijo backtracking) - uniform kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 2))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            absu = abs(u)

            if absu <= 2.0:
                k2 = 0.5 * (1.0 - 0.5 * absu)
                local_sums[tid, 0] += k2

            if i != j and absu <= 1.0:
                local_sums[tid, 1] += 0.5

    sum_K2 = local_sums[:, 0].sum()
    sum_K = local_sums[:, 1].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    return sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)


@numba.njit(fastmath=True, parallel=True)
def lscv_numba_biweight(x: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LSCV score, gradient, Hessian for biweight kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 6))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            k, kp, kpp, k2, k2p, k2pp = _biweight_all(u)

            local_sums[tid, 0] += k2
            local_sums[tid, 1] += k2 + u * k2p
            local_sums[tid, 2] += 2.0 * k2p + u * k2pp

            if i != j:
                local_sums[tid, 3] += k
                local_sums[tid, 4] += k + u * kp
                local_sums[tid, 5] += 2.0 * kp + u * kpp

    sum_K2 = local_sums[:, 0].sum()
    sum_SF = local_sums[:, 1].sum()
    sum_SF2 = local_sums[:, 2].sum()
    sum_K = local_sums[:, 3].sum()
    sum_SK = local_sums[:, 4].sum()
    sum_SK2 = local_sums[:, 5].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    h2 = h * h
    h3 = h2 * h

    score = sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)
    grad = -sum_SF / (n2 * h2) + 2.0 * sum_SK / (nn1 * h2)
    hess = (
        2.0 * sum_SF / (n2 * h3)
        - sum_SF2 / (n2 * h2)
        - 4.0 * sum_SK / (nn1 * h3)
        + 2.0 * sum_SK2 / (nn1 * h2)
    )

    return score, grad, hess


@numba.njit(fastmath=True, parallel=True)
def lscv_score_numba_biweight(x: NDArray, h: float) -> float:
    """Compute LSCV score only (for Armijo backtracking) - biweight kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 2))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            absu = abs(u)
            uu = u * u

            if absu <= 2.0:
                k2 = (
                    5.0 / 7.0
                    - 15.0 * uu / 14.0
                    + 15.0 * uu * uu / 16.0
                    - 15.0 * absu**5 / 32.0
                    + 15.0 * absu**7 / 448.0
                    - 5.0 * absu**9 / 3584.0
                )
                local_sums[tid, 0] += k2

            if i != j and absu <= 1.0:
                one_minus_uu = 1.0 - uu
                k = (15.0 / 16.0) * one_minus_uu * one_minus_uu
                local_sums[tid, 1] += k

    sum_K2 = local_sums[:, 0].sum()
    sum_K = local_sums[:, 1].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    return sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)


@numba.njit(fastmath=True, parallel=True)
def lscv_numba_triweight(x: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LSCV score, gradient, Hessian for triweight kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 6))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            k, kp, kpp, k2, k2p, k2pp = _triweight_all(u)

            local_sums[tid, 0] += k2
            local_sums[tid, 1] += k2 + u * k2p
            local_sums[tid, 2] += 2.0 * k2p + u * k2pp

            if i != j:
                local_sums[tid, 3] += k
                local_sums[tid, 4] += k + u * kp
                local_sums[tid, 5] += 2.0 * kp + u * kpp

    sum_K2 = local_sums[:, 0].sum()
    sum_SF = local_sums[:, 1].sum()
    sum_SF2 = local_sums[:, 2].sum()
    sum_K = local_sums[:, 3].sum()
    sum_SK = local_sums[:, 4].sum()
    sum_SK2 = local_sums[:, 5].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    h2 = h * h
    h3 = h2 * h

    score = sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)
    grad = -sum_SF / (n2 * h2) + 2.0 * sum_SK / (nn1 * h2)
    hess = (
        2.0 * sum_SF / (n2 * h3)
        - sum_SF2 / (n2 * h2)
        - 4.0 * sum_SK / (nn1 * h3)
        + 2.0 * sum_SK2 / (nn1 * h2)
    )

    return score, grad, hess


@numba.njit(fastmath=True, parallel=True)
def lscv_score_numba_triweight(x: NDArray, h: float) -> float:
    """Compute LSCV score only (for Armijo backtracking) - triweight kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 2))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            absu = abs(u)
            uu = u * u

            if absu <= 2.0:
                k2 = (
                    350.0 / 429.0
                    - 35.0 * uu / 22.0
                    + 35.0 * uu * uu / 24.0
                    - 35.0 * absu**6 / 32.0
                    + 35.0 * absu**7 / 64.0
                    - 35.0 * absu**9 / 768.0
                    + 35.0 * absu**11 / 11264.0
                    - 175.0 * absu**13 / 1757184.0
                )
                local_sums[tid, 0] += k2

            if i != j and absu <= 1.0:
                one_minus_uu = 1.0 - uu
                k = (35.0 / 32.0) * one_minus_uu * one_minus_uu * one_minus_uu
                local_sums[tid, 1] += k

    sum_K2 = local_sums[:, 0].sum()
    sum_K = local_sums[:, 1].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    return sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)


@numba.njit(fastmath=True, parallel=True)
def lscv_numba_cosine(x: NDArray, h: float) -> tuple[float, float, float]:
    """Compute LSCV score, gradient, Hessian for cosine kernel using fused loop."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 6))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            k, kp, kpp, k2, k2p, k2pp = _cosine_all(u)

            local_sums[tid, 0] += k2
            local_sums[tid, 1] += k2 + u * k2p
            local_sums[tid, 2] += 2.0 * k2p + u * k2pp

            if i != j:
                local_sums[tid, 3] += k
                local_sums[tid, 4] += k + u * kp
                local_sums[tid, 5] += 2.0 * kp + u * kpp

    sum_K2 = local_sums[:, 0].sum()
    sum_SF = local_sums[:, 1].sum()
    sum_SF2 = local_sums[:, 2].sum()
    sum_K = local_sums[:, 3].sum()
    sum_SK = local_sums[:, 4].sum()
    sum_SK2 = local_sums[:, 5].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    h2 = h * h
    h3 = h2 * h

    score = sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)
    grad = -sum_SF / (n2 * h2) + 2.0 * sum_SK / (nn1 * h2)
    hess = (
        2.0 * sum_SF / (n2 * h3)
        - sum_SF2 / (n2 * h2)
        - 4.0 * sum_SK / (nn1 * h3)
        + 2.0 * sum_SK2 / (nn1 * h2)
    )

    return score, grad, hess


@numba.njit(fastmath=True, parallel=True)
def lscv_score_numba_cosine(x: NDArray, h: float) -> float:
    """Compute LSCV score only (for Armijo backtracking) - cosine kernel."""
    n = len(x)
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 2))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            u = (x[i] - x[j]) * inv_h
            absu = abs(u)

            if absu <= 2.0:
                cos_half = math.cos(_PI_2 * absu)
                sin_half = math.sin(_PI_2 * absu)
                k2 = _PI / 32.0 * (_PI * (2.0 - absu) * cos_half + 2.0 * sin_half)
                local_sums[tid, 0] += k2

            if i != j and absu <= 1.0:
                k = _PI_4 * math.cos(_PI_2 * u)
                local_sums[tid, 1] += k

    sum_K2 = local_sums[:, 0].sum()
    sum_K = local_sums[:, 1].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    return sum_K2 / (n2 * h) - 2.0 * sum_K / (nn1 * h)


@numba.njit(fastmath=True, parallel=True)
def lscv_mv_numba_gauss(data: NDArray, h: float) -> tuple[float, float, float]:
    """Compute multivariate LSCV for Gaussian product kernel using fused loop."""
    n, d = data.shape
    inv_h = 1.0 / h
    n_threads = numba.get_num_threads()

    local_sums = np.zeros((n_threads, 6))

    for i in numba.prange(n):
        tid = numba.get_thread_id()
        for j in range(n):
            prod_k = 1.0
            prod_k2 = 1.0
            sum_ratio_k = 0.0
            sum_ratio_k2 = 0.0
            sum_d2_k = 0.0
            sum_d2_k2 = 0.0

            for dim in range(d):
                u = (data[i, dim] - data[j, dim]) * inv_h

                exp_half = math.exp(-0.5 * u * u)
                k = exp_half / _SQRT_2PI
                kp = -u * k
                kpp = (u * u - 1.0) * k

                exp_quarter = math.exp(-0.25 * u * u)
                k2 = exp_quarter / _SQRT_4PI
                k2p = -0.5 * u * k2
                k2pp = (0.25 * u * u - 0.5) * k2

                prod_k *= k
                prod_k2 *= k2

                if k > 0:
                    r = kp / k
                    sum_ratio_k += u * r
                    sum_d2_k += 2.0 * u * r + u * u * kpp / k

                if k2 > 0:
                    r2 = k2p / k2
                    sum_ratio_k2 += u * r2
                    sum_d2_k2 += 2.0 * u * r2 + u * u * k2pp / k2

            local_sums[tid, 0] += prod_k2
            local_sums[tid, 1] += prod_k2 * (d + sum_ratio_k2)
            local_sums[tid, 2] += prod_k2 * (
                (d + 1) * d + 2.0 * (d + 1) * sum_ratio_k2 + sum_d2_k2
            )

            if i != j:
                local_sums[tid, 3] += prod_k
                local_sums[tid, 4] += prod_k * (d + sum_ratio_k)
                local_sums[tid, 5] += prod_k * (
                    (d + 1) * d + 2.0 * (d + 1) * sum_ratio_k + sum_d2_k
                )

    sum_K2 = local_sums[:, 0].sum()
    sum_SF = local_sums[:, 1].sum()
    sum_SF2 = local_sums[:, 2].sum()
    sum_K = local_sums[:, 3].sum()
    sum_SK = local_sums[:, 4].sum()
    sum_SK2 = local_sums[:, 5].sum()

    n2 = float(n * n)
    nn1 = float(n * (n - 1))
    hd = h**d
    hd1 = h ** (d + 1)
    hd2 = h ** (d + 2)

    score = sum_K2 / (n2 * hd) - 2.0 * sum_K / (nn1 * hd)
    grad = -sum_SF / (n2 * hd1) + 2.0 * sum_SK / (nn1 * hd1)
    hess = (
        (d + 1) * sum_SF / (n2 * hd2)
        - sum_SF2 / (n2 * hd1)
        - 2.0 * (d + 1) * sum_SK / (nn1 * hd2)
        + 2.0 * sum_SK2 / (nn1 * hd1)
    )

    return score, grad, hess


def warmup() -> None:
    """Trigger JIT compilation with small arrays."""
    x = np.array([0.0, 1.0, 2.0])
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lscv_numba_gauss(x, 1.0)
    lscv_score_numba_gauss(x, 1.0)
    lscv_numba_epan(x, 1.0)
    lscv_score_numba_epan(x, 1.0)
    lscv_numba_unif(x, 1.0)
    lscv_score_numba_unif(x, 1.0)
    lscv_numba_biweight(x, 1.0)
    lscv_score_numba_biweight(x, 1.0)
    lscv_numba_triweight(x, 1.0)
    lscv_score_numba_triweight(x, 1.0)
    lscv_numba_cosine(x, 1.0)
    lscv_score_numba_cosine(x, 1.0)
    lscv_mv_numba_gauss(data, 1.0)
