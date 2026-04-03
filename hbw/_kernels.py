"""Kernel functions and their derivatives for bandwidth selection."""

from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

_SQRT_2PI = sqrt(2 * np.pi)
_SQRT_4PI = sqrt(4 * np.pi)


def _gauss(u: NDArray[Any]) -> NDArray[Any]:
    """Gaussian kernel K(u)."""
    return np.exp(-0.5 * u * u) / _SQRT_2PI


def _gauss_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Gaussian kernel."""
    return -u * _gauss(u)


def _gauss_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Gaussian kernel."""
    return (u * u - 1.0) * _gauss(u)


def _gauss_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of Gaussian kernel."""
    return np.exp(-0.25 * u * u) / _SQRT_4PI


def _gauss_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Gaussian kernel convolution."""
    return -0.5 * u * _gauss_conv(u)


def _gauss_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Gaussian kernel convolution."""
    return (0.25 * u * u - 0.5) * _gauss_conv(u)


def _poly_mask(u: NDArray[Any], mask: NDArray[Any], expr: NDArray[Any]) -> NDArray[Any]:
    """Return piecewise polynomial values for Epanechnikov kernels."""
    out = np.zeros_like(u, dtype=float)
    out[mask] = expr[mask]
    return out


def _epan(u: NDArray[Any]) -> NDArray[Any]:
    """Epanechnikov kernel K(u)."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, 0.75 * (1 - u * u))


def _epan_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Epanechnikov kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, -1.5 * u)


def _epan_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Epanechnikov kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, np.full_like(u, -1.5))


def _epan_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of Epanechnikov kernel (valid for |u|<=2)."""
    absu = np.abs(u)
    mask = absu <= 2
    poly = 0.6 - 0.75 * absu**2 + 0.375 * absu**3 - 0.01875 * absu**5
    return _poly_mask(u, mask, poly)


def _epan_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of Epanechnikov kernel convolution."""
    absu = np.abs(u)
    mask = absu <= 2
    poly = np.sign(u) * (-0.09375 * absu**4 + 1.125 * absu**2 - 1.5 * absu)
    return _poly_mask(u, mask, poly)


def _epan_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of Epanechnikov kernel convolution."""
    absu = np.abs(u)
    mask = absu <= 2
    poly = -0.375 * absu**3 + 2.25 * absu - 1.5
    return _poly_mask(u, mask, poly)


def _unif(u: NDArray[Any]) -> NDArray[Any]:
    """Uniform (rectangular) kernel K(u) = 0.5 for |u| <= 1."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, np.full_like(u, 0.5))


def _unif_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of uniform kernel (zero inside support)."""
    return np.zeros_like(u, dtype=float)


def _unif_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of uniform kernel (zero inside support)."""
    return np.zeros_like(u, dtype=float)


def _unif_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of uniform kernel (triangle/tent function)."""
    absu = np.abs(u)
    mask = absu <= 2
    return _poly_mask(u, mask, 0.5 * (1 - 0.5 * absu))


def _unif_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of uniform kernel convolution."""
    absu = np.abs(u)
    mask = absu <= 2
    return _poly_mask(u, mask, -0.25 * np.sign(u))


def _unif_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of uniform kernel convolution (zero except at u=0)."""
    return np.zeros_like(u, dtype=float)


def _biweight(u: NDArray[Any]) -> NDArray[Any]:
    """Biweight (quartic) kernel K(u) = (15/16)(1-u²)² for |u| ≤ 1."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, (15 / 16) * (1 - u * u) ** 2)


def _biweight_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of biweight kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, -(15 / 4) * u * (1 - u * u))


def _biweight_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of biweight kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, (15 / 4) * (3 * u * u - 1))


def _biweight_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of biweight kernel (valid for |u| ≤ 2).

    Formula: 5/7 - 15u²/14 + 15u⁴/16 - 15|u|⁵/32 + 15|u|⁷/448 - 5|u|⁹/3584
    """
    absu = np.abs(u)
    mask = absu <= 2
    uu = u * u
    poly = (
        5 / 7
        - 15 * uu / 14
        + 15 * uu**2 / 16
        - 15 * absu**5 / 32
        + 15 * absu**7 / 448
        - 5 * absu**9 / 3584
    )
    return _poly_mask(u, mask, poly)


def _biweight_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of biweight kernel convolution.

    Formula: 15u(-3u⁷ + 56u⁵ - 560u³ + 896u² - 512) / 3584
    """
    absu = np.abs(u)
    mask = absu <= 2
    uu = u * u
    poly = 15 * u * (-3 * absu**7 + 56 * absu**5 - 560 * absu**3 + 896 * uu - 512) / 3584
    return _poly_mask(u, mask, poly)


def _biweight_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of biweight kernel convolution.

    Formula: -45|u|⁷/448 + 45|u|⁵/32 - 75|u|³/8 + 45u²/4 - 15/7
    """
    absu = np.abs(u)
    mask = absu <= 2
    uu = u * u
    poly = -45 * absu**7 / 448 + 45 * absu**5 / 32 - 75 * absu**3 / 8 + 45 * uu / 4 - 15 / 7
    return _poly_mask(u, mask, poly)


def _triweight(u: NDArray[Any]) -> NDArray[Any]:
    """Triweight kernel K(u) = (35/32)(1-u²)³ for |u| ≤ 1."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, (35 / 32) * (1 - u * u) ** 3)


def _triweight_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of triweight kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, -(105 / 16) * u * (1 - u * u) ** 2)


def _triweight_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of triweight kernel."""
    mask = np.abs(u) <= 1
    uu = u * u
    return _poly_mask(u, mask, -(105 / 16) * (1 - 6 * uu + 5 * uu * uu))


def _triweight_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of triweight kernel (valid for |u| ≤ 2).

    Formula: 350/429 - 35u²/22 + 35u⁴/24 - 35|u|⁶/32 + 35|u|⁷/64
             - 35|u|⁹/768 + 35|u|¹¹/11264 - 175|u|¹³/1757184
    """
    absu = np.abs(u)
    mask = absu <= 2
    uu = u * u
    poly = (
        350 / 429
        - 35 * uu / 22
        + 35 * uu**2 / 24
        - 35 * absu**6 / 32
        + 35 * absu**7 / 64
        - 35 * absu**9 / 768
        + 35 * absu**11 / 11264
        - 175 * absu**13 / 1757184
    )
    return _poly_mask(u, mask, poly)


def _triweight_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of triweight kernel convolution.

    Formula: 35u(-5|u|¹¹ + 132|u|⁹ - 1584|u|⁷ + 14784|u|⁵ - 25344u⁴ + 22528u² - 12288) / 135168
    """
    absu = np.abs(u)
    mask = absu <= 2
    uu = u * u
    poly = (
        35
        * u
        * (
            -5 * absu**11
            + 132 * absu**9
            - 1584 * absu**7
            + 14784 * absu**5
            - 25344 * uu**2
            + 22528 * uu
            - 12288
        )
        / 135168
    )
    return _poly_mask(u, mask, poly)


def _triweight_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of triweight kernel convolution.

    Formula: -175|u|¹¹/11264 + 175|u|⁹/512 - 105|u|⁷/32 + 735|u|⁵/32
             - 525u⁴/16 + 35u²/2 - 35/11
    """
    absu = np.abs(u)
    mask = absu <= 2
    uu = u * u
    poly = (
        -175 * absu**11 / 11264
        + 175 * absu**9 / 512
        - 105 * absu**7 / 32
        + 735 * absu**5 / 32
        - 525 * uu**2 / 16
        + 35 * uu / 2
        - 35 / 11
    )
    return _poly_mask(u, mask, poly)


def _cosine(u: NDArray[Any]) -> NDArray[Any]:
    """Cosine kernel K(u) = (π/4)cos(πu/2) for |u| ≤ 1."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, (np.pi / 4) * np.cos(np.pi * u / 2))


def _cosine_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of cosine kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, -(np.pi**2 / 8) * np.sin(np.pi * u / 2))


def _cosine_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of cosine kernel."""
    mask = np.abs(u) <= 1
    return _poly_mask(u, mask, -(np.pi**3 / 16) * np.cos(np.pi * u / 2))


def _cosine_conv(u: NDArray[Any]) -> NDArray[Any]:
    """Convolution K*K of cosine kernel (valid for |u| ≤ 2).

    Formula: π/32 * (π*(2-|u|)*cos(π|u|/2) + 2*sin(π|u|/2))
    """
    absu = np.abs(u)
    mask = absu <= 2
    poly = (
        np.pi / 32 * (np.pi * (2 - absu) * np.cos(np.pi * absu / 2) + 2 * np.sin(np.pi * absu / 2))
    )
    return _poly_mask(u, mask, poly)


def _cosine_conv_p(u: NDArray[Any]) -> NDArray[Any]:
    """First derivative of cosine kernel convolution.

    Formula: sign(u) * (-π³/64) * sin(π|u|/2) * (2 - |u|)
    """
    absu = np.abs(u)
    mask = absu <= 2
    poly = np.sign(u) * (-(np.pi**3) / 64) * np.sin(np.pi * absu / 2) * (2 - absu)
    return _poly_mask(u, mask, poly)


def _cosine_conv_pp(u: NDArray[Any]) -> NDArray[Any]:
    """Second derivative of cosine kernel convolution.

    Formula: π³/128 * (π*(|u|-2)*cos(π|u|/2) + 2*sin(π|u|/2))
    """
    absu = np.abs(u)
    mask = absu <= 2
    poly = (
        np.pi**3
        / 128
        * (np.pi * (absu - 2) * np.cos(np.pi * absu / 2) + 2 * np.sin(np.pi * absu / 2))
    )
    return _poly_mask(u, mask, poly)


_KERNELS = {
    "gauss": (_gauss, _gauss_p, _gauss_pp, _gauss_conv, _gauss_conv_p, _gauss_conv_pp),
    "epan": (_epan, _epan_p, _epan_pp, _epan_conv, _epan_conv_p, _epan_conv_pp),
    "unif": (_unif, _unif_p, _unif_pp, _unif_conv, _unif_conv_p, _unif_conv_pp),
    "biweight": (
        _biweight,
        _biweight_p,
        _biweight_pp,
        _biweight_conv,
        _biweight_conv_p,
        _biweight_conv_pp,
    ),
    "triweight": (
        _triweight,
        _triweight_p,
        _triweight_pp,
        _triweight_conv,
        _triweight_conv_p,
        _triweight_conv_pp,
    ),
    "cosine": (
        _cosine,
        _cosine_p,
        _cosine_pp,
        _cosine_conv,
        _cosine_conv_p,
        _cosine_conv_pp,
    ),
}
