"""Kernel functions and their derivatives for bandwidth selection."""

from __future__ import annotations

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


_KERNELS = {
    "gauss": (_gauss, _gauss_p, _gauss_pp, _gauss_conv, _gauss_conv_p, _gauss_conv_pp),
    "epan": (_epan, _epan_p, _epan_pp, _epan_conv, _epan_conv_p, _epan_conv_pp),
}
