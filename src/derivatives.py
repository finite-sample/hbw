import numpy as np

SQRT_2PI = np.sqrt(2 * np.pi)


def _poly_mask(u: np.ndarray, mask: np.ndarray, expr: np.ndarray) -> np.ndarray:
    """Return piecewise polynomial values for Epanechnikov kernels.

    Parameters
    ----------
    u : np.ndarray
        Evaluation points.
    mask : np.ndarray
        Boolean mask where the polynomial expression is valid.
    expr : np.ndarray
        Polynomial expression evaluated elementwise on ``u``.
    """
    out = np.zeros_like(u, dtype=float)
    if np.isscalar(u):
        return expr if mask else 0.0
    out[mask] = expr[mask]
    return out


# Gaussian kernel and its convolution -----------------------------------------

def gauss(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel K(u)."""
    return np.exp(-0.5 * u * u) / SQRT_2PI


def gauss_p(u: np.ndarray) -> np.ndarray:
    """First derivative of the Gaussian kernel."""
    return -u * gauss(u)


def gauss_pp(u: np.ndarray) -> np.ndarray:
    """Second derivative of the Gaussian kernel."""
    return (u * u - 1.0) * gauss(u)


def gauss_conv(u: np.ndarray) -> np.ndarray:
    """Convolution K*K of the Gaussian kernel."""
    return np.exp(-0.25 * u * u) / np.sqrt(4 * np.pi)


def gauss_conv_p(u: np.ndarray) -> np.ndarray:
    """First derivative of the Gaussian kernel convolution."""
    return -0.5 * u * gauss_conv(u)


def gauss_conv_pp(u: np.ndarray) -> np.ndarray:
    """Second derivative of the Gaussian kernel convolution."""
    return (0.25 * u * u - 0.5) * gauss_conv(u)


# Epanechnikov kernel and its convolution ------------------------------------

def _abs(u: np.ndarray) -> np.ndarray:
    return np.abs(u)


def epan(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel K(u)."""
    return _poly_mask(u, _abs(u) <= 1, 0.75 * (1 - u * u))


def epan_p(u: np.ndarray) -> np.ndarray:
    """First derivative of the Epanechnikov kernel."""
    return _poly_mask(u, _abs(u) <= 1, -1.5 * u)


def epan_pp(u: np.ndarray) -> np.ndarray:
    """Second derivative of the Epanechnikov kernel."""
    return _poly_mask(u, _abs(u) <= 1, -1.5 + 0.0 * u)


def epan_conv(u: np.ndarray) -> np.ndarray:
    """Convolution K*K of the Epanechnikov kernel (valid for |u|≤2)."""
    absu = _abs(u)
    poly = 0.6 - 0.75 * absu**2 + 0.375 * absu**3 - 0.01875 * absu**5
    return _poly_mask(u, absu <= 2, poly)


def epan_conv_p(u: np.ndarray) -> np.ndarray:
    """First derivative of the Epanechnikov kernel convolution."""
    absu = _abs(u)
    poly = np.sign(u) * (-0.09375 * absu**4 + 1.125 * absu**2 - 1.5 * absu)
    return _poly_mask(u, absu <= 2, poly)


def epan_conv_pp(u: np.ndarray) -> np.ndarray:
    """Second derivative of the Epanechnikov kernel convolution."""
    absu = _abs(u)
    poly = -0.375 * absu**3 + 2.25 * absu - 1.5
    return _poly_mask(u, absu <= 2, poly)


# Convenience dictionaries ----------------------------------------------------

KERNELS = {
    "gauss": (gauss, gauss_p, gauss_pp, gauss_conv, gauss_conv_p, gauss_conv_pp),
    "epan": (epan, epan_p, epan_pp, epan_conv, epan_conv_p, epan_conv_pp),
}


# NW regression weight functions (derivatives w.r.t. h) -----------------------

def nw_weights_gauss(u: np.ndarray, h: float) -> tuple:
    """Gaussian weights and h-derivatives for Nadaraya-Watson regression.

    Returns w, dw/dh, d²w/dh² for the NW kernel weight K(u)/h.
    """
    base = np.exp(-0.5 * u * u) / (h * SQRT_2PI)
    w1 = base * (u * u - 1) / h
    w2 = base * (u**4 - 3 * u * u + 1) / (h * h)
    return base, w1, w2


def nw_weights_epan(u: np.ndarray, h: float) -> tuple:
    """Epanechnikov weights and h-derivatives for Nadaraya-Watson regression.

    Returns w, dw/dh, d²w/dh² for the NW kernel weight K(u)/h.
    """
    mask = _abs(u) <= 1
    w = np.zeros_like(u, dtype=float)
    w1 = np.zeros_like(u, dtype=float)
    w2 = np.zeros_like(u, dtype=float)
    uu = u * u
    w[mask] = 0.75 * (1 - uu[mask]) / h
    w1[mask] = 0.75 * (-1 + 3 * uu[mask]) / (h * h)
    w2[mask] = 1.5 * (1 - 6 * uu[mask]) / (h ** 3)
    return w, w1, w2


NW_WEIGHTS = {
    "gauss": nw_weights_gauss,
    "epan": nw_weights_epan,
}
