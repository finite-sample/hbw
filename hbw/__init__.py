"""Fast kernel bandwidth selection via analytic Hessian Newton optimization.

This module provides optimal bandwidth selection for:
- Univariate kernel density estimation (KDE) via LSCV minimization
- Nadaraya-Watson regression via LOOCV-MSE minimization

The key innovation is using closed-form analytic gradients and Hessians
of the cross-validation objectives, enabling Newton optimization that
converges in 6-12 evaluations vs 50-100 for grid search.
"""

from __future__ import annotations

from ._kernels import _KERNELS
from .kde import kde_bandwidth, kde_bandwidth_mv, lscv, lscv_mv, lscv_score
from .nw import loocv_mse, loocv_mse_score, nw_bandwidth

__all__ = [
    "kde_bandwidth",
    "kde_bandwidth_mv",
    "nw_bandwidth",
    "lscv",
    "lscv_score",
    "lscv_mv",
    "loocv_mse",
    "loocv_mse_score",
    "_KERNELS",
]
__version__ = "0.2.0"
