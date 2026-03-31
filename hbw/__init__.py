"""Fast kernel bandwidth selection via analytic Hessian Newton optimization.

This module provides optimal bandwidth selection for:
- Univariate kernel density estimation (KDE) via LSCV minimization
- Nadaraya-Watson regression via LOOCV-MSE minimization

The key innovation is using closed-form analytic gradients and Hessians
of the cross-validation objectives, enabling Newton optimization that
converges in 6-12 evaluations vs 50-100 for grid search.
"""

from __future__ import annotations

from .kde import kde_bandwidth, lscv
from .nw import loocv_mse, nw_bandwidth
from ._kernels import _KERNELS

__all__ = ["kde_bandwidth", "nw_bandwidth", "lscv", "loocv_mse", "_KERNELS"]
__version__ = "0.1.0"
