"""Fast kernel bandwidth selection via analytic Hessian Newton optimization.

This module provides optimal bandwidth selection for:
- Univariate kernel density estimation (KDE) via LSCV minimization
- Nadaraya-Watson regression via LOOCV-MSE minimization

The key innovation is using closed-form analytic gradients and Hessians
of the cross-validation objectives, enabling Newton optimization that
converges in 6-12 evaluations vs 50-100 for grid search.
"""

from importlib.metadata import version

from ._kernels import _KERNELS
from .kde import (
    kde_bandwidth,
    kde_bandwidth_mv,
    kde_evaluate,
    kde_evaluate_mv,
    lscv,
    lscv_grad,
    lscv_mv,
    lscv_score,
)
from .nw import (
    loocv_mse,
    loocv_mse_grad,
    loocv_mse_mv,
    loocv_mse_score,
    nw_bandwidth,
    nw_bandwidth_mv,
    nw_predict,
    nw_predict_mv,
)

__all__ = [
    "_KERNELS",
    "kde_bandwidth",
    "kde_bandwidth_mv",
    "kde_evaluate",
    "kde_evaluate_mv",
    "loocv_mse",
    "loocv_mse_grad",
    "loocv_mse_mv",
    "loocv_mse_score",
    "lscv",
    "lscv_grad",
    "lscv_mv",
    "lscv_score",
    "nw_bandwidth",
    "nw_bandwidth_mv",
    "nw_predict",
    "nw_predict_mv",
]
__version__ = version("hbw")
