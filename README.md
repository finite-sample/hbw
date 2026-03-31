# hbw

[![PyPI](https://img.shields.io/pypi/v/hbw)](https://pypi.org/project/hbw/)
[![CI](https://github.com/soodoku/analytic-bw/actions/workflows/ci.yml/badge.svg)](https://github.com/soodoku/analytic-bw/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast kernel bandwidth selection via analytic Hessian Newton optimization.

## Installation

```bash
pip install hbw
```

## Quick Start

```python
import numpy as np
from hbw import kde_bandwidth, nw_bandwidth

# KDE bandwidth selection
x = np.random.randn(1000)
h = kde_bandwidth(x)
print(f"Optimal KDE bandwidth: {h:.4f}")

# Nadaraya-Watson regression bandwidth
x = np.linspace(-2, 2, 500)
y = np.sin(2 * x) + 0.3 * np.random.randn(len(x))
h = nw_bandwidth(x, y)
print(f"Optimal NW bandwidth: {h:.4f}")

# Large datasets: automatic subsampling
x_large = np.random.randn(100_000)
h = kde_bandwidth(x_large, max_n=5000, seed=42)  # Uses 5000 random points
```

## API Reference

### `kde_bandwidth(x, kernel="gauss", h0=None, max_n=5000, seed=None)`

Select optimal KDE bandwidth via LSCV minimization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like | Sample data |
| `kernel` | str | `"gauss"` or `"epan"` (Epanechnikov) |
| `h0` | float | Initial bandwidth (default: Silverman's rule) |
| `max_n` | int | Subsample size for large data (None to disable) |
| `seed` | int | Random seed for reproducible subsampling |

**Returns:** `float` - optimal bandwidth

### `nw_bandwidth(x, y, kernel="gauss", h0=None, max_n=5000, seed=None)`

Select optimal Nadaraya-Watson bandwidth via LOOCV-MSE minimization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like | Predictor values |
| `y` | array-like | Response values |
| `kernel` | str | `"gauss"` or `"epan"` |
| `h0` | float | Initial bandwidth (default: Silverman's rule) |
| `max_n` | int | Subsample size for large data |
| `seed` | int | Random seed |

**Returns:** `float` - optimal bandwidth

### `lscv(x, h, kernel="gauss")`

Compute LSCV score, gradient, and Hessian for KDE.

**Returns:** `tuple[float, float, float]` - (score, gradient, hessian)

### `loocv_mse(x, y, h, kernel="gauss")`

Compute LOOCV-MSE, gradient, and Hessian for NW regression.

**Returns:** `tuple[float, float, float]` - (loss, gradient, hessian)

## How It Works

**Problem:** Cross-validation bandwidth selection requires O(n²) per evaluation. Grid search needs 50-100 evaluations.

**Solution:** We derive closed-form gradients *and* Hessians for the LSCV (KDE) and LOOCV-MSE (NW) objectives. This enables Newton optimization that converges in 6-12 evaluations—same optimum, 4-10x fewer evaluations.

**Supported kernels:**
- Gaussian: `K(u) = exp(-u²/2) / √(2π)`
- Epanechnikov: `K(u) = 0.75(1-u²)` for |u| ≤ 1

For full mathematical details, see the [paper](ms/).

## Citation

```bibtex
@misc{hbw2024,
  author = {Sood, Gaurav},
  title = {Analytic-Hessian Bandwidth Selection for Kernel Density Estimation and Nadaraya-Watson Regression},
  year = {2024},
  url = {https://github.com/soodoku/analytic-bw}
}
```

## License

MIT
