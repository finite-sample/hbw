# hbw

[![PyPI](https://img.shields.io/pypi/v/hbw)](https://pypi.org/project/hbw/)
[![CI](https://github.com/finite-sample/hbw/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/hbw/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/hbw)](https://pepy.tech/project/hbw)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://finite-sample.github.io/hbw)

Fast kernel bandwidth selection via analytic Hessian Newton optimization. **16× faster for KDE, 25-49× faster for NW regression** at n≥10,000.

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

# Multivariate KDE (2D example)
from hbw import kde_bandwidth_mv
X = np.random.randn(500, 2)
h = kde_bandwidth_mv(X)
print(f"Optimal 2D KDE bandwidth: {h:.4f}")

# Multivariate NW regression (2D predictors)
from hbw import nw_bandwidth_mv
X = np.random.randn(500, 2)
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.3 * np.random.randn(500)
h = nw_bandwidth_mv(X, y)
print(f"Optimal 2D NW bandwidth: {h:.4f}")
```

## When to Use Data-Driven Bandwidth Selection

Silverman's rule-of-thumb assumes your data is unimodal and roughly Gaussian. Use `hbw` when:

- **Multimodal distributions**: Silverman oversmooths multiple peaks into a single blob. LSCV adapts to reveal distinct modes.
- **Non-Gaussian data**: Heavy tails or skewness cause Silverman to choose suboptimal bandwidths. Cross-validation optimizes for your actual data shape.
- **Regression (Nadaraya-Watson)**: Silverman's rule is designed for density estimation, not the x-y relationship in regression. NW bandwidth selection requires data-driven LOOCV.
- **Bootstrap/uncertainty quantification**: Each resample has different structure; rule-of-thumb bandwidths don't adapt. CV selection per resample is critical—and with Newton optimization, now practical.

## API Reference

### `kde_bandwidth(x, kernel="gauss", h0=None, max_n=5000, seed=None)`

Select optimal KDE bandwidth via LSCV minimization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like | Sample data |
| `kernel` | str | `"gauss"`, `"epan"`, `"unif"`, `"biweight"`, `"triweight"`, or `"cosine"` |
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
| `kernel` | str | `"gauss"`, `"epan"`, `"unif"`, `"biweight"`, `"triweight"`, or `"cosine"` |
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

### `kde_bandwidth_mv(data, kernel="gauss", h0=None, max_n=3000, seed=None, standardize=True)`

Select optimal multivariate KDE bandwidth via LSCV minimization with product kernel.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | Sample data, shape (n, d) |
| `kernel` | str | `"gauss"`, `"epan"`, `"unif"`, `"biweight"`, `"triweight"`, or `"cosine"` |
| `h0` | float | Initial bandwidth (default: Scott's rule) |
| `max_n` | int | Subsample size for large data |
| `seed` | int | Random seed |
| `standardize` | bool | Standardize each dimension to unit variance |

**Returns:** `float` - optimal isotropic bandwidth

### `lscv_mv(data, h, kernel="gauss")`

Compute LSCV score, gradient, and Hessian for multivariate KDE.

**Returns:** `tuple[float, float, float]` - (score, gradient, hessian)

### `nw_bandwidth_mv(data, y, kernel="gauss", h0=None, max_n=3000, seed=None, standardize=True)`

Select optimal multivariate NW bandwidth via LOOCV-MSE minimization with product kernel.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | Predictor values, shape (n, d) |
| `y` | array-like | Response values |
| `kernel` | str | `"gauss"`, `"epan"`, `"unif"`, `"biweight"`, `"triweight"`, or `"cosine"` |
| `h0` | float | Initial bandwidth (default: Scott's rule) |
| `max_n` | int | Subsample size for large data |
| `seed` | int | Random seed |
| `standardize` | bool | Standardize each predictor dimension to unit variance |

**Returns:** `float` - optimal isotropic bandwidth

### `loocv_mse_mv(data, y, h, kernel="gauss")`

Compute LOOCV-MSE, gradient, and Hessian for multivariate NW regression.

**Returns:** `tuple[float, float, float]` - (loss, gradient, hessian)

## How It Works

**Problem:** Cross-validation bandwidth selection requires O(n²) per evaluation. Grid search needs 50-100 evaluations.

**Solution:** We derive closed-form gradients *and* Hessians for the LSCV (KDE) and LOOCV-MSE (NW) objectives. Newton optimization converges in 6-12 iterations, but the key insight is that each iteration shares O(n²) pairwise computations across objective, gradient, and Hessian—yielding speedups that grow with sample size (16× for KDE, 25-49× for NW at n≥10,000).

**Supported kernels:**
- Gaussian: `K(u) = exp(-u²/2) / √(2π)`
- Epanechnikov: `K(u) = 0.75(1-u²)` for |u| ≤ 1
- Uniform: `K(u) = 0.5` for |u| ≤ 1
- Biweight: `K(u) = (15/16)(1-u²)²` for |u| ≤ 1
- Triweight: `K(u) = (35/32)(1-u²)³` for |u| ≤ 1
- Cosine: `K(u) = (π/4)cos(πu/2)` for |u| ≤ 1

For full mathematical details, see the [paper](ms/).

## Results

Newton-Armijo with analytic Hessian achieves identical accuracy to grid search with speedups that grow with sample size. All implementations use Numba with parallel execution.

**Speedup vs. Grid Search (Gaussian kernel):**
| n | KDE Speedup | NW Speedup |
|---|-------------|------------|
| 100 | 0.3× | 1.1× |
| 500 | 4.2× | 7.8× |
| 2,000 | 7.9× | 28× |
| 5,000 | 8.7× | 26× |
| 10,000 | 16× | 25× |
| 20,000 | — | 49× |

**Bootstrap use case**: 200 resamples at n=10,000 takes ~100 minutes with grid search vs. ~4 minutes with Newton.

Tested across sample sizes, noise levels, four DGPs (bimodal, unimodal, skewed, heavy-tailed), and all six kernels. See [ms/](ms/) for full details.

## Limitations

- **Multivariate KDE/NW**: Isotropic bandwidth only (same h in all dimensions)
- **Not supported**: Anisotropic bandwidth (dimension-specific bandwidths), local/adaptive bandwidth selection

## Citation

```bibtex
@misc{hbw2025,
  author = {Sood, Gaurav},
  title = {Analytic-Hessian Bandwidth Selection for Kernel Density Estimation and Nadaraya-Watson Regression},
  year = {2025},
  url = {https://github.com/finite-sample/hbw}
}
```

## License

MIT
