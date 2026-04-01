# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-04-01

### Added
- `lscv_score()` - Score-only LSCV computation for efficient grid search
- `loocv_mse_score()` - Score-only LOOCV-MSE for NW regression
- Multivariate KDE support via `kde_bandwidth_mv()` and `lscv_mv()`

### Changed
- Optimized `lscv()` with kernel value caching (no redundant computations)
- Optimized Newton-Armijo backtracking to use score-only function
- Improved timing comparisons in benchmarks

### Performance
- Newton backtracking ~2x faster (uses score-only instead of full objective)
- Grid search timing now uses score-only for fair comparison

## [0.1.0] - Initial Release

- Core bandwidth selection: `kde_bandwidth()`, `nw_bandwidth()`
- LSCV and LOOCV-MSE objective functions with analytic gradients/Hessians
- Gaussian, Epanechnikov, and Uniform kernel support
- Newton-Armijo optimization
