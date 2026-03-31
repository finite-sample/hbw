hbw
===

Fast kernel bandwidth selection via analytic Hessian Newton optimization.

**hbw** provides optimal bandwidth selection for:

- Kernel density estimation (KDE) via LSCV minimization
- Nadaraya-Watson regression via LOOCV-MSE minimization

The key innovation is using closed-form analytic gradients and Hessians,
enabling Newton optimization that converges in 6-12 evaluations vs 50-100
for grid search.

Installation
------------

.. code-block:: bash

   pip install hbw

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from hbw import kde_bandwidth, nw_bandwidth

   # KDE bandwidth selection
   x = np.random.randn(1000)
   h = kde_bandwidth(x)

   # Nadaraya-Watson bandwidth selection
   x = np.linspace(-2, 2, 200)
   y = np.sin(x) + 0.1 * np.random.randn(len(x))
   h = nw_bandwidth(x, y)

API Reference
-------------

.. automodule:: hbw
   :members: kde_bandwidth, nw_bandwidth, lscv, loocv_mse
   :undoc-members:
   :show-inheritance:
