# Batched Procrustes numerical-evidence ownership

The `numerics.py` module owns pure Procrustes-specific reductions shared by contract
validation and solver assembly. It does not call SVD or expose names through the public
façade.

## Owned class

- [`ResidualReconstruction`](ResidualReconstruction/index.md) reconstructs optimal
  squared residuals and their squared-unit roundoff cutoffs from powers and complete
  singular spectra.

## Supporting operations

The module also owns exact batch-size reduction, deterministic C-order index conversion,
first-failure indexing, and dimension-scaled validation bounds. These operations have no
scientific-policy meaning and import neither PhysKit nor KSD.
