# Least-squares optimization

`projectkoios.optimization.least_squares` owns reusable problems and solvers that select
coefficients or matrices by minimizing declared residual functions.

## Modules

| Module | Owned behavior |
|---|---|
| `affine.py` | real weighted affine residual definitions and completed-square construction |
| `linear.py` | ordinary and positive-weighted real/complex linear least squares with one or multiple right-hand sides |
| `matrix_subspaces.py` | real fitting over an ordered matrix basis |
| `numpy/*` | dense implementations |
| `scipy/*` | sparse or SciPy-specific implementations |

## Public vocabulary

- `WeightedAffineResidualDefinition`
- `WeightedAffineResidualQuadraticConstructor`
- `LinearLeastSquaresTolerance`
- `LinearLeastSquaresResult`
- `LinearLeastSquaresSolver`
- `LinearLeastSquaresMethod`
- `LinearLeastSquaresSolverFactory`
- `NumpyLinearLeastSquaresSolver`
- `RealMatrixSubspaceLeastSquaresResult`
- `NumpyRealMatrixSubspaceLeastSquaresSolver`

## Boundary

Numerics owns decompositions, norms, and conditioning calculations. Least Squares owns
the residual objective, coefficient selection, fitted values, and residual diagnostics.
Consumers own design-column meaning, units, model-class labels, and scientific adequacy.

## Current state

Ordinary and positive-weight real/complex linear least squares with one or multiple
right-hand sides is implemented under `projectkoios.optimization.least_squares`. Real
weighted-affine construction remains implemented under
`projectkoios.optimization.affine`; it moves behind a compatibility path only after a
separate identity review. Matrix-subspace fitting is not implemented.

## Pages

- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Optimization namespace](../index.md)
