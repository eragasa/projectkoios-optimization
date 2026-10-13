# Weighted affine residuals

## Purpose

`WeightedAffineResidualTerm` and `WeightedAffineResidualDefinition` are immutable,
unit-free core contracts for

$$
L(x)=\sum_i w_i(a_i^T x-b_i)^2, \qquad w_i>0.
$$

Every term has the same positive coefficient dimension, and tuple order defines the
coordinate convention. Weights are used exactly as supplied. Scientific coordinate
names, channel alignment, sample normalization, units, meaning, and acceptance
thresholds remain caller-owned.

## Direct evaluation

`WeightedAffineResidualDefinition.squared_loss()` validates exact tuple inputs, finite
non-Boolean real coordinates, and dimension. Scalar products and sums fail closed on
nonfinite binary64 results. Weighted products use square-root rescaling when a direct
multiplication order would prematurely overflow or underflow.

## Quadratic construction

`affine/quadratic.py` publicly owns checked `weighted_design()` and
`weighted_normal_matrix()` construction plus the lazy
`WeightedAffineResidualQuadraticConstructor` backend selector. Importing this module
does not import NumPy or SciPy.

`affine/numpy/affine.py` applies `numpy.linalg.lstsq(..., rcond=None)` directly to the
weighted design and returns a dense quadratic. `affine/scipy/affine.py` applies
`scipy.linalg.lstsq(..., cond=None, lapack_driver="gelsd")` and returns a SciPy sparse
quadratic. Both require reported numerical full column rank and do not regularize a
rejected system.

After rank acceptance, construction uses the exactly mirrored binary64 normal matrix
and delegates final represented positive-definiteness, storage, and evaluation to the
quadratic capability. Rank and positive-definiteness acceptance are backend numerical
decisions, not exact proofs, outward-rounded certificates, conditioning guarantees, or
scientific validation.

## Exclusions

This capability does not own physical dimensions, normalization policy, box
feasibility, support optimization, witness or set-distance search, interval arithmetic,
directed rounding, certificate interpretation, or scientific disposition.
