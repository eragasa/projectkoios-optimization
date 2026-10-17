# Shared-grid batched finite Fourier evaluation

This capability evaluates ordered coefficient channels over one common coordinate and
integer-mode grid while preserving a dependency-free public boundary.

- Public values use point-major shape `P × S`.
- Channel column order exactly matches coefficient-channel order.
- The positive exponential and absence of normalization are fixed.
- The NumPy backend reuses one point-batch-by-mode phase matrix and bounds large
  evaluations to one million phase elements per batch.

## Mathematical operation

For coordinates $x_i$, integer modes $m_j$, and coefficient channels $c_{a,j}$, the
operation returns

$$
y_{i,a}=\sum_{j=0}^{M-1} c_{a,j}\exp\!\left(+2\pi i\,x_i\cdot m_j\right).
$$

- Coordinates, modes, and channels retain caller order.
- Duplicate integer modes remain distinct columns.
- All channels have the same ordered mode inventory.
- No coordinate, mode, or coefficient inference occurs.

## Public vocabulary

The names describe a distinct batched operation rather than changing the exact ordered
scalar finite-series contract.

- `BatchedFiniteFourierSeriesEvaluationDefinition` records the shared grid and channels.
- `BatchedFiniteFourierSeriesEvaluationResult` records point-major values and evidence.
- `BatchedFiniteFourierSeriesEvaluator` is the nominal backend boundary.
- `NumpyBatchedFiniteFourierSeriesEvaluator` owns dense NumPy execution.

## Ownership boundary

Numerics owns only unit-free finite-periodic arithmetic and numerical evidence.

- PhysKit may map hopping-model entries to coefficient channels and restore units.
- PhysKit owns lattices, displacements, truncation inventories, models, and bands.
- KSD owns scientific meshes, cutoffs, errors, thresholds, and interpretation.
- This capability owns no caching, persistent phase plan, optimization, or physics API.

## Distinction from scalar evaluation

The existing scalar evaluator retains its reviewed observable reduction order; this
batched capability instead permits backend-owned dense channel reduction.

- Both operations use the same positive exponent and no normalization.
- Coordinate-component accumulation remains in declared component order.
- Dense mode reduction may differ from scalar accumulation by binary64 rounding.
- Consumers requiring exact scalar reduction continue using the scalar evaluator.

## Pages

The supporting pages state implementation, numerical, and flow details.

- [Implementation organization](implementation.md)
- [Numerical contract](numerics.md)
- [Schematics](schematics.md)
- [Ordered scalar finite-series evaluation](../finite_series/index.md)
