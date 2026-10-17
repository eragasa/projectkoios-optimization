# NumpyBatchedComplexUnitaryProcrustesDefinition

`NumpyBatchedComplexUnitaryProcrustesDefinition` is the immutable input owner for one
shape-preserving batch of right-unitary Procrustes problems.

## Contract

- Source and target must be exact `numpy.ndarray` objects with exact
  `numpy.complex128` dtype.
- Both arrays have identical shape `(..., m, n)`, at least one leading batch dimension,
  and positive `m` and `n`.
- Values are finite.
- The tolerance has the exact owner record type.
- Construction makes immutable byte-backed defensive copies.

The record derives `batch_shape`, `batch_size`, `row_count`, and `column_count` without
materializing per-item Python records. Zero-sized leading dimensions are admitted.

## Failures and verification

Wrong nominal types or dtypes raise `TypeError`; invalid shapes or nonfinite values raise
`ValueError`. Dedicated unit tests cover shape preservation, immutable ownership,
multidimensional and empty batches, exact-type rejection, shape rejection, and
nonfinite rejection.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
