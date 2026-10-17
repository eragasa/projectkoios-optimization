# NumpyBatchedComplexUnitaryProcrustesResult

`NumpyBatchedComplexUnitaryProcrustesResult` is the immutable evidence owner for a
completed batch. It validates stored evidence independently instead of trusting solver
assembly.

## Contract

- `from_arrays(...)` normalizes all public arrays to exact dtypes and immutable
  byte-backed copies.
- `from_immutable_arrays(...)` retains only canonical immutable arrays by identity.
- The raw frozen constructor performs relational validation without replacing fields.
- Shapes, nonnegative ordered spectra, additive rank cutoffs, numerical ranks,
  `full_rank`, powers, reconstructed residuals, roundoff cutoffs, and right-unitarity
  must agree.
- Nonempty results require coupled backend identifier and version evidence; empty
  results forbid backend claims.
- `all_full_rank`, `rank_deficient_count`, and `first_rank_deficient_index` derive only
  from immutable rank status.
- `require_unique_right_unitary_transformations()` returns the existing completion array
  only when every item is full rank and otherwise raises at the first deficient C-order
  index.

Rank-deficient completions remain available as diagnostics, but the result never labels
them unique transformations.

## Failures and verification

`TypeError` covers wrong nominal fields or dtypes. `ValueError` covers inconsistent
shape, rank, residual, completion, or backend evidence. Numerical reconstruction may
raise `FloatingPointError`. Dedicated tests exercise immutable backing, every relational
invariant family, multidimensional rank summaries, empty semantics, fail-closed unique
access, copying and zero-copy construction, canonical-storage rejection, and absence of
construction-time field replacement.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
