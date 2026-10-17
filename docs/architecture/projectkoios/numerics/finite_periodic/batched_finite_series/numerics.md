# Shared-grid batched finite Fourier numerics

The numerical contract fixes representation, phase convention, shape, and failure
semantics while leaving the final dense reduction order to the identified backend.

- Coordinates and modes use binary64 in the NumPy implementation.
- Coefficients, phases, accumulators, and outputs use complex128.
- Integer modes are bounded so conversion to binary64 is exact.
- All public and backend-produced values must be finite.

## Phase convention

Every point-mode phase uses the fixed positive-exponent convention without
normalization.

- The phase is `exp(+2*pi*1j*dot(coordinate, mode))`.
- Coordinate components accumulate in supplied order.
- Mode columns retain supplied order, including duplicates.
- Coefficients are neither normalized nor coalesced.

## Bounded dense reductions

Channel evaluation processes point batches while sharing each batch's phase work across
all channels.

- Every phase matrix has shape `B × M_b` with `B * M_b <= 1_000_000`.
- Corresponding coefficient-channel slices have shape `M_b × S`.
- The complete result has point-major shape `P × S`.
- Problems below the bound use one phase matrix and one multiplication.
- Dense mode reduction may be reassociated by NumPy or its linked numerical libraries.
- Bitwise equality with the scalar ordered evaluator is not promised.

## Numerical verification

Bounded verification compares dense columns with the unchanged scalar evaluator rather
than claiming mathematical exactness.

- Fixtures include real and complex coefficients.
- Fixtures include shifted one-dimensional coordinates.
- Fixtures include seeded multidimensional grids.
- Zero-masked channels represent finite-range consumer families.
- The comparison bound scales with coefficient magnitude, mode count, and binary64 epsilon.

## Failures and evidence

The template and backend fail closed for malformed structure and nonfinite arithmetic.

- Invalid public types and shapes raise `TypeError` or `ValueError`.
- Nonfinite phase or result arithmetic raises `FloatingPointError`.
- Undeclared backend programming failures propagate unchanged.
- Nontrivial results identify `numpy.matmul` and the installed NumPy version.
- Template-resolved empty results carry no fabricated backend evidence.
