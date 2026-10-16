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

## Dense reduction

Channel evaluation uses one backend-owned dense matrix multiplication after the phase
matrix is formed once.

- The phase matrix has shape `P × M`.
- Transposed coefficient channels have shape `M × S`.
- The result has point-major shape `P × S`.
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
