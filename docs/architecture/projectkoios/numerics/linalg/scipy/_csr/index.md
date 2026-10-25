# `projectkoios.numerics.linalg.scipy._csr`

The private source module
`src/python/projectkoios/numerics/linalg/scipy/_csr.py` owns one shared CSR representation
boundary for matrix-norm implementations.

## Operation

`_canonical_binary64_csr(matrix)`:

1. requires a SciPy `csr_matrix` or `csr_array`;
2. requires exact `float64` or `complex128` dtype;
3. requires nonempty two-dimensional shape;
4. requires finite stored source entries;
5. copies the caller representation;
6. consolidates duplicate coordinates under local floating-warning control;
7. requires finite consolidated represented entries;
8. removes explicit zeros; and
9. sorts column indices.

The returned object is independent canonical CSR storage. Every mutation applies to the
copy. Duplicate overflow is arithmetic failure and raises `FloatingPointError`.

## Consumers

- `ScipyCsrFrobeniusPower`
- `ScipyCsrMaximumAbsoluteEntryNorm`
- `ScipyCsrSpectralNorm`

The operation is private and has no package export, class wrapper, cache, dtype coercion,
or sparse-format conversion.
