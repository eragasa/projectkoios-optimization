# NumPy typing

## Status

This page specifies the current shared typing boundary for concrete optional NumPy
capabilities.

## Ownership

`projectkoios.numerics.numpy.typing` owns reusable aliases for arrays whose scalar dtype
is part of a numerical implementation contract. It does not define mathematical problem
records, array-shape validation, backend selection, or scientific semantics.

The module currently exports:

- `Float64Array` and `Complex128Array` for rank-independent binary64 storage;
- `Float64Matrix` and `Complex128Matrix` for matrix-consuming capability signatures;
- `DenseMatrix`, the closed union of those two matrix aliases;
- `Int64Array`, `IntpArray`, and `BoolArray`; and
- `NumpyArrayLike`.

The dtype-specific names identify static scalar storage. NumPy annotations do not enforce
rank, nonemptiness, exact runtime dtype, or finiteness, so each matrix-consuming
capability validates those invariants at its public boundary. The matrix names communicate
the required semantic rank without claiming static shape enforcement.

## Optional-dependency boundary

Importing `projectkoios.numerics.numpy` alone imports no backend. Importing its `typing`
module explicitly imports NumPy, so dependency-free public contracts continue to use
built-in immutable records. Concrete NumPy implementations and tests may share these
aliases rather than spelling broad `numpy.ndarray`, explicit `Any` dimensions, or local
dtype aliases repeatedly.

## Verification

Mapped integration tests type-check representative arrays and real/complex dense
matrices of every exported concrete dtype and verify their runtime dtypes. Strict mypy
includes the mapped NumPy typing tests.
