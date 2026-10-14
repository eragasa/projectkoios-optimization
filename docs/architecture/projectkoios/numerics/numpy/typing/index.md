# NumPy typing

## Status

This page specifies the current shared typing boundary for concrete optional NumPy
capabilities.

## Ownership

`projectkoios.numerics.numpy.typing` owns reusable aliases for arrays whose scalar dtype
is part of a numerical implementation contract. It does not define mathematical problem
records, array-shape validation, backend selection, or scientific semantics.

The module currently exports:

- `Float64Array`;
- `Complex128Array`;
- `Int64Array`;
- `IntpArray`;
- `BoolArray`; and
- `NumpyArrayLike`.

The names specify dtype, not rank or shape. Each consuming capability remains responsible
for checking its required dimension and shape at runtime.

## Optional-dependency boundary

Importing `projectkoios.numerics.numpy` alone imports no backend. Importing its `typing`
module explicitly imports NumPy, so dependency-free public contracts continue to use
built-in immutable records. Concrete NumPy implementations and tests may share these
aliases rather than spelling broad `numpy.ndarray`, explicit `Any` dimensions, or local
dtype aliases repeatedly.

## Verification

Mapped integration tests type-check representative arrays of every exported concrete
dtype and verify their runtime dtypes. Strict mypy includes the mapped NumPy typing tests.
