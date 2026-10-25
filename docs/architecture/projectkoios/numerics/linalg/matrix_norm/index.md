# Matrix-norm contract foundation

`projectkoios.numerics.linalg.matrix_norm` owns the dependency-free semantic vocabulary
and nominal callable boundary shared by matrix-norm implementations.

## Public API

- `MatrixNormEnum`
- `MatrixNormABC[MatrixT]`

`MatrixNormEnum` is a pure `Enum` with the semantic members `frobenius`, `spectral`, and
`maximum_absolute_entry`. Its opaque values are not display labels or serialization
identifiers. Consumers reuse the owner enum and maintain any presentation mapping at
their own facade boundary.

`MatrixNormABC` is generic only in the concrete matrix representation accepted by a
subclass. Its final callable boundary returns an exact built-in finite nonnegative float;
representation validation and numerical evaluation remain subclass responsibilities.

## Status boundary

The enum names the reviewed matrix-norm semantics without claiming that every norm has a
concrete evaluator. `projectkoios.numerics.linalg.numpy.frobenius` supplies the current
dense NumPy Frobenius implementation. The foundation supplies no factory, dispatcher,
unit handling, or scientific threshold policy. Detailed Frobenius values remain owned by
`projectkoios.numerics.linalg.frobenius` and its capability-local backend modules.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
