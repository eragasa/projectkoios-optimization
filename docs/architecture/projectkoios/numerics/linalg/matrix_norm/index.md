# Matrix-norm contract foundation

`projectkoios.numerics.linalg.matrix_norm` owns the dependency-free semantic vocabulary
and nominal construction-plus-`result()` boundary shared by matrix-norm implementations.

## Public API

- `MatrixNormEnum`
- `MatrixNormABC[MatrixT]`

`MatrixNormEnum` is a pure `Enum` with the semantic members `frobenius`, `spectral`, and
`maximum_absolute_entry`. Its opaque values are not display labels or serialization
identifiers. Consumers reuse the owner enum and maintain any presentation mapping at
their own facade boundary.

`MatrixNormABC` is generic only in the concrete matrix representation accepted by a
subclass. Construction retains the matrix and its final `result()` boundary returns an
exact built-in finite nonnegative float; representation validation and numerical
evaluation remain subclass responsibilities.

## Status boundary

The enum names the reviewed matrix-norm semantics without claiming that every norm has a
concrete evaluator. `projectkoios.numerics.linalg.numpy.frobenius` and
`projectkoios.numerics.linalg.scipy.frobenius` supply the current dense and CSR Frobenius
implementations. Capability-local NumPy and SciPy modules supply the current dense and
CSR maximum-entry implementations. The foundation supplies no factory, dispatcher, unit
handling, or scientific threshold policy. Detailed numerical behavior
remains owned by each capability-local operation module.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
