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

The enum defines three matrix-norm semantic identities. Capability-local NumPy and SciPy
modules implement dense and CSR Frobenius norms and maximum-entry norms. The NumPy
spectral-norm module implements complete dense largest-singular-value evaluation. No
sparse spectral-norm implementation exists.

The foundation supplies no factory, dispatcher, unit handling, or scientific threshold
policy. Each capability-local operation module owns its numerical behavior.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
