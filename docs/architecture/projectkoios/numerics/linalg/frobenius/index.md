# Matrix Frobenius values

`projectkoios.numerics.linalg.frobenius` owns the dependency-free unit-free reduction

$$
P_F(A)=\sum_{i,j}|A_{ij}|^2,
\qquad
\lVert A\rVert_F=\sqrt{P_F(A)}
$$

for one nonempty finite rectangular complex matrix.

## Public API

Dependency-free values:

- `ComplexMatrixFrobenius`
- `complex_matrix_frobenius(matrix)`

Optional NumPy dense evaluation:

- `numpy_matrix_frobenius(matrix)`
- `NumpyFrobeniusNorm`

The dependency-free input annotation uses `FiniteComplexMatrix` from its owning
`projectkoios.numerics.typing` module; the Frobenius module does not re-export that shared
representation alias. The NumPy operation accepts exact finite nonempty two-dimensional
float64 or complex128 arrays directly without mutation or dtype conversion. Both paths
return the same immutable detailed result retaining row count, column count, raw
Frobenius power, and nonnegative Frobenius norm. The nominal evaluator returns only the
built-in scalar norm through `MatrixNormABC`.

## Boundary

This capability owns only representation validation, generic arithmetic, and binary64
failure behavior. It does not own physical units, block identities, lattice displacement
labels, basis meaning, normalization, thresholds, or interpretation. Consumers adapt
the unit-free output to their own semantic records.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
