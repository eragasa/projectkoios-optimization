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

Optional backend scalar evaluation:

- `NumpyFrobeniusPower(matrix).result()` and
  `NumpyFrobeniusNorm(matrix).result()` for dense NumPy arrays;
- `ScipyCsrFrobeniusPower(matrix).result()` and
  `ScipyCsrFrobeniusNorm(matrix).result()` for SciPy CSR matrices and arrays.

The dependency-free input annotation uses `FiniteComplexMatrix` from its owning
`projectkoios.numerics.typing` module; the Frobenius module does not re-export that shared
representation alias. The NumPy evaluations accept exact finite nonempty two-dimensional
float64 or complex128 arrays directly without mutation or dtype conversion. The SciPy
evaluations accept the corresponding CSR matrix and array representations, canonicalize
a private copy, and never densify or mutate the caller object. Backend power and norm
capabilities return separate built-in scalar results. The dependency-free tuple operation
continues to return the immutable detailed record with row count, column count, power,
and norm. Nominal norm evaluations return through `MatrixNormABC.result()`.

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
