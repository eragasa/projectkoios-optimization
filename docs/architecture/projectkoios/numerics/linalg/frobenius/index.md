# Complex-matrix Frobenius values

`projectkoios.numerics.linalg.frobenius` owns the dependency-free unit-free reduction

$$
P_F(A)=\sum_{i,j}|A_{ij}|^2,
\qquad
\lVert A\rVert_F=\sqrt{P_F(A)}
$$

for one nonempty finite rectangular complex matrix.

## Public API

- `ComplexMatrixFrobenius`
- `complex_matrix_frobenius(matrix)`

The input annotation uses `FiniteComplexMatrix` from its owning
`projectkoios.numerics.typing` module; the Frobenius module does not re-export that shared
representation alias. The immutable result retains the row count, column count, raw
Frobenius power, and nonnegative Frobenius norm. Exact built-in tuples and complex values
keep the contract independent of optional array libraries.

## Boundary

This capability owns only the generic arithmetic and binary64 failure behavior. It does
not own physical units, block identities, lattice displacement labels, basis meaning,
normalization, thresholds, or interpretation. Consumers adapt the unit-free output to
their own semantic records.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
