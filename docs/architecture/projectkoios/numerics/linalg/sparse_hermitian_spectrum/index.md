# Sparse partial Hermitian spectra

`projectkoios.numerics.linalg.sparse_hermitian_spectrum` owns unit-free selection of a
strict subset of eigenvalues from a finite sparse Hermitian matrix.

- The matrix contract uses canonical compressed sparse row storage.
- The selection rule and requested count are explicit.
- The public result contains eigenvalues and residual evidence but no eigenvectors.

## Mathematical operation

The capability returns $k<n$ selected eigenvalues of a represented-Hermitian sparse
matrix $H\in\mathbb{C}^{n\times n}$.

- Selection supports smallest or largest algebraic values.
- Selection supports smallest or largest magnitudes.
- Returned eigenvalues use ascending algebraic order regardless of selection rule.
- The represented matrix is $H=(A+A^\dagger)/2$ after Hermiticity qualification.
- The result reports the maximum Euclidean eigenpair residual computed from backend-local
  eigenvectors.

## Public API

The public vocabulary separates dependency-free sparse records from the SciPy ARPACK
implementation.

- Matrix record: `SparseComplexCsrMatrix`.
- Selection enum: `SparseHermitianSpectrumSelection`.
- Hermiticity policy: `BatchedHermitianSpectrumTolerance`, shared with dense batch
  qualification because the represented-input formula is identical.
- Definition: `SparseHermitianPartialSpectrumDefinition`.
- Result: `SparseHermitianPartialSpectrumResult`.
- Nominal template: `SparseHermitianPartialSpectrumSolver`.
- SciPy adapter: `scipy_sparse_hermitian_csr_matrix`.
- SciPy backend: `ScipySparseHermitianPartialSpectrumSolver`.

## Boundary

The sparse capability is distinct from dense complete-spectrum evaluation because its
output and convergence contract are different.

- It requires `eigenvalue_count < dimension`.
- It owns explicit iterative tolerance, iteration limit, Krylov dimension, and seeded
  start-vector controls.
- It raises `RuntimeError` when ARPACK does not converge.
- It does not claim a complete spectrum, return public eigenvectors, or infer scientific
  acceptance from the residual.
- Consumers choose whether a partial spectrum is mathematically sufficient.

## Pages

The child pages document numerical semantics, implementation, and control flow.

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
