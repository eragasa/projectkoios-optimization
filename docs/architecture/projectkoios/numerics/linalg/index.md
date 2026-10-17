# Linear algebra numerics

`projectkoios.numerics.linalg` owns unit-free matrix decompositions, invariant spectral
records, and matrix reductions that do not select optimizer arguments.

- Current capabilities include invariant Hermitian subspaces, dense batched complete
  Hermitian spectra, sparse partial Hermitian spectra, and finite complex-matrix
  Frobenius values.
- Prospective capabilities require separate mathematical contracts and qualification.
- Physical semantics and objective-driven selection remain outside this namespace.

## Status

The namespace grows by capability rather than through a universal decomposition base.

- Hermitian spectral-subspace decomposition is implemented and independently qualified.
- Dense batched complete Hermitian-spectrum evaluation is implemented with NumPy and
  SciPy backends.
- Sparse partial Hermitian-spectrum evaluation is implemented with a SciPy ARPACK
  backend.
- Dependency-free finite complex-matrix Frobenius power and norm are implemented.
- Frame-producing eigensystems, orthonormalization, other norms, and compression remain
  prospective.

## Capability inventory

The inventory identifies owned behavior and prevents planned modules from inheriting
acceptance from implemented modules.

- `hermitian_spectral_subspace.py` owns a complete ascending Hermitian spectrum and an
  orthogonal projector for a caller-selected contiguous index range; it is implemented
  and independently qualified.
- `hermitian_spectrum.py` owns batched complete ascending spectra of explicit Hermitian
  representatives with represented-input diagnostics; it is implemented and locally
  qualified.
- `sparse_hermitian_spectrum.py` owns canonical CSR partial-spectrum selection and
  residual evidence; it is implemented with a SciPy sparse backend.
- `frobenius.py` owns finite complex-matrix Frobenius power and norm; it is implemented.
- `orthonormalization.py` would own symmetric Gram inverse-square-root construction and
  common-space injection; it remains planned.
- `compression.py` would own projector-based compression and embedding; it remains
  planned.
- Other matrix norms and each optional backend require capability-specific review.
- A generic `eigensystems.py` module is not introduced merely to collect unrelated
  spectral behaviors.

## Public vocabulary

The public vocabulary reflects separate mathematical problems rather than a universal
solver or certificate hierarchy.

- Hermitian subspace records: `HermitianSpectralMatrix`,
  `HermitianSpectralSubspaceTolerance`, `HermitianSpectralSubspaceDefinition`,
  `HermitianSpectralSubspaceDiagnostics`, and `HermitianSpectralSubspaceResult`.
- Hermitian subspace execution: `HermitianSpectralSubspaceSolver`,
  `HermitianSpectralSubspaceMethod`, `HermitianSpectralSubspaceSolverFactory`, and
  `NumpyHermitianSpectralSubspaceSolver`.
- Batched spectrum records: `BatchedHermitianSpectrumDefinition`,
  `BatchedHermitianSpectrumTolerance`, `HermitianSpectrumMatrixDiagnostics`,
  `BatchedHermitianSpectrumDiagnostics`, and `BatchedHermitianSpectrumResult`.
- Dense batched spectrum execution: `BatchedHermitianSpectrumSolver`,
  `NumpyBatchedHermitianSpectrumSolver`, and `ScipyBatchedHermitianSpectrumSolver`.
- Sparse partial-spectrum records: `SparseComplexCsrMatrix`,
  `SparseHermitianSpectrumSelection`, `SparseHermitianPartialSpectrumDefinition`, and
  `SparseHermitianPartialSpectrumResult`.
- Sparse execution: `SparseHermitianPartialSpectrumSolver`,
  `ScipySparseHermitianPartialSpectrumSolver`, and
  `scipy_sparse_hermitian_csr_matrix`.
- Frobenius values: `FiniteComplexMatrix`, `ComplexMatrixFrobenius`, and
  `complex_matrix_frobenius`.

## Frame nonuniqueness boundary

Invariant subspaces do not determine a unique basis, so the implemented spectral
capabilities avoid implicit coordinate-dependent frame policy.

- A selected frame $F$ and $FU$ span the same subspace for every selected-space unitary
  $U$.
- Exact degeneracy therefore admits no canonical eigenbasis without additional policy.
- The invariant-subspace capability returns a projector rather than public eigenvectors.
- Frame extraction remains blocked pending reviewed clustering, phase fixing,
  degenerate-block completion, coordinate dependence, and failure behavior.
- KSD remains responsible for stating whether a consumer needs eigenvector-diagonal
  frames or only an orthonormal basis for a retained subspace.

## Optimization boundary

Linear algebra numerics transform supplied matrices without selecting transformations by
minimizing objectives.

- Numerics owns decompositions, spectra, projectors, and matrix reductions.
- Least squares and Procrustes selection belong in `projectkoios.optimization`.
- Optimized transport belongs under `projectkoios.optimization.alignment`.

## Academic basis

The invariant spectral-subspace pages maintain an inspected bibliography for terminology
and algorithmic rationale.

- References cover the spectral theorem, spectral projectors, invariant-subspace
  perturbation, dense Hermitian eigensystems, finite-precision error analysis, and LAPACK
  behavior.
- Every academic reference carries a DOI or valid ISBN.
- References support owner contracts but do not replace them.

## Pages

The child pages document implemented capabilities, organization, and namespace-level
schematics.

- [Hermitian spectral-subspace decomposition](hermitian_spectral_subspace/index.md)
- [Batched complete Hermitian spectra](hermitian_spectrum/index.md)
- [Sparse partial Hermitian spectra](sparse_hermitian_spectrum/index.md)
- [Complex-matrix Frobenius values](frobenius/index.md)
- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Numerics namespace](../index.md)
