# Linear algebra numerics

`projectkoios.numerics.linalg` owns unit-free matrix decompositions, invariant spectral
records, and matrix reductions that do not select optimizer arguments.

- Current capabilities include invariant Hermitian subspaces, dense batched complete
  Hermitian spectra, sparse partial Hermitian spectra, Frobenius values, dense
  maximum-absolute-entry and spectral norms, and the shared nominal matrix-norm contract
  foundation.
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
- The dependency-free semantic matrix-norm enum and nominal `result()` boundary are
  implemented, together with dense NumPy and SciPy CSR Frobenius and
  maximum-absolute-entry evaluation and dense NumPy spectral-norm evaluation.
- Frame-producing eigensystems, orthonormalization, sparse spectral norms, additional
  norm semantics, and compression remain prospective.

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
- `matrix_norm.py` owns `MatrixNormEnum` and the generic nominal `MatrixNormABC`
  `result()` boundary; capability-local NumPy and SciPy modules provide concrete
  evaluations.
- `numpy/maximum_absolute_entry.py` and `scipy/maximum_absolute_entry.py` own direct dense
  and CSR maximum-entry evaluation.
- `numpy/spectral_norm.py` owns complete dense largest-singular-value evaluation; sparse
  spectral evaluation remains prospective pending a separate iterative contract.
- `orthonormalization.py` would own symmetric Gram inverse-square-root construction and
  common-space injection; it remains planned.
- `compression.py` would own projector-based compression and embedding; it remains
  planned.
- Each additional dense or sparse matrix-norm evaluator requires capability-specific
  review; semantic enum membership alone does not establish implementation.
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
- Frobenius values: `ComplexMatrixFrobenius` and `complex_matrix_frobenius`; the shared
  `FiniteComplexMatrix` representation remains owned and exported only by
  `projectkoios.numerics.typing`.
- Matrix-norm foundation: `MatrixNormEnum` and `MatrixNormABC`.
- Dense Frobenius execution: `NumpyFrobeniusPower` and `NumpyFrobeniusNorm`.
- CSR Frobenius execution: `ScipyCsrFrobeniusPower` and
  `ScipyCsrFrobeniusNorm`.
- Dense maximum-entry execution: `NumpyMaximumAbsoluteEntryNorm`.
- CSR maximum-entry execution: `ScipyCsrMaximumAbsoluteEntryNorm`.
- Dense spectral-norm execution: `NumpySpectralNorm`.

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
- [Matrix Frobenius values](frobenius/index.md)
- [Maximum-absolute-entry matrix norm](maximum_absolute_entry/index.md)
- [Dense spectral norm](spectral_norm/index.md)
- [Matrix-norm contract foundation](matrix_norm/index.md)
- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Numerics namespace](../index.md)
