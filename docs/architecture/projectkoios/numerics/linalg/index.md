# Linear algebra numerics

`projectkoios.numerics.linalg` owns unit-free matrix decompositions, invariant
spectral subspaces, orthonormalization, projectors, compression, and matrix norms that do
not select an optimizer argument through their public contract.

## Status

The invariant-only dense Hermitian spectral-subspace capability is implemented and
independently qualified. Dependency-free finite complex-matrix Frobenius power and norm
evaluation and batched complete Hermitian-spectrum evaluation are also implemented.
Frame-producing eigensystems, orthonormalization, other matrix norms, and compression
remain prospective and require separate capability-level review.

## Capability inventory

| Proposed module | Owned behavior | Status |
|---|---|---|
| `hermitian_spectral_subspace.py` | complete ascending Hermitian spectrum and projector for a caller-selected contiguous index range whose present boundaries pass a declared cutoff policy | implemented and independently qualified locally |
| `hermitian_spectrum.py` | batched complete ascending spectra of explicit Hermitian representatives with per-matrix and aggregate input diagnostics | implemented and locally qualified |
| `orthonormalization.py` | symmetric Gram inverse-square-root construction and common-space injection | planned |
| `compression.py` | projector-based compression and embedding of supplied operators | planned |
| `frobenius.py` | finite complex-matrix Frobenius power and norm | implemented |
| other matrix norms | operator-2, maximum-entry, and reference-normalized norms | planned |
| `numpy/*` | dense concrete implementations | capability-specific review required |

A generic `eigensystems.py` module is not introduced merely to hold unrelated spectral
behaviors. Eigenframes, generalized problems, sparse partial spectra, and non-Hermitian
decompositions require distinct contracts.

## First proposed public vocabulary

- `HermitianSpectralMatrix`
- `HermitianSpectralSubspaceTolerance`
- `HermitianSpectralSubspaceDefinition`
- `HermitianSpectralSubspaceDiagnostics`
- `HermitianSpectralSubspaceResult`
- `HermitianSpectralSubspaceSolver`
- `HermitianSpectralSubspaceMethod`
- `HermitianSpectralSubspaceSolverFactory`
- `NumpyHermitianSpectralSubspaceSolver`
- `BatchedHermitianSpectrumDefinition`
- `BatchedHermitianSpectrumTolerance`
- `HermitianSpectrumMatrixDiagnostics`
- `BatchedHermitianSpectrumDiagnostics`
- `BatchedHermitianSpectrumResult`
- `BatchedHermitianSpectrumSolver`
- `NumpyBatchedHermitianSpectrumSolver`
- `FiniteComplexMatrix`
- `ComplexMatrixFrobenius`
- `complex_matrix_frobenius`

## Frame nonuniqueness boundary

The first spectral slice returns eigenvalues and an orthogonal spectral projector, not
eigenvectors. A selected-subspace frame $F$ is nonunique because $FU$ spans the same
subspace for every selected-space unitary $U$. Exact degeneracy therefore does not admit
a canonical eigenbasis without an additional coordinate-dependent policy.

Frame-producing extraction is blocked until spectral clustering, phase fixing,
degenerate-block completion, coordinate dependence, and failure behavior are reviewed.
KSD must separately state whether M2 requires eigenvector-diagonal frames or only an
orthonormal frame spanning its retained subspace.

## Optimization boundary

This capability decomposes and transforms supplied matrices. It does not fit parameters
or select a transformation by minimizing an objective. Least squares and Procrustes
therefore belong in `projectkoios.optimization`; optimized transport remains under
`projectkoios.optimization.alignment`.

## Academic basis

The spectral-subspace pages include an annotated, inspected bibliography in which every
academic reference has a DOI or valid ISBN. The references cover the spectral theorem,
spectral projectors, invariant-subspace perturbation, dense Hermitian eigensystems,
finite-precision error analysis, and LAPACK behavior. They support terminology and
algorithmic rationale but do not replace ProjectKoios contracts.

## Pages

- [Hermitian spectral-subspace decomposition](hermitian_spectral_subspace/index.md)
- [Batched complete Hermitian spectra](hermitian_spectrum/index.md)
- [Complex-matrix Frobenius values](frobenius/index.md)
- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
