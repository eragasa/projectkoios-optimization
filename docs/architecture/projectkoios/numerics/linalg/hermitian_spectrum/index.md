# Batched complete Hermitian spectra

`projectkoios.numerics.linalg.hermitian_spectrum` owns a dependency-free contract for
complete eigenvalue spectra of finite dense complex Hermitian matrix batches.

- The contract is unit-free and backend-neutral.
- The NumPy and SciPy backends remain explicit capability-local imports.
- The result contains spectra and represented-input numerical diagnostics.

## Mathematical operation

The capability evaluates the complete spectrum of an explicit Hermitian representative
for every matrix in the supplied batch.

- Each input is a finite square matrix $A_b\in\mathbb{C}^{n\times n}$.
- The representative is
  $$
  H_b=\frac{A_b+A_b^\dagger}{2}.
  $$
- The result retains all $n$ eigenvalues of $H_b$ in ascending order with algebraic
  multiplicity.
- The operation returns no eigenvectors or basis-dependent frame choices.

## Shape contract

The definition infers a nonempty dimension from canonical immutable matrix storage.

- `BatchedHermitianSpectrumDefinition(matrices, tolerance)` accepts only exact tuple
  containers and exact finite built-in complex entries.
- Every matrix must share the inferred positive square dimension.
- `BatchedHermitianSpectrumDefinition.empty(dimension, tolerance)` names empty-batch
  construction without inventing a zero-dimensional matrix.
- NumPy and SciPy arrays remain transient backend representations rather than immutable
  contract storage.

## Public API

The public vocabulary separates immutable mathematical records from the concrete NumPy
implementation.

- Matrix types: `HermitianSpectrumMatrix`, `HermitianSpectrumMatrices`, and
  `HermitianSpectra`.
- Input records: `BatchedHermitianSpectrumTolerance` and
  `BatchedHermitianSpectrumDefinition`.
- Evidence records: `HermitianSpectrumMatrixDiagnostics` and
  `BatchedHermitianSpectrumDiagnostics`.
- Output record: `BatchedHermitianSpectrumResult`.
- Nominal template: `BatchedHermitianSpectrumSolver`.
- Concrete backends: `NumpyBatchedHermitianSpectrumSolver` and
  `ScipyBatchedHermitianSpectrumSolver`.
- Empty results use `diagnostics is None` because a maximum over no observations is
  undefined.

## Usage

The following example shows canonical nonempty storage and explicit empty construction
without importing NumPy through the contract module.

- A nonempty batch uses the strict definition constructor, while an empty batch uses the
  named empty constructor.
- NumPy and SciPy provide interchangeable dense complete-spectrum implementations.

  ```python
  from projectkoios.numerics.linalg.hermitian_spectrum import (
      BatchedHermitianSpectrumDefinition,
      BatchedHermitianSpectrumTolerance,
  )
  from projectkoios.numerics.linalg.numpy.hermitian_spectrum import (
      NumpyBatchedHermitianSpectrumSolver,
  )
  from projectkoios.numerics.linalg.scipy.hermitian_spectrum import (
      ScipyBatchedHermitianSpectrumSolver,
  )

  policy = BatchedHermitianSpectrumTolerance(
      hermiticity_absolute=1.0e-12,
      hermiticity_relative=1.0e-12,
  )
  solver = NumpyBatchedHermitianSpectrumSolver()
  result = solver.solve(
      BatchedHermitianSpectrumDefinition(
          (
              (
                  (2.0 + 0.0j, 1.0 - 1.0j),
                  (1.0 + 1.0j, 3.0 + 0.0j),
              ),
              (
                  (4.0 + 0.0j, 0.0 + 0.0j),
                  (0.0 + 0.0j, 5.0 + 0.0j),
              ),
          ),
          policy,
      )
  )
  empty = solver.solve(BatchedHermitianSpectrumDefinition.empty(2, policy))
  scipy_result = ScipyBatchedHermitianSpectrumSolver().solve(
      BatchedHermitianSpectrumDefinition(
          (((2.0 + 0.0j, 1.0 - 1.0j), (1.0 + 1.0j, 3.0 + 0.0j)),),
          policy,
      )
  )
  assert result.dimension == 2
  assert empty.diagnostics is None
  assert scipy_result.backend_identifier == "scipy.linalg.eigvalsh"
  ```

## Boundary

The capability owns only reusable matrix numerics and represented-input evidence.

- It owns canonical storage validation, Hermiticity measurement, representative construction, complete
  spectra, and backend-neutral result correlation.
- It does not own eigenvectors, projectors, spectral-gap policy, degeneracy decisions,
  units, lattice or hopping semantics, band identity, mesh meaning, or scientific
  interpretation.
- Exception classes and documented inducing conditions are public behavior; message text
  is diagnostic context rather than compatibility surface.

## Pages

The architecture is divided into focused pages that preserve the same capability
boundary.

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
