# Batched complete Hermitian spectra

`projectkoios.numerics.linalg.hermitian_spectrum` owns a dependency-free contract for
complete eigenvalue spectra of finite dense complex Hermitian matrix batches.

- The contract is unit-free and backend-neutral.
- The NumPy backend remains an explicit capability-local import.
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

The definition stores one matrix dimension without requiring callers to duplicate shape
information that already exists in a nonempty batch.

- `BatchedHermitianSpectrumDefinition(matrices, tolerance)` infers the common dimension
  from a nonempty batch.
- `BatchedHermitianSpectrumDefinition.empty(dimension, tolerance)` supplies the dimension
  only when the batch contains no matrix.
- A dimension supplied alongside a nonempty batch is rejected as redundant.
- Python and array-like iterables are normalized to immutable built-in tuples.
- Finite real and complex numeric scalars are normalized to built-in complex values.

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
- Concrete backend: `NumpyBatchedHermitianSpectrumSolver`.
- Empty results use `diagnostics is None` because a maximum over no observations is
  undefined.

## Usage

The following example shows nonempty dimension inference and explicit empty construction
without importing NumPy through the contract module.

- A nonempty batch uses the ordinary definition constructor, while an empty batch uses
  the named empty constructor.

  ```python
  from projectkoios.numerics.linalg.hermitian_spectrum import (
      BatchedHermitianSpectrumDefinition,
      BatchedHermitianSpectrumTolerance,
  )
  from projectkoios.numerics.linalg.numpy.hermitian_spectrum import (
      NumpyBatchedHermitianSpectrumSolver,
  )

  policy = BatchedHermitianSpectrumTolerance(
      hermiticity_absolute=1.0e-12,
      hermiticity_relative=1.0e-12,
  )
  solver = NumpyBatchedHermitianSpectrumSolver()
  result = solver.solve(
      BatchedHermitianSpectrumDefinition(
          [
              [[2.0, 1.0 - 1.0j], [1.0 + 1.0j, 3.0]],
              [[4.0, 0.0], [0.0, 5.0]],
          ],
          policy,
      )
  )
  empty = solver.solve(BatchedHermitianSpectrumDefinition.empty(2, policy))
  assert result.dimension == 2
  assert empty.diagnostics is None
  ```

## Boundary

The capability owns only reusable matrix numerics and represented-input evidence.

- It owns normalization, Hermiticity measurement, representative construction, complete
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
