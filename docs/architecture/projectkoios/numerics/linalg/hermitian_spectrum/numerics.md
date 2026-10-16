# Batched Hermitian-spectrum numerics

The numerical contract distinguishes represented-input qualification, Hermitian
representative construction, eigensolver execution, and result validation.

- Every tolerance and diagnostic is unit-free.
- Every accepted nonempty batch reaches its backend as one complete matrix stack.
- Every empty batch returns without a numerical backend operation.

## Input and cutoff

Each matrix receives an independent scale and Hermiticity cutoff, so one large matrix
cannot relax acceptance for another matrix in the same batch.

- Nonempty batches infer their common positive dimension.
- Empty batches require a positive integral dimension through the named empty
  constructor.
- Finite real and complex numeric inputs are normalized to immutable built-in complex
  values.
- Matrix scale and represented Hermiticity defect are
  $$
  s_b=\max_{i,j}|(A_b)_{ij}|,
  \qquad
  \delta_b=\max_{i,j}|(A_b)_{ij}-\overline{(A_b)_{ji}}|.
  $$
- The applied cutoff is
  $$
  \tau_b=\tau_{\mathrm{abs}}+\tau_{\mathrm{rel}}s_b.
  $$
- Equality is accepted, and only $\delta_b>\tau_b$ fails.
- One failed matrix rejects the complete batch before backend invocation.

## Representative

Every accepted nonempty input is replaced by an explicit represented-Hermitian matrix
before eigensolver execution.

- The representative is
  $$
  H_b=\frac{A_b+A_b^\dagger}{2}.
  $$
- Checked complex binary64 arithmetic rejects nonfinite construction.
- An exact postcondition verifies $H_b=H_b^\dagger$ in the represented values.
- Representative adjustment is
  $$
  \eta_b=\max_{i,j}|(H_b)_{ij}-(A_b)_{ij}|.
  $$
- Separate defect and adjustment fields preserve evidence about tolerated
  non-Hermiticity.

## Backend behavior

The backend boundary preserves batching without assigning empty-set behavior to a
third-party eigensolver.

- A nonempty batch becomes one complex128 array stack.
- The NumPy backend invokes `numpy.linalg.eigvalsh(..., UPLO="L")` exactly once.
- The NumPy backend does not loop through scalar owner eigensolvers in Python.
- An empty batch returns empty spectra and per-matrix diagnostics without calling NumPy.
- An empty result retains its declared dimension and uses `diagnostics=None`.
- The conceptual empty spectrum shape is `(0, dimension)`.
- The shared template validates spectrum width, finiteness, nondecreasing order, and batch
  correlation.
- The template does not reconstruct eigenvectors or introduce an undeclared residual
  tolerance.

## Failure classification

Failure classes distinguish malformed public input, represented arithmetic failure, and
backend nonconvergence without making message strings part of the API.

- Unsupported representations raise `TypeError`.
- Invalid dimensions, shapes, finiteness, ordering, or Hermiticity policy raise
  `ValueError`.
- Nonfinite shared arithmetic or backend eigenvalues raise `FloatingPointError`.
- NumPy eigensolver nonconvergence raises `RuntimeError`.
- Exception classes and inducing conditions are public behavior.
- Human-readable exception messages remain diagnostic context.
- Diagnostics constitute software and bounded numerical evidence, not scientific
  validation or an uncertainty certificate.
