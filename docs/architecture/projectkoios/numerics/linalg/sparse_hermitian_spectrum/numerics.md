# Sparse partial-spectrum numerics

The numerical contract preserves sparse storage through qualification and ARPACK
execution while making partial selection and convergence controls explicit.

- Canonical CSR storage omits explicit zeros and duplicate coordinates.
- Represented-input Hermiticity uses the same maximum-entry cutoff semantics as the
  complete-spectrum capability.
- A deterministic seeded start vector supports repeatable owner calls.

## Sparse representation

The owner CSR record defines one finite square matrix without importing SciPy into the
contract module.

- `indptr` has length `dimension + 1`, starts at zero, and is nondecreasing.
- `indices` and `data` have the terminal count stored in `indptr`.
- Column indices increase strictly within each row.
- Stored values are finite, complex-capable, and nonzero.
- Missing coordinates represent exact zero.

## Hermiticity qualification

Sparse qualification operates on the union of stored coordinates and their transposes so
missing asymmetric entries contribute to the measured defect.

- Scale is $s=\max_{i,j}|A_{ij}|$ over stored values with a zero default.
- Defect is $\delta=\max_{i,j}|A_{ij}-\overline{A_{ji}}|$ over the coordinate union.
- Cutoff is $\tau=\tau_{\mathrm{abs}}+\tau_{\mathrm{rel}}s$.
- Equality is accepted, and only $\delta>\tau$ fails.
- The explicit sparse representative stores nonzeros of $(A+A^\dagger)/2$.

## Iterative selection

SciPy ARPACK computes a strict subset whose count and ordering semantics remain visible in
the definition and result.

- `SMALLEST_ALGEBRAIC` maps to SciPy `which="SA"`.
- `LARGEST_ALGEBRAIC` maps to SciPy `which="LA"`.
- `SMALLEST_MAGNITUDE` maps to SciPy `which="SM"`.
- `LARGEST_MAGNITUDE` maps to SciPy `which="LM"`.
- `eigensolver_tolerance=0.0` delegates precision choice to SciPy.
- `maximum_iterations` and `krylov_subspace_dimension` remain optional explicit controls.
- `random_seed` generates the normalized ARPACK start vector.
- Complex SciPy problems require `eigenvalue_count < dimension - 1` because SciPy routes
  complex Hermitian problems through ARPACK's nonsymmetric interface.

## Residual evidence

Backend-local eigenvectors support verification without becoming public frame data.

- Each residual is $r_i=Hv_i-\lambda_i v_i$.
- The result reports $\max_i\lVert r_i\rVert_2$.
- Returned eigenvalues are sorted in ascending algebraic order.
- Residual evidence is bounded numerical verification, not a scientific threshold or
  uncertainty certificate.
