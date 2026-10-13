# Least-squares implementation organization

## Target modules

```text
projectkoios/optimization/least_squares/
├── __init__.py
├── affine.py
├── linear.py
├── matrix_subspaces.py
├── factory.py
├── numpy/
│   ├── __init__.py
│   ├── affine.py
│   ├── linear.py
│   └── matrix_subspaces.py
└── scipy/
    ├── __init__.py
    ├── affine.py
    └── linear.py
```

## Nominal solver and factory

`LinearLeastSquaresSolver` is a capability-local nominal abstract class and the public
SDK integration point. Its public `solve(...)` template method owns shared validation and
result invariants; concrete backends explicitly inherit it and implement only the
protected numerical operation.

`LinearLeastSquaresMethod` is a closed enum of supported implementations.
`LinearLeastSquaresSolverFactory.create(method)` lazily imports and returns a supported
concrete subclass. External structural lookalikes are not accepted as solver
implementations.

## Linear least-squares contract

The final public template signature is:

```python
def solve(
    design_matrix: LinearLeastSquaresMatrix,
    target_matrix: LinearLeastSquaresMatrix,
    /,
    *,
    weights: tuple[float, ...] | None = None,
    rank_tolerance: float | None = None,
) -> LinearLeastSquaresResult: ...
```

Both matrices are exact nonempty rectangular tuples. The design has shape `(m, n)` and
the target has shape `(m, r)`, where all dimensions are positive. Finite real and complex
scalars are normalized to built-in complex binary64 values. Booleans and nonfinite or
unrepresentable inputs fail.

`weights=None` defines ordinary least squares. Otherwise exactly one finite strictly
positive weight is required per row and the objective is
`sum_i weights[i] * sum_j |(A C - B)[i, j]|**2`; the backend applies square-root row
scaling. `rank_tolerance=None` selects the owner-defined binary64 cutoff
`sys.float_info.epsilon * max(m, n)`, matching the current NumPy default. An explicit
relative cutoff is finite, at least zero, and less than one. Singular values strictly
greater than the cutoff times the largest singular value contribute to rank.

`LinearLeastSquaresResult` retains:

- coefficient matrix `(n, r)`;
- fitted target matrix `(m, r)`;
- raw residual matrix `fitted_values - target_matrix`;
- raw and weighted per-right-hand-side L2 residual norms;
- raw and weighted whole-matrix Frobenius residual norms;
- maximum raw row L2 residual norm;
- numerical rank of the weighted design;
- nonincreasing singular values of the weighted design; and
- its spectral condition number only when numerical rank equals `n`.

`LinearLeastSquaresTolerance` defaults to absolute `1.0e-12` and relative `1.0e-10`.
It governs final backend-result consistency only; it does not change the caller's rank
tolerance or any scientific decision. Final validation checks fitted values, raw and
weighted diagnostics, the weighted-design Frobenius/singular-value identity, rank against
the effective cutoff, and the condition number. It checks full-gradient stationarity
when no positive singular direction was discarded. Explicitly or numerically truncated
solutions are not full-gradient stationary, so that check is inapplicable to them.

Individual singular values remain decomposition-backend output; dependency-free final
validation authenticates their shape, order, finite state, total spectral energy, rank
cutoff, and derived condition but does not recompute an independent SVD. Independent QR
and normal-equation tests provide bounded verification for the built-in backend.

## Matrix-subspace contract

The caller supplies an ordered real matrix basis and target matrix.
`NumpyRealMatrixSubspaceLeastSquaresSolver` returns a
`RealMatrixSubspaceLeastSquaresResult` containing coefficients, best-fit matrix, residual
matrix, Frobenius residual, rank, singular values, and condition number. The API does not
carry model-class names or acceptance labels.

## Failure behavior

- Empty observations fail.
- Shape mismatches fail.
- Nonfinite values fail.
- Nonpositive weights fail.
- Rank deficiency is returned as numerical state when the operation supports incomplete
  design; it is not silently converted to a full-rank fit.
- Condition number is reported only under its declared rank convention.

## Verification

- Verify ordinary least squares for real and complex inputs.
- Reproduce M1 uniform, nonuniform-weight, and incomplete-design routes.
- Reproduce periodic-1D and periodic-2D direct fits.
- Compare with independently reconstructed normal-equation and QR/SVD routes.
- Verify multi-right-hand-side equivalence with separate column fits.
- Verify rank, condition, fitted values, and residual aggregates.
- Reproduce Appendix-D and Stage-C matrix-subspace fits.

## Status

The NumPy dense linear capability is implemented and has passed its software, bounded
numerical, packaging, and independent code-review gate. It supports ProjectKoios
consumers that must not own a reusable generic least-squares kernel. Matrix-subspace and
SciPy-specific least-squares capabilities remain planned and require separate review. No
cross-project provenance record is required for internal ProjectKoios ownership work.
