# Slice 02 — real/complex linear least-squares support

## Status

This support slice is standing-authorized at the Optimization ownership boundary: KSD and
other consumers must not own a reusable generic mathematical least-squares kernel. The
implementation has passed its software, bounded numerical, packaging, and independent
code-review gate. This does not authorize consumer scientific policy, dependency
publication, commit, merge, or push.

## Owned capability

`projectkoios.optimization.least_squares` owns ordinary and finite-positive-row-weighted
linear least squares for finite real or complex matrices with one or multiple right-hand
sides.

The first backend is dense NumPy:

```text
projectkoios/optimization/least_squares/
├── __init__.py
├── linear.py
├── factory.py
└── numpy/
    ├── __init__.py
    └── linear.py
```

Package initializers remain import-empty. `LinearLeastSquaresSolver` is a capability-local
nominal ABC with a final validated `solve` template. `LinearLeastSquaresMethod` is a closed
string enum and `LinearLeastSquaresSolverFactory` imports NumPy lazily. The built-in
`NumpyLinearLeastSquaresSolver` is final.

## Exact numerical boundary

The caller supplies exact nonempty rectangular tuples:

- design matrix `A` with shape `(m, n)`;
- target matrix `B` with shape `(m, r)`;
- either no weights or exactly `m` finite strictly positive row weights; and
- either the owner-defined binary64 default rank cutoff or one finite explicit relative
  cutoff at least zero and less than one.

All dimensions are positive. Real and complex inputs are normalized to built-in complex
binary64. Weighted mode minimizes

```text
sum_i weights[i] * sum_j |(A C - B)[i, j]|**2
```

through square-root row scaling. It does not form normal equations.

The result retains coefficients, fitted values, raw residuals, raw and weighted per-RHS
norms, raw and weighted whole-matrix Frobenius norms, maximum raw row norm, numerical
rank, weighted-design singular values, and the weighted-design spectral condition number
only at numerical full column rank.

Raw residuals are `fitted_values - B`. `rank_tolerance=None` means
`sys.float_info.epsilon * max(m, n)`; rank counts singular values strictly above the
relative cutoff times the largest singular value. Backend acceptance tolerance validates
returned relationships but never changes rank or scientific acceptance. Full-gradient
stationarity is checked only when no positive singular direction was discarded, because
a truncated SVD solution is stationary only in its retained subspace.

The final template validates singular-value shape, order, finite state, total spectral
energy against the weighted-design Frobenius norm, explicit rank-cutoff consistency, and
the derived condition number. It does not independently recompute an SVD; individual
singular values remain backend decomposition evidence qualified by independent numerical
tests.

## Consumer boundary

Consumers retain:

- sample/domain records and coordinate meaning;
- design construction and basis meaning;
- units and conversion to common numeric magnitudes;
- coefficient interpretation and domain-record reconstruction;
- route meaning and comparisons;
- scientific diagnostics, thresholds, adequacy, and validation; and
- independent reconstruction used to qualify an owner version.

For KSD single-band hopping reduction, KSD constructs the Fourier design and flattened
matrix targets, calls this solver through its domain fitter, and maps coefficient rows
back to hopping blocks. Optimization does not own reciprocal representatives, hopping
ranges, bands, Hermiticity, Parseval checks, or band-error interpretation.

## Explicit exclusions

- no KSD adapter or KSD record;
- no Fourier transform or hopping model;
- no units;
- no scientific fit disposition;
- no matrix-subspace fitter;
- no sparse backend;
- no regularization or constrained regression; and
- no movement of the existing affine-quadratic compatibility path.

## Verification gate

- dependency-free result, tolerance, and input validation;
- nominal ABC, final template, exact backend-result type, and forged relational or
  spectral-energy result rejection;
- lazy closed factory and import-empty initializers;
- real and complex ordinary solves;
- positive-weight complex multiple-RHS solve;
- equivalence with separate RHS solves;
- exact rank deficiency, order-one discarded-direction truncation, and explicit-rank-
  tolerance behavior;
- full-rank condition-number convention;
- forced backend failure translation and chaining;
- stable large- and small-magnitude residual diagnostics without naive norm squaring;
- bounded independent normal-equation and QR references;
- KSD-shaped Fourier-design regression;
- isolated wheel import and backend laziness;
- full tests, Ruff, formatting, strict mypy, reproducible builds; and
- independent technical review with no blocking findings.

Verification is software and bounded numerical verification only. It is not scientific
validation.
