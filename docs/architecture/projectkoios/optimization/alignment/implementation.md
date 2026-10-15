# Alignment implementation organization

## Source

```text
projectkoios/optimization/alignment/
├── __init__.py
├── procrustes.py
└── numpy/
    ├── __init__.py
    └── procrustes.py
```

Package initializers are import-empty. The dependency-free contract owns exact input
validation, cross-matrix construction, SVD conformance checks, numerical-rank policy,
transformation withholding, residual reconstruction, and result construction. The
NumPy backend performs only the complete square complex128 SVD.

## Nominal solver boundary

`ComplexUnitaryProcrustesSolver.solve` is a final public template. Concrete backends
inherit nominally and implement only `_svd_validated`. The closed factory imports the
NumPy backend lazily. Structural solver substitution and mutable backend registration
are not supported.

## Verification

Mapped tests cover exact typing, dimensions, direction, tolerance cutoffs, malformed
backend output, SVD reconstruction, unitary evidence, deficient withholding, residual
reconstruction, seeded numerical optimality, scale covariance, isolated-wheel laziness,
and the exact rank-one neighbor-overlap failure shape.

Earlier planning for rectangular, batched, frame-path, transport, closure, and phase
synchronization APIs has been removed from current-state architecture. It grants no
future implementation authority.
