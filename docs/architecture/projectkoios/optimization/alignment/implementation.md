# Alignment implementation organization

## Source

```text
projectkoios/optimization/alignment/
├── __init__.py
├── closed_loop_polar_transport.py
├── procrustes.py
├── numpy/
│   ├── __init__.py
│   ├── closed_loop_polar_transport.py
│   ├── procrustes.py
│   └── batched_procrustes/
│       ├── __init__.py
│       ├── base.py
│       ├── contract.py
│       ├── numerics.py
│       └── solver.py
└── scipy/
    ├── __init__.py
    └── closed_loop_polar_transport.py
```

Parent package initializers remain import-empty. The encapsulated batched capability
initializer exposes only names approved by its `base.py` façade. Dependency-free scalar
contracts and shared concrete policy remain in `procrustes.py`; exact-array batch records,
numerics, and execution have explicit module owners. Closed-loop transport separates
dependency-free diagnostics and failure types, exact NumPy records, and SciPy execution.
Its stateful SciPy solver receives the concrete Procrustes owner explicitly and delegates
every link selection to it.

## Nominal solver boundary

`ComplexUnitaryProcrustesSolver` owns final shared numerical-policy methods without a
generic solve return. `ScalarComplexUnitaryProcrustesSolver` owns the concrete final
one-pair template; concrete scalar backends implement `_svd_validated`.
`BatchedComplexUnitaryProcrustesSolver` owns vectorized rank-cutoff policy; the concrete
NumPy batch solver owns execution. The scalar factory still imports its NumPy backend
lazily.

## Verification

Mapped tests cover exact typing, dimensions, direction, tolerance cutoffs, malformed
backend output, SVD reconstruction, unitary evidence, deficient withholding, residual
reconstruction, seeded numerical optimality, scale covariance, isolated-wheel laziness,
and the exact rank-one neighbor-overlap failure shape.

Closed-loop unit and integration tests additionally cover exact complex128 frame
storage, self-validating result relations, scalar and composite rank, nondiagonal closure
distribution, direct right-action and correction identities, deficient and unattempted
progression, bounded backend failure classification, both explicit phase branches, exact
branch-cut roots, and metric discrimination.
Rectangular transformations remain excluded. No capability grants downstream scientific
authority.
