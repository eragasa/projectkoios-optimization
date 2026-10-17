# Alignment implementation organization

## Source

```text
projectkoios/optimization/alignment/
├── __init__.py
├── procrustes.py
└── numpy/
    ├── __init__.py
    ├── procrustes.py
    └── batched_procrustes/
        ├── __init__.py
        ├── base.py
        ├── contract.py
        ├── numerics.py
        └── solver.py
```

Parent package initializers remain import-empty. The encapsulated batched capability
initializer exposes only names approved by its `base.py` façade. Dependency-free scalar
contracts and shared generic policy remain in `procrustes.py`; exact-array batch records,
numerics, and execution have explicit module owners.

## Nominal solver boundary

`ComplexUnitaryProcrustesSolver[DefinitionT, ResultT]` owns final shared numerical-policy
methods and an abstract typed solve boundary. `ScalarComplexUnitaryProcrustesSolver`
owns the final one-pair template; concrete scalar backends implement `_svd_validated`.
`BatchedComplexUnitaryProcrustesSolver` owns vectorized rank-cutoff policy; the concrete
NumPy batch solver owns execution. The scalar factory still imports its NumPy backend
lazily.

## Verification

Mapped tests cover exact typing, dimensions, direction, tolerance cutoffs, malformed
backend output, SVD reconstruction, unitary evidence, deficient withholding, residual
reconstruction, seeded numerical optimality, scale covariance, isolated-wheel laziness,
and the exact rank-one neighbor-overlap failure shape.

Rectangular transformations, frame paths, transport, closure, and phase synchronization
remain excluded. The implemented batched capability provides square right-unitary
completions only and grants no downstream scientific authority.
