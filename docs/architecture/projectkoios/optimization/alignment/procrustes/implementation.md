# Procrustes implementation

## Source

```text
src/python/projectkoios/optimization/alignment/
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

`procrustes.py` contains the dependency-free immutable definition, tolerance, status,
result, generic policy-owning solver base, scalar nominal solver, closed method enum, and
lazy scalar factory. Scalar result construction delegates to named owners for structural,
rank, residual, transformation, and backend-evidence validation; the final frozen record
performs no postconstruction mutation. `numpy/procrustes.py` contains the one-pair built-in SVD
implementation. `numpy/batched_procrustes/` owns the encapsulated direct-array family:
façade, nominal batch contract, immutable result, numerical evidence, and vectorized
solver have separate owners.

## Tests

```text
tests/projectkoios/optimization/alignment/
├── conftest.py
├── fixtures/
│   ├── batched_procrustes.py
│   └── scalar_procrustes.py
├── test__ComplexUnitaryProcrustesDefinition__unit.py
├── test__ComplexUnitaryProcrustesResult__unit.py
├── test__ComplexUnitaryProcrustesResultValidation__unit.py
├── test__ComplexUnitaryProcrustesSolver__contract.py
├── test__ComplexUnitaryProcrustesSolver__unit.py
├── test__ComplexUnitaryProcrustesSolverFactory__unit.py
├── test__ComplexUnitaryProcrustesWheel__integration.py
└── numpy/
    ├── test__NumpyComplexUnitaryProcrustesSolver__integration.py
    ├── test__NumpyComplexUnitaryProcrustesSolver__numerical.py
    └── batched_procrustes/
        ├── test__BatchedComplexUnitaryProcrustesSolver__unit.py
        ├── test__CompletionEvidence__unit.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesDefinition__unit.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesResult__unit.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesSolver__integration.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesSolver__numerical.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesWheel__integration.py
        ├── test__ResidualArrays__unit.py
        ├── test__ResidualReconstruction__unit.py
        └── test__ResultArrays__unit.py
```

The result-validation tests use a named analytically derived one-dimensional alignment
case to exercise each named validation owner as well as the complete result constructor.
The complete mapped tree is included in strict mypy. Wheel tests separately verify the
lazy scalar factory and the direct NumPy batched module from isolated built artifacts.

## Exclusions

There is no partial-isometry, aggregate, frame-path, transport, closure,
phase-synchronization, or campaign module in this slice. The batched capability supports
rectangular source and target matrices with square right-unitary completions, but it does
not define rectangular transformations. Parent package initializers remain import-empty;
the batched capability initializer deliberately exposes only its approved façade.
