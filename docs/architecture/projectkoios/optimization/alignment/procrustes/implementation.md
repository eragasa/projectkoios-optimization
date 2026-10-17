# Procrustes implementation

## Source

```text
src/python/projectkoios/optimization/alignment/
├── __init__.py
├── procrustes.py
└── numpy/
    ├── __init__.py
    ├── procrustes.py
    └── batched_procrustes.py
```

`procrustes.py` contains the dependency-free immutable definition, tolerance, status,
result, nominal solver, closed method enum, and lazy factory. `numpy/procrustes.py`
contains the one-pair built-in SVD implementation. `numpy/batched_procrustes.py` owns the
direct-array batch definition, immutable result, and vectorized solver.

## Tests

```text
tests/projectkoios/optimization/alignment/
├── test__ComplexUnitaryProcrustesDefinition__unit.py
├── test__ComplexUnitaryProcrustesResult__unit.py
├── test__ComplexUnitaryProcrustesSolver__contract.py
├── test__ComplexUnitaryProcrustesSolverFactory__unit.py
├── test__ComplexUnitaryProcrustesWheel__integration.py
└── numpy/
    ├── test__NumpyComplexUnitaryProcrustesSolver__integration.py
    ├── test__NumpyComplexUnitaryProcrustesSolver__numerical.py
    └── batched_procrustes/
        ├── test__NumpyBatchedComplexUnitaryProcrustesDefinition__unit.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesResult__unit.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesSolver__integration.py
        ├── test__NumpyBatchedComplexUnitaryProcrustesSolver__numerical.py
        └── test__NumpyBatchedComplexUnitaryProcrustesWheel__integration.py
```

The complete mapped tree is included in strict mypy. Wheel tests separately verify the
lazy scalar factory and the direct NumPy batched module from isolated built artifacts.

## Exclusions

There is no partial-isometry, aggregate, frame-path, transport, closure,
phase-synchronization, or campaign module in this slice. The batched capability supports
rectangular source and target matrices with square right-unitary completions, but it does
not define rectangular transformations. Package initializers re-export nothing.
