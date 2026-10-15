# Procrustes implementation

## Source

```text
src/python/projectkoios/optimization/alignment/
├── __init__.py
├── procrustes.py
└── numpy/
    ├── __init__.py
    └── procrustes.py
```

`procrustes.py` contains the dependency-free immutable definition, tolerance, status,
result, nominal solver, closed method enum, and lazy factory. `numpy/procrustes.py`
contains the sole built-in SVD implementation.

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
    └── test__NumpyComplexUnitaryProcrustesSolver__numerical.py
```

The complete mapped tree is included in strict mypy. The wheel test separately verifies
that the contract imports without NumPy and that explicit factory selection executes the
NumPy backend.

## Exclusions

There is no rectangular, partial-isometry, batch, aggregate, frame-path, transport,
closure, phase-synchronization, or campaign module in this slice. Package initializers
re-export nothing.
