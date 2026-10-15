# Frobenius implementation

## Source

```text
src/python/projectkoios/numerics/linalg/
└── frobenius.py
```

The module validates immutable built-in values, performs the reduction directly, and
imports no optional backend. Package initializers remain import-empty.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/
├── test__ComplexMatrixFrobenius__unit.py
└── test__ComplexMatrixFrobeniusWheel__integration.py
```

Tests cover rectangular and zero matrices, result correlation, exact public typing,
ragged and nonfinite rejection, term overflow, accumulated-power overflow, and
isolated dependency-free wheel execution.
