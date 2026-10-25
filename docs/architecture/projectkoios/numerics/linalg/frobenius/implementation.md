# Frobenius implementation

## Source

```text
src/python/projectkoios/numerics/linalg/
├── frobenius.py
└── numpy/
    └── frobenius.py
```

The dependency-free module owns the detailed result, exact tuple operation, and shared
ordered scalar reduction without importing an optional backend. The NumPy module owns
exact ndarray representation checks and direct dense-array evaluation.
`ComplexMatrixFrobenius` validates its canonical fields explicitly and never replaces
them after frozen construction. `NumpyFrobeniusNorm` is a stateless nominal evaluator
that delegates to the detailed NumPy operation. Package initializers remain import-empty.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/
├── test__ComplexMatrixFrobenius__unit.py
├── test__ComplexMatrixFrobeniusWheel__integration.py
└── numpy/frobenius/
    └── test__NumpyFrobeniusNorm__integration.py
```

Tests cover rectangular real and complex matrices, zero matrices, result correlation,
logical row-major traversal across storage layouts and views, caller nonmutation,
read-only input, exact public typing, malformed and nonfinite rejection, term overflow,
accumulated-power overflow, nominal evaluator behavior, and isolated dependency-free
wheel execution.
