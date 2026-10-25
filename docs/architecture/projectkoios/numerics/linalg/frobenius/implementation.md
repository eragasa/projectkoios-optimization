# Frobenius implementation

## Source

```text
src/python/projectkoios/numerics/linalg/
├── frobenius.py
├── numpy/
│   └── frobenius.py
└── scipy/
    └── frobenius.py
```

The dependency-free module owns the detailed result, exact tuple operation, and shared
ordered scalar reduction without importing an optional backend. The NumPy module owns
exact ndarray representation checks and direct dense-array evaluation. The SciPy module
owns the concrete CSR boundary, private-copy canonicalization, and sparse evaluation.
`ComplexMatrixFrobenius` validates its canonical fields explicitly and never replaces
them after frozen construction. `NumpyFrobeniusNorm` and `ScipyCsrFrobeniusNorm` are
stateless nominal evaluators that delegate to their detailed backend operations. Package
initializers remain import-empty.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/
├── test__ComplexMatrixFrobenius__unit.py
├── test__ComplexMatrixFrobeniusWheel__integration.py
├── numpy/frobenius/
│   └── test__NumpyFrobeniusNorm__integration.py
└── scipy/frobenius/
    └── test__ScipyCsrFrobeniusNorm__integration.py
```

Tests cover rectangular real and complex matrices, zero matrices, result correlation,
logical row-major traversal across storage layouts and views, caller nonmutation,
read-only input, exact public typing, malformed and nonfinite rejection, CSR duplicate
and zero canonicalization, canonicalization overflow, sparse caller nonmutation, absence
of densification, term overflow, accumulated-power overflow, nominal evaluator behavior,
and isolated dependency-free wheel execution.
