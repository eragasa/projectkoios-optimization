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
exact ndarray representation checks and separate dense power and norm evaluations. The
SciPy module owns separate sparse power and norm evaluations and consumes the single
private CSR canonicalization operation in `scipy/_csr.py`.
`ComplexMatrixFrobenius` validates its canonical fields explicitly and never replaces
them after frozen construction. The backend norm
classes compose their matching power classes and take the binary64 square root. Package
initializers remain import-empty.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/
├── test__ComplexMatrixFrobenius__unit.py
├── test__ComplexMatrixFrobeniusWheel__integration.py
├── numpy/frobenius/
│   ├── test__NumpyFrobeniusNorm__integration.py
│   └── test__NumpyFrobeniusPower__integration.py
└── scipy/frobenius/
    ├── test__ScipyCsrFrobeniusNorm__integration.py
    └── test__ScipyCsrFrobeniusPower__integration.py
```

Tests cover rectangular real and complex matrices, zero matrices, result correlation,
logical row-major traversal across storage layouts and views, caller nonmutation,
read-only input, exact public typing, malformed and nonfinite rejection, CSR duplicate
and zero canonicalization, canonicalization overflow, sparse caller nonmutation, absence
of densification, term overflow, accumulated-power overflow, distinct scalar power and
norm behavior, nominal norm semantics, and isolated dependency-free wheel execution.
