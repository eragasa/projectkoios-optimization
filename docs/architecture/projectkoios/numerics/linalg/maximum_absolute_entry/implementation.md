# Maximum-absolute-entry implementation

## Source

```text
src/python/projectkoios/numerics/linalg/
├── numpy/
│   └── maximum_absolute_entry.py
└── scipy/
    └── maximum_absolute_entry.py
```

The NumPy module owns exact dense-array validation and direct magnitude reduction. The
SciPy module owns the concrete CSR boundary, private-copy canonicalization, and stored-
value reduction. `NumpyMaximumAbsoluteEntryNorm` and
`ScipyCsrMaximumAbsoluteEntryNorm` retain the matrix supplied at construction and
implement their representation-specific validation and reduction directly when
`result()` is called. No parallel function API, factory,
dispatcher, static method, class method, or additional helper layer is present. Package
initializers remain import-empty.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/
├── numpy/maximum_absolute_entry/
│   └── test__NumpyMaximumAbsoluteEntryNorm__integration.py
└── scipy/maximum_absolute_entry/
    └── test__ScipyCsrMaximumAbsoluteEntryNorm__integration.py
```

Tests cover real and complex matrices, rectangular shape, both CSR runtime classes,
storage layouts, noncontiguous and read-only dense input, caller nonmutation, exact zero,
CSR duplicates and cancellation, explicit and implicit zeros, absence of densification,
malformed shapes and dtypes, nonfinite entries, canonicalization and magnitude overflow,
semantic identity, and exact built-in scalar output.
