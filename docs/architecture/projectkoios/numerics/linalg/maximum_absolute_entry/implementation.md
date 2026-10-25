# Maximum-absolute-entry implementation

## Source

```text
src/python/projectkoios/numerics/linalg/numpy/
└── maximum_absolute_entry.py
```

The NumPy module owns exact dense-array validation and direct magnitude reduction.
`NumpyMaximumAbsoluteEntryNorm` is stateless and delegates to the detailed function. No
factory, dispatcher, static method, class method, or additional helper layer is present.
Package initializers remain import-empty.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/numpy/maximum_absolute_entry/
└── test__NumpyMaximumAbsoluteEntryNorm__integration.py
```

Tests cover real and complex matrices, rectangular shape, storage layouts,
noncontiguous and read-only input, caller nonmutation, exact zero, malformed shapes,
nonfinite entries, magnitude overflow, semantic identity, and exact built-in scalar
output.
