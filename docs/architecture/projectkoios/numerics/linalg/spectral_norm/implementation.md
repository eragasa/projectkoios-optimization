# Dense spectral-norm implementation

## Source

```text
src/python/projectkoios/numerics/linalg/numpy/
└── spectral_norm.py
```

`NumpySpectralNorm` retains the matrix supplied at construction, performs exact dense
representation validation in `_evaluate`, calls `numpy.linalg.svdvals` once, classifies
backend nonconvergence and nonfinite output, and returns the largest singular value
through the final `MatrixNormABC.result()` boundary. No function facade, helper layer,
factory, dispatcher, classmethod, static method, cache, scaling path, or result record is
present. Package initializers remain import-empty.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/numpy/spectral_norm/
└── test__NumpySpectralNorm__integration.py
```

Tests cover real and complex matrices, rectangular and nonnormal shape, rank deficiency,
one-row and one-column matrices, exact zero, storage layouts and views, read-only input,
caller nonmutation, invalid shape and dtype, nonfinite source entries, unrepresentable
finite-input output, NumPy two-norm compatibility, nonconvergence translation, nonfinite
backend output, semantic identity, and exact built-in scalar output.
