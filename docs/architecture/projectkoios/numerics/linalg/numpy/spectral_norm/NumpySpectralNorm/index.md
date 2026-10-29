# `NumpySpectralNorm`

## Identity

`projectkoios.numerics.linalg.numpy.spectral_norm.NumpySpectralNorm` is the final concrete
NumPy implementation of `MatrixNormEnum.spectral` for dense binary64 matrices.

## Construction

```python
from projectkoios.numerics.linalg.numpy.spectral_norm import NumpySpectralNorm

spectral_norm = NumpySpectralNorm(matrix)
value = spectral_norm.result()
```

`matrix` is an exact nonempty finite two-dimensional NumPy ndarray with exact `float64`
or `complex128` storage. Construction retains the array reference. `result()` validates
and evaluates the represented matrix without mutation or dtype conversion.

## Result

`result()` returns the largest singular value as an exact built-in finite nonnegative
`float`. `matrix_norm` returns `MatrixNormEnum.spectral`.

## Ownership boundary

The class owns dense representation validation, one complete `numpy.linalg.svdvals`
call, largest-value selection, nonconvergence translation, and nonfinite-output
rejection. It owns no units, sparse iteration, scaling fallback, residual evidence,
scientific threshold, or acceptance policy.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Schematics](schematics.md)
- [Scientific boundary](scientific.md)
- [`spectral_norm` module](../index.md)
- [Dense spectral-norm capability](../../../spectral_norm/index.md)
