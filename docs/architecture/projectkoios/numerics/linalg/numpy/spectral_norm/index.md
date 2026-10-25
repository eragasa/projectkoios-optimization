# `projectkoios.numerics.linalg.numpy.spectral_norm`

The source module
`src/python/projectkoios/numerics/linalg/numpy/spectral_norm.py` owns complete dense NumPy
spectral-norm evaluation for exact float64 and complex128 matrices.

## Owned classes

- [`NumpySpectralNorm`](NumpySpectralNorm/index.md) validates one retained dense matrix,
  calls `numpy.linalg.svdvals`, and returns the largest singular value through
  `MatrixNormABC.result()`.

## Public exports

```python
__all__ = ["NumpySpectralNorm"]
```

The module exposes no function facade, result record, factory, dispatcher, or fallback
algorithm.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/numpy/spectral_norm/
└── test__NumpySpectralNorm__integration.py
```

## Related architecture

- [Dense spectral-norm capability](../../spectral_norm/index.md)
- [NumPy linear-algebra backends](../index.md)
- [Matrix-norm foundation](../../matrix_norm/index.md)
