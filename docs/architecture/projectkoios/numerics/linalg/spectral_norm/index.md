# Dense spectral norm

`projectkoios.numerics.linalg.numpy.spectral_norm` owns direct dense NumPy evaluation of

$$
\lVert A\rVert_2 = \sigma_{\max}(A),
$$

the largest singular value of one general nonempty rectangular matrix. This is not the
largest eigenvalue magnitude except for special matrix classes.

## Public API

- `NumpySpectralNorm(matrix).result()`

The evaluation accepts an exact finite two-dimensional NumPy ndarray with exact
`float64` or `complex128` storage. It returns an exact built-in finite nonnegative float
and identifies `MatrixNormEnum.spectral` through `MatrixNormABC`.

## Boundary

The capability owns complete dense singular-value evaluation and backend failure
classification. It does not own physical units, scaling policy for specialized extreme-
range diagnostics, sparse iteration controls, residual evidence, scientific thresholds,
or acceptance. Sparse spectral evaluation remains separately prospective.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
