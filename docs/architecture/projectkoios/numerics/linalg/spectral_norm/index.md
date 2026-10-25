# Dense spectral norm

`projectkoios.numerics.linalg.numpy.spectral_norm` owns direct dense NumPy evaluation of

$$
\lVert A\rVert_2 = \sigma_{\max}(A),
$$

the largest singular value of one general nonempty rectangular matrix. This is the
induced Euclidean matrix norm. For a general matrix, eigenvalue magnitudes do not define
this norm. For a normal square matrix, the largest eigenvalue magnitude equals this norm.

## Public API

- `NumpySpectralNorm(matrix).result()`

Construction retains the supplied matrix reference. `result()` validates and evaluates
the current represented matrix, then returns an exact built-in finite nonnegative
`float`. `matrix_norm` identifies `MatrixNormEnum.spectral`.

## Representation contract

The accepted representation is an exact two-dimensional NumPy ndarray with:

- positive row and column counts;
- exact `float64` or `complex128` storage; and
- finite real and imaginary components.

The operation accepts C-order, Fortran-order, strided, writable, and read-only storage.
It performs no dtype conversion, normalization, or caller mutation.

## Capability boundary

The capability owns the complete dense singular-value call, largest-value selection,
and classification of input, nonconvergence, and failed finite arithmetic. It does not
own physical units, power-of-two scaling policy for specialized extreme-range
residuals, sparse iteration controls, residual evidence, scientific thresholds, or
acceptance.

Consumers restore the matrix-entry unit to the scalar norm. No sparse spectral-norm
implementation exists. Its iterative architecture and verification are independent of
this dense implementation.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
