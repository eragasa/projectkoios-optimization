# Maximum-absolute-entry matrix norm

The maximum-absolute-entry norm is

$$
\lVert A\rVert_{\max}=\max_{i,j}|A_{ij}|.
$$

`projectkoios.numerics.linalg.numpy.maximum_absolute_entry` currently evaluates this
unit-free scalar directly for nonempty finite float64 or complex128 NumPy matrices.

## Public API

- `numpy_matrix_maximum_absolute_entry(matrix)`
- `NumpyMaximumAbsoluteEntryNorm`

The detailed function returns an exact built-in finite nonnegative float. The nominal
evaluator exposes the same operation through `MatrixNormABC` and identifies
`MatrixNormEnum.maximum_absolute_entry`.

## Boundary

This capability owns representation validation, entry-magnitude evaluation, and
binary64 failure behavior. It does not own units, normalization, thresholds, matrix
meaning, display labels, or scientific interpretation. SciPy CSR support requires its
own capability-local backend implementation.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
