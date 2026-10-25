# Maximum-absolute-entry matrix norm

The maximum-absolute-entry norm is

$$
\lVert A\rVert_{\max}=\max_{i,j}|A_{ij}|.
$$

Capability-local NumPy and SciPy modules evaluate this unit-free scalar directly for
nonempty finite float64 or complex128 dense and CSR matrix representations.

## Public API

- `NumpyMaximumAbsoluteEntryNorm`
- `ScipyCsrMaximumAbsoluteEntryNorm`

Each nominal evaluator validates its representation, performs the reduction, returns an
exact built-in finite nonnegative float through `MatrixNormABC`, and identifies
`MatrixNormEnum.maximum_absolute_entry`. No parallel function API or detailed result
record is needed for this scalar-only operation.

## Boundary

This capability owns representation validation, entry-magnitude evaluation, CSR
canonicalization, and binary64 failure behavior. It does not own units, normalization,
thresholds, matrix meaning, display labels, or scientific interpretation. The SciPy path
copies before canonicalization and neither densifies nor mutates the caller object.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
