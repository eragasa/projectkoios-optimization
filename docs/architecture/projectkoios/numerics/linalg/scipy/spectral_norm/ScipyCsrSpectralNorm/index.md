# `ScipyCsrSpectralNorm`

## Identity

`projectkoios.numerics.linalg.scipy.spectral_norm.ScipyCsrSpectralNorm` is the final
concrete SciPy implementation of `MatrixNormEnum.spectral` for binary64 CSR matrices.

## Construction

```python
from projectkoios.numerics.linalg.scipy.spectral_norm import ScipyCsrSpectralNorm

spectral_norm = ScipyCsrSpectralNorm(matrix, controls)
value = spectral_norm.result()
```

`matrix` is a nonempty two-dimensional SciPy CSR matrix or array with exact `float64` or
`complex128` storage. `controls` is an exact `ScipyCsrSpectralNormControls` instance.
Construction retains both references. Evaluation reads caller storage and mutates only a
private canonical copy.

## Result

`result()` returns the largest singular value as an exact built-in finite nonnegative
`float`. Small shapes use scale-safe direct arithmetic. Larger shapes return an
internally qualified deterministic ARPACK approximation.

## Ownership boundary

The class owns representation validation, private canonicalization, dimension-defined
algorithm selection, deterministic starts, pinned solver arguments, internal triplet
qualification, and numerical failure classification. It owns no units, scientific
acceptance, public evidence record, dense fallback, or alternate sparse solver.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Schematics](schematics.md)
- [Scientific boundary](scientific.md)
- [`spectral_norm` module](../index.md)
- [Matrix spectral-norm capability](../../../spectral_norm/index.md)
