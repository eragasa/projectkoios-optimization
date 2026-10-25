# Matrix spectral norm

The spectral norm of a nonempty rectangular matrix is

$$
\lVert A\rVert_2 = \sigma_{\max}(A),
$$

the largest singular value. For a general matrix, eigenvalue magnitudes do not define
this norm. The capability provides separate dense-complete and sparse-piecewise
implementations under the canonical `MatrixNormEnum.spectral` identity.

## Public API

- `NumpySpectralNorm(matrix).result()` performs a complete dense NumPy decomposition.
- `ScipyCsrSpectralNormControls(tolerance, maximum_iterations, krylov_dimension, seed)`
  records mandatory sparse numerical controls.
- `ScipyCsrSpectralNorm(matrix, controls).result()` performs deterministic sparse
  evaluation.

Every evaluator retains the supplied matrix reference and returns an exact built-in
finite nonnegative `float`. Dense and sparse implementations remain distinct public
classes because their numerical contracts differ.

## Representation contracts

The dense implementation accepts exact nonempty finite two-dimensional NumPy ndarrays
with exact `float64` or `complex128` storage.

The sparse implementation accepts nonempty finite two-dimensional SciPy `csr_matrix` or
`csr_array` values with exact `float64` or `complex128` storage. It canonicalizes a
private copy, consolidates duplicate coordinates, removes explicit zeros, sorts indices,
and preserves caller storage.

## Sparse algorithm identity

The sparse evaluator uses one piecewise primary contract:

- exact-zero canonical storage returns exact `0.0`;
- `min(shape) <= 2` uses scale-safe direct sparse small-Gram arithmetic; and
- `min(shape) >= 3` uses deterministic approximate
  `scipy.sparse.linalg.svds(k=1, which="LM", solver="arpack")`.

The small-shape branch exists because pinned SciPy's complex ARPACK route rejects the
minimum-dimension-two request. It is not a fallback. Neither sparse branch materializes
the caller matrix as dense or delegates to the dense evaluator.

## Ownership boundary

Numerics owns representation validation, backend selection inside each concrete class,
convergence controls, deterministic start construction, internal residual qualification,
and numerical failure classification. Consumers restore matrix-entry units and own
scientific interpretation, thresholds, acceptance, and uncertainty.

## Pages

- [`NumpySpectralNorm` class](../numpy/spectral_norm/NumpySpectralNorm/index.md)
- [`ScipyCsrSpectralNorm` class](../scipy/spectral_norm/ScipyCsrSpectralNorm/index.md)
- [`ScipyCsrSpectralNormControls` class](../scipy/spectral_norm/ScipyCsrSpectralNormControls/index.md)
- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Linear algebra numerics](../index.md)
