# ResidualReconstruction

`ResidualReconstruction` owns cancellation-aware reconstruction of the optimal squared
right-Procrustes residual.

## Numerical operation

For source power `p`, target power `q`, and complete cross singular spectrum
`(sigma_i)`, it evaluates

$$
r^2 = p + q - 2\sum_i \sigma_i
$$

in a normalized scale. It also computes an additive squared-unit roundoff cutoff from
the absolute normalized contribution sum and the number of reduced terms.

- Values within the cutoff become exact zero.
- A negative value beyond the cutoff raises `FloatingPointError` with the first C-order
  batch index.
- Nonfinite nuclear-norm contributions, reconstruction state, cutoffs, or final values
  raise `FloatingPointError`.
- Returned arrays have exact `numpy.float64` dtype.

The class is an encapsulated numerical evidence record. It is consumed by both result
validation and solver assembly but is not exported by the package façade.

## Verification

Dedicated tests cover exact positive residuals, zero-scale input, cancellation clamping,
multidimensional shape preservation, deterministic negative-value indexing, nonfinite
arithmetic rejection, exact dtypes, and frozen field binding.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
