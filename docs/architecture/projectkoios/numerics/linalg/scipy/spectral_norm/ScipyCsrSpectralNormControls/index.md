# `ScipyCsrSpectralNormControls`

## Identity

`projectkoios.numerics.linalg.scipy.spectral_norm.ScipyCsrSpectralNormControls` is the
final frozen slotted owner record for deterministic sparse spectral-norm controls.

## Construction

```python
from projectkoios.numerics.linalg.scipy.spectral_norm import (
    ScipyCsrSpectralNormControls,
)

controls = ScipyCsrSpectralNormControls(
    tolerance=1.0e-10,
    maximum_iterations=1_000,
    krylov_dimension=None,
    seed=17,
)
```

All four fields are mandatory. The record supplies no canonical defaults and performs no
coercion.

## Fields

- `tolerance: float` is exact, finite, and satisfies `0.0 <= tolerance < 1.0`.
- `maximum_iterations: int` is exact and positive.
- `krylov_dimension: int | None` is `None` or an exact positive integer; matrix-dependent
  validity belongs to the evaluator.
- `seed: int` is exact and nonnegative.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Schematics](schematics.md)
- [Scientific boundary](scientific.md)
- [`spectral_norm` module](../index.md)
- [Matrix spectral-norm capability](../../../spectral_norm/index.md)
