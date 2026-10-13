# `WeightedAffineResidualQuadraticConstructor`

## Purpose

`projectkoios.optimization.affine.quadratic.WeightedAffineResidualQuadraticConstructor`
selects a capability-local implementation that converts a dependency-free
`WeightedAffineResidualDefinition` into the completed-square quadratic capability.

## Shared construction

The public `affine/quadratic.py` module owns two independently tested operations:

- `weighted_design()` returns square-root-weighted design rows and targets; and
- `weighted_normal_matrix()` returns an exactly mirrored binary64 Gram matrix using
  checked scalar products and deterministic summation.

The constructor validates its definition and built-in Boolean selector without loading
an optional backend. `sparse=False` lazily selects
`affine.numpy.affine.NumpyWeightedAffineResidualQuadraticConstructor`; `sparse=True`
lazily selects `affine.scipy.affine.ScipyWeightedAffineResidualQuadraticConstructor`.

## Numerical acceptance

The NumPy implementation calls `numpy.linalg.lstsq(..., rcond=None)`. The SciPy
implementation calls `scipy.linalg.lstsq(..., cond=None, lapack_driver="gelsd")`. Each
rejects a reported numerical rank smaller than the coordinate dimension and never
regularizes a rejected system. This direct weighted-design gate prevents rounded normal
equations from inventing strict positive definiteness for an exactly dependent design.

After rank acceptance, the selected constructor evaluates the minimum from the original
residual definition and delegates represented SPD acceptance and evaluation to the
matching quadratic backend.

These rank and SPD decisions are backend-dependent numerical acceptance, not exact-real
proofs, interval enclosures, outward-rounded certificates, or scientific validation.

## Import boundary

Package initializers export nothing. Supported imports identify the exact owner:

```python
from projectkoios.optimization.affine.quadratic import (
    WeightedAffineResidualQuadraticConstructor,
)
from projectkoios.optimization.affine.numpy.affine import (
    NumpyWeightedAffineResidualQuadraticConstructor,
)
from projectkoios.optimization.affine.scipy.affine import (
    ScipyWeightedAffineResidualQuadraticConstructor,
)
```
