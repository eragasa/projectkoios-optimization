# Positive-definite quadratic loss

`projectkoios.optimization.quadratic.quadratic.PositiveDefiniteQuadraticLoss` is the
public lazy facade for the unit-free completed-square loss

$$
q(x)=q_{\min}+(x-c)^TQ(x-c).
$$

Construction requires a nonempty finite center, a finite exactly symmetric matrix of
the same dimension, and a finite nonnegative minimum squared loss. `squared_loss()`
evaluates the quadratic and `root_loss()` returns its nonnegative square root. The
object does not select units, physical interpretation, residual weights, normalization,
sample count, or scientific policy.

## Backend selection

```python
from projectkoios.optimization.quadratic.quadratic import (
    PositiveDefiniteQuadraticLoss,
)

dense = PositiveDefiniteQuadraticLoss(center, matrix, minimum)
sparse = PositiveDefiniteQuadraticLoss(center, matrix, minimum, sparse=True)
```

The facade imports no optional backend until construction. The default implementation
is
`quadratic.numpy.quadratic.NumpyPositiveDefiniteQuadraticLoss`; it stores immutable
tuple state and uses NumPy dense Cholesky validation and matrix-vector evaluation.

With `sparse=True`, the facade selects
`quadratic.scipy.quadratic.ScipySparsePositiveDefiniteQuadraticLoss`. It canonicalizes a
tuple or SciPy sparse input to private CSR storage, checks exact represented symmetry,
and uses the smallest algebraic eigenvalue reported by `scipy.sparse.linalg.eigsh` for
positive-definiteness acceptance. Iterative solver failure fails closed.

Sparse eigenvalue and dense Cholesky decisions are backend numerical acceptance, not
exact mathematical certificates. Both implementations reject nonfinite or negative
evaluated quadratic values and result overflow. Verification establishes software
behavior and bounded numerical agreement only, not scientific validation.

## Ownership

Optimization owns this reusable unit-free loss and its capability-local backend choice.
Scientific consumers adapt already normalized coordinates and matrices to this API.
PhysKit owns unit-bearing wrappers and dimensional conversion. Callers own loss meaning,
normalization, tolerances, thresholds, and acceptance policy.

## Periodic-1D migration architecture

Reusable two-dimensional rectangle minimization, full-ellipsoid support, and ordered
finite-family selection are specified in the
[constrained-quadratic architecture](constrained/index.md) and implemented beneath the
capability-local dense, sparse, factory, and selection modules.

Arbitrary-positive-dimensional minimization over a finite closed axis-aligned box is a
separate dense capability with a bounded deterministic primal active-set method,
active-bound classification, and independently reconstructed KKT evidence. Its reviewed
boundary is specified in the [box-quadratic architecture](box/index.md).

## Navigation

- [`PositiveDefiniteQuadraticLoss`](PositiveDefiniteQuadraticLoss/index.md)
- [Constrained-quadratic migration architecture](constrained/index.md)
- [Finite box minimization](box/index.md)
- [`WeightedAffineResidualQuadraticConstructor`](../affine/WeightedAffineResidualQuadraticConstructor/index.md)
- [Package ownership index](../index.md)
