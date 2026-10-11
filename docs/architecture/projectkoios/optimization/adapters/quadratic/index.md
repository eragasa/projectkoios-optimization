# Positive-definite quadratic loss adapters

`projectkoios.optimization.adapters.PositiveDefiniteQuadraticLoss` is the public
object-oriented facade for a unit-free completed-square loss

$$
q(x)=q_{\min}+(x-c)^TQ(x-c).
$$

Construction requires a nonempty finite center, a finite exactly symmetric matrix of
the same dimension, and a finite nonnegative minimum squared loss. `squared_loss`
evaluates the quadratic and `root_loss` returns its nonnegative square root. The object
does not select units, physical interpretation, residual weights, normalization,
sample count, or scientific policy.

## Backend selection

The facade owns backend selection at construction:

```python
from projectkoios.optimization.adapters import PositiveDefiniteQuadraticLoss

dense = PositiveDefiniteQuadraticLoss(center, matrix, minimum)
sparse = PositiveDefiniteQuadraticLoss(center, matrix, minimum, sparse=True)
```

The default implementation is
`adapters.numpy.quadratic.NumpyPositiveDefiniteQuadraticLoss`; it stores immutable tuple
state and uses NumPy dense Cholesky validation and matrix-vector evaluation. It requires
the `numpy` extra.

With `sparse=True`, the facade constructs
`adapters.scipy.quadratic.ScipySparsePositiveDefiniteQuadraticLoss`. It canonicalizes a
tuple or SciPy sparse input to private CSR storage, checks exact represented symmetry,
and uses the smallest algebraic eigenvalue reported by `scipy.sparse.linalg.eigsh` for
positive-definiteness acceptance. Iterative solver failure fails closed. It requires the
`scipy` extra, which also installs NumPy.

The sparse eigenvalue decision is backend numerical acceptance, not an exact
mathematical certificate. Both implementations reject nonfinite or negative evaluated
quadratic values and result overflow. Verification establishes software behavior and
bounded numerical agreement only, not scientific validation.

## Ownership

Optimization owns this reusable unit-free loss and its backend choice. A consuming
scientific repository adapts its already normalized coordinates and matrices to this
API. PhysKit owns unit-bearing wrappers and dimensional conversion. Scientific callers
own the meaning of the loss, normalization, tolerances, thresholds, and acceptance
policy.

## Navigation

- [`PositiveDefiniteQuadraticLoss`](../PositiveDefiniteQuadraticLoss/index.md)
- [Package ownership index](../../index.md)
