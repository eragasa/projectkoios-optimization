# `PositiveDefiniteQuadraticLoss`

## Purpose

`projectkoios.optimization.adapters.PositiveDefiniteQuadraticLoss` is the public
object-oriented facade for the unit-free completed-square loss

$$
q(x)=q_{\min}+(x-c)^TQ(x-c).
$$

The object owns backend selection, validated finite state, squared-loss evaluation, and
root-loss evaluation. It does not assign units, choose residual weights, normalize
samples, or make scientific decisions.

## Construction

```python
from projectkoios.optimization.adapters import PositiveDefiniteQuadraticLoss

dense = PositiveDefiniteQuadraticLoss(center, matrix, minimum_squared_loss)
sparse = PositiveDefiniteQuadraticLoss(
    center,
    matrix,
    minimum_squared_loss,
    sparse=True,
)
```

The default requires the `numpy` extra. `sparse=True` requires the `scipy` extra.
Scientific consumers adapt their already normalized coordinates and matrices to this
owner API.

## Pages

- [Schematics](schematics.md)
- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Quadratic adapter namespace](../quadratic/index.md)
