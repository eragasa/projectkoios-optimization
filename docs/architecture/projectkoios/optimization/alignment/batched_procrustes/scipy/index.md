# SciPy batched Procrustes status

No SciPy batched Procrustes contract or solver is currently implemented or exported.
This chapter records the boundary explicitly so users do not infer backend availability
from other SciPy capabilities in the distribution.

## Current guidance

- Dense exact complex128 batches use the [NumPy backend](../numpy/index.md).
- Consumers must not import or probe a nonexistent SciPy solver.
- A future backend requires a concrete consumer need, an explicit representation
  contract, independent numerical qualification, and wheel verification.

## Pages

- [Implementation status](implementation.md)
- [Prospective numerical requirements](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
