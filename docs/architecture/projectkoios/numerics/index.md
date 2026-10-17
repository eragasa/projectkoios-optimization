# `projectkoios.numerics`

`projectkoios.numerics` owns reusable mathematical transformations, decompositions,
geometry, and numerical analysis that do not select an optimum through their public
contract.

## Status

The finite Fourier-series evaluator is implemented at owner commit
`6c3f73cd5f1810c8de915e1559658ea4b79c2357`. Invariant-only Hermitian
spectral-subspace decomposition is implemented and independently qualified locally under
explicit authorization; it is not committed or bound downstream. Other capabilities remain architecture candidates unless their pages
explicitly identify implemented source.

## Capability organization

| Capability | Package | Public operation |
|---|---|---|
| Validation | `validation` | normalize finite dependency-free inputs |
| Linear algebra | `linalg` | decompose matrices and construct subspaces |
| Geometry | `geometry` | evaluate primal and dual mathematical geometry |
| Deterministic sequences | `sequences` | calculate indexed low-discrepancy points |
| Finite-periodic numerics | `finite_periodic` | transform, sew, project, calculate holonomy, reconstruct supplied states, and calculate real-space moments |
| Numerical analysis | `numerical_analysis` | evaluate convergence, derivatives, tensors, and roots |
| Statistical diagnostics | `statistics` | calculate covariance, conditioning, intervals, and probabilities |

## Boundary examples

- Symmetric eigensystem decomposition belongs here.
- A least-squares fit belongs in Optimization.
- Reciprocal sewing-map construction belongs here.
- Procrustes alignment belongs in Optimization.
- Wilson-product and phase-spectrum calculation belongs here.
- Assignment of one phase multiset to another belongs in Optimization.
- Halton point generation belongs here.
- Mapping Halton points into decision candidates belongs in Optimization sampling.

## Import boundary

```python
from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
    FiniteFourierSeriesEvaluatorFactory,
    FiniteFourierSeriesMethod,
)
```

Package initializers remain import-empty. Consumers import exact defining modules.

## Pages

- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Linear algebra](linalg/index.md)
- [Finite-periodic numerics](finite_periodic/index.md)
