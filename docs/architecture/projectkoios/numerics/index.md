# `projectkoios.numerics`

`projectkoios.numerics` owns reusable mathematical transformations, decompositions,
geometry, and numerical analysis that do not select an optimum through their public
contract.

## Status

Finite-series evaluation, centered uniform finite-mesh matrix transforms, uniform
closed-interval trapezoidal quadrature, Frobenius values, dense
maximum-absolute-entry and spectral norms, the nominal matrix-norm contract foundation,
Hermitian spectral-subspace decomposition, dense complete Hermitian spectra, and sparse
partial Hermitian spectra are
implemented. Other capabilities remain
architecture candidates unless their pages explicitly identify implemented source.

## Capability organization

| Capability | Package | Public operation |
|---|---|---|
| Validation | `validation` | normalize finite dependency-free inputs |
| Linear algebra | `linalg` | decompose matrices and construct subspaces |
| Geometry | `geometry` | evaluate primal and dual mathematical geometry |
| Deterministic sequences | `sequences` | calculate indexed low-discrepancy points |
| Finite-periodic numerics | `finite_periodic` | transform, sew, project, calculate holonomy, reconstruct supplied states, and calculate real-space moments |
| Quadrature | `quadrature` | construct unit-free integration weights and diagonal mass algebra |
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
from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.finite_periodic.numpy.matrix_evaluation import (
    NumpyFiniteFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.tensor_product_evaluation import (
    NumpyTensorProductFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformer,
)
from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)
from projectkoios.numerics.quadrature.numpy.uniform_1d import (
    NumpyUniformClosedIntervalTrapezoidalRule,
)
```

Package initializers remain import-empty. Consumers import exact defining modules.

## Pages

- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Dependency-free numerical typing](typing/index.md)
- [Dependency-free numerical validation](validation/index.md)
- [NumPy numerical support](numpy/index.md)
- [Linear algebra](linalg/index.md)
- [Finite-periodic numerics](finite_periodic/index.md)
- [Uniform one-dimensional quadrature](quadrature/index.md)
