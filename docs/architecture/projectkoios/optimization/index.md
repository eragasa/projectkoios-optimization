# `projectkoios.optimization`

## Ownership index

| Capability | Current implementation | Owned behavior |
|---|---|---|
| Validation | [`validation/`](../../../../src/python/projectkoios/optimization/validation/) | finite scalar and identifier boundaries shared by maintained numerical capabilities |
| Engines | [`engines/`](../../../../src/python/projectkoios/optimization/engines/) | injected evaluation/result handling and scalar weighted handling |
| Multi-objective | [`multiobjective/`](../../../../src/python/projectkoios/optimization/multiobjective/) | optimizer base, Pareto front/set pullback, and optimized PyPosPack selection |
| Parameter spaces | [`spaces/`](../../../../src/python/projectkoios/optimization/spaces/) | finite ordered closed parameter boxes |
| Algorithms | [`algorithms/`](../../../../src/python/projectkoios/optimization/algorithms/) | scalar coordinate search and intrinsic typed state |
| Sampling | [`sampling/`](../../../../src/python/projectkoios/optimization/sampling/) | uniform sampling, PyPosPack-lineage global and GMM-conditioned KDE, owner-authored shell contrast, and generic scalar sampling contracts |
| Probability | [`probability/`](../../../../src/python/projectkoios/optimization/probability/) | dependency-free distributions and sampling plus capability-local NumPy/SciPy implementations |
| Affine residuals | [`affine/`](../../../../src/python/projectkoios/optimization/affine/) | immutable weighted affine residuals, checked quadratic inputs, and capability-local NumPy/SciPy constructors |
| Quadratic loss | [`quadratic/`](../../../../src/python/projectkoios/optimization/quadratic/) | unit-free completed-square facade and capability-local NumPy dense/SciPy sparse implementations |
| Metric geometry | [`metric/`](../../../../src/python/projectkoios/optimization/metric/) | positive diagonal unit-free primal and dual geometry over ordered tuples |
| Reports | [`reports/`](../../../../src/python/projectkoios/optimization/reports/) | deterministic scalar ranking using the engine block's Pareto selector |

## Import boundary

`projectkoios.optimization.__init__` intentionally imports no capabilities. Capability
and backend package initializers are also import-empty. Consumers import the exact
owning module they use, for example:

```python
from projectkoios.optimization.affine.affine import WeightedAffineResidualDefinition
from projectkoios.optimization.affine.numpy.affine import (
    NumpyWeightedAffineResidualQuadraticConstructor,
)
from projectkoios.optimization.quadratic.quadratic import (
    PositiveDefiniteQuadraticLoss,
)
```

This prevents package integration from eagerly importing unrelated behavior or optional
NumPy/SciPy backends. The core package remains dependency-free; optional implementations
are located beneath the capability they implement.

The package has no calculator, filesystem, subprocess, network, dispatch, retry,
persistence, or workflow authority. Atomistic and potential models, campaigns, GMM
plotting and scientific analysis, unrelated PyPosPack/PyPosMat machinery, and PyMatMC2
are outside this repository. The retained PyPosPack Pareto and GMM/KDE numerical
behaviors are provenance- and license-bound under their owning capabilities.

See [multi-objective optimization](multiobjective/index.md),
[engines](engines/index.md), [parameter spaces](spaces/index.md),
[affine residuals](affine/index.md), [quadratic loss](quadratic/index.md),
[diagonal metric geometry](metric/index.md), and
[source mapping](../../../provenance/source-mapping.md) for detailed contracts and
lineage evidence.
