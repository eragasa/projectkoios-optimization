# Project Koios Optimization

`projectkoios-optimization` owns reusable mathematical numerics and optimization under
`projectkoios.numerics` and `projectkoios.optimization`.

The package provides explicit ask/tell engines, concrete typed optimizer state,
SplitMix64 uniform-box sampling, generic sampling and probability contracts, weighted
squared loss, weighted affine residual definitions, positive diagonal metric geometry,
bounded coordinate pattern search, iterative Pareto optimization over interchangeable
samplers, standalone Gaussian-mixture clustering, recursive Pareto-shell pullback,
owner-authored shell-contrastive KDE sampling, and deterministic ranking reports.
Capability-local optional implementations provide NumPy/SciPy probability behavior,
PyPosPack-lineage global and GMM-conditioned Gaussian KDE parameter sampling,
KSD-lineage BIC-selected Gaussian-mixture clustering, positive-definite quadratic losses,
and affine-residual quadratic construction. Linear-algebra numerics provide dense batched
complete Hermitian spectra through NumPy and SciPy, plus sparse partial Hermitian spectra
through SciPy ARPACK with explicit convergence controls and residual evidence.
Finite-periodic numerics provide shared-grid batched positive-exponent Fourier-series
evaluation with point-major values and one dense NumPy channel reduction.

Package initializers intentionally perform no eager capability or backend imports.
Consumers import exact owning modules:

```python
from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.numpy.batched_finite_series import (
    NumpyBatchedFiniteFourierSeriesEvaluator,
)
from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
)
from projectkoios.numerics.linalg.numpy.hermitian_spectrum import (
    NumpyBatchedHermitianSpectrumSolver,
)
from projectkoios.numerics.linalg.scipy.hermitian_spectrum import (
    ScipyBatchedHermitianSpectrumSolver,
)
from projectkoios.numerics.linalg.scipy.sparse_hermitian_spectrum import (
    ScipySparseHermitianPartialSpectrumSolver,
    scipy_sparse_hermitian_csr_matrix,
)
from projectkoios.numerics.linalg.sparse_hermitian_spectrum import (
    SparseHermitianPartialSpectrumDefinition,
    SparseHermitianSpectrumSelection,
)

from projectkoios.optimization.sampling.uniform import UniformRandomSampler
from projectkoios.optimization.clustering.scipy.gmm import (
    ScipyGaussianMixtureClusterer,
)
from projectkoios.optimization.sampling.scipy.gmm_kde import (
    CovarianceStabilizationMethod,
    ScipyGmmConditionedKdeSampler,
)
from projectkoios.optimization.sampling.scipy.kde import (
    KdeBandwidthMethod,
    ScipyGaussianKdeSampler,
)
from projectkoios.optimization.sampling.scipy.pareto_shells import (
    ScipyParetoShellContrastiveKdeSampler,
)
from projectkoios.optimization.affine.affine import WeightedAffineResidualDefinition
from projectkoios.optimization.affine.numpy.affine import (
    NumpyWeightedAffineResidualQuadraticConstructor,
)
from projectkoios.optimization.multiobjective.factory import (
    MultiObjectiveOptimizerFactory,
)
from projectkoios.optimization.probability.factory import ProbabilitySamplerFactory
from projectkoios.optimization.metric.metric import PositiveDiagonalMetric
from projectkoios.optimization.quadratic.quadratic import (
    PositiveDefiniteQuadraticLoss,
)
```

NumPy implementations live beneath each capability's `numpy` namespace; SciPy
implementations live beneath its `scipy` namespace. Importing
`projectkoios.optimization` alone imports neither capability modules nor optional
backends. Affine-quadratic construction requires numerical full column rank under the
selected backend's direct weighted-design least-squares policy and never regularizes a
rejected system. Affine definitions and diagonal metrics consume ordered numeric tuples;
scientific coordinate names, channel alignment, units, and nondimensionalization remain
consumer-owned.

Forward evaluation and result handling remain separate injected ports. The package does
not execute calculators, filesystems, subprocesses, networks, workflows, dispatch,
retries, or persistence.

The provenance-bound `projectkoios.optimization.multiobjective.pareto` block originates
in Eugene J. Ragasa's PhD work preserved across PyPosPack and multiple research
repositories. Its iterative Pareto optimizer can use a different injected sampler at
each stage. The
performance-sensitive PyPosPack Pareto routine retains adaptive chunk pruning, with only
the incomplete-final-batch defect corrected. This protected block must not be moved,
split, or refactored without explicit human authorization. The recursive shell pullback
and contrastive sampler are separately documented owner-authored research additions;
they are not represented as recovered PyPosPack behavior. The reconstructed
GMM-conditioned KDE remains separately provenance-bound to the historical GMM and
cluster-sampling pipeline. Standard and irregular forward-evaluation comparisons are
documented in [`benchmarks/README.md`](benchmarks/README.md).

See [`TRANSFER.toml`](TRANSFER.toml) and
[`docs/provenance/source-mapping.md`](docs/provenance/source-mapping.md) for recovery and
design-evidence mapping. Verification is software and bounded numerical verification
only; it is not scientific validation.

## Development

```bash
PYTHONPATH=src/python python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/python/projectkoios/optimization
python -m build --sdist --wheel
```
