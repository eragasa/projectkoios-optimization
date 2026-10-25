# Project Koios Optimization

`projectkoios-optimization` owns reusable mathematical numerics and optimization for
Project Koios. Mathematical transformations live under `projectkoios.numerics`, while
fitting and search policy remain under `projectkoios.optimization`.

The package provides explicit ask/tell engines, concrete typed optimizer state,
SplitMix64 uniform-box sampling, generic sampling and probability contracts, weighted
squared loss, weighted affine residual definitions, positive diagonal metric geometry,
bounded coordinate pattern search, iterative Pareto optimization over interchangeable
samplers, standalone Gaussian-mixture clustering, recursive Pareto-shell pullback,
owner-authored shell-contrastive KDE sampling, and deterministic ranking reports.
Capability-local optional implementations provide NumPy/SciPy probability behavior,
PyPosPack-lineage global and GMM-conditioned Gaussian KDE parameter sampling,
KSD-lineage BIC-selected Gaussian-mixture clustering, positive-definite quadratic losses,
exact two-dimensional rectangle minimization, dense and sparse full-ellipsoid support,
deterministic finite-objective selection, ordinary and positive-weight real/complex
multiple-right-hand-side linear least squares, affine-residual quadratic construction,
one-pair and tensor-batched complex-unitary right Procrustes alignment,
positive-exponent finite Fourier-series evaluation at supplied real coordinates,
shared-grid batched finite Fourier evaluation with bounded phase batches, parity-generic
centered uniform one-, two-, and three-dimensional matrix Fourier transforms, bounded selected-
mode matrix evaluation, Parseval and cyclic-Hermiticity diagnostics, endpoint-inclusive
uniform trapezoidal quadrature with explicit homogeneous-Dirichlet restriction and
diagonal mass algebra, invariant dense Hermitian spectral-subspace decomposition with complete-spectrum output
and selected-projector construction, dense batched complete Hermitian-spectrum
evaluation, and sparse partial Hermitian-spectrum selection with explicit convergence
controls and residual evidence. A dependency-free nominal matrix-norm foundation defines
the shared semantic enum and callable result-validation boundary; concrete dense and CSR
norm evaluators remain separately pending.

Package initializers intentionally perform no eager capability or backend imports.
Consumers import exact owning modules:

```python
from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.numpy.batched_finite_series import (
    NumpyBatchedFiniteFourierSeriesEvaluator,
)
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
    NumpyTensorProductFourierMatrixEvaluationDefinition,
    NumpyTensorProductFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
    NumpyCenteredUniformMeshMatrixFourierTransformer,
)
from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceDefinition,
    HermitianSpectralSubspaceMethod,
    HermitianSpectralSubspaceSolverFactory,
    HermitianSpectralSubspaceTolerance,
)
from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
)
from projectkoios.numerics.linalg.matrix_norm import (
    MatrixNormABC,
    MatrixNormEnum,
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
from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)
from projectkoios.numerics.quadrature.numpy.uniform_1d import (
    NumpyUniformClosedIntervalTrapezoidalRule,
)

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesTolerance,
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
from projectkoios.optimization.least_squares.factory import (
    LinearLeastSquaresMethod,
    LinearLeastSquaresSolverFactory,
)
from projectkoios.optimization.multiobjective.factory import (
    MultiObjectiveOptimizerFactory,
)
from projectkoios.optimization.probability.factory import ProbabilitySamplerFactory
from projectkoios.optimization.metric.metric import PositiveDiagonalMetric
from projectkoios.optimization.quadratic.factory import (
    QuadraticSublevelSupportMethod,
    QuadraticSublevelSupportSolverFactory,
    RectangleQuadraticMinimizerFactory,
    RectangleQuadraticMinimizerMethod,
)
from projectkoios.optimization.quadratic.quadratic import (
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.selection.finite import (
    first_minimum,
    indices_at_or_below,
)
```

NumPy implementations live beneath each capability's `numpy` namespace; SciPy
implementations live beneath its `scipy` namespace. Importing
`projectkoios.optimization` alone imports neither capability modules nor optional
backends. Affine-quadratic construction requires numerical full column rank under the
selected backend's direct weighted-design least-squares policy and never regularizes a
rejected system. Affine definitions and diagonal metrics consume ordered numeric tuples;
scientific coordinate names, channel alignment, units, and nondimensionalization remain
consumer-owned. Constrained-quadratic factories expose explicit dense and sparse
methods; the rectangle solver remains exactly two-dimensional, support applies to the
complete ellipsoid without a box-containment claim, and finite selection preserves
first-on-exact-tie declaration order. Linear least squares consumes unit-free ordered
complex-capable tuple matrices, uses square-root row scaling for strictly positive
weights, and reports rank deficiency rather than converting it to a full-rank fit.

Forward evaluation and result handling remain separate injected ports. The package does
not execute calculators, filesystems, subprocesses, networks, workflows, dispatch,
retries, or persistence.

The provenance-bound `projectkoios.optimization.multiobjective.pareto` block originates
in Eugene J. Ragasa's PhD work. PyPosPack and PyMatMC2 are the more authoritative
historical mechanism repositories; KSD copies are recovery evidence rather than
mechanism origin. PyPosPack is the exact source authority for the transferred selector
and iterative sampling policy, while no PyMatMC2 source is currently transferred. The
protected block, its exact identities, adaptations, and owner-authored shell extensions
are documented in the
[Pareto architecture record](docs/architecture/projectkoios/optimization/multiobjective/pareto/index.md).
The reconstructed GMM-conditioned KDE remains separately provenance-bound to the
historical GMM and cluster-sampling pipeline. Standard and irregular forward-evaluation
comparisons are documented in [`benchmarks/README.md`](benchmarks/README.md).

See [`TRANSFER.toml`](TRANSFER.toml) and
[`docs/provenance/source-mapping.md`](docs/provenance/source-mapping.md) for recovery and
design-evidence mapping. Verification is software and bounded numerical verification
only; it is not scientific validation.

## Development

```bash
PYTHONPATH=src/python python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict \
  src/python/projectkoios/numerics \
  src/python/projectkoios/optimization \
  tests/projectkoios/numerics/linalg
python -m build --sdist --wheel
```
