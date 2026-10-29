# Implementation strategy

This draft organizes the KSD migration across two namespaces owned by this repository.
`projectkoios.numerics` owns reusable transformations, decompositions, mathematical
geometry, and numerical analysis. `projectkoios.optimization` owns algorithms whose
public contract minimizes, fits, selects, assigns, clusters, samples, or searches.
Physical and scientific interpretation is the responsibility of each consumer.

## Classification rule

Place a capability under `projectkoios.optimization` whenever its public operation
accepts a function or set of functions and selects arguments, parameters, candidates, or
transformations that optimize them. This includes operations that:

- minimize or maximize an objective;
- fit parameters or a model;
- selects candidates by objective values;
- solves an assignment objective;
- clusters observations;
- samples a decision space; or
- manages optimizer observations, state, and reports.

Place a capability under `projectkoios.numerics` when its public operation:

- transforms a representation;
- decomposes or factors an operator;
- constructs mathematical geometry;
- evaluates a norm, residual, derivative, or convergence diagnostic;
- constructs holonomy without selecting an optimizing transformation;
- evaluates finite-periodic geometry; or
- supplies deterministic numerical sequences and root-finding primitives.

A composition that selects transformations through internal objectives belongs to
Optimization. Reciprocal-frame transport therefore belongs to Optimization Alignment;
its mesh, sewing, frame, and holonomy calculations come from Numerics.

## Dependency direction

- `projectkoios.optimization` imports `projectkoios.numerics`.
- `projectkoios.numerics` never imports `projectkoios.optimization`.
- Consumers import the exact numerical or optimization capability they use.
- Neither namespace imports a consumer.
- Concrete dependency cycles are prohibited.

## Package-design rules

- Package initializers remain import-empty.
- Public imports name the defining module.
- Capability records live in dependency-free modules.
- Optional backends live beneath the capability they implement.
- Backend classes carry `Numpy`, `Scipy`, or `Sklearn` prefixes.
- Backend-neutral records carry no backend prefix.
- Stateless single operations use module functions.
- Configured algorithms with structured diagnostics use classes.
- Class suffixes state behavior: `Solver`, `Selector`, `Constructor`, `Calculator`,
  `Transporter`, `Matcher`, `Clusterer`, or `Sampler`.
- Result suffixes state output: `Result`, `Minimum`, `Support`, `Assignment`, or
  `Clustering`.
- `utils.py`, `helpers.py`, `common.py`, and behaviorless universal bases are prohibited.
- Ordering, tie behavior, tolerance meaning, and failure conditions are explicit.
- Scientific names, units, thresholds, and dispositions remain consumer responsibilities.

## Proposed source tree

```text
src/python/projectkoios/
├── numerics/
│   ├── __init__.py
│   ├── py.typed
│   ├── validation/
│   │   ├── __init__.py
│   │   └── values.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   └── diagonal_metrics.py
│   ├── linalg/
│   │   ├── __init__.py
│   │   ├── eigensystems.py
│   │   ├── subspaces.py
│   │   ├── orthonormalization.py
│   │   ├── norms.py
│   │   └── numpy/
│   ├── sequences/
│   │   ├── __init__.py
│   │   ├── radical_inverse.py
│   │   └── halton.py
│   ├── finite_periodic/
│   │   ├── __init__.py
│   │   ├── meshes.py
│   │   ├── transforms.py
│   │   ├── sewing.py
│   │   ├── frames.py
│   │   ├── wilson.py
│   │   ├── real_space.py
│   │   ├── symmetry.py
│   │   ├── density.py
│   │   ├── numpy/
│   │   └── scipy/
│   ├── numerical_analysis/
│   │   ├── __init__.py
│   │   ├── observed_order.py
│   │   ├── hessian.py
│   │   ├── roots.py
│   │   ├── numpy/
│   │   └── scipy/
│   └── statistics/
│       ├── __init__.py
│       ├── sandwich.py
│       └── numpy/
└── optimization/
    ├── __init__.py
    ├── affine/                    # compatibility path during realignment
    ├── least_squares/
    │   ├── __init__.py
    │   ├── affine.py
    │   ├── linear.py
    │   ├── matrix_subspaces.py
    │   ├── numpy/
    │   └── scipy/
    ├── alignment/
    │   ├── __init__.py
    │   ├── procrustes.py
    │   ├── frames.py
    │   ├── transport.py
    │   ├── phase_synchronization.py
    │   └── numpy/
    ├── quadratic/
    │   ├── __init__.py
    │   ├── quadratic.py
    │   ├── constrained.py
    │   ├── numpy/
    │   └── scipy/
    ├── selection/
    │   ├── __init__.py
    │   └── finite.py
    ├── assignment/
    │   ├── __init__.py
    │   ├── linear_sum.py
    │   ├── minimax.py
    │   └── scipy/
    ├── likelihood/
    │   ├── __init__.py
    │   ├── censored_log_normal.py
    │   └── scipy/
    ├── algorithms/
    ├── clustering/
    ├── engines/
    ├── multiobjective/
    │   └── pareto/
    ├── probability/
    ├── sampling/
    ├── spaces/
    └── reports/
```

The tree is a target map. A package or module appears only with its first accepted
behavior; no empty placeholder hierarchy is created.

## Optimization capabilities added by the migration

### Ordinary and weighted linear least squares

`projectkoios.optimization.least_squares.linear`

- `LinearLeastSquaresResult`
- `LinearLeastSquaresSolver` capability-local nominal abstract class
- `LinearLeastSquaresMethod`
- `LinearLeastSquaresSolverFactory`

`projectkoios.optimization.least_squares.numpy.linear`

- `NumpyLinearLeastSquaresSolver`

The solver accepts real or complex design and target matrices. Ordinary least squares
uses no weights; weighted least squares accepts finite strictly positive row weights.
One or multiple target columns use the same contract. The result carries coefficients,
fitted values, residuals, numerical rank, singular values, and full-rank condition
number.

### Real matrix-subspace fitting

`projectkoios.optimization.least_squares.matrix_subspaces`

- `RealMatrixSubspaceLeastSquaresResult`

`projectkoios.optimization.least_squares.numpy.matrix_subspaces`

- `NumpyRealMatrixSubspaceLeastSquaresSolver`

The consumer supplies an ordered real matrix basis. The result carries coefficients,
best-fit matrix, residual matrix, Frobenius residual, numerical rank, singular values,
and condition number. Model-class names and scientific adequacy remain external.

### Complex Procrustes and optimized alignment

`projectkoios.optimization.alignment.procrustes`

- `ComplexProcrustesResult`
- `BatchedComplexProcrustesResult`
- `ComplexProcrustesSolver` capability-local nominal abstract class
- `ComplexProcrustesMethod`
- `ComplexProcrustesSolverFactory`

`projectkoios.optimization.alignment.numpy.procrustes`

- `NumpyComplexProcrustesSolver`

Concrete operations are:

- `solve_unitary(source, target)`;
- `solve_partial_isometry(source, target)`;
- `solve_batched(sources, targets)`; and
- `solve_aggregate(sources, targets)`.

Results carry the transform, singular values, numerical rank, condition number, and
residual. Rank-deficient nonuniqueness and deterministic completion receive explicit
contracts before implementation.

`projectkoios.optimization.alignment.frames`, `transport`, and
`phase_synchronization` define:

- `FrameAlignmentResult`, `FrameAlignmentSolver`, and
  `NumpyFrameAlignmentSolver`;
- implemented `ClosedLoopPolarFrameTransportPrincipalPhaseBranch`,
  `NumpyClosedLoopPolarFrameTransportDefinition`,
  `NumpyClosedLoopPolarFrameTransportResult`, and the explicitly composed
  `ScipyClosedLoopPolarFrameTransportSolver`; and
- future `PhaseSynchronizationResult` and `NumpyPhaseSynchronizationSolver` for
  synchronization behavior beyond one closed loop.

These operations select transforms or phases by optimizing explicit alignment or
synchronization objectives.

### Constrained quadratic minimization

Existing `projectkoios.optimization.quadratic` remains the owner of quadratic objectives.

`projectkoios.optimization.quadratic.constrained`

- `ConstrainedQuadraticTolerance`
- `RectangleQuadraticMinimum`
- `RectangleQuadraticMinimizer2D`
- `QuadraticSublevelSupportResult`
- `QuadraticSublevelSupportSolver`

`projectkoios.optimization.quadratic.factory`

- `RectangleQuadraticMinimizerMethod`
- `RectangleQuadraticMinimizerFactory`
- `QuadraticSublevelSupportMethod`
- `QuadraticSublevelSupportSolverFactory`

`projectkoios.optimization.quadratic.numpy.constrained`

- `NumpyRectangleQuadraticMinimizer2D`
- `NumpyQuadraticSublevelSupportSolver`

`projectkoios.optimization.quadratic.scipy.constrained`

- `ScipySparseRectangleQuadraticMinimizer2D`
- `ScipySparseQuadraticSublevelSupportSolver`

The closed factory methods are `NUMPY_DENSE` and `SCIPY_SPARSE`. The `2D` suffix
exposes the rectangle algorithm's exact dimensional limit. Dense and sparse backends
share candidate order, tie behavior, result contracts, and failures. A capability-
specific absolute/relative tolerance governs final backend-result validation and bounded
parity tests; it never changes strict candidate ranking or scientific acceptance.

This slice intentionally stops at fixed-angle convex quadratics and ordered finite-family
enumeration. It adds no generic nonconvex optimizer, continuous alignment, reusable
box-clipped-set-distance solver, phase map, KSD adapter, or publication policy. KSD owns
scientific metric selection, admissible-set and requested-distance definitions, backend
qualification against retained proof objects, immutable owner-version binding, and
independent certificate reconstruction. A reusable distance algorithm requires separate
Optimization architecture and must not be reimplemented in KSD.

### Finite objective selection

`projectkoios.optimization.selection.finite`

- `FiniteMinimum`
- `FiniteAmbiguitySet`
- `first_minimum(values)`
- `indices_at_or_below(values, ceiling)`
- `absolute_ambiguity_set(values, tolerance)`

The functions consume objective values, preserve declaration order, and return numeric
indices. Consumers retain candidate labels and scientific dispositions.

### Assignment

`projectkoios.optimization.assignment.linear_sum`

- `LinearSumAssignment`
- `LinearSumAssignmentSolver` capability-local nominal abstract class
- `LinearSumAssignmentMethod`
- `LinearSumAssignmentSolverFactory`

`projectkoios.optimization.assignment.scipy.linear_sum`

- `ScipyLinearSumAssignmentSolver`

`projectkoios.optimization.assignment.minimax`

- `MinimaxAssignment`
- `MinimaxAssignmentSolver` capability-local nominal abstract class
- `MinimaxAssignmentMethod`
- `MinimaxAssignmentSolverFactory`

`projectkoios.optimization.assignment.scipy.minimax`

- `ScipyMinimaxAssignmentSolver`

Linear-sum and minimax objectives remain separate modules and types.

### Likelihood-based fitting

`projectkoios.optimization.likelihood.censored_log_normal`

- `RightCensoredLogNormalAftResult`
- `RightCensoredLogNormalAftSolver` capability-local nominal abstract class
- `RightCensoredLogNormalAftMethod`
- `RightCensoredLogNormalAftSolverFactory`

`projectkoios.optimization.likelihood.scipy.censored_log_normal`

- `ScipyRightCensoredLogNormalAftSolver`

Optimization owns the likelihood objective, bounded parameter fit, fitted parameters,
convergence result, score, and gradient. Numerics owns reusable Hessian, clustered
sandwich covariance, conditioning, and interval calculations. Consumers own the meaning
of censoring, predictors, clusters, and reported conclusions.

### Representative clustering

`projectkoios.optimization.clustering.representatives`

- `RepresentativeCluster`
- `RepresentativeClustering`
- `RepresentativeClusterer`

The clusterer preserves input order, retains the first representative, and returns every
membership index. The caller supplies the equivalence predicate or distance and
threshold.

### Halton optimization sampling

`projectkoios.numerics.sequences.halton`

- `HaltonSequence`

`projectkoios.optimization.sampling.halton`

- `HaltonSampler`
- `HaltonSamplerState`

Numerics owns sequence generation. Optimization maps unit-cube points into a
`ParameterSpace`, produces decision vectors, and owns resumable sampler state.

## Numerics capabilities added by the migration

### Eigensystems and subspaces

`projectkoios.numerics.linalg.eigensystems`

- `SymmetricEigensystem`

`projectkoios.numerics.linalg.numpy.eigensystems`

- `NumpySymmetricEigensolver`

`projectkoios.numerics.linalg.subspaces`

- `SpectralSubspace`
- `OperatorCompression`

`projectkoios.numerics.linalg.numpy.subspaces`

- `NumpySpectralSubspaceSelector`
- `orthogonal_projector(frame)`
- `orthogonal_complement(projector)`
- `compress_operator(frame, operator)`
- `embed_operator(frame, reduced_operator)`

Projection, basis pullback, compression, and truncation remain distinct operations.

### Symmetric orthonormalization

`projectkoios.numerics.linalg.orthonormalization`

- `SymmetricOrthonormalizationResult`

`projectkoios.numerics.linalg.numpy.orthonormalization`

- `NumpySymmetricOrthonormalizer`

The result carries the Gram matrix, inverse square root, injection, condition number, and
isometry defect.

### Deterministic sequences

`projectkoios.numerics.sequences.radical_inverse`

- `radical_inverse(index, base)`

`projectkoios.numerics.sequences.halton`

- `HaltonSequence`

`HaltonSequence.point(index)` is deterministic and random-state free.

### Finite-periodic numerics

`projectkoios.numerics.finite_periodic` owns:

- mesh neighbors and reciprocal translations;
- finite Fourier transforms and truncation;
- reciprocal sewing maps;
- frame, projector, and projected-operator paths;
- Wilson products, phase spectra, circular-residual cost construction, unwrapping, and
  winding;
- direct finite-supercell real-space reconstruction and moment diagnostics for supplied
  states;
- finite group actions and periodic-center geometry; and
- deterministic density reconstruction.

Primary records and operations include:

- `PeriodicMeshNeighbor`;
- `ReciprocalSewingMap`;
- `OrthonormalFramePath`;
- `ProjectedOperatorPath`;
- `WilsonLoopResult`;
- `PhaseSpectrum`;
- `CircularPhaseCostMatrix`;
- `FinitePeriodicRealSpaceMomentResult`; and
- `PeriodicDensityResult`.

Backend implementations include:

- `NumpyFinitePeriodicTransformer`;
- `NumpyReciprocalSewingConstructor`;
- `NumpyProjectedOperatorPathConstructor`;
- `NumpyWilsonLoopCalculator`;
- `NumpyFinitePeriodicRealSpaceMomentCalculator`; and
- `NumpyPeriodicDensityReconstructor`.

Optimized transport, phase synchronization, and phase matching consume these numerical
records from `projectkoios.optimization`; Numerics has no reverse dependency. Direct
real-space reconstruction accepts already supplied states and does not optimize a
Wannier gauge. Wannier localization remains Wannier90-owned.

### Numerical analysis

`projectkoios.numerics.numerical_analysis.observed_order`

- `observed_order(errors, scales)`

`projectkoios.numerics.numerical_analysis.hessian`

- `FiniteDifferenceHessianResult`
- `NumpyCenteredFiniteDifferenceHessianCalculator`

`projectkoios.numerics.numerical_analysis.roots`

- `BracketedRootResult`
- `BisectionRootFinder`

### Statistical diagnostics

`projectkoios.numerics.statistics.sandwich`

- `ClusteredSandwichCovariance`

Optimization owns right-censored likelihood fitting. Numerics owns reusable Hessian,
covariance, conditioning, and interval diagnostics that do not select fitted parameters.

## Existing-package realignment

### Move to Numerics

- `optimization.validation` → `numerics.validation`.
- `optimization.metric` → `numerics.geometry.diagonal_metrics`.

### Reorganize inside Optimization

- `optimization.affine` → canonical
  `optimization.least_squares.affine`, with compatibility modules at the old path.
- `optimization.quadratic` remains canonical for objective representation and quadratic
  minimization.
- `optimization.clustering` remains canonical for clustering algorithms.
- Protected Pareto behavior remains under
  `projectkoios.optimization.multiobjective.pareto`.

### Compatibility rule

- Define each class in exactly one canonical module.
- Old paths import the canonical class during the compatibility window.
- Package initializers remain import-empty.
- Compatibility modules contain no duplicate implementation.
- Compatibility tests assert object identity between old and canonical imports.
- Removing compatibility paths requires a separate API decision.

## Human- and LLM-readable imports

```python
from projectkoios.optimization.least_squares.numpy.linear import (
    NumpyLinearLeastSquaresSolver,
)
from projectkoios.optimization.alignment.numpy.procrustes import (
    NumpyComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.quadratic.numpy.constrained import (
    NumpyRectangleQuadraticMinimizer2D,
)
```

```python
from projectkoios.numerics.finite_periodic.numpy.sewing import (
    NumpyReciprocalSewingConstructor,
)
from projectkoios.optimization.alignment.closed_loop_polar_transport import (
    ClosedLoopPolarFrameTransportPrincipalPhaseBranch,
)
from projectkoios.optimization.alignment.numpy.closed_loop_polar_transport import (
    NumpyClosedLoopPolarFrameTransportDefinition,
)
from projectkoios.optimization.alignment.scipy.closed_loop_polar_transport import (
    ScipyClosedLoopPolarFrameTransportSolver,
)
from projectkoios.numerics.finite_periodic.numpy.wilson import (
    NumpyWilsonLoopCalculator,
)
```

Each import communicates namespace, capability, backend, and operation without an alias
table.

## Testing layout

Tests mirror source paths:

```text
tests/projectkoios/
├── numerics/
│   ├── linalg/
│   ├── finite_periodic/
│   └── numerical_analysis/
└── optimization/
    ├── least_squares/
    ├── alignment/
    ├── quadratic/
    ├── selection/
    └── assignment/
```

- Unit tests verify dependency-free records and validation.
- Integration tests verify optional backend behavior.
- Numerical tests compare independent formulas and owner-native regression fixtures.
- Property tests use deterministic generated cases.
- Fixtures contain owner-native numeric regression values, not KSD objects.
- No test imports KSD at runtime.

## Documentation layout

```text
docs/architecture/projectkoios/
├── numerics/
│   ├── index.md
│   ├── linalg/
│   ├── finite_periodic/
│   ├── numerical_analysis/
│   └── statistics/
└── optimization/
    ├── index.md
    ├── least_squares/
    ├── alignment/
    ├── quadratic/
    ├── selection/
    └── assignment/
```

- Namespace indexes list current capabilities only.
- Capability pages identify contracts, implementations, failures, and ownership.
- Class pages mirror source ownership.
- Task pages retain sequencing and migration status.

## Packaging and static analysis

- Package both namespaces in the existing `projectkoios-optimization` distribution.
- Add `src/python/projectkoios/numerics/py.typed`.
- Expand strict mypy coverage to `src/python/projectkoios`.
- Preserve dependency-free installation.
- Keep NumPy, SciPy, and scikit-learn optional.
- Add isolated imports for both namespaces to wheel tests.
- Verify that importing either namespace loads no optional backend.
- Add dependency checks that prohibit Numerics-to-Optimization imports.

## Delivery sequence

### Slice 0 — namespace foundation

- Add `projectkoios.numerics` and its architecture index.
- Move shared validation and diagonal metric geometry.
- Add compatibility modules and identity tests.
- Expand packaging and strict typing gates.

### Slice 1 — M3 constrained quadratic optimization

Detailed plan: [constrained-quadratic implementation specification](slice-01-constrained-quadratic.md).

- Replace the private quadratic backend structural protocol with a nominal abstract
  backend while preserving public facade behavior.
- Add constrained-quadratic records beneath `projectkoios.optimization.quadratic`.
- Add exact two-dimensional dense and sparse rectangle minimizers.
- Add dense and sparse full-ellipsoid support solvers.
- Verify dense/sparse agreement and sparse no-densification.
- Add finite objective selection.
- Add M3 regression tests without a KSD runtime dependency.
- Stop at the capability acceptance gate.

### Slice 2 — ordinary and weighted linear least squares

- Review the completed M1 implementation and verification evidence.
- Add ordinary and weighted real/complex least-squares records and implementations
  beneath `projectkoios.optimization.least_squares`.
- Verify uniform, nonuniform-weight, incomplete-design, rank, condition, and residual
  behavior.
- Stop at the capability acceptance gate.

### Slice 3 — M2 and blind-alignment Procrustes

- Review the completed M2 and blind-alignment implementation and verification evidence.
- Add square and rectangular Procrustes beneath `projectkoios.optimization.alignment`.
- Verify pointwise, batched, aggregate, full-rank, and rank-deficient behavior.
- Stop at the capability acceptance gate.

### Later slices

- Deliver each fitting, assignment, sequence, clustering, finite-periodic,
  numerical-analysis, and statistical capability independently.
- Write and review architecture before each implementation.
- Stop at the acceptance gate after every slice.

## Acceptance criteria

- Every completed behavior maps to one proposed owner module.
- Every class has one canonical defining module.
- Optimization owns every capability that selects arguments or parameters to optimize a
  function or set of functions, including least squares, Procrustes, likelihood fitting,
  assignment, and finite selection.
- Numerics owns transformations, decompositions, geometry, and numerical analysis.
- Namespace dependencies remain one-way from Optimization to Numerics.
- Protected Pareto code remains unmoved.
- Package initializers remain import-empty.
- Optional backend imports remain lazy and capability-local.
- Consumer interpretation does not enter reusable records.
- Compatibility paths contain no duplicate behavior.
- Tests and documentation mirror source ownership.
- Implementation proceeds only in reviewed, independently verified slices.
