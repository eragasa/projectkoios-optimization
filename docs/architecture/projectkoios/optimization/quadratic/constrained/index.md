# Periodic-1D constrained-quadratic migration architecture

## Status

This page defines the owner architecture for the first reusable optimization slice from
KSD's periodic-1D toy-model work. The capability is implemented and has passed the
Slice 01 software, bounded numerical, packaging, and independent code-review gate in
`projectkoios.optimization`.

The completed KSD calculation historically called M3 combines a continuous
two-parameter rectangle with a finite ordered family of represented-operator
alternatives. The Optimization owner will provide only reusable unit-free quadratic
geometry and ordered finite-family selection. New KSD calculations consume these owner
APIs instead of retaining or rewriting the numerical mechanisms. Physical and scientific
interpretation remains the responsibility of each consumer; KSD owns that interpretation
for this experiment family.

## Pages

- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Quadratic optimization](../index.md)
- [Finite objective selection](../../selection/index.md)

## Completed KSD use case

For each member of a finite rotation family, KSD constructs a positive-definite
completed-square loss

$$
q_j(x)=m_j+(x-c_j)^TQ_j(x-c_j),\qquad x\in [\ell_0,u_0]\times[\ell_1,u_1].
$$

The completed calculation uses three reusable numerical mechanisms:

1. minimize each `q_j` over the closed rectangle;
2. calculate an unclipped quadratic-sublevel support point in a supplied direction;
3. select or filter members of the finite ordered quadratic family.

KSD uses those mechanisms to reason about a finite family of alignment angles and to
construct a witness or separation certificate. The angle meaning, sublevel thresholds,
unclipped-domain premise, and certificate interpretation are not part of the reusable
optimization capability.

## Successor-experiment contract

For every fixed allowed alignment angle, the spectral and operator squared losses are
positive-definite quadratics. Rectangle-constrained feasibility is convex, and each
accepted region is a convex box-clipped ellipsoid. The complete operator region is a
nonconvex finite union, but its global comparison requires only exhaustive ordered
enumeration:

$$
d(A_S,\bigcup_i A_{O_i})=\min_i d(A_S,A_{O_i}).
$$

The current owner scope therefore needs no generic nonconvex optimizer. Optimization
owns the exact two-dimensional rectangle minimum, full-ellipsoid support/extrema, and
deterministic finite-family filtering and selection. KSD owns per-angle loss
construction, allowed-family definition, selection of model-induced and Euclidean
scientific metrics, admissible-set definitions, requested distance, phase-map assembly,
witnesses, certificates, resolutions, dispositions, and interpretation. KSD does not own
or reimplement a reusable generic box-clipped-set-distance algorithm. Such an algorithm
requires a separate Optimization architecture and API review when a concrete experiment
needs it.

KSD will first qualify dense and sparse backend parity against its retained completed
proof objects. KSD then freezes a publication-independent immutable binding to an exact
Optimization release or commit and to its own adapter contract. That binding and adapter
are KSD responsibilities; no KSD-specific adapter or publication policy belongs here.

## Existing Optimization capabilities

The migration must compose existing owner APIs rather than duplicate them:

- `PositiveDefiniteQuadraticLoss` already owns unit-free completed-square state and
  evaluation;
- `ParameterSpace` already owns ordered finite closed bounds; and
- `WeightedAffineResidualQuadraticConstructor` already owns real weighted-affine
  least-squares construction after direct weighted-design rank acceptance.

KSD's private `_quadratic(...)` construction is therefore internal design context for
those existing capabilities, not a new implementation target. Complex
represented-operator feature construction and normalization remain consumer adapter
responsibilities; KSD owns its successor adapter.

## Planned module boundary

```text
projectkoios.optimization/
├── quadratic/
│   ├── constrained.py
│   │   ├── ConstrainedQuadraticTolerance
│   │   ├── RectangleQuadraticCandidate
│   │   ├── RectangleQuadraticMinimum
│   │   ├── RectangleQuadraticMinimizer2D
│   │   ├── QuadraticSublevelSupportResult
│   │   └── QuadraticSublevelSupportSolver
│   ├── factory.py
│   │   ├── RectangleQuadraticMinimizerMethod
│   │   ├── RectangleQuadraticMinimizerFactory
│   │   ├── QuadraticSublevelSupportMethod
│   │   └── QuadraticSublevelSupportSolverFactory
│   ├── numpy/
│   │   └── constrained.py
│   │       ├── NumpyRectangleQuadraticMinimizer2D
│   │       └── NumpyQuadraticSublevelSupportSolver
│   └── scipy/
│       └── constrained.py
│           ├── ScipySparseRectangleQuadraticMinimizer2D
│           └── ScipySparseQuadraticSublevelSupportSolver
└── selection/
    └── finite.py
        ├── FiniteMinimum
        ├── first_minimum
        └── indices_at_or_below
```

Package initializers remain import-empty. Contract and result modules remain
dependency-free. Dense numerical linear algebra stays in the capability-local NumPy module. Sparse
numerical linear algebra stays in the capability-local SciPy module. Finite objective
selection remains a separate Optimization capability rather than a quadratic-specific
selector.

## Contracts

### `ConstrainedQuadraticTolerance`

An immutable capability-specific numerical acceptance record with finite nonnegative
`absolute` and `relative` fields, at least one of which is positive. The default is
`absolute=1.0e-12` and `relative=1.0e-10`.

For finite scalars `left` and `right`, numerical consistency means

$$
|left-right| \leq absolute + relative\max(|left|,|right|).
$$

This tolerance validates backend results and numerical tests only. It never changes
strict candidate comparison, inserts an epsilon into finite selection, or defines a
scientific threshold.

### `RectangleQuadraticMinimum`

An immutable result containing:

- the two finite minimizing coordinates;
- the finite squared loss at those coordinates; and
- the closed `RectangleQuadraticCandidate` enum member identifying the winning ordered
  candidate.

It does not identify a scientific parameter, claim uniqueness outside the accepted
positive-definite contract, or interpret the value against a threshold.

### `NumpyRectangleQuadraticMinimizer2D`

The first implementation accepts exactly:

- one dense `PositiveDefiniteQuadraticLoss` of dimension two; and
- one `ParameterSpace` containing exactly two finite closed bounds, including fixed
  coordinates with equal endpoints.

It returns the global minimum over that closed rectangle. The implementation deliberately
remains two-dimensional because the completed algorithm is exact only for a rectangle in two
coordinates. It must not present the clipped-center and four-edge enumeration as a
general-dimensional box-QP algorithm.

The deterministic candidate order is:

1. the unconstrained center clipped into the rectangle;
2. coordinate 0 fixed at its lower bound;
3. coordinate 0 fixed at its upper bound;
4. coordinate 1 fixed at its lower bound; and
5. coordinate 1 fixed at its upper bound.

For each fixed coordinate, the other coordinate is minimized analytically and clipped to
its interval. The lowest squared loss wins; exact ties retain the earliest candidate in
that order. Returning the point as well as its loss makes the behavior independently
checkable without adding scientific interpretation.

`ScipySparseRectangleQuadraticMinimizer2D` applies the same exact candidate order to a
sparse quadratic representation. It does not change dimensional scope or tie behavior.
Dense and sparse coordinates and losses must agree under the declared constrained-
quadratic tolerance on bounded parity fixtures. Candidate enums must agree when the
winner is separated from the runner-up beyond the parity margin or when an exact tie is
constructed from exactly represented duplicate candidates. Each backend independently
uses strict candidate comparison; no cross-backend candidate-enum promise is made for
roundoff-level near ties.

A later n-dimensional box optimizer requires a separately designed and independently
verified active-set or finite-enumeration algorithm. It is not an implied extension of
this migration.

### `QuadraticSublevelSupportResult`

An immutable result containing:

- the finite support point;
- the exact finite nonzero direction supplied by the caller after conversion to built-in
  binary64 values; and
- the finite support value, equal to the direction dot the point; and
- the finite squared loss evaluated at the returned point.

The direction is not normalized: scaling it by a positive factor leaves the support
point unchanged and scales the retained support value by that factor. The numerical
operation attaches no axis or scientific coordinate name.

### Quadratic sublevel-support solvers

For a squared-loss ceiling `s`, direction `d`, and

$$
b=s-m,
$$

the sublevel set is empty when `b < 0`. Otherwise the calculator solves

$$
Qy=d
$$

and returns the positive-direction support point

$$
x_\star=c+\sqrt{\frac{b}{d^Ty}}\,y.
$$

The dense implementation uses `numpy.linalg.solve`; the sparse implementation uses a
SciPy sparse linear solve. Neither forms `Q^{-1}`. Both accept any positive dimension
supported by their quadratic backend, a finite nonzero direction of matching dimension,
and a finite nonnegative squared-loss ceiling. They fail closed for an empty sublevel
set, representation mismatch, dimension mismatch, nonfinite arithmetic, or linear-solve
failure.

The final nominal solver methods validate exact result types and relational
postconditions. Rectangle results must lie in the supplied bounds and their retained loss
must agree with fresh quadratic evaluation. Support results must preserve the normalized
direction exactly; their support value must agree with the direction-point dot product;
and their retained loss must agree with both fresh quadratic evaluation and the requested
ceiling under `ConstrainedQuadraticTolerance`.

This operation computes support for the complete ellipsoid. It does **not** check that
the ellipsoid or support point lies inside a consumer's box. The consumer retains the
unclipped-ellipse premise and must check domain containment before interpreting the
result as certificate evidence; KSD owns that check for M3.

### Finite-family composition

The consumer evaluates the ordered quadratic family and passes the resulting numeric
values to `projectkoios.optimization.selection.finite`:

- `first_minimum(values)` returns the first minimizing index and value; and
- `indices_at_or_below(values, ceiling)` returns every feasible index in declaration
  order.

Selection carries no consumer label or scientific meaning. Support-result ranking remains
ordinary consumer composition because M3 requires no distinct reusable optimization
contract for it.

## Ownership boundary

| Behavior | Owner after migration |
|---|---|
| Unit-free completed-square quadratic | Optimization |
| Exact two-dimensional rectangle minimum | Optimization dense and sparse capability |
| Full-ellipsoid support point/value | Optimization dense and sparse capability |
| Ordered finite-family numeric selection and first-on-tie behavior | Optimization |
| Complex operator feature construction and affine candidate matrices | Consumer adapter; KSD for M3 |
| Hamiltonian, basis, unit, and energy-scale meaning | Consumer; KSD for M3 |
| Finite rotation inventory and angle identities | Consumer; KSD for M3 |
| Spectral and represented-operator RMS definitions | Consumer; KSD for M3 |
| Training and withheld evaluation roles | Consumer; KSD for M3 |
| Threshold selection and scientific meaning | Consumer; KSD for M3 |
| Scientific metric choice, admissible sets, requested distance, and resolution | Consumer; KSD for M3 |
| Reusable generic box-clipped-sublevel-set distance algorithm | Deferred Optimization capability requiring separate review |
| Phase-map controls and sensitivity inventories | Consumer; KSD for M3 |
| Box containment and unclipped-certificate premise | Consumer; KSD for M3 |
| Common-witness policy and separation-bound assembly | Consumer; KSD for M3 |
| Certificate, disposition, and acceptance interpretation | Consumer; KSD for M3 |
| Locality diagnostics, provenance, and serialization | Consumer; KSD for M3 |
| Execution, authorization, persistence, and retries | Consumers |

Optimization accepts ordered unit-free tuples and numeric ceilings only. It does not
learn the consumer coordinate names `energy_shift_ratio` and `splitting_scale`, nor the
finite-family label `alignment_angle`.

## ProjectKoios source context

The completed KSD M3 implementation and tests provide internal ProjectKoios design and
verification context. This migration does not require cross-project provenance or a
transfer-manifest entry. The reusable owner capability remains independently specified
and tested here.

## Migration integrity

The completed KSD source, independent verifier, proof objects, and behavioral tests
remain untouched internal ProjectKoios evidence. New KSD calculations call the owner API
rather than retaining or rewriting these numerical mechanisms.

Optimization will create owner-native records, capability-local optional backends,
strict typing, deterministic behavior, and independent tests. The implementation will
not import `ksdft2effmass` or its schemas at runtime. Owner-native regression fixtures will be compact numeric reproductions of the completed
M3 behavior.

Consumer adapters remain outside the owned numerical namespaces and retain all
conversion, labeling, threshold, containment, certificate, and interpretation policy.

## Verification plan

Implementation is not accepted until source-mirrored tests establish:

1. interior, one-edge, and corner rectangle minima;
2. correlated off-diagonal quadratics where independent coordinate clipping is wrong;
3. exact first-on-tie behavior for duplicate minimum candidates, including singleton,
   fixed-coordinate, and center-on-boundary rectangles;
4. regression agreement for the retained M3 quadratics;
5. independent agreement with a bounded reference minimization over adversarial SPD
   rectangles;
6. full-ellipsoid support equality `q(x_star) == s` within an explicit numerical
   tolerance;
7. support optimality against sampled boundary directions;
8. empty-sublevel, zero-direction, dimension, nonfinite, backend-failure, and
   nonconforming-subclass-result cases;
9. family-order preservation and threshold-boundary inclusion; and
10. scale-aware tolerance checks and bounded dense/sparse parity without asserting enum
    identity for roundoff-level near ties; and
11. import-empty initializers plus isolated wheel import without NumPy until the backend
    module or dense quadratic is constructed.

These checks are software and bounded numerical verification. They establish neither
KSD's witness/certificate claims nor scientific validation.

## Explicit exclusions and future roadmap

The current constrained-quadratic slice does not authorize:

- a generic local or global nonconvex optimizer;
- continuous constant-angle alignment;
- smooth momentum-dependent gauge search;
- nonlinear hopping or Slater–Koster fitting;
- a reusable general distance solver for box-clipped quadratic sublevel sets;
- model-induced or Euclidean scientific distance policy;
- phase-map, bounds, angle-family, mesh, or weight sensitivity policy;
- a KSD adapter or publication-specific binding; or
- Wannier localization, which remains Wannier90-owned.

Future continuous alignment, nonlinear fitting, and reusable box-clipped-set distance
require separate Optimization architecture tied to a concrete experiment; KSD must not
reimplement those generic algorithms. The immediate experiment uses only certificates
constructible from the approved minima, support, and enumeration kernels or waits for the
reviewed extension. Local optimization may establish witnesses, but it cannot establish
incompatibility. Positive-separation claims require certified global lower bounds through
methods such as analytic elimination, complete enumeration, interval branch-and-bound,
or convex relaxation. Prefer Procrustes or polar closed forms, symmetry reduction, and
finite enumeration whenever they solve the declared problem.

M2 square Procrustes and blind-alignment rectangular Procrustes remain separately planned
Alignment capabilities. They are not authorized by this M3 slice.
