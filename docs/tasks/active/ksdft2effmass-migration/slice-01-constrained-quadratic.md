# Slice 01 — constrained-quadratic implementation specification

## Status

This document fixes the class-, method-, backend-, factory-, and test-level contract for
the first implementation slice. Source implementation proceeded only after explicit
operator authorization and has passed its software, bounded numerical, packaging, and
independent code-review gate.

The slice owns the reusable mechanisms needed by the periodic-1D constrained-admissible-
set experiment historically called M3: exact two-dimensional rectangle minimization,
full-ellipsoid support, and ordered finite-family selection. It adds dense and sparse
owner implementations with one nominal SDK contract. New KSD calculations consume this
API instead of retaining or rewriting the numerical mechanisms.

## Scope guardrail

For a fixed allowed alignment angle, each accepted region is a convex box-clipped
ellipsoid. The operator region is a finite union, handled by complete ordered enumeration.
No generic nonconvex optimizer is required or authorized.

This slice does not include continuous alignment, smooth gauge search, nonlinear model
fitting, a reusable general distance solver for box-clipped sublevel sets, scientific
metric policy, phase-map construction, sensitivity policy, KSD adapters, publication
bindings, or Wannier localization. KSD owns scientific metric and requested-distance
definitions; any reusable distance algorithm requires separate Optimization architecture
and must not be reimplemented in KSD. The immediate experiment uses only certificates
constructible from the approved kernels or waits for that reviewed extension.

## Prerequisite nominalization

`quadratic/quadratic.py` currently uses a private structural `_QuadraticLossBackend`
protocol. Before adding solvers:

- replace that protocol with a nominal abstract backend class;
- make the NumPy and SciPy quadratic implementations inherit it explicitly;
- preserve the existing `PositiveDefiniteQuadraticLoss` compatibility facade;
- preserve constructor, properties, evaluation behavior, lazy imports, and exceptions;
- add nominal-subclass and facade-identity tests; and
- make no scientific or numerical behavior change.

This prerequisite removes structural backend escape without forcing a breaking public
quadratic-loss construction change.

## Public source layout

```text
src/python/projectkoios/optimization/
├── quadratic/
│   ├── constrained.py
│   ├── factory.py
│   ├── numpy/
│   │   └── constrained.py
│   └── scipy/
│       └── constrained.py
└── selection/
    └── finite.py
```

All package initializers remain import-empty.

## Public result records

### `ConstrainedQuadraticTolerance`

```python
@dataclass(frozen=True, slots=True)
class ConstrainedQuadraticTolerance:
    absolute: float = 1.0e-12
    relative: float = 1.0e-10
```

Both fields are finite nonnegative built-in floats after normalization, and at least one
is positive. Scalar consistency uses:

```text
abs(left - right) <= absolute + relative * max(abs(left), abs(right))
```

This is a backend-result acceptance tolerance, not a scientific threshold. It never
changes strict candidate comparisons or finite-selection behavior.

### `RectangleQuadraticCandidate`

A closed string enum identifying the winning ordered candidate:

```python
class RectangleQuadraticCandidate(StrEnum):
    CLIPPED_CENTER = "clipped-center"
    COORDINATE_0_LOWER = "coordinate-0-lower"
    COORDINATE_0_UPPER = "coordinate-0-upper"
    COORDINATE_1_LOWER = "coordinate-1-lower"
    COORDINATE_1_UPPER = "coordinate-1-upper"
```

The enum exposes deterministic algorithm evidence without attaching parameter meaning.

### `RectangleQuadraticMinimum`

```python
@dataclass(frozen=True, slots=True)
class RectangleQuadraticMinimum:
    coordinates: tuple[float, float]
    squared_loss: float
    candidate: RectangleQuadraticCandidate
```

Validation:

- `coordinates` is an exact two-item tuple normalized to finite built-in floats;
- `squared_loss` is finite and nonnegative; and
- `candidate` is an exact enum member.

### `QuadraticSublevelSupportResult`

```python
@dataclass(frozen=True, slots=True)
class QuadraticSublevelSupportResult:
    point: tuple[float, ...]
    direction: tuple[float, ...]
    support_value: float
    squared_loss: float
```

Validation:

- `point` and `direction` are nonempty finite tuples with equal dimensions;
- `direction` is not the zero vector;
- `support_value` and `squared_loss` are finite;
- `squared_loss` is nonnegative; and
- retained values are built-in binary64 values.

The result retains the evaluated boundary loss so callers can verify the requested
ceiling without re-evaluating through a backend.

## Nominal solver contracts

### `RectangleQuadraticMinimizer2D`

```python
class RectangleQuadraticMinimizer2D(ABC):
    def __init__(
        self,
        *,
        tolerance: ConstrainedQuadraticTolerance = ConstrainedQuadraticTolerance(),
    ) -> None: ...

    @final
    def solve(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum: ...

    @abstractmethod
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        bounds: ParameterSpace,
        /,
    ) -> RectangleQuadraticMinimum: ...
```

The public template method validates:

- exact `PositiveDefiniteQuadraticLoss` facade input;
- exact `ParameterSpace` input;
- quadratic dimension two;
- bounds dimension two; and
- matching dimensions.

After `_solve_validated`, the final public method requires the exact
`RectangleQuadraticMinimum` type, coordinates admitted by the supplied bounds, exact
candidate-enum type, and retained squared loss consistent with
`loss.squared_loss(coordinates)` under the configured tolerance.

Zero-width bounds are accepted. `ParameterBound` already defines equal endpoints as a
fixed coordinate, and the exact candidate algorithm remains valid for a line segment or
point rectangle.

The abstract class is the SDK integration point. The public template method is marked
`final`; external implementations explicitly inherit the class and implement only
`_solve_validated`. Built-in concrete implementations are marked `final`.

### `QuadraticSublevelSupportSolver`

```python
class QuadraticSublevelSupportSolver(ABC):
    def __init__(
        self,
        *,
        tolerance: ConstrainedQuadraticTolerance = ConstrainedQuadraticTolerance(),
    ) -> None: ...

    @final
    def solve(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult: ...

    @abstractmethod
    def _solve_validated(
        self,
        loss: PositiveDefiniteQuadraticLoss,
        squared_loss_ceiling: float,
        direction: tuple[float, ...],
        /,
    ) -> QuadraticSublevelSupportResult: ...
```

The public template method validates and normalizes:

- exact quadratic facade input;
- finite nonnegative ceiling;
- finite nonempty direction tuple;
- dimension agreement;
- nonzero direction; and
- nonempty sublevel condition `ceiling >= minimum_squared_loss`.

After `_solve_validated`, the final public method requires the exact
`QuadraticSublevelSupportResult` type, matching point dimension, direction exactly equal
to the normalized caller direction, support value consistent with the direction-point
dot product, and retained squared loss consistent with both fresh quadratic evaluation
and the requested ceiling under the configured tolerance.

The solver operates on the complete ellipsoid. It accepts no bounds and makes no box
containment claim.

## Factory contracts

```python
class RectangleQuadraticMinimizerMethod(StrEnum):
    NUMPY_DENSE = "numpy-dense"
    SCIPY_SPARSE = "scipy-sparse"


class QuadraticSublevelSupportMethod(StrEnum):
    NUMPY_DENSE = "numpy-dense"
    SCIPY_SPARSE = "scipy-sparse"
```

```python
class RectangleQuadraticMinimizerFactory:
    @staticmethod
    def create(
        method: RectangleQuadraticMinimizerMethod,
        /,
        *,
        tolerance: ConstrainedQuadraticTolerance = ConstrainedQuadraticTolerance(),
    ) -> RectangleQuadraticMinimizer2D: ...


class QuadraticSublevelSupportSolverFactory:
    @staticmethod
    def create(
        method: QuadraticSublevelSupportMethod,
        /,
        *,
        tolerance: ConstrainedQuadraticTolerance = ConstrainedQuadraticTolerance(),
    ) -> QuadraticSublevelSupportSolver: ...
```

Factory behavior:

- requires the exact enum and tolerance types;
- lazily imports only the selected backend;
- returns a nominal solver subclass;
- contains no numerical algorithm; and
- provides no `AUTO` method or string registry.

## Finite-family selection contract

```python
def first_minimum(values: tuple[float, ...], /) -> FiniteMinimum: ...


def indices_at_or_below(
    values: tuple[float, ...],
    ceiling: float,
    /,
) -> tuple[int, ...]: ...


def absolute_ambiguity_set(
    values: tuple[float, ...],
    tolerance: float,
    /,
) -> FiniteAmbiguitySet: ...
```

The exact nonempty value tuple contains finite real values excluding Boolean values.
Ceilings and tolerances are finite real values excluding Boolean values; tolerances are
nonnegative. Exact minima retain the first index, ceiling equality qualifies, no
qualifying ceiling value returns `()`, and every returned index preserves declaration
order. The ambiguity set includes all exact minima at zero tolerance.

## Dense rectangle algorithm

`NumpyRectangleQuadraticMinimizer2D` requires `loss.sparse is False`.

It reads the two-dimensional center and dense matrix, then constructs candidates in this
exact order:

1. clip both center coordinates to their bounds;
2. fix coordinate 0 at its lower bound, analytically minimize coordinate 1, then clip;
3. fix coordinate 0 at its upper bound, analytically minimize coordinate 1, then clip;
4. fix coordinate 1 at its lower bound, analytically minimize coordinate 0, then clip;
5. fix coordinate 1 at its upper bound, analytically minimize coordinate 0, then clip.

For fixed coordinate `axis`, free coordinate `other`, and fixed value `bound`:

```text
point[other] = center[other]
             - Q[other, axis] / Q[other, other]
             * (bound - center[axis])
```

Every candidate is evaluated through `loss.squared_loss`. Selection uses strict `<` in
candidate order, so an exact tie retains the first candidate. The solver returns the
winning coordinates, evaluated squared loss, and candidate enum.

## Sparse rectangle algorithm

`ScipySparseRectangleQuadraticMinimizer2D` requires `loss.sparse is True`.

It applies the identical candidate formula and order. It reads only the required CSR
scalar entries and evaluates candidates through the sparse loss. It must not call
`toarray`, `.A`, `.todense`, or otherwise materialize a dense matrix.

Sparse and dense coordinates and squared losses must agree under the configured
tolerance on bounded parity fixtures. Candidate enums must agree for score-separated
winners and exactly represented duplicate-candidate ties. Each backend uses its own
strict `<` comparisons; candidate-enum parity is not promised for roundoff-level near
ties.

## Dense support algorithm

`NumpyQuadraticSublevelSupportSolver` requires `loss.sparse is False`.

For normalized ceiling `s`, minimum `m`, direction `d`, and matrix `Q`:

```text
budget = s - m
y = numpy.linalg.solve(Q, d)
denominator = dot(d, y)
scale = sqrt(budget / denominator)
point = center + scale * y
support_value = dot(d, point)
squared_loss = loss.squared_loss(point)
```

The solver fails if the solve fails, any intermediate is nonfinite, or the denominator is
not strictly positive. It never forms an inverse.

When `budget == 0`, the returned point is exactly the normalized center and does not
depend on direction.

## Sparse support algorithm

`ScipySparseQuadraticSublevelSupportSolver` requires `loss.sparse is True`.

It uses canonical CSR state and `scipy.sparse.linalg.spsolve(Q, d)`. It treats singular or
rank warnings as failure, checks finite output and a strictly positive denominator, and
never densifies or forms an inverse. Result construction is otherwise identical to the
dense solver.

## Numerical acceptance and parity domain

All backend-result comparisons use `ConstrainedQuadraticTolerance`; the optimization
algorithm itself never uses that tolerance to rank candidates.

Deterministic dense/sparse parity fixtures use identical binary64 center, matrix, bounds,
ceiling, and direction values. They have matrix 2-norm condition number at most
`1.0e6`, finite magnitudes at most `1.0e6`, and no underflowing nonzero direction. A
score-separated rectangle fixture has a best-to-runner-up loss gap greater than four
times the tolerance bound for those two scores. Exact-tie fixtures use exactly
represented matrices, bounds, and duplicate candidate coordinates. Near ties inside that
margin test each backend's deterministic local order but do not require cross-backend
enum identity.

Expected degenerate candidate identities are explicit:

- a singleton rectangle returns `CLIPPED_CENTER` because every later candidate duplicates
  the first point;
- a fixed-coordinate line returns `CLIPPED_CENTER` when that point is optimal, otherwise
  the earlier lower edge for the fixed coordinate wins its duplicate upper-edge tie;
- a center exactly on a rectangle boundary returns `CLIPPED_CENTER`; and
- exact dense/sparse tie fixtures use the same earliest enum.

Reference checks use the same scalar tolerance formula. Coordinate vectors are compared
componentwise. Bounded-reference loss agrees with the returned loss.

For positive support budget, let

```text
y = solve(Q, direction)
denominator = dot(direction, y)
alpha = sqrt(budget / denominator)
lagrange_multiplier = 1 / (2 * alpha)
stationarity = -direction + 2 * lagrange_multiplier * Q @ (point - center)
```

The KKT stationarity check uses the infinity norm and requires

```text
norm_inf(stationarity)
<= absolute
 + relative * max(
       norm_inf(direction),
       norm_inf(2 * lagrange_multiplier * Q @ (point - center)),
   )
```

Boundary loss independently satisfies the scalar tolerance formula against the requested
ceiling. When `budget == 0`, the feasible ellipsoid is the singleton center; the test
requires the exact normalized center and boundary-loss agreement and does not apply the
positive-budget stationarity multiplier.

For every deterministic sampled point admitted by the requested quadratic sublevel set,
let `sample_value = dot(direction, sample)` and `returned_value = support_value`. Support
optimality requires

```text
sample_value
<= returned_value
 + absolute
 + relative * max(abs(sample_value), abs(returned_value))
```

Positive direction scaling produces componentwise-consistent points and a correspondingly
scaled support value under the same scalar/componentwise formula.

## Failure contract

| Condition | Failure |
|---|---|
| Wrong public object kind | `TypeError` naming the argument |
| Wrong factory enum or tolerance kind | `TypeError` naming the argument |
| Quadratic or bound dimension is not two | `ValueError` |
| Dense solver receives sparse loss | `ValueError` |
| Sparse solver receives dense loss | `ValueError` |
| Direction dimension mismatch | `ValueError` |
| Zero direction | `ValueError` |
| Negative or nonfinite ceiling | `ValueError` |
| Ceiling below quadratic minimum | `ValueError` with `sublevel set is empty` |
| Nonpositive support denominator | `ValueError` |
| Nonfinite backend result | `ValueError` |
| Dense or sparse solve failure | `ValueError` chained from backend failure |
| Backend returns wrong result type or inconsistent result | `TypeError` or `ValueError` from final public validation |

The slice introduces no scientific exception and no acceptance or certificate status.

## Source-mirrored tests

```text
tests/projectkoios/optimization/
├── quadratic/
│   ├── test__ConstrainedQuadraticResults__unit.py
│   ├── test__ConstrainedQuadraticFactories__unit.py
│   ├── test__ConstrainedQuadraticWheel__integration.py
│   ├── test__RectangleQuadraticMinimizer2D__contract.py
│   ├── test__QuadraticSublevelSupportSolver__contract.py
│   ├── numpy/
│   │   ├── test__NumpyRectangleQuadraticMinimizer2D__integration.py
│   │   ├── test__NumpyRectangleQuadraticMinimizer2D__numerical.py
│   │   └── test__NumpyQuadraticSublevelSupportSolver__numerical.py
│   └── scipy/
│       ├── test__ScipySparseRectangleQuadraticMinimizer2D__integration.py
│       └── test__ScipySparseQuadraticSublevelSupportSolver__numerical.py
└── selection/
    ├── test__FiniteMinimum__unit.py
    ├── test__FiniteAmbiguitySet__unit.py
    └── test__FiniteSelection__unit.py
```

## Acceptance matrix

| Behavior | Verification |
|---|---|
| Nominal SDK boundary | abstract-class, explicit-subclass, final built-in, and malicious-result rejection tests |
| Factory construction | exact concrete type for each closed enum member |
| Lazy optional backends | isolated subprocess imports |
| Interior minimum | analytic fixture |
| Each edge orientation | four analytic fixtures |
| Corner minimum | analytic fixture |
| Fixed coordinate and point rectangle | explicit candidate-enum awkward-case fixtures |
| Correlated quadratic | independent bounded reference |
| Exact candidate tie | candidate enum and first-order assertion |
| Full-ellipsoid support | boundary equality and KKT direction check |
| Zero-budget support | exact center assertion |
| Direction scaling | componentwise-consistent point for positive scale and scaled support value |
| Dense/sparse agreement | bounded, conditioned, score-separated and exact-tie fixtures under declared tolerance |
| Sparse no-densification | fail-on-densification instrumentation |
| Tolerance contract | defaults, scale formula, Boolean/nonfinite/negative/all-zero rejection |
| Invalid inputs | exact exception kind and stable message fragment |
| Finite minimum | first-on-exact-tie index and value |
| Ceiling filter | inclusive equality, empty result, and declaration order |
| Absolute ambiguity | zero/nonzero tolerance and all qualifying indices |
| Selection validation | Boolean, empty, nonreal, nonfinite, and negative-tolerance failures |
| Completed M3 behavior | owner-native numeric regression fixtures |
| Independent reference | bounded minimizer and sampled support comparisons |

## Internal ProjectKoios source context

KSD's completed M3 implementation uses explicit inversion for support; the owner
capability uses linear solves. Dense and sparse owner implementations receive independent
verification. No cross-project provenance or transfer-manifest record is required.

Before scientific use, KSD qualifies both owner backends against its retained completed
proof objects and freezes an immutable binding to an exact Optimization release or commit
plus the KSD adapter contract. KSD's standalone verifier independently reconstructs
certificates and must not call the same high-level Optimization routines being checked.
These are consumer qualification requirements, not additional owner API behavior.

## Stop gate

The slice stops after:

- prerequisite nominalization passes existing quadratic tests unchanged;
- new dense and sparse solvers pass their conformance and numerical suites;
- wheel imports prove optional-backend isolation; and
- independent review reports no blocking technical findings.

No least-squares, Procrustes, finite-periodic, assignment, likelihood, generic nonconvex,
box-clipped-set-distance, KSD-adapter, or publication-policy implementation is part of
this slice.
