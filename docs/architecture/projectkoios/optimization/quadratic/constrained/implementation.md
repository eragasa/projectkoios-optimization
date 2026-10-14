# Constrained-quadratic implementation organization

## Target modules

```text
projectkoios/optimization/quadratic/
├── constrained.py
├── factory.py
├── numpy/
│   └── constrained.py
└── scipy/
    └── constrained.py
```

Finite-family value selection is implemented separately in
`projectkoios.optimization.selection.finite`. No generic nonconvex optimizer,
box-clipped-set-distance solver, KSD adapter, or scientific metric is part of this
module. A future reusable distance solver requires separate Optimization architecture;
consumers must not reimplement it as an application-local generic algorithm.

## Dependency-free results

`ConstrainedQuadraticTolerance` retains finite nonnegative absolute and relative backend-
acceptance tolerances, requires at least one positive value, and defaults to `1.0e-12`
and `1.0e-10`, respectively. Its scale-aware scalar comparison is
`abs(left - right) <= absolute + relative * max(abs(left), abs(right))`. It never changes
candidate ordering or defines scientific acceptance.

`RectangleQuadraticMinimum` retains:

- exactly two minimizing coordinates;
- finite squared loss; and
- the closed `RectangleQuadraticCandidate` enum member for the winning candidate.

`QuadraticSublevelSupportResult` retains:

- support point;
- exact caller direction after binary64 normalization; and
- directional support value; and
- squared loss evaluated at the returned point.

## Nominal solvers and factories

`RectangleQuadraticMinimizer2D` and `QuadraticSublevelSupportSolver` are capability-local
nominal abstract classes and SDK integration points. Their final public `solve` methods
own both input validation and relational result validation. Their factories accept closed
method enums with `NUMPY_DENSE` and `SCIPY_SPARSE` members, accept a
`ConstrainedQuadraticTolerance`, import the selected backend lazily, and return the
abstract solver type. Built-in concrete solver classes are marked `final`.

Post-backend validation requires exact result types. Rectangle coordinates must lie in
the supplied bounds and retained loss must agree with fresh evaluation. Support direction
must equal the normalized caller direction exactly; support value must agree with its dot
product; and retained loss must agree with fresh evaluation and the requested ceiling
under the declared tolerance.

## NumPy implementation

`NumpyRectangleQuadraticMinimizer2D` accepts a dense
`PositiveDefiniteQuadraticLoss` and exactly two finite closed bounds. Equal endpoints
are accepted as fixed coordinates. It evaluates the clipped center and four analytically
minimized edge candidates in the specified deterministic order.

`NumpyQuadraticSublevelSupportSolver` accepts a dense quadratic, finite squared-loss
ceiling, and finite nonzero direction. It uses `numpy.linalg.solve`; it never forms an
explicit inverse.

## SciPy sparse implementation

`ScipySparseRectangleQuadraticMinimizer2D` preserves the dense implementation's exact
candidate and tie order over sparse quadratic state.

`ScipySparseQuadraticSublevelSupportSolver` uses SciPy sparse linear solving without
densifying the quadratic matrix or forming an inverse.

## Failures

- Non-two-dimensional rectangle minimization fails.
- Dimension mismatch fails.
- Empty sublevels fail.
- Zero directions fail.
- Nonfinite inputs or backend arithmetic fail.
- Linear-solve failures fail closed.

## Verification

- Interior, edge, and corner minima.
- Correlated off-diagonal cases where independent coordinate clipping is wrong.
- Exact first-on-tie candidate order.
- Independent bounded-reference agreement.
- Sublevel-boundary equality and directional optimality.
- Empty-sublevel and invalid-input failures.
- Dense/sparse coordinate and loss agreement on bounded, well-conditioned fixtures.
- Candidate-enum parity for score-separated winners and exactly represented duplicate
  ties; no parity promise for roundoff-level near ties.
- Nonconforming nominal-subclass result rejection.
- Sparse no-densification regression.
- Owner-native M3 regression fixtures.
- Consumer-side dense/sparse parity qualification against retained KSD proof objects
  before scientific use.

## Status

Architecture and source implementation are complete. The Slice 01 software, bounded
numerical, packaging, and independent code-review gate passes.
