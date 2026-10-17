# Finite box minimization of a positive-definite quadratic

## Status

This page defines the reviewed owner boundary for deterministic minimization of an
existing unit-free `PositiveDefiniteQuadraticLoss` over a finite, nonempty,
positive-dimensional axis-aligned box. The capability belongs in Optimization because
it selects a feasible parameter vector. It does not own the meaning, units, scientific
bounds, threshold, disposition, or experiment policy supplied by a consumer.

## Public operation

For

$$
q(x)=m+(x-c)^TQ(x-c),\qquad Q=Q^T\succ 0,
$$

and ordered finite bounds

$$
B=\{x\in\mathbb{R}^n:\ell_i\leq x_i\leq u_i\},\qquad n\geq 1,
$$

`BoxQuadraticMinimizer.solve(loss, bounds)` returns the unique global minimizer,
its evaluated squared loss, the active-bound classification, the full gradient,
per-coordinate KKT acceptance bounds, maximum KKT violation and normalized ratio,
bounded active-set diagnostics, and backend metadata.

The first implementation is `NumpyBoxQuadraticMinimizer`. It accepts the existing dense
quadratic representation and uses a deterministic primal active-set method with a caller-
visible positive iteration limit. Sparse variants are not implied.

## Contract decisions

- `PositiveDefiniteQuadraticLoss` and `ParameterSpace` are reused without a duplicate
  quadratic or bounds record.
- Dimensions must agree and must be positive.
- Equal endpoints are admitted and reported as `fixed` coordinates.
- Exact built-in nominal records remain required. Value factories perform explicit
  finite-real normalization before frozen construction; raw frozen constructors reject
  implicit coercion and never replace fields after construction.
- The initial point is the unconstrained center clipped into the box. Nonfixed clipped
  coordinates begin active at their exact endpoint; all other nonfixed coordinates begin
  free.
- When free-coordinate stationarity fails, the backend solves the unique Newton system
  on the current free subspace and moves by the largest feasible step up to one.
- Every coordinate reaching a bound on that step becomes active. Exact simultaneous
  hits are processed in ascending coordinate order.
- When free coordinates are stationary but active-bound signs fail, the smallest-index
  violating active coordinate is released. This is the exact tie rule.
- The solve performs at most `maximum_iterations` KKT-check/active-set iterations and
  fails closed rather than return an unchecked point when that limit is reached.
- The default limit is 10,000. It is numerical execution control, not a scientific
  threshold and not a promise that every admitted binary64 problem will converge.
- The nominal capability is dense-only. Its final template rejects sparse loss state
  before backend dispatch so every nominal subclass has the same representation boundary.
- The public active-bound classification is recomputed from the returned point and bounds,
  rather than trusted from backend face labels.
- KKT evidence is recomputed coordinate by coordinate by the final nominal template from
  the original quadratic, bounds, and returned point. One coordinate's scale cannot
  relax another coordinate's optimality condition.
- Numerical tolerance is software acceptance policy, not a scientific threshold and not
  a proof of exact arithmetic.

## Pages

- [Scientific and ownership boundary](science.md)
- [Mathematics and deterministic algorithm](numerics.md)
- [Implementation and failure contract](implementation.md)
- [Schematics](schematics.md)
- [Inspected references](references.md)

## Explicit exclusions

This capability is not a universal quadratic-programming facade, a nonconvex optimizer,
an intersection-feasibility certificate, a strict-separation certificate, or a
box-clipped ellipsoid distance solver. It does not import PhysKit or KSD, attach units,
select bounds, interpret a minimum against a threshold, or authorize an experiment.
