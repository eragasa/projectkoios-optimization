# Independent architecture review

## Scope

The review covered the planned ProjectKoios Numerics and Optimization ownership model,
with a bounded implementation-readiness review of the first constrained-quadratic slice.
It inspected:

- namespace ownership and one-way dependency direction;
- nominal abstract solver and closed factory SDK conventions;
- dense and sparse constrained-quadratic contracts;
- exact two-dimensional rectangle candidate completeness and tie behavior;
- full-ellipsoid support mathematics and failure semantics;
- shared relational result validation;
- finite-family selection signatures, results, failures, and tests;
- scale-aware numerical acceptance and dense/sparse parity scope;
- KSD scientific ownership and independent verification responsibilities;
- the deferred box-clipped-sublevel-set distance capability; and
- direct real-space reconstruction versus Wannier90-owned localization.

The review was read-only and did not authorize implementation or modify KSD.

## Resolved findings

### Ownership inventory

Disposition: `NO_ACTION_REQUIRED`

The inventory now keeps affine fitting, quadratic objectives, optimized reciprocal-frame
transport, and phase synchronization in Optimization. Numerics owns direct frame,
sewing, closure, holonomy, transformation, and diagnostic records only.

### Nominal result invariants

Disposition: `NO_ACTION_REQUIRED`

Final public solver methods now own input and relational result validation. Built-in
implementations are final. Rectangle and support results must satisfy exact type,
dimension, direction, bounds, objective, and support-value invariants under the declared
capability tolerance.

### Finite-family selection

Disposition: `NO_ACTION_REQUIRED`

The slice now includes exact selection signatures and return types, empty qualifying-set
behavior, Boolean rejection, exact first-on-tie behavior, inclusive ceilings,
declaration-order preservation, source layout, tests, and acceptance cases.

### Numerical acceptance and parity

Disposition: `NO_ACTION_REQUIRED`

The architecture now defines the absolute/relative tolerance formula, bounded parity
fixtures, conditioning and magnitude limits, exact degenerate candidate expectations,
score-separation requirements, and the absence of a cross-backend candidate-enum promise
for roundoff-level near ties.

Positive-budget support verification defines the step scale, Lagrange multiplier,
stationarity vector, infinity-norm acceptance formula, boundary check, and sampled-support
bound. Zero-budget support is checked separately as a singleton-center case.

### Wannier ownership wording

Disposition: `NO_ACTION_REQUIRED`

Numerics is limited to direct reconstruction and real-space moments for supplied states.
Wannier gauge optimization and localization remain Wannier90-owned.

## Final technical result

ReviewOutcome: `NO_BLOCKING_FINDINGS`

OperatorRequest: `NONE`

The bounded architecture review found no remaining technical blocker. Implementation,
runtime numerical verification, scientific validation, commit, merge, release, and
publication remain separate future actions and are not authorized by this review.
