# Algorithms

## Bounded uniform

`BoundedUniformOptimizer` retains the donor SplitMix64 equations, sample-major
counter order, upper-53-bit conversion, and affine bound mapping. Seed and
proposal budget are explicit. Empty spaces produce empty tuples; fixed bounds
produce their endpoint. Multiple asks may be outstanding and tells may arrive
in any order. Equal loss is resolved by proposal sequence.

## Bounded coordinate pattern

`BoundedCoordinatePatternOptimizer` evaluates the initial point, then feasible
negative and positive neighbors in coordinate order. It accepts only strict
loss improvement. A failed pattern reduces every active step. Termination is
one of `step_tolerance`, `evaluation_limit`, or `iteration_limit`. Repeated ask
while one proposal is pending returns that exact proposal. The algorithm is a
local optimizer and makes no global-convergence claim.

Both engines checkpoint by recording construction inputs and accepted feedback;
restore deterministically replays state and verifies any pending proposal.
