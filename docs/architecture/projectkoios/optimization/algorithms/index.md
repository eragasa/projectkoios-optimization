# Algorithms

## Bounded uniform

`BoundedUniformOptimizer` retains the donor SplitMix64 equations, sample-major
counter order, upper-53-bit conversion, and affine bound mapping. Its unit
fraction primitive is owned by `SplitMix64UnitUniformSampler` and shared with
slot-composed probability sampling.
Seed and
proposal budget are explicit; the proposal budget is a positive u32. Empty
spaces produce empty tuples; fixed bounds produce their endpoint. Multiple asks
may be outstanding and tells may arrive in any order. Equal loss is resolved by
proposal sequence.

## Bounded coordinate pattern

`BoundedCoordinatePatternOptimizer` evaluates the initial point, then feasible
negative and positive neighbors in coordinate order. Its evaluation budget is
a positive u32 and its iteration budget is an unsigned u32. It accepts only
strict loss improvement. A failed pattern reduces every active step.
Termination is one of `step_tolerance`, `evaluation_limit`, or
`iteration_limit`. Repeated ask while one proposal is pending returns that exact
proposal. The algorithm is a local optimizer and makes no global-convergence
claim.

Both engines validate at construction that the run identifier plus the longest
possible `.candidate.<sequence>` suffix fits the 256-character candidate-ID
limit; identifiers are never truncated. Both engines checkpoint by recording
construction inputs and accepted feedback; restore deterministically replays
state and verifies any pending proposal.
