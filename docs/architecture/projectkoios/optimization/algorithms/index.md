# Scalar optimization algorithms

## Coordinate search

`CoordinateSearch` is a deterministic scalar direct-search method. It evaluates the
initial decision vector, then feasible negative and positive coordinate directions in
order. Strict improvement is accepted; otherwise every active step is reduced.
Termination is `step_tolerance`, `evaluation_limit`, or `iteration_limit`.

`CoordinateSearchSettings` and `CoordinateSearchState` follow the repository's
canonical frozen-construction pattern. Their dataclass constructors accept canonical
built-in storage and validate without field replacement. No alternate normalization
constructor is exposed.

Validation names each field explicitly; neither record uses reflective `getattr` nor
`object.__setattr__`. `CoordinateSearchState` contains intrinsic replay state only.
Encoding, checksums, persistence, and workflow lifecycle checkpoints remain external.

This scalar algorithm is independent of `multiobjective.pareto.ParetoOptimizer`.
