# Alignment numerics

Alignment capabilities select transformations that compare unit-free mathematical
representations. Numerical controls belong to Optimization only when they govern
software validation, deterministic decomposition, or explicit optimizer rank behavior.

- Scalar Procrustes evaluates one right-unitary alignment problem.
- Batched Procrustes preserves leading tensor dimensions and performs bounded vectorized
  reductions.
- Rank cutoffs, residual roundoff cutoffs, and validation bounds are distinct evidence.
- Equality at the additive rank cutoff is deficient.
- A numerically selected deficient completion is diagnostic rather than unique.

Alignment numerics do not define physical frames, lattice paths, scientific thresholds,
or acceptance policy.
