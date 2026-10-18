# Rank-cutoff rule implementation

The closed enum is defined beside the dependency-free tolerance and shared solver policy
in `projectkoios.optimization.alignment.procrustes`.

- Tolerance construction requires the exact enum type.
- Scalar cutoff construction branches in the shared generic policy owner.
- NumPy batched cutoff construction broadcasts the same selected rule.
- Scalar and batched result validation recompute the declared rule independently.
