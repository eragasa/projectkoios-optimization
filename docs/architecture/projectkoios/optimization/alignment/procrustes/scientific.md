# Scalar Procrustes consumer needs

A consumer uses scalar Procrustes when one source-target matrix pair requires a best
right-unitary transformation. The consumer owns why the matrices should be compared and
what any residual or rank status means.

- PhysKit may adapt physically meaningful frames into unit-free matrices and restore
  semantic context after the solve.
- KSD may use the result as numerical evidence under separately defined scientific
  thresholds and failure policy.
- A deficient SVD completion must not support a claim that the transformation is unique.
- Software and bounded numerical verification do not establish scientific validity.

The scalar contract does not import consumer types, units, topology, persistence,
scientific dispositions, or acceptance policy.
