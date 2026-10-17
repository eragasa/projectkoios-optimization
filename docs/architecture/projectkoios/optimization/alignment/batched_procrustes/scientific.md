# Batched Procrustes consumer needs

Consumers use batched Procrustes when a shared tensor batch must be aligned without
per-item Python materialization. They retain ownership of every domain interpretation.

- A consumer supplies exact unit-free complex matrices and explicit numerical controls.
- The consumer preserves the meaning and topology of leading indices.
- The consumer may persist ranks, residuals, and roundoff cutoffs as evidence.
- Any deficient item must be handled under consumer policy; diagnostic completions are
  not unique transformations.
- KSD can fail closed for closure or plaquette interpretation without moving that policy
  into Optimization.
- PhysKit can adapt frame-bearing objects without moving physical types or units into
  Optimization.

This library owns only transformation selection and numerical evidence. It does not own
scientific thresholds, lattice semantics, closure claims, dispositions, or human
acceptance.
