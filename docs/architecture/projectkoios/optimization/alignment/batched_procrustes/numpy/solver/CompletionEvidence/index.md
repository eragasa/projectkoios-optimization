# CompletionEvidence

`CompletionEvidence` is an encapsulated frozen solver record for one vectorized
completion stage.

## Ownership

- `completions` holds SVD-selected right-unitary matrices.
- `ranks` and `full_rank` hold additive-cutoff uniqueness evidence.
- `rank_cutoffs` records the exact applied per-item cutoffs.
- The record performs no backend work and does not claim deficient completions are
  unique.
- It is consumed only while assembling the validated public result and is not exported
  by the façade.

## Verification

Dedicated unit tests verify exact field identity, frozen field binding, slots, and
absence from the supported façade. Solver integration tests verify the numerical
relations represented by all four fields.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
