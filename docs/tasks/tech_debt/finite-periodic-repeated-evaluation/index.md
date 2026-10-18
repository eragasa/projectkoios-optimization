# Repeated finite-periodic tensor evaluation

This task records possible API and allocation improvements observed while KSD adopted the
finite-periodic owner contracts. The current API is correct and sufficient for the
qualified workload; implementation work is deferred until campaign profiling establishes
a material cost.

## Status

The KSD consumer directly uses the owner transform, tensor-product evaluator, Parseval
analyzer, and cyclic Hermiticity analyzer without a blocking owner defect.

- Owner checkpoint reviewed: `2c6dd7f717722f6b0495feb8cfe1287a104268a6`.
- Focused KSD parity evidence: 51 passing locality tests.
- Retained dense smoke: `N=32` to `65**3`, matrix size two.
- Reported smoke runtime: approximately `0.11 s`.
- Reported maximum temporary elements: `997100` under a `1000000` element bound.
- Recommendation: do not expand the owner API before profiling a complete multi-shell
  campaign.

## Observed repeated work

KSD evaluates several related matrix series on the same construction mesh and coordinate
axes for every retained physical-metric shell.

- Each shell constructs selected mode and coefficient arrays.
- Each shell constructs a new tensor-product evaluation definition.
- Defensive construction copies the coordinate axes unless the existing explicit
  immutable-array boundary is used.
- Each evaluation rebuilds its dense coefficient grid and phase matrices.
- KSD currently evaluates four correlated shell streams: reference and comparison fields
  on construction and evaluation grids.
- Complete-shell definitions, physical metrics, normalization, errors, thresholds, and
  scientific dispositions remain KSD-owned policy.

## Deferred API candidates

The first justified extension should be a reusable tensor-grid evaluation plan, but only
if profiling shows that repeated setup materially affects a campaign.

- Validate and canonicalize the source mesh and coordinate axes once.
- Cache reusable axis phase matrices under an explicit memory bound.
- Accept multiple compatible finite matrix series while checking dimension, matrix size,
  and construction-box mode bounds.
- Expose cached, temporary, output, and algorithm-visible live element or byte counts.
- Specify whether any reusable workspace is immutable, per-call, or stateful and define
  thread-safety accordingly.
- Keep shell selection and other physical support policy out of the plan.

A correlated series-selection operation may accompany that work if repeated manual
selection remains common.

- A candidate `series.select_terms(mask)` operation would require one Boolean value per
  term, preserve declared order and duplicates, allow an empty selection, and return
  canonical immutable storage.
- The caller would continue to define the mask and its scientific meaning.
- A centered-mesh mode-bound property could replace repeated consumer derivation of the
  lower twisted-Nyquist boundary.

A streaming evaluation API is a separate option if full-result canonicalization becomes
a measured memory or runtime cost.

- Prefer bounded immutable batches with explicit tensor-grid slice metadata over a
  mutable-array escape hatch or arbitrary reduction callback.
- Preserve downstream ownership of matrix, band, normalization, and threshold
  reductions.
- Clarify that `maximum_working_elements` excludes retained output and does not claim to
  bound opaque NumPy or BLAS internal workspace.

## Revisit gate

No implementation should begin from the single-evaluation smoke result alone.

- Profile a complete production-representative multi-shell campaign.
- Record wall time, repeated setup time, result-copy time, and peak resident memory.
- Compare the existing `from_arrays(...)` path with reuse through available
  `from_immutable_arrays(...)` boundaries.
- Proceed only when measured evidence identifies repeated phase construction, axis
  copying, coefficient-grid construction, or result copying as material.

## Excluded extensions

The observed consumer does not justify broader Fourier or scientific-policy APIs.

- Do not add physical shell or radius abstractions to Numerics.
- Do not add automatic generic-versus-tensor backend selection without a demonstrated
  need and explicit memory semantics.
- Do not add sparse FFT, sparse block storage, NUFFT, or approximate discovery.
- Do not combine numerical diagnostics with KSD scientific acceptance or disposition.
