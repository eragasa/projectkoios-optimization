# Migration recommendations

This disposition converts the completed-work inventory into an ordered migration plan.
This repository owns reusable mathematical numerics and optimization; each consumer owns
its physical and scientific interpretation.

## Disposition

- Review outcome: **NO_BLOCKING_FINDINGS**.
- Independent architecture review:
  [NO_BLOCKING_FINDINGS](independent-architecture-review.md).
- Operator request: **NONE**.
- Inventory reviewed with KSD: yes.
- KSD review result: **NO_BLOCKING_FINDINGS**.
- Package and API draft: [implementation strategy](implementation-strategy.md).
- Ownership decision: transformations, decompositions, mathematical geometry, and
  numerical analysis belong to `projectkoios.numerics`.
- Ownership decision: every capability that optimizes a function or set of functions
  belongs to `projectkoios.optimization`.
- Ownership decision: physical and scientific interpretation belongs to each consumer.
- Implementation authorization recorded here: Slice 01 plus standing authorization for
  owner support needed to keep consumer repositories outside reusable mathematical
  kernels; current support scope is Slice 02 linear least squares.
- Commit, merge, push, and dependency authorization recorded here: none.

## Internal source review gate

KSD and this repository are parts of ProjectKoios. Cross-project provenance records and
transfer-manifest entries are not required for this migration.

- Review each completed ProjectKoios implementation and its tests before extracting the
  reusable contract.
- Reproduce only the reusable behavior under the owner namespace.
- Verify owner behavior independently rather than importing KSD at runtime.
- Begin implementation only after capability architecture and verification cases are
  complete.

## M3 successor boundary

- New KSD calculations consume the owner APIs instead of retaining or rewriting their
  numerical mechanisms.
- Each fixed-angle region is convex; the nonconvex operator region is a finite union
  handled by complete ordered enumeration.
- The current owner slice is limited to exact two-dimensional rectangle quadratic
  minima, full-ellipsoid support/extrema, and deterministic ordered finite-family
  filtering and selection.
- KSD retains Hamiltonians, loss construction, allowed alignments, thresholds, selection
  of model-induced and Euclidean scientific metrics, admissible-set definitions,
  requested distance, resolution, phase maps, independent frozen controls,
  bounds/angle-family/mesh/weight sensitivity, dispositions, locality, provenance,
  version binding, and scientific interpretation.
- KSD qualifies dense/sparse parity against retained proof objects and independently
  reconstructs certificates without calling the same high-level routines under test.
- Generic nonconvex optimization, continuous alignment, smooth gauge search, nonlinear
  hopping fitting, and reusable box-clipped-set distance require separate Optimization
  architecture; KSD must not reimplement those generic algorithms.
- The immediate experiment uses only certificates constructible from the approved
  minima, support, and enumeration kernels or waits for a separately reviewed extension.
- Local optimization establishes witnesses only. Incompatibility and positive separation
  require certified global lower bounds.
- Wannier localization remains Wannier90-owned.

## `projectkoios.optimization` ownership

- Weighted real and complex least squares.
- Multiple-right-hand-side fitting.
- Real linear and structured matrix-subspace fitting.
- Positive-definite quadratic objectives and closed-box minimization.
- Finite-family and tolerance-aware candidate selection.
- Square and rectangular complex Procrustes.
- Pointwise and aggregate frame alignment objectives.
- Optimized reciprocal-frame transport and phase synchronization.
- Linear-sum assignment.
- Minimax assignment.
- Deterministic representative clustering.
- Likelihood-based parameter fitting.
- Optimization samplers over deterministic numerical sequences.

## `projectkoios.numerics` ownership

- Symmetric eigensystems and spectral-subspace construction.
- Projectors, complements, compression, and basis pullback.
- Symmetric Gram orthonormalization and common-space injection.
- Represented-matrix norms.
- Observed-order numerical analysis.
- Finite-periodic Fourier transforms and diagnostics.
- Reciprocal sewing maps.
- Frame, projector, projected-operator, and holonomy records.
- Wilson-loop products and circular phase-cost construction.
- Principal-phase unwrapping and winding.
- Direct finite-supercell real-space reconstruction and moment diagnostics for supplied
  states.
- Radical-inverse and Halton sequence generation.
- Finite group actions, periodic-center geometry, and density reconstruction.
- Finite-difference Hessians and tensor inversion.
- Bounded root finding.
- Statistical covariance, conditioning, interval, and probability diagnostics.

## Consumer responsibility

- Physical units and material interpretation.
- Model and state-space meaning.
- Scientific coordinate and parameter meaning.
- Study-specific input construction.
- Scientific thresholds and acceptance.
- Evidence identity and claim interpretation.
- Paper and manuscript conclusions.

## Extraction rule

- Extract the complete reusable mathematical contract.
- Keep arrays, records, state, algorithms, diagnostics, and numerical failure semantics
  in the owning `projectkoios.numerics` or `projectkoios.optimization` capability.
- Keep physical labels, units, scientific thresholds, and interpretation in consumer
  adapters.
- Preserve distinctions between projection, basis transformation, truncation, alignment,
  sewing, transport, and optimization.
- Preserve distinctions between linear-sum and minimax assignment.
- Preserve distinctions between frame defect and projector defect.

## Implementation order

### Phase 1 — conference-paper kernels

1. Implement M3 constrained-quadratic geometry and exact finite-family selection.
2. Verify the M3 slice and stop at its acceptance gate.
3. Write and review ordinary and weighted linear least-squares architecture.
4. Implement and verify the M1 complex multiple-right-hand-side specialization.
5. Write and review Procrustes architecture.
6. Implement and verify the M2 square Procrustes and blind-alignment rectangular
   Procrustes core.

### Phase 2 — fitting and deterministic selection

1. Add the real linear and structured matrix-subspace fit facade.
2. Add tolerance-aware finite-candidate ambiguity selection.
3. Add explicit rank, condition, fitted-value, and residual diagnostics.

### Phase 3 — sampling, assignment, and clustering

1. Add radical-inverse and Halton generation.
2. Add linear-sum assignment.
3. Add minimax point-set assignment.
4. Add deterministic representative clustering.

### Phase 4 — reusable linear algebra and finite-periodic numerics

1. Add symmetric eigensystems, spectral subspaces, projectors, and compression.
2. Add represented-matrix norms and observed-order diagnostics.
3. Add symmetric Gram orthonormalization and common-space pullback.
4. Add finite-periodic Fourier transforms, truncation, and Parseval diagnostics.
5. Add reciprocal sewing maps and mesh-neighbor translations.
6. Add frame projectors and projected-operator paths.
7. Add direct frame, projector, projected-operator, and holonomy records.
8. Add Wilson products, circular phase-cost construction, unwrapping, and winding.
9. Add direct finite-supercell real-space reconstruction and moment diagnostics for
   supplied states; do not implement Wannier gauge optimization or localization.
10. Add finite group actions, periodic-center geometry, and density reconstruction.
11. Add finite-difference Hessian and tensor numerics.
12. Add bounded root finding.
13. Add reusable statistical likelihood and inference numerics.

## Finite-selection contract

- Accept a nonempty ordered numeric inventory.
- Return the first index attaining the exact minimum.
- Return every index at or below a caller-supplied ceiling.
- Return every index within a caller-supplied absolute ambiguity tolerance of the
  minimum.
- Preserve declaration order.
- Return indices and numeric values only.
- Keep scientific labels, thresholds, and dispositions with the consumer.

## Procrustes contract

- Implement one reusable complex Procrustes numerical kernel.
- Support square unitary solutions.
- Support rectangular partial-isometry solutions.
- Support pointwise, batched, and aggregate operations.
- Return the transform, singular values, numerical rank, condition number, and residual.
- Define deterministic behavior for full-rank and rank-deficient inputs.
- Build optimized reciprocal transport and phase synchronization as separate Alignment
  solvers over the shared Procrustes kernel and Numerics-owned finite-periodic records.

## Assignment contracts

- Implement linear-sum assignment against a caller-supplied cost matrix.
- Implement minimax assignment as a distinct objective and result contract.
- Keep circular-residual cost construction in Numerics and assignment-based phase
  matching in Optimization.
- Keep the physical meaning of Wilson spectra and centers with the consumer.

## Capability acceptance gate

- Completed internal ProjectKoios behavior and tests have been reviewed.
- Owner-native regression fixtures pass.
- An independent numerical oracle agrees within an explicit tolerance.
- Rank-deficient, degenerate, boundary, tie, empty, and nonfinite cases pass.
- Output ordering is deterministic.
- Optional numerical backends remain capability-local and lazy.
- Package initializers remain import-empty.
- Full tests pass.
- Ruff passes.
- Formatting passes.
- Strict mypy passes.
- Isolated wheel import passes.
- Reproducible builds match.
- Independent technical review reports no blocking findings.

## Immediate next action

- Keep KSD domain-shaped and replace its direct generic least-squares kernel through the
  existing fitter seam after KSD receives a durable owner commit, release, or artifact
  locator.
- Treat the completed Slice 02 gate and local artifact digests as technical evidence, not
  as a published dependency locator.
- Do not infer authorization for commit, merge, push, publication, dependency release,
  or capabilities that do not enforce the consumer-not-math boundary.
