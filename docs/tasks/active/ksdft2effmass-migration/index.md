# KSD completed-work numerical inventory

This task records reusable numerical behavior already exercised by the completed ICMSEP
2026 Conference Paper 1 and completed research-manuscript studies. The inventory records
all completed behavior before assigning the final repository owner.

## Status

- Branch: `migration/ksdft2effmass`.
- Base commit: `8424caad898a0002ef08d41d235cd0a25d4852c9`.
- KSD access: read-only.
- KSD files modified: none.
- Migration disposition: [recommendations](recommendations.md).
- Package and API draft: [implementation strategy](implementation-strategy.md).
- First-slice class and test plan:
  [constrained quadratic](slice-01-constrained-quadratic.md).
- Standing-authorized support slice:
  [linear least squares](slice-02-linear-least-squares.md).
- Independent architecture review:
  [no blocking findings](independent-architecture-review.md).
- Slice 01 independent implementation review:
  [no blocking findings](independent-implementation-review.md).
- Slice 02 independent least-squares review:
  [no blocking findings](independent-least-squares-review.md).
- Commit, merge, push, and dependency authorization: not granted.

## ICMSEP 2026 Conference Paper 1

### Direct admissible-set demonstration — Section 3.6 and Appendix B

- Producer: `calculations/ICMSEP2026/conference/paper_1/run.py`.
- Verifier: adjacent `verify_result.py`.
- Evidence: retained result, exact-rational verification, figure data, and checksums.
- Reusable behavior: weighted real affine residual construction.
- Reusable behavior: positive-definite completed-square quadratic evaluation.
- Reusable behavior: diagonal parameter metric.
- Repository status: affine fitting and quadratic objectives remain under
  `projectkoios.optimization`; only diagonal metric geometry has a canonical move to
  `projectkoios.numerics` pending.
- Owner APIs: `WeightedAffineResidualQuadraticConstructor`,
  `PositiveDefiniteQuadraticLoss`, and diagonal metric geometry.

### M1 isolated-band reduction — Section 3.2 and Appendix S2

- Producer:
  `python/src/ksdft2effmass/analysis/hopping_fits.py::BlockHoppingLeastSquaresFitter1D`.
- Context: `python/src/ksdft2effmass/periodic1d/isolated_band/`.
- Verifier:
  `calculations/ICMSEP2026/conference/paper_1/isolated-band/verify_result.py`.
- Evidence: retained M1 result, library verification, standalone reconstruction, fit
  diagnostics, and withheld diagnostics.
- Reusable behavior: positive-weight complex linear least squares with multiple
  right-hand sides.
- Required result: coefficients, numerical rank, full-rank condition number, fitted
  values, L2-of-Frobenius residual, maximum-Frobenius residual, and identification
  disposition.
- Optimization status: owner solver implemented; acceptance review and consumer version
  binding are pending.
- Numerics status: reusable Fourier-design construction missing.
- Consumer responsibility: physical units, hopping-record meaning, route meaning, and
  identification interpretation.

### M2 multiband alignment — Section 3.3 and Appendix S3

- Pointwise producer:
  `python/src/ksdft2effmass/solid_state/frame_alignment.py::BandFrameAligner1D`.
- Aggregate producer:
  `python/src/ksdft2effmass/periodic1d/multiband_alignment/calculate.py::_global_rotation`.
- Verifier:
  `python/src/ksdft2effmass/periodic1d/multiband_alignment/verify.py`.
- Evidence: retained M2 result and independent reconstruction of pointwise and
  one-global-unitary channels.
- Reusable behavior: square complex unitary Procrustes polar-SVD solve.
- Required forms: pointwise, batched pointwise, and aggregate one-unitary solve.
- Retained outputs: rotations, aligned frames, frame defects, and projector defects.
- Optimization status: square complex Procrustes/polar core missing.
- Numerics status: frame construction, reciprocal path, and closure distribution missing.
- Rectangular partial-isometry evidence: supplied by the completed blind-alignment study,
  not M2.
- Consumer responsibility: admissible-family choice and physical operator
  interpretation.

### M3 constrained admissible sets — Section 3.4 and Appendix S5

- Producer:
  `python/src/ksdft2effmass/periodic1d/admissible_sets/calculate.py`.
- Verifier:
  `python/src/ksdft2effmass/periodic1d/admissible_sets/verify.py`.
- Evidence: witness case, separated case, library verification, standalone verification,
  and checksums.
- Reusable behavior: exact two-dimensional SPD-quadratic minimum over a closed
  rectangle.
- Reusable behavior: full-ellipsoid sublevel support in a supplied nonzero direction.
- Reusable behavior: ordered finite-quadratic-family pointwise argmin and numeric
  feasibility filtering.
- Tie behavior: first member on exact ties.
- Optimization status: owner implementation and acceptance gate complete.
- Owner architecture:
  [constrained quadratics](../../../architecture/projectkoios/optimization/quadratic/constrained/index.md).
- Successor plan: KSD performs new calculations against the owner API rather than
  retaining or rewriting these numerical mechanisms.
- Consumer responsibility: Hamiltonian and loss construction, allowed angle inventory,
  scientific thresholds, selection of scientific metrics, admissible-set definitions,
  requested distance and resolution, phase maps, containment claims, witnesses,
  certificates, locality, provenance, dispositions, publication-independent version
  binding, and interpretation.
- Deferred owner capability: a reusable generic distance algorithm for box-clipped
  quadratic sublevel sets requires separate Optimization architecture and API review;
  KSD must not reimplement it.
- Scope guardrail: the finite nonconvex union is handled by exhaustive ordered
  enumeration; no generic nonconvex optimizer is required or authorized.

### M3 threshold sensitivity — Section 3.5

- Producer:
  `calculations/ICMSEP2026/conference/paper_1/constrained-admissible-sets-threshold-sensitivity/analyze.py`.
- Verifier: adjacent `verify_result.py`.
- Evidence: retained grid, transition thresholds, and reconstructed verification.
- Reusable behavior: quadratic evaluation, directional sublevel support, ordered
  extrema, and deterministic filtering.
- Optimization status: covered by existing quadratic evaluation and the missing M3
  extraction.
- Additional numerical primitive: none.

## Research manuscript Appendix D — particle in a box

### Symmetric eigensystem and retained subspaces

- Producers: `python/src/ksdft2effmass/campaigns/piab1d/residual_study.py`,
  `python/src/ksdft2effmass/operators/eigenpairs.py`, and
  `python/src/ksdft2effmass/operators/subspaces.py`.
- Evidence:
  `calculations/research-monograph/particle-in-box/result.json` and corresponding
  verifier/checksums.
- Completed behavior: deterministic symmetric eigensolve.
- Completed behavior: lowest spectral-subspace selection.
- Completed behavior: projector and orthogonal complement.
- Completed behavior: coordinate compression and embedded compression.
- Completed behavior: discarded-sector identities.
- Numerics status: eigensystem, spectral-subspace, projector, and compression capability
  missing.

### Adjacent-level observed order

- Producers: `python/src/ksdft2effmass/campaigns/piab1d/convergence.py` and
  `python/src/ksdft2effmass/analysis/convergence.py`.
- Evidence:
  `calculations/research-monograph/particle-in-box/convergence-result.json` and
  corresponding verifier/checksums.
- Completed behavior: validated adjacent-level order
  `log(e_i/e_j) / log(h_i/h_j)`.
- Numerics status: observed-order capability missing.

### Eigenpair sweep diagnostics

- Producer: `python/src/ksdft2effmass/campaigns/piab1d/eigenpair_sweep.py`.
- Evidence:
  `calculations/research-monograph/particle-in-box/eigenpair-sweep-result.json` and
  corresponding verifier/checksums.
- Completed behavior: complete spectrum.
- Completed behavior: deterministic eigenvector sign convention.
- Completed behavior: overlap diagnostics.
- Completed behavior: scaled algebraic residuals.
- Numerics status: canonical eigensystem and residual-diagnostic capability missing.

### Matrix-norm sweep

- Producers: `python/src/ksdft2effmass/campaigns/piab1d/norm_sweep.py` and
  `python/src/ksdft2effmass/operators/matrix_norms.py`.
- Evidence:
  `calculations/research-monograph/particle-in-box/norm-sweep-result.json` and
  corresponding verifier/checksums.
- Completed behavior: Frobenius norm.
- Completed behavior: spectral/operator-2 norm.
- Completed behavior: maximum-entry norm.
- Completed behavior: same-reference normalization.
- Numerics status: represented-matrix norm and reference-normalization capability
  missing.

### Ordered model-class fitting

- Producer:
  `python/src/ksdft2effmass/campaigns/piab1d/identifiability.py::RetainedModelClassFitter`.
- Evidence:
  `calculations/research-monograph/particle-in-box/identifiability-result.json` and
  corresponding verifier/checksums.
- Completed class: `scalar_identity`, with projection `tr(A) / n I`.
- Completed class: `diagonal_in_retained_basis`.
- Completed class: `real_symmetric_tridiagonal`.
- Completed class: `arbitrary_real_symmetric`.
- Retained result: best fit, unexplained residual, and Frobenius norm.
- Repository status: full-rank vectorized minimizer present through the real affine
  core.
- Optimization status: explicit real linear and matrix-subspace fit facade missing.
- Proposed facade extension: numerical rank and condition diagnostics.
- Optimization status: tolerance-aware ordered class selection missing.
- Consumer responsibility: class identities, scientific adequacy, and interpretation of
  declared masks.

## Research manuscript Appendix E — harmonic oscillator

### Symmetric Gram orthonormalization and common-space pullback

- Producer:
  `python/src/ksdft2effmass/analysis/model_systems/harmonic_oscillator/comparison.py`.
- Evidence: completed harmonic-oscillator result and independent verifier under
  `calculations/research-monograph/harmonic-oscillator/`.
- Completed behavior: construct the sampled Gram matrix.
- Completed gate: reject a non-SPD Gram matrix.
- Completed behavior: symmetric inverse square root.
- Completed behavior: isometric injection into common coordinates.
- Completed behavior: operator pullback into common coordinates.
- Completed diagnostics: Gram conditioning and isometry defect.
- Numerics status: symmetric orthonormalization and common-space pullback missing.

## Research manuscript Appendix G — periodic 1-D

### Complex hopping least squares

- Producer:
  `python/src/ksdft2effmass/analysis/hopping_fits.py::BlockHoppingLeastSquaresFitter1D`.
- Workflow source:
  `python/src/ksdft2effmass/campaigns/periodic_1d/workflows.py`.
- Evidence: completed nominal, stress, composite, and native-comparison results with
  verifiers and checksums.
- Completed route: uniform positive weights.
- Completed route: nonuniform positive weights.
- Completed route: incomplete design.
- Required result: coefficients, rank, identified disposition, full-rank condition
  number, fitted values, L2-of-Frobenius residual, and maximum-Frobenius residual.
- Optimization status: weighted complex multiple-right-hand-side least squares missing.

### Reciprocal sewing maps

- 1-D producers:
  `python/src/ksdft2effmass/solid_state/reciprocal_paths.py::PlaneWaveReciprocalSewingConstructor`,
  `PlaneWaveReciprocalSewingResult`, and `ReciprocalSewingDirection1D`.
- 2-D producers:
  `python/src/ksdft2effmass/analysis/model_systems/periodic2d/reciprocal_mesh.py::ReciprocalMeshNeighbor2DConstructor`
  and `PlaneWaveReciprocalSewing2DConstructor`.
- Evidence: completed Appendix-G nominal, stress, and composite results plus periodic-2D
  boundary calculations and independent verifiers.
- Completed input: declared ordered finite plane-wave reciprocal-index basis.
- Completed behavior: explicit coefficient map for `k -> k + G_alpha`.
- Completed 1-D map: one-index shift in retained `-P, ..., P` order.
- Completed 2-D map: lookup-based shift in either primitive reciprocal direction.
- Cutoff behavior: discard coefficients shifted outside the finite basis.
- Map type: nonunitary partial shift, not a periodic permutation.
- Mesh companion result: wrapped mesh index plus distinct integer reciprocal translation.
- Boundary use: sewn overlap `F_last† S F_first`.
- Downstream use: closure polar transport, Wilson products, frame-path compatibility,
  and gauge-covariance checks.
- Distinction: sewing identifies representations across reciprocal-cell boundaries; it
  is not gauge alignment, Procrustes optimization, phase unwrapping, or truncated-basis
  periodic wrapping.
- Numerics status: reciprocal sewing and mesh-translation capabilities missing.
- Consumer responsibility: physical reciprocal-basis meaning.

### Pointwise frame alignment

- Producer:
  `python/src/ksdft2effmass/solid_state/frame_alignment.py::BandFrameAligner1D`.
- Evidence: completed nominal, stress, composite, and native-comparison results with
  independent verification.
- Completed behavior: square complex unitary Procrustes/polar solve.
- Completed behavior: pointwise and batched candidate-to-reference alignment.
- Completed result: exact rotation at every point and aligned frame path.
- Completed diagnostic: frame Frobenius defect.
- Completed diagnostic: projector Frobenius defect.
- Completed scalar case: rank-one phase alignment.
- Optimization status: generic Procrustes/polar core missing.

### Frame projectors and projected-operator paths

- Projector producer:
  `python/src/ksdft2effmass/solid_state/frame_alignment.py::BandProjectorPathConstructor1D`.
- Projected-operator producer:
  `python/src/ksdft2effmass/solid_state/hopping_transforms.py::BandProjectedOperatorPathConstructor1D`.
- Evidence: completed Appendix-G composite controlled/rough-gauge studies, independent
  verification, and Conference Paper M2 alignment channels.
- Completed behavior: gauge-invariant projector `P(k) = F(k) F(k)†`.
- Completed behavior: basis pullback `A_red(k) = F(k)† A(k) F(k)`.
- Completed gates: coordinate count, coordinate tolerance, ambient dimension, units,
  mesh, rank, and sewing-map agreement.
- Completed distinction: frame defect and projector defect remain separate.
- Numerics status: projector-path and projected-operator-path capabilities missing.
- Consumer responsibility: coordinate, state-space, and unit interpretation.
- Semantic constraint: projection, basis transformation, and truncation remain distinct
  operations.

### Reciprocal-frame transport and closure-holonomy distribution

- Producer types:
  `python/src/ksdft2effmass/solid_state/band_frames.py::ReciprocalBandFramePath1D`,
  `PolarBandFrameTransporter1D`, and `PolarBandFrameTransportResult1D`.
- Evidence: completed isolated, composite, and stress results with independent stress
  and composite verification.
- Input: ordered orthonormal complex frames with fixed ambient dimension and rank on a
  declared half-open reciprocal mesh.
- Input: explicit endpoint sewing map.
- Completed behavior: neighbor-to-neighbor propagation by SVD polar alignment.
- Completed local rule: for `M = L Σ R†`, rotate the next frame by `R L†`.
- Completed diagnostic: retain every neighbor singular value.
- Completed gate: reject paths whose minimum singular value does not exceed the declared
  threshold.
- Completed closure: compare the final frame with the sewn first frame.
- Completed behavior: polar-project the closure overlap to a unitary holonomy.
- Completed behavior: obtain closure eigenphases and eigenvectors by complex Schur
  decomposition.
- Completed behavior: distribute a fractional unitary root continuously over the ordered
  mesh.
- Completed result: closed transported path, minimum singular value, threshold, and
  closure eigenphase multiset.
- Completed scalar specialization: rank-one phase parallel transport and closure-phase
  distribution.
- Completed composite specialization: rank-r non-Abelian polar transport.
- Shared numerical kernel: local complex Procrustes/polar solve.
- Optimization status: Procrustes core and objective-selected reciprocal-frame transport
  solver missing.
- Numerics status: direct frame, sewing, closure, and holonomy records and diagnostics
  missing.
- Consumer responsibility: physical meaning assigned to the path and holonomy.
- Distinction: this transport is separate from sewing-map construction, pointwise
  candidate alignment, Wilson link products, and spread minimization.

### Centered finite-mesh hopping transform

- Producer: `python/src/ksdft2effmass/solid_state/hopping_transforms.py`.
- Evidence: completed periodic-1D packages and independent route checks.
- Completed behavior: centered finite-mesh complex matrix Fourier transform.
- Completed behavior: exact inverse transform.
- Completed behavior: symmetric range truncation.
- Completed behavior: omitted-block L2 norm.
- Numerics status: finite-periodic transform and truncation capability missing.

### Hopping diagnostics

- Producer: `python/src/ksdft2effmass/analysis/hopping_diagnostics.py`.
- Evidence: completed periodic-1D packages and independent route checks.
- Completed behavior: Parseval residual.
- Completed behavior: modular block-Hermiticity pairing.
- Completed behavior: coefficient/sample Frobenius route comparison.
- Completed behavior: bandwidth evaluation.
- Completed behavior: analytic center-curvature evaluation.
- Numerics status: hopping and finite-periodic diagnostic capability missing.

### Finite Born–von Karman real-space reconstruction

- Producer: `python/src/ksdft2effmass/analysis/wannier_localization.py`.
- Evidence: completed periodic-1D packages and independent route checks.
- Completed behavior: finite Born–von Karman inverse Bloch transform.
- Completed behavior: quadrature normalization.
- Completed behavior: branch-declared center and spread.
- Numerics status: direct finite-periodic real-space reconstruction and moment
  diagnostics for supplied states are missing.
- Ownership boundary: Wannier gauge optimization and localization remain
  Wannier90-owned.

## Generic Wilson-loop machinery

- Historical producer:
  `calculations/research-monograph/periodic-1d/run_composite.py`.
- Public records and comparison:
  `python/src/ksdft2effmass/solid_state/wilson_loops.py`.
- Independent verifier:
  `python/src/ksdft2effmass/campaigns/periodic_1d/verification.py::Periodic1DWannier90WilsonVerifier`.
- Completed input: ordered closed path of square complex neighbor-overlap matrices and an
  explicit closure edge.
- Completed behavior: closest unitary polar factor `L R†` from `M = L Σ R†`.
- Completed diagnostic: minimum overlap singular value.
- Completed behavior: path-ordered unitary-link product.
- Completed behavior: unitary-loop eigenvalues and principal phases in `[-π, π)`.
- Completed storage: sorted canonical representation of an unordered phase multiset.
- Completed diagnostic: `||W†W - I||F`.
- Completed behavior: gauge-transformed overlap covariance check.
- Completed behavior: principal circular residual cost matrix.
- Completed behavior: minimum-total-absolute linear-sum assignment.
- Required comparison result: permutation, signed residuals, maximum defect, and L2
  defect.
- Completed behavior: explicit phase-to-center convention.
- Shared numerical kernel: complex polar/Procrustes.
- Shared numerical kernel: finite linear-sum assignment on a caller-supplied cost matrix.
- Numerics status: Wilson products, phase-multiset comparison, and circular residuals
  missing.
- Optimization status: phase-multiset linear-sum matching capability missing.
- Consumer responsibility: physical and topological interpretation.

## Completed 1-D retained models

- Sources:
  `python/src/ksdft2effmass/campaigns/periodic_1d/model/retained/isolated.py`,
  `stress.py`, and `composite.py`.
- Status: immutable exact-byte data models.
- Numerical behavior: supplied by the complex least-squares, Procrustes, finite-periodic
  transform, diagnostics, direct real-space reconstruction, and Wilson-loop producers
  above.
- Numerical extraction from retained models: none; extract from behaviorful producers.

## Completed 1-D defect studies

### Blind alignment

- Producer:
  `python/src/ksdft2effmass/campaigns/periodic_1d/defects/blind_alignment/inference.py`.
- Evidence:
  `calculations/research-monograph/impurity-defect-1d-blind-alignment/` result and
  independent verification.
- Completed behavior: rank-aware rectangular complex Procrustes inference.
- Completed result: full or partial outcome, singular rank, and conditioning.
- Completed behavior: structured identifiability stops.
- Completed behavior: scalar shift minimization.
- Optimization status: generic rectangular Procrustes missing; scalar affine core
  present.

### Spin-independent operator projection

- Producer:
  `calculations/research-monograph/impurity-spin-spaces/spin_spaces_calculation/run_experiment.py`.
- Evidence: completed human-accepted result and independent verification.
- Completed behavior: Frobenius-optimal projection onto operators of form `B ⊗ I_2` by
  partial spin trace.
- Completed diagnostics: absolute residual, relative residual, RMS residual, and analytic
  residual check.
- Optimization status: second completed consumer for an explicit structured
  matrix-subspace fit facade.
- Consumer responsibility: spin-space meaning and scientific acceptance.

### Finite-rank analytical oracle

- Producer:
  `python/src/ksdft2effmass/campaigns/periodic_1d/defects/finite_rank_oracle/workflow.py`
  and retained oracle runner.
- Completed behavior: bracket expansion.
- Completed behavior: bounded bisection.
- Required result: final bracket, residual, and iteration count.
- Numerics status: bracket expansion and bounded bisection capability missing.

### Extraction and reconciliation

- Completed behavior: matched extraction under declared transforms.
- Completed behavior: independent-route reconciliation.
- Completed behavior: continuum refinement with spectral and operator comparisons.
- Hidden parameter optimizer: none.
- Numerics status: reusable linear algebra and declared-transform reconciliation belong
  in `projectkoios.numerics`.
- Consumer responsibility: physical interpretation of transforms and comparisons.

## Completed 2-D manuscript studies

### Appendix-H finite-periodic and topological numerics

- Producers: `calculations/research-monograph/periodic-2d/run_experiment.py`,
  `run_composite.py`, and `run_topological.py`.
- Evidence: retained accepted results, independent verifiers, and the periodic-2D
  evidence index.
- Completed behavior: exact two-dimensional finite-mesh complex matrix Fourier transform
  and inverse.
- Completed behavior: radial shell truncation and omitted-block norm.
- Completed diagnostics: Parseval residual and training/withheld route comparison.
- Completed scalar topology: normalized links, Bargmann plaquette phases, Chern sum, and
  integer defect.
- Completed non-Abelian topology: polar links, determinant plaquettes, Wilson-loop phase
  spectra, and gauge covariance.
- Completed behavior: principal-phase unwrapping and Wilson winding.
- Completed behavior: finite-supercell two-dimensional real-space centers and spreads
  for supplied states.
- Completed numerical analysis: centered finite-difference Hessian, tensor inversion,
  and principal values.
- Shared numerical kernels: complex polar/Procrustes and caller-cost linear-sum
  assignment.
- Numerics status: two-dimensional finite-periodic, topological, direct real-space
  moment, and tensor capabilities missing. Wannier gauge optimization and localization
  remain Wannier90-owned.
- Consumer responsibility: physical and scientific interpretation.

### Direct hopping/model fits

- Producers:
  `calculations/research-monograph/periodic-2d/run_experiment.py` and
  `run_composite.py`.
- Evidence: completed scalar and composite results with independent verification.
- Reusable behavior: weighted complex multiple-right-hand-side least squares.
- Optimization status: missing; same primitive as M1 and periodic 1-D.

### Defect Stage B blind finite search

- Producer:
  `calculations/research-monograph/impurity-defect-2d/run_stage_b.py::StageBBlindAlignment`.
- Verifier:
  `calculations/research-monograph/impurity-defect-2d/verify_stage_b.py`.
- Evidence: retained result, 2,304 candidate rows, and independent criteria
  reconstruction.
- Reusable behavior: ordered finite-candidate minimum with a caller-supplied absolute
  ambiguity tolerance.
- Required result: best index, best value, and complete ambiguity index set.
- Optimization status: finite candidate selection missing.
- Numerics status: finite-group actions and twists missing.
- Optimization status: phase-synchronization solver missing.
- Consumer responsibility: candidate meaning and energy-shift interpretation.

### Defect Stage C model hierarchy

- Producer:
  `calculations/research-monograph/impurity-defect-2d/stage_c_parent/model_fitting.py::ParentModelFitter`.
- Verifier:
  `calculations/research-monograph/impurity-defect-2d/stage_c_parent_verification/verifier.py::_fit`.
- Evidence: accepted parent result, 1,040 fits, and independent verification.
- Reusable behavior: real full-rank least-squares fit to a caller-supplied basis.
- Reusable behavior: tolerance-aware ordered comparison across nested classes.
- Repository status: fit mechanics present; explicit numerical fit facade and
  tolerance-aware selection missing.
- Numerics ownership: basis construction.
- Optimization ownership: ordered support-index selection.
- Consumer responsibility: locality meaning and scientific acceptance criteria.

### Optimizer-basin deterministic starts

- Producer:
  `calculations/research-monograph/periodic-2d-optimizer-basin/generate_standalone_starts.py::StandaloneStartDesign._radical_inverse`.
- Verifier:
  `calculations/research-monograph/periodic-2d-optimizer-basin/verify_standalone_proposal.py`.
- Evidence: 16-start table consumed by the completed 256-endpoint study.
- Reusable behavior: deterministic radical-inverse and Halton coordinate generation for
  explicit bases and indices.
- Numerics status: sequence generation missing.
- Optimization status: bounded decision-space mapping and sampler state missing.
- Consumer responsibility: generator, harmonic, and amplitude meaning.

### Optimizer-basin equivalence and clustering

- Producer:
  `calculations/research-monograph/periodic-2d-optimizer-basin/reanalyze_study.py::SquareSymmetryBasinClassifier`.
- Density-aware producer:
  `calculations/research-monograph/periodic-2d-optimizer-basin/extract_standalone_results.py::DensityAwareBasinAction`.
- Verifier:
  `calculations/research-monograph/periodic-2d-optimizer-basin/verify_standalone_results.py`.
- Evidence: completed original and standalone results, threshold sensitivity, and
  independent verification.
- Reusable behavior: finite minimax assignment between equal-size point sets under an
  injected distance.
- Reusable behavior: deterministic ordered representative clustering under an injected
  equivalence predicate or distance.
- Required clustering result: first representative, ordered membership, and complete
  assignments.
- Optimization status: minimax assignment and representative clustering missing.
- Numerics status: finite-group transforms, periodic-center geometry, and density
  reconstruction missing.
- Existing GMM clustering: different behavior.
- Consumer responsibility: equivalence thresholds and scientific interpretation.

### Optimizer-basin right-censored convergence regression

- Producer:
  `calculations/research-monograph/periodic-2d-optimizer-basin/analyze_standalone_convergence_regression.py`.
- Verifier:
  `calculations/research-monograph/periodic-2d-optimizer-basin/verify_standalone_convergence_regression.py`.
- Evidence: retained completed regression and independent reconstruction.
- Completed model: right-censored log-normal accelerated-failure-time likelihood.
- Completed behavior: analytic observation scores and analytic gradient.
- Completed fit: bounded SciPy L-BFGS-B optimization.
- Completed numerical behavior: finite-difference Hessian.
- Completed inference: start-clustered sandwich covariance, conditioning, intervals,
  adjusted time ratios, and convergence probabilities.
- Optimization status: reusable censored-likelihood parameter fitting missing.
- Numerics status: derivative, Hessian, clustered covariance, interval, and probability
  diagnostics missing.
- Backend boundary: SciPy supplies L-BFGS-B; owned capabilities define the reusable
  model, fit records, diagnostics, and composition.

## Confirmed `projectkoios.optimization` backlog

- [ ] Constrained positive-quadratic minimization and exact finite-family selection.
- [ ] Positive-weight complex multiple-right-hand-side linear least squares.
- [ ] Complex square and rectangular Procrustes/polar optimization.
- [ ] Objective-selected reciprocal-frame transport and phase synchronization.
- [ ] Explicit real linear and matrix-subspace fitting.
- [ ] Tolerance-aware finite-candidate ambiguity selection.
- [ ] Linear-sum and minimax assignment.
- [ ] Deterministic representative clustering.
- [ ] Halton decision-space sampler and resumable state.
- [ ] Censored-likelihood parameter fitting.

## Confirmed `projectkoios.numerics` backlog

- [ ] Symmetric eigensystems, subspaces, projectors, and compression.
- [ ] Symmetric Gram orthonormalization and common-space pullback.
- [ ] Represented-matrix norms and observed-order diagnostics.
- [ ] Radical-inverse and Halton sequence generation.
- [ ] Finite-periodic transforms, sewing, frame/projector paths, direct closure and
  holonomy diagnostics, Wilson numerics, and direct real-space reconstruction and
  moments.
- [ ] Finite-group actions, periodic-center geometry, and density reconstruction.
- [ ] Finite-difference Hessian and tensor numerics.
- [ ] Bracket expansion and bounded bisection.
- [ ] Clustered covariance, conditioning, interval, and probability diagnostics.

## Existing repository coverage

- [x] Weighted real affine residual construction.
- [x] Positive-definite completed-square quadratic evaluation.
- [x] Positive diagonal metric geometry.
- [x] Full-rank real vectorized minimizer through affine quadratic construction.
- [x] Scalar affine fitting mechanics.

## Current `projectkoios.optimization` implementation slice

- [x] Add immutable constrained-quadratic results and numerical acceptance tolerance.
- [x] Add nominal solver ABCs, final shared input/result validation, and closed factories.
- [x] Implement exact dense and sparse two-dimensional rectangle minimizers.
- [x] Implement dense and sparse solve-based ellipsoid support.
- [x] Implement ordered family argmin, inclusive filtering, and ambiguity selection.
- [x] Add owner-native M3 regression fixtures.
- [x] Verify correlated edge, corner, tie, empty-sublevel, and failure cases.
- [x] Run tests, Ruff, formatting, strict mypy, wheel import, and reproducible builds.
- [x] Complete independent implementation review and resolve any blocking findings.

## Current least-squares support slice

- [x] Define the unit-free complex-capable multiple-RHS boundary.
- [x] Add nominal solver, immutable result, tolerance, and lazy closed factory.
- [x] Implement NumPy ordinary and positive-weight dense solving without normal equations.
- [x] Add KSD-shaped Fourier-design, rank, condition, failure, and independent-reference tests.
- [x] Run the complete validation and packaging gate.
- [x] Complete independent implementation review and resolve any blocking findings.
