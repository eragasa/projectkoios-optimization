# Closed-loop polar frame transport

`projectkoios.optimization.alignment.closed_loop_polar_transport`, its NumPy contract,
and its SciPy executor own one unit-free, rank-generic, closed-loop right-action
transport capability. It accepts an
immutable exact `complex128` inventory `F` with shape `(M, ambient, rank)`, where
`M >= 1` and `0 < rank <= ambient`, plus an explicit endpoint sewing map `S` with shape
`(ambient, ambient)`. `S @ F[0]` is the first frame represented at the terminal endpoint.

## Public API

The dependency-free owner module exports:

- `ClosedLoopPolarFrameTransportStatus`;
- `ClosedLoopPolarFrameTransportLinkKind`;
- `ClosedLoopPolarFrameTransportLinkStatus`;
- `ClosedLoopPolarFrameTransportPrincipalPhaseBranch`;
- `ClosedLoopPolarFrameTransportLinkDiagnostic`;
- `ClosedLoopPolarFrameTransportBackendOperation`;
- `ClosedLoopPolarFrameTransportBackendFailureReason`; and
- `ClosedLoopPolarFrameTransportBackendFailure`.

The NumPy contract module exports:

- `ClosedLoopPolarFrameInventory`;
- `ClosedLoopPolarFrameRightActions`;
- `NumpyClosedLoopPolarFrameTransportDefinition`; and
- `NumpyClosedLoopPolarFrameTransportResult`.

The SciPy execution module exports `ScipyClosedLoopPolarFrameTransportSolver`. Its
constructor requires an explicit `NumpyComplexUnitaryProcrustesSolver`; the action owner
has no empty constructor or static operation.

## Class pages

- [`ClosedLoopPolarFrameTransportStatus`](ClosedLoopPolarFrameTransportStatus/index.md)
- [`ClosedLoopPolarFrameTransportLinkKind`](ClosedLoopPolarFrameTransportLinkKind/index.md)
- [`ClosedLoopPolarFrameTransportLinkStatus`](ClosedLoopPolarFrameTransportLinkStatus/index.md)
- [`ClosedLoopPolarFrameTransportPrincipalPhaseBranch`](ClosedLoopPolarFrameTransportPrincipalPhaseBranch/index.md)
- [`ClosedLoopPolarFrameTransportLinkDiagnostic`](ClosedLoopPolarFrameTransportLinkDiagnostic/index.md)
- [`ClosedLoopPolarFrameTransportBackendOperation`](ClosedLoopPolarFrameTransportBackendOperation/index.md)
- [`ClosedLoopPolarFrameTransportBackendFailureReason`](ClosedLoopPolarFrameTransportBackendFailureReason/index.md)
- [`ClosedLoopPolarFrameTransportBackendFailure`](ClosedLoopPolarFrameTransportBackendFailure/index.md)
- [`NumpyClosedLoopPolarFrameTransportDefinition`](NumpyClosedLoopPolarFrameTransportDefinition/index.md)
- [`NumpyClosedLoopPolarFrameTransportResult`](NumpyClosedLoopPolarFrameTransportResult/index.md)
- [`ScipyClosedLoopPolarFrameTransportSolver`](ScipyClosedLoopPolarFrameTransportSolver/index.md)

The definition factories are `from_arrays(...)`, which makes immutable defensive copies,
and `from_immutable_arrays(...)`, which accepts only canonical immutable byte-backed
arrays. Both require an explicit `principal_phase_branch`: `negative_pi_open` selects
`(-pi, pi]`, while `positive_pi_open` selects `[-pi, pi)`. Frames must have orthonormal
columns within the Procrustes validation tolerance. The sewing map is finite and
shape-exact; the capability does not assign it physical meaning or assume it is unitary.

## Orientation and closure

Let `T[0] = F[0]`. For neighbor `j`, the existing scalar Procrustes owner solves

\[
\min_{Q^\dagger Q=I}\lVert F_{j+1}Q-T_j\rVert_F,
\]

using the cross matrix `F[j+1]† @ T[j]`. Only its full-rank transformation is admitted,
and then `T[j+1] = F[j+1] @ Q`. The closure link is last in edge order and solves

\[
\min_{H^\dagger H=I}\lVert T_{M-1}H-S T_0\rVert_F.
\]

Thus its cross matrix is `T[M-1]† @ S @ T[0]`. Call the admitted closure right action
`A`; it maps the primitive last frame toward the sewn first frame. The endpoint-mismatch
orientation is retained separately as `E = A†`. A complex Schur decomposition represents

\[
E = Z\,\operatorname{diag}(e^{i\gamma_a})\,Z^\dagger
\]

on the definition's explicit principal-phase branch. The distributed correction is

\[
W = Z\,\operatorname{diag}(e^{-i\gamma_a/M})\,Z^\dagger,
\qquad W^M = E^\dagger = A.
\]

The returned qualified frames are `X[j] = T[j] @ W**j`. This explicit inverse-orientation
crosswalk matches protocols that define endpoint mismatch as `E` and apply negative
fractional phases. At an exact `-1` eigenvalue, `negative_pi_open` selects `gamma = +pi`
and therefore the negative root `exp(-i*pi/M)`; `positive_pi_open` selects the conjugate
root through `gamma = -pi`. There are `M` edge diagnostics: `M-1` neighbors and one sewn
closure edge.

## Qualification and diagnostics

Every edge reuses `ComplexUnitaryProcrustesTolerance`, including its additive or maximum
rank-cutoff rule and strict `singular_value > cutoff` comparison. Equality is deficient.
Each edge diagnostic retains its full singular-value vector, applied cutoff, signed
smallest-singular-value margin, numerical rank, and
`maximum_absolute_entry_unitarity_defect` when a unique map exists.

A deficient neighbor stops progression. Its dependent neighbor and closure links are
explicitly `UNATTEMPTED`; no SVD null-space completion is used. Closure deficiency has no
dependent unattempted link. A deficient result exposes no transported frames, actions,
endpoint mismatch, distributed correction, Schur vectors, phases, or closure evidence. A
qualified result additionally retains:

- the exact input definition;
- immutable transported frames and their per-frame right actions;
- closure alignment right action `A`, endpoint mismatch `E = A†`, distributed correction
  step `W`, and ordered Schur vectors;
- `maximum_absolute_entry_closure_alignment_residual`, equal to
  `max(abs(T[-1] @ A - S @ T[0]))`;
- `maximum_absolute_entry_closure_alignment_unitarity_defect`, equal to
  `max(abs(A† @ A - I))`; and
- `ordered_closure_endpoint_mismatch_phases` in ascending storage order.

Every named defect states its metric. These maximum-absolute-entry quantities are
software-validation evidence, not normalized Frobenius protocol diagnostics. Consumers
can compute raw or rank-normalized Frobenius quantities from the retained frames,
actions, and closure matrices without treating deterministic phase storage order as a
scientific phase labeling.

SVD and Schur nonconvergence, nonfinite output, and invalid decomposition evidence raise
`ClosedLoopPolarFrameTransportBackendFailure`. Its exact `operation`, `reason`, and
`link_index` let consumers map unresolved computation without parsing exception text.
Mathematical rank deficiency is a successful typed result and remains distinct.

The result constructor independently verifies progression order, spectrum length and
cutoff policy, right-action identities, Schur reconstruction, phase branch, correction
power, orientation identities, metric-explicit unitarity evidence, and closure-alignment
residual evidence.

## Excluded scope

This capability owns no units, reciprocal-lattice convention, mesh or protocol identity,
scientific overlap threshold, acceptance status, physical interpretation, or topology
claim. Consumers supply path order and sewing orientation and decide what the diagnostics
mean.

Full sparse-frame transport is excluded: dense rank-space right actions generally destroy
frame sparsity, while complete overlap SVD and closure Schur operations are dense. A
future complex-CSR sewing-map representation requires concrete evidence that dense
ambient-square sewing storage is unacceptable; it does not require a sparse transport
solver.

## Pages

- [Numerics](numerics.md)
- [Implementation](implementation.md)
- [Schematics](schematics.md)
- [Alignment](../index.md)
