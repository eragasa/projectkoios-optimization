# ScipyClosedLoopPolarFrameTransportSolver

`ScipyClosedLoopPolarFrameTransportSolver` owns ordered execution and complex Schur
closure distribution. Its constructor requires an explicit
`NumpyComplexUnitaryProcrustesSolver`; it has no empty constructor, static operation, or
module-level numerical helper.

For each neighbor it applies the unique full-rank right Procrustes transformation. It
stops at the first deficient link and marks all dependent links unattempted. The final
link obtains closure action `A` by aligning the primitive last frame toward
`endpoint_sewing_map @ first_frame`. The endpoint-mismatch orientation is retained as
`E = A†`.

Qualified closure applies `scipy.linalg.schur` to `E`, uses the definition's explicit
principal-phase branch, stably sorts phase/vector pairs for deterministic storage, and
constructs `W = exp(-i gamma/M)` so `W**M = E† = A`. At eigenvalue `-1`,
`negative_pi_open` selects `gamma = +pi` and the negative fractional root required by
that convention.

Nonconvergence, nonfinite output, and invalid backend evidence raise the typed backend
failure with the exact operation, reason, and link index.
