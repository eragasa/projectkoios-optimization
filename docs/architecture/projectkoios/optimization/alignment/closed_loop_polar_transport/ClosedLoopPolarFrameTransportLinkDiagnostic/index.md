# ClosedLoopPolarFrameTransportLinkDiagnostic

`ClosedLoopPolarFrameTransportLinkDiagnostic` is the dependency-free immutable evidence
record for one ordered link. Attempted links retain:

- the complete nonincreasing singular-value tuple;
- the exact applied Procrustes rank cutoff;
- `smallest_singular_value - applied_cutoff` as a signed margin;
- the numerical rank; and
- `maximum_absolute_entry_unitarity_defect` only when the link is qualified.

The defect is exactly `max(abs(Q† @ Q - I))`; it is not a raw or normalized Frobenius
quantity.

The record validates local type, finiteness, ordering, margin, rank, and status
relations. `NumpyClosedLoopPolarFrameTransportResult` additionally validates spectrum
length and cutoff policy against the route rank and tolerance.
