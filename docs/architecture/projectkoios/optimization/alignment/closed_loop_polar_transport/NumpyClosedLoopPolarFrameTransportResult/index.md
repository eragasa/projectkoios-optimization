# NumpyClosedLoopPolarFrameTransportResult

`NumpyClosedLoopPolarFrameTransportResult` is the immutable, self-validating exact-array
route result. It retains the exact definition and exhaustive ordered diagnostics.

A deficient result exposes no transported frames, actions, endpoint mismatch,
distributed correction, Schur vectors, phases, or closure evidence. A qualified result
additionally retains immutable:

- transported frames and per-frame right actions;
- closure alignment right action `A` and endpoint mismatch `E = A†`;
- distributed correction step `W`, with `W**M = E† = A`;
- ordered Schur vectors and ascending endpoint-mismatch phases;
- metric-explicit maximum-absolute-entry closure residual and unitarity evidence; and
- SVD and Schur backend identity/version evidence.

Construction verifies progression order, spectrum length and cutoff policy, array shape
and immutability, right-action identities, Schur reconstruction, the definition's
principal-phase branch, correction power, orientation relations, and metric-explicit
closure evidence. Ascending phase order is deterministic storage only; it does not assign
scientific labels to the phase multiset. The retained frames, actions, and matrices let
consumers calculate separate raw or rank-normalized Frobenius protocol diagnostics.
