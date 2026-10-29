# ClosedLoopPolarFrameTransportPrincipalPhaseBranch

`ClosedLoopPolarFrameTransportPrincipalPhaseBranch` is the required dependency-free
closure phase convention carried by every definition.

- `negative_pi_open` selects `(-pi, pi]`.
- `positive_pi_open` selects `[-pi, pi)`.

The branch controls the distributed correction at the `-1` eigenvalue. It does not
assign an ordering or scientific band label to the phase spectrum. Results use ascending
storage order only so serialization is deterministic; consumers treat the values as a
multiset when that is their scientific contract.
