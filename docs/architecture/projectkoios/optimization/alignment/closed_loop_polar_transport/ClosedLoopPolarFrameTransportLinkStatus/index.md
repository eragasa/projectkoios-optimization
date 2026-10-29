# ClosedLoopPolarFrameTransportLinkStatus

`ClosedLoopPolarFrameTransportLinkStatus` records progression at one ordered link.

- `QUALIFIED` retains full-rank spectrum, cutoff, margin, rank, and unitary defect.
- `RANK_DEFICIENT` retains spectrum, cutoff, margin, and numerical rank but no unitary
  map or unitary defect.
- `UNATTEMPTED` retains no numerical evidence because an earlier deficient link stopped
  progression.

A route result admits exactly the sequence `QUALIFIED*`, optionally followed by one
`RANK_DEFICIENT` and then `UNATTEMPTED*`.
