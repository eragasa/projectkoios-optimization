# ClosedLoopPolarFrameTransportStatus

`ClosedLoopPolarFrameTransportStatus` is the closed route-level numerical disposition in
`projectkoios.optimization.alignment.closed_loop_polar_transport`.

- `QUALIFIED` means all `M-1` neighbor links and the sewn closure link have full
  numerical rank and qualified closure output is present.
- `RANK_DEFICIENT` means progression encountered exactly one deficient link; dependent
  links are explicitly unattempted and no transported route or closure output is
  exposed.

This enum is numerical software evidence, not a workflow, protocol, or scientific
acceptance status.
