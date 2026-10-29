# ClosedLoopPolarFrameTransportLinkKind

`ClosedLoopPolarFrameTransportLinkKind` identifies the fixed edge role in one ordered
closed route.

- `NEIGHBOR` applies at indices `0` through `M-2`.
- `CLOSURE` applies only at index `M-1` and compares the transported last frame with the
  explicitly sewn first frame.

The result validates kind and index together, so callers cannot relabel closure evidence
as ordinary neighbor evidence.
