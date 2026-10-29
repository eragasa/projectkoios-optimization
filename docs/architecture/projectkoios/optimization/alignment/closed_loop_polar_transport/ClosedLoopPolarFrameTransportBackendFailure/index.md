# ClosedLoopPolarFrameTransportBackendFailure

`ClosedLoopPolarFrameTransportBackendFailure` is the bounded typed `RuntimeError` raised
for decomposition failure. It retains exact:

- `operation: ClosedLoopPolarFrameTransportBackendOperation`;
- `reason: ClosedLoopPolarFrameTransportBackendFailureReason`; and
- `link_index: int`.

The original backend exception is chained when present. Downstream mappings use the
typed fields and never exception text.
