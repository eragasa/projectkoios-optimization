# ClosedLoopPolarFrameTransportBackendOperation

`ClosedLoopPolarFrameTransportBackendOperation` identifies which decomposition failed:

- `PROCRUSTES_SVD` for one neighbor or closure Procrustes decomposition;
- `CLOSURE_SCHUR` for endpoint-mismatch eigenphase and distributed-correction
  construction.

It is carried by `ClosedLoopPolarFrameTransportBackendFailure` together with a typed
reason and exact link index.
