# Batched Procrustes numerical contract

Every backend must preserve the same mathematical orientation and explicit evidence.

- Cross matrices are `source.conj().swapaxes(-1, -2) @ target`.
- Selected completions act as `source @ transformation`.
- Complete singular spectra support additive rank cutoffs and residual reconstruction.
- Equality at the rank cutoff is deficient.
- Cancellation-scale squared residuals may become exact zero only with their applied
  squared-unit roundoff cutoff retained.
- Empty leading dimensions preserve shape and do not imply backend execution.
- Backend failure rejects the whole batch; numerical deficiency remains successful,
  status-bearing evidence.

The [NumPy numerics](numpy/numerics.md) specify the implemented binary64 realization.
