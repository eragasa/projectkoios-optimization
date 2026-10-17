# Batched Procrustes solver ownership

The `solver.py` module owns NumPy backend execution and rejects the whole batch when any
backend or post-SVD validation fails.

## Owned classes

- [`NumpyBatchedComplexUnitaryProcrustesSolver`](NumpyBatchedComplexUnitaryProcrustesSolver/index.md)
  owns vectorized execution and public result assembly.
- [`CompletionEvidence`](CompletionEvidence/index.md) groups selected completions and
  rank evidence during solver execution.
- [`ResidualArrays`](ResidualArrays/index.md) groups power and residual evidence during
  solver execution.

## Boundary

The solver consumes the immutable contract and numerical operations. It does not redefine
result invariants, expose internal evidence records through the façade, or own downstream
scientific thresholds and interpretations.
