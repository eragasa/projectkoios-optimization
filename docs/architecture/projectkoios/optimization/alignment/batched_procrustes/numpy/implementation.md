# Batched Procrustes implementation

The implementation is an encapsulated NumPy capability package with explicit contract,
numerical-evidence, and solver ownership.

- No one-option factory is introduced.
- Parent alignment and NumPy package initializers remain import-empty.
- The capability initializer is a deliberate façade over `base.py`.
- Scalar and batched nominal solvers inherit one generic shared-policy owner.
- Façade `__all__` lists only the four approved batched names.

## Source organization

The source lives beside the existing one-pair NumPy backend because NumPy arrays are
part of the new public boundary.

```text
src/python/projectkoios/optimization/alignment/
├── procrustes.py
└── numpy/
    ├── procrustes.py
    └── batched_procrustes/
        ├── __init__.py
        ├── base.py
        ├── contract.py
        ├── numerics.py
        └── solver.py
```

- `__init__.py` re-exports only the approved façade vocabulary.
- `base.py` is the canonical supported API surface and owns no numerical behavior.
- `contract.py` owns the nominal batched solver, definition, result, immutability,
  relational invariants, vectorized rank-cutoff policy, and fail-closed uniqueness
  access.
- `numerics.py` owns Procrustes-specific residual reconstruction, bounds, and indexed
  numerical evidence without exporting them through the façade.
- `solver.py` owns NumPy SVD execution, backend validation, failure isolation, and result
  assembly.
- Definition and result `from_arrays(...)` factories own immutable byte-backed copies,
  while explicit `from_immutable_arrays(...)` factories retain canonical storage by
  identity without copying.
- Raw frozen constructors accept only C-contiguous, read-only arrays whose base chain
  terminates in immutable bytes; callers cannot re-enable NumPy writes.
- The solver owns vectorized cross products, one batched SVD, and evidence construction.
- The keyword-only result validates shape-preserving evidence relationally without
  postconstruction mutation or mutable-then-sealed state.
- Result-level rank summaries and the fail-closed unique-transformation accessor derive
  solely from the immutable `full_rank` array and allocate no matrix copies.

## Memory behavior

Nonempty execution uses vectorized arrays whose sizes are bounded by fixed multiples of
the input and required output sizes.

- Cross matrices and SVD factors scale as `batch_size × n × n`.
- Required completions scale as `batch_size × n × n`.
- Spectra scale as `batch_size × n`.
- Scalar evidence scales as `batch_size`.
- No matrix pair becomes a nested Python tuple object.

## Verification organization

Tests mirror every class and reuse named Procrustes fixtures across scalar and batched
ownership boundaries.

- Definition tests cover dtype, shape, finiteness, copy ownership, and empty batches.
- Result tests cover read-only arrays, deficient completion status, rank summaries,
  fail-closed unique access, and relational evidence.
- `ResultArrays`, `CompletionEvidence`, and `ResidualArrays` each have direct tests for
  exact field identity, frozen slots, and façade exclusion; owning result and solver
  tests cover their transported relations.
- `ResidualReconstruction` has direct tests for positive, zero-scale, cancellation,
  multidimensional, indexed-negative, nonfinite, dtype, frozen-slot, and façade behavior.
- Solver integration tests cover right action, one SVD call, additive cutoff, and indexed
  failure.
- Numerical tests compare all full-rank items with the scalar owner.
- Deficient parity compares singular/rank/residual evidence but not transformations.
- The wheel test executes the package façade from an isolated built artifact.
