# Batched Procrustes implementation

The implementation is one direct-import NumPy module with mirrored contract,
integration, numerical, and wheel verification.

- No one-option factory is introduced.
- Alignment package initializers remain import-empty.
- The existing scalar module and factory remain unchanged.
- Module `__all__` lists the three public batched names.

## Source organization

The source lives beside the existing one-pair NumPy backend because NumPy arrays are
part of the new public boundary.

```text
src/python/projectkoios/optimization/alignment/
├── procrustes.py
└── numpy/
    ├── procrustes.py
    └── batched_procrustes.py
```

- The definition validates and owns source and target array copies.
- Array copies use immutable byte backing, so callers cannot re-enable NumPy writes.
- The solver owns vectorized cross products, one batched SVD, and evidence construction.
- The keyword-only result validates and owns shape-preserving read-only evidence arrays.

## Memory behavior

Nonempty execution uses vectorized arrays whose sizes are bounded by fixed multiples of
the input and required output sizes.

- Cross matrices and SVD factors scale as `batch_size × n × n`.
- Required completions scale as `batch_size × n × n`.
- Spectra scale as `batch_size × n`.
- Scalar evidence scales as `batch_size`.
- No matrix pair becomes a nested Python tuple object.

## Verification organization

Tests mirror the public records and numerical action without changing scalar fixtures.

- Definition tests cover dtype, shape, finiteness, copy ownership, and empty batches.
- Result tests cover read-only arrays, deficient completion status, and relational evidence.
- Integration tests cover right action, one SVD call, additive cutoff, and indexed failure.
- Numerical tests compare all full-rank items with the scalar owner.
- Deficient parity compares singular/rank/residual evidence but not transformations.
- The wheel test executes the direct public module from an isolated built artifact.
