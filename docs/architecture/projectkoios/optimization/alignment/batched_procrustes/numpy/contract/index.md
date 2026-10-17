# Batched Procrustes contract ownership

The `contract.py` module owns the exact NumPy definition and result boundary. It is the
only module allowed to normalize caller arrays into immutable public records or decide
whether stored rank, residual, completion, and backend evidence are relationally
consistent.

## Owned classes

- [`BatchedComplexUnitaryProcrustesSolver`](BatchedComplexUnitaryProcrustesSolver/index.md)
  specializes shared solver policy to the exact-array batch contract.
- [`NumpyBatchedComplexUnitaryProcrustesDefinition`](NumpyBatchedComplexUnitaryProcrustesDefinition/index.md)
  owns validated source and target batches.
- [`NumpyBatchedComplexUnitaryProcrustesResult`](NumpyBatchedComplexUnitaryProcrustesResult/index.md)
  owns immutable evidence, rank summaries, and fail-closed unique access.
- [`ResultArrays`](ResultArrays/index.md) groups normalized arrays during result
  construction and does not cross the public façade.

## Boundary

The contract module does not call SVD, choose a backend, define scientific acceptance,
or interpret deficient completions. Numerical reconstruction is delegated to
`numerics.py`; backend execution is delegated to `solver.py`.
