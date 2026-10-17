# NumpyBatchedComplexUnitaryProcrustesSolver

`NumpyBatchedComplexUnitaryProcrustesSolver` owns vectorized NumPy execution for one
validated batch.

## Operation

- It constructs `source.conj().swapaxes(-1, -2) @ target`.
- A nonempty batch makes one public batched `numpy.linalg.svd` call.
- It selects the right-unitary completion as `left @ right_adjoint`.
- It validates SVD factors, reconstruction, completion unitarity, direct residuals, and
  finite evidence before constructing the public result.
- It applies the owner additive rank cutoff; equality is rank deficient.
- An empty batch returns exact shape-preserving evidence without calling SVD or claiming
  backend evidence.

## Failures and verification

The solver rejects wrong definitions with `TypeError`, inconsistent backend evidence
with `ValueError`, nonfinite arithmetic with `FloatingPointError`, and SVD
nonconvergence with `RuntimeError`. Indexed failures report the first reproducible
C-order item where possible.

Dedicated integration and numerical tests cover orientation, multidimensional batches,
one-call vectorization, cutoff equality, empty batches, scalar parity, indexed failures,
overflow, and isolated-wheel execution.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
