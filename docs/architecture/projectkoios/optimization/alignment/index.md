# Alignment optimization

`projectkoios.optimization.alignment` owns reusable capabilities that select
transformations by minimizing unit-free alignment objectives.

## Current capability

`procrustes.py` owns one-pair complex-unitary right alignment:

$$
\min_{Q^\dagger Q=I}\lVert A Q-B\rVert_F.
$$

The source matrix is $A$, the target matrix is $B$, and the returned transformation
acts on the right of the source. The result retains the cross-matrix singular values,
applied numerical-rank cutoff, numerical rank, source and target squared Frobenius
norms, and independently reconstructable optimal residual.

A full transformation is returned only when the declared numerical-rank policy finds
$A^\dagger B$ full rank. A deficient one-pair result contains no full transformation,
because the data do not identify its null-space action. A distinct NumPy batch contract
retains SVD-selected diagnostic completions for homogeneous array processing while
marking every item's effective rank and full-rank status explicitly.

## Public vocabulary

- `ComplexUnitaryProcrustesMatrix`
- `ComplexUnitaryProcrustesTolerance`
- `ComplexUnitaryProcrustesDefinition`
- `ComplexUnitaryProcrustesStatus`
- `ComplexUnitaryProcrustesResult`
- `ComplexUnitaryProcrustesSolver`
- `ComplexUnitaryProcrustesMethod`
- `ComplexUnitaryProcrustesSolverFactory`
- `NumpyComplexUnitaryProcrustesSolver`
- `NumpyBatchedComplexUnitaryProcrustesDefinition`
- `NumpyBatchedComplexUnitaryProcrustesResult`
- `NumpyBatchedComplexUnitaryProcrustesSolver`

## Excluded scope

This slice does not own real-orthogonal, partial-isometry, aggregate, frame-path,
transport, closure, synchronization, assignment, or scientific null-space acceptance
behavior. Batched rectangular source/target arrays are supported only with square
right-unitary completions; rectangular transformations remain outside scope.

Consumers own frame meaning, path ordering, retained-frame protocols, scientific rank
thresholds, periodic closure, meshes, failure policy, and acceptance. The numerical-rank
tolerances here are software controls and do not establish scientific identifiability.

## Pages

- [Complex-unitary Procrustes alignment](procrustes/index.md)
- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Optimization namespace](../index.md)
