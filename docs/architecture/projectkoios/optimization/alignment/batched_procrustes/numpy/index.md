# NumPy batched complex-unitary right Procrustes alignment

This NumPy-specific capability solves a complete tensor batch without converting matrix
pairs into Python tuple objects.

- Inputs have exact complex128 shape `(..., m, n)` with at least one leading dimension.
- Each completion acts on the right of its corresponding source matrix.
- Leading batch dimensions are preserved exactly in every result array.
- The existing one-pair dependency-free contract remains unchanged.

## Mathematical operation

For every leading index $b$, source $A_b$, and target $B_b$, the solver evaluates the
right-unitary Procrustes problem

$$
\min_{Q_b^\dagger Q_b=I}\lVert A_bQ_b-B_b\rVert_F.
$$

- The cross matrix is $A_b^\dagger B_b$.
- One public batched `numpy.linalg.svd` call decomposes all nonempty cross matrices.
- The SVD-selected completion is $Q_b=U_bV_b^\dagger$.
- Source, target, matrix-row, and matrix-column order remain explicit.

## Rank semantics

Every item retains a unitary SVD completion and separate evidence saying whether that
completion is the uniquely identified polar factor.

- `numerical_ranks` counts singular values strictly above the applied cutoff.
- `full_rank` is true exactly when the numerical rank equals the column count.
- A full-rank completion is the unique Procrustes transformation.
- A deficient completion is diagnostic and must not be represented as a unique solution.
- `all_full_rank`, `rank_deficient_count`, and `first_rank_deficient_index` summarize
  the same per-item evidence without introducing downstream policy.
- `require_unique_right_unitary_transformations()` fails closed at the first deficient
  C-order index and otherwise returns the existing immutable completion array.
- Whole-batch success does not erase deficient-item status.

## Public vocabulary

The public names identify their NumPy ownership because array inputs and outputs are part
of the contract.

- `NumpyBatchedComplexUnitaryProcrustesDefinition`
- `NumpyBatchedComplexUnitaryProcrustesResult`
- `NumpyBatchedComplexUnitaryProcrustesSolver`
- `right_unitary_completions`
- `cross_singular_values`
- `numerical_ranks` and `full_rank`
- `all_full_rank`, `rank_deficient_count`, and `first_rank_deficient_index`
- `require_unique_right_unitary_transformations()`

## Ownership boundary

Optimization owns transformation selection and numerical evidence, while scientific
meaning and acceptance remain downstream.

- Consumers own frame meaning, tensor-grid topology, path direction, and closure.
- Consumers own scientific rank thresholds and whether deficient diagnostics fail a gate.
- This capability owns no lattice, hopping, unit, mesh, retry, or persistence semantics.
- No downstream dependency or publication follows from implementation alone.

## Pages

The supporting pages specify numerical behavior, implementation structure, and flow.

- [Numerical contract](numerics.md)
- [Implementation organization](implementation.md)
- [Consumer needs](scientific.md)
- [Contract and result ownership](contract/index.md)
- [Numerical-evidence ownership](numerics/index.md)
- [Solver ownership](solver/index.md)
- [Schematics](schematics.md)
- [Backend-neutral batched contract](../index.md)
- [One-pair Procrustes contract](../../procrustes/index.md)
