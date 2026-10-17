# Batched right-unitary Procrustes alignment

Batched Procrustes selects one right-unitary completion for every source-target pair in a
shared leading tensor shape. The capability preserves per-item rank and residual evidence
without converting the batch into Python tuples.

## Contract direction

For every leading index $b$, the mathematical operation is

$$
\min_{Q_b^\dagger Q_b=I}\lVert A_bQ_b-B_b\rVert_F.
$$

The cross matrix is $A_b^\dagger B_b$, the selected orientation acts on the right, and
rank-deficient completions remain diagnostics rather than unique transformations.

## Backend status

- [NumPy backend](numpy/index.md) is implemented and is the supported array contract.
- [SciPy backend](scipy/index.md) is not implemented; the chapter records consumer needs
  and prevents accidental implication of a second backend.
- The scalar capability remains documented under [Procrustes](../procrustes/index.md).

## Pages

- [Implementation boundary](implementation.md)
- [Numerical contract](numerics.md)
- [Consumer needs](scientific.md)
- [Schematics](schematics.md)
