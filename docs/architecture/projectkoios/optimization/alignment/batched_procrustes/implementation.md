# Batched Procrustes implementation boundary

The batched capability separates backend-neutral mathematical direction from implemented
array ownership.

- The NumPy package owns the only current definition, result, and solver.
- A package façade exposes approved public objects while contract, numerical evidence,
  and solver behavior remain in explicit implementation modules.
- No nominal one-option factory is introduced.
- Adding another backend requires its own concrete contract and qualification rather than
  silent coercion through NumPy.

See the [NumPy implementation](numpy/implementation.md) for the current source tree and
the [SciPy chapter](scipy/implementation.md) for the intentionally absent backend.
