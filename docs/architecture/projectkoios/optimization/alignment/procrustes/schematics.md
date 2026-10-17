# Procrustes schematics

```mermaid
flowchart TD
    D[ComplexUnitaryProcrustesDefinition] --> C[Construct A† B]
    C --> S[Backend square SVD]
    S --> V[Shared SVD validation]
    V --> K[Apply declared numerical-rank cutoff]
    K --> F{Full numerical rank?}
    F -->|yes| Q[Construct Q = U V†]
    Q --> U[Validate Q† Q and direct residual]
    F -->|no| N[Return transformation = None]
    V --> O[Reconstruct optimal residual]
    U --> R[ComplexUnitaryProcrustesResult]
    N --> R
    O --> R
```

Transient backend null-space vectors may occur inside a complete SVD, but they are not
used to construct or expose a deficient full transformation.

## Batched distinction

The NumPy batch capability preserves homogeneous array shape while making uniqueness a
separate per-item status.

- Full-rank items expose unique right transformations.
- Deficient items expose diagnostic SVD completions marked non-full-rank.
- See the [batched schematics](../batched_procrustes/schematics.md) for vectorized flow.
