# Frobenius schematics

```mermaid
flowchart LR
    A[Finite complex tuple matrix] --> V[Shape and finiteness validation]
    V --> P[Stable row-major sum of squared magnitudes]
    P --> N[Binary64 square root]
    P --> R[ComplexMatrixFrobenius]
    N --> R
    R --> C[Consumer-owned semantic or unit adapter]
```

The consumer adapter is outside Numerics and cannot alter the retained unit-free power
or norm.
