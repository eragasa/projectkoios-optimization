# Alignment schematics

## One-pair right alignment

```mermaid
flowchart LR
    A[Source matrix A] --> C[Cross matrix A† B]
    B[Target matrix B] --> C
    C --> S[Validated square SVD]
    S --> E[Singular values and rank cutoff]
    E --> R{Numerically full rank?}
    R -->|yes| Q[Unique right-unitary Q]
    R -->|no| N[No full transformation]
    A --> O[Optimal residual evidence]
    B --> O
    E --> O
    Q --> X[Apply A Q toward B]
```

The deficient branch deliberately does not use the SVD's arbitrary null-space bases to
construct a public unitary completion. Closed-loop transport reuses this branch for every
ordered neighbor and closure link; its separate
[schematic](closed_loop_polar_transport/schematics.md) shows progression,
endpoint-mismatch orientation, and branch-explicit correction distribution. Consumers retain frame interpretation and path/sewing meaning.
