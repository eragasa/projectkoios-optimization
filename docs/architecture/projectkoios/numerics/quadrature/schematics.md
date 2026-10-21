# Uniform one-dimensional quadrature schematics

## Capability flow

```mermaid
flowchart TD
    D[Closed interval and endpoint-inclusive count] --> R[Uniform trapezoidal rule]
    R --> W[Immutable full weights]
    W --> M[Diagonal full mass matrix]
    W --> Q{At least three full-grid points?}
    Q -->|Yes| I[Indices 1 through N minus 2]
    I --> IW[Equal interior weights h]
    IW --> IM[Diagonal interior mass matrix h I]
    Q -->|No| X[Interior restriction rejected]
```

## Ownership boundary

```mermaid
flowchart LR
    P[PhysKit unit-bearing interval and state] --> U[Unit-stripped interval definition]
    U --> N[projectkoios.numerics.quadrature]
    N --> W[Unit-free immutable weights and diagonal mass]
    W --> P
    P --> S[Sampled-state normalization and physical interpretation]
```

Numerics selects no physical unit, boundary-condition model, eigenvector normalization,
or scientific acceptance policy. The homogeneous-Dirichlet operation only declares how
a caller-established zero-endpoint representation restricts the complete quadrature.
