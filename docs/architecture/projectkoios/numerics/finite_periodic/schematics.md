# Finite-periodic schematics

## Representation flow

```mermaid
flowchart LR
    M[Centered uniform mesh] --> T[Implemented matrix forward/inverse transform]
    T --> D[Parseval and cyclic Hermiticity diagnostics]
    M --> S[Prospective sewing map]
    F[Orthonormal frame path] --> P[Projector path]
    F --> O[Projected-operator path]
    S --> B[Sewn boundary overlap]
    B --> W[Wilson product]
    W --> H[Eigenphases and holonomy diagnostics]
    T --> L[Direct real-space reconstruction and moments]
```

## Optimization composition

```mermaid
flowchart LR
    N[Finite-periodic records] --> A[Optimization alignment]
    N --> X[Optimization assignment]
    A --> R[Aligned or transported result]
    X --> C[Matched phase or point-set result]
```

Numerics constructs and evaluates the finite-periodic representation. Optimization
chooses transformations or assignments that optimize declared objectives. Wannier gauge
optimization and localization are not part of either planned ProjectKoios namespace;
they remain Wannier90-owned.

## Sewing distinction

```mermaid
flowchart TD
    I[Finite reciprocal index basis] --> S[Nonunitary partial coefficient shift]
    M[Mesh neighbor request] --> W[Wrapped mesh index]
    M --> G[Integer reciprocal translation]
    S --> B[Sewn endpoint frame]
    W --> B
    G --> B
```

The coefficient sewing map, wrapped mesh index, and integer reciprocal translation are
separate mathematical objects.
