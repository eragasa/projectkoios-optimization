# Optimization schematics

## Objective-based ownership

```mermaid
flowchart LR
    F[Function or set of functions] --> P[Optimization problem]
    P --> S[Solver, fitter, selector, matcher, clusterer, or sampler]
    S --> R[Selected arguments and objective diagnostics]
```

## Capability composition

```mermaid
flowchart TD
    N[projectkoios.numerics] --> L[Least squares]
    N --> A[Alignment and Procrustes]
    N --> Q[Quadratic minimization]
    N --> M[Assignment]
    N --> C[Clustering]
    L --> R[Optimization results]
    A --> R
    Q --> R
    M --> R
    C --> R
```

## Consumer boundary

```mermaid
flowchart LR
    C[Consumer] --> O[Optimization problem inputs]
    O --> R[Selected mathematical result]
    R --> C
    C --> I[Physical or scientific interpretation]
```

Optimization owns objective definitions and selection behavior. Consumers own physical
meaning, units, scientific thresholds, and acceptance.
