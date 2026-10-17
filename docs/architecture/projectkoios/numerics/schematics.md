# Numerics schematics

## Capability flow

```mermaid
flowchart TD
    I[Finite numeric inputs] --> V[Validation]
    V --> L[Linear algebra]
    V --> G[Geometry]
    V --> S[Sequences]
    L --> P[Finite-periodic numerics]
    G --> P
    P --> D[Diagnostics]
    L --> D
    D --> R[Immutable numerical results]
```

## Backend composition

```mermaid
flowchart LR
    C[Dependency-free contract and records] --> N[NumPy implementation]
    C --> S[SciPy implementation]
    C --> K[scikit-learn implementation]
    N --> R[Backend-neutral immutable result]
    S --> R
    K --> R
```

Only the selected implementation imports its optional backend.

## Consumer boundary

```mermaid
flowchart LR
    C[Consumer inputs and meanings] --> N[projectkoios.numerics]
    N --> R[Mathematical result]
    R --> C
    C --> I[Physical or scientific interpretation]
```

Numerics returns mathematical state and diagnostics. It does not attach material,
physical, evidentiary, or acceptance meaning.
