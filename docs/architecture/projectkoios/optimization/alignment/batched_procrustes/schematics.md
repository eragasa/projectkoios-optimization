# Batched Procrustes schematics

The capability keeps consumer semantics outside its mathematical and backend boundaries.

```mermaid
flowchart LR
    C[Consumer arrays and controls] --> B[Batched Procrustes contract]
    B --> N[Concrete backend]
    N --> E[Immutable completions and evidence]
    E --> C
    C --> P[Consumer interpretation and policy]
```

A singleton batch can represent one vertically stacked aggregate without changing the
solver contract.

```mermaid
flowchart LR
    C[Source members] --> S[Scale by one over square root of member count]
    R[Target members] --> T[Scale by one over square root of member count]
    S --> VS[Vertically stack source]
    T --> VT[Vertically stack target]
    VS --> B[Singleton rectangular batch]
    VT --> B
    B --> X[Mean member cross matrix]
    X --> P[Right-unitary Procrustes evidence]
```

Backend ownership is explicit rather than selected implicitly.

```mermaid
flowchart TD
    A[Batched Procrustes] --> NP[NumPy: implemented]
    A --> SP[SciPy: not implemented]
    NP --> R[Validated result]
    SP --> X[No public solver]
```
