# Closed-loop polar transport schematics

```mermaid
flowchart TD
    D[Immutable frames F, sewing S, and explicit phase branch] --> N[Next neighbor Procrustes link]
    N --> R{Full numerical rank?}
    R -->|no| X[Retain deficiency; mark dependent links unattempted]
    R -->|yes| A[Apply unique right action to next frame]
    A --> M{More neighbors?}
    M -->|yes| N
    M -->|no| C[Closure Procrustes: last toward S first]
    C --> G{Closure full rank?}
    G -->|no| Y[Retain closure deficiency; withhold route output]
    G -->|yes| H[Closure action A and endpoint mismatch E = A dagger]
    H --> Q[Validated Schur factors and branch-selected phases of E]
    Q --> P[Correction W with W to the M equals E dagger]
    P --> T[Return frame j multiplied by W to the j]
```

There are exactly `M` edge diagnostics: neighbor indices `0` through `M-2`, followed by
the sewn closure at index `M-1`. Backend decomposition failure raises a typed exception;
rank deficiency remains represented evidence. Phase sorting is deterministic storage,
not a scientific ordering of the phase multiset.
