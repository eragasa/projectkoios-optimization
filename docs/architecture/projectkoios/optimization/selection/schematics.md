# Selection schematics

## Selection operations

```mermaid
flowchart TD
    V[Ordered objective values] --> M[first_minimum]
    V --> C[indices_at_or_below]
    V --> A[absolute_ambiguity_set]
    M --> MR[First minimizing index and value]
    C --> CR[Ordered feasible indices]
    A --> AR[Minimum and complete ambiguity set]
```

## Consumer composition

```mermaid
flowchart LR
    C[Consumer candidates and labels] --> E[Objective evaluation]
    E --> S[Finite objective selection]
    S --> I[Numeric selected indices]
    I --> C
    C --> D[Interpretation and disposition]
```

Selection never stores consumer labels or assigns scientific meaning to ceilings and
tolerances.

## Closed-form continuous selection

```mermaid
flowchart LR
    K[Single-harmonic coefficients] --> H[Periodic harmonic maximizer]
    H --> V[Mathematical maximum]
    H --> D{Amplitude resolved?}
    D -->|yes| A[Canonical argument]
    D -->|no| N[Numerically constant status]
```

Only the explicitly supported single-harmonic family uses this continuous route. It is
not a generic continuous optimizer.
