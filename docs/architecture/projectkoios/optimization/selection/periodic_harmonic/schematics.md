# Periodic harmonic schematics

## Ownership flow

```mermaid
flowchart LR
    C[Consumer-derived offset and harmonic coefficients] --> H[PeriodicHarmonic]
    H --> M[PeriodicHarmonicMaximizer]
    T[Owner degeneracy tolerance] --> M
    M --> R[Maximum value and amplitude]
    M --> S{Amplitude resolved?}
    S -->|yes| A[Canonical argument in half-open interval]
    S -->|no| N[No unique argument]
    R --> E[Consumer evidence and interpretation]
    A --> E
    N --> E
```

## Canonicalization

```mermaid
flowchart TD
    P[atan2 of sine and cosine coefficients] --> D[Divide phase by harmonic order]
    D --> W[Reduce modulo the fundamental period]
    W --> I[Shift into the declared half-open interval]
    I --> B{Rounded to excluded endpoint?}
    B -->|yes| L[Use included interval start]
    B -->|no| C[Use reduced argument]
```

The owner selects only a numeric periodic argument. Coefficient construction and all
scientific labels remain outside the capability.
