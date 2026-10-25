# Frobenius schematics

```mermaid
flowchart LR
    T[Finite complex tuple matrix] --> V[Tuple validation]
    D[Dense NumPy float64 or complex128 matrix] --> W[Dense validation]
    S[SciPy float64 or complex128 CSR matrix or array] --> K[Private-copy CSR canonicalization]
    V --> P[Stable ordered sum of squared magnitudes]
    W --> P
    K --> P
    P --> R[ComplexMatrixFrobenius detailed tuple result]
    P --> U[NumpyFrobeniusPower scalar result]
    P --> C[ScipyCsrFrobeniusPower scalar result]
    P --> N[Binary64 square root]
    N --> R
    N --> E[NumpyFrobeniusNorm scalar result]
    N --> Q[ScipyCsrFrobeniusNorm scalar result]
```

The dependency-free tuple path retains dimensions, power, and norm in one immutable
record. Optional backend paths expose power and norm as separate unit-free scalar
capabilities. Consumer-owned semantic and unit adapters remain outside Numerics.
