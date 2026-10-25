# Dense spectral-norm schematics

```mermaid
flowchart LR
    D[Dense NumPy float64 or complex128 matrix]
    V[Exact representation, shape, dtype, and finiteness validation]
    S[Complete NumPy singular-value decomposition]
    L[Largest singular value]
    B[MatrixNormABC final result validation]
    F[Exact finite nonnegative built-in float]

    D --> V --> S --> L --> B --> F
```

The complete dense path has no iterative controls or residual record. Sparse spectral
norms require a separate reviewed architecture and do not inherit this implementation's
acceptance.
