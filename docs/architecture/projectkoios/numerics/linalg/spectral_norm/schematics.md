# Dense spectral-norm schematics

```mermaid
flowchart LR
    D[Exact dense NumPy ndarray]
    V[Class, dtype, shape, emptiness, and finiteness validation]
    S[Complete numpy.linalg.svdvals call]
    L[First descending singular value]
    B[MatrixNormABC scalar validation]
    F[Exact finite nonnegative built-in float]

    D --> V --> S --> L --> B --> F
    V -. malformed input .-> X[TypeError or ValueError]
    S -. NumPy LinAlgError .-> R[RuntimeError]
    S -. nonfinite output .-> P[FloatingPointError]
```

The dense path contains no iterative controls, residual record, owner-level scaling,
fallback algorithm, unit handling, or sparse conversion. No sparse spectral-norm
implementation exists. Its iterative contract requires a separate architecture review.
