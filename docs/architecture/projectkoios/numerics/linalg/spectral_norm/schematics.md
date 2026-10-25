# Matrix spectral-norm schematics

## Implementation selection

```mermaid
flowchart TB
    E[MatrixNormEnum.spectral]
    D[NumpySpectralNorm]
    S[ScipyCsrSpectralNorm plus mandatory controls]
    DS[Complete numpy.linalg.svdvals]
    C[Private canonical CSR copy]
    M{Minimum dimension}
    Z[Exact canonical zero]
    G[Scale-safe sparse small Gram including zero]
    A[Deterministic ARPACK svds]
    Q[Internal unit-vector and two-equation residual qualification]
    R[MatrixNormABC scalar validation]

    E --> D --> DS --> R
    E --> S --> C --> M
    M -->|at most 2| G --> R
    M -->|at least 3 and no stored entries| Z --> R
    M -->|at least 3 and nonzero| A --> Q --> R
```

## Sparse failure flow

```mermaid
flowchart LR
    V[CSR representation and control validation]
    K[Canonicalization]
    N{Direct or iterative numerical branch}
    T[Triplet qualification for iterative branch]
    F[Built-in finite nonnegative float]

    V --> K --> N
    N -->|direct| F
    N -->|iterative| T --> F
    V -. malformed public input .-> X[TypeError or ValueError]
    K -. nonfinite duplicate sum .-> P[FloatingPointError]
    N -. ARPACK or decomposition nonconvergence .-> R[RuntimeError]
    N -. nonfinite arithmetic .-> P
    T -. finite excessive residual .-> R
    T -. nonfinite arithmetic or nonunit vectors .-> P
```

The sparse evaluator has no dense conversion, dense-evaluator delegation, solver
fallback, public residual record, cache, function facade, or implicit controls object.
