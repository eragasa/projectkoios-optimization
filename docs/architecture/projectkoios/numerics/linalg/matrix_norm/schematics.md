# Matrix-norm contract schematics

```mermaid
flowchart LR
    E[MatrixNormEnum semantic identity]
    M[Concrete MatrixT representation]
    A[MatrixNormABC construction retains MatrixT]
    C[Final result boundary]
    I[Subclass representation checks and evaluation]
    V[Implementation result]
    B[Final type, finiteness, and sign validation]
    R[Exact finite nonnegative built-in float]

    E --> A
    M --> A --> C --> I --> V --> B --> R
```

Concrete dense NumPy and SciPy CSR Frobenius and maximum-entry evaluations and the dense
NumPy spectral-norm evaluation implement this foundation in capability-local backend
modules. Consumers attach units and presentation labels after receiving the unit-free
scalar.
