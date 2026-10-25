# Matrix-norm contract schematics

```mermaid
flowchart LR
    E[MatrixNormEnum semantic identity]
    M[Concrete MatrixT representation]
    A[MatrixNormABC final callable boundary]
    I[Subclass representation checks and evaluation]
    V[Implementation result]
    B[Final type, finiteness, and sign validation]
    R[Exact finite nonnegative built-in float]

    E --> A
    M --> A --> I --> V --> B --> R
```

Concrete dense NumPy and SciPy CSR Frobenius evaluators and the dense NumPy
maximum-entry evaluator implement this foundation in capability-local backend modules.
Consumers attach units and presentation labels after receiving the unit-free scalar.
