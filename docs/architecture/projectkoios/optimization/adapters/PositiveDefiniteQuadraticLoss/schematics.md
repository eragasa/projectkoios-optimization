# Schematics

## Object composition

```mermaid
classDiagram
    class PositiveDefiniteQuadraticLoss {
        +center: tuple
        +matrix: object
        +minimum_squared_loss: float
        +dimension: int
        +sparse: bool
        +squared_loss(coordinates) float
        +root_loss(coordinates) float
    }
    class NumpyPositiveDefiniteQuadraticLoss {
        +squared_loss(coordinates) float
        +root_loss(coordinates) float
    }
    class ScipySparsePositiveDefiniteQuadraticLoss {
        +matrix: csr_array
        +squared_loss(coordinates) float
        +root_loss(coordinates) float
    }

    PositiveDefiniteQuadraticLoss *-- NumpyPositiveDefiniteQuadraticLoss : sparse=False
    PositiveDefiniteQuadraticLoss *-- ScipySparsePositiveDefiniteQuadraticLoss : sparse=True
```

## Call flow

```mermaid
flowchart LR
    A[Normalized unit-free inputs] --> B{PositiveDefiniteQuadraticLoss}
    B -->|sparse=False| C[NumPy dense Cholesky]
    B -->|sparse=True| D[SciPy CSR and eigsh]
    C --> E[squared_loss]
    D --> E
    E --> F[root_loss]
```

The facade composes exactly one backend object. Backend imports are lazy, so importing
the facade does not import NumPy or SciPy and does not weaken the dependency-free
package root.
