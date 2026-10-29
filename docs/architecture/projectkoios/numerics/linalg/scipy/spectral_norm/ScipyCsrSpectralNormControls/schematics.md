# `ScipyCsrSpectralNormControls` schematics

```mermaid
classDiagram
    class ScipyCsrSpectralNormControls {
        +float tolerance
        +int maximum_iterations
        +int_or_None krylov_dimension
        +int seed
    }
    class ScipyCsrSpectralNorm {
        -ScipyCsrSpectralNormControls _controls
        +result() float
    }

    ScipyCsrSpectralNorm *-- ScipyCsrSpectralNormControls : exact mandatory instance
```

```mermaid
flowchart LR
    C[Explicit control values]
    T[Exact type checks]
    V[Shape-independent value checks]
    R[Frozen controls record]
    E[Evaluator shape and dtype checks]
    D[Direct branch or ARPACK arguments]

    C --> T --> V --> R --> E --> D
```

No evaluator or consumer synthesizes, replaces, or silently ignores the controls record.
The documented direct branch does not consume iterative fields numerically.
