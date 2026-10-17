# Solver-template schematics

## Nominal SDK boundary

```mermaid
classDiagram
    class ProblemSolver {
        <<abstract>>
        +solve(inputs) ProblemResult
    }
    class NumpyProblemSolver {
        +solve(inputs) ProblemResult
    }
    class ExternalProblemSolver {
        +solve(inputs) ProblemResult
    }
    class ProblemSolverFactory {
        +create(method) ProblemSolver
    }
    class ProblemMethod {
        <<enumeration>>
    }

    ProblemSolver <|-- NumpyProblemSolver
    ProblemSolver <|-- ExternalProblemSolver
    ProblemSolverFactory --> ProblemMethod
    ProblemSolverFactory --> NumpyProblemSolver
```

## Construction and injection

```mermaid
flowchart LR
    M[Closed built-in method enum] --> F[Capability factory]
    F --> B[Built-in nominal subclass]
    E[External nominal subclass] --> H[Higher-level SDK consumer]
    B --> H
    H --> R[Backend-neutral result]
```

The factory controls built-in construction. The abstract class controls nominal SDK
substitution.

## Template-method validation

```mermaid
flowchart LR
    I[solve inputs] --> V[Shared public validation]
    V --> P[Protected backend operation]
    P --> C[Shared result validation]
    C --> R[Immutable result]
```

A capability can place shared invariant enforcement in the abstract class and delegate
only backend-specific numerical work to a protected abstract operation.
