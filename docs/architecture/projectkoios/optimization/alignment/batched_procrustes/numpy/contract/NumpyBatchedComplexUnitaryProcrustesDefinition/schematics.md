# Definition schematics

```mermaid
flowchart LR
    Ordinary[Ordinary exact arrays] --> Copy[from_arrays]
    Copy --> Defensive[Immutable defensive copies]
    Canonical[Canonical immutable arrays] --> Zero[from_immutable_arrays]
    Defensive --> Raw[Canonical frozen constructor]
    Zero --> Raw
    Raw --> Validate[Shape and relation validation]
    Validate --> Definition[Immutable definition]
    Definition --> Solver[Solver]
```

Consumer semantics remain outside the record.
