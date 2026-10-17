# Batched nominal solver schematics

```mermaid
flowchart LR
    G[Generic shared policy] --> B[Batched nominal policy]
    B --> N[Concrete NumPy solver]
    N --> R[Exact-array batch result]
```
