# Batched Procrustes schematics

The diagrams distinguish vectorized numerical work, item-level uniqueness status, and
downstream interpretation.

- One phase of array validation precedes backend execution.
- One public SVD handles every nonempty batch item.
- Every item retains its own rank and residual evidence.

## Vectorized solve

The solver preserves all leading dimensions through cross construction, decomposition,
and result assembly.

- Full-rank and deficient items share one homogeneous array result.
- Deficient completions remain explicitly nonunique diagnostics.

```mermaid
flowchart LR
    A[Source batch ... x m x n] --> C[Cross batch A-dagger B]
    B[Target batch ... x m x n] --> C
    C --> S[One batched NumPy SVD]
    S --> Q[Right-unitary completions U V-dagger]
    S --> E[Singular, rank, cutoff, residual evidence]
    E --> F{Full rank per item?}
    F -->|yes| U[Unique Procrustes transformation]
    F -->|no| D[Diagnostic nonunique completion]
    Q --> R[Read-only shape-preserving result]
    U --> R
    D --> R
```

## Failure boundary

The batch has an all-or-nothing failure policy for invalid arithmetic while rank
deficiency is ordinary per-item evidence.

- Validation reports the first failing leading index in C order.
- No partial result escapes a failed solve.

```mermaid
flowchart TD
    I[Validated finite complex128 arrays] --> N{Empty batch?}
    N -->|yes| Z[Exact empty result without backend evidence]
    N -->|no| V[Vectorized backend and evidence validation]
    V --> P{Programming or numerical failure?}
    P -->|yes| X[Reject whole batch with indexed condition]
    P -->|no| K[Return all items including rank status]
```

## Consumer boundary

Optimization supplies selected completions and numerical evidence without deciding their
scientific use.

- Consumers may admit only full-rank transformations.
- Consumers may retain deficient completions for diagnostics only.

```mermaid
flowchart LR
    O[Optimization batched Procrustes result] --> T[Consumer tensor-grid links]
    T --> G[Consumer rank qualification gate]
    G --> C[Consumer closure and scientific diagnostics]
```
