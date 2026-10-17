# Shared-grid batched finite Fourier schematics

The diagrams show the dependency-free boundary, shared numerical work, and downstream
ownership separation.

- Public inputs contain no array backend or physics semantics.
- Empty cases return before backend execution.
- Nontrivial channels share one phase matrix and dense reduction.

## Evaluation flow

The public template validates structure before choosing an exact empty result or the
concrete backend path.

- Every nonempty channel shares the same coordinate and mode arrays.
- Backend output is checked again at the nominal boundary.

```mermaid
flowchart LR
    D[Dependency-free definition] --> V[Validate dimension, grid, modes, channels]
    V --> E[_trivial_values checks exact empty cases]
    E -->|exact values| Z[Return P x S result without backend evidence]
    E -->|nontrivial| A[Convert shared arrays]
    A --> P[Construct one P x M phase matrix]
    P --> M[One dense P x M by M x S multiplication]
    M --> S[Correlate result metadata]
    S --> R[Validate complete value surface]
    R --> B[Require paired backend evidence]
```

## Consumer composition

Numerics provides unit-free channels while downstream owners attach their own semantic
records and policies.

- PhysKit maps block entries and truncation families to coefficient channels.
- PhysKit restores model identities, matrices, units, and band records.
- KSD computes scientific error channels and interpretation.

```mermaid
flowchart LR
    H[PhysKit hopping models and explicit inventories] --> C[Unit-free coefficient channels]
    C --> N[Numerics shared-grid batched finite Fourier evaluation]
    N --> B[PhysKit Bloch matrices and band family]
    B --> K[KSD truncation evidence and scientific analysis]
```

## Scalar distinction

The scalar and batched capabilities share mathematical orientation but make different
floating-point reduction guarantees.

- Scalar evaluation exposes ordered accumulation.
- Batched evaluation exposes dense backend evidence and tolerance-qualified values.

```mermaid
flowchart TD
    Q[Finite Fourier request] --> O{Required numerical behavior}
    O -->|observable scalar reduction order| S[Scalar finite-series evaluator]
    O -->|shared-grid channel throughput| B[Batched dense evaluator]
```
