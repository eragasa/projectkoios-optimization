# Batched Hermitian-spectrum schematics

The schematics show evaluation control flow and the boundary between reusable matrix
numerics and consumer-owned semantics.

- Every diagram uses Mermaid.
- Solid arrows represent current data or control flow.
- No diagram introduces eigenvectors, physics, or scientific acceptance policy.

## Evaluation flow

The evaluation flow separates empty construction, represented-input qualification, and
one nonempty batched eigensolver operation.

- Empty definitions return before backend invocation.
- Nonempty definitions fail as a complete batch when any matrix exceeds its own cutoff.
- Accepted matrices retain per-matrix and aggregate diagnostics.

```mermaid
flowchart TD
    D[Definition] --> V[Validate canonical matrices and retain shape]
    V --> E{Batch empty?}
    E -->|yes| Z[Empty spectra and diagnostics absent]
    E -->|no| M[Measure scale and Hermiticity defect per matrix]
    M --> C{Any defect above its cutoff?}
    C -->|yes| F[Reject complete batch before backend]
    C -->|no| H[Construct represented-Hermitian matrices]
    H --> B{Selected dense backend}
    B --> N[One batched NumPy eigvalsh call]
    B --> P[One batched SciPy eigvalsh call]
    N --> S[Validate complete ascending spectra]
    P --> S
    S --> R[Correlate spectra and diagnostics]
```

## Ownership flow

The ownership flow keeps physical meaning and scientific decisions outside the numerical
spectrum capability.

- Numerics produces unit-free complete spectra and represented-input evidence.
- Consumer adapters correlate those records with physical objects.
- Consumers own units, block identity, meshes, thresholds, and interpretation.

```mermaid
flowchart LR
    N[Unit-free complete spectra and diagnostics]
    A[Consumer-owned semantic adapter]
    P[Units, block identity, mesh, thresholds, interpretation]

    N --> A --> P
```
