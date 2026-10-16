# Sparse partial-spectrum schematics

The diagrams show sparse qualification, partial selection, and the boundary between
backend-local eigenvectors and public invariant evidence.

- Every diagram uses Mermaid.
- Sparse storage remains explicit through backend invocation.
- Public results omit eigenvectors.

## Evaluation flow

The evaluation flow qualifies canonical CSR input before invoking one SciPy ARPACK
problem with explicit controls.

- Hermiticity failure occurs before backend execution.
- The backend computes only the requested strict subset.
- Backend-local vectors produce residual evidence and are then discarded.

```mermaid
flowchart TD
    C[Canonical owner CSR matrix]
    H[Measure sparse Hermiticity defect]
    Q{Defect above cutoff?}
    F[Reject before backend]
    R[Construct sparse represented-Hermitian matrix]
    A[Adapt to SciPy CSR array]
    E[Run scipy sparse eigsh with explicit controls]
    V[Backend-local eigenvalues and eigenvectors]
    D[Compute maximum eigenpair residual]
    O[Return ascending selected eigenvalues and evidence]

    C --> H --> Q
    Q -->|yes| F
    Q -->|no| R --> A --> E --> V --> D --> O
```

## Capability boundary

The capability boundary keeps complete dense spectra and partial sparse spectra as
separate mathematical operations.

- Dense complete evaluation returns every eigenvalue.
- Sparse partial evaluation returns a caller-selected strict subset.
- Consumers choose the operation required by their scientific or physical model.

```mermaid
flowchart LR
    M[Hermitian matrix problem]
    C[Dense complete-spectrum contract]
    P[Sparse partial-spectrum contract]
    A[Consumer-owned sufficiency decision]

    M --> C --> A
    M --> P --> A
```
