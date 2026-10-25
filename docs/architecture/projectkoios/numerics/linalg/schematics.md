# Linear algebra schematics

The diagrams summarize current invariant-first matrix capabilities and preserve explicit
boundaries around prospective or consumer-owned behavior.

- Every diagram uses Mermaid.
- Current paths use solid arrows.
- Prospective behavior is labeled explicitly and receives no authorization from a
  diagram.

## Invariant-first spectral flow

The spectral-subspace capability returns an invariant projector without choosing a basis
inside the selected subspace.

- The complete spectrum precedes caller-supplied index selection.
- Boundary qualification precedes projector construction.
- Frame selection remains a separate blocked capability.

```mermaid
flowchart LR
    A[Finite complex matrix]
    H[Validated Hermitian representative]
    E[Ascending complete spectrum]
    I[Caller-supplied index range and boundary policy]
    P[Orthogonal spectral projector]
    F[Frame selection: prospective]

    A --> H --> E --> I --> P
    P -. separate future policy .-> F
```

## Batched complete-spectrum flow

The batched capability infers nonempty shape, measures each input independently, and
executes one batched eigensolver operation.

- Nonempty batches infer dimension from their matrices.
- Empty batches use a named constructor and return before backend invocation.
- Empty aggregate diagnostics remain absent rather than using sentinel maxima.

```mermaid
flowchart LR
    B[Nonempty ordered matrix batch]
    D[Inferred common dimension]
    Q[Per-matrix Hermiticity diagnostics]
    H[Represented-Hermitian matrices]
    E[One batched eigvalsh operation]
    S[Complete ascending spectra]

    B --> D --> Q --> H --> E --> S
```

## Sparse partial-spectrum flow

The sparse capability preserves canonical CSR storage and makes subset selection and
iterative controls explicit.

- Sparse evaluation returns a strict subset rather than claiming a complete spectrum.
- Backend-local eigenvectors support residual evidence without becoming public outputs.
- Consumers decide whether the selected subset is sufficient.

```mermaid
flowchart LR
    C[Canonical CSR matrix]
    H[Sparse Hermiticity qualification]
    R[Sparse represented-Hermitian matrix]
    E[SciPy sparse eigsh]
    V[Selected eigenvalues and backend-local vectors]
    D[Maximum eigenpair residual]
    O[Partial-spectrum result without public vectors]

    C --> H --> R --> E --> V --> D --> O
```

## Linear-algebra composition

The composition view distinguishes implemented reductions from independently reviewed
prospective transformations.

- Frobenius power and norm are current.
- The matrix-norm semantic enum and nominal callable boundary are current; concrete
  backend evaluators remain prospective.
- Projector complement, compression, orthonormalization, and common-space injection are
  prospective.
- Prospective nodes do not inherit acceptance from implemented capabilities.

```mermaid
flowchart LR
    P[Spectral projector] --> C[Operator compression: prospective]
    P --> K[Projector complement: prospective]
    G[Sampled Gram matrix] --> O[Symmetric orthonormalization: prospective]
    O --> J[Common-space injection: prospective]
    N[Finite complex matrix] --> M[Implemented Frobenius power and norm]
    E[MatrixNormEnum] --> A[MatrixNormABC callable boundary]
    A -. reviewed backend required .-> B[Concrete matrix-norm evaluator: prospective]
```

## Optimization boundary

The ownership boundary distinguishes numerical matrix records from objective-driven
selection of transformations.

- Numerics owns decompositions, spectra, projectors, and backend-neutral evidence.
- Optimization owns objectives and selected optimizer arguments.
- Optimized transport belongs under `projectkoios.optimization.alignment`.

```mermaid
flowchart LR
    N[Numerics decomposition]
    R[Backend-neutral numerical record]
    O[Optimization objective]
    T[Selected transformation]

    N --> R --> O --> T
```

## Consumer semantics

The consumer boundary prevents use by a physics or KSD package from transferring domain
meaning into Numerics.

- Consumers own operators, units, basis identities, and sample roles.
- Numerics owns only unit-free matrix operations and numerical evidence.
- Scientific gap and interpretation policies remain consumer-owned.

```mermaid
flowchart LR
    C[Consumer operator, unit, basis, and role]
    A[Unit-free adapter]
    N[Numerics spectral capability]
    W[Consumer-correlated result]
    S[Scientific gap and interpretation]

    C --> A --> N --> W --> S
```
