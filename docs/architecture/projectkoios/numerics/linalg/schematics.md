# Linear algebra schematics

## Invariant-first spectral flow

```mermaid
flowchart LR
    A["finite complex matrix"]
    H["validated Hermitian representative"]
    E["ascending complete spectrum"]
    I["caller-supplied index range<br/>declared boundary policy"]
    P["orthogonal spectral projector"]
    F["frame selection<br/>separate blocked capability"]

    A --> H --> E --> I --> P
    P -. future explicit policy .-> F
```

The implemented contract, when authorized, stops at the projector. It does not hide
phase or degenerate-basis choices inside an eigensystem result.

## Batched complete-spectrum flow

```mermaid
flowchart LR
    B["nonempty ordered matrix batch"]
    D["inferred common dimension"]
    Q["per-matrix Hermiticity diagnostics"]
    H["represented-Hermitian matrices"]
    E["one batched NumPy eigvalsh operation"]
    S["complete ascending spectra"]

    B --> D --> Q --> H --> E --> S
```

An empty batch uses an explicit constructor and returns before the numerical backend,
with aggregate diagnostics absent rather than represented by sentinel maxima.

## Linear-algebra composition

```mermaid
flowchart LR
    P["spectral projector"] --> C["operator compression"]
    P --> K["projector complement"]
    G["sampled Gram matrix"] --> O["symmetric orthonormalization"]
    O --> J["common-space injection"]
    N["finite complex matrix"] --> M["implemented Frobenius power and norm"]
```

The Frobenius path is current. The projector-complement, compression,
orthonormalization, and injection boxes remain separate prospective capabilities; this
diagram does not authorize them.

## Optimization boundary

```mermaid
flowchart LR
    N["Numerics decomposition<br/>spectrum and projector"]
    R["backend-neutral numerical record"]
    O["Optimization objective"]
    T["selected transformation"]

    N --> R --> O --> T
```

The left side belongs to Numerics. The objective and selected optimizer argument belong
to Optimization.

## Consumer semantics

```mermaid
flowchart LR
    C["consumer operator, unit, basis, role"]
    A["unit-free adapter"]
    N["Numerics spectral subspace"]
    W["consumer-correlated result"]
    S["scientific gap and interpretation"]

    C --> A --> N --> W --> S
```

Numerics does not acquire physical units, band meaning, sample roles, or evidentiary
policy merely because a physics or KSD consumer uses the projector.
