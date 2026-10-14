# Constrained-quadratic schematics

## Solver construction

```mermaid
flowchart LR
    M[Closed method enum] --> F[Capability factory]
    F -->|NUMPY_DENSE| N[NumPy dense solver]
    F -->|SCIPY_SPARSE| S[SciPy sparse solver]
    N --> V[Final shared result validation]
    S --> V
    V --> R[Backend-neutral result]
```

## Rectangle minimization

```mermaid
flowchart TD
    Q[2-D positive-definite quadratic] --> C[Clipped unconstrained center]
    B[Closed rectangle] --> C
    B --> E[Four analytically minimized edges]
    Q --> E
    C --> S[Ordered candidate selection]
    E --> S
    S --> M[RectangleQuadraticMinimum]
```

Candidate order is the clipped center, coordinate-0 lower edge, coordinate-0 upper edge,
coordinate-1 lower edge, and coordinate-1 upper edge. Exact ties retain the earliest
candidate.

## Ellipsoid support

```mermaid
flowchart LR
    Q[Positive-definite quadratic] --> B[Squared-loss ceiling]
    D[Nonzero direction d] --> S[Solve Q y = d]
    B --> P[Support-point formula]
    S --> P
    P --> R[QuadraticSublevelSupportResult]
```

The operation optimizes the directional linear functional over the full quadratic
sublevel ellipsoid. Box containment remains a consumer check.

## Finite family

```mermaid
flowchart LR
    F[Ordered fixed-angle quadratic family] --> E[Evaluate values or constrained minima]
    E --> M[first_minimum]
    E --> C[indices_at_or_below]
    M --> R[Selected minimum index]
    C --> I[Ordered feasible indices]
    D[Consumer-supplied per-angle distances] --> G[first_minimum]
    G --> H[Global finite-union distance index]
```

The nonconvexity is only a finite union. Complete ordered enumeration reduces the global
comparison to the minimum of per-member values. This capability does not calculate a
general distance between box-clipped sublevel sets.
