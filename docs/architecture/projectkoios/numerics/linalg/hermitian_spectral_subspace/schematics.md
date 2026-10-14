# Hermitian spectral-subspace schematics

## Status

These diagrams describe the proposed invariant-only capability. They do not authorize
source implementation or a frame-producing eigensystem.

## Shared validation and backend flow

```mermaid
flowchart TD
    D["HermitianSpectralSubspaceDefinition<br/>matrix, [start, stop), tolerances"]
    H["shared input-Hermiticity measurement"]
    A{"defect <= applied cutoff?"}
    S["shared representative<br/>(A + A†) / 2"]
    B["nominal backend<br/>complete spectrum + transient complete frame"]
    E["shared eigenpair, orthonormality,<br/>and reconstruction validation"]
    G["shared boundary-gap calculation"]
    I{"every present gap > policy cutoff?"}
    P["shared selected-column<br/>projector construction"]
    V["shared projector and spectral relation validation"]
    R["HermitianSpectralSubspaceResult<br/>frame discarded"]
    X["fail closed"]

    D --> H --> A
    A -- no --> X
    A -- yes --> S --> B --> E
    E --> G --> I
    I -- no --> X
    I -- yes --> P --> V --> R
```

The unsymmetrized input is measured before the representative is constructed. The
backend never chooses which input triangle represents the matrix.

## Selected range and optional boundaries

```mermaid
flowchart LR
    L0["λ0 ... λ(a-1)"]
    LB{"lower gap<br/>present only if a > 0"}
    SEL["selected indices [a, b)<br/>projector P"]
    UB{"upper gap<br/>present only if b < n"}
    HI["λb ... λ(n-1)"]

    L0 --> LB --> SEL --> UB --> HI
```

A present boundary passes only when its gap is strictly larger than the applied declared
cutoff. This policy is not itself a backward-error certificate. The lowest group has no
lower boundary. The complete range has neither boundary.

## Internal degeneracy is not a boundary defect

```mermaid
flowchart TD
    E["ordered eigenvalues with multiplicity"] --> Q{"where is a repeated or near cluster?"}
    Q -- "entirely inside [a, b)" --> P["admit complete block<br/>return summed projector"]
    Q -- "crosses a or b" --> F["reject boundary under declared policy"]
    P --> INV["invariant under internal unitary rotation"]
```

The operation does not split a boundary-degenerate eigenspace and does not reject an
internal degeneracy merely because individual eigenvectors are nonunique.

## Frame nonuniqueness gate

```mermaid
flowchart LR
    P["selected projector P"]
    F1["frame F<br/>F†F = I, FF† = P"]
    U["arbitrary selected-space unitary U"]
    F2["equally valid frame FU"]
    OUT["public invariant result"]
    BLOCK["frame-producing contract<br/>separate review required"]

    P --> F1 --> U --> F2
    P --> OUT
    F1 -. not returned .-> BLOCK
    F2 -. not returned .-> BLOCK
```

No phase rule, eigenvector ordering within an exact degenerate block, or canonical frame
is implied by the projector result.

## Relational validation

```mermaid
flowchart TD
    H["Hermitian representative H"]
    F["protected complete frame V"]
    P["shared-constructed projector P"]
    E["returned eigenvalues λ"]
    C0["V†V ≈ I<br/>HV ≈ V diag(λ)<br/>H ≈ V diag(λ)V†"]
    C1["P ≈ P†"]
    C2["P² ≈ P"]
    C3["tr(P) ≈ selected rank"]
    C4["HP ≈ PH"]
    C5["tr(HP) ≈ sum selected λ"]
    C6["tr(H) ≈ sum all λ"]

    H --> C0
    F --> C0
    E --> C0
    F --> P
    P --> C1
    P --> C2
    P --> C3
    H --> C4
    P --> C4
    H --> C5
    P --> C5
    E --> C5
    H --> C6
    E --> C6
```

The shared template validates the complete protected eigensystem before it constructs
$P$ from the requested ordered columns. This rejects trace-preserving wrong spectra and
wrong-subspace counterexamples that pass projector-only identities. A backend supplies
no self-authored quality diagnostics.

## Nominal SDK and lazy factory

```mermaid
classDiagram
    class HermitianSpectralSubspaceSolver {
        <<abstract>>
        +solve(definition) HermitianSpectralSubspaceResult
        #_backend_identifier str
        #_backend_version str
        #_eigendecompose_validated(representative) BackendOutput
    }
    class NumpyHermitianSpectralSubspaceSolver {
        #_eigendecompose_validated(representative) BackendOutput
    }
    class ExternalHermitianSpectralSubspaceSolver {
        #_eigendecompose_validated(representative) BackendOutput
    }
    class HermitianSpectralSubspaceSolverFactory {
        +create(method) HermitianSpectralSubspaceSolver
    }
    class HermitianSpectralSubspaceMethod {
        <<enumeration>>
        NUMPY_DENSE
    }

    HermitianSpectralSubspaceSolver <|-- NumpyHermitianSpectralSubspaceSolver
    HermitianSpectralSubspaceSolver <|-- ExternalHermitianSpectralSubspaceSolver
    HermitianSpectralSubspaceSolverFactory --> HermitianSpectralSubspaceMethod
    HermitianSpectralSubspaceSolverFactory --> NumpyHermitianSpectralSubspaceSolver
```

The factory constructs only reviewed built-ins. External implementations use nominal
inheritance and direct injection.

## Ownership and dependency graph

```mermaid
flowchart TB
    NUM["projectkoios.numerics<br/>Hermitian representative, spectrum, projector"]
    OPT["projectkoios.optimization.alignment<br/>future transformation selection"]
    PHY["projectkoios.physkit<br/>units, operators, basis and lattice identity"]
    KSD["ksdft2effmass<br/>retained rank, sample role, gap meaning, evidence"]

    NUM --> OPT
    NUM --> PHY
    NUM --> KSD
    OPT --> KSD
    PHY --> KSD
```

Numerics imports none of the downstream packages. PhysKit is not a mandatory facade;
KSD may consume the unit-free contract directly when it owns the mapping.

## KSD lowest-group composition

```mermaid
flowchart LR
    KH["KSD energy-valued Hermitian operator"]
    KM["KSD converts magnitudes in one declared unit"]
    ND["Numerics definition<br/>range [0, retained_rank)"]
    NR["all eigenvalues + retained projector"]
    KR["KSD restores units and identities"]
    KE["KSD external-gap and scientific evidence"]

    KH --> KM --> ND --> NR --> KR --> KE
```

The numerical boundary cutoff and KSD's scientific external-gap lower bound remain
separate controls. Invariant output does not replace M2's still-local eigenframe path.

## Reference and documentation audit

```mermaid
flowchart LR
    REF["references.md<br/>DOI or ISBN on every entry"]
    OFF["offline identifier and checksum audit"]
    ON["Crossref/Open Library metadata inspection"]
    READ["human reading of abstracts, contents, publisher descriptions"]
    DOC["bounded architecture claims"]

    REF --> OFF
    REF --> ON
    ON --> READ
    READ --> DOC
```

Metadata retrieval helps detect mistaken citations. It does not substitute for reading
the source material or for defining explicit software behavior.
