# Linear algebra implementation organization

## Current status

The package contains the invariant-only Hermitian spectral-subspace solver documented
under [`hermitian_spectral_subspace/`](hermitian_spectral_subspace/index.md) and the
dependency-free complex-matrix Frobenius reduction documented under
[`frobenius/`](frobenius/index.md). Source continues to be added capability by
capability; no universal decomposition base is authorized.

## Target modules

```text
projectkoios/numerics/linalg/
├── __init__.py
├── hermitian_spectral_subspace.py
├── frobenius.py
├── orthonormalization.py
├── compression.py
└── numpy/
    ├── __init__.py
    ├── hermitian_spectral_subspace.py
    ├── orthonormalization.py
    └── compression.py
```

The Hermitian spectral-subspace and standard-library-only Frobenius paths are current.
Orthonormalization, compression, other norms, and their optional backends remain
prospective.

## Record boundary

Dependency-free capability modules define immutable tuple-oriented definitions,
tolerances, diagnostics, and results. Dense backend modules own optional-array conversion
and the narrow third-party decomposition call. Shared final template methods retain input
validation, declared matrix representation, relational checks, failure classification,
and backend-neutral output construction.

Package initializers remain import-empty. Concrete classes carry explicit backend prefixes.

## Invariant-before-frame rule

- Eigenvalues use ascending order with multiplicity retained.
- Spectral selection is caller-supplied rather than domain-inferred.
- Internal exact degeneracies are admitted only when the complete block is selected.
- A selected boundary that cuts an exact cluster or fails the declared isolation policy
  fails closed.
- Subspace results expose basis-independent projectors before any basis-dependent frame.
- No backend eigenvector is represented as canonical.
- Compression, embedding, projection, frame completion, and transformation selection
  remain distinct operations.

## Verification organization

Each accepted capability receives mirrored core and backend tests. The first spectral
slice requires:

- exact built-in input and tolerance validation;
- analytic and high-precision spectrum/projector references;
- projector Hermiticity, idempotence, trace, and commutator checks;
- internal-degeneracy and boundary-cutting cases;
- phase, internal-unitary, shift, scaling, and coordinate-permutation cases;
- malicious nominal backend outputs;
- precise backend exception translation;
- lazy optional imports and isolated wheel smoke; and
- DOI/ISBN reference auditing plus human source inspection.

Future orthonormalization, compression, and norm modules require their own verification
contracts rather than inheriting acceptance from the spectral slice.

## Ownership constraint

Linear algebra source may depend on the Python standard library and its selected optional
backend. It must not import Optimization, PhysKit, or KSD. Optimization may consume these
records for objectives and transformation selection; downstream consumers retain units,
model identities, thresholds, evidence, and scientific interpretation.
