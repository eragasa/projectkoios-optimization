# Numerics implementation organization

## Target source layout

```text
projectkoios/numerics/
├── validation/
├── geometry/
├── linalg/
├── sequences/
├── finite_periodic/
├── numerical_analysis/
└── statistics/
```

Capability packages contain dependency-free records and backend subpackages. The source
tree is created incrementally; no placeholder source package is added before its first
accepted behavior.

## Backend rule

```text
<capability>/<module>.py
<capability>/numpy/<module>.py
<capability>/scipy/<module>.py
```

The capability module defines immutable records and, when multiple implementations are
supported, a capability-local nominal abstract class. Backend modules define concrete
subclasses. Capability factories select supported subclasses through closed method
enums. Structural protocols are not used. Public result records remain backend-neutral.

## Implemented namespace foundation

The first implemented namespace slice provides:

- import-empty `projectkoios.numerics` and `finite_periodic` packages;
- dependency-free finite Fourier-series records and nominal evaluator;
- a lazy ordered NumPy finite-series backend;
- strict typing and `projectkoios/numerics/py.typed`; and
- isolated dependency-free and NumPy wheel imports.

The next local slice adds invariant-only dense Hermitian spectral-subspace decomposition
under `linalg`. Its source and qualification were separately authorized and
independent review closed without blocking findings; no commit or downstream binding is
implied.

## Cross-namespace use

Optimization consumes Numerics through exact module imports. Numerics does not import
Optimization. Composite algorithms that optimize an internal objective belong in
Optimization rather than introducing reverse dependencies.

## Status

The namespace organization is an accepted incremental target. Only capabilities whose
pages and exact owner revisions identify implemented source are current APIs; all others
remain prospective.
