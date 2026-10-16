# Linear algebra implementation organization

The namespace organizes source by mathematical capability and keeps optional backends
beneath the capability-neutral contract layer.

- No universal decomposition base is authorized.
- Package initializers remain import-empty.
- Concrete optional implementations carry explicit backend prefixes.

## Current status

Three linear algebra capabilities currently have owner implementations and mirrored
architecture pages.

- [`hermitian_spectral_subspace/`](hermitian_spectral_subspace/index.md) documents
  invariant-only Hermitian spectral-subspace decomposition.
- [`hermitian_spectrum/`](hermitian_spectrum/index.md) documents batched complete
  Hermitian spectra.
- [`sparse_hermitian_spectrum/`](sparse_hermitian_spectrum/index.md) documents sparse
  partial Hermitian-spectrum selection.
- [`frobenius/`](frobenius/index.md) documents dependency-free complex-matrix Frobenius
  values.
- Orthonormalization, compression, other norms, and their optional backends remain
  prospective.

## Module organization

Current and prospective modules remain capability-local so each public operation can
carry its own validation, evidence, and qualification.

- `projectkoios/numerics/linalg/__init__.py` remains import-empty.
- `hermitian_spectral_subspace.py`, `hermitian_spectrum.py`,
  `sparse_hermitian_spectrum.py`, and `frobenius.py` are current contract modules.
- `numpy/hermitian_spectral_subspace.py` and `numpy/hermitian_spectrum.py` are current
  concrete backend modules.
- `scipy/hermitian_spectrum.py` and `scipy/sparse_hermitian_spectrum.py` are current
  dense-complete and sparse-partial SciPy backend modules.
- `orthonormalization.py`, `compression.py`, and matching backend modules remain planned.
- `numpy/__init__.py` remains import-empty.

## Record boundary

Dependency-free modules define immutable mathematical records, while backend modules own
third-party representation conversion and narrow numerical calls.

- Contract modules own definitions, tolerances, diagnostics, results, and nominal solver
  templates.
- Backend modules own optional-array conversion and concrete decomposition calls.
- Final template methods retain validation, represented matrix construction, relational
  checks, failure classification, and backend-neutral result construction.
- Backend records do not acquire units, domain identifiers, or scientific policy.

## Invariant-before-frame rule

Spectral capabilities expose invariant information before any basis-dependent frame
selection.

- Eigenvalues use ascending order with multiplicity retained.
- Spectral selection is caller-supplied rather than domain-inferred.
- Internal exact degeneracies are admitted only when the complete block is selected.
- A selected boundary that cuts an exact cluster or fails declared isolation fails
  closed.
- Subspace results expose basis-independent projectors before basis-dependent frames.
- No backend eigenvector is represented as canonical.
- Compression, embedding, projection, frame completion, and transformation selection
  remain separate operations.

## Verification organization

Each accepted capability receives mirrored contract, backend, bounded numerical, and
wheel verification appropriate to its mathematical output.

- Spectral-subspace verification covers strict inputs, analytic and high-precision
  references, projector identities, degeneracy boundaries, invariances, malicious
  backend outputs, failure translation, lazy imports, and reference auditing.
- Batched complete-spectrum verification covers normalized Python and NumPy inputs,
  inferred nonempty shape, explicit empty construction, whole-batch failure,
  representative evidence, absent empty aggregates, NumPy/SciPy batched calls, and
  high-precision spectra.
- Sparse partial-spectrum verification covers canonical CSR storage, explicit selection
  and convergence controls, sparse Hermiticity evidence, SciPy ARPACK behavior,
  deterministic start vectors, and residual bounds.
- Future capabilities require independent verification contracts and do not inherit
  acceptance from either spectral slice.

## Ownership constraint

Linear algebra source depends only on the standard library and its selected optional
backend, while downstream layers retain their own semantics.

- Numerics must not import Optimization, PhysKit, or KSD.
- Optimization may consume numerical records for objective construction and
  transformation selection.
- PhysKit and KSD retain units, model identities, thresholds, evidence, and scientific
  interpretation.
