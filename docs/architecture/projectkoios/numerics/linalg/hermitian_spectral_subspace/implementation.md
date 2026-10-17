# Hermitian spectral-subspace implementation organization

## Status and inspected foundations

This page records the current local implementation organization. It is informed by the
[inspected academic references](references.md), especially the invariant-subspace and
gap-dependent perturbation treatment of Kato, Davis–Kahan, and Stewart–Sun and the dense
Hermitian eigensystem/error treatment of Golub–Van Loan and the LAPACK guide.

The source and mapped tests described here now exist locally under explicit
implementation authorization. They are not committed, pushed, bound downstream,
released, or published.

## Source layout

```text
src/python/projectkoios/numerics/linalg/
├── __init__.py
├── hermitian_spectral_subspace.py
└── numpy/
    ├── __init__.py
    └── hermitian_spectral_subspace.py
```

The dependency-free module owns the public matrix alias, exact records, shared scalar
validation and relational checks, shared projector construction, the nominal solver, the
closed method enum, and the lazy factory. The NumPy module owns only dense conversion,
`numpy.linalg.eigh`, finite-output checks, built-in tuple conversion, and exact backend
failure translation.

Package initializers remain import-empty. Importing the core module must not import NumPy.
The module name is capability-specific and must not become a collection of unrelated
linear-algebra helpers.

## Dependency-free public vocabulary

### `HermitianSpectralMatrix`

A public representation alias:

```python
type HermitianSpectralMatrix = tuple[tuple[complex, ...], ...]
```

The alias alone does not prove Hermiticity. Its exact structural and finite-state
invariants are enforced by the owning definition, result, and final solver boundaries.

### `HermitianSpectralSubspaceTolerance`

A final frozen slotted record with the seven exact fields specified in
[numerics.md](numerics.md). Constructor validation rejects booleans, NumPy scalars,
nonfinite values, negative values, and relative tolerances outside $[0,1)$. It
validates each canonical field explicitly and performs no field replacement.

Its documentation states that the boundary cutoff is a ProjectKoios isolation policy,
not a backward-error theorem or scientific external-gap criterion.

### `HermitianSpectralSubspaceDefinition`

A final frozen slotted record containing the exact tuple matrix, half-open selected
range, and exact tolerance record. Construction validates all dependency-free structural
invariants and exposes derived `dimension` and `selected_rank` properties. It validates
canonical storage directly and does not replace fields after construction.

The constructor performs no symmetrization or eigendecomposition. Retaining the original
matrix lets the final solver report its pre-symmetrization Hermiticity defect.

### `HermitianSpectralSubspaceDiagnostics`

A final frozen slotted record with these exact `float` fields:

```python
input_hermiticity_maximum_absolute_defect: float
representative_adjustment_maximum_absolute_defect: float
eigenvector_orthonormality_maximum_absolute_defect: float
eigensystem_residual_maximum_absolute_defect: float
representative_reconstruction_maximum_absolute_defect: float
projector_hermiticity_maximum_absolute_defect: float
projector_idempotence_maximum_absolute_defect: float
projector_trace_absolute_defect: float
projector_commutator_maximum_absolute_defect: float
selected_spectral_trace_absolute_defect: float
full_spectral_trace_absolute_defect: float
input_matrix_maximum_absolute_entry: float
representative_matrix_maximum_absolute_entry: float
applied_hermiticity_cutoff: float
spectral_diameter: float
applied_boundary_cutoff: float
applied_projector_validation_cutoff: float
applied_spectral_validation_cutoff: float
```

Every field is an exact finite nonnegative built-in float. Its sole constructor validates
each named field explicitly, without reflective attribute access or postconstruction
field replacement. The formulas and correlation rules are fixed in
[numerics.md](numerics.md). Consumers may retain this evidence but
must not reinterpret it as physical or scientific acceptance without an owning policy.

### `HermitianSpectralSubspaceResult`

A final frozen slotted record containing the correlated dimension, exact selected range,
tolerance, complete ascending spectrum, selected projector, optional boundary gaps,
shared diagnostics, and trusted API-level backend metadata.

The result constructor validates local field shape and finite-state invariants without
replacing canonical fields after construction. The final solver performs the stronger
correlation checks against the definition, representative,
complete protected eigensystem, and shared-constructed projector.

## Protected backend eigensystem

Concrete subclasses do not construct the public projector, diagnostics, or result. Their
protected operation returns a capability-private exact tuple equivalent to

```python
type _ComplexSquareMatrix = tuple[tuple[complex, ...], ...]
type _HermitianSpectralSubspaceBackendOutput = tuple[
    tuple[float, ...],
    _ComplexSquareMatrix,
]
```

The two elements are the complete ascending eigenvalue tuple and its corresponding
complete eigenvector matrix. The final template requires exact built-in nested/scalar
types and validates count, shape, finiteness, ascending order, orthonormality, complete
eigenpair residual, and reconstruction.

The protected eigenvectors are transient validation evidence. They never enter a public
record and establish no public phase or degenerate-basis convention. After validation,
the shared template uses the ordered selected columns to construct the projector and
then discards the frame.

## Nominal solver

`HermitianSpectralSubspaceSolver` is a capability-local abstract class:

```python
from abc import ABC, abstractmethod
from typing import final


class HermitianSpectralSubspaceSolver(ABC):
    @property
    @abstractmethod
    def _backend_identifier(self) -> str: ...

    @property
    @abstractmethod
    def _backend_version(self) -> str: ...

    @final
    def solve(
        self,
        definition: HermitianSpectralSubspaceDefinition,
        /,
    ) -> HermitianSpectralSubspaceResult: ...

    @abstractmethod
    def _eigendecompose_validated(
        self,
        representative: HermitianSpectralMatrix,
        /,
    ) -> _HermitianSpectralSubspaceBackendOutput:
        raise NotImplementedError
```

The shared `solve()` method follows this fixed sequence:

1. require the exact nominal definition type;
2. calculate original matrix scale and Hermiticity defect;
3. apply the inclusive input-Hermiticity cutoff;
4. construct and verify the explicit Hermitian representative;
5. call `_eigendecompose_validated()` exactly once;
6. validate the protected output's exact structure and finite built-in values;
7. validate spectrum length, eigenvector shape, and ascending order;
8. independently validate eigenvector orthonormality, complete eigenpair residual, and
   reconstruction;
9. calculate the spectral diameter and declared boundary cutoff;
10. calculate optional boundary gaps and enforce strict isolation;
11. construct the selected projector from the validated ordered eigenvectors;
12. independently calculate projector and spectral-correlation diagnostics;
13. apply the declared projector and spectral validation policies;
14. validate exact nonempty backend metadata strings; and
15. construct the public result from shared values.

A backend receives only a nonempty, finite, square, constructed-Hermitian representative.
It does not receive the selection range, so it cannot return a self-attested selected
projector. Complete eigenpair validation binds the ordered columns to the corresponding
spectrum; shared column selection binds the public projector to the requested range.

Built-in implementations are final nominal subclasses. External implementations opt in
through nominal inheritance and must pass the owner conformance suite. Structural
protocols, duck-typed substitution, entry-point discovery, and mutable factory
registration are forbidden.

## Backend metadata trust boundary

`_backend_identifier` and `_backend_version` are trusted declarations by the nominal
backend. The final template validates only exact nonempty string structure; it cannot
authenticate arbitrary external labels.

The conformance suite checks the exact identity and version only for the final built-in
NumPy class. An external class may provide its own truthful values, but a result does not
prove that declaration cryptographically. For NumPy, the two values identify the API and
NumPy release, not the linked BLAS/LAPACK implementation.

## Shared scalar arithmetic

The core module remains dependency-free. It implements narrow private operations for:

- maximum absolute entry;
- conjugate transpose and explicit symmetrization;
- deterministic complex matrix multiplication;
- multiplication by a diagonal spectrum;
- selected-column projector construction;
- trace;
- Hermiticity and maximum-entry difference defects; and
- finite scale-aware relation validation.

These operations remain local until another reviewed contract proves an identical
reusable primitive. They must not enter a generic `utils.py`, `helpers.py`, or
`common.py` module.

The implementation uses stable real/imaginary summation where practical and checks
finiteness after multiplication, addition, trace, norm, scale, and cutoff operations. It
does not clamp a negative gap, replace a nonfinite diagnostic, or change a strict
comparison into an inclusive one.

The shared $O(n^3)$ eigensystem, projector, and commutator validation is intentional for
the initial dense capability. Dense eigendecomposition is already $O(n^3)$, and the
final template must not trust a nominal extension's spectrum-to-subspace association. A
future large-scale iterative capability requires a distinct verification and complexity
contract.

## NumPy implementation

`NumpyHermitianSpectralSubspaceSolver` is a final subclass. Its module imports NumPy and
no consumer package. The protected operation:

1. converts the representative to a C-order `numpy.complex128` array;
2. executes `numpy.linalg.eigh(representative, UPLO="L")`;
3. verifies every eigenvalue and eigenvector component is finite;
4. converts all eigenvalues to built-in floats;
5. converts the complete transient eigenvector matrix to built-in complex tuples; and
6. returns only those two values.

Its metadata properties return `numpy.linalg.eigh` and the exact runtime NumPy version.

The backend must not:

- receive or inspect the original approximately Hermitian input;
- choose another Hermitian triangle or symmetrization rule;
- expose eigenvectors in the public result;
- construct or self-attest to the selected projector;
- fix phases or degenerate-block bases;
- infer a cluster or selected rank;
- use a general nonsymmetric eigensolver;
- sort or perturb eigenvalues after `eigh`; or
- claim bitwise reproducibility or complete linked-library provenance.

The implementation translates exactly `numpy.linalg.LinAlgError` to the reviewed
`RuntimeError` with chaining. NumPy `FloatingPointError` and explicit nonfinite-output
detection produce `FloatingPointError`. Every other undeclared exception propagates.

## Method enum and lazy factory

`HermitianSpectralSubspaceMethod` is a closed `StrEnum` with one initial member:

```python
NUMPY_DENSE = "numpy-dense"
```

`HermitianSpectralSubspaceSolverFactory.create()` accepts only the exact enum type,
lazily imports `NumpyHermitianSpectralSubspaceSolver`, and returns the nominal solver
type. It accepts no consumer default, tolerance, or scientific label.

External subclasses are injected directly rather than registered with the factory.

## Deferred capabilities

Separate architecture is required for:

- public eigenvectors or deterministic frames;
- generalized Hermitian eigenproblems;
- real-symmetric-specialized records;
- sparse Lanczos or other iterative partial-spectrum methods;
- interval or directed-rounding eigenvalue enclosures;
- batched matrices;
- derivatives of eigenvalues or projectors;
- non-Hermitian Schur or eigensystem calculations; and
- GPU or distributed implementations.

The inspected Golub–Van Loan reference treats large sparse eigenvalue methods separately
from dense symmetric eigensystems, supporting this split.

## Test mapping

```text
tests/projectkoios/numerics/linalg/
├── hermitian_spectral_subspace/
│   ├── test__HermitianSpectralSubspaceTolerance__unit.py
│   ├── test__HermitianSpectralSubspaceDefinition__unit.py
│   ├── test__HermitianSpectralSubspaceDiagnostics__unit.py
│   ├── test__HermitianSpectralSubspaceResult__unit.py
│   ├── test__HermitianSpectralSubspaceSolver__contract.py
│   ├── test__HermitianSpectralSubspaceSolverFactory__unit.py
│   └── test__HermitianSpectralSubspaceWheel__integration.py
└── numpy/
    └── hermitian_spectral_subspace/
        ├── test__NumpyHermitianSpectralSubspaceSolver__integration.py
        └── test__NumpyHermitianSpectralSubspaceSolver__numerical.py
```

The contract suite covers exact nominal typing, shared symmetrization, cutoff equality,
malicious backend eigensystems, diagnostics recomputation, trusted-metadata limitations,
and exception propagation. It includes the exact wrong-projector and wrong-spectrum
counterexamples from independent review. The NumPy suite covers analytic matrices,
complex phases, internal degeneracy, boundary-cutting rejection, coordinate covariance,
shift/scale relations, independently converged high-precision references,
convergence-failure translation, and repeated-call behavior.

KSD-shaped fixtures use owner-native tuple matrices and index ranges; owner tests do not
import KSD or PhysKit. The deterministic stress procedure and retained environment/error
evidence are documented in
[`benchmarks/hermitian_spectral_subspace.md`](../../../../../../benchmarks/hermitian_spectral_subspace.md).

## Documentation and reference tooling

The six architecture pages, including the annotated bibliography, are part of the slice.
Every academic-reference bullet carries a valid DOI or ISBN. Before review:

```bash
REFERENCE_PAGE="docs/architecture/projectkoios/numerics/linalg/\
hermitian_spectral_subspace/references.md"

python scripts/check_reference_identifiers.py --minimum 8 "${REFERENCE_PAGE}"
python scripts/inspect_reference_metadata.py \
  --output /tmp/hermitian-spectral-subspace-reference-metadata.json \
  "${REFERENCE_PAGE}"
```

The offline checker validates DOI form, percent encoding, ISBN checksums, missing
identifiers, and duplicate identifiers. The online inspector retrieves Crossref DOI
metadata and Open Library ISBN metadata for human review. Metadata checks supplement
rather than replace reading abstracts, contents, and publisher descriptions.

The normalized JSON remains an ephemeral audit output rather than a committed artifact:
provider records can change, and the annotated bibliography records what was inspected.
The executable scripts and resolving identifiers are the reproducible audit path.

## Packaging and dependency requirements

- The core module imports with no NumPy installation.
- Constructing `NUMPY_DENSE` is the first operation that imports NumPy.
- `projectkoios.numerics` imports neither Optimization, PhysKit, nor KSD.
- The wheel includes the new modules and Numerics `py.typed` marker.
- An isolated core environment exercises record construction without NumPy.
- An isolated NumPy environment constructs the backend and solves an analytic complex
  Hermitian fixture from the built wheel.
- Source distributions and wheels remain reproducible under explicit
  `SOURCE_DATE_EPOCH`.

## Acceptance gate

Qualification is not complete until the exact scoped tree passes:

1. all mapped unit, contract, integration, adversarial, and numerical tests;
2. high-precision oracle residual, orthogonality, reconstruction, and precision-convergence
   checks;
3. deterministic stress calibration of repository-owned validation factors and bounded
   fixture constants across available NumPy/LAPACK environments, with retained seeds,
   observed maxima, platform, Python, exactly resolved mpmath, NumPy, and
   `numpy.show_config()` evidence;
4. strict mypy over source and mapped tests;
5. Ruff lint and formatting;
6. local documentation-link and DOI/ISBN validation;
7. independent inspection of the cited academic sources;
8. reproducible wheel and source-distribution builds;
9. isolated dependency-free and NumPy wheel smoke tests;
10. independent implementation review; and
11. KSD consumer-adapter review against an exact owner revision.

These gates establish bounded software and numerical verification only. Commit, push,
release, publication, and downstream binding remain separately authorized actions.
