# Hermitian spectral-subspace numerical contract

## Status and references

This page defines the current local implementation's mathematical, tolerance,
diagnostic, and failure contract. Its rationale uses the
[inspected DOI/ISBN references](references.md): the spectral theorem and projector
vocabulary follow Bhatia and Kato; boundary sensitivity follows Davis–Kahan and
Stewart–Sun; dense eigensystem and finite-precision behavior follow Golub–Van Loan,
Higham, and the LAPACK guide.

The references support the mathematical direction and qualitative error scaling. The
exact tolerance formulas, validation factors, sanitized-fixture bounds, and failure
classes below are ProjectKoios-owned software policies rather than cited theorems.

## Dependency-free matrix representations

The public matrix representation is

```python
type HermitianSpectralMatrix = tuple[tuple[complex, ...], ...]
```

`HermitianSpectralMatrix` is part of the public import surface. It denotes representation,
not proof that a value is Hermitian; the definition and final solver establish the
applicable invariants.

The definition requires:

1. an exact outer built-in tuple;
2. at least one row;
3. exact built-in tuple rows;
4. square shape with common positive dimension $n$;
5. exact built-in complex entries; and
6. finite real and imaginary components.

Real Hermitian matrices use built-in complex values with zero imaginary parts. NumPy
arrays, NumPy scalars, lists, quantity objects, and arbitrary numeric subclasses are
rejected at the public dependency-free boundary.

A protected backend may return an eigenvector matrix in the same exact nested-tuple
representation. That matrix is never placed in a public result and does not imply a
public phase or frame convention.

## Tolerance record

`HermitianSpectralSubspaceTolerance` is a frozen, slotted record containing:

- `hermiticity_absolute: float`;
- `hermiticity_relative: float`;
- `boundary_absolute: float`;
- `boundary_relative: float`;
- `projector_validation_absolute: float`;
- `spectral_validation_absolute: float`; and
- `spectral_validation_relative: float`.

Every value is an exact built-in finite float. Absolute tolerances are nonnegative.
Relative tolerances are in $[0,1)$. Zero is permitted for every field, including a
consumer that intentionally requests absolute-only behavior.

The fields have distinct roles:

- Hermiticity tolerances decide whether the original input admits the declared Hermitian
  representative.
- Boundary tolerances implement a declared index-boundary isolation policy.
- Projector validation bounds dimensionless eigenvector and projector relations.
- Spectral validation bounds matrix-scale eigensystem, reconstruction, commutator, and
  trace-correlation relations.

The boundary formula is not by itself a backward-error theorem or proof of “numerical
resolvability.” A consumer scientific gap requirement is also separate.

## Definition record

`HermitianSpectralSubspaceDefinition` is a frozen, slotted record containing:

- `matrix: HermitianSpectralMatrix`;
- `start_index: int`;
- `stop_index: int`; and
- `tolerance: HermitianSpectralSubspaceTolerance`.

The index fields are exact built-in integers; booleans and integer subclasses are
rejected. For inferred matrix dimension $n$, construction requires

$$
0\leq \texttt{start\_index}<\texttt{stop\_index}\leq n.
$$

The selected rank is

$$
r=\texttt{stop\_index}-\texttt{start\_index}>0.
$$

The record exposes read-only `dimension` and `selected_rank` properties. It performs
shape, scalar, finite-state, index, and tolerance validation but does not import NumPy or
perform an eigendecomposition.

## Input Hermiticity measure

For original input $A$, define

$$
s_A=\max_{i,j}|A_{ij}|
$$

and

$$
d_H(A)=\max_{i,j}|A_{ij}-\overline{A_{ji}}|.
$$

For a one-by-one matrix, this includes twice the magnitude of a nonzero imaginary
diagonal component. The applied input cutoff is

$$
\eta_H
=\texttt{hermiticity\_absolute}
+\texttt{hermiticity\_relative}\,s_A.
$$

The final solver accepts the input exactly when

$$
d_H(A)\leq\eta_H.
$$

Equality is accepted. Any nonfinite scale, defect, or cutoff raises
`FloatingPointError` rather than being accepted, clamped, or returned partially.

## Explicit Hermitian representative

After acceptance, the shared solver template constructs

$$
H=\frac{A+A^\dagger}{2}
$$

in deterministic row-major scalar complex128-equivalent arithmetic. It records

$$
d_{\mathrm{adjust}}=\max_{i,j}|H_{ij}-A_{ij}|.
$$

The backend receives $H$, not $A$. This prevents it from silently selecting a lower or
upper input triangle. The operation does not mutate the definition and does not claim
that symmetrization repairs a scientifically invalid operator.

The shared template verifies that $H$ is finite and exactly Hermitian under its
constructed built-in-complex representation. A backend requiring another representative
needs separate review.

## Protected complete eigensystem

The backend returns a complete candidate eigensystem:

$$
(\lambda_0,\ldots,\lambda_{n-1}),\qquad
V=(v_0\;\cdots\;v_{n-1}).
$$

The eigenvalues are exact built-in finite floats in ascending order with multiplicity:

$$
\lambda_0\leq\lambda_1\leq\cdots\leq\lambda_{n-1}.
$$

The eigenvector matrix is an exact square tuple matrix of finite built-in complex values.
It is protected transient evidence only. A decreasing pair, wrong count or shape,
non-built-in scalar, nonfinite value, or complex eigenvalue is an invalid backend output.

The shared template checks the backend's index-to-eigenspace association through the
complete eigenpair residual, not merely through projector trace identities. Define

$$
d_{V,I}=\max_{i,j}|(V^\dagger V-I)_{ij}|,
$$

$$
d_{eig}=\max_{i,j}|(HV-V\operatorname{diag}(\lambda))_{ij}|,
$$

and

$$
d_{recon}=\max_{i,j}|(H-V\operatorname{diag}(\lambda)V^\dagger)_{ij}|.
$$

These relations reject, for example, an ascending spectrum with the correct total trace
but incorrect eigenvalues. They also bind each ordered eigenvalue to the corresponding
column used later to construct the selected projector.

## Spectrum scale and boundary-isolation policy

The spectral diameter is

$$
D=\lambda_{n-1}-\lambda_0\geq 0.
$$

It is invariant under adding a scalar multiple of the identity and scales linearly under
positive matrix scaling. The declared boundary-isolation cutoff is

$$
\eta_B
=\texttt{boundary\_absolute}
+\texttt{boundary\_relative}\,D.
$$

For $a=\texttt{start\_index}$ and $b=\texttt{stop\_index}$, define

$$
g_{\mathrm{lower}}=
\begin{cases}
\lambda_a-\lambda_{a-1}, & a>0,\\
\text{absent}, & a=0,
\end{cases}
$$

and

$$
g_{\mathrm{upper}}=
\begin{cases}
\lambda_b-\lambda_{b-1}, & b<n,\\
\text{absent}, & b=n.
\end{cases}
$$

Every present boundary must satisfy

$$
g>\eta_B.
$$

A zero, negative, equal-to-cutoff, or below-cutoff boundary fails closed with
`ValueError`. Absent boundaries are `None`, not infinite or sentinel gaps. The complete
range $[0,n)$ has no external boundary-isolation condition.

This is a declared contract policy. Davis–Kahan and LAPACK establish that subspace error
depends on perturbation or residual magnitude relative to separation; they do not derive
this exact diameter-based cutoff. The independently checked eigensystem residual is
therefore retained rather than claiming that $D$ alone proves resolvability.

Internal gaps do not participate in acceptance. An exact or near degeneracy wholly
inside $[a,b)$ is permitted.

## Shared projector construction

After the complete eigensystem and boundary checks pass, the shared template—not the
backend—constructs

$$
P=V_{[:,a:b]}V_{[:,a:b]}^\dagger.
$$

The public projector is an exact finite `HermitianSpectralMatrix` of shape $(n,n)$. The
protected eigenvector matrix is then discarded.

This construction closes the target-association defect that would occur if an arbitrary
backend-supplied projector were tested only by rank, commutation, and first moment. For
$H=\operatorname{diag}(0,1,2,3)$ and $[a,b)=[1,3)$, the wrong projector
$\operatorname{diag}(1,0,0,1)$ satisfies those weaker identities, but it cannot arise from
columns one and two of a complete ascending eigensystem that passes the eigenpair
residual.

In exact arithmetic, the constructed result satisfies

$$
P=P^\dagger,
\qquad
P^2=P,
\qquad
\operatorname{tr}(P)=r,
\qquad
HP=PH,
$$

$$
\operatorname{tr}(HP)=\sum_{j=a}^{b-1}\lambda_j,
\qquad
\operatorname{tr}(H)=\sum_{j=0}^{n-1}\lambda_j.
$$

## Shared relational diagnostics and acceptance

The final template independently recomputes finite nonnegative binary64 diagnostics from
$A$, $H$, the protected complete eigensystem, and the shared projector. Let

$$
s_H=\max_{i,j}|H_{ij}|,
$$

$$
\eta_S
=\texttt{spectral\_validation\_absolute}
+\texttt{spectral\_validation\_relative}\,s_H,
$$

and

$$
\eta_P=\texttt{projector\_validation\_absolute}.
$$

In addition to $d_H(A)$, $d_{\mathrm{adjust}}$, $d_{V,I}$, $d_{eig}$, and $d_{recon}$,
the template calculates

$$
d_{P,H}=\max_{i,j}|P_{ij}-\overline{P_{ji}}|,
$$

$$
d_{P,2}=\max_{i,j}|(P^2-P)_{ij}|,
$$

$$
d_{P,tr}=|\operatorname{tr}(P)-r|,
$$

$$
d_{comm}=\max_{i,j}|(HP-PH)_{ij}|,
$$

$$
d_{sel}=\left|\operatorname{tr}(HP)-\sum_{j=a}^{b-1}\lambda_j\right|,
$$

and

$$
d_{tr}=\left|\operatorname{tr}(H)-\sum_j\lambda_j\right|.
$$

Matrix products use deterministic row, column, and inner-index scalar order. Stable
real/imaginary component summation is required where it prevents avoidable overflow or
loss. Any nonfinite intermediate raises `FloatingPointError`.

The shared template requires

$$
d_{V,I}\leq n\eta_P,
\qquad
d_{P,H}\leq 2n\eta_P,
\qquad
d_{P,2}\leq 2n\eta_P,
\qquad
d_{P,tr}\leq r\eta_P,
$$

and

$$
d_{eig}\leq n\eta_S,
\qquad
d_{recon}\leq 2n\eta_S,
\qquad
d_{comm}\leq 2n\eta_S,
\qquad
d_{sel}\leq r\eta_S,
\qquad
d_{tr}\leq n\eta_S.
$$

The dimension factors are ProjectKoios-owned conservative acceptance policies for
bounded dense cases, not sharp academic error bounds. They must be stress-calibrated
against the reference families described below before implementation approval. A
relation outside its declared bound is rejected even when the output is structurally
valid.

## Exact diagnostics record

`HermitianSpectralSubspaceDiagnostics` is a frozen, slotted record with these exact
built-in finite nonnegative float fields:

- `input_hermiticity_maximum_absolute_defect` for $d_H(A)$;
- `representative_adjustment_maximum_absolute_defect` for $d_{\mathrm{adjust}}$;
- `eigenvector_orthonormality_maximum_absolute_defect` for $d_{V,I}$;
- `eigensystem_residual_maximum_absolute_defect` for $d_{eig}$;
- `representative_reconstruction_maximum_absolute_defect` for $d_{recon}$;
- `projector_hermiticity_maximum_absolute_defect` for $d_{P,H}$;
- `projector_idempotence_maximum_absolute_defect` for $d_{P,2}$;
- `projector_trace_absolute_defect` for $d_{P,tr}$;
- `projector_commutator_maximum_absolute_defect` for $d_{comm}$;
- `selected_spectral_trace_absolute_defect` for $d_{sel}$;
- `full_spectral_trace_absolute_defect` for $d_{tr}$;
- `input_matrix_maximum_absolute_entry` for $s_A$;
- `representative_matrix_maximum_absolute_entry` for $s_H$;
- `applied_hermiticity_cutoff` for $\eta_H$;
- `spectral_diameter` for $D$;
- `applied_boundary_cutoff` for $\eta_B$;
- `applied_projector_validation_cutoff` for $\eta_P$; and
- `applied_spectral_validation_cutoff` for $\eta_S$.

The final template constructs all fields. A backend cannot provide or override them. This
schema lets KSD preserve the original-input defect and representative adjustment without
duplicating owner mathematics.

## Result record

`HermitianSpectralSubspaceResult` is a frozen, slotted record containing:

- `dimension: int`;
- `start_index: int`;
- `stop_index: int`;
- `tolerance: HermitianSpectralSubspaceTolerance`;
- `eigenvalues: tuple[float, ...]`;
- `projector: HermitianSpectralMatrix`;
- `lower_boundary_gap: float | None`;
- `upper_boundary_gap: float | None`;
- `diagnostics: HermitianSpectralSubspaceDiagnostics`;
- `backend_identifier: str`; and
- `backend_version: str`.

The backend strings are exact nonempty built-in strings. The nominal backend declares
them through protected metadata properties; they are trusted API-level provenance, not
authenticated identity. For the initial class they identify `numpy.linalg.eigh` and the
runtime NumPy version. They do not identify the linked BLAS/LAPACK library or promise
bytewise reproducibility across builds.

The final solver correlates the dimension, range, tolerance, complete eigensystem,
shared-constructed projector, gaps, and diagnostics with the exact definition before it
constructs the public result.

## Nominal solver semantics

`HermitianSpectralSubspaceSolver.solve(definition, /)` is final. It:

1. requires the exact `HermitianSpectralSubspaceDefinition` type;
2. measures and validates the original input Hermiticity;
3. constructs the explicit Hermitian representative;
4. invokes the protected backend complete eigendecomposition;
5. validates exact output structure, scalar types, shape, finiteness, and ascending order;
6. verifies complete eigenvector orthonormality, eigenpair residual, and reconstruction;
7. applies the declared strict selected-boundary policy;
8. constructs the selected projector from the validated ordered eigenvectors;
9. recomputes projector and spectral-correlation diagnostics;
10. applies the declared validation policies;
11. validates protected backend metadata strings; and
12. constructs the exact public result without exposing eigenvectors.

External implementations use nominal inheritance and implement only the protected
complete decomposition and metadata properties. Structural protocols, automatic backend
discovery, and mutable registries are excluded.

## Failure semantics

The dependency-free records and final template raise:

- `TypeError` for wrong nominal records, exact scalar types, semantic containers, or
  protected output types;
- `ValueError` for empty or nonsquare matrices, row-width mismatch, invalid indices,
  invalid tolerance ranges, input beyond Hermiticity tolerance, a boundary that fails the
  declared isolation policy, nonascending eigenvalues, malformed eigenvector shape,
  failed eigenpair/reconstruction checks, or violated projector relations;
- `FloatingPointError` for nonfinite shared or backend arithmetic and nonfinite output;
  and
- `RuntimeError`, chained from exactly `numpy.linalg.LinAlgError`, when the initial NumPy
  eigendecomposition does not converge.

No other backend exception is translated. `MemoryError`, `KeyboardInterrupt`,
`SystemExit`, programming exceptions, and undeclared third-party failures propagate.

## Initial NumPy backend

`NumpyHermitianSpectralSubspaceSolver` converts the already symmetrized tuple matrix to
`numpy.complex128` and calls

```python
numpy.linalg.eigh(matrix, UPLO="L")
```

The representative is already Hermitian, so `UPLO="L"` is a storage choice rather than a
hidden choice about the original input. NumPy returns eigenvalues ascending. The backend
converts all eigenvalues and the complete transient eigenvector matrix to exact built-in
tuples for shared validation and projector construction.

The backend executes with invalid and overflow conditions configured to raise, verifies
all returned arrays are finite, and translates only `numpy.linalg.LinAlgError` as
specified. It never returns NumPy arrays publicly. The complete eigenvector tuple exists
only across the protected subclass seam and is discarded by the final template.

## Bounded numerical verification

The suite must not compare one `numpy.linalg.eigh` call with itself. It includes:

1. exact diagonal matrices with analytically known eigenvalues and projectors;
2. real and complex unitary rotations of known diagonal spectra;
3. exact degeneracies wholly inside the selected range;
4. exact degeneracies cut by a selected boundary and therefore rejected;
5. below-, equal-, and above-cutoff boundary fixtures;
6. lowest, interior, and complete index ranges;
7. scalar identity shifts demonstrating boundary-policy shift invariance while separately
   checking matrix-scale eigensystem residuals;
8. positive matrix rescaling with consistently rescaled absolute tolerances;
9. coordinate permutations verifying $H'=SHS^\dagger$ and $P'=SPS^\dagger$;
10. independent phase and internal-block rotations yielding the same projector;
11. approximate-Hermitian inputs verifying original defect, representative, and
    adjustment;
12. malicious nominal subclasses returning wrong eigenvalue counts, decreasing or
    trace-preserving-but-wrong spectra, malformed eigenvectors, nonorthonormal bases,
    inconsistent eigenpairs, nonfinite values, or empty metadata;
13. the exact adversarial matrix $H=\operatorname{diag}(0,1,2,3)$ with range $[1,3)$,
    proving that an unrelated equal-first-moment projector is not accepted or constructed;
14. the exact full-range matrix $H=\operatorname{diag}(0,1,2)$, proving that the false
    trace-preserving spectrum $(-1,1,3)$ fails eigenpair validation;
15. unchanged propagation of undeclared backend exceptions;
16. NumPy convergence-failure translation through a controlled backend seam;
17. one- through eight-dimensional cases against an independent high-precision oracle;
    and
18. KSD-shaped lowest-group fixtures without importing KSD or PhysKit.

### Independent high-precision oracle

The development/test-only oracle uses `mpmath.eigh` on an explicitly constructed
Hermitian `mpmath.matrix`, never NumPy. Qualification pins one exact mpmath release in
its resolved development environment, and calibration evidence records
`mpmath.__version__`. Each binary64 real and imaginary input component is converted
exactly using
`float.as_integer_ratio()` followed by high-precision integer division. The oracle runs
at 100 and 160 decimal digits.

For dimension at most eight, input entry magnitudes at most four, and every present
reference boundary gap at least $10^{-3}$ times the reference matrix scale, it requires:

- high-precision eigenpair residual, orthonormality, and reconstruction maximum defects
  below $10^{-80}\max(1,\lVert H\rVert_F)$ at 160 digits;
- 100-to-160-digit eigenvalue agreement below $10^{-80}$ times that scale;
- 100-to-160-digit selected-projector Frobenius agreement below
  $10^{-75}\max(1,\lVert H\rVert_F/g)$ when a boundary is present; and
- the analogous projector identity and rank checks at both precisions.

Internal exact degeneracy is tested through the projector, not through individual
high-precision eigenvectors.

Let $\epsilon$ be binary64 machine epsilon, $S=\max(1,\lVert H\rVert_F)$, and $g$ the
smallest present reference boundary gap. Sanitized binary64 fixtures require

$$
\max_j|\lambda_j-\lambda_j^{ref}|
\leq 512\epsilon nS
$$

and, when a boundary is present,

$$
\lVert P-P^{ref}\rVert_F
\leq 4096\epsilon n\max\!\left(1,\frac{S}{g}\right).
$$

For the complete range, compare the projector with the identity using
$512\epsilon n$. These constants are repository-owned acceptance criteria, not universal
forward-error theorems. Before implementation review they must be exercised over all
analytic cases plus at least 256 deterministic seeded complex Hermitian cases across the
NumPy/LAPACK environments available in continuous integration. Any systematic false-rejection triggers recalibration and renewed review rather than
silent relaxation. Retained calibration evidence records Python, mpmath and NumPy
versions, `numpy.show_config()` linked-library details, platform identity, case seeds, and
all observed maxima.

Tests establish bounded software and numerical behavior only. They do not validate a
physical Hamiltonian, retained-rank choice, or scientific gap criterion.
