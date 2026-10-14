# Hermitian spectral-subspace decomposition

## Status

This directory defines the independently reviewed architecture requirements for the
locally implemented invariant-only
`projectkoios.numerics.linalg.hermitian_spectral_subspace` slice. Architecture
and KSD consumer review found no remaining contract blocker after the corrections
recorded here. Local source and qualification were subsequently authorized. Independent implementation,
numerical-evidence, and command-backed qualification reviews closed with no blocking
findings. No commit, push, downstream dependency change, release, publication, or
frame-producing eigensystem is authorized by that local action.

The first slice will decompose one finite dense complex Hermitian representative, return
all eigenvalues in ascending order, and return the orthogonal projector onto one
caller-selected contiguous eigenvalue-index range. It will not expose eigenvectors.

## Mathematical operation

For a finite matrix $A\in\mathbb{C}^{n\times n}$, the operation first measures the
Hermiticity defect of the unsymmetrized input. If that defect satisfies an explicit
input tolerance, it constructs the declared Hermitian representative

$$
H=\frac{A+A^\dagger}{2}.
$$

Let

$$
H=V\operatorname{diag}(\lambda_0,\ldots,\lambda_{n-1})V^\dagger,
\qquad
\lambda_0\leq\cdots\leq\lambda_{n-1}.
$$

For a caller-supplied half-open contiguous range $I=[a,b)$ satisfying
$0\leq a<b\leq n$, the returned invariant object is

$$
P_I=V_I V_I^\dagger
   =\sum_{j=a}^{b-1}v_jv_j^\dagger.
$$

The individual columns of $V_I$ are not returned. Internal phase changes and unitary
rotations within a degenerate selected block therefore leave $P_I$ unchanged.

## Why the boundary is an index range

The numerical owner does not decide which eigenvalues have physical or scientific
meaning. A consumer explicitly supplies $[a,b)$:

- KSD may select the lowest retained group as $[0,r)$ while retaining the meaning of
  $r$, the energy unit, and the scientific external-gap criterion;
- another consumer may select an interior range when both boundaries pass the declared
  isolation policy; and
- selecting the complete range $[0,n)$ returns the complete-space projector without a
  boundary-isolation test.

The owner applies only a declared cutoff policy to each present selected boundary and
retains eigensystem residual diagnostics separately. The diameter-based cutoff alone is
not a proof of backward-error resolvability. The owner does not claim a physical band
gap, an admissibility threshold, or scientific model adequacy.

## Public-operation boundary

The capability owns:

- exact dependency-free matrix, index-range, tolerance, diagnostic, and result records;
- validation of square shape, finite complex entries, and nonempty dimension;
- measurement of the unsymmetrized input-Hermiticity defect;
- explicit symmetrization after that input passes validation;
- ascending eigenvalue order with multiplicity retained;
- declared isolation checks at the lower and upper selected boundaries;
- shared validation of the complete protected eigensystem and shared construction of the
  selected orthogonal spectral projector;
- eigenpair, reconstruction, invariant-projector, commutator, trace, and
  spectral-correlation diagnostics;
- trusted API-level backend identifier and version reporting, without authentication or
  linked-library provenance claims;
- a nominal solver boundary, closed built-in method enum, and lazy factory; and
- an initial dense NumPy implementation.

It does not own:

- eigenvector phases, canonical eigenvectors, or deterministic degenerate-block frames;
- physical units, lattice or basis identity, band labels, occupied-state meaning, or
  retained-rank policy;
- the scientific meaning of a spectral gap or the lower bound required by an
  experiment;
- Procrustes alignment, frame transport, gauge selection, or closure optimization;
- sparse, iterative, partial-spectrum, generalized, or non-Hermitian eigenproblems; or
- scientific validation of a consumer model.

## Proposed public vocabulary

- `HermitianSpectralMatrix`
- `HermitianSpectralSubspaceTolerance`
- `HermitianSpectralSubspaceDefinition`
- `HermitianSpectralSubspaceDiagnostics`
- `HermitianSpectralSubspaceResult`
- `HermitianSpectralSubspaceSolver`
- `HermitianSpectralSubspaceMethod`
- `HermitianSpectralSubspaceSolverFactory`
- `NumpyHermitianSpectralSubspaceSolver`

The names describe a specific decomposition problem. They do not introduce a universal
`Eigensolver`, `Decomposition`, `Projector`, or `Certificate` hierarchy.

## Nonuniqueness gate

This slice intentionally stops at invariant output. A frame $F$ satisfying

$$
F^\dagger F=I,
\qquad
FF^\dagger=P_I
$$

is not unique: $FU$ is an equally valid frame for every unitary $U$ of the selected
rank. Returning backend eigenvectors would therefore add phase and degenerate-subspace
policy that this contract does not claim.

A future frame-producing slice requires separate review of spectral clustering, phase
pivots, degenerate-block completion, coordinate dependence, failure behavior, and
independent frame-residual verification. Nothing in this architecture resolves whether
KSD M2 needs eigenvector-diagonal frames or only an orthonormal frame spanning the
retained subspace.

## Qualification gates

The local implementation may be considered qualified only after review closes:

1. exact record fields and scalar/container types;
2. the input-Hermiticity measure and explicit symmetrization rule;
3. the strict boundary-isolation policy, its shift-invariant scale, and its explicitly
   non-certifying interpretation;
4. complete-eigensystem correlation, shared projector construction, and relational
   validation;
5. precise NumPy failure translation and trusted API-level backend metadata;
6. bounded independent high-precision and analytic numerical criteria;
7. malicious-subclass, cluster-cutoff, degeneracy, covariance, and wheel tests; and
8. KSD consumer review confirming that scientific controls remain downstream.

Architecture approval does not authorize implementation, commit, push, release,
publication, or downstream dependency binding.

## Pages

- [Scientific and interpretation boundary](science.md)
- [Numerical contract](numerics.md)
- [Implementation organization](implementation.md)
- [Schematics](schematics.md)
- [Inspected academic references](references.md)
