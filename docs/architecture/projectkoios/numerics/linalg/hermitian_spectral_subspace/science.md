# Hermitian spectral-subspace interpretation boundary

## Status

This page defines the mathematical and scientific-interpretation boundary for the
locally implemented invariant-only Hermitian spectral-subspace solver. Local
implementation was separately authorized; the page does not authorize a commit,
downstream binding, frame-producing extension, or scientific interpretation.

The terminology and perturbation rationale are grounded in the
[inspected DOI/ISBN bibliography](references.md). In particular, Bhatia and Kato support
the matrix/spectral-projector vocabulary, Davis–Kahan and Stewart–Sun support the
spectral-gap dependence of invariant subspaces, and Golub–Van Loan, Higham, and the
LAPACK guide support the numerical-analysis distinctions.

## Spectral theorem and invariant object

For a finite Hermitian matrix $H=H^\dagger$, the spectral theorem gives a unitary
matrix $V$ and real eigenvalues $\lambda_j$ such that

$$
H=V\operatorname{diag}(\lambda_0,\ldots,\lambda_{n-1})V^\dagger.
$$

An ordered individual eigenvector is not generally an invariant coordinate object:

- a nondegenerate eigenvector may be multiplied by any complex phase;
- a degenerate eigenspace may be represented by any orthonormal basis related by an
  internal unitary transformation; and
- a numerically near-degenerate group may rotate strongly under perturbations that are
  small relative to the matrix scale but not small relative to the separating gap.

For an isolated selected set of eigenvalue indices $I$, the orthogonal spectral
projector

$$
P_I=\sum_{j\in I}v_jv_j^\dagger
$$

is invariant under all phase changes and internal unitary rotations within the selected
subspace. This is why the first owner slice returns $P_I$ rather than the columns used by
a particular eigensolver to construct it.

## Contiguous index selection

The operation accepts a half-open contiguous index range $I=[a,b)$ after sorting all
eigenvalues in ascending order with multiplicity retained. The range is caller-supplied
rather than inferred from a domain label or a physical threshold.

This choice separates three questions:

1. **Numerical ordering:** Numerics orders the real eigenvalues ascending.
2. **Declared boundary policy:** Numerics rejects a selected boundary whose returned gap
   is no larger than the supplied diameter-scaled cutoff and retains eigensystem residual
   evidence separately.
3. **Scientific selection:** the consumer decides why $[a,b)$ is the relevant group and
   whether its physical gap is scientifically adequate.

A lowest-$r$ consumer supplies $[0,r)$. An interior group supplies two boundaries. The
complete range $[0,n)$ has no external boundary. Empty ranges are excluded from the
initial API because no current consumer needs a rank-zero result and because admitting
one would add avoidable boundary semantics.

## Internal degeneracy versus boundary degeneracy

Internal degeneracy is admitted. If $\lambda_j=\lambda_{j+1}$ and both indices lie
inside $[a,b)$, the selected projector includes the complete degenerate block and does
not depend on the backend's internal basis.

A degeneracy cut by $a$ or $b$ is different. The requested index set then asks for only
part of one eigenspace, so no invariant projector is determined by the matrix and the
index range alone. The solver must reject that request. A near-degenerate boundary is also rejected when its gap is no larger than the declared
cutoff.

The strict boundary rule is a declared representational policy, not a theorem that every
larger returned gap is numerically resolved or physically meaningful. The Davis–Kahan
analysis summarized in the inspected reference explains why subspace sensitivity
depends on perturbation or residual size relative to a separating gap. The solver
therefore validates and reports complete-eigensystem residuals separately; it does not
convert the diameter-based policy into an unbounded accuracy guarantee.

## Approximate Hermiticity and declared representative

A numerical consumer may supply a matrix that is Hermitian only within a declared input
tolerance. A Hermitian eigensolver nevertheless acts on a Hermitian representation; many
backends consume only one stored triangle. Silently relying on that triangle would make
the represented matrix backend-dependent.

The owner contract therefore makes the choice explicit:

1. measure and retain the Hermiticity defect of the original input $A$;
2. reject $A$ if the defect exceeds the supplied Hermiticity cutoff; and
3. otherwise decompose

$$
H=\frac{A+A^\dagger}{2}.
$$

The result reports both the original defect and the maximum adjustment from $A$ to $H$.
The symmetrized matrix is the unique orthogonal projection onto the real vector space of
Hermitian matrices under the Frobenius inner product; the inspected Higham reference
uses the corresponding symmetric part in the real structured-matrix setting.

This behavior is numerical representation, not scientific repair. A consumer remains
free to require exact Hermiticity or a stricter physical defect threshold. A later KSD
adapter must preserve the historical direct-NumPy path as evidence because NumPy's
triangle-consumption behavior need not equal explicit symmetrization for a merely
approximately Hermitian matrix.

## Meaning of returned eigenvalues

The result returns every eigenvalue of the explicit representative $H$, ordered
ascending and repeated according to multiplicity. Returning the full spectrum allows a
consumer to reconstruct its own retained/excluded gap and to verify the supplied index
range without asking Numerics to attach a label such as “occupied,” “valence,” “low
energy,” or “retained.”

The eigenvalues are unit-free numeric values at this boundary. A physical consumer may:

- convert a unit-bearing matrix to magnitudes in one declared unit;
- pass those magnitudes and numeric absolute tolerances to Numerics; and
- restore the unit and physical identities around the result.

Numerics does not accept or return PhysKit quantity objects.

## KSD M2 boundary

For the current M2 consumer, KSD may supply $[0,r)$, where $r$ is the KSD-owned retained
rank. KSD continues to own:

- the parent operator and its energy unit;
- the meaning of the lowest retained group;
- the scientific retained/excluded gap requirement;
- training, attack, and withheld sample roles;
- path, mesh, basis, and channel identities;
- evidence retention, failure policy, disposition, and interpretation; and
- independent reconstruction in the verifier.

The owner cutoff that implements the declared boundary-isolation policy is not the KSD
scientific lower bound on the external gap. It is also not, by itself, a backward-error
certificate. KSD must retain these distinct controls and claims separately.

The invariant result cannot replace M2's current raw-eigenframe path. Polar transport,
the controlled attack, pointwise Procrustes, and the global-unitary channel still consume
frames. Frame-producing extraction remains blocked until KSD decides whether it needs
eigenvector-diagonal frames or merely an orthonormal frame spanning the retained
subspace, and until the relevant phase and degenerate-completion policy is reviewed.

## Optimization boundary

No operation in this slice selects a transformation by minimizing an objective. It
decomposes a supplied matrix, validates a caller-supplied index range, and constructs the
associated invariant projector. It therefore belongs in Numerics.

The following remain Optimization responsibilities when separately authorized:

- pointwise or aggregate Procrustes transformation selection;
- optimized frame transport;
- gauge synchronization;
- assignment between eigenvalue or phase multisets; and
- closure transformation selection.

## Interpretation limits

A successful result establishes only that, under the declared numeric tolerances:

- the input was close enough to Hermitian to admit the declared representative;
- every present external boundary passed the declared boundary-isolation cutoff policy;
- the protected complete eigensystem satisfied the declared residual, orthonormality,
  reconstruction, and ordering checks; and
- the shared-constructed projector satisfied bounded invariant and spectral-correlation
  checks.

It does not establish:

- a physical band gap;
- correctness or adequacy of a Hamiltonian;
- material validity;
- occupancy, topology, polarization, or locality;
- continuity of a frame along a parameter path;
- convergence of a basis or discretization;
- uncertainty quantification; or
- scientific validation.

Software verification, bounded numerical verification, and scientific interpretation
remain separate claims.
