# Closed-loop polar transport numerics

## Reused Procrustes kernel

Every link is delegated to `NumpyComplexUnitaryProcrustesSolver`; transport does not
reimplement SVD, polar-factor, rank-cutoff, residual, or uniqueness logic. For a complete
singular spectrum `sigma`, the applied cutoff is the existing additive or maximum
Procrustes cutoff. Rank uses strict comparison, and the retained signed margin is

\[
\sigma_{\min}-\tau_{\mathrm{rank}}.
\]

A nonpositive margin makes the rank deficient. A deficient transformation is never
completed or applied.

## Schur distribution

Let `A` be the closure Procrustes right action from the primitive last frame toward the
sewn first frame. The result also retains the opposite endpoint-mismatch orientation
`E = A†`. SciPy computes the complex Schur form `E = Z T Z†` and verifies finite output,
unitary `Z`, reconstruction, near-diagonality of `T`, and unit-modulus diagonal entries.

The required definition branch controls only exact branch-cut representatives:

- `negative_pi_open` maps `-pi` to `+pi` and uses `(-pi, pi]`;
- `positive_pi_open` maps `+pi` to `-pi` and uses `[-pi, pi)`.

Phases and corresponding Schur vectors are stably sorted together for deterministic
storage. The storage order has no scientific band-label or ordered-spectrum meaning. The
distributed correction is

\[
W=Z\operatorname{diag}(\exp(-i\gamma_a/M))Z^\dagger.
\]

The solver verifies correction unitarity and `W**M = E† = A` within the declared
maximum-absolute-entry validation bound. The same ordered Schur vectors construct every
fractional correction. The result independently reconstructs `E`, `A`, and `W` without
invoking another decomposition.

## Metric-explicit evidence

Every reported defect names its maximum-absolute-entry metric:

- `maximum_absolute_entry_unitarity_defect` on each qualified link is
  `max(abs(Q† @ Q - I))`;
- `maximum_absolute_entry_closure_alignment_residual` is
  `max(abs(T[-1] @ A - S @ T[0]))`;
- `maximum_absolute_entry_closure_alignment_unitarity_defect` is
  `max(abs(A† @ A - I))`.

The result also retains each final right action `K[j]` and verifies
`transported[j] = input[j] @ K[j]`. Removing the distributed correction recovers the
primitive actions and independently reconstructs the closure-alignment residual and
per-link evidence.

These values must not be read as raw or normalized Frobenius diagnostics. Consumers that
require normalized Frobenius frame or unitary evidence divide the applicable Frobenius
norm by `sqrt(rank)` and retain that separate metric identity. The retained matrices and
actions are sufficient for that calculation; the generic owner does not assign a
consumer protocol tolerance or acceptance meaning.
