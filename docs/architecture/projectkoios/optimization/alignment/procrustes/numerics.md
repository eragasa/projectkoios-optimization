# Procrustes numerics

## Backend evidence

The dependency-free final template constructs $A^\dagger B$ with checked complex
binary64 products and componentwise `math.fsum`. It requires a complete square SVD from
the backend and independently validates:

- nonnegative nonincreasing finite singular values;
- left and right factor unitarity;
- reconstruction of the cross matrix; and
- for a returned full-rank transformation, right-unitarity and direct-residual
  agreement with the singular-value optimum.

The validation cutoff has separate absolute and relative components. It validates the
software/backend correlation only and does not affect the numerical-rank decision.

## Numerical rank

The numerical-rank cutoff uses explicit absolute and relative components and the largest
cross singular value. Equality with the cutoff is deficient. Numerical rank is evidence
under a declared software policy, not exact algebraic rank, a condition certificate, or
a scientific overlap threshold.

## Optimal residual

To reduce avoidable overflow, the residual identity is reconstructed after scaling all
terms by their largest magnitude. A reconstructed value whose magnitude lies within an
epsilon-scaled roundoff envelope that grows with the reconstruction term count is
clamped to zero. A negative value outside that envelope raises `FloatingPointError`.
This means residuals below the stated binary64 reconstruction resolution are reported as
zero rather than distinguished from cancellation noise. Source and target powers, all
singular values, and the retained squared result let consumers independently reconstruct
the same identity and cutoff.

## Failure behavior

- malformed public inputs raise `TypeError` or `ValueError`;
- nonfinite input is rejected before backend invocation;
- nonfinite/overflowing shared arithmetic raises `FloatingPointError`;
- malformed or inconsistent backend evidence raises `TypeError`, `ValueError`, or
  `FloatingPointError`;
- NumPy SVD nonconvergence becomes `RuntimeError`;
- numerical rank deficiency is a successful `RANK_DEFICIENT` result, not an exception.

No public full transform is formed on the deficient branch.
