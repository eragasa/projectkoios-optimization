# Numerics

## Represented function

For finite binary64 coordinates, both backends evaluate

$$
q(x)=q_{\min}+(x-c)^TQ(x-c)
$$

and reject nonfinite or negative backend results. `root_loss(x)` is exactly
`sqrt(squared_loss(x))`; it is not named RMS because normalization remains caller-owned.

## Dense acceptance

The dense backend requires exact represented symmetry and successful
`numpy.linalg.cholesky`. This is NumPy/LAPACK numerical acceptance for the represented
binary64 matrix, not an exact proof over ideal real coefficients.

## Sparse acceptance

The sparse backend requires exact represented symmetry after CSR canonicalization. For
$n>1$, it accepts only when the smallest algebraic eigenvalue returned by ARPACK through
`scipy.sparse.linalg.eigsh` is finite and strictly positive. The solver uses its machine-
precision convergence mode (`tol=0.0`). Failure to converge fails closed.

Sparse eigenvalue acceptance is iterative and backend-dependent. It must not be reused as
a scientific certificate or an outward-rounded feasibility proof.

## Evaluation failures

Both backends reject:

- coordinate container or dimension mismatch;
- NaN, infinity, Boolean, complex, or non-real coordinates;
- nonfinite matrix-vector arithmetic;
- a negative computed quadratic contribution; and
- overflow when adding `minimum_squared_loss`.

## Verification scope

Integration tests cover known completed squares, dense/sparse agreement, invalid
curvature and symmetry, result overflow, defensive sparse copies, and wheel import with
and without optional backends. These checks establish software behavior and bounded
numerical agreement only. They do not establish scientific validation, conditioning
adequacy, or fitness for a campaign.
