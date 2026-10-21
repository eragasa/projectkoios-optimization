# Uniform one-dimensional quadrature numerics

## Complete closed-interval rule

For an ordered finite interval $[a,b]$ and an endpoint-inclusive point count $N\geq2$,
the uniform spacing is

\[
 h=\frac{b-a}{N-1}.
\]

The composite trapezoidal weights are

\[
 w_0=w_{N-1}=\frac{h}{2},
 \qquad
 w_i=h\quad(1\leq i\leq N-2).
\]

Thus $\sum_i w_i=b-a$. The rule integrates constant and affine functions exactly in
exact arithmetic. The implementation retains binary64 weights and reports no continuum
error bound.

## Homogeneous-Dirichlet interior restriction

For $N\geq3$, homogeneous endpoint values leave the full-grid indices

\[
 1,2,\ldots,N-2
\]

as active degrees of freedom. Every active weight is $h$. For factors $u$ and $v$ with
$u_0=u_{N-1}=v_0=v_{N-1}=0$,

\[
 \sum_{i=0}^{N-1}w_i u_i v_i
 =h\sum_{i=1}^{N-2}u_i v_i.
\]

Dropping endpoints is not valid for a generic integrand with nonzero endpoint values.
The API therefore exposes the restriction explicitly rather than silently changing the
complete rule.

## Diagonal mass convention

The supported mass matrix is only

\[
 M=\operatorname{diag}(w).
\]

For the interior restriction this becomes $M_{\mathrm{int}}=hI_{N-2}$. This is the
diagonal or lumped quadrature mass convention. A consistent linear finite-element mass
matrix, generalized eigenproblem, and basis-function construction are distinct
capabilities and are excluded.

## Failure semantics

- `TypeError`: Boolean, coerced, or otherwise non-exact public scalar types; incorrect
  nominal records; incorrect NumPy dtypes.
- `ValueError`: nonfinite endpoints or arrays, reversed or degenerate intervals, invalid
  point counts, unrepresentable positive spacing or half-weight, malformed arrays, or
  inconsistent result correlation.

Physical units and model interpretation remain downstream.
