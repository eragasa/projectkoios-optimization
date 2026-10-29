# Matrix spectral-norm numerics

## Mathematical operation

For $A \in \mathbb{F}^{m \times n}$ with positive $m$ and $n$,

$$
\lVert A\rVert_2 = \sigma_1(A),
\qquad
\sigma_1(A) \geq \cdots \geq \sigma_{\min(m,n)}(A) \geq 0.
$$

The result has the same physical unit as one matrix entry. This repository evaluates only
the unit-free represented value.

## Dense complete path

`numpy.linalg.svdvals` computes the complete descending singular-value sequence. Index
zero supplies the dense result. The path exposes no tolerance, initial vector, iteration
limit, residual estimate, scaling fallback, or sparse conversion.

## Sparse direct path

Canonical all-zero storage returns exact `0.0`. For `min(shape) <= 2`, the evaluator
scales canonical stored values by their largest magnitude and forms only a sparse
one-by-one or two-by-two Gram matrix on the smaller axis. For the two-dimensional case,
with Hermitian Gram entries

$$
G = \begin{bmatrix}a & b\\ \overline{b} & d\end{bmatrix},
$$

the largest scaled squared singular value is

$$
\lambda_{\max}(G)
= \frac{a+d+\operatorname{hypot}(a-d,2|b|)}{2}.
$$

The evaluator takes its square root and restores the stored-value scale. Scaling occurs
before multiplication, so finite large entries do not overflow merely because a raw
square is unrepresentable. A mathematically unrepresentable final binary64 norm still
fails with `FloatingPointError`.

## Sparse ARPACK path

For `min(shape) >= 3`, the evaluator calls

```text
scipy.sparse.linalg.svds(
    k=1,
    which="LM",
    solver="arpack",
    tol=controls.tolerance,
    maxiter=controls.maximum_iterations,
    ncv=controls.krylov_dimension,
    v0=deterministic_normalized_start,
)
```

A real input receives seeded real standard-normal components. A complex input receives
independently drawn seeded real and imaginary components in complex128 storage. The
vector is normalized with SciPy's scale-safe dense-vector norm. Repeated calls with the
same matrix, controls, and pinned numerical environment receive the same start.

`krylov_dimension=None` delegates selection to pinned SciPy. The evaluator maps the
public `krylov_dimension` field to SciPy's private-boundary `ncv` keyword. It defines no
public `ncv` alias. Explicit values obey the actual SciPy 1.18 ARPACK boundaries:

- real input: `2 <= krylov_dimension < min(shape)`;
- complex input: `3 <= krylov_dimension < min(shape)`.

The direct branch requires `krylov_dimension=None`. This rule avoids pretending an
unused or invalid ARPACK subspace controls a direct calculation.

## Internal triplet qualification

For returned unit-vector candidates $u$ and $v$ and finite nonnegative $\sigma$, the
evaluator forms

$$
r_L = Av - \sigma u,
\qquad
r_R = A^H u - \sigma v.
$$

The scale-free relative residuals are

$$
\rho_L =
\frac{\lVert r_L\rVert_2}
{\max(\lVert Av\rVert_2, |\sigma|\lVert u\rVert_2)},
\qquad
\rho_R =
\frac{\lVert r_R\rVert_2}
{\max(\lVert A^H u\rVert_2, |\sigma|\lVert v\rVert_2)}.
$$

When a denominator and its residual numerator are both zero, the corresponding ratio is
exact zero. A zero denominator with nonzero numerator is infinity and fails.

Both vector norms must differ from one by no more than

$$
f = 64\,\operatorname{ulp}(1)\,\max(m,n).
$$

This structural bound is independent of caller tolerance. Both residual ratios must
satisfy

$$
\max(\rho_L,\rho_R) \leq \max(\text{tolerance}, f).
$$

All matrix-vector products, residual arrays, scale-safe vector norms, and
`|sigma| * vector_norm` normalization products must remain finite at their owning
arithmetic boundaries. Qualification evidence remains internal; the public result stays
a scalar.

## Approximation boundary

ARPACK completion plus owner residual qualification establishes a reproducible software
approximation under the pinned numerical environment. It does not establish a scientific
acceptance threshold, uncertainty statement, or scale-safe guarantee for every extreme
finite sparse matrix. Specialized correlated residual diagnostics remain separate
consumer-owned capabilities.
