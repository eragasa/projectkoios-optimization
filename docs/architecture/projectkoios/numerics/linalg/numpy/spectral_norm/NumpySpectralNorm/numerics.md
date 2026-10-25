# `NumpySpectralNorm` numerics

## Definition

For $A \in \mathbb{F}^{m \times n}$ with positive $m$ and $n$,

$$
\operatorname{result}(A)
= \lVert A\rVert_2
= \sigma_1(A),
$$

where the complete singular-value sequence satisfies

$$
\sigma_1(A) \geq \cdots \geq \sigma_{\min(m,n)}(A) \geq 0.
$$

The class selects index zero from the descending sequence returned by
`numpy.linalg.svdvals`. This definition applies to square, tall, wide, real, complex,
rank-deficient, one-row, one-column, and zero matrices.

## Binary64 behavior

The class accepts only float64 and complex128 storage. It rejects nonfinite source
entries before decomposition and rejects nonfinite backend output after decomposition.
A finite matrix whose largest singular value exceeds binary64 range raises
`FloatingPointError`.

The implementation applies no owner-level scaling. It exposes no tolerance, starting
vector, iteration limit, residual, or approximate convergence policy because the backend
operation computes a complete dense singular-value sequence.

## Verification bound

Analytic fixtures use the bound

$$
64\,\max(m,n)\,\operatorname{ulp}(1)\,\max(1,\sigma_1).
$$

This bound is the test suite's binary64 roundoff allowance. It is not a production
algorithm parameter or scientific acceptance threshold. The zero matrix has a separate
exact `0.0` assertion.
