# `ScipyCsrSpectralNorm` numerics

## Piecewise primary algorithm

For canonical $A$:

- no stored entries gives exact `0.0`;
- $\min(m,n)\leq2$ uses a scaled sparse Gram matrix on the smaller axis; and
- $\min(m,n)\geq3$ uses one largest-magnitude singular triplet from deterministic SciPy
  ARPACK `svds`.

The direct branch is a defined dimension-specific algorithm, not a failure fallback. The
pinned SciPy complex ARPACK route rejects the minimum-dimension-two request, so the same
direct rule applies to real and complex inputs.

## Scaling and representability

The direct branch divides stored data by its maximum entry magnitude before sparse Gram
multiplication. Restoring scale occurs after the square root. Intermediate raw squares
therefore do not reject a representable large norm. Nonfinite entry magnitude, scaled
Gram arithmetic, or final rescaling raises `FloatingPointError`.

The ARPACK branch does not rescale the sparse matrix. Extreme finite matrices that cause
nonfinite backend or residual arithmetic fail explicitly.

## Approximate qualification

Returned singular vectors must have exact expected shapes, finite components, and unit
norms within the dimension floor

$$
64\,\operatorname{ulp}(1)\,\max(m,n).
$$

Both left and right normalized singular-equation residuals must not exceed the larger of
that floor and the explicit caller tolerance. A denominator and numerator both equal to
zero define a zero relative residual. A zero denominator with nonzero numerator fails.
The products of the reported singular value and each finite vector norm must remain
finite before either denominator is constructed.

Residual qualification detects malformed or numerically failed triplets. It does not
independently prove that a returned exact singular triplet is the largest; largest-
selection semantics remain part of the fixed `which="LM"` backend contract.

## Reproducibility boundary

A fixed matrix, exact controls, and pinned NumPy/SciPy/BLAS/ARPACK environment receive the
same owner start vector and algorithm arguments. Reproducibility does not promise
bit-identical output across different numerical backends or hardware environments.
