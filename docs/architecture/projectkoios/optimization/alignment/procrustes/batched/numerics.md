# Batched Procrustes numerics

The NumPy capability uses vectorized complex128 arithmetic and preserves per-item owner
evidence under the existing tolerance record.

- All inputs must be exact `numpy.ndarray` instances with exact `complex128` dtype.
- Real, complex64, object, Boolean, and implicit-coercion inputs are rejected.
- Public definition and result arrays are defensive read-only copies.
- Nonfinite arithmetic fails closed.

## Rank cutoff

The batched solver uses the same additive owner cutoff as the one-pair scalar contract.

$$
\tau_b=\tau_{\mathrm{absolute}}+
\tau_{\mathrm{relative}}\,\sigma_{b,\max}.
$$

- Equality with the cutoff is deficient.
- Every item exposes its actual applied cutoff.
- `full_rank` is explicit and does not depend on consumer interpretation.
- A different downstream `max(absolute, relative)` convention is not encoded upstream.

## Residual evidence

The optimal squared residual is reconstructed dimension-generally from source power,
target power, and the complete singular spectrum.

$$
r_b^2=\lVert A_b\rVert_F^2+\lVert B_b\rVert_F^2
-2\sum_j\sigma_{b,j}.
$$

- Reconstruction scales terms before subtraction.
- Cancellation-scale values are represented as exact zero.
- The applied squared-residual roundoff cutoff remains public per item.
- This avoids reproducing spurious square-root residual floors from unclamped subtraction.

## Backend validation

Vectorized validation correlates NumPy output with the public problem before a result is
returned.

- Left and right singular-vector unitarity are checked per item.
- Cross-matrix reconstruction is checked per item.
- Every SVD-selected completion is checked for right unitarity.
- Direct residuals are checked against reconstructed optimal residuals.
- The first failing leading index is reported in deterministic C order.

## Failure policy

Failures reject the whole batch and never expose partial results, while numerical rank
deficiency remains successful status-bearing evidence.

- Invalid arrays raise `TypeError` or `ValueError` before backend execution.
- Shared or backend nonfinite arithmetic raises `FloatingPointError`.
- SVD nonconvergence raises `RuntimeError`.
- A rare failed batched SVD uses scalar diagnostic retries only to locate the first failing index.
- Empty batch dimensions return shape-preserving empty evidence without invoking SVD.
