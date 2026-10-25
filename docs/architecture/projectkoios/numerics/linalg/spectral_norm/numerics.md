# Dense spectral-norm numerics

The implementation validates the dense representation before calling
`numpy.linalg.svdvals`. NumPy computes the complete singular spectrum, and the first
singular value is the induced Euclidean matrix norm. No tolerance, initial vector,
iteration limit, residual estimate, or fallback algorithm is part of this dense
contract.

One-row, one-column, rank-deficient, and all-zero matrices use the same complete-SVD
path. Real and complex rectangular matrices are supported. The operation reads but does
not mutate or normalize caller storage.

Malformed representation, dtype, shape, or source finiteness raises `TypeError` or
`ValueError`. Exact NumPy `LinAlgError` nonconvergence is translated to `RuntimeError`.
A nonfinite singular-value result raises `FloatingPointError`; infinity is never returned
as a successful norm.

The direct algorithm deliberately does not implement power-of-two scaling for
specialized extreme-range residual diagnostics. Such policy-bearing diagnostics remain
consumer-owned unless separately extracted and reviewed.
