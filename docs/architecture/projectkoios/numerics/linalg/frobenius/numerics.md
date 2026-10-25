# Frobenius numerics

For every validated entry, each representation path converts the binary64 scalar to a
built-in complex value, evaluates `abs(entry)`, and then evaluates its binary64 square.
It accumulates all powers with `math.fsum` in deterministic logical row-major order. The
norm is exactly the binary64 `sqrt` of the retained power. NumPy arrays are traversed
directly through their logical flat iterator, so contiguous layout and view strides do
not change term order. SciPy CSR values are copied, duplicate coordinates are summed,
explicit zeros are eliminated, and indices are sorted before canonical stored values are
reduced; implicit zeros make no contribution.

The contract distinguishes malformed data from failed finite arithmetic:

- malformed dependency-free tuple input or an optional-backend value with the wrong
  representation, shape, dtype, or finiteness raises `TypeError` or `ValueError`;
- overflow during CSR duplicate consolidation, a squared magnitude, or the accumulated
  power raises `FloatingPointError`;
- infinity is never returned as either public value.

The capability intentionally requires a finite raw power. A matrix whose norm could be
represented by a scaled norm algorithm but whose raw power overflows is therefore
rejected, because returning the raw power is part of this API.

The dependency-free detailed-result constructor independently checks positive
dimensions, finite nonnegative values, and the exact relation
`norm == sqrt(power)`. Optional backend power and norm classes expose those two scalar
quantities separately while using the same ordered reduction. This is deterministic
binary64 software verification, not a claim of exact real-arithmetic summation or
scientific validation.
