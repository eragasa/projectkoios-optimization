# Frobenius numerics

For every validated entry, each representation path converts the binary64 scalar to a
built-in complex value, evaluates `abs(entry)`, and then evaluates its binary64 square.
It accumulates all powers with `math.fsum` in deterministic logical row-major order. The
norm is exactly the binary64 `sqrt` of the retained power. NumPy arrays are traversed
directly through their logical flat iterator, so contiguous layout and view strides do
not change term order.

The contract distinguishes malformed data from failed finite arithmetic:

- malformed dependency-free tuple input or a NumPy value with the wrong representation,
  shape, dtype, or finiteness raises `TypeError` or `ValueError`;
- overflow of a squared magnitude or the accumulated power raises
  `FloatingPointError`;
- infinity is never returned as either public value.

The capability intentionally requires a finite raw power. A matrix whose norm could be
represented by a scaled norm algorithm but whose raw power overflows is therefore
rejected, because returning the raw power is part of this API.

The result constructor independently checks positive dimensions, finite nonnegative
values, and the exact relation `norm == sqrt(power)`. This is deterministic binary64
software verification, not a claim of exact real-arithmetic summation or scientific
validation.
