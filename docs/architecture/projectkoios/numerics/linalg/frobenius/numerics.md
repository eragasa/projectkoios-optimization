# Frobenius numerics

For every validated entry, the implementation evaluates `abs(entry)` and then its
binary64 square. It accumulates all powers with `math.fsum` in deterministic row-major
order. The norm is exactly the binary64 `sqrt` of the retained power.

The contract distinguishes malformed data from failed finite arithmetic:

- non-tuple, empty, ragged, non-complex, or nonfinite input raises `TypeError` or
  `ValueError`;
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
