# Numerics

Finite-float normalization accepts non-Boolean `numbers.Real` values and produces finite built-in binary64 values. Exact validators reject implicit coercion. Relative values occupy `[0, 1)`. Unsigned bounds are the exact u32 and u64 integer intervals.

Arithmetic-result validators distinguish failed numerical arithmetic with `FloatingPointError` from malformed public values.
