# Numerics

Finite-float normalization accepts non-Boolean `numbers.Real` values and produces finite built-in binary64 values. Exact finite-float validation retains signed values without coercion; nonnegative and relative validators add their range relations. Exact finite-float tuple validation requires a nonempty exact tuple and retains its canonical immutable storage without copying. Exact validators reject implicit coercion. Relative values occupy `[0, 1)`. Unsigned bounds are the exact u32 and u64 integer intervals.

Arithmetic-result validators distinguish failed numerical arithmetic with `FloatingPointError` from malformed public values.
