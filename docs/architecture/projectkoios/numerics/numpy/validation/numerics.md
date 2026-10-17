# Numerics

Finiteness is evaluated elementwise with `numpy.isfinite`. Nonfinite-copy failures identify the first offending multidimensional index. Float64 nonnegative validation rejects NaN, infinities, and negative entries without changing shape or values.

No tolerance or domain interpretation is introduced.
