# Definition numerics

The definition admits exact finite complex128 arrays with common shape `(..., m, n)`, at
least one leading dimension, and positive matrix dimensions. Zero-sized leading batch
dimensions are allowed.

It performs no Procrustes arithmetic. Its numerical responsibility is limited to
preserving exact values, dimensions, dtype, and tolerance without implicit coercion.
