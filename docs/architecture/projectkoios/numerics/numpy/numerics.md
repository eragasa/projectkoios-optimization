# NumPy numerics

NumPy implementations preserve the explicit dtypes, shapes, axis conventions,
finiteness checks, storage ownership, and deterministic operation ordering required by
their owner contracts.

The dense spectral-norm backend passes validated float64 or complex128 matrix storage
directly to `numpy.linalg.svdvals`. NumPy returns the complete descending singular-value
sequence, and the backend selects index zero. The backend performs no dtype conversion,
normalization, mutation, sparse conversion, or iterative approximation.
