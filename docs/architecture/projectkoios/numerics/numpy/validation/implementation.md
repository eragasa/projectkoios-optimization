# Implementation

The module requires exact `numpy.ndarray` objects and exact requested dtypes. `immutable_array` validates finiteness and copies through immutable bytes backing so callers cannot re-enable writes with `setflags`.

`validate_immutable_array` is the zero-copy canonical-storage boundary. It requires C-contiguous, read-only storage whose NumPy base chain terminates in exact built-in `bytes`; a merely read-only view over mutable storage is rejected.

Specialized validators may validate without copying when a consumer only needs checked arithmetic input rather than retained immutable ownership.
