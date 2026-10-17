# Implementation

The module requires exact `numpy.ndarray` objects and exact requested dtypes. `immutable_array` validates finiteness and copies through immutable byte backing so callers cannot re-enable writes with `setflags`.

Specialized validators may validate without copying when a consumer only needs a checked arithmetic input.
