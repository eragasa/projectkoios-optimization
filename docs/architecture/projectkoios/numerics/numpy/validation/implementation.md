# Implementation

The module exposes four nominal action owners:

- `NumpyFiniteArrayValidator` validates an exact `numpy.ndarray`, its exact requested
  dtype, and elementwise finiteness without copying;
- `NumpyImmutableArrayConstructor` validates the same representation and copies it
  through immutable `bytes` backing;
- `NumpyImmutableArrayValidator` is the zero-copy canonical-storage boundary; and
- `NumpyFiniteNonnegativeFloat64ArrayValidator` validates exact finite,
  nonnegative `float64` arithmetic input without copying.

Each action binds its complete input at construction and exposes zero-argument
`execute()`. The generic owners retain their input array's static array type. Consumers
therefore use the same nominal owners for complex128, float64, int64, bool, and other
exact dtypes instead of function wrappers for individual dtype aliases.

`NumpyImmutableArrayConstructor` produces a C-contiguous, read-only array whose NumPy
base chain terminates in exact built-in `bytes`, so callers cannot re-enable writes with
`setflags`. `NumpyImmutableArrayValidator` requires that same representation and rejects
a merely read-only view over mutable storage.

The actions validate rather than coerce dtype, shape, or values. Specialized validators
may validate without copying when a consumer needs checked arithmetic input rather than
retained immutable ownership.
