# `NumpySpectralNorm` implementation

## Source mapping

```text
src/python/projectkoios/numerics/linalg/numpy/spectral_norm.py
└── NumpySpectralNorm
```

The class is final, defines no instance fields beyond the retained `_matrix` slot owned
by `MatrixNormABC`, and inherits the final `result()` method. It implements the
`matrix_norm` property and `_evaluate(matrix)` backend hook.

## Evaluation stages

`_evaluate` executes these stages in order:

1. require exact `numpy.ndarray` runtime class;
2. require exact `float64` or `complex128` dtype;
3. require two nonempty matrix axes;
4. require finite entries;
5. call `numpy.linalg.svdvals` once;
6. translate exact `numpy.linalg.LinAlgError` to `RuntimeError`;
7. reject nonfinite singular-value output with `FloatingPointError`; and
8. convert the first descending singular value to built-in `float`.

`MatrixNormABC.result()` then rejects a non-built-in, nonfinite, or negative scalar.
Unrelated backend exceptions propagate unchanged.

## Storage behavior

The class retains the caller array by reference. Evaluation passes that array directly to
NumPy. It performs no dtype conversion, normalization, caller-storage write, cache
installation, sparse conversion, or fallback execution. C-order, Fortran-order, strided,
writable, and read-only arrays use the same path.

## Mapped verification

```text
tests/projectkoios/numerics/linalg/numpy/spectral_norm/
└── test__NumpySpectralNorm__integration.py
```

The mapped test module verifies analytic values, runtime representation boundaries,
caller ownership, backend failure translation, nonfinite arithmetic, semantic identity,
and exact scalar type. Backend substitution occurs only at the `numpy.linalg.svdvals`
trust boundary for nonconvergence and malformed numerical output.
