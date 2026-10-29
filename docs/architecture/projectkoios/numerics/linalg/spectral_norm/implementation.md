# Dense spectral-norm implementation

## Source

```text
src/python/projectkoios/numerics/linalg/numpy/
└── spectral_norm.py
```

`NumpySpectralNorm` is final and specializes `MatrixNormABC[DenseMatrix]`. The inherited
constructor retains the supplied array reference. `_evaluate` performs the following
ordered stages:

1. validate exact ndarray representation and exact binary64 dtype;
2. validate two-dimensional nonempty shape;
3. validate finite source entries;
4. call `numpy.linalg.svdvals` once under local floating-warning control;
5. translate exact NumPy SVD nonconvergence;
6. reject nonfinite singular-value output; and
7. return the first descending singular value as a built-in `float`.

`MatrixNormABC.result()` then enforces exact built-in type, finiteness, and nonnegativity.
The module contains no function facade, helper layer, factory, dispatcher, class method,
static method, cache, scaling branch, sparse branch, or result record. The NumPy package
initializer remains import-empty.

The module, class, property, and evaluation method contain NumPy-style ownership,
parameter, return, failure, and numerical-boundary documentation. Inline comments state
the runtime limits of static NumPy aliases, floating-warning policy, exception translation
boundary, and descending singular-value selection.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/numpy/spectral_norm/
└── test__NumpySpectralNorm__integration.py
```

Every test function documents its public invariant and input role. Tests cover:

- analytically derived real, complex, rectangular, nonnormal, rank-deficient, one-row,
  one-column, and zero cases;
- an explicit dimension- and scale-dependent binary64 error bound;
- C-order, Fortran-order, strided, writable, and read-only storage;
- caller-value and writeability preservation;
- distinct dimensionality, empty-shape, dtype, and source-finiteness failures;
- finite input with an unrepresentable binary64 norm;
- exact NumPy SVD nonconvergence translation;
- nonfinite backend-output rejection;
- `MatrixNormEnum.spectral` identity; and
- exact built-in scalar result type.

The two backend substitutions are scoped test fixtures for nondeterministic failure
paths. They require no production hook, fallback, environment branch, or compatibility
layer.
