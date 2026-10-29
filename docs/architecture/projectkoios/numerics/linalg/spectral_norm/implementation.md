# Matrix spectral-norm implementation

## Sources

```text
src/python/projectkoios/numerics/linalg/
├── numpy/spectral_norm.py
├── scipy/_csr.py
└── scipy/spectral_norm.py
```

`NumpySpectralNorm` and `ScipyCsrSpectralNorm` are final concrete specializations of
`MatrixNormABC`. Both report `MatrixNormEnum.spectral`; neither hides the other behind a
factory or representation dispatcher.

## Dense stages

`NumpySpectralNorm._evaluate`:

1. validates exact ndarray representation, dtype, shape, emptiness, and finiteness;
2. calls `numpy.linalg.svdvals` once;
3. translates NumPy SVD nonconvergence;
4. rejects nonfinite output; and
5. returns the first descending singular value.

## Sparse stages

`ScipyCsrSpectralNorm._evaluate`:

1. validates the CSR representation and exact binary64 dtype;
2. canonicalizes an independent copy and verifies duplicate consolidation remains
   finite;
3. validates shape- and dtype-dependent Krylov controls;
4. uses a scaled sparse one-by-one or two-by-two Gram calculation when the smaller axis
   has size at most two, including its exact-zero case;
5. returns exact zero for empty canonical data on larger shapes; or
6. constructs a deterministic normalized real or complex start vector and calls
   `scipy.sparse.linalg.svds` with fixed `k=1`, `which="LM"`, and `solver="arpack"`;
7. validates output arrays, finiteness, singular-vector unit norms, and both singular
   equations; and
8. returns the finite nonnegative singular value.

The sparse class catches ARPACK nonconvergence and decomposition nonconvergence. Finite
triplets outside the residual bound fail with `RuntimeError`. Nonfinite backend or
residual arithmetic fails with `FloatingPointError`.

## Shared CSR canonicalization

`scipy/_csr.py` contains one private canonicalization operation used by Frobenius,
maximum-entry, and spectral CSR evaluators. This single operation prevents duplicated CSR
validation and normalization kernels. It is not exported from the package.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/
├── numpy/spectral_norm/test__NumpySpectralNorm__integration.py
└── scipy/spectral_norm/
    ├── test__ScipyCsrSpectralNormControls__unit.py
    └── test__ScipyCsrSpectralNorm__integration.py
```

The sparse tests cover analytic real and complex cases, direct and iterative dimensions,
deterministic complex starts, control validation, pinned SciPy Krylov boundaries, zero
short-circuiting, duplicate consolidation, caller nonmutation, read-only storage,
non-densification in both numerical branches, extreme-range direct arithmetic, large
implicit dimensions, source failures, nonconvergence, nonfinite output, residual
arithmetic and normalization products, unit-vector qualification, and excessive finite
residuals.
