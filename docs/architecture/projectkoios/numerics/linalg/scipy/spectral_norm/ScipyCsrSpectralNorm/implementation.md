# `ScipyCsrSpectralNorm` implementation

## Source mapping

```text
src/python/projectkoios/numerics/linalg/scipy/spectral_norm.py
└── ScipyCsrSpectralNorm
```

The final class specializes `MatrixNormABC[CsrMatrix]`, adds one `_controls` slot, and
inherits final scalar validation from `MatrixNormABC.result()`.

## Ordered evaluation

1. `_canonical_binary64_csr` validates representation, dtype, shape, source finiteness,
   and post-duplicate finiteness while preserving caller storage.
2. `_evaluate` validates the exact Krylov boundary against canonical shape and dtype.
3. `_direct_result` handles minimum dimensions one and two, including zero storage,
   through stored-value scaling and a bounded sparse Gram matrix.
4. Larger canonical all-zero storage returns exact `0.0`.
5. `_start_vector` constructs a seeded normalized vector without global random state.
6. `_iterative_result` calls `scipy.sparse.linalg.svds` with fixed owner choices and
   caller controls.
7. Backend array class, shape, finiteness, singular-value sign, and vector unit norms are
   checked.
8. Both singular equations are formed against the exact canonical matrix passed to
   SciPy, normalized, and compared with the owner cutoff.

## Storage behavior

The class never calls `toarray`, `todense`, or `NumpySpectralNorm`. The direct branch
creates one sparse scaled copy and a sparse Gram matrix of order at most two. The
iterative branch creates dense start, singular, matrix-vector-product, and residual
vectors; no dense matrix representation is created. Caller data, indices, index pointers,
canonical flags, and sparse class remain unchanged.

## Failure classification

- malformed public representation or controls: `TypeError` or `ValueError`;
- nonfinite duplicate consolidation, direct arithmetic, backend output, vector structure,
  residual arithmetic, or residual-normalization products: `FloatingPointError`;
- ARPACK/decomposition nonconvergence or excessive finite residual: `RuntimeError`.

Unrelated backend programming errors propagate rather than selecting another algorithm.

## Mapped verification

```text
tests/projectkoios/numerics/linalg/scipy/spectral_norm/
└── test__ScipyCsrSpectralNorm__integration.py
```

Backend substitution is limited to exact-zero short-circuit observation, deterministic
start observation, nonunit output, nonconvergence, nonfinite output, residual arithmetic,
residual-normalization overflow, and inconsistent finite triplets. These substitutions
create no production injection port.
