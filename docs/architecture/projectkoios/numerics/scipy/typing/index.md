# SciPy typing

## Status

This page specifies the current shared typing boundary for concrete optional SciPy
capabilities.

## Ownership

`projectkoios.numerics.scipy.typing` owns concrete representation aliases and narrow
static interfaces for third-party SciPy objects consumed by Project Koios where upstream
annotations are absent or incomplete. The module currently exports:

- `CsrMatrix`, the concrete union of SciPy `csr_matrix` and `csr_array`;
- `Float64SparseMatrix`, covering the sparse operations used by quadratic capabilities;
- `GaussianKde`, covering density evaluation, integration, log-density, and resampling;
  and
- `OptimizeResult`, covering the result fields inspected after minimization.

`CsrMatrix` identifies the accepted concrete CSR representation family. It does not
statically enforce dtype, shape, nonemptiness, canonical storage, or finiteness; each
consuming capability performs those runtime checks. The structural interfaces describe
only operations consumed from incompletely typed third-party values. Neither form is a
substitute for Project Koios solver contracts, selects a backend, or adds a universal
numerical hierarchy.

## Dependency boundary

Importing `projectkoios.numerics.scipy` alone imports no optional backend. Importing its
`typing` module explicitly imports NumPy and SciPy so it can expose the concrete CSR
union. Concrete capability modules continue to own their numerical SciPy operations, and
package initializers remain import-empty.

Consumers use `CsrMatrix` where the public input must be an actual SciPy CSR matrix or
array. They cast to a narrow structural interface only after constructing a value through
SciPy or checking its representation. This confines unavoidable trust in an incompletely
typed third-party API to one visible boundary instead of spreading `Any` through
numerical code.

## Verification

Mapped integration tests type-check both concrete CSR classes with float64 and complex128
storage and exercise the exact sparse, KDE, and optimization-result operations declared
by the interfaces against installed SciPy objects. Separate unit tests verify that the
package initializer is import-empty. Strict mypy includes both the NumPy and SciPy typing
test trees.
