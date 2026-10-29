# `projectkoios.numerics.linalg.scipy`

This namespace owns concrete SciPy implementations of linear-algebra capabilities. Each
backend class remains under the source module that defines it. The package initializer is
import-empty.

## Source-module documentation

- [`_csr.py`](_csr/index.md) owns private canonical binary64 CSR normalization shared by
  matrix-norm implementations.
- [`frobenius.py`](../frobenius/index.md) implements CSR Frobenius power and norm.
- [`hermitian_spectrum.py`](../hermitian_spectrum/index.md) implements complete batched
  Hermitian spectra.
- [`maximum_absolute_entry.py`](../maximum_absolute_entry/index.md) implements the CSR
  maximum-absolute-entry norm.
- [`sparse_hermitian_spectrum.py`](../sparse_hermitian_spectrum/index.md) implements
  partial sparse Hermitian spectra.
- [`spectral_norm.py`](spectral_norm/index.md) implements deterministic CSR spectral-norm
  evaluation and owns source-mirrored class documentation.

## Boundary

The namespace imports SciPy only inside exact capability modules. It defines no eager
package export, unit policy, scientific interpretation, or cross-representation fallback.
