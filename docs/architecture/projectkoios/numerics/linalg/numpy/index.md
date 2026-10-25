# `projectkoios.numerics.linalg.numpy`

This namespace owns concrete NumPy implementations of linear-algebra capabilities. Each
backend class remains under the source module that defines it. Package initializers are
import-empty.

## Source-module documentation

- [`frobenius.py`](../frobenius/index.md) implements dense Frobenius power and norm.
- [`hermitian_spectral_subspace.py`](../hermitian_spectral_subspace/index.md) implements
  dense invariant spectral-subspace decomposition.
- [`hermitian_spectrum.py`](../hermitian_spectrum/index.md) implements dense complete
  Hermitian spectra.
- [`maximum_absolute_entry.py`](../maximum_absolute_entry/index.md) implements the dense
  maximum-absolute-entry norm.
- [`spectral_norm.py`](spectral_norm/index.md) implements the dense spectral norm and
  contains source-mirrored class documentation.

## Boundary

The namespace imports NumPy only inside exact capability modules. It defines no generic
backend registry, eager package export, unit policy, or scientific interpretation.
