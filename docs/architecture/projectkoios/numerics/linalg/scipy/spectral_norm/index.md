# `projectkoios.numerics.linalg.scipy.spectral_norm`

The source module
`src/python/projectkoios/numerics/linalg/scipy/spectral_norm.py` owns deterministic CSR
spectral-norm controls and evaluation.

## Owned classes

- [`ScipyCsrSpectralNormControls`](ScipyCsrSpectralNormControls/index.md) is the immutable
  mandatory owner record for tolerance, maximum iterations, Krylov dimension, and seed.
- [`ScipyCsrSpectralNorm`](ScipyCsrSpectralNorm/index.md) validates and privately
  canonicalizes one CSR value, selects its dimension-defined primary numerical branch,
  and returns a qualified scalar largest singular value.

## Public exports

```python
__all__ = ["ScipyCsrSpectralNorm", "ScipyCsrSpectralNormControls"]
```

The module exposes no function facade, factory, representation dispatcher, public
residual evidence, dense conversion, or solver fallback.

## Mapped tests

```text
tests/projectkoios/numerics/linalg/scipy/spectral_norm/
├── test__ScipyCsrSpectralNormControls__unit.py
└── test__ScipyCsrSpectralNorm__integration.py
```

## Related architecture

- [Matrix spectral-norm capability](../../spectral_norm/index.md)
- [SciPy linear-algebra backends](../index.md)
- [Matrix-norm foundation](../../matrix_norm/index.md)
