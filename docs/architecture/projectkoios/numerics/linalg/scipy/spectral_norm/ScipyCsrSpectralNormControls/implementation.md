# `ScipyCsrSpectralNormControls` implementation

## Source mapping

```text
src/python/projectkoios/numerics/linalg/scipy/spectral_norm.py
└── ScipyCsrSpectralNormControls
```

The class is a final `dataclass(frozen=True, slots=True)`. `__post_init__` validates every
field without replacing it:

1. tolerance exact type, finiteness, and half-open range;
2. maximum-iteration exact type and positivity;
3. optional Krylov-dimension exact type and positivity; and
4. seed exact type and nonnegativity.

Boolean values fail every integer and float field because validation requires exact
built-in runtime types.

Shape- and dtype-dependent Krylov rules cannot be established by a matrix-independent
record. `ScipyCsrSpectralNorm._evaluate` applies those relational checks against the
canonical matrix before any numerical branch.

## Mapped verification

```text
tests/projectkoios/numerics/linalg/scipy/spectral_norm/
└── test__ScipyCsrSpectralNormControls__unit.py
```

The tests cover full value retention, machine-precision tolerance, explicit SciPy Krylov
policy, Boolean rejection, nonfinite tolerance, range endpoints, positive iteration and
Krylov bounds, and nonnegative seed.
