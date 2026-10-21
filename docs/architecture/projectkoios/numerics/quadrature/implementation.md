# Uniform one-dimensional quadrature implementation

## Source layout

```text
projectkoios/numerics/quadrature/
├── __init__.py
├── uniform_1d.py
└── numpy/
    ├── __init__.py
    └── uniform_1d.py
```

Both package initializers are import-empty. The dependency-free module owns the interval,
point-count, rule-identity, version, and convention record. The NumPy module owns
canonical float64/int64 arrays and concrete calculation.

## Immutable storage

`NumpyUniformClosedIntervalTrapezoidalRuleResult.from_array(...)`,
`NumpyHomogeneousDirichletInteriorQuadrature1D.from_arrays(...)`, and
`NumpyDiagonalQuadratureMassMatrix.from_array(...)` defensively copy exact NumPy arrays
into C-contiguous immutable byte-backed storage. Raw record constructors accept only that
canonical storage and form the zero-copy boundary between correlated owner records.

The complete result reconstructs and validates the spacing and every trapezoidal weight.
The interior result reconstructs and validates the exact full-grid index sequence and
its equal spacing weights. A mass record stores only the diagonal, avoiding an implicit
quadratic allocation; `to_dense()` performs an explicit defensive dense expansion.

## PIAB1D adoption

A downstream endpoint-inclusive PIAB1D grid can construct the definition from its
unit-stripped endpoints and full point count, calculate the rule, and request the
homogeneous-Dirichlet restriction. PhysKit remains responsible for wrapping the returned
magnitudes with length units and applying them through its existing sampled-state
normalizer. Numerics does not normalize physical states or eigenvectors.

## Verification

Tests independently verify:

- interval, rule, and convention identity;
- weight sum and constant/linear exactness;
- complete-versus-interior bilinear equality for zero endpoint values;
- diagonal mass-matrix bilinear algebra;
- defensive copying and byte-backed immutability;
- exact interior indexing;
- two-point full-rule support and rejection of its unavailable interior restriction;
- invalid type, finiteness, interval, count, shape, index, and correlation cases; and
- isolated-wheel dependency-free and NumPy imports.
