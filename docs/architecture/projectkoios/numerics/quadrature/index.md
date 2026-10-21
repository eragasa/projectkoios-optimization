# Uniform one-dimensional quadrature

`projectkoios.numerics.quadrature` owns unit-free numerical quadrature rules. The first
implemented capability is the endpoint-inclusive composite trapezoidal rule on one
uniform closed interval.

## Public operations

- `UniformClosedIntervalTrapezoidalRuleDefinition` fixes exact binary64 endpoints and an
  endpoint-inclusive point count.
- `NumpyUniformClosedIntervalTrapezoidalRule.calculate(...)` returns canonical immutable
  full-grid weights.
- `result.restrict_homogeneous_dirichlet()` returns the explicit full-grid indices
  `1, ..., N - 2` and one spacing weight per interior node.
- Full and interior results expose `NumpyDiagonalQuadratureMassMatrix`, stored by its
  positive diagonal and expandable to a defensive immutable dense array.

The package initializers remain import-empty. Consumers import the exact defining
modules:

```python
from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)
from projectkoios.numerics.quadrature.numpy.uniform_1d import (
    NumpyUniformClosedIntervalTrapezoidalRule,
)
```

## Boundary

The capability owns unit-free nodes-by-convention, weights, the declared
homogeneous-Dirichlet restriction, and diagonal lumped mass algebra. It does not own
physical units, sampled-state or eigenvector normalization, finite-element basis
functions, consistent finite-element mass matrices, generalized eigensolvers,
quadrature-error acceptance, or scientific interpretation.

## Pages

- [Implementation](implementation.md)
- [Numerics](numerics.md)
- [Schematics](schematics.md)
