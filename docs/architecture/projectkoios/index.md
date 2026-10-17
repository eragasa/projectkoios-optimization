# Project Koios mathematical numerics and optimization

This repository owns two public Python namespaces in the
`projectkoios-optimization` distribution:

- `projectkoios.numerics` owns reusable transformations, decompositions, mathematical
  geometry, and numerical analysis; and
- `projectkoios.optimization` owns every reusable capability whose public contract
  selects variables, parameters, candidates, assignments, or transformations by
  optimizing one or more functions.

Physical and scientific interpretation is the responsibility of each consumer.

## Ownership test

A capability belongs to Optimization when the public operation answers a question such
as:

- Which parameters minimize this residual?
- Which transformation best aligns these inputs?
- Which candidate has the best objective value?
- Which assignment minimizes this cost?
- Which fitted model maximizes this likelihood?

A capability belongs to Numerics when the public operation answers a question such as:

- What is this matrix decomposition?
- What is the transformed representation?
- What sewing map identifies these finite bases?
- What is the Wilson product or phase spectrum?
- What norm, derivative, convergence order, or root was computed?

Internal use of linear algebra does not change an optimization problem into a numerical
transformation. Mathematical equivalence to a variational principle does not change a
decomposition API into an optimizer.

## Dependency rule

- Optimization may import Numerics.
- Numerics never imports Optimization.
- Consumers may import either namespace.
- Neither owned namespace imports a consumer.
- Workflow execution remains outside both namespaces.

## Architecture pages

- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [`projectkoios.numerics`](numerics/index.md)
- [`projectkoios.optimization`](optimization/index.md)
- [Architecture templates](templates/index.md)
