# Optimization implementation organization

## Capability layout

```text
projectkoios/optimization/
├── least_squares/
├── alignment/
├── quadratic/
├── selection/
├── assignment/
├── algorithms/
├── clustering/
├── engines/
├── multiobjective/
├── probability/
├── sampling/
├── spaces/
└── reports/
```

Existing implemented capabilities remain in place until a bounded compatibility plan
moves them. Protected Pareto source remains under `multiobjective/pareto`.

## Optimization result rule

A public result records both the selected mathematical object and the diagnostics needed
to verify why it was selected. Examples include:

- fitted coefficients and residuals;
- selected transformation and singular diagnostics;
- minimizing point and objective value;
- assignment indices and total or maximum cost;
- selected candidate indices and ambiguity set; and
- sampler state required for deterministic continuation.

## Backend rule

Dependency-free modules define problems, immutable results, and capability-local nominal
abstract solver classes. Concrete subclasses live in capability-local backend packages.
Factories select supported subclasses with closed method enums and lazy imports.
Structural protocols are not used. Optional backends are imported only when their
implementations are constructed or imported directly.

## Numerics dependency

Optimization imports matrix decompositions, geometry, sequences, finite-periodic records,
and numerical diagnostics from `projectkoios.numerics`. Numerics does not import
Optimization. Algorithms that select an optimum remain entirely in this namespace.

## Status

Dense linear least squares, constrained quadratics, and finite selection are implemented.
Alignment, matrix-subspace fitting, assignment, and SciPy-specific least squares remain
planned. Current implemented packages and public paths remain authoritative until each
later migration slice passes its acceptance gate.
