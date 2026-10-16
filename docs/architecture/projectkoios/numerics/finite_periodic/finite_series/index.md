# Finite Fourier-series evaluation

## Status

This directory documents the current architecture of the implemented
`projectkoios.numerics.finite_periodic.finite_series` slice. The source includes the
backend-neutral records and evaluator, a lazy factory, and the initial ordered NumPy
backend. These pages describe that implementation; they do not authorize commit, push,
release, publication, or downstream dependency binding.

The capability evaluates one finite Fourier series at an ordered set of real
coordinates:

$$
f(x_i)=\sum_{j=0}^{T-1} c_j\exp\!\left(+2\pi i\,x_i\cdot m_j\right).
$$

where each $x_i$ is a real coordinate vector, each $m_j$ is an integer mode vector, and
each $c_j$ is a complex coefficient. The positive exponent, absence of normalization,
coordinate order, mode order, and coefficient order are part of the contract.

## Public-operation boundary

The public operation is **finite Fourier-series evaluation**, also describable as the
evaluation of a finite trigonometric polynomial. It is not named a discrete Fourier
transform because the coordinates need not form a uniform mesh, the modes need not form
a complete reciprocal index set, and no forward/inverse transform pair is implied.

The capability owns:

- dependency-free unit-free input and result records;
- validation of dimension, shapes, finite values, and bounded integer representation;
- the fixed positive-exponent convention;
- deterministic preservation of supplied point, coordinate, mode, and coefficient order;
- direct evaluation without mode inference, symmetry reduction, or normalization;
- explicit empty-point and empty-term behavior;
- bounded failure semantics and backend-result conformance checks; and
- a lazy built-in NumPy implementation behind a nominal SDK boundary.

It does not own:

- physical reciprocal or direct coordinates;
- Bloch, hopping, orbital, gauge, lattice, or energy-unit meaning;
- selection or canonicalization of an integer representative domain;
- Hermiticity or model-symmetry acceptance;
- fitting, loss construction, parameter selection, or optimization;
- experiment meshes, thresholds, admissibility, certificates, or interpretation; or
- a general forward/inverse discrete Fourier-transform API.

## Public vocabulary

The implemented source uses capability-specific names rather than a universal transform
or evaluator base:

- `FiniteFourierSeriesEvaluationDefinition` — immutable unit-free series and evaluation points;
- `FiniteFourierSeriesEvaluationResult` — immutable ordered complex values and correlated sizes;
- `FiniteFourierSeriesEvaluator` — nominal abstract SDK boundary;
- `FiniteFourierSeriesMethod` — closed built-in method enum;
- `FiniteFourierSeriesEvaluatorFactory` — lazy built-in constructor; and
- `NumpyFiniteFourierSeriesEvaluator` — initial dense binary64/complex128 backend.

The defining evaluator contract fixes the positive exponential. A future negative-sign
operation or a forward/inverse transform pair requires separate architecture review; it
must not be added through an unbounded sign flag.

## Dependency position

`projectkoios.numerics` is upstream of Optimization, PhysKit, and KSD. Consumers may use
this contract directly when they need only the unit-free mathematical operation.
PhysKit may wrap it in a physics-semantic scalar-hopping evaluator and attach units and
model identity, but PhysKit is not a required facade or re-export layer. Numerics never
imports PhysKit, Optimization, or KSD.

## Qualification gates

An owner revision containing this capability is not ready for downstream binding until
review and verification confirm:

1. the public names and exact dependency-free record fields;
2. the positive-exponent and no-normalization convention;
3. zero-dimension rejection and empty collection behavior;
4. integer-mode representation limits and nonfinite phase behavior;
5. the initial backend summation order and reproducibility claim;
6. nominal subclass, lazy factory, and closed failure semantics;
7. high-precision numerical criteria and adversarial contract tests; and
8. isolated wheel behavior with NumPy remaining lazy.

Commit, push, release, downstream dependency binding, and publication remain separate
actions requiring their own authorization.

## Pages

- [Scientific and interpretation boundary](science.md)
- [Numerical contract](numerics.md)
- [Implementation organization](implementation.md)
- [Schematics](schematics.md)
- [Distinct shared-grid batched finite-series evaluation](../batched_finite_series/index.md)
