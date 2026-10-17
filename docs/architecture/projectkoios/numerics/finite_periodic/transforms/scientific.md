# Centered uniform transform scientific boundary

## Numerical claim

The implementation claims only a deterministic finite Fourier convention over reduced
coordinates and integer modes, together with represented reconstruction, Parseval, and
coefficient-Hermiticity evidence.

## Geometry boundary

No lattice metric appears in the API or formulas. Nonorthogonal direct lattices are
supported because reduced-coordinate duality removes Cartesian geometry from the phase.
Numerics owns only `exp(±2πi q·n)`; it does not define physical reciprocal vectors or
select angular versus cycles-per-length conventions. Consumers validate the
physical direct/reciprocal primitive relation, declare the convention, and use a
physical metric when defining distances or shells.

An oblique cell, an FCC primitive cell, and an orthogonal cell therefore use identical
Numerics records once their points are expressed in reduced reciprocal coordinates.
This statement does not cover a nonorthogonal retained-state or orbital overlap metric.

## Consumer-owned interpretation

PhysKit may attach lattice records, units, and physical hopping semantics. KSD may own:

- complete-shell policies and shell metrics;
- onsite/non-onsite classifications;
- energy-reference or rank normalization;
- interpolation, truncation, and total-error channels;
- numerical thresholds; and
- scientific dispositions and human acceptance.

Those policies must not be inferred by the transform or diagnostic analyzers.

## Verification versus validation

- Round-trip tests verify the declared finite pair.
- Parseval tests verify normalization and represented arithmetic.
- cyclic Hermiticity tests verify origin-aware coefficient pairing.
- oblique and FCC tests verify geometry-independent phase representation.

None of these tests validates a material model, gauge choice, physical truncation,
locality claim, or continuum limit.
