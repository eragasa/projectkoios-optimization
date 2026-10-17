# Periodic harmonic implementation

## Source organization

```text
src/python/projectkoios/optimization/selection/
├── __init__.py
├── finite.py
└── harmonic.py
```

The package initializer remains import-empty. The harmonic implementation uses only the
Python standard library and shared dependency-free scalar validators.

## Records and algorithm

`PeriodicHarmonic.from_values(...)` normalizes finite real coefficients and the interval
start before frozen construction. Its raw constructor accepts canonical exact built-in
floats and validates them without postconstruction replacement. It also rejects an
interval whose positive mathematical period cannot be represented as a distinct binary64
endpoint at the declared start.

`PeriodicHarmonicMaximum` is an immutable evidence record. `from_values(...)` performs
explicit real-to-binary64 normalization before frozen construction. The raw constructor
accepts canonical exact built-in floats and independently checks amplitude, maximum value,
fundamental period, status, degeneracy cutoff, and the canonical argument/status
relationship against the retained harmonic without postconstruction mutation. Named
value, relational, and canonical-argument validators own those separate checks.

`PeriodicHarmonicMaximizer` validates the two tolerance controls at construction. Its
final `maximize` method performs the `hypot`, `atan2`, and half-open-interval reduction
specified in [the numerical contract](numerics.md). There is no backend extension slot
because the exact closed-form algorithm is the capability.

## Tests

Mapped tests live under:

```text
tests/projectkoios/optimization/selection/
├── conftest.py
├── fixtures/
│   └── periodic_harmonic.py
├── test__PeriodicHarmonic__unit.py
├── test__PeriodicHarmonicMaximum__unit.py
├── test__PeriodicHarmonicMaximumValidation__unit.py
├── test__PeriodicHarmonicMaximizer__unit.py
├── test__PeriodicHarmonicMaximizer__numerical.py
└── test__PeriodicHarmonicWheel__integration.py
```

They use a named Pythagorean harmonic case with owner-derived maximum evidence to
directly exercise each named maximum-validation owner and immutable factory, in addition
to covering input validation, public-result relational validation, arbitrary harmonic
orders and interval starts, boundary wrapping, signed-zero phases, numerical degeneracy,
nonfinite arithmetic, repeatability, dense independent grid comparison, the rank-two
second-harmonic consumer shape, and isolated dependency-free wheel execution. Strict
mypy includes the complete selection test directory.

## Verification scope

Passing tests establish dependency-free software behavior and bounded binary64 numerical
agreement only. They do not validate a consumer's coefficient construction, physical
model, threshold, continuous-field quadrature, or scientific interpretation.
