# Finite-selection implementation organization

## Target module

```text
projectkoios/optimization/selection/
├── __init__.py
└── finite.py
```

The capability is dependency-free and requires no backend package.

## Records

`FiniteMinimum` retains:

- selected zero-based exact built-in integer index; and
- selected finite objective value.

`FiniteAmbiguitySet` retains:

- the `FiniteMinimum`;
- nonnegative absolute tolerance; and
- every qualifying zero-based index in declaration order.

Its index tuple is nonempty and strictly increasing and contains the retained minimum
index.

## Validation and failures

- The value inventory must be an exact nonempty tuple.
- Booleans, non-real values, NaN, and infinity fail in values, ceilings, and tolerances.
- Ceilings must be finite; an empty qualifying set returns `()`.
- Absolute tolerances must be finite and nonnegative.
- No epsilon is inserted into exact-tie or ceiling comparisons.

## Source-mirrored tests

```text
tests/projectkoios/optimization/selection/
├── test__FiniteMinimum__unit.py
├── test__FiniteAmbiguitySet__unit.py
└── test__FiniteSelection__unit.py
```

## Verification

- Exact public signatures and return types.
- First-on-exact-tie behavior.
- Equality at a feasibility ceiling.
- Zero-tolerance ambiguity equivalence with all exact minima.
- Multiple candidates inside and outside a nonzero tolerance.
- Stable declaration order.
- Empty qualifying ceiling result.
- Empty, nonfinite, Boolean, and negative-tolerance failures.
- Owner-native M3 regression fixture.

## Status

Finite selection is implemented as part of the M3 constrained-quadratic slice. The
Slice 01 software, bounded numerical, packaging, and independent code-review gate passes.
