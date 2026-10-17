# Objective selection

`projectkoios.optimization.selection` owns deterministic selection from finite ordered
objective inventories and closed-form selection for explicitly supported continuous
families.

## Module

`finite.py` defines:

- `FiniteMinimum`;
- `FiniteAmbiguitySet`;
- `first_minimum(values)`;
- `indices_at_or_below(values, ceiling)`; and
- `absolute_ambiguity_set(values, tolerance)`.

## Public API

```python
@dataclass(frozen=True, slots=True)
class FiniteMinimum:
    index: int
    value: float


@dataclass(frozen=True, slots=True)
class FiniteAmbiguitySet:
    minimum: FiniteMinimum
    tolerance: float
    indices: tuple[int, ...]


def first_minimum(values: tuple[float, ...], /) -> FiniteMinimum: ...


def indices_at_or_below(
    values: tuple[float, ...],
    ceiling: float,
    /,
) -> tuple[int, ...]: ...


def absolute_ambiguity_set(
    values: tuple[float, ...],
    tolerance: float,
    /,
) -> FiniteAmbiguitySet: ...
```

## Contract

- Inputs are exact nonempty tuples of finite real numeric values excluding Boolean
  values.
- Exact minima retain the first index on ties.
- Ceilings are finite real values excluding Boolean values.
- Ceiling filtering includes equality and returns an empty tuple when no value qualifies.
- Absolute tolerances are finite nonnegative real values excluding Boolean values.
- Absolute ambiguity includes every value within the declared nonnegative tolerance of
  the minimum.
- Returned indices preserve declaration order.
- Results retain numeric indices and values only.
- `FiniteMinimum.index` is an exact nonnegative built-in integer and its value is finite.
- `FiniteAmbiguitySet.indices` is nonempty, strictly increasing, contains the minimum
  index, and preserves declaration order.
- Both result records use canonical frozen construction. Dataclass constructors require
  exact stored types and validate without field replacement. No alternate normalization
  constructors are exposed.
- Result validation names fields explicitly and uses neither reflective attribute access
  nor `object.__setattr__`.

Consumers own candidate labels, threshold meaning, model-class meaning, and scientific
dispositions.

## Periodic single-harmonic selection

`harmonic.py` defines a dependency-free closed-form maximum for
`offset + c*cos(order*argument) + s*sin(order*argument)`. It returns the mathematical
maximum and either a canonical argument in one half-open fundamental interval or an
explicit tolerance-qualified numerically-constant status. See the
[periodic-harmonic architecture](periodic_harmonic/index.md).

## Uses

- M3 finite quadratic family selection.
- Appendix-D ordered model-class comparison.
- Appendix-J Stage-B ambiguity-set construction.
- Appendix-J Stage-C ordered hierarchy comparison.
- Continuous maximization of a consumer-derived single harmonic.

## Pages

- [Periodic single-harmonic maximization](periodic_harmonic/index.md)
- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Optimization namespace](../index.md)
