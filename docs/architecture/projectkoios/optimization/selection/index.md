# Finite objective selection

`projectkoios.optimization.selection` owns deterministic selection from a finite ordered
inventory of objective values.

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

Consumers own candidate labels, threshold meaning, model-class meaning, and scientific
dispositions.

## Uses

- M3 finite quadratic family selection.
- Appendix-D ordered model-class comparison.
- Appendix-J Stage-B ambiguity-set construction.
- Appendix-J Stage-C ordered hierarchy comparison.

## Pages

- [Schematics](schematics.md)
- [Implementation organization](implementation.md)
- [Optimization namespace](../index.md)
