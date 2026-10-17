"""Backend-neutral contracts for evaluating finite Fourier series.

For real coordinates ``x_i``, integer modes ``m_j``, and complex coefficients
``c_j``, this capability evaluates the ordered sum

``y_i = sum_j c_j * exp(+2*pi*1j*dot(x_i, m_j))``.

The positive exponent and absence of normalization are fixed parts of the public
contract. Points, terms, and coordinate components retain their supplied order.
This module intentionally imports no numerical backend; NumPy is loaded only when the
built-in NumPy evaluator is constructed through the factory.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import final

# Integer modes are stored as ordinary Python integers at the public boundary. The
# backend converts them to binary64, so the tighter 2**53 bound guarantees that the
# conversion itself is exact. The signed-64-bit check also keeps the serialized public
# domain explicit even though the binary64-exactness bound is narrower.
_MAX_EXACT_BINARY64_INTEGER = 2**53
_SIGNED_INT64_MINIMUM = -(2**63)
_SIGNED_INT64_MAXIMUM = 2**63 - 1

type FiniteFourierCoordinates = tuple[tuple[float, ...], ...]
type FiniteFourierIntegerModes = tuple[tuple[int, ...], ...]
type FiniteFourierCoefficients = tuple[complex, ...]


def _exact_positive_integer(value: int, name: str, /) -> int:
    """Validate a positive built-in integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _exact_nonnegative_integer(value: int, name: str, /) -> int:
    """Validate a nonnegative built-in integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact built-in integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _coordinates(
    value: FiniteFourierCoordinates,
    dimension: int,
    /,
) -> FiniteFourierCoordinates:
    """Validate ordered real coordinates against the explicit dimension."""
    if type(value) is not tuple:
        raise TypeError("coordinates must be an exact tuple")
    normalized: list[tuple[float, ...]] = []
    for row in value:
        if type(row) is not tuple:
            raise TypeError("coordinate rows must be exact tuples")
        if len(row) != dimension:
            raise ValueError("coordinate rows must match dimension")
        entries: list[float] = []
        for entry in row:
            if type(entry) is not float:
                raise TypeError("coordinate entries must be exact built-in floats")
            if not math.isfinite(entry):
                raise ValueError("coordinate entries must be finite")
            entries.append(entry)
        normalized.append(tuple(entries))
    return tuple(normalized)


def _integer_modes(
    value: FiniteFourierIntegerModes,
    dimension: int,
    /,
) -> FiniteFourierIntegerModes:
    """Validate ordered integer modes and their exact-conversion range."""
    if type(value) is not tuple:
        raise TypeError("integer_modes must be an exact tuple")
    normalized: list[tuple[int, ...]] = []
    for row in value:
        if type(row) is not tuple:
            raise TypeError("integer mode rows must be exact tuples")
        if len(row) != dimension:
            raise ValueError("integer mode rows must match dimension")
        entries: list[int] = []
        for entry in row:
            if type(entry) is not int:
                raise TypeError("integer mode entries must be exact built-in integers")
            if not _SIGNED_INT64_MINIMUM <= entry <= _SIGNED_INT64_MAXIMUM:
                raise ValueError("integer mode entries must fit signed 64-bit range")
            if abs(entry) > _MAX_EXACT_BINARY64_INTEGER:
                raise ValueError("integer mode magnitudes must not exceed 2**53")
            entries.append(entry)
        normalized.append(tuple(entries))
    return tuple(normalized)


def _complex_values(
    value: tuple[complex, ...],
    name: str,
    /,
) -> tuple[complex, ...]:
    """Validate a named tuple of finite exact built-in complex values."""
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    normalized: list[complex] = []
    for entry in value:
        if type(entry) is not complex:
            raise TypeError(f"{name} must contain exact built-in complex values")
        if not math.isfinite(entry.real) or not math.isfinite(entry.imag):
            raise ValueError(f"{name} must be finite")
        normalized.append(entry)
    return tuple(normalized)


@final
@dataclass(frozen=True, slots=True)
class FiniteFourierSeriesEvaluationDefinition:
    """Define one ordered positive-exponent finite Fourier-series evaluation.

    ``dimension`` is explicit so empty point and term collections remain unambiguous.
    Each coordinate and mode row must have exactly that width. ``coefficients[j]`` is
    paired with ``integer_modes[j]``; duplicate modes remain separate ordered terms.

    The record accepts exact built-in tuples and scalar types only. It is intentionally
    independent of NumPy arrays, consumer units, lattice records, and scientific labels.
    """

    dimension: int
    coordinates: FiniteFourierCoordinates
    integer_modes: FiniteFourierIntegerModes
    coefficients: FiniteFourierCoefficients

    def __post_init__(self) -> None:
        """Reject malformed, nonfinite, or ambiguously typed public inputs."""
        dimension = _exact_positive_integer(self.dimension, "dimension")
        _coordinates(self.coordinates, dimension)
        _integer_modes(self.integer_modes, dimension)
        _complex_values(self.coefficients, "coefficients")
        if len(self.coefficients) != len(self.integer_modes):
            raise ValueError("coefficients must contain one value per integer mode")

    @property
    def point_count(self) -> int:
        """Return the number of ordered evaluation points."""
        return len(self.coordinates)

    @property
    def term_count(self) -> int:
        """Return the number of ordered series terms."""
        return len(self.integer_modes)


@final
@dataclass(frozen=True, slots=True)
class FiniteFourierSeriesEvaluationResult:
    """Record ordered values and structural metadata for one evaluation.

    The metadata establishes only structural correlation with a definition. Numerical
    correspondence remains the evaluator implementation's responsibility and is checked
    by independent numerical tests.
    """

    dimension: int
    point_count: int
    term_count: int
    values: tuple[complex, ...]

    def __post_init__(self) -> None:
        """Validate dependency-free result shape and finite values."""
        _exact_positive_integer(self.dimension, "dimension")
        point_count = _exact_nonnegative_integer(self.point_count, "point_count")
        _exact_nonnegative_integer(self.term_count, "term_count")
        _complex_values(self.values, "values")
        if len(self.values) != point_count:
            raise ValueError("values length must equal point_count")


class FiniteFourierSeriesEvaluator(ABC):
    """Nominal SDK boundary for positive-exponent finite Fourier series.

    Subclasses implement only :meth:`_evaluate_validated`. The final public template
    method owns exact input typing, backend-independent empty cases, and result
    correlation. External backends opt in through nominal inheritance;
    structural duck typing is not part of this contract.
    """

    __slots__ = ()

    @final
    def evaluate(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult:
        """Evaluate a definition and return values in supplied point order.

        Empty point and empty term collections are resolved without loading or invoking
        an optional numerical backend. For nontrivial definitions, undeclared backend
        exceptions propagate unchanged. Malformed backend records raise ``TypeError`` or
        ``ValueError``.
        """
        if type(definition) is not FiniteFourierSeriesEvaluationDefinition:
            raise TypeError(
                "definition must be FiniteFourierSeriesEvaluationDefinition"
            )
        # Resolve mathematically exact boundary cases in the dependency-free template.
        if definition.point_count == 0:
            return FiniteFourierSeriesEvaluationResult(
                definition.dimension,
                0,
                definition.term_count,
                (),
            )
        if definition.term_count == 0:
            return FiniteFourierSeriesEvaluationResult(
                definition.dimension,
                definition.point_count,
                0,
                (0.0 + 0.0j,) * definition.point_count,
            )

        result = self._evaluate_validated(definition)

        # Correlate the already validated immutable result with its definition.
        if type(result) is not FiniteFourierSeriesEvaluationResult:
            raise TypeError("backend must return FiniteFourierSeriesEvaluationResult")
        if result.dimension != definition.dimension:
            raise ValueError("backend result dimension is inconsistent")
        if result.point_count != definition.point_count:
            raise ValueError("backend result point_count is inconsistent")
        if result.term_count != definition.term_count:
            raise ValueError("backend result term_count is inconsistent")
        return result

    @abstractmethod
    def _evaluate_validated(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult:
        """Evaluate a definition whose point and term collections are nonempty.

        Implementations receive an already validated immutable definition. They must
        preserve point, term, and component order and return the exact public result
        type. Optional-backend-specific programming failures must not be broadly
        wrapped.
        """
        raise NotImplementedError


class FiniteFourierSeriesMethod(StrEnum):
    """Closed selection of owner-provided finite Fourier-series implementations."""

    NUMPY_DENSE = "numpy-dense"


@final
class FiniteFourierSeriesEvaluatorFactory:
    """Construct an owner-provided evaluator without eager backend imports.

    External evaluator subclasses are injected directly by consumers rather than added
    through mutable registration or automatic discovery.
    """

    __slots__ = ()

    @staticmethod
    def create(
        method: FiniteFourierSeriesMethod,
        /,
    ) -> FiniteFourierSeriesEvaluator:
        """Construct the exact requested built-in evaluator.

        Importing this module does not import NumPy. Selecting ``NUMPY_DENSE`` is the
        first operation that imports the NumPy backend module.
        """
        if type(method) is not FiniteFourierSeriesMethod:
            raise TypeError("method must be FiniteFourierSeriesMethod")
        if method is FiniteFourierSeriesMethod.NUMPY_DENSE:
            from .numpy.finite_series import NumpyFiniteFourierSeriesEvaluator

            return NumpyFiniteFourierSeriesEvaluator()
        raise AssertionError("unhandled finite Fourier-series method")


__all__ = [
    "FiniteFourierCoefficients",
    "FiniteFourierCoordinates",
    "FiniteFourierIntegerModes",
    "FiniteFourierSeriesEvaluationDefinition",
    "FiniteFourierSeriesEvaluationResult",
    "FiniteFourierSeriesEvaluator",
    "FiniteFourierSeriesEvaluatorFactory",
    "FiniteFourierSeriesMethod",
]
