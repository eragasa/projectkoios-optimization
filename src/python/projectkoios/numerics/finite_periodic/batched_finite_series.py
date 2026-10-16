"""Backend-neutral contracts for shared-grid batched finite Fourier series.

For real coordinates ``x_i``, integer modes ``m_j``, and ordered coefficient
channels ``c_{a,j}``, this capability evaluates

``y_{i,a} = sum_j c_{a,j} * exp(+2*pi*1j*dot(x_i, m_j))``.

Public values use point-major shape ``P x S``: point row ``i`` and channel column ``a``.
All channels share one ordered coordinate grid and one ordered mode inventory. Channel
and point order are public structure, while floating-point reduction order is owned by
the selected backend. This contract is therefore distinct from the scalar ordered
finite-series capability whose scalar accumulation order is observable behavior.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from .finite_series import (
    FiniteFourierCoordinates,
    FiniteFourierIntegerModes,
    _complex_values,
    _coordinates,
    _exact_nonnegative_integer,
    _exact_positive_integer,
    _integer_modes,
)

type FiniteFourierCoefficientChannels = tuple[tuple[complex, ...], ...]
type BatchedFiniteFourierValues = tuple[tuple[complex, ...], ...]


def _coefficient_channels(
    value: object,
    term_count: int,
    /,
) -> FiniteFourierCoefficientChannels:
    """Normalize ordered finite coefficient channels against a shared term count.

    Parameters
    ----------
    value : object
        Candidate exact tuple of coefficient tuples.
    term_count : int
        Required number of coefficients in every channel.

    Returns
    -------
    tuple of tuple of complex
        Immutable built-in coefficient channels in supplied order.
    """
    if type(value) is not tuple:
        raise TypeError("coefficient_channels must be an exact tuple")
    channels: list[tuple[complex, ...]] = []
    for channel in value:
        coefficients = _complex_values(channel, "coefficient channels")
        if len(coefficients) != term_count:
            raise ValueError(
                "each coefficient channel must contain one value per integer mode"
            )
        channels.append(coefficients)
    return tuple(channels)


def _point_values(
    value: object,
    point_count: int,
    channel_count: int,
    /,
) -> BatchedFiniteFourierValues:
    """Normalize point-major finite values against declared result shape.

    Parameters
    ----------
    value : object
        Candidate exact tuple of point value tuples.
    point_count : int
        Required number of point rows.
    channel_count : int
        Required number of channel values per point.

    Returns
    -------
    tuple of tuple of complex
        Immutable point-major values in supplied order.
    """
    if type(value) is not tuple:
        raise TypeError("values must be an exact tuple")
    if len(value) != point_count:
        raise ValueError("values length must equal point_count")
    points: list[tuple[complex, ...]] = []
    for point in value:
        values = _complex_values(point, "point values")
        if len(values) != channel_count:
            raise ValueError("each point value row length must equal channel_count")
        points.append(values)
    return tuple(points)


def _backend_evidence(
    identifier: object,
    version: object,
    /,
) -> tuple[str | None, str | None]:
    """Validate paired optional backend evidence.

    Parameters
    ----------
    identifier : object
        Backend operation identifier or ``None``.
    version : object
        Backend version or ``None``.

    Returns
    -------
    tuple of (str or None, str or None)
        Validated paired backend evidence.
    """
    if identifier is None and version is None:
        return None, None
    if type(identifier) is not str or not identifier:
        raise TypeError("backend_identifier must be a nonempty exact string or None")
    if type(version) is not str or not version:
        raise TypeError("backend_version must be a nonempty exact string or None")
    return identifier, version


@final
@dataclass(frozen=True, slots=True)
class BatchedFiniteFourierSeriesEvaluationDefinition:
    """Define ordered coefficient channels on one shared finite Fourier grid.

    Parameters
    ----------
    dimension : int
        Positive coordinate and mode width, retained for empty collections.
    coordinates : tuple of tuple of float
        Ordered finite real evaluation coordinates.
    integer_modes : tuple of tuple of int
        Ordered modes shared by every coefficient channel.
    coefficient_channels : tuple of tuple of complex
        Ordered channels, each with one coefficient per shared mode.

    Notes
    -----
    Empty coordinate, mode, and channel collections are explicit and unambiguous because
    ``dimension`` and all other collection counts remain available independently.
    """

    dimension: int
    coordinates: FiniteFourierCoordinates
    integer_modes: FiniteFourierIntegerModes
    coefficient_channels: FiniteFourierCoefficientChannels

    def __post_init__(self) -> None:
        """Validate and normalize all dependency-free public inputs."""
        dimension = _exact_positive_integer(self.dimension, "dimension")
        coordinates = _coordinates(self.coordinates, dimension)
        integer_modes = _integer_modes(self.integer_modes, dimension)
        channels = _coefficient_channels(
            self.coefficient_channels,
            len(integer_modes),
        )
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "integer_modes", integer_modes)
        object.__setattr__(self, "coefficient_channels", channels)

    @property
    def point_count(self) -> int:
        """Return the number of ordered evaluation points."""
        return len(self.coordinates)

    @property
    def term_count(self) -> int:
        """Return the number of shared ordered modes."""
        return len(self.integer_modes)

    @property
    def channel_count(self) -> int:
        """Return the number of ordered coefficient channels."""
        return len(self.coefficient_channels)


@final
@dataclass(frozen=True, slots=True)
class BatchedFiniteFourierSeriesEvaluationResult:
    """Record point-major values, structural metadata, and backend evidence.

    Parameters
    ----------
    dimension : int
        Coordinate and mode width.
    point_count : int
        Number of point rows.
    term_count : int
        Number of shared modes represented by each channel.
    channel_count : int
        Number of channel columns in every point row.
    values : tuple of tuple of complex
        Evaluated values in point-major, then channel-major order.
    backend_identifier : str or None
        Numerical backend operation, absent for template-resolved empty cases.
    backend_version : str or None
        Numerical backend version, absent exactly when the identifier is absent.
    """

    dimension: int
    point_count: int
    term_count: int
    channel_count: int
    values: BatchedFiniteFourierValues
    backend_identifier: str | None
    backend_version: str | None

    def __post_init__(self) -> None:
        """Validate result shape, finite values, and paired backend evidence."""
        dimension = _exact_positive_integer(self.dimension, "dimension")
        point_count = _exact_nonnegative_integer(self.point_count, "point_count")
        term_count = _exact_nonnegative_integer(self.term_count, "term_count")
        channel_count = _exact_nonnegative_integer(self.channel_count, "channel_count")
        values = _point_values(self.values, point_count, channel_count)
        backend_identifier, backend_version = _backend_evidence(
            self.backend_identifier,
            self.backend_version,
        )
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "term_count", term_count)
        object.__setattr__(self, "channel_count", channel_count)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "backend_identifier", backend_identifier)
        object.__setattr__(self, "backend_version", backend_version)


class BatchedFiniteFourierSeriesEvaluator(ABC):
    """Nominal SDK boundary for shared-grid batched finite Fourier series.

    Subclasses implement only :meth:`_evaluate_validated`. The final public method owns
    exact input typing, backend-independent empty cases, structural correlation, finite
    output validation, and backend-evidence requirements.
    """

    __slots__ = ()

    @final
    def evaluate(
        self,
        definition: BatchedFiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> BatchedFiniteFourierSeriesEvaluationResult:
        """Evaluate all channels on their shared coordinate and mode grid.

        Parameters
        ----------
        definition : BatchedFiniteFourierSeriesEvaluationDefinition
            Validated dependency-free shared-grid definition.

        Returns
        -------
        BatchedFiniteFourierSeriesEvaluationResult
            Point-major values and correlated backend evidence.

        Raises
        ------
        TypeError
            If the definition or backend record has the wrong nominal type.
        ValueError
            If backend structural metadata is inconsistent or evidence is absent.
        FloatingPointError
            If a forged backend result contains nonfinite values.

        Notes
        -----
        Empty channels, empty coordinates, and empty modes bypass the backend. Empty
        modes produce an exact complex zero for every declared channel and point.
        """
        if type(definition) is not BatchedFiniteFourierSeriesEvaluationDefinition:
            raise TypeError(
                "definition must be BatchedFiniteFourierSeriesEvaluationDefinition"
            )
        if definition.point_count == 0:
            return self._trivial_result(definition, ())
        if definition.channel_count == 0:
            return self._trivial_result(
                definition,
                ((),) * definition.point_count,
            )
        if definition.term_count == 0:
            zeros = (0.0 + 0.0j,) * definition.channel_count
            return self._trivial_result(
                definition,
                (zeros,) * definition.point_count,
            )

        result = self._evaluate_validated(definition)
        if type(result) is not BatchedFiniteFourierSeriesEvaluationResult:
            raise TypeError(
                "backend must return BatchedFiniteFourierSeriesEvaluationResult"
            )
        # Exact result instances can still be forged around their dataclass constructor.
        # Revalidate the complete public surface at the nominal backend boundary rather
        # than trusting nominal type alone.
        if (
            type(result.dimension) is not int
            or result.dimension != definition.dimension
        ):
            raise ValueError("backend result dimension is inconsistent")
        if (
            type(result.point_count) is not int
            or result.point_count != definition.point_count
        ):
            raise ValueError("backend result point_count is inconsistent")
        if (
            type(result.term_count) is not int
            or result.term_count != definition.term_count
        ):
            raise ValueError("backend result term_count is inconsistent")
        if (
            type(result.channel_count) is not int
            or result.channel_count != definition.channel_count
        ):
            raise ValueError("backend result channel_count is inconsistent")
        if (
            type(result.values) is not tuple
            or len(result.values) != definition.point_count
        ):
            raise ValueError("backend result values point shape is inconsistent")
        for point in result.values:
            if type(point) is not tuple or len(point) != definition.channel_count:
                raise ValueError("backend result values channel shape is inconsistent")
            for value in point:
                if type(value) is not complex:
                    raise TypeError(
                        "backend result values must be built-in complex values"
                    )
                if not math.isfinite(value.real) or not math.isfinite(value.imag):
                    raise FloatingPointError("backend result values must be finite")
        backend_identifier, backend_version = _backend_evidence(
            result.backend_identifier,
            result.backend_version,
        )
        if backend_identifier is None or backend_version is None:
            raise ValueError("nontrivial backend result must include backend evidence")
        return result

    @staticmethod
    def _trivial_result(
        definition: BatchedFiniteFourierSeriesEvaluationDefinition,
        values: BatchedFiniteFourierValues,
    ) -> BatchedFiniteFourierSeriesEvaluationResult:
        """Construct a backend-free result for a mathematically exact case.

        Parameters
        ----------
        definition : BatchedFiniteFourierSeriesEvaluationDefinition
            Correlated public definition.
        values : tuple of tuple of complex
            Exact point-major result values.

        Returns
        -------
        BatchedFiniteFourierSeriesEvaluationResult
            Structurally correlated result with absent backend evidence.
        """
        return BatchedFiniteFourierSeriesEvaluationResult(
            definition.dimension,
            definition.point_count,
            definition.term_count,
            definition.channel_count,
            values,
            None,
            None,
        )

    @abstractmethod
    def _evaluate_validated(
        self,
        definition: BatchedFiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> BatchedFiniteFourierSeriesEvaluationResult:
        """Evaluate a definition whose three public collections are nonempty.

        Parameters
        ----------
        definition : BatchedFiniteFourierSeriesEvaluationDefinition
            Fully validated nontrivial definition.

        Returns
        -------
        BatchedFiniteFourierSeriesEvaluationResult
            Exact public result type with backend evidence.
        """
        raise NotImplementedError


__all__ = [
    "BatchedFiniteFourierSeriesEvaluationDefinition",
    "BatchedFiniteFourierSeriesEvaluationResult",
    "BatchedFiniteFourierSeriesEvaluator",
    "BatchedFiniteFourierValues",
    "FiniteFourierCoefficientChannels",
]
