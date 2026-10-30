"""NumPy evaluation of finite matrix Fourier series.

Arbitrary selected modes use bounded point-and-mode batches. Complete or large
construction-box inventories use separable tensor contractions with a separate bound on
all non-output working arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Self, final

import numpy as np

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    _backend_evidence,
)
from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array
from projectkoios.numerics.numpy.validation import (
    NumpyImmutableArrayConstructor,
    NumpyImmutableArrayValidator,
)
from projectkoios.numerics.validation import exact_positive_integer

from .batched_finite_series import _evaluate_coefficient_channels
from .matrix_series import NumpyFiniteFourierMatrixSeries


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyFiniteFourierMatrixEvaluationDefinition:
    """Define bounded evaluation of a selected matrix Fourier series.

    Parameters
    ----------
    series
        Ordered selected integer modes and square matrix coefficients.
    coordinates
        Canonical immutable float64 array with shape ``(..., dimension)`` and at least
        one leading point dimension.
    maximum_phase_elements
        Positive upper bound on elements in any temporary point-by-mode phase matrix.

    Notes
    -----
    Use :meth:`from_array` for a defensive coordinate copy or
    :meth:`from_immutable_array` for an explicit zero-copy boundary.
    """

    series: NumpyFiniteFourierMatrixSeries
    coordinates: Float64Array = field(repr=False)
    maximum_phase_elements: int = 1_000_000

    @classmethod
    def from_array(
        cls,
        series: NumpyFiniteFourierMatrixSeries,
        coordinates: Float64Array,
        maximum_phase_elements: int = 1_000_000,
    ) -> Self:
        """Construct from an exact coordinate array using a defensive copy.

        Parameters
        ----------
        series
            Canonical finite matrix Fourier series.
        coordinates
            Exact finite float64 coordinate array.
        maximum_phase_elements
            Positive temporary phase-matrix element bound.

        Returns
        -------
        NumpyFiniteFourierMatrixEvaluationDefinition
            Canonical independently owned evaluation definition.
        """
        return cls(
            series,
            NumpyImmutableArrayConstructor(
                coordinates,
                "coordinates",
                np.dtype(np.float64),
            ).execute(),
            maximum_phase_elements,
        )

    @classmethod
    def from_immutable_array(
        cls,
        series: NumpyFiniteFourierMatrixSeries,
        coordinates: Float64Array,
        maximum_phase_elements: int = 1_000_000,
    ) -> Self:
        """Construct without copying a canonical immutable coordinate array.

        Parameters
        ----------
        series
            Canonical finite matrix Fourier series.
        coordinates
            Canonical immutable finite float64 coordinate array.
        maximum_phase_elements
            Positive temporary phase-matrix element bound.

        Returns
        -------
        NumpyFiniteFourierMatrixEvaluationDefinition
            Definition retaining the supplied coordinate array by identity.
        """
        return cls(series, coordinates, maximum_phase_elements)

    def __post_init__(self) -> None:
        """Validate nominal series, coordinates, and memory bound.

        Raises
        ------
        TypeError
            If fields have the wrong exact nominal or array representation.
        ValueError
            If coordinate rank, width, finiteness, or memory bound is invalid.
        """
        if type(self.series) is not NumpyFiniteFourierMatrixSeries:
            raise TypeError("series must be NumpyFiniteFourierMatrixSeries")
        coordinates = NumpyImmutableArrayValidator(
            self.coordinates,
            "coordinates",
            np.dtype(np.float64),
        ).execute()
        if coordinates.ndim < 2 or coordinates.shape[-1] != self.series.dimension:
            raise ValueError("coordinates must have shape (..., series.dimension)")
        exact_positive_integer(self.maximum_phase_elements, "maximum_phase_elements")

    @property
    def point_shape(self) -> tuple[int, ...]:
        """Return the leading tensor shape of supplied coordinates.

        Returns
        -------
        tuple of int
            One or more point-grid dimensions.
        """
        return self.coordinates.shape[:-1]

    @property
    def point_count(self) -> int:
        """Return the flattened number of supplied coordinates.

        Returns
        -------
        int
            Product of :attr:`point_shape`, including zero.
        """
        return math.prod(self.point_shape)


@final
@dataclass(frozen=True, slots=True, eq=False)
class NumpyFiniteFourierMatrixEvaluationResult:
    """Record immutable evaluated matrix values and backend evidence.

    Parameters
    ----------
    dimension
        Coordinate and mode dimension.
    point_shape
        Leading shape of the evaluated coordinate tensor.
    term_count
        Number of ordered series terms used.
    matrix_values
        Canonical immutable complex128 array with shape ``point_shape + (n, n)``.
    backend_identifier, backend_version
        Exact executed NumPy backend evidence.
    """

    dimension: int
    point_shape: tuple[int, ...]
    term_count: int
    matrix_values: Complex128Array = field(repr=False)
    backend_identifier: str
    backend_version: str

    @classmethod
    def from_array(
        cls,
        *,
        dimension: int,
        point_shape: tuple[int, ...],
        term_count: int,
        matrix_values: Complex128Array,
        backend_identifier: str,
        backend_version: str,
    ) -> Self:
        """Construct from an exact value array using a defensive copy.

        Parameters
        ----------
        dimension, point_shape, term_count
            Declared evaluation structure.
        matrix_values
            Exact finite complex128 matrix-value tensor.
        backend_identifier, backend_version
            Executed backend evidence.

        Returns
        -------
        NumpyFiniteFourierMatrixEvaluationResult
            Canonical independently owned evaluation result.
        """
        return cls(
            dimension,
            point_shape,
            term_count,
            NumpyImmutableArrayConstructor(
                matrix_values,
                "matrix_values",
                np.dtype(np.complex128),
            ).execute(),
            backend_identifier,
            backend_version,
        )

    def __post_init__(self) -> None:
        """Validate structural metadata, immutable values, and backend evidence.

        Raises
        ------
        TypeError
            If metadata, arrays, or evidence have the wrong exact representation.
        ValueError
            If dimensions, shapes, counts, or finiteness is invalid.
        """
        dimension = exact_positive_integer(self.dimension, "dimension")
        if dimension not in (1, 2, 3):
            raise ValueError("dimension must be one, two, or three")
        if type(self.point_shape) is not tuple or not self.point_shape:
            raise TypeError("point_shape must be a nonempty exact tuple")
        if not all(type(extent) is int for extent in self.point_shape):
            raise TypeError("point_shape must contain exact built-in integers")
        if not all(extent >= 0 for extent in self.point_shape):
            raise ValueError("point_shape extents must be nonnegative")
        if type(self.term_count) is not int:
            raise TypeError("term_count must be an exact built-in integer")
        if self.term_count < 0:
            raise ValueError("term_count must be nonnegative")
        values = NumpyImmutableArrayValidator(
            self.matrix_values,
            "matrix_values",
            np.dtype(np.complex128),
        ).execute()
        if values.ndim != len(self.point_shape) + 2:
            raise ValueError("matrix_values rank is inconsistent with point_shape")
        if values.shape[:-2] != self.point_shape:
            raise ValueError("matrix_values leading shape must equal point_shape")
        if values.shape[-2] <= 0 or values.shape[-1] != values.shape[-2]:
            raise ValueError("matrix values must be nonempty and square")
        _backend_evidence(self.backend_identifier, self.backend_version)


@final
class NumpyFiniteFourierMatrixSeriesEvaluator:
    """Evaluate selected matrix coefficients with bounded phase batches."""

    __slots__ = ()

    def evaluate(
        self,
        definition: NumpyFiniteFourierMatrixEvaluationDefinition,
        /,
    ) -> NumpyFiniteFourierMatrixEvaluationResult:
        """Evaluate the selected series at all supplied reduced coordinates.

        Parameters
        ----------
        definition
            Canonical series, coordinates, and phase-memory bound.

        Returns
        -------
        NumpyFiniteFourierMatrixEvaluationResult
            Matrix values retaining the coordinate tensor shape and backend evidence.

        Raises
        ------
        TypeError
            If ``definition`` has the wrong exact nominal type.
        FloatingPointError
            If phase or reduction arithmetic becomes nonfinite.
        """
        if type(definition) is not NumpyFiniteFourierMatrixEvaluationDefinition:
            raise TypeError(
                "definition must be NumpyFiniteFourierMatrixEvaluationDefinition"
            )
        series = definition.series
        coordinates = definition.coordinates.reshape(
            (definition.point_count, series.dimension)
        )
        modes = series.integer_modes.astype(np.float64)
        channels = series.matrix_coefficients.reshape(
            (series.term_count, series.matrix_size * series.matrix_size)
        ).T
        values = _evaluate_coefficient_channels(
            coordinates,
            modes,
            channels,
            definition.maximum_phase_elements,
        ).reshape((*definition.point_shape, series.matrix_size, series.matrix_size))
        return NumpyFiniteFourierMatrixEvaluationResult.from_array(
            dimension=series.dimension,
            point_shape=definition.point_shape,
            term_count=series.term_count,
            matrix_values=values,
            backend_identifier="numpy.matmul.bounded",
            backend_version=np.__version__,
        )


__all__ = [
    "NumpyFiniteFourierMatrixEvaluationDefinition",
    "NumpyFiniteFourierMatrixEvaluationResult",
    "NumpyFiniteFourierMatrixSeriesEvaluator",
]
