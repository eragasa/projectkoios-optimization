"""Immutable unit-free positive diagonal metric geometry.

Physical callers must nondimensionalize coordinates and metric entries before crossing
this boundary. The object assigns no physical interpretation or acceptance policy.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from numbers import Real
from typing import SupportsFloat

from projectkoios.numerics.validation import exact_finite_float


@dataclass(frozen=True, slots=True)
class PositiveDiagonalMetric:
    """Represent one strictly positive diagonal unit-free metric.

    Parameters
    ----------
    diagonal
        Nonempty exact tuple of finite strictly positive real diagonal entries. Tuple
        order defines the coordinate convention.

    Notes
    -----
    Numeric entries must be canonical built-in binary64 floats. The object is frozen
    and slotted. Physical callers must nondimensionalize coordinates, covectors, and
    metric entries before construction. Scientific names, units, channel alignment,
    thresholds, separating-direction requirements, and certificate meaning remain
    outside this numerical object.
    """

    diagonal: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate dimension, finiteness, and strict positivity.

        Raises
        ------
        TypeError
            If ``diagonal`` is not an exact tuple or an entry is Boolean, non-real, or
            otherwise unsupported.
        ValueError
            If the tuple is empty, an entry is nonfinite or unrepresentable, or an
            entry is not strictly positive.
        """
        if type(self.diagonal) is not tuple:
            raise TypeError("diagonal must be a tuple")
        if not self.diagonal:
            raise ValueError("diagonal must be nonempty")
        for index, value in enumerate(self.diagonal):
            exact_finite_float(value, f"diagonal[{index}]")
            if value <= 0.0:
                raise ValueError("metric entries must be strictly positive")

    @property
    def dimension(self) -> int:
        """Return the positive coordinate dimension.

        Returns
        -------
        int
            Number of ordered metric entries.
        """
        return len(self.diagonal)

    def squared_distance(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
        /,
    ) -> float:
        """Evaluate the squared primal distance ``(left-right)^T M (left-right)``.

        Parameters
        ----------
        left, right
            Exact tuples of finite real coordinates in the caller-established order and
            with exactly :attr:`dimension` entries.

        Returns
        -------
        float
            Finite nonnegative binary64 squared distance.

        Raises
        ------
        TypeError
            If either input is not an exact tuple or an entry is unsupported.
        ValueError
            If dimensions differ, an entry is nonfinite or unrepresentable, a
            coordinate difference is nonfinite, or weighted arithmetic overflows.
        """
        left_values = self._point(left)
        right_values = self._point(right)
        contributions: list[float] = []
        for weight, first, second in zip(
            self.diagonal, left_values, right_values, strict=True
        ):
            difference = first - second
            if not math.isfinite(difference):
                raise ValueError("coordinate difference is nonfinite")
            contributions.append(_weighted_square(weight, difference))
        return _finite_sum(contributions)

    def distance(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
        /,
    ) -> float:
        """Evaluate the nonnegative primal metric distance.

        Parameters
        ----------
        left, right
            Exact tuples of finite real coordinates in the caller-established order and
            with exactly :attr:`dimension` entries.

        Returns
        -------
        float
            Finite nonnegative binary64 distance.

        Raises
        ------
        TypeError
            If either input is not an exact tuple or an entry is unsupported.
        ValueError
            If dimensions differ, an input or scaled difference is nonfinite, or the
            resulting norm is nonfinite.
        """
        return _norm(self._primal_components(left, right))

    def squared_dual_norm(self, covector: tuple[float, ...], /) -> float:
        """Evaluate the squared dual norm ``u^T M^{-1} u``.

        Parameters
        ----------
        covector
            Exact tuple of finite real components in the caller-established order and
            with exactly :attr:`dimension` entries. The zero covector is valid.

        Returns
        -------
        float
            Finite nonnegative binary64 squared dual norm.

        Raises
        ------
        TypeError
            If the input is not an exact tuple or an entry is unsupported.
        ValueError
            If dimensions differ, an entry is nonfinite or unrepresentable, or divided
            square arithmetic produces a nonfinite result.
        """
        values = self._covector(covector)
        contributions = [
            _divided_square(value, weight)
            for weight, value in zip(self.diagonal, values, strict=True)
        ]
        return _finite_sum(contributions)

    def dual_norm(self, covector: tuple[float, ...], /) -> float:
        """Evaluate the nonnegative dual norm ``sqrt(u^T M^{-1}u)``.

        Parameters
        ----------
        covector
            Exact tuple of finite real components in the caller-established order and
            with exactly :attr:`dimension` entries. The zero covector is valid.

        Returns
        -------
        float
            Finite nonnegative binary64 dual norm.

        Raises
        ------
        TypeError
            If the input is not an exact tuple or an entry is unsupported.
        ValueError
            If dimensions differ, an entry or inverse-square-root-scaled component is
            nonfinite, or the resulting norm is nonfinite.
        """
        return _norm(self._dual_components(covector))

    def _primal_components(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> tuple[float, ...]:
        left_values = self._point(left)
        right_values = self._point(right)
        components: list[float] = []
        for weight, first, second in zip(
            self.diagonal, left_values, right_values, strict=True
        ):
            difference = first - second
            if not math.isfinite(difference):
                raise ValueError("coordinate difference is nonfinite")
            components.append(_scaled_coordinate(weight, difference))
        return tuple(components)

    def _dual_components(self, covector: tuple[float, ...]) -> tuple[float, ...]:
        values = self._covector(covector)
        components: list[float] = []
        for weight, value in zip(self.diagonal, values, strict=True):
            component = value / math.sqrt(weight)
            if not math.isfinite(component):
                raise ValueError("scaled dual component is nonfinite")
            components.append(component)
        return tuple(components)

    def _point(self, value: tuple[float, ...]) -> tuple[float, ...]:
        if type(value) is not tuple:
            raise TypeError("metric point must be a tuple")
        if len(value) != self.dimension:
            raise ValueError("metric point dimension must match the metric")
        return tuple(_point_coordinate(item) for item in value)

    def _covector(self, value: tuple[float, ...]) -> tuple[float, ...]:
        if type(value) is not tuple:
            raise TypeError("covector must be a tuple")
        if len(value) != self.dimension:
            raise ValueError("covector dimension must match the metric")
        return tuple(_covector_component(item) for item in value)


def _point_coordinate(value: SupportsFloat) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("metric point coordinate must be a real number excluding bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "metric point coordinate must be representable as a finite float"
        ) from error
    if not math.isfinite(result):
        raise ValueError(
            "metric point coordinate must be representable as a finite float"
        )
    return result


def _covector_component(value: SupportsFloat) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("covector component must be a real number excluding bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "covector component must be representable as a finite float"
        ) from error
    if not math.isfinite(result):
        raise ValueError("covector component must be representable as a finite float")
    return result


def _norm(values: tuple[float, ...]) -> float:
    result = math.hypot(*values)
    if not math.isfinite(result):
        raise ValueError("metric norm is nonfinite")
    return result


def _weighted_square(weight: float, value: float) -> float:
    square = value * value
    if math.isfinite(square) and (
        abs(square) >= sys.float_info.min or (square == 0.0 and value == 0.0)
    ):
        result = weight * square
        if math.isfinite(result) and (
            abs(result) >= sys.float_info.min or (result == 0.0 and square == 0.0)
        ):
            return result
    scaled = math.sqrt(weight) * value
    if not math.isfinite(scaled):
        raise ValueError("metric contribution is nonfinite")
    result = scaled * scaled
    if not math.isfinite(result):
        raise ValueError("metric contribution is nonfinite")
    return result


def _divided_square(value: float, weight: float) -> float:
    square = value * value
    if math.isfinite(square) and (
        abs(square) >= sys.float_info.min or (square == 0.0 and value == 0.0)
    ):
        result = square / weight
        if math.isfinite(result) and (
            abs(result) >= sys.float_info.min or (result == 0.0 and square == 0.0)
        ):
            return result
    scaled = value / math.sqrt(weight)
    if not math.isfinite(scaled):
        raise ValueError("dual-metric contribution is nonfinite")
    result = scaled * scaled
    if not math.isfinite(result):
        raise ValueError("dual-metric contribution is nonfinite")
    return result


def _scaled_coordinate(weight: float, difference: float) -> float:
    result = math.sqrt(weight) * difference
    if not math.isfinite(result):
        raise ValueError("scaled coordinate difference is nonfinite")
    return result


def _finite_sum(values: list[float]) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError("metric sum is nonfinite") from error
    if not math.isfinite(result):
        raise ValueError("metric sum is nonfinite")
    return result


__all__ = ["PositiveDiagonalMetric"]
