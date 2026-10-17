"""Immutable unit-free weighted affine residual definitions."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from numbers import Real
from typing import SupportsFloat

from projectkoios.numerics.validation import exact_finite_float


@dataclass(frozen=True, slots=True)
class WeightedAffineResidualTerm:
    """Define one finite positive-weight affine residual.

    Parameters
    ----------
    coefficients
        Nonempty exact tuple of finite real coordinate coefficients. Tuple order defines
        the coordinate convention.
    target
        Finite real target subtracted from the coefficient-vector product.
    weight
        Finite strictly positive multiplier for the squared residual.

    Notes
    -----
    Numeric inputs must be canonical built-in binary64 floats. Booleans, numeric
    strings, non-real values, and nonfinite or unrepresentable values are rejected.
    The object is frozen and slotted; it assigns no units or scientific meaning.
    """

    coefficients: tuple[float, ...]
    target: float
    weight: float

    def __post_init__(self) -> None:
        """Validate the canonical immutable residual term.

        Raises
        ------
        TypeError
            If ``coefficients`` is not an exact tuple or a numeric input is Boolean,
            non-real, or otherwise unsupported.
        ValueError
            If coefficients are empty, a number is nonfinite or unrepresentable, or
            ``weight`` is not strictly positive.
        """
        if type(self.coefficients) is not tuple:
            raise TypeError("coefficients must be a tuple")
        if not self.coefficients:
            raise ValueError("coefficients must be nonempty")
        for index, value in enumerate(self.coefficients):
            exact_finite_float(value, f"coefficients[{index}]")
        exact_finite_float(self.target, "target")
        exact_finite_float(self.weight, "weight")
        if self.weight <= 0.0:
            raise ValueError("weight must be strictly positive")


@dataclass(frozen=True, slots=True)
class WeightedAffineResidualDefinition:
    """Define an ordered finite unit-free weighted affine residual system.

    Parameters
    ----------
    terms
        Nonempty exact tuple of :class:`WeightedAffineResidualTerm` instances with one
        common positive coefficient dimension. Term order is preserved.

    Notes
    -----
    The represented function is
    ``sum(term.weight * (term.coefficients @ x - term.target) ** 2)``. Weights are
    used exactly as supplied. A caller requiring mean-square or another normalization
    must apply it before construction. Scientific names, units, channel alignment,
    sample meaning, thresholds, and certificate interpretation remain outside this
    numerical object.
    """

    terms: tuple[WeightedAffineResidualTerm, ...]

    def __post_init__(self) -> None:
        """Validate term ownership and a common positive dimension.

        Raises
        ------
        TypeError
            If ``terms`` is not an exact tuple or an item is not exactly a
            :class:`WeightedAffineResidualTerm`.
        ValueError
            If ``terms`` is empty or coefficient dimensions differ.
        """
        if type(self.terms) is not tuple or any(
            type(term) is not WeightedAffineResidualTerm for term in self.terms
        ):
            raise TypeError("terms must be a tuple of WeightedAffineResidualTerm")
        if not self.terms:
            raise ValueError("terms must be nonempty")
        dimension = len(self.terms[0].coefficients)
        if any(len(term.coefficients) != dimension for term in self.terms):
            raise ValueError("every residual term must have one common dimension")

    @property
    def dimension(self) -> int:
        """Return the positive coordinate dimension.

        Returns
        -------
        int
            Number of coefficients in every residual term.
        """
        return len(self.terms[0].coefficients)

    def squared_loss(self, coordinates: tuple[float, ...], /) -> float:
        """Evaluate the weighted residual sum directly.

        Parameters
        ----------
        coordinates
            Exact tuple of finite real coordinates in the caller-established order and
            with exactly :attr:`dimension` entries.

        Returns
        -------
        float
            Finite nonnegative binary64 weighted squared-residual sum.

        Raises
        ------
        TypeError
            If ``coordinates`` is not an exact tuple or an entry is Boolean, non-real,
            or otherwise unsupported.
        ValueError
            If the dimension differs, an entry is nonfinite or unrepresentable, or
            affine or weighted arithmetic produces a nonfinite result.

        Notes
        -----
        Scalar summation uses :func:`math.fsum`. Weighted products use square-root
        rescaling when direct multiplication would prematurely overflow or underflow.
        This is checked binary64 evaluation, not interval arithmetic or directed
        rounding.
        """
        if type(coordinates) is not tuple:
            raise TypeError("coordinates must be a tuple")
        if len(coordinates) != self.dimension:
            raise ValueError("coordinate dimension must match the residual definition")
        values = tuple(_coordinate(value) for value in coordinates)
        contributions: list[float] = []
        for term in self.terms:
            products = tuple(
                _finite_product(coefficient, value)
                for coefficient, value in zip(term.coefficients, values, strict=True)
            )
            residual = _finite_sum(products) - term.target
            if not math.isfinite(residual):
                raise ValueError("affine residual is nonfinite")
            contributions.append(_weighted_product(term.weight, residual, residual))
        return _finite_sum(contributions)


def _coordinate(value: SupportsFloat) -> float:
    """Return one finite built-in binary64 affine coordinate."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("coordinate must be a real number excluding bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "coordinate must be representable as a finite float"
        ) from error
    if not math.isfinite(result):
        raise ValueError("coordinate must be representable as a finite float")
    return result


def _weighted_product(weight: float, first: float, second: float) -> float:
    """Return ``weight * first * second`` with a scale-aware fallback."""
    product = first * second
    if math.isfinite(product) and (
        abs(product) >= sys.float_info.min
        or (product == 0.0 and (first == 0.0 or second == 0.0))
    ):
        result = weight * product
        if math.isfinite(result) and (
            abs(result) >= sys.float_info.min or (result == 0.0 and product == 0.0)
        ):
            return result
    scale = math.sqrt(weight)
    scaled_first = _finite_product(scale, first)
    scaled_second = _finite_product(scale, second)
    return _finite_product(scaled_first, scaled_second)


def _finite_product(first: float, second: float) -> float:
    result = first * second
    if not math.isfinite(result):
        raise ValueError("floating-point product is nonfinite")
    return result


def _finite_sum(values: tuple[float, ...] | list[float]) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError("floating-point sum is nonfinite") from error
    if not math.isfinite(result):
        raise ValueError("floating-point sum is nonfinite")
    return result


__all__ = [
    "WeightedAffineResidualDefinition",
    "WeightedAffineResidualTerm",
]
