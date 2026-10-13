"""Backend-composed quadratic construction for weighted affine residuals."""

from __future__ import annotations

import math

from ..quadratic.quadratic import PositiveDefiniteQuadraticLoss
from .affine import WeightedAffineResidualDefinition, _weighted_product


class WeightedAffineResidualQuadraticConstructor:
    """Select a numerical backend for affine-residual quadratic construction.

    Notes
    -----
    Construction is lazy: importing this module does not import NumPy or SciPy. The
    default execution path selects the NumPy implementation; ``sparse=True`` selects
    the SciPy implementation. The action assigns no units, residual normalization,
    regularization, conditioning tolerance, scientific weights, acceptance thresholds,
    or certificate meaning.
    """

    __slots__ = ()

    def execute(
        self,
        definition: WeightedAffineResidualDefinition,
        *,
        sparse: bool = False,
    ) -> PositiveDefiniteQuadraticLoss:
        """Construct a completed-square loss with the selected backend.

        Parameters
        ----------
        definition
            Exact :class:`WeightedAffineResidualDefinition` whose square-root-weighted
            design has numerical full column rank under the selected backend.
        sparse
            Built-in Boolean selecting SciPy sparse construction when true. False
            selects NumPy dense construction.

        Returns
        -------
        PositiveDefiniteQuadraticLoss
            Backend-composed completed-square loss.

        Raises
        ------
        TypeError
            If ``definition`` is not exactly the required definition type or ``sparse``
            is not a built-in Boolean.
        ValueError
            If weighted construction is nonfinite, least squares does not converge,
            numerical rank is deficient, or represented positive-definiteness fails.
        ImportError
            If the selected optional numerical backend is unavailable.

        Notes
        -----
        Rank and positive-definiteness decisions are backend numerical acceptance, not
        exact-real proofs, interval enclosures, or outward-rounded certificates.
        Rejected systems are not regularized.
        """
        if type(definition) is not WeightedAffineResidualDefinition:
            raise TypeError("definition must be WeightedAffineResidualDefinition")
        if type(sparse) is not bool:
            raise TypeError("sparse must be a built-in bool")
        if sparse:
            from .scipy.affine import (
                ScipyWeightedAffineResidualQuadraticConstructor,
            )

            return ScipyWeightedAffineResidualQuadraticConstructor().execute(definition)

        from .numpy.affine import NumpyWeightedAffineResidualQuadraticConstructor

        return NumpyWeightedAffineResidualQuadraticConstructor().execute(definition)


def weighted_design(
    definition: WeightedAffineResidualDefinition,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return the finite square-root-weighted design and target.

    Parameters
    ----------
    definition
        Exact validated affine residual definition.

    Returns
    -------
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]
        Ordered binary64 design rows and target entries, each multiplied by the square
        root of its term weight.

    Raises
    ------
    TypeError
        If ``definition`` is not exactly a
        :class:`WeightedAffineResidualDefinition`.
    ValueError
        If scaling produces a nonfinite binary64 entry.
    """
    if type(definition) is not WeightedAffineResidualDefinition:
        raise TypeError("definition must be WeightedAffineResidualDefinition")
    rows: list[tuple[float, ...]] = []
    targets: list[float] = []
    for term in definition.terms:
        scale = math.sqrt(term.weight)
        rows.append(
            tuple(
                _finite_product(scale, coefficient, "weighted design entry")
                for coefficient in term.coefficients
            )
        )
        targets.append(_finite_product(scale, term.target, "weighted target entry"))
    return tuple(rows), tuple(targets)


def weighted_normal_matrix(
    definition: WeightedAffineResidualDefinition,
) -> tuple[tuple[float, ...], ...]:
    """Return the exactly mirrored binary64 weighted normal matrix.

    Parameters
    ----------
    definition
        Exact validated affine residual definition.

    Returns
    -------
    tuple[tuple[float, ...], ...]
        Finite symmetric matrix ``sum(weight * a * a.T)``. Mirrored entries are copied
        from one deterministically summed upper-triangle value and therefore compare
        exactly equal.

    Raises
    ------
    TypeError
        If ``definition`` is not exactly a
        :class:`WeightedAffineResidualDefinition`.
    ValueError
        If a weighted product or deterministic sum is nonfinite.
    """
    if type(definition) is not WeightedAffineResidualDefinition:
        raise TypeError("definition must be WeightedAffineResidualDefinition")
    dimension = definition.dimension
    rows = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    for first in range(dimension):
        for second in range(first, dimension):
            products = tuple(
                _weighted_product(
                    term.weight,
                    term.coefficients[first],
                    term.coefficients[second],
                )
                for term in definition.terms
            )
            value = _finite_sum(products, "normal matrix entry")
            rows[first][second] = value
            rows[second][first] = value
    return tuple(tuple(row) for row in rows)


def _finite_product(first: float, second: float, name: str) -> float:
    result = first * second
    if not math.isfinite(result):
        raise ValueError(f"{name} is nonfinite")
    return result


def _finite_sum(values: tuple[float, ...], name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError(f"{name} is nonfinite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is nonfinite")
    return result


__all__ = [
    "WeightedAffineResidualQuadraticConstructor",
    "weighted_design",
    "weighted_normal_matrix",
]
