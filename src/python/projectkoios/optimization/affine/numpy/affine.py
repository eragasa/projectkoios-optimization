"""NumPy affine-residual quadratic construction."""

from __future__ import annotations

import math

import numpy as np

from ...quadratic.quadratic import PositiveDefiniteQuadraticLoss
from ..affine import WeightedAffineResidualDefinition
from ..quadratic import weighted_design, weighted_normal_matrix


class NumpyWeightedAffineResidualQuadraticConstructor:
    """Construct a dense affine-residual quadratic with NumPy least squares."""

    __slots__ = ()

    def execute(
        self,
        definition: WeightedAffineResidualDefinition,
    ) -> PositiveDefiniteQuadraticLoss:
        """Construct a dense quadratic after direct weighted-design rank acceptance.

        Parameters
        ----------
        definition
            Exact affine residual definition whose square-root-weighted design has
            numerical full column rank under ``numpy.linalg.lstsq`` with ``rcond=None``.

        Returns
        -------
        PositiveDefiniteQuadraticLoss
            Dense NumPy-backed completed-square loss.

        Raises
        ------
        TypeError
            If ``definition`` is not exactly a
            :class:`WeightedAffineResidualDefinition`.
        ValueError
            If weighted construction is nonfinite, least squares does not converge,
            NumPy reports deficient numerical rank, the center is nonfinite, or the
            represented normal matrix fails dense positive-definiteness acceptance.

        Notes
        -----
        The solve acts on the weighted design rather than normal equations. Numerical
        rank and positive-definiteness acceptance are not exact-real certificates.
        Rejected systems are not regularized.
        """
        if type(definition) is not WeightedAffineResidualDefinition:
            raise TypeError("definition must be WeightedAffineResidualDefinition")
        rows, targets = weighted_design(definition)
        design = np.asarray(rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        try:
            center_values, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "weighted affine least-squares solve did not converge"
            ) from error
        if int(rank) != definition.dimension:
            raise ValueError(
                "affine residual system must have numerical full column rank"
            )
        center = tuple(float(value) for value in center_values)
        if not all(math.isfinite(value) for value in center):
            raise ValueError("quadratic center is nonfinite")
        return PositiveDefiniteQuadraticLoss(
            center,
            weighted_normal_matrix(definition),
            definition.squared_loss(center),
        )


__all__ = ["NumpyWeightedAffineResidualQuadraticConstructor"]
