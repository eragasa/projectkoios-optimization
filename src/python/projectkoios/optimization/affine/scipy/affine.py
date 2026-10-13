"""SciPy affine-residual quadratic construction."""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as spla

from ...quadratic.quadratic import PositiveDefiniteQuadraticLoss
from ..affine import WeightedAffineResidualDefinition
from ..quadratic import weighted_design, weighted_normal_matrix


class ScipyWeightedAffineResidualQuadraticConstructor:
    """Construct a sparse affine-residual quadratic with SciPy least squares."""

    __slots__ = ()

    def execute(
        self,
        definition: WeightedAffineResidualDefinition,
    ) -> PositiveDefiniteQuadraticLoss:
        """Construct a sparse quadratic after direct weighted-design rank acceptance.

        Parameters
        ----------
        definition
            Exact affine residual definition whose square-root-weighted design has
            numerical full column rank under ``scipy.linalg.lstsq`` with ``cond=None``.

        Returns
        -------
        PositiveDefiniteQuadraticLoss
            SciPy CSR-backed completed-square loss.

        Raises
        ------
        TypeError
            If ``definition`` is not exactly a
            :class:`WeightedAffineResidualDefinition`.
        ValueError
            If weighted construction is nonfinite, least squares does not converge,
            SciPy reports deficient numerical rank, the center is nonfinite, or the
            represented normal matrix fails sparse positive-definiteness acceptance.

        Notes
        -----
        The solve acts on the weighted design rather than normal equations and uses the
        divide-and-conquer SVD driver. Numerical rank, ARPACK, and represented
        positive-definiteness acceptance are not exact-real certificates. Rejected
        systems are not regularized.
        """
        if type(definition) is not WeightedAffineResidualDefinition:
            raise TypeError("definition must be WeightedAffineResidualDefinition")
        rows, targets = weighted_design(definition)
        design = np.asarray(rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        try:
            center_values, _, rank, _ = spla.lstsq(
                design,
                target,
                cond=None,
                lapack_driver="gelsd",
                check_finite=False,
            )
        except (spla.LinAlgError, ValueError) as error:
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
            sparse=True,
        )


__all__ = ["ScipyWeightedAffineResidualQuadraticConstructor"]
