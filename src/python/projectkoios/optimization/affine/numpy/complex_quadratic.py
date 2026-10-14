"""NumPy complex-observation affine quadratic construction."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from ...quadratic.quadratic import PositiveDefiniteQuadraticLoss
from ..complex_quadratic import (
    ComplexAffineResidualMatrix,
    ComplexAffineResidualQuadraticConstructor,
    ComplexAffineResidualVector,
    _weighted_normal_matrix,
    _weighted_squared_loss,
)


@final
class NumpyComplexAffineResidualQuadraticConstructor(
    ComplexAffineResidualQuadraticConstructor
):
    """Construct dense quadratics for complex observations and real coordinates."""

    __slots__ = ()

    def _construct_validated(
        self,
        design_matrix: ComplexAffineResidualMatrix,
        target_vector: ComplexAffineResidualVector,
        weights: tuple[float, ...] | None,
        /,
    ) -> PositiveDefiniteQuadraticLoss:
        """Solve the stacked weighted real system and construct its quadratic."""
        design = np.asarray(design_matrix, dtype=np.complex128)
        targets = np.asarray(target_vector, dtype=np.complex128)
        row_weights = np.ones(design.shape[0], dtype=np.float64)
        if weights is not None:
            row_weights = np.asarray(weights, dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                square_root_weights = np.sqrt(row_weights)
                real_design = np.concatenate((design.real, design.imag), axis=0)
                real_targets = np.concatenate((targets.real, targets.imag), axis=0)
                repeated_scales = np.concatenate(
                    (square_root_weights, square_root_weights)
                )
                weighted_design = repeated_scales[:, None] * real_design
                weighted_targets = repeated_scales * real_targets
            center_values, _, rank, _ = np.linalg.lstsq(
                weighted_design,
                weighted_targets,
                rcond=None,
            )
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            raise ValueError(
                "NumPy complex affine least-squares solve failed"
            ) from error
        if int(rank) != design.shape[1]:
            raise ValueError(
                "complex affine residual system must have numerical full "
                "real-coordinate rank"
            )
        if not np.isfinite(center_values).all():
            raise ValueError("quadratic center must be finite")
        center = tuple(float(value) for value in center_values)
        if not all(math.isfinite(value) for value in center):
            raise ValueError("quadratic center must be finite")
        normalized_weights = tuple(float(value) for value in row_weights)
        matrix = _weighted_normal_matrix(design_matrix, normalized_weights)
        minimum = _weighted_squared_loss(
            design_matrix,
            target_vector,
            normalized_weights,
            center,
        )
        return PositiveDefiniteQuadraticLoss(center, matrix, minimum)


__all__ = ["NumpyComplexAffineResidualQuadraticConstructor"]
