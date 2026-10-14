"""NumPy implementation of real and complex linear least squares."""

from __future__ import annotations

import math
from typing import final

import numpy as np

from ....numerics.numpy.typing import Complex128Array
from ..linear import (
    LinearLeastSquaresMatrix,
    LinearLeastSquaresResult,
    LinearLeastSquaresSolver,
    _column_norms,
    _frobenius_norm,
    _maximum_row_norm,
)


@final
class NumpyLinearLeastSquaresSolver(LinearLeastSquaresSolver):
    """Solve dense ordinary or positive-weight least squares with NumPy SVD."""

    __slots__ = ()

    def _solve_validated(
        self,
        design_matrix: LinearLeastSquaresMatrix,
        target_matrix: LinearLeastSquaresMatrix,
        weights: tuple[float, ...] | None,
        rank_tolerance: float,
        /,
    ) -> LinearLeastSquaresResult:
        design = np.asarray(design_matrix, dtype=np.complex128)
        targets = np.asarray(target_matrix, dtype=np.complex128)
        row_weights = np.ones(design.shape[0], dtype=np.float64)
        if weights is not None:
            row_weights = np.asarray(weights, dtype=np.float64)

        try:
            with np.errstate(over="raise", invalid="raise"):
                square_root_weights = np.sqrt(row_weights)
                weighted_design = square_root_weights[:, None] * design
                weighted_targets = square_root_weights[:, None] * targets
            coefficients, _, rank, singular_values = np.linalg.lstsq(
                weighted_design,
                weighted_targets,
                rcond=rank_tolerance,
            )
            with np.errstate(over="raise", invalid="raise"):
                fitted_values = design @ coefficients
                residuals = fitted_values - targets
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            raise ValueError("NumPy linear least-squares solve failed") from error

        arrays = (coefficients, fitted_values, residuals, singular_values)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("NumPy linear least-squares result must be finite")

        normalized_residuals = _complex_matrix(residuals)
        normalized_weights = tuple(float(value) for value in row_weights)
        residual_norms = _column_norms(normalized_residuals)
        weighted_residual_norms = _column_norms(
            normalized_residuals,
            normalized_weights,
        )
        residual_frobenius_norm = _frobenius_norm(normalized_residuals)
        weighted_residual_frobenius_norm = _frobenius_norm(
            normalized_residuals,
            normalized_weights,
        )
        maximum_row_residual_norm = _maximum_row_norm(normalized_residuals)
        diagnostic_values = (
            *residual_norms,
            *weighted_residual_norms,
            residual_frobenius_norm,
            weighted_residual_frobenius_norm,
            maximum_row_residual_norm,
        )
        if any(not math.isfinite(float(value)) for value in diagnostic_values):
            raise ValueError("NumPy linear least-squares diagnostics must be finite")

        normalized_singular_values = tuple(float(value) for value in singular_values)
        coefficient_count = design.shape[1]
        normalized_rank = int(rank)
        condition = None
        if normalized_rank == coefficient_count:
            smallest = normalized_singular_values[-1]
            if smallest <= 0.0:
                raise ValueError("full-rank singular values must be positive")
            condition = normalized_singular_values[0] / smallest
            if not math.isfinite(condition):
                raise ValueError("full-rank condition number must be finite")

        return LinearLeastSquaresResult(
            _complex_matrix(coefficients),
            _complex_matrix(fitted_values),
            normalized_residuals,
            residual_norms,
            weighted_residual_norms,
            residual_frobenius_norm,
            weighted_residual_frobenius_norm,
            maximum_row_residual_norm,
            normalized_rank,
            normalized_singular_values,
            condition,
        )


def _complex_matrix(values: Complex128Array, /) -> LinearLeastSquaresMatrix:
    return tuple(tuple(complex(value) for value in row) for row in values)


__all__ = ["NumpyLinearLeastSquaresSolver"]
