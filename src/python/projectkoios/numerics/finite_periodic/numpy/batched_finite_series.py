"""Dense NumPy backend for shared-grid batched finite Fourier series.

The backend constructs the point-by-mode phase matrix exactly once and multiplies it by
all coefficient channels in one dense operation. Coordinate-component accumulation is
performed in declared component order. The final mode reduction is owned by NumPy's
dense matrix multiplication and is intentionally not the observable scalar reduction
contract of :mod:`projectkoios.numerics.finite_periodic.finite_series`.
"""

from __future__ import annotations

from typing import final

import numpy as np

from ..batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
    BatchedFiniteFourierSeriesEvaluationResult,
    BatchedFiniteFourierSeriesEvaluator,
)


@final
class NumpyBatchedFiniteFourierSeriesEvaluator(BatchedFiniteFourierSeriesEvaluator):
    """Evaluate shared-grid coefficient channels with one dense phase matrix.

    Notes
    -----
    The implementation uses binary64 coordinate, mode, and phase arithmetic and
    complex128 coefficients, exponentials, reductions, and output. Backend evidence
    identifies NumPy's dense matrix multiplication and exact installed version.
    """

    __slots__ = ()

    def _evaluate_validated(
        self,
        definition: BatchedFiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> BatchedFiniteFourierSeriesEvaluationResult:
        """Evaluate a nonempty shared-grid definition with dense NumPy operations.

        Parameters
        ----------
        definition : BatchedFiniteFourierSeriesEvaluationDefinition
            Fully validated definition with nonempty points, modes, and channels.

        Returns
        -------
        BatchedFiniteFourierSeriesEvaluationResult
            Point-major values and NumPy backend evidence.

        Raises
        ------
        FloatingPointError
            If phase construction, exponentiation, reduction, or output becomes
            nonfinite.
        """
        coordinates = np.asarray(definition.coordinates, dtype=np.float64)
        modes = np.asarray(definition.integer_modes, dtype=np.float64)
        coefficients = np.asarray(
            definition.coefficient_channels,
            dtype=np.complex128,
        )

        # Accumulate coordinate-mode inner products by component so component order
        # remains explicit even though all points and modes share each vectorized step.
        inner_products = np.zeros(
            (definition.point_count, definition.term_count),
            dtype=np.float64,
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for component in range(definition.dimension):
                products = coordinates[:, component, None] * modes[None, :, component]
                inner_products = inner_products + products
            if not bool(np.isfinite(inner_products).all()):
                raise FloatingPointError(
                    "batched finite Fourier phase inner products became nonfinite"
                )

            phases = np.exp(
                np.complex128(2.0j * np.pi) * inner_products.astype(np.complex128)
            )
            if not bool(np.isfinite(phases).all()):
                raise FloatingPointError(
                    "batched finite Fourier phase matrix became nonfinite"
                )

            # One dense reduction evaluates all coefficient channels while reusing the
            # exact same point-mode phases. Transposition makes public output P x S.
            values = np.matmul(phases, coefficients.T)
            if not bool(np.isfinite(values).all()):
                raise FloatingPointError(
                    "batched finite Fourier evaluation produced nonfinite values"
                )

        return BatchedFiniteFourierSeriesEvaluationResult(
            definition.dimension,
            definition.point_count,
            definition.term_count,
            definition.channel_count,
            tuple(tuple(complex(value) for value in row) for row in values),
            "numpy.matmul",
            np.__version__,
        )


__all__ = ["NumpyBatchedFiniteFourierSeriesEvaluator"]
