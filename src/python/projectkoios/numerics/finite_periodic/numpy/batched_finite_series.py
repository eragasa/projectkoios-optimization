"""Bounded NumPy backend for shared-grid batched finite Fourier series.

The backend constructs one point-batch-by-mode phase matrix at a time and multiplies it
by all coefficient channels. Small problems use one shared dense operation; larger
problems are bounded to one million phase elements per batch. Coordinate-component
accumulation is performed in declared component order. The final mode reduction is owned
by NumPy's dense matrix multiplication and is intentionally not the observable scalar
reduction contract of :mod:`projectkoios.numerics.finite_periodic.finite_series`.
"""

from __future__ import annotations

from typing import cast, final

import numpy as np

from projectkoios.numerics.numpy.typing import Complex128Array, Float64Array

from ..batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
    BatchedFiniteFourierSeriesEvaluationResult,
    BatchedFiniteFourierSeriesEvaluator,
)

_MAXIMUM_PHASE_ELEMENTS = 1_000_000


def _evaluate_coefficient_channels(
    coordinates: Float64Array,
    modes: Float64Array,
    coefficient_channels: Complex128Array,
    maximum_phase_elements: int,
    /,
) -> Complex128Array:
    """Evaluate positive-exponent channels in bounded point batches.

    Parameters
    ----------
    coordinates
        Float64 point-by-dimension array.
    modes
        Float64 mode-by-dimension array.
    coefficient_channels
        Complex128 channel-by-mode array.
    maximum_phase_elements
        Positive upper bound on elements in any constructed point-by-mode phase matrix.

    Returns
    -------
    numpy.ndarray
        Complex128 point-by-channel values.

    Raises
    ------
    FloatingPointError
        If phase construction, exponentiation, reduction, or output becomes nonfinite.
    """
    point_count = int(coordinates.shape[0])
    term_count = int(modes.shape[0])
    channel_count = int(coefficient_channels.shape[0])
    if term_count == 0:
        return np.zeros((point_count, channel_count), dtype=np.complex128)
    mode_batch_size = min(term_count, maximum_phase_elements)
    point_batch_size = max(1, maximum_phase_elements // mode_batch_size)
    values = np.zeros((point_count, channel_count), dtype=np.complex128)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for point_start in range(0, point_count, point_batch_size):
            point_stop = min(point_start + point_batch_size, point_count)
            point_batch = coordinates[point_start:point_stop]
            for mode_start in range(0, term_count, mode_batch_size):
                mode_stop = min(mode_start + mode_batch_size, term_count)
                mode_batch = modes[mode_start:mode_stop]
                inner_products = np.zeros(
                    (point_stop - point_start, mode_stop - mode_start),
                    dtype=np.float64,
                )
                for component in range(int(coordinates.shape[1])):
                    products = (
                        point_batch[:, component, None] * mode_batch[None, :, component]
                    )
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
                values[point_start:point_stop] = values[
                    point_start:point_stop
                ] + np.matmul(
                    phases,
                    coefficient_channels[:, mode_start:mode_stop].T,
                )
        if not bool(np.isfinite(values).all()):
            raise FloatingPointError(
                "batched finite Fourier evaluation produced nonfinite values"
            )
    return cast(Complex128Array, values)


@final
class NumpyBatchedFiniteFourierSeriesEvaluator(BatchedFiniteFourierSeriesEvaluator):
    """Evaluate shared-grid coefficient channels with bounded phase batches.

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
        """Evaluate a nonempty shared-grid definition with bounded NumPy operations.

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

        values = _evaluate_coefficient_channels(
            coordinates,
            modes,
            coefficients,
            _MAXIMUM_PHASE_ELEMENTS,
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
