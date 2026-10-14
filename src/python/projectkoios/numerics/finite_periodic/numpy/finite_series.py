"""Ordered NumPy implementation of finite Fourier-series evaluation.

The evaluator computes

``y_i = sum_j c_j * exp(+2*pi*1j*dot(x_i, m_j))``

with binary64 coordinates and phase arithmetic and complex128 coefficients,
exponentials, accumulators, and outputs. It deliberately uses scalar loops rather than a
vectorized matrix reduction: point order, term order, and coordinate-component order are
observable parts of the initial numerical contract. The implementation performs no
normalization and does not coalesce duplicate integer modes.

NumPy floating-point overflow and invalid operations fail closed with
``FloatingPointError``. The public template method handles empty inputs before this
backend is called and verifies the returned backend-neutral record.
"""

from __future__ import annotations

from typing import final

import numpy as np

from ..finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
    FiniteFourierSeriesEvaluationResult,
    FiniteFourierSeriesEvaluator,
)


@final
class NumpyFiniteFourierSeriesEvaluator(FiniteFourierSeriesEvaluator):
    """Evaluate finite Fourier series with ordered scalar NumPy arithmetic.

    This implementation is deterministic within a pinned Python, NumPy, and platform
    environment. It does not promise bitwise-identical results across libm versions,
    NumPy releases, hardware, or alternative backends.
    """

    __slots__ = ()

    def _evaluate_validated(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult:
        """Evaluate nonempty points and terms in their declared order.

        The public evaluator has already validated all shapes, scalar types, finite
        inputs, and integer-mode conversion bounds. This method therefore focuses only
        on the required scalar numerical procedure. Overflow or invalid arithmetic is
        reported as ``FloatingPointError`` by NumPy's error-state machinery.
        """
        values: list[complex] = []
        two_pi = np.float64(2.0) * np.float64(np.pi)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Do not replace these loops with dot products or matrix multiplication:
            # their reduction order is part of the reviewed binary64 contract.
            for coordinate in definition.coordinates:
                total = np.complex128(0.0 + 0.0j)
                for mode, coefficient in zip(
                    definition.integer_modes,
                    definition.coefficients,
                    strict=True,
                ):
                    dot = np.float64(0.0)
                    for coordinate_component, mode_component in zip(
                        coordinate,
                        mode,
                        strict=True,
                    ):
                        # The public 2**53 mode limit makes this integer-to-float64
                        # conversion exact. The multiplication and sum remain ordinary
                        # binary64 operations in coordinate-component order.
                        product = np.float64(coordinate_component) * np.float64(
                            mode_component
                        )
                        dot = dot + product

                    # The positive sign and lack of normalization are intentional.
                    phase = two_pi * dot
                    exponential = np.exp(np.complex128(1j) * phase)
                    term = np.complex128(coefficient) * exponential
                    total = total + term

                # The error state catches normal NumPy failures. Keep an explicit final
                # guard in case a backend operation returns a nonfinite value silently.
                if not np.isfinite(total.real) or not np.isfinite(total.imag):
                    raise FloatingPointError(
                        "finite Fourier-series evaluation produced a nonfinite value"
                    )
                # Convert NumPy's scalar to the backend-neutral built-in complex type.
                values.append(complex(total))

        return FiniteFourierSeriesEvaluationResult(
            definition.dimension,
            definition.point_count,
            definition.term_count,
            tuple(values),
        )


__all__ = ["NumpyFiniteFourierSeriesEvaluator"]
