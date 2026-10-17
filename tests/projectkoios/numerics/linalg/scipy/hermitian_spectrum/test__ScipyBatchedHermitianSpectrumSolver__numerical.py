"""Bounded numerical verification for SciPy batched Hermitian spectra."""

from __future__ import annotations

import mpmath
import numpy as np
import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
    HermitianSpectrumMatrix,
)
from projectkoios.numerics.linalg.scipy.hermitian_spectrum import (
    ScipyBatchedHermitianSpectrumSolver,
)
from projectkoios.numerics.numpy.typing import Complex128Array

pytestmark = pytest.mark.numerical


def _matrix(value: Complex128Array) -> HermitianSpectrumMatrix:
    return tuple(tuple(complex(entry) for entry in row) for row in value)


def _exact_mpf(value: float) -> mpmath.mpf:
    numerator, denominator = value.as_integer_ratio()
    return mpmath.mpf(numerator) / denominator


def _reference(matrix: HermitianSpectrumMatrix) -> tuple[mpmath.mpf, ...]:
    dimension = len(matrix)
    with mpmath.workdps(100):
        high = mpmath.matrix(dimension, dimension)
        for row in range(dimension):
            for column in range(dimension):
                entry = matrix[row][column]
                high[row, column] = mpmath.mpc(
                    _exact_mpf(entry.real),
                    _exact_mpf(entry.imag),
                )
        eigenvalues, _ = mpmath.eigh(high)
        return tuple(+eigenvalues[index] for index in range(dimension))


def test_seeded_scipy_batches_agree_with_high_precision_complete_spectra() -> None:
    rng = np.random.default_rng(20261017)
    solver = ScipyBatchedHermitianSpectrumSolver()

    for dimension in (1, 2, 3, 4):
        matrices: list[HermitianSpectrumMatrix] = []
        for _ in range(6):
            raw = np.asarray(
                rng.normal(size=(dimension, dimension))
                + 1.0j * rng.normal(size=(dimension, dimension)),
                dtype=np.complex128,
            )
            hermitian = np.asarray(
                (raw + raw.conj().T) * 0.5,
                dtype=np.complex128,
            )
            matrices.append(_matrix(hermitian))
        result = solver.solve(
            BatchedHermitianSpectrumDefinition(
                tuple(matrices),
                BatchedHermitianSpectrumTolerance(1.0e-14, 1.0e-14),
            )
        )

        for matrix, actual in zip(matrices, result.eigenvalues, strict=True):
            expected = _reference(matrix)
            scale = max(1.0, *(abs(float(value)) for value in expected))
            for observed, reference in zip(actual, expected, strict=True):
                assert abs(observed - float(reference)) <= 3.0e-14 * scale


def test_scipy_uses_the_explicit_tolerance_accepted_representative() -> None:
    matrix = (
        (1.0 + 2.0e-10j, 2.0 + 4.0e-10j),
        (2.0 - 2.0e-10j, 3.0 + 0.0j),
    )
    representative = (
        (1.0 + 0.0j, 2.0 + 3.0e-10j),
        (2.0 - 3.0e-10j, 3.0 + 0.0j),
    )

    result = ScipyBatchedHermitianSpectrumSolver().solve(
        BatchedHermitianSpectrumDefinition(
            (matrix,),
            BatchedHermitianSpectrumTolerance(4.0e-10, 0.0),
        )
    )
    expected = _reference(representative)

    np.testing.assert_allclose(
        np.asarray(result.eigenvalues[0]),
        np.asarray(tuple(float(value) for value in expected)),
        rtol=0.0,
        atol=1.0e-14,
    )
