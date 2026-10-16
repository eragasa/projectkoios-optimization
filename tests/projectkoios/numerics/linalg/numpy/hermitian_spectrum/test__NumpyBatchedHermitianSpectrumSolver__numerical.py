"""Bounded numerical verification for batched complete Hermitian spectra."""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
    HermitianSpectrumMatrices,
    HermitianSpectrumMatrix,
)
from projectkoios.numerics.linalg.numpy.hermitian_spectrum import (
    NumpyBatchedHermitianSpectrumSolver,
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


def _seeded_batch(
    rng: np.random.Generator,
    batch_size: int,
    dimension: int,
) -> HermitianSpectrumMatrices:
    matrices: list[HermitianSpectrumMatrix] = []
    for _ in range(batch_size):
        raw = np.asarray(
            rng.normal(size=(dimension, dimension))
            + 1.0j * rng.normal(size=(dimension, dimension)),
            dtype=np.complex128,
        )
        hermitian = np.asarray((raw + raw.conj().T) * 0.5, dtype=np.complex128)
        matrices.append(_matrix(hermitian))
    return tuple(matrices)


def test_seeded_batches_agree_with_high_precision_complete_spectra() -> None:
    rng = np.random.default_rng(20261016)
    solver = NumpyBatchedHermitianSpectrumSolver()

    for dimension in (1, 2, 3, 4):
        matrices = _seeded_batch(rng, 8, dimension)
        result = solver.solve(
            BatchedHermitianSpectrumDefinition(
                matrices,
                BatchedHermitianSpectrumTolerance(1.0e-14, 1.0e-14),
            )
        )

        for matrix, actual in zip(matrices, result.eigenvalues, strict=True):
            expected = _reference(matrix)
            scale = max(1.0, *(abs(float(value)) for value in expected))
            for observed, reference in zip(actual, expected, strict=True):
                assert abs(observed - float(reference)) <= 2.0e-14 * scale


def test_batch_order_and_uniform_real_shift_are_covariant() -> None:
    matrices = (
        ((1.0 + 0.0j, 1.0 - 2.0j), (1.0 + 2.0j, 4.0 + 0.0j)),
        ((-2.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 3.0 + 0.0j)),
        ((0.0 + 0.0j, 0.0 + 1.0j), (0.0 - 1.0j, 0.0 + 0.0j)),
    )
    shift = 7.25
    shifted = tuple(
        tuple(
            tuple(
                entry + (shift if row == column else 0.0)
                for column, entry in enumerate(matrix[row])
            )
            for row in range(2)
        )
        for matrix in matrices
    )
    solver = NumpyBatchedHermitianSpectrumSolver()
    tolerance = BatchedHermitianSpectrumTolerance(0.0, 0.0)

    base = solver.solve(BatchedHermitianSpectrumDefinition(matrices, tolerance))
    moved = solver.solve(BatchedHermitianSpectrumDefinition(shifted, tolerance))
    reversed_result = solver.solve(
        BatchedHermitianSpectrumDefinition(tuple(reversed(matrices)), tolerance)
    )

    assert reversed_result.eigenvalues == tuple(reversed(base.eigenvalues))
    for base_spectrum, shifted_spectrum in zip(
        base.eigenvalues,
        moved.eigenvalues,
        strict=True,
    ):
        for base_value, shifted_value in zip(
            base_spectrum,
            shifted_spectrum,
            strict=True,
        ):
            assert shifted_value == pytest.approx(
                base_value + shift,
                rel=0.0,
                abs=8.0 * math.ulp(base_value + shift),
            )


def test_tolerance_accepted_input_uses_explicit_representative_spectrum() -> None:
    matrix = (
        (1.0 + 2.0e-10j, 2.0 + 4.0e-10j),
        (2.0 - 2.0e-10j, 3.0 + 0.0j),
    )
    representative = (
        (1.0 + 0.0j, 2.0 + 3.0e-10j),
        (2.0 - 3.0e-10j, 3.0 + 0.0j),
    )

    result = NumpyBatchedHermitianSpectrumSolver().solve(
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
