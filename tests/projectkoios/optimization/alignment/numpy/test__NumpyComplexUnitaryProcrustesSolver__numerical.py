"""Bounded numerical verification of complex-unitary Procrustes alignment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projectkoios.numerics.numpy.typing import Complex128Array
from projectkoios.optimization.alignment.numpy.procrustes import (
    NumpyComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesDefinition,
    ComplexUnitaryProcrustesMatrix,
    ComplexUnitaryProcrustesStatus,
    ComplexUnitaryProcrustesTolerance,
)

pytestmark = pytest.mark.numerical


def _matrix(value: Complex128Array) -> ComplexUnitaryProcrustesMatrix:
    return tuple(tuple(complex(entry) for entry in row) for row in value)


def _unitary(rng: np.random.Generator, dimension: int) -> Complex128Array:
    matrix = rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(
        size=(dimension, dimension)
    )
    orthonormal, triangular = np.linalg.qr(matrix)
    phases = np.diag(triangular)
    phases = np.where(np.abs(phases) == 0.0, 1.0 + 0.0j, phases / np.abs(phases))
    return np.asarray(orthonormal * phases.conj(), dtype=np.complex128)


def test_seeded_solutions_beat_independent_random_unitary_candidates() -> None:
    rng = np.random.default_rng(20261015)
    solver = NumpyComplexUnitaryProcrustesSolver()

    for row_count, column_count in ((2, 1), (4, 2), (7, 3)):
        for _ in range(12):
            source = rng.normal(size=(row_count, column_count)) + 1.0j * rng.normal(
                size=(row_count, column_count)
            )
            target = rng.normal(size=(row_count, column_count)) + 1.0j * rng.normal(
                size=(row_count, column_count)
            )
            source = np.asarray(source, dtype=np.complex128)
            target = np.asarray(target, dtype=np.complex128)
            result = solver.solve(
                ComplexUnitaryProcrustesDefinition(_matrix(source), _matrix(target))
            )

            assert result.status is ComplexUnitaryProcrustesStatus.UNIQUE
            assert result.transformation is not None
            transformation = np.asarray(result.transformation, dtype=np.complex128)
            direct_residual = float(np.linalg.norm(source @ transformation - target))
            assert direct_residual == pytest.approx(
                result.optimal_residual_frobenius_norm,
                rel=1.0e-8,
                abs=1.0e-8,
            )
            reconstructed_squared = (
                result.source_squared_frobenius_norm
                + result.target_squared_frobenius_norm
                - 2.0 * math.fsum(result.cross_singular_values)
            )
            assert reconstructed_squared == pytest.approx(
                result.optimal_squared_residual,
                rel=1.0e-12,
                abs=1.0e-12,
            )
            for _ in range(20):
                candidate = _unitary(rng, column_count)
                candidate_residual = float(np.linalg.norm(source @ candidate - target))
                assert result.optimal_residual_frobenius_norm <= (
                    candidate_residual + 1.0e-10
                )


def test_known_right_transform_is_recovered_for_nonsingular_source() -> None:
    rng = np.random.default_rng(814)
    source = np.asarray(
        rng.normal(size=(6, 3)) + 1.0j * rng.normal(size=(6, 3)),
        dtype=np.complex128,
    )
    expected = _unitary(rng, 3)
    target = source @ expected

    result = NumpyComplexUnitaryProcrustesSolver().solve(
        ComplexUnitaryProcrustesDefinition(_matrix(source), _matrix(target))
    )

    assert result.transformation is not None
    np.testing.assert_allclose(
        np.asarray(result.transformation),
        expected,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert result.optimal_squared_residual == 0.0
    assert result.optimal_residual_frobenius_norm == 0.0
    assert result.applied_residual_roundoff_cutoff > 0.0


def test_relative_rank_policy_is_covariant_under_common_matrix_scaling() -> None:
    source = np.asarray(((1.0, 0.0), (0.0, 1.0e-4)), dtype=np.complex128)
    target = source.copy()
    tolerance = ComplexUnitaryProcrustesTolerance(
        rank_absolute=0.0,
        rank_relative=1.0e-7,
        validation_absolute=1.0e-12,
        validation_relative=1.0e-10,
    )
    solver = NumpyComplexUnitaryProcrustesSolver()

    base = solver.solve(
        ComplexUnitaryProcrustesDefinition(
            _matrix(source),
            _matrix(target),
            tolerance,
        )
    )
    scaled = solver.solve(
        ComplexUnitaryProcrustesDefinition(
            _matrix(1.0e6 * source),
            _matrix(1.0e6 * target),
            tolerance,
        )
    )

    assert base.numerical_rank == scaled.numerical_rank == 1
    assert base.transformation is None
    assert scaled.transformation is None
    assert scaled.applied_rank_cutoff == pytest.approx(
        1.0e12 * base.applied_rank_cutoff,
        rel=1.0e-15,
    )


def test_scalar_complex_case_matches_independent_phase_formula() -> None:
    source = 2.0 - 3.0j
    target = -4.0 + 1.0j

    result = NumpyComplexUnitaryProcrustesSolver().solve(
        ComplexUnitaryProcrustesDefinition(((source,),), ((target,),))
    )

    assert result.transformation is not None
    expected = source.conjugate() * target / abs(source.conjugate() * target)
    assert result.transformation[0][0] == pytest.approx(
        expected,
        rel=0.0,
        abs=1.0e-15,
    )
    assert result.optimal_residual_frobenius_norm == pytest.approx(
        abs(abs(source) - abs(target)),
        rel=1.0e-14,
        abs=1.0e-14,
    )
