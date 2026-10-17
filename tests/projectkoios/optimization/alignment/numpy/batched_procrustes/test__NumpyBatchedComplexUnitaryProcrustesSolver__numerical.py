"""Bounded numerical verification for NumPy batched Procrustes solving."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.numpy.procrustes import (
    NumpyComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesDefinition,
    ComplexUnitaryProcrustesStatus,
    ComplexUnitaryProcrustesTolerance,
)

pytestmark = pytest.mark.numerical


def _matrix(value: np.ndarray) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(complex(entry) for entry in row) for row in value)


def test_full_rank_batch_matches_every_scalar_owner_result() -> None:
    random = np.random.default_rng(1837)
    source = np.asarray(
        random.normal(size=(3, 4, 6, 3)) + 1.0j * random.normal(size=(3, 4, 6, 3)),
        dtype=np.complex128,
    )
    target = np.asarray(
        random.normal(size=(3, 4, 6, 3)) + 1.0j * random.normal(size=(3, 4, 6, 3)),
        dtype=np.complex128,
    )
    tolerance = ComplexUnitaryProcrustesTolerance(
        1.0e-14,
        1.0e-11,
        1.0e-12,
        1.0e-10,
    )
    batch = NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition(source, target, tolerance)
    )
    scalar = NumpyComplexUnitaryProcrustesSolver()
    unique_transformations = batch.require_unique_right_unitary_transformations()

    for index in np.ndindex(batch.batch_shape):
        expected = scalar.solve(
            ComplexUnitaryProcrustesDefinition(
                _matrix(source[index]),
                _matrix(target[index]),
                tolerance,
            )
        )
        assert expected.status is ComplexUnitaryProcrustesStatus.UNIQUE
        assert expected.transformation is not None
        np.testing.assert_allclose(
            unique_transformations[index],
            np.asarray(expected.transformation),
            rtol=0.0,
            atol=3.0e-14,
        )
        np.testing.assert_allclose(
            batch.cross_singular_values[index],
            expected.cross_singular_values,
            rtol=2.0e-15,
            atol=2.0e-15,
        )
        assert batch.numerical_ranks[index] == expected.numerical_rank
        assert batch.applied_rank_cutoffs[index] == pytest.approx(
            expected.applied_rank_cutoff,
            rel=2.0e-15,
            abs=1.0e-15,
        )
        assert batch.optimal_residual_frobenius_norms[index] == pytest.approx(
            expected.optimal_residual_frobenius_norm,
            rel=2.0e-12,
            abs=2.0e-12,
        )


def test_deficient_batch_matches_scalar_evidence_without_transform_parity_claim() -> (
    None
):
    source = np.asarray(
        (
            ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
        ),
        dtype=np.complex128,
    )
    target = np.asarray(
        (
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
            ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
        ),
        dtype=np.complex128,
    )
    batch = NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition(source, target)
    )
    scalar = NumpyComplexUnitaryProcrustesSolver()

    for index in np.ndindex(batch.batch_shape):
        expected = scalar.solve(
            ComplexUnitaryProcrustesDefinition(
                _matrix(source[index]),
                _matrix(target[index]),
            )
        )
        assert expected.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT
        assert expected.transformation is None
        assert not bool(batch.full_rank[index])
        assert batch.numerical_ranks[index] == expected.numerical_rank
        np.testing.assert_allclose(
            batch.cross_singular_values[index],
            expected.cross_singular_values,
            rtol=0.0,
            atol=2.0e-15,
        )
        assert batch.applied_rank_cutoffs[index] == pytest.approx(
            expected.applied_rank_cutoff,
            rel=0.0,
            abs=1.0e-15,
        )
        assert batch.optimal_squared_residuals[index] == pytest.approx(
            expected.optimal_squared_residual,
            rel=0.0,
            abs=2.0e-15,
        )
        completion = batch.right_unitary_completions[index]
        np.testing.assert_allclose(
            completion.conjugate().T @ completion,
            np.eye(2),
            rtol=0.0,
            atol=2.0e-15,
        )


def test_rank_two_frame_fixture_matches_consumer_residual_formula() -> None:
    angles = np.linspace(0.0, 0.7, 35).reshape(5, 7)
    source = np.zeros((5, 7, 4, 2), dtype=np.complex128)
    source[..., 0, 0] = 1.0
    source[..., 1, 1] = 1.0
    target = np.zeros_like(source)
    target[..., 0, 0] = np.cos(angles)
    target[..., 2, 0] = np.sin(angles)
    target[..., 1, 1] = 1.0

    result = NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition(source, target)
    )
    consumer_squared = np.maximum(
        0.0,
        4.0 - 2.0 * np.sum(result.cross_singular_values, axis=-1),
    )

    assert np.all(
        np.abs(result.optimal_squared_residuals - consumer_squared)
        <= result.applied_residual_roundoff_cutoffs
    )
    direct_residual = np.linalg.norm(
        source @ result.right_unitary_completions - target,
        axis=(-2, -1),
    )
    np.testing.assert_allclose(
        result.optimal_residual_frobenius_norms,
        direct_residual,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
