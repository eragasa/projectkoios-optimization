"""Integration tests for NumPy complex-unitary right alignment."""

from __future__ import annotations

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
)

pytestmark = pytest.mark.integration


def _matrix(value: Complex128Array) -> ComplexUnitaryProcrustesMatrix:
    return tuple(tuple(complex(entry) for entry in row) for row in value)


def test_aligns_raw_source_on_right_to_accepted_target() -> None:
    accepted = np.asarray(
        (
            (1.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 1.0 + 0.0j),
            (0.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 0.0 + 0.0j),
        ),
        dtype=np.complex128,
    )
    rotation = np.asarray(
        (
            (0.0 + 1.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 0.6 - 0.8j),
        ),
        dtype=np.complex128,
    )
    raw = accepted @ rotation

    result = NumpyComplexUnitaryProcrustesSolver().solve(
        ComplexUnitaryProcrustesDefinition(_matrix(raw), _matrix(accepted))
    )

    assert result.status is ComplexUnitaryProcrustesStatus.UNIQUE
    assert result.transformation is not None
    transform = np.asarray(result.transformation, dtype=np.complex128)
    np.testing.assert_allclose(transform, rotation.conj().T, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(raw @ transform, accepted, rtol=0.0, atol=1.0e-14)
    assert result.optimal_residual_frobenius_norm <= 1.0e-7
    assert result.backend_identifier == "numpy.linalg.svd"
    assert result.backend_version == np.__version__


def test_exact_rank_one_frame_overlap_returns_evidence_without_transform() -> None:
    raw = np.asarray(
        (
            (1.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 1.0 + 0.0j),
            (0.0 + 0.0j, 0.0 + 0.0j),
        ),
        dtype=np.complex128,
    )
    accepted = np.asarray(
        (
            (1.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 1.0 + 0.0j),
            (0.0 + 0.0j, 0.0 + 0.0j),
            (0.0 + 0.0j, 0.0 + 0.0j),
        ),
        dtype=np.complex128,
    )

    result = NumpyComplexUnitaryProcrustesSolver().solve(
        ComplexUnitaryProcrustesDefinition(_matrix(raw), _matrix(accepted))
    )

    assert result.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT
    assert result.cross_singular_values == (1.0, 0.0)
    assert result.numerical_rank == 1
    assert result.transformation is None
    assert result.source_squared_frobenius_norm == 2.0
    assert result.target_squared_frobenius_norm == 2.0
    assert result.optimal_squared_residual == 2.0


def test_underdetermined_zero_problem_is_deficient_without_completion() -> None:
    source = ((0.0 + 0.0j, 0.0 + 0.0j),)

    result = NumpyComplexUnitaryProcrustesSolver().solve(
        ComplexUnitaryProcrustesDefinition(source, source)
    )

    assert result.status is ComplexUnitaryProcrustesStatus.RANK_DEFICIENT
    assert result.cross_singular_values == (0.0, 0.0)
    assert result.numerical_rank == 0
    assert result.transformation is None
    assert result.optimal_squared_residual == 0.0
    assert result.optimal_residual_frobenius_norm == 0.0


def test_repeated_positive_singular_values_still_identify_unique_polar_factor() -> None:
    identity = (
        (1.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 1.0 + 0.0j),
    )

    result = NumpyComplexUnitaryProcrustesSolver().solve(
        ComplexUnitaryProcrustesDefinition(identity, identity)
    )

    assert result.status is ComplexUnitaryProcrustesStatus.UNIQUE
    assert result.cross_singular_values == (1.0, 1.0)
    assert result.transformation == identity
