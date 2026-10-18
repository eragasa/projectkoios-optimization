from __future__ import annotations

import sys
from typing import cast

import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    BatchedComplexUnitaryProcrustesSolver,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.numpy.procrustes import (
    NumpyComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesBackendEvidence,
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesSolver,
    ComplexUnitaryProcrustesTolerance,
    ScalarComplexUnitaryProcrustesSolver,
)


def test_solver_hierarchy_has_one_generic_behavior_owner() -> None:
    assert issubclass(
        ScalarComplexUnitaryProcrustesSolver,
        ComplexUnitaryProcrustesSolver,
    )
    assert issubclass(
        BatchedComplexUnitaryProcrustesSolver,
        ComplexUnitaryProcrustesSolver,
    )
    assert issubclass(
        NumpyComplexUnitaryProcrustesSolver,
        ScalarComplexUnitaryProcrustesSolver,
    )
    assert issubclass(
        NumpyBatchedComplexUnitaryProcrustesSolver,
        BatchedComplexUnitaryProcrustesSolver,
    )


def test_shared_tolerance_and_rank_policy(
    procrustes_rank_tolerance: ComplexUnitaryProcrustesTolerance,
) -> None:
    tolerance = procrustes_rank_tolerance

    assert ComplexUnitaryProcrustesSolver.validate_tolerance(tolerance) is tolerance
    assert ComplexUnitaryProcrustesSolver.rank_cutoff(tolerance, 2.0) == 1.25
    maximum = ComplexUnitaryProcrustesTolerance(
        rank_absolute=tolerance.rank_absolute,
        rank_relative=tolerance.rank_relative,
        validation_absolute=tolerance.validation_absolute,
        validation_relative=tolerance.validation_relative,
        rank_cutoff_rule=ComplexUnitaryProcrustesRankCutoffRule.MAXIMUM,
    )
    assert ComplexUnitaryProcrustesSolver.rank_cutoff(maximum, 2.0) == 1.0

    with pytest.raises(TypeError, match="ComplexUnitaryProcrustesTolerance"):
        ComplexUnitaryProcrustesSolver.validate_tolerance(
            cast(ComplexUnitaryProcrustesTolerance, "wrong")
        )


@pytest.mark.parametrize(
    ("singular_value_count", "epsilon_multiplier"),
    [
        pytest.param(1, 64.0, id="minimum-floor"),
        pytest.param(11, 104.0, id="dimension-scaled"),
    ],
)
def test_shared_residual_roundoff_coefficient_is_dimension_aware(
    singular_value_count: int,
    epsilon_multiplier: float,
) -> None:
    assert ComplexUnitaryProcrustesSolver.residual_roundoff_coefficient(
        singular_value_count
    ) == (epsilon_multiplier * sys.float_info.epsilon)

    with pytest.raises(ValueError, match="positive"):
        ComplexUnitaryProcrustesSolver.residual_roundoff_coefficient(0)


@pytest.mark.parametrize(
    ("identifier", "version"),
    [pytest.param("svd", "1", id="executed-backend")],
)
def test_shared_backend_evidence_distinguishes_execution(
    identifier: str,
    version: str,
) -> None:
    assert ComplexUnitaryProcrustesSolver.validate_backend_evidence(
        identifier,
        version,
    ) == ComplexUnitaryProcrustesBackendEvidence(identifier, version)
    assert (
        ComplexUnitaryProcrustesSolver.validate_absent_backend_evidence(None, None)
        is None
    )

    with pytest.raises(ValueError, match="unexecuted backend"):
        ComplexUnitaryProcrustesSolver.validate_absent_backend_evidence("svd", "1")
    with pytest.raises(ValueError, match="empty"):
        ComplexUnitaryProcrustesSolver.validate_backend_evidence("", "1")
