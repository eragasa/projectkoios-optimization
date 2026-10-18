from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    BatchedComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesTolerance,
)


@dataclass(frozen=True, slots=True)
class _RankCutoffCase:
    tolerance: ComplexUnitaryProcrustesTolerance
    largest_singular_values: np.ndarray
    expected_cutoffs: np.ndarray


@pytest.fixture
def rank_cutoff_case(
    procrustes_rank_tolerance: ComplexUnitaryProcrustesTolerance,
) -> _RankCutoffCase:
    """Return a multidimensional additive-cutoff case."""
    return _RankCutoffCase(
        tolerance=procrustes_rank_tolerance,
        largest_singular_values=np.array(
            [[2.0, 4.0], [6.0, 8.0]],
            dtype=np.float64,
        ),
        expected_cutoffs=np.array(
            [[1.25, 2.25], [3.25, 4.25]],
            dtype=np.float64,
        ),
    )


def test_nominal_batched_solver_is_abstract() -> None:
    with pytest.raises(TypeError):
        BatchedComplexUnitaryProcrustesSolver()  # type: ignore[abstract]


def test_nominal_batched_solver_constructs_vectorized_additive_cutoffs(
    rank_cutoff_case: _RankCutoffCase,
) -> None:
    cutoffs = BatchedComplexUnitaryProcrustesSolver.rank_cutoffs(
        rank_cutoff_case.tolerance,
        rank_cutoff_case.largest_singular_values,
    )

    np.testing.assert_array_equal(cutoffs, rank_cutoff_case.expected_cutoffs)
    assert cutoffs.dtype == np.dtype(np.float64)


def test_nominal_batched_solver_constructs_vectorized_maximum_cutoffs() -> None:
    tolerance = ComplexUnitaryProcrustesTolerance(
        rank_absolute=0.25,
        rank_relative=0.5,
        rank_cutoff_rule=ComplexUnitaryProcrustesRankCutoffRule.MAXIMUM,
    )
    largest = np.array([[0.25, 2.0], [4.0, 8.0]], dtype=np.float64)

    cutoffs = BatchedComplexUnitaryProcrustesSolver.rank_cutoffs(tolerance, largest)

    np.testing.assert_array_equal(
        cutoffs,
        np.array([[0.25, 1.0], [2.0, 4.0]], dtype=np.float64),
    )
    assert cutoffs.dtype == np.dtype(np.float64)


@pytest.mark.parametrize(
    ("largest_singular_values", "error", "match"),
    [
        pytest.param([1.0], TypeError, "exact numpy.ndarray", id="array-like"),
        pytest.param(
            np.array([1.0], dtype=np.float32),
            TypeError,
            "exact dtype float64",
            id="wrong-dtype",
        ),
        pytest.param(
            np.array([np.nan], dtype=np.float64),
            ValueError,
            "finite and nonnegative",
            id="nonfinite",
        ),
        pytest.param(
            np.array([-1.0], dtype=np.float64),
            ValueError,
            "finite and nonnegative",
            id="negative",
        ),
    ],
)
def test_nominal_batched_solver_rejects_invalid_largest_singular_values(
    procrustes_rank_tolerance: ComplexUnitaryProcrustesTolerance,
    largest_singular_values: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        BatchedComplexUnitaryProcrustesSolver.rank_cutoffs(
            procrustes_rank_tolerance,
            largest_singular_values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("tolerance", "largest_singular_values"),
    [
        pytest.param(
            ComplexUnitaryProcrustesTolerance(
                rank_absolute=float(np.finfo(np.float64).max),
                rank_relative=0.5,
            ),
            np.array([np.finfo(np.float64).max], dtype=np.float64),
            id="absolute-plus-relative-overflow",
        )
    ],
)
def test_nominal_batched_solver_rejects_nonfinite_cutoff_arithmetic(
    tolerance: ComplexUnitaryProcrustesTolerance,
    largest_singular_values: np.ndarray,
) -> None:
    with (
        np.errstate(over="ignore"),
        pytest.raises(FloatingPointError, match="rank cutoff became nonfinite"),
    ):
        BatchedComplexUnitaryProcrustesSolver.rank_cutoffs(
            tolerance,
            largest_singular_values,
        )
