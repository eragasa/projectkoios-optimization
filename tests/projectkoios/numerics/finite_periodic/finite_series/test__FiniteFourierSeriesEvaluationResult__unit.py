"""Unit tests for finite Fourier-series evaluation results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationResult,
)

pytestmark = pytest.mark.unit


def test_result_retains_finite_structural_metadata_and_values() -> None:
    result = FiniteFourierSeriesEvaluationResult(
        2,
        2,
        3,
        (1.0 + 2.0j, -3.0 + 0.5j),
    )

    assert result.dimension == 2
    assert result.point_count == 2
    assert result.term_count == 3
    assert result.values == (1.0 + 2.0j, -3.0 + 0.5j)
    with pytest.raises(FrozenInstanceError):
        result.point_count = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("dimension", "point_count", "term_count", "values", "error", "message"),
    (
        (0, 0, 0, (), ValueError, "dimension"),
        (True, 0, 0, (), TypeError, "dimension"),
        (1, -1, 0, (), ValueError, "point_count"),
        (1, True, 0, (), TypeError, "point_count"),
        (1, 0, -1, (), ValueError, "term_count"),
        (1, 0, True, (), TypeError, "term_count"),
        (1, 0, 0, [], TypeError, "values"),
        (1, 1, 0, (), ValueError, "values length"),
        (1, 1, 0, (1.0,), TypeError, "complex"),
        (
            1,
            1,
            0,
            (complex(float("nan"), 0.0),),
            ValueError,
            "finite",
        ),
    ),
)
def test_result_rejects_invalid_state(
    dimension: object,
    point_count: object,
    term_count: object,
    values: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        FiniteFourierSeriesEvaluationResult(
            cast(Any, dimension),
            cast(Any, point_count),
            cast(Any, term_count),
            cast(Any, values),
        )
