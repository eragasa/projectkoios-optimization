"""Unit tests for batched finite Fourier-series results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationResult,
)

pytestmark = pytest.mark.unit


def test_result_records_point_major_values_and_backend_evidence() -> None:
    result = BatchedFiniteFourierSeriesEvaluationResult(
        2,
        2,
        3,
        2,
        ((1.0 + 0.0j, 2.0 + 1.0j), (3.0 - 1.0j, 4.0 + 0.0j)),
        "numpy.matmul",
        "2.5.4",
    )

    assert result.values[0] == (1.0 + 0.0j, 2.0 + 1.0j)
    assert result.values[1] == (3.0 - 1.0j, 4.0 + 0.0j)
    assert result.backend_identifier == "numpy.matmul"
    assert result.backend_version == "2.5.4"


def test_result_permits_backend_free_empty_point_shape() -> None:
    result = BatchedFiniteFourierSeriesEvaluationResult(
        3,
        0,
        4,
        2,
        (),
        None,
        None,
    )

    assert result.point_count == 0
    assert result.channel_count == 2
    assert result.values == ()


def test_result_permits_point_rows_with_no_channels() -> None:
    result = BatchedFiniteFourierSeriesEvaluationResult(
        1,
        2,
        3,
        0,
        ((), ()),
        None,
        None,
    )

    assert result.values == ((), ())


@pytest.mark.parametrize(
    ("point_count", "channel_count", "values", "message"),
    (
        (2, 1, ((1.0 + 0.0j,),), "point_count"),
        (1, 2, ((1.0 + 0.0j,),), "channel_count"),
    ),
)
def test_result_rejects_inconsistent_point_major_shape(
    point_count: int,
    channel_count: int,
    values: tuple[tuple[complex, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BatchedFiniteFourierSeriesEvaluationResult(
            1,
            point_count,
            1,
            channel_count,
            values,
            None,
            None,
        )


@pytest.mark.parametrize(
    ("identifier", "version"),
    (("numpy.matmul", None), (None, "2.5.4"), ("", "2.5.4"), ("numpy.matmul", "")),
)
def test_result_rejects_unpaired_or_empty_backend_evidence(
    identifier: object,
    version: object,
) -> None:
    with pytest.raises(TypeError):
        BatchedFiniteFourierSeriesEvaluationResult(
            1,
            1,
            1,
            1,
            ((1.0 + 0.0j,),),
            identifier,  # type: ignore[arg-type]
            version,  # type: ignore[arg-type]
        )


def test_result_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        BatchedFiniteFourierSeriesEvaluationResult(
            1,
            1,
            1,
            1,
            ((complex(float("nan"), 0.0),),),
            "backend",
            "version",
        )


def test_result_is_frozen() -> None:
    result = BatchedFiniteFourierSeriesEvaluationResult(1, 0, 0, 0, (), None, None)

    with pytest.raises(FrozenInstanceError):
        result.point_count = 1  # type: ignore[misc]
