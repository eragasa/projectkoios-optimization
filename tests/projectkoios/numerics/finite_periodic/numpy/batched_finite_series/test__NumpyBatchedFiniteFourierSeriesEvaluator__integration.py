"""Integration tests for dense NumPy batched finite Fourier evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
)
from projectkoios.numerics.finite_periodic.numpy.batched_finite_series import (
    NumpyBatchedFiniteFourierSeriesEvaluator,
)

pytestmark = pytest.mark.integration


def test_backend_returns_point_major_values_and_evidence() -> None:
    result = NumpyBatchedFiniteFourierSeriesEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            ((0.0,), (0.25,), (0.5,)),
            ((0,), (1,)),
            (
                (2.0 - 0.5j, 1.0 + 0.0j),
                (-1.0 + 0.0j, 0.0 + 2.0j),
            ),
        )
    )

    expected = (
        (3.0 - 0.5j, -1.0 + 2.0j),
        (2.0 + 0.5j, -3.0 + 0.0j),
        (1.0 - 0.5j, -1.0 - 2.0j),
    )
    assert np.asarray(result.values) == pytest.approx(
        np.asarray(expected), abs=3.0e-15, rel=0.0
    )
    assert result.backend_identifier == "numpy.matmul"
    assert result.backend_version == np.__version__


def test_backend_forms_phase_matrix_and_channel_reduction_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projectkoios.numerics.finite_periodic.numpy.batched_finite_series as module

    original_exp = module.np.exp
    original_matmul = module.np.matmul
    exp_shapes: list[tuple[int, ...]] = []
    matmul_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def recording_exp(values: np.ndarray) -> np.ndarray:
        exp_shapes.append(values.shape)
        return original_exp(values)

    def recording_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        matmul_shapes.append((left.shape, right.shape))
        return original_matmul(left, right)

    monkeypatch.setattr(module.np, "exp", recording_exp)
    monkeypatch.setattr(module.np, "matmul", recording_matmul)

    result = module.NumpyBatchedFiniteFourierSeriesEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            2,
            ((0.0, 0.0), (0.25, 0.5), (-0.5, 0.125)),
            ((0, 0), (1, -1), (-2, 3), (4, 1)),
            (
                (1.0 + 0.0j,) * 4,
                (2.0 + 1.0j,) * 4,
                (-1.0 + 0.5j,) * 4,
                (0.25 - 2.0j,) * 4,
                (3.0 + 0.0j,) * 4,
            ),
        )
    )

    assert result.point_count == 3
    assert result.channel_count == 5
    assert exp_shapes == [(3, 4)]
    assert matmul_shapes == [((3, 4), (4, 5))]


def test_duplicate_and_negative_modes_remain_distinct_columns() -> None:
    result = NumpyBatchedFiniteFourierSeriesEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            ((0.125,),),
            ((1,), (-1,), (1,)),
            (
                (1.0 + 2.0j, 1.0 - 2.0j, -0.5 + 0.25j),
                (0.0 + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j),
            ),
        )
    )

    phases = np.exp(
        2.0j
        * np.pi
        * np.asarray([0.125])[:, None]
        * np.asarray([1.0, -1.0, 1.0])[None, :]
    )
    coefficient_channels = (
        (1.0 + 2.0j, 1.0 - 2.0j, -0.5 + 0.25j),
        (0.0 + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j),
    )
    expected = phases @ np.asarray(coefficient_channels).T

    assert coefficient_channels[0][0] != coefficient_channels[0][2]
    assert np.asarray(result.values) == pytest.approx(expected, abs=3.0e-15, rel=0.0)


def test_coordinate_components_accumulate_in_declared_order() -> None:
    result = NumpyBatchedFiniteFourierSeriesEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            3,
            ((1.0e16, 0.25, -1.0e16),),
            ((1, 1, 1),),
            ((1.0 + 0.0j,), (0.0 + 2.0j,)),
        )
    )

    assert result.values == ((1.0 + 0.0j, 0.0 + 2.0j),)


def test_nonfinite_phase_intermediate_fails_closed() -> None:
    definition = BatchedFiniteFourierSeriesEvaluationDefinition(
        1,
        ((float.fromhex("0x1.fffffffffffffp+1023"),),),
        ((2,),),
        ((1.0 + 0.0j,),),
    )

    with pytest.raises(FloatingPointError):
        NumpyBatchedFiniteFourierSeriesEvaluator().evaluate(definition)
