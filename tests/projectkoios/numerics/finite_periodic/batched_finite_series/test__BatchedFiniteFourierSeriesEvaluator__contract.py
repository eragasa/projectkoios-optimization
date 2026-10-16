"""Contract tests for batched finite Fourier-series evaluators."""

from __future__ import annotations

from typing import cast

import pytest

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
    BatchedFiniteFourierSeriesEvaluationResult,
    BatchedFiniteFourierSeriesEvaluator,
)

pytestmark = pytest.mark.unit


class _ForgedEvaluator(BatchedFiniteFourierSeriesEvaluator):
    __slots__ = ("_result",)

    def __init__(self, result: object) -> None:
        self._result = result

    def _evaluate_validated(
        self,
        definition: BatchedFiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> BatchedFiniteFourierSeriesEvaluationResult:
        del definition
        return cast(BatchedFiniteFourierSeriesEvaluationResult, self._result)


class _UnexpectedCallEvaluator(BatchedFiniteFourierSeriesEvaluator):
    __slots__ = ()

    def _evaluate_validated(
        self,
        definition: BatchedFiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> BatchedFiniteFourierSeriesEvaluationResult:
        del definition
        raise AssertionError("trivial definitions must bypass the backend")


class _FailingEvaluator(BatchedFiniteFourierSeriesEvaluator):
    __slots__ = ()

    def _evaluate_validated(
        self,
        definition: BatchedFiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> BatchedFiniteFourierSeriesEvaluationResult:
        del definition
        raise KeyError("backend defect")


def _definition() -> BatchedFiniteFourierSeriesEvaluationDefinition:
    return BatchedFiniteFourierSeriesEvaluationDefinition(
        1,
        ((0.25,),),
        ((1,),),
        ((1.0 + 0.0j,), (2.0 - 1.0j,)),
    )


def _result(**changes: object) -> BatchedFiniteFourierSeriesEvaluationResult:
    values: dict[str, object] = {
        "dimension": 1,
        "point_count": 1,
        "term_count": 1,
        "channel_count": 2,
        "values": ((0.0 + 1.0j, 1.0 + 2.0j),),
        "backend_identifier": "test.backend",
        "backend_version": "1",
    }
    values.update(changes)
    return BatchedFiniteFourierSeriesEvaluationResult(**values)  # type: ignore[arg-type]


def _forged_result(**changes: object) -> BatchedFiniteFourierSeriesEvaluationResult:
    values: dict[str, object] = {
        "dimension": 1,
        "point_count": 1,
        "term_count": 1,
        "channel_count": 2,
        "values": ((0.0 + 1.0j, 1.0 + 2.0j),),
        "backend_identifier": "test.backend",
        "backend_version": "1",
    }
    values.update(changes)
    forged = object.__new__(BatchedFiniteFourierSeriesEvaluationResult)
    for name, value in values.items():
        object.__setattr__(forged, name, value)
    return forged


def test_evaluator_is_abstract_and_template_is_final() -> None:
    with pytest.raises(TypeError):
        BatchedFiniteFourierSeriesEvaluator()  # type: ignore[abstract]

    assert vars(BatchedFiniteFourierSeriesEvaluator.evaluate)["__final__"] is True


def test_evaluator_rejects_wrong_definition_type() -> None:
    with pytest.raises(
        TypeError, match="BatchedFiniteFourierSeriesEvaluationDefinition"
    ):
        _UnexpectedCallEvaluator().evaluate(object())  # type: ignore[arg-type]


def test_empty_points_return_shape_zero_by_channels_without_backend() -> None:
    result = _UnexpectedCallEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            2,
            (),
            ((1, -1),),
            ((1.0 + 0.0j,), (2.0 + 0.0j,)),
        )
    )

    assert result.point_count == 0
    assert result.channel_count == 2
    assert result.values == ()
    assert result.backend_identifier is None
    assert result.backend_version is None


def test_empty_channels_return_one_empty_row_per_point_without_backend() -> None:
    result = _UnexpectedCallEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            ((0.0,), (0.5,)),
            ((1,),),
            (),
        )
    )

    assert result.values == ((), ())
    assert result.backend_identifier is None


def test_empty_modes_return_exact_zero_matrix_without_backend() -> None:
    result = _UnexpectedCallEvaluator().evaluate(
        BatchedFiniteFourierSeriesEvaluationDefinition(
            1,
            ((0.0,), (0.5,)),
            (),
            ((), (), ()),
        )
    )

    assert result.values == (
        (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
    )
    assert result.backend_identifier is None


def test_evaluator_accepts_correlated_result_with_backend_evidence() -> None:
    result = _result()

    assert _ForgedEvaluator(result).evaluate(_definition()) is result


@pytest.mark.parametrize(
    ("result", "message"),
    (
        (object(), "must return BatchedFiniteFourierSeriesEvaluationResult"),
        (_result(dimension=2), "dimension"),
        (
            _result(point_count=2, values=((1.0 + 0.0j, 2.0 + 0.0j),) * 2),
            "point_count",
        ),
        (_result(term_count=2), "term_count"),
        (_result(channel_count=1, values=((1.0 + 0.0j,),)), "channel_count"),
        (
            _result(backend_identifier=None, backend_version=None),
            "backend evidence",
        ),
    ),
)
def test_evaluator_rejects_inconsistent_backend_result(
    result: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _ForgedEvaluator(result).evaluate(_definition())


@pytest.mark.parametrize(
    ("result", "exception", "message"),
    (
        (_forged_result(dimension=True), ValueError, "dimension"),
        (_forged_result(values=()), ValueError, "point shape"),
        (
            _forged_result(values=((1.0 + 0.0j,),)),
            ValueError,
            "channel shape",
        ),
        (
            _forged_result(values=((1.0, 2.0 + 0.0j),)),
            TypeError,
            "built-in complex",
        ),
        (
            _forged_result(backend_identifier=1),
            TypeError,
            "backend_identifier",
        ),
    ),
)
def test_evaluator_rejects_forged_malformed_backend_surface(
    result: BatchedFiniteFourierSeriesEvaluationResult,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        _ForgedEvaluator(result).evaluate(_definition())


def test_evaluator_rejects_forged_nonfinite_backend_values() -> None:
    forged = _forged_result(values=((complex(float("nan"), 0.0), 1.0 + 0.0j),))

    with pytest.raises(FloatingPointError, match="finite"):
        _ForgedEvaluator(forged).evaluate(_definition())


def test_evaluator_propagates_undeclared_backend_exception() -> None:
    with pytest.raises(KeyError, match="backend defect"):
        _FailingEvaluator().evaluate(_definition())
