"""Contract tests for nominal finite Fourier-series evaluators."""

from __future__ import annotations

from typing import cast

import pytest

from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
    FiniteFourierSeriesEvaluationResult,
    FiniteFourierSeriesEvaluator,
)

pytestmark = pytest.mark.unit


class _ForgedEvaluator(FiniteFourierSeriesEvaluator):
    __slots__ = ("_result",)

    def __init__(self, result: object) -> None:
        self._result = result

    def _evaluate_validated(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult:
        del definition
        return cast(FiniteFourierSeriesEvaluationResult, self._result)


class _FailingEvaluator(FiniteFourierSeriesEvaluator):
    __slots__ = ()

    def _evaluate_validated(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult:
        del definition
        raise KeyError("backend defect")


class _UnexpectedCallEvaluator(FiniteFourierSeriesEvaluator):
    __slots__ = ()

    def _evaluate_validated(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult:
        del definition
        raise AssertionError("trivial definitions must bypass the backend")


def _definition() -> FiniteFourierSeriesEvaluationDefinition:
    return FiniteFourierSeriesEvaluationDefinition(
        1,
        ((0.25,),),
        ((1,),),
        (1.0 + 0.0j,),
    )


def test_evaluator_is_abstract_and_public_template_is_final() -> None:
    with pytest.raises(TypeError):
        FiniteFourierSeriesEvaluator()  # type: ignore[abstract]

    assert vars(FiniteFourierSeriesEvaluator.evaluate)["__final__"] is True


def test_evaluator_rejects_wrong_definition_type() -> None:
    with pytest.raises(TypeError, match="FiniteFourierSeriesEvaluationDefinition"):
        _UnexpectedCallEvaluator().evaluate(object())  # type: ignore[arg-type]


def test_evaluator_returns_empty_points_without_calling_backend() -> None:
    result = _UnexpectedCallEvaluator().evaluate(
        FiniteFourierSeriesEvaluationDefinition(
            3,
            (),
            ((1, 2, 3),),
            (2.0 + 1.0j,),
        )
    )

    assert result == FiniteFourierSeriesEvaluationResult(3, 0, 1, ())


def test_evaluator_returns_exact_zero_for_empty_terms_without_backend() -> None:
    result = _UnexpectedCallEvaluator().evaluate(
        FiniteFourierSeriesEvaluationDefinition(
            2,
            ((0.25, -0.5), (0.0, 0.0)),
            (),
            (),
        )
    )

    assert result == FiniteFourierSeriesEvaluationResult(
        2,
        2,
        0,
        (0.0 + 0.0j, 0.0 + 0.0j),
    )


def test_evaluator_accepts_structurally_correlated_backend_result() -> None:
    result = FiniteFourierSeriesEvaluationResult(1, 1, 1, (0.0 + 1.0j,))

    actual = _ForgedEvaluator(result).evaluate(_definition())

    assert actual is result


@pytest.mark.parametrize(
    ("result", "message"),
    (
        (object(), "must return FiniteFourierSeriesEvaluationResult"),
        (
            FiniteFourierSeriesEvaluationResult(2, 1, 1, (1.0 + 0.0j,)),
            "dimension",
        ),
        (
            FiniteFourierSeriesEvaluationResult(1, 2, 1, (1.0 + 0.0j,) * 2),
            "point_count",
        ),
        (
            FiniteFourierSeriesEvaluationResult(1, 1, 2, (1.0 + 0.0j,)),
            "term_count",
        ),
    ),
)
def test_evaluator_rejects_inconsistent_backend_result(
    result: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _ForgedEvaluator(result).evaluate(_definition())


def test_evaluator_rejects_forged_nonfinite_backend_values() -> None:
    forged = object.__new__(FiniteFourierSeriesEvaluationResult)
    object.__setattr__(forged, "dimension", 1)
    object.__setattr__(forged, "point_count", 1)
    object.__setattr__(forged, "term_count", 1)
    object.__setattr__(forged, "values", (complex(float("nan"), 0.0),))

    with pytest.raises(FloatingPointError, match="finite"):
        _ForgedEvaluator(forged).evaluate(_definition())


def test_evaluator_propagates_undeclared_backend_exception_unchanged() -> None:
    with pytest.raises(KeyError, match="backend defect"):
        _FailingEvaluator().evaluate(_definition())
