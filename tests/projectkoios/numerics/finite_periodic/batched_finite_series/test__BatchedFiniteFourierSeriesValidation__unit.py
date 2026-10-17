"""Direct tests for named batched finite-series template operations."""

from __future__ import annotations

import pytest

from projectkoios.numerics.finite_periodic.batched_finite_series import (
    BatchedFiniteFourierSeriesEvaluationDefinition,
    BatchedFiniteFourierSeriesEvaluationResult,
    _require_backend_evidence,
    _trivial_values,
    _validate_result_metadata,
    _validate_result_values,
)

pytestmark = pytest.mark.unit


def _nontrivial_definition() -> BatchedFiniteFourierSeriesEvaluationDefinition:
    return BatchedFiniteFourierSeriesEvaluationDefinition(
        1,
        ((0.25,),),
        ((1,),),
        ((1.0 + 0.0j,), (2.0 - 1.0j,)),
    )


def _nontrivial_result() -> BatchedFiniteFourierSeriesEvaluationResult:
    return BatchedFiniteFourierSeriesEvaluationResult(
        dimension=1,
        point_count=1,
        term_count=1,
        channel_count=2,
        values=((0.0 + 1.0j, 1.0 + 2.0j),),
        backend_identifier="test.backend",
        backend_version="1",
    )


def _forged_result(
    **changes: object,
) -> BatchedFiniteFourierSeriesEvaluationResult:
    source = _nontrivial_result()
    values: dict[str, object] = {
        "dimension": source.dimension,
        "point_count": source.point_count,
        "term_count": source.term_count,
        "channel_count": source.channel_count,
        "values": source.values,
        "backend_identifier": source.backend_identifier,
        "backend_version": source.backend_version,
    }
    values.update(changes)
    forged = object.__new__(BatchedFiniteFourierSeriesEvaluationResult)
    for name, value in values.items():
        object.__setattr__(forged, name, value)
    return forged


@pytest.mark.parametrize(
    ("definition", "expected"),
    (
        (
            BatchedFiniteFourierSeriesEvaluationDefinition(
                1,
                (),
                ((1,),),
                ((1.0 + 0.0j,),),
            ),
            (),
        ),
        (
            BatchedFiniteFourierSeriesEvaluationDefinition(
                1,
                ((0.0,), (0.5,)),
                ((1,),),
                (),
            ),
            ((), ()),
        ),
        (
            BatchedFiniteFourierSeriesEvaluationDefinition(
                1,
                ((0.0,),),
                (),
                ((), ()),
            ),
            ((0.0 + 0.0j, 0.0 + 0.0j),),
        ),
        (_nontrivial_definition(), None),
    ),
)
def test_trivial_values_preserve_declared_shape(
    definition: BatchedFiniteFourierSeriesEvaluationDefinition,
    expected: tuple[tuple[complex, ...], ...] | None,
) -> None:
    assert _trivial_values(definition) == expected


def test_result_metadata_validation_correlates_all_counts() -> None:
    definition = _nontrivial_definition()
    result = _nontrivial_result()

    _validate_result_metadata(definition, result)

    with pytest.raises(ValueError, match="dimension"):
        _validate_result_metadata(definition, _forged_result(dimension=True))


def test_result_value_validation_checks_exact_type_and_finiteness() -> None:
    definition = _nontrivial_definition()
    result = _nontrivial_result()

    _validate_result_values(definition, result)

    nonfinite = _forged_result(values=((complex(float("nan"), 0.0), 1.0 + 2.0j),))
    with pytest.raises(FloatingPointError, match="finite"):
        _validate_result_values(definition, nonfinite)


def test_backend_evidence_is_required_for_nontrivial_results() -> None:
    _require_backend_evidence(_nontrivial_result())

    absent = _forged_result(backend_identifier=None, backend_version=None)
    with pytest.raises(ValueError, match="backend evidence"):
        _require_backend_evidence(absent)
