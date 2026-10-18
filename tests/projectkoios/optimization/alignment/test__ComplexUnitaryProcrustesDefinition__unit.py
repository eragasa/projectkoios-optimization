"""Unit tests for complex-unitary Procrustes inputs and tolerances."""

from __future__ import annotations

from typing import Any, cast

import pytest

from projectkoios.optimization.alignment.procrustes import (
    DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE,
    ComplexUnitaryProcrustesDefinition,
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesTolerance,
)

pytestmark = pytest.mark.unit


def test_default_tolerance_separates_rank_and_validation_controls() -> None:
    tolerance = DEFAULT_COMPLEX_UNITARY_PROCRUSTES_TOLERANCE

    assert tolerance.rank_absolute == 0.0
    assert tolerance.rank_relative == 1.0e-12
    assert tolerance.validation_absolute == 1.0e-12
    assert tolerance.validation_relative == 1.0e-10
    assert tolerance.rank_cutoff_rule is ComplexUnitaryProcrustesRankCutoffRule.ADDITIVE


@pytest.mark.parametrize("field", ["rank_absolute", "validation_absolute"])
@pytest.mark.parametrize("value", [True, 1, float("nan"), float("inf"), -1.0])
def test_absolute_tolerances_reject_invalid_values(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "rank_absolute": 1.0e-12,
        "rank_relative": 1.0e-10,
        "validation_absolute": 1.0e-12,
        "validation_relative": 1.0e-10,
    }
    arguments[field] = value

    with pytest.raises((TypeError, ValueError)):
        ComplexUnitaryProcrustesTolerance(**cast(Any, arguments))


@pytest.mark.parametrize("field", ["rank_relative", "validation_relative"])
@pytest.mark.parametrize(
    "value",
    [True, 1, float("nan"), float("inf"), -1.0, 1.0, 2.0],
)
def test_relative_tolerances_reject_invalid_values(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "rank_absolute": 1.0e-12,
        "rank_relative": 1.0e-10,
        "validation_absolute": 1.0e-12,
        "validation_relative": 1.0e-10,
    }
    arguments[field] = value

    with pytest.raises((TypeError, ValueError)):
        ComplexUnitaryProcrustesTolerance(**cast(Any, arguments))


def test_rank_cutoff_rule_requires_exact_closed_enum() -> None:
    with pytest.raises(TypeError, match="ComplexUnitaryProcrustesRankCutoffRule"):
        ComplexUnitaryProcrustesTolerance(
            rank_cutoff_rule=cast(Any, "maximum"),
        )


def test_each_tolerance_pair_requires_a_positive_component() -> None:
    with pytest.raises(ValueError, match="numerical-rank"):
        ComplexUnitaryProcrustesTolerance(0.0, 0.0, 1.0e-12, 0.0)
    with pytest.raises(ValueError, match="validation tolerance"):
        ComplexUnitaryProcrustesTolerance(1.0e-12, 0.0, 0.0, 0.0)


def test_definition_retains_direction_and_shape() -> None:
    source = (
        (1.0 + 0.0j, 0.0 + 1.0j),
        (2.0 - 1.0j, 3.0 + 0.0j),
        (0.0 + 0.0j, -1.0 + 2.0j),
    )
    target = (
        (0.0 + 1.0j, 1.0 + 0.0j),
        (3.0 + 0.0j, 2.0 + 0.0j),
        (1.0 - 1.0j, 0.0 + 0.0j),
    )

    definition = ComplexUnitaryProcrustesDefinition(source, target)

    assert definition.source_matrix == source
    assert definition.target_matrix == target
    assert definition.row_count == 3
    assert definition.column_count == 2


@pytest.mark.parametrize(
    "source, target, error, match",
    [
        ([], ((1.0 + 0.0j,),), TypeError, "source_matrix.*exact tuple"),
        ((), ((1.0 + 0.0j,),), ValueError, "source_matrix.*nonempty"),
        (((1.0,),), ((1.0 + 0.0j,),), TypeError, "exact built-in complex"),
        (
            ((complex(float("nan"), 0.0),),),
            ((1.0 + 0.0j,),),
            ValueError,
            "finite",
        ),
        (
            ((1.0 + 0.0j,),),
            ((1.0 + 0.0j,), (2.0 + 0.0j,)),
            ValueError,
            "row count",
        ),
        (
            ((1.0 + 0.0j, 2.0 + 0.0j),),
            ((1.0 + 0.0j,),),
            ValueError,
            "column count",
        ),
    ],
)
def test_definition_rejects_malformed_or_mismatched_matrices(
    source: object,
    target: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ComplexUnitaryProcrustesDefinition(
            cast(Any, source),
            cast(Any, target),
        )


def test_definition_requires_nominal_tolerance() -> None:
    with pytest.raises(TypeError, match="ComplexUnitaryProcrustesTolerance"):
        ComplexUnitaryProcrustesDefinition(
            ((1.0 + 0.0j,),),
            ((1.0 + 0.0j,),),
            cast(Any, object()),
        )
