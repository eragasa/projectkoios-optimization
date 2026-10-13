import pytest

from projectkoios.optimization.engines.records import ObjectiveFeedback, ObjectiveValue

pytestmark = pytest.mark.unit


def test_objective_feedback_preserves_residual_objective_order() -> None:
    feedback = ObjectiveFeedback("c", 2.0, (ObjectiveValue("q", 1.0),))

    assert feedback.objectives[0].value == 1.0


def test_objective_feedback_rejects_duplicate_objective_names() -> None:
    with pytest.raises(ValueError, match="objective names"):
        ObjectiveFeedback(
            "c",
            1.0,
            (ObjectiveValue("q", 1.0), ObjectiveValue("q", 2.0)),
        )
