import pytest

from projectkoios.optimization.engines.records import Measurement, RawEvaluation

pytestmark = pytest.mark.unit


def test_raw_evaluation_preserves_measurement_order() -> None:
    evaluation = RawEvaluation("c", (Measurement("q", 1.0),))

    assert evaluation.measurements[0].name == "q"


def test_raw_evaluation_rejects_duplicate_measurement_names() -> None:
    with pytest.raises(ValueError, match="measurement names"):
        RawEvaluation("c", (Measurement("q", 1.0), Measurement("q", 2.0)))
