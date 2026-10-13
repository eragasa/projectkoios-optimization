import pytest

from projectkoios.optimization.engines.records import OptimizerUpdate

pytestmark = pytest.mark.unit


def test_optimizer_update_preserves_best_candidate_identity() -> None:
    update = OptimizerUpdate("c", 1, "best", False, None)

    assert update.best_candidate_id == "best"
