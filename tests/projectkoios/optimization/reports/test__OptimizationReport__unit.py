import pytest

from projectkoios.optimization.engines.records import (
    Candidate,
    ObjectiveFeedback,
    ObjectiveValue,
)
from projectkoios.optimization.reports.reports import ScoredResult, build_report

pytestmark = pytest.mark.unit


def test_report_uses_loss_identity_tie_break_and_pareto_dominance() -> None:
    def scored(
        identity: str,
        loss: float,
        objectives: tuple[float, float],
    ) -> ScoredResult:
        candidate = Candidate(identity, "run", int(identity[-1]), (0.0,))
        feedback = ObjectiveFeedback(
            identity,
            loss,
            (
                ObjectiveValue("a", objectives[0]),
                ObjectiveValue("b", objectives[1]),
            ),
        )
        return ScoredResult(candidate, feedback)

    report = build_report(
        (
            scored("c2", 1.0, (3.0, 4.0)),
            scored("c1", 1.0, (2.0, 2.0)),
            scored("c0", 3.0, (1.0, 3.0)),
        )
    )

    assert tuple(item.candidate.candidate_id for item in report.ranked) == (
        "c1",
        "c2",
        "c0",
    )
    assert report.pareto_candidate_ids == ("c0", "c1")
