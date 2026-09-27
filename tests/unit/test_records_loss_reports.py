import math

import pytest

from projectkoios.optimization import (
    Candidate,
    Measurement,
    ObjectiveFeedback,
    ObjectiveValue,
    OptimizerDescriptor,
    OptimizerIdentity,
    ParameterBound,
    ParameterSpace,
    RawEvaluation,
    ScoredResult,
    WeightedLossTerm,
    WeightedSquaredLoss,
    build_report,
)


def test_identity_and_descriptor_are_immutable_and_versioned() -> None:
    identity = OptimizerIdentity("bounded.uniform", 1)
    descriptor = OptimizerDescriptor(identity, "splitmix64:1", 1)
    assert identity.value == "bounded.uniform:1"
    assert descriptor.identity is identity
    with pytest.raises(AttributeError):
        identity.version = 2  # type: ignore[misc]


@pytest.mark.parametrize("family", ["", "Upper", "bad space", "bad..part"])
def test_identity_rejects_nonportable_family(family: str) -> None:
    with pytest.raises(ValueError):
        OptimizerIdentity(family, 1)


def test_zero_dimensional_space_and_fixed_coordinate_are_explicit() -> None:
    empty = ParameterSpace(())
    fixed = ParameterSpace((ParameterBound("fixed", 2.0, 2.0),))
    assert empty.dimension == 0
    assert empty.admits(())
    assert fixed.admits((2.0,))
    assert not fixed.admits((2.1,))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(2.0, 1.0), (math.nan, 1.0), (0.0, math.inf)],
)
def test_invalid_bounds_fail_closed(lower: float, upper: float) -> None:
    with pytest.raises(ValueError):
        ParameterBound("x", lower, upper)


@pytest.mark.parametrize("weight", [0.0, -1.0, math.nan, math.inf])
def test_nonpositive_or_nonfinite_weight_is_rejected(weight: float) -> None:
    with pytest.raises(ValueError):
        WeightedLossTerm("q", 0.0, weight)


def test_weighted_loss_preserves_term_order_and_rejects_missing_values() -> None:
    loss = WeightedSquaredLoss(
        (
            WeightedLossTerm("x", 1.0, 1.0),
            WeightedLossTerm("y", -2.0, 2.0),
        )
    )
    result = loss.evaluate(
        RawEvaluation("c", (Measurement("y", 0.0), Measurement("x", 3.0)))
    )
    assert result.value == 12.0
    assert result.objectives == (ObjectiveValue("x", 2.0), ObjectiveValue("y", 2.0))
    with pytest.raises(ValueError, match="missing measurement"):
        loss.evaluate(RawEvaluation("c", (Measurement("x", 1.0),)))


def test_report_uses_loss_identity_tie_break_and_pareto_dominance() -> None:
    def scored(
        identity: str, loss: float, objectives: tuple[float, float]
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
