import copy
from fractions import Fraction
from typing import Any

import pytest

from projectkoios.optimization import (
    BoundedCoordinatePatternOptimizer,
    BoundedUniformOptimizer,
    Candidate,
    CheckpointError,
    Measurement,
    ObjectiveFeedback,
    ObjectiveValue,
    OptimizerCheckpoint,
    ParameterBound,
    ParameterSpace,
    PatternSearchSettings,
    WeightedLoss,
    WeightedLossTerm,
)

HUGE_INTEGER = 10**400
MAX_U32 = 4_294_967_295
MAX_IDENTIFIER_LENGTH = 256


def pattern_settings(
    max_evaluations: int = 4,
    max_iterations: int = 2,
) -> PatternSearchSettings:
    return PatternSearchSettings(max_evaluations, max_iterations, 0.25, 0.5, 0.01)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ParameterBound("x", 0.0, HUGE_INTEGER),
        lambda: Candidate("candidate", "run", 0, (HUGE_INTEGER,)),
        lambda: Measurement("q", HUGE_INTEGER),
        lambda: WeightedLossTerm("q", HUGE_INTEGER, 1.0),
        lambda: WeightedLossTerm("q", 0.0, HUGE_INTEGER),
        lambda: WeightedLoss(HUGE_INTEGER, (ObjectiveValue("q", 0.0),)),
        lambda: PatternSearchSettings(2, 1, HUGE_INTEGER, 0.5, 0.1),
        lambda: PatternSearchSettings(2, 1, 0.5, HUGE_INTEGER, 0.1),
        lambda: PatternSearchSettings(2, 1, 0.5, 0.5, HUGE_INTEGER),
    ],
)
def test_huge_real_values_are_closed_as_value_errors(factory: Any) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("value", [True, "1", object()])
def test_public_float_fields_reject_bool_and_non_real_values(value: object) -> None:
    with pytest.raises(TypeError):
        Measurement("q", value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PatternSearchSettings(2, 1, value, 0.5, 0.1)  # type: ignore[arg-type]


def test_real_values_are_normalized_to_builtin_float() -> None:
    measurement = Measurement("q", Fraction(1, 2))  # type: ignore[arg-type]
    assert type(measurement.value) is float
    assert measurement.value == 0.5


def test_bounds_reject_nonfinite_representable_width() -> None:
    with pytest.raises(ValueError, match="width"):
        ParameterBound("x", -1e308, 1e308)


@pytest.mark.parametrize(
    "settings",
    [
        (HUGE_INTEGER, 1),
        (1, HUGE_INTEGER),
        (MAX_U32 + 1, 1),
        (1, MAX_U32 + 1),
    ],
)
def test_pattern_integer_budgets_are_explicitly_bounded(
    settings: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        pattern_settings(*settings)


def uniform_checkpoint_with_feedback() -> OptimizerCheckpoint:
    optimizer = BoundedUniformOptimizer(
        ParameterSpace((ParameterBound("x", 0.0, 1.0),)),
        seed=1,
        max_evaluations=2,
        run_id="uniform.numeric",
    )
    candidate = optimizer.ask()
    optimizer.tell(
        ObjectiveFeedback(
            candidate.candidate_id,
            1.0,
            (ObjectiveValue("q", 1.0),),
        )
    )
    return optimizer.checkpoint()


def pattern_checkpoint_with_pending() -> OptimizerCheckpoint:
    optimizer = BoundedCoordinatePatternOptimizer(
        ParameterSpace((ParameterBound("x", -1.0, 1.0),)),
        (0.0,),
        pattern_settings(),
        run_id="pattern.numeric",
    )
    first = optimizer.ask()
    optimizer.tell(ObjectiveFeedback(first.candidate_id, 1.0))
    optimizer.ask()
    return optimizer.checkpoint()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("space", 0, "upper"), HUGE_INTEGER),
        (("space", 0, "lower"), -1e308),
        (("feedback", 0, "loss"), HUGE_INTEGER),
        (("feedback", 0, "objectives", 0, "value"), HUGE_INTEGER),
        (("seed",), HUGE_INTEGER),
    ],
)
def test_digest_valid_uniform_checkpoint_numeric_corruption_is_closed(
    path: tuple[str | int, ...], value: int | float
) -> None:
    payload: Any = copy.deepcopy(uniform_checkpoint_with_feedback().payload_object())
    if path == ("space", 0, "lower"):
        payload["space"][0]["upper"] = 1e308
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    checkpoint = OptimizerCheckpoint.create("bounded-uniform:1", payload)
    with pytest.raises(CheckpointError):
        BoundedUniformOptimizer.from_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "path",
    [
        ("space", 0, "upper"),
        ("initial_values", 0),
        ("settings", "initial_step_fraction"),
        ("outstanding", "values", 0),
    ],
)
def test_digest_valid_pattern_checkpoint_numeric_corruption_is_closed(
    path: tuple[str | int, ...],
) -> None:
    payload: Any = copy.deepcopy(pattern_checkpoint_with_pending().payload_object())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = HUGE_INTEGER
    checkpoint = OptimizerCheckpoint.create("bounded-coordinate-pattern:1", payload)
    with pytest.raises(CheckpointError):
        BoundedCoordinatePatternOptimizer.from_checkpoint(checkpoint)


def test_measurement_and_weighted_term_share_exact_name_boundary() -> None:
    accepted = "m" * MAX_IDENTIFIER_LENGTH
    rejected = accepted + "m"
    assert Measurement(accepted, 0.0).name == accepted
    assert WeightedLossTerm(accepted, 0.0, 1.0).measurement == accepted
    with pytest.raises(ValueError):
        Measurement(rejected, 0.0)
    with pytest.raises(ValueError):
        WeightedLossTerm(rejected, 0.0, 1.0)


@pytest.mark.parametrize(
    "engine_type",
    [BoundedUniformOptimizer, BoundedCoordinatePatternOptimizer],
)
def test_run_id_capacity_boundary_is_checked_at_engine_construction(
    engine_type: type[BoundedUniformOptimizer]
    | type[BoundedCoordinatePatternOptimizer],
) -> None:
    accepted = "r" * 235
    rejected = accepted + "r"
    space = ParameterSpace(())
    if engine_type is BoundedUniformOptimizer:
        optimizer = engine_type(space, seed=0, max_evaluations=MAX_U32, run_id=accepted)
        with pytest.raises(ValueError):
            engine_type(space, seed=0, max_evaluations=MAX_U32, run_id=rejected)
        restored = BoundedUniformOptimizer.from_checkpoint(
            OptimizerCheckpoint.from_json(optimizer.checkpoint().to_json())
        )
    else:
        settings = pattern_settings(MAX_U32, 0)
        optimizer = engine_type(space, (), settings, run_id=accepted)
        with pytest.raises(ValueError):
            engine_type(space, (), settings, run_id=rejected)
        restored = BoundedCoordinatePatternOptimizer.from_checkpoint(
            OptimizerCheckpoint.from_json(optimizer.checkpoint().to_json())
        )
    assert restored.run_id == accepted


def test_single_evaluation_budget_uses_full_candidate_id_capacity() -> None:
    run_id = "r" * 237
    uniform = BoundedUniformOptimizer(
        ParameterSpace(()), seed=0, max_evaluations=1, run_id=run_id
    )
    pattern = BoundedCoordinatePatternOptimizer(
        ParameterSpace(()), (), pattern_settings(1, 0), run_id=run_id
    )
    assert len(uniform.ask().candidate_id) == MAX_IDENTIFIER_LENGTH
    assert len(pattern.ask().candidate_id) == MAX_IDENTIFIER_LENGTH
