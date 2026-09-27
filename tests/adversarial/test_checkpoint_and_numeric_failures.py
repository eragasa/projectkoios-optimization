import json
import math

import pytest

from projectkoios.optimization import (
    BoundedUniformOptimizer,
    Candidate,
    CheckpointError,
    Measurement,
    ObjectiveFeedback,
    OptimizerCheckpoint,
    ParameterBound,
    ParameterSpace,
)


def checkpoint() -> OptimizerCheckpoint:
    return BoundedUniformOptimizer(
        ParameterSpace((ParameterBound("x", 0.0, 1.0),)),
        seed=1,
        max_evaluations=2,
        run_id="corruption",
    ).checkpoint()


def test_checkpoint_corruption_and_version_mismatch_fail_closed() -> None:
    valid = checkpoint()
    corrupted = valid.payload.replace('"max_evaluations":2', '"max_evaluations":3')
    with pytest.raises(CheckpointError, match="digest mismatch"):
        OptimizerCheckpoint(1, valid.optimizer_identity, corrupted, valid.sha256)
    envelope = json.loads(valid.to_json())
    envelope["schema_version"] = 2
    with pytest.raises(CheckpointError, match="unsupported"):
        OptimizerCheckpoint.from_json(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True)
        )


def test_checkpoint_identity_mismatch_fails_restore() -> None:
    valid = checkpoint()
    wrong = OptimizerCheckpoint.create("other:1", valid.payload_object())
    with pytest.raises(CheckpointError, match="identity"):
        BoundedUniformOptimizer.from_checkpoint(wrong)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_public_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        Candidate("candidate", "run", 0, (value,))
    with pytest.raises(ValueError):
        ObjectiveFeedback("candidate", value)
    with pytest.raises(ValueError):
        ParameterBound("x", 0.0, value)
    with pytest.raises(ValueError):
        Measurement("q", value)


def test_shape_mismatch_is_not_admitted() -> None:
    space = ParameterSpace((ParameterBound("x", 0.0, 1.0),))
    assert not space.admits(())
    assert not space.admits((0.5, 0.5))
