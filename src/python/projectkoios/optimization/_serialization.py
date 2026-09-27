"""Internal strict JSON record codecs."""

from __future__ import annotations

from ._validation import finite_float, integer
from .checkpoint import JsonValue
from .exceptions import CheckpointError
from .records import (
    Candidate,
    ObjectiveFeedback,
    ObjectiveValue,
    ParameterBound,
    ParameterSpace,
)


def object_value(value: JsonValue, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be an object")
    return value


def list_value(value: JsonValue, name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise CheckpointError(f"{name} must be an array")
    return value


def string_value(value: JsonValue, name: str) -> str:
    if type(value) is not str:
        raise CheckpointError(f"{name} must be a string")
    return value


def integer_value(value: JsonValue, name: str) -> int:
    try:
        return integer(value, name)
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"{name} must be an integer") from error


def number_value(value: JsonValue, name: str) -> float:
    if type(value) not in (int, float):
        raise CheckpointError(f"{name} must be a number")
    try:
        return finite_float(value, name)
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"{name} must be a finite number") from error


def required(
    value: dict[str, JsonValue], keys: set[str], name: str
) -> dict[str, JsonValue]:
    if set(value) != keys:
        raise CheckpointError(f"{name} has incompatible fields")
    return value


def encode_space(space: ParameterSpace) -> list[JsonValue]:
    return [
        {"lower": bound.lower, "name": bound.name, "upper": bound.upper}
        for bound in space.bounds
    ]


def decode_space(value: JsonValue) -> ParameterSpace:
    bounds: list[ParameterBound] = []
    for item in list_value(value, "space"):
        data = required(
            object_value(item, "bound"), {"lower", "name", "upper"}, "bound"
        )
        bounds.append(
            ParameterBound(
                string_value(data["name"], "bound.name"),
                number_value(data["lower"], "bound.lower"),
                number_value(data["upper"], "bound.upper"),
            )
        )
    return ParameterSpace(tuple(bounds))


def encode_candidate(candidate: Candidate) -> dict[str, JsonValue]:
    return {
        "candidate_id": candidate.candidate_id,
        "optimizer_run_id": candidate.optimizer_run_id,
        "sequence": candidate.sequence,
        "values": list(candidate.values),
    }


def decode_candidate(value: JsonValue) -> Candidate:
    data = required(
        object_value(value, "candidate"),
        {"candidate_id", "optimizer_run_id", "sequence", "values"},
        "candidate",
    )
    values = tuple(
        number_value(item, "candidate coordinate")
        for item in list_value(data["values"], "candidate.values")
    )
    return Candidate(
        string_value(data["candidate_id"], "candidate.candidate_id"),
        string_value(data["optimizer_run_id"], "candidate.optimizer_run_id"),
        integer_value(data["sequence"], "candidate.sequence"),
        values,
    )


def encode_feedback(feedback: ObjectiveFeedback) -> dict[str, JsonValue]:
    return {
        "candidate_id": feedback.candidate_id,
        "loss": feedback.loss,
        "objectives": [
            {"name": objective.name, "value": objective.value}
            for objective in feedback.objectives
        ],
    }


def decode_feedback(value: JsonValue) -> ObjectiveFeedback:
    data = required(
        object_value(value, "feedback"),
        {"candidate_id", "loss", "objectives"},
        "feedback",
    )
    objectives: list[ObjectiveValue] = []
    for item in list_value(data["objectives"], "feedback.objectives"):
        objective = required(
            object_value(item, "objective"), {"name", "value"}, "objective"
        )
        objectives.append(
            ObjectiveValue(
                string_value(objective["name"], "objective.name"),
                number_value(objective["value"], "objective.value"),
            )
        )
    return ObjectiveFeedback(
        string_value(data["candidate_id"], "feedback.candidate_id"),
        number_value(data["loss"], "feedback.loss"),
        tuple(objectives),
    )
