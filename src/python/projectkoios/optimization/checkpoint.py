"""Canonical, integrity-checked optimizer checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast

from ._validation import integer, text
from .exceptions import CheckpointError

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def canonical_json(value: JsonValue) -> str:
    """Return compact canonical JSON, rejecting non-finite values."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointError(
            "checkpoint payload is not canonical JSON data"
        ) from error


def parse_json(document: str) -> JsonValue:
    """Parse strict JSON and close malformed/non-finite input as checkpoint errors."""
    if type(document) is not str:
        raise TypeError("JSON document must be a built-in string")
    try:
        value: object = json.loads(document, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise CheckpointError("checkpoint contains invalid JSON") from error
    return cast(JsonValue, value)


@dataclass(frozen=True, slots=True)
class OptimizerCheckpoint:
    """Versioned canonical optimizer state with a payload digest."""

    schema_version: int
    optimizer_identity: str
    payload: str
    sha256: str

    def __post_init__(self) -> None:
        integer(self.schema_version, "schema_version")
        if self.schema_version != 1:
            raise CheckpointError("unsupported checkpoint schema version")
        try:
            text(self.optimizer_identity, "optimizer_identity")
        except ValueError as error:
            raise CheckpointError("optimizer_identity is invalid") from error
        if type(self.payload) is not str:
            raise TypeError("payload must be a built-in string")
        parsed = parse_json(self.payload)
        if canonical_json(parsed) != self.payload:
            raise CheckpointError("checkpoint payload is not canonical JSON")
        if type(self.sha256) is not str or _DIGEST.fullmatch(self.sha256) is None:
            raise CheckpointError("checkpoint digest must be lowercase SHA-256")
        actual = hashlib.sha256(self.payload.encode()).hexdigest()
        if actual != self.sha256:
            raise CheckpointError("checkpoint payload digest mismatch")

    @classmethod
    def create(
        cls, optimizer_identity: str, payload: dict[str, JsonValue]
    ) -> OptimizerCheckpoint:
        """Create a schema-one checkpoint from JSON-compatible state."""
        document = canonical_json(payload)
        return cls(
            schema_version=1,
            optimizer_identity=optimizer_identity,
            payload=document,
            sha256=hashlib.sha256(document.encode()).hexdigest(),
        )

    def payload_object(self) -> dict[str, JsonValue]:
        """Return the validated payload object."""
        value = parse_json(self.payload)
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint payload must be an object")
        return value

    def to_json(self) -> str:
        """Serialize the complete checkpoint canonically."""
        return canonical_json(
            {
                "optimizer_identity": self.optimizer_identity,
                "payload": self.payload,
                "schema_version": self.schema_version,
                "sha256": self.sha256,
            }
        )

    @classmethod
    def from_json(cls, document: str) -> OptimizerCheckpoint:
        """Parse and validate a complete checkpoint."""
        value = parse_json(document)
        if not isinstance(value, dict) or set(value) != {
            "optimizer_identity",
            "payload",
            "schema_version",
            "sha256",
        }:
            raise CheckpointError("checkpoint envelope has incompatible fields")
        schema = value["schema_version"]
        identity = value["optimizer_identity"]
        payload = value["payload"]
        digest = value["sha256"]
        if type(schema) is not int:
            raise CheckpointError("checkpoint schema_version must be an integer")
        if (
            type(identity) is not str
            or type(payload) is not str
            or type(digest) is not str
        ):
            raise CheckpointError("checkpoint envelope field types are invalid")
        return cls(schema, identity, payload, digest)
