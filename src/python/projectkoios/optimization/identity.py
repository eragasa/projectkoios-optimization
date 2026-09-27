"""Immutable optimizer identity and descriptor records."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import (
    MAX_OPTIMIZER_FAMILY_LENGTH,
    OPTIMIZER_FAMILY_PATTERN,
    integer,
    text,
    u32,
)


@dataclass(frozen=True, slots=True)
class OptimizerIdentity:
    """Stable family and positive contract version for an optimizer."""

    family: str
    version: int

    def __post_init__(self) -> None:
        family = text(self.family, "family", maximum=MAX_OPTIMIZER_FAMILY_LENGTH)
        if OPTIMIZER_FAMILY_PATTERN.fullmatch(family) is None:
            raise ValueError("family must be a portable lowercase optimizer identity")
        u32(self.version, "version", allow_zero=False)

    @property
    def value(self) -> str:
        """Return the canonical family/version value."""
        return f"{self.family}:{self.version}"


@dataclass(frozen=True, slots=True)
class OptimizerDescriptor:
    """Identify an algorithm contract and its checkpoint schema."""

    identity: OptimizerIdentity
    algorithm: str
    checkpoint_schema_version: int

    def __post_init__(self) -> None:
        if type(self.identity) is not OptimizerIdentity:
            raise TypeError("identity must be OptimizerIdentity")
        text(self.algorithm, "algorithm")
        integer(self.checkpoint_schema_version, "checkpoint_schema_version")
        if self.checkpoint_schema_version != 1:
            raise ValueError("only checkpoint schema version 1 is supported")
