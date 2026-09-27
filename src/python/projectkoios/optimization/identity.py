"""Immutable optimizer identity and descriptor records."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FAMILY = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z", re.ASCII)
_MAX_U32 = 4_294_967_295


@dataclass(frozen=True, slots=True)
class OptimizerIdentity:
    """Stable family and positive contract version for an optimizer."""

    family: str
    version: int

    def __post_init__(self) -> None:
        if type(self.family) is not str:
            raise TypeError("family must be a built-in string")
        if len(self.family) > 128 or _FAMILY.fullmatch(self.family) is None:
            raise ValueError("family must be a portable lowercase optimizer identity")
        if type(self.version) is not int:
            raise TypeError("version must be a built-in integer excluding bool")
        if not 0 < self.version <= _MAX_U32:
            raise ValueError("version must be in the positive u32 range")

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
        if type(self.algorithm) is not str:
            raise TypeError("algorithm must be a built-in string")
        if not self.algorithm or self.algorithm.strip() != self.algorithm:
            raise ValueError("algorithm must be nonempty trimmed text")
        if type(self.checkpoint_schema_version) is not int:
            raise TypeError("checkpoint_schema_version must be a built-in integer")
        if self.checkpoint_schema_version != 1:
            raise ValueError("only checkpoint schema version 1 is supported")
