"""Dependency-free immutable numerical representation types."""

from __future__ import annotations

type FiniteComplexMatrix = tuple[tuple[complex, ...], ...]
"""Nonempty rectangular finite complex matrix after runtime validation."""

__all__ = ["FiniteComplexMatrix"]
