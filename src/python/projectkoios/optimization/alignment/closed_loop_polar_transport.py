"""Dependency-free contracts for closed-loop polar frame transport."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import final


class ClosedLoopPolarFrameTransportStatus(StrEnum):
    """Describe whether every neighbor and closure link qualified."""

    QUALIFIED = "qualified"
    RANK_DEFICIENT = "rank-deficient"


class ClosedLoopPolarFrameTransportLinkKind(StrEnum):
    """Identify an ordered neighbor link or the final sewn closure link."""

    NEIGHBOR = "neighbor"
    CLOSURE = "closure"


class ClosedLoopPolarFrameTransportLinkStatus(StrEnum):
    """Describe the progression state of one ordered link."""

    QUALIFIED = "qualified"
    RANK_DEFICIENT = "rank-deficient"
    UNATTEMPTED = "unattempted"


class ClosedLoopPolarFrameTransportPrincipalPhaseBranch(StrEnum):
    """Select one principal-phase branch for closure eigenvalues."""

    negative_pi_open = "(-pi, pi]"
    positive_pi_open = "[-pi, pi)"


class ClosedLoopPolarFrameTransportBackendOperation(StrEnum):
    """Identify the failed decomposition operation."""

    PROCRUSTES_SVD = "procrustes-svd"
    CLOSURE_SCHUR = "closure-schur"


class ClosedLoopPolarFrameTransportBackendFailureReason(StrEnum):
    """Classify a backend failure without exception-text parsing."""

    NONCONVERGENCE = "nonconvergence"
    NONFINITE_OUTPUT = "nonfinite-output"
    INVALID_DECOMPOSITION = "invalid-decomposition"


class ClosedLoopPolarFrameTransportBackendFailure(RuntimeError):
    """Report one typed decomposition failure at the capability boundary."""

    __slots__ = ("operation", "reason", "link_index")

    operation: ClosedLoopPolarFrameTransportBackendOperation
    reason: ClosedLoopPolarFrameTransportBackendFailureReason
    link_index: int

    def __init__(
        self,
        operation: ClosedLoopPolarFrameTransportBackendOperation,
        reason: ClosedLoopPolarFrameTransportBackendFailureReason,
        link_index: int,
    ) -> None:
        if type(operation) is not ClosedLoopPolarFrameTransportBackendOperation:
            raise TypeError(
                "operation must be ClosedLoopPolarFrameTransportBackendOperation"
            )
        if type(reason) is not ClosedLoopPolarFrameTransportBackendFailureReason:
            raise TypeError(
                "reason must be ClosedLoopPolarFrameTransportBackendFailureReason"
            )
        if type(link_index) is not int:
            raise TypeError("link_index must be an exact built-in integer")
        if link_index < 0:
            raise ValueError("link_index must be nonnegative")
        self.operation = operation
        self.reason = reason
        self.link_index = link_index
        super().__init__(
            f"{operation.value} backend failed with {reason.value} "
            f"at link index {link_index}"
        )


@final
@dataclass(frozen=True, slots=True)
class ClosedLoopPolarFrameTransportLinkDiagnostic:
    """Retain the complete available evidence for one ordered route link."""

    index: int
    kind: ClosedLoopPolarFrameTransportLinkKind
    status: ClosedLoopPolarFrameTransportLinkStatus
    singular_values: tuple[float, ...] | None
    applied_rank_cutoff: float | None
    rank_margin: float | None
    numerical_rank: int | None
    maximum_absolute_entry_unitarity_defect: float | None

    def __post_init__(self) -> None:
        if type(self.index) is not int:
            raise TypeError("index must be an exact built-in integer")
        if self.index < 0:
            raise ValueError("index must be nonnegative")
        if type(self.kind) is not ClosedLoopPolarFrameTransportLinkKind:
            raise TypeError("kind must be ClosedLoopPolarFrameTransportLinkKind")
        if type(self.status) is not ClosedLoopPolarFrameTransportLinkStatus:
            raise TypeError("status must be ClosedLoopPolarFrameTransportLinkStatus")
        evidence = (
            self.singular_values,
            self.applied_rank_cutoff,
            self.rank_margin,
            self.numerical_rank,
        )
        if self.status is ClosedLoopPolarFrameTransportLinkStatus.UNATTEMPTED:
            if any(
                value is not None
                for value in (
                    *evidence,
                    self.maximum_absolute_entry_unitarity_defect,
                )
            ):
                raise ValueError("unattempted links must not claim numerical evidence")
            return
        if any(value is None for value in evidence):
            raise ValueError("attempted links require complete rank evidence")
        singular_values = self.singular_values
        if singular_values is None:
            raise ValueError("attempted links require singular_values")
        if type(singular_values) is not tuple or not singular_values:
            raise TypeError("singular_values must be a nonempty exact tuple")
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in singular_values
        ):
            raise ValueError("singular_values must be finite nonnegative floats")
        if any(
            left < right
            for left, right in zip(
                singular_values,
                singular_values[1:],
                strict=False,
            )
        ):
            raise ValueError("singular_values must be nonincreasing")
        cutoff = self.applied_rank_cutoff
        margin = self.rank_margin
        numerical_rank = self.numerical_rank
        if cutoff is None or margin is None or numerical_rank is None:
            raise ValueError("attempted links require complete rank evidence")
        if type(cutoff) is not float or not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("applied_rank_cutoff must be finite and nonnegative")
        if type(margin) is not float or not math.isfinite(margin):
            raise ValueError("rank_margin must be finite")
        if margin != singular_values[-1] - cutoff:
            raise ValueError("rank_margin is inconsistent")
        if type(numerical_rank) is not int:
            raise TypeError("numerical_rank must be an exact built-in integer")
        if not 0 <= numerical_rank <= len(singular_values):
            raise ValueError("numerical_rank lies outside the rank bound")
        expected_rank = sum(value > cutoff for value in singular_values)
        if numerical_rank != expected_rank:
            raise ValueError("numerical_rank is inconsistent")
        if self.status is ClosedLoopPolarFrameTransportLinkStatus.QUALIFIED:
            if numerical_rank != len(singular_values):
                raise ValueError("qualified links require full numerical rank")
            maximum_absolute_entry_unitarity_defect = (
                self.maximum_absolute_entry_unitarity_defect
            )
            if (
                type(maximum_absolute_entry_unitarity_defect) is not float
                or not math.isfinite(maximum_absolute_entry_unitarity_defect)
                or maximum_absolute_entry_unitarity_defect < 0.0
            ):
                raise ValueError(
                    "qualified links require a finite maximum-absolute-entry "
                    "unitarity defect"
                )
        else:
            if numerical_rank == len(singular_values):
                raise ValueError("rank-deficient links must be numerically deficient")
            if self.maximum_absolute_entry_unitarity_defect is not None:
                raise ValueError("rank-deficient links must not claim a unitary map")


__all__ = [
    "ClosedLoopPolarFrameTransportBackendFailure",
    "ClosedLoopPolarFrameTransportBackendFailureReason",
    "ClosedLoopPolarFrameTransportBackendOperation",
    "ClosedLoopPolarFrameTransportLinkDiagnostic",
    "ClosedLoopPolarFrameTransportLinkKind",
    "ClosedLoopPolarFrameTransportLinkStatus",
    "ClosedLoopPolarFrameTransportPrincipalPhaseBranch",
    "ClosedLoopPolarFrameTransportStatus",
]
