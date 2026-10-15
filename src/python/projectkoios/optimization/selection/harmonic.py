"""Dependency-free closed-form selection for one real periodic harmonic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import final

from ..validation.validation import finite_float, u32

DEFAULT_PERIODIC_HARMONIC_ABSOLUTE_DEGENERACY_TOLERANCE = 1.0e-12
DEFAULT_PERIODIC_HARMONIC_RELATIVE_DEGENERACY_TOLERANCE = 1.0e-10


class PeriodicHarmonicMaximumStatus(StrEnum):
    """Whether one canonical maximizing argument is numerically resolved."""

    UNIQUE = "unique"
    NUMERICALLY_CONSTANT = "numerically-constant"


@final
@dataclass(frozen=True, slots=True)
class PeriodicHarmonic:
    r"""Define ``offset + c*cos(order*argument) + s*sin(order*argument)``.

    The canonical fundamental interval is half open. It begins at
    ``fundamental_interval_start`` and has length ``2*pi/harmonic_order``.
    """

    offset: float
    cosine_coefficient: float
    sine_coefficient: float
    harmonic_order: int
    fundamental_interval_start: float

    def __post_init__(self) -> None:
        """Normalize finite binary64 coefficients and validate the interval."""
        offset = finite_float(self.offset, "offset")
        cosine = finite_float(self.cosine_coefficient, "cosine_coefficient")
        sine = finite_float(self.sine_coefficient, "sine_coefficient")
        order = u32(self.harmonic_order, "harmonic_order", allow_zero=False)
        start = finite_float(
            self.fundamental_interval_start,
            "fundamental_interval_start",
        )
        period = math.tau / float(order)
        stop = start + period
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("fundamental period must be finite and positive")
        if not math.isfinite(stop) or stop <= start:
            raise ValueError(
                "fundamental interval endpoint must be finite and distinguishable"
            )
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "cosine_coefficient", cosine)
        object.__setattr__(self, "sine_coefficient", sine)
        object.__setattr__(self, "harmonic_order", order)
        object.__setattr__(self, "fundamental_interval_start", start)

    @property
    def fundamental_period(self) -> float:
        """Return ``2*pi/harmonic_order`` in binary64."""
        return math.tau / float(self.harmonic_order)

    @property
    def fundamental_interval_stop(self) -> float:
        """Return the excluded endpoint of the canonical interval."""
        return self.fundamental_interval_start + self.fundamental_period


@final
@dataclass(frozen=True, slots=True)
class PeriodicHarmonicMaximum:
    """Retain one harmonic maximum and canonical-argument disposition."""

    harmonic: PeriodicHarmonic
    maximum_value: float
    amplitude: float
    canonical_maximizing_argument: float | None
    fundamental_period: float
    applied_degeneracy_cutoff: float
    status: PeriodicHarmonicMaximumStatus

    def __post_init__(self) -> None:
        """Validate all derived values against the retained harmonic."""
        if type(self.harmonic) is not PeriodicHarmonic:
            raise TypeError("harmonic must be PeriodicHarmonic")
        maximum = finite_float(self.maximum_value, "maximum_value")
        amplitude = finite_float(self.amplitude, "amplitude")
        period = finite_float(self.fundamental_period, "fundamental_period")
        cutoff = finite_float(
            self.applied_degeneracy_cutoff,
            "applied_degeneracy_cutoff",
        )
        if amplitude < 0.0:
            raise ValueError("amplitude must be nonnegative")
        if period <= 0.0:
            raise ValueError("fundamental_period must be positive")
        if cutoff < 0.0:
            raise ValueError("applied_degeneracy_cutoff must be nonnegative")
        if type(self.status) is not PeriodicHarmonicMaximumStatus:
            raise TypeError("status must be PeriodicHarmonicMaximumStatus")

        expected_amplitude = math.hypot(
            self.harmonic.cosine_coefficient,
            self.harmonic.sine_coefficient,
        )
        expected_maximum = self.harmonic.offset + expected_amplitude
        if not math.isfinite(expected_amplitude) or not math.isfinite(expected_maximum):
            raise ValueError("harmonic maximum arithmetic must be finite")
        if amplitude != expected_amplitude:
            raise ValueError("amplitude is inconsistent with harmonic coefficients")
        if maximum != expected_maximum:
            raise ValueError("maximum_value is inconsistent with harmonic coefficients")
        if period != self.harmonic.fundamental_period:
            raise ValueError("fundamental_period is inconsistent with harmonic order")

        is_degenerate = amplitude <= cutoff
        expected_status = (
            PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT
            if is_degenerate
            else PeriodicHarmonicMaximumStatus.UNIQUE
        )
        if self.status is not expected_status:
            raise ValueError("status is inconsistent with amplitude and cutoff")
        argument = self.canonical_maximizing_argument
        if is_degenerate:
            if argument is not None:
                raise ValueError(
                    "numerically constant harmonics must not select one argument"
                )
        else:
            normalized_argument = finite_float(
                argument,
                "canonical_maximizing_argument",
            )
            expected_argument = _canonical_maximizing_argument(self.harmonic)
            if normalized_argument != expected_argument:
                raise ValueError(
                    "canonical_maximizing_argument is inconsistent with the harmonic"
                )
            argument = normalized_argument

        object.__setattr__(self, "maximum_value", maximum)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "canonical_maximizing_argument", argument)
        object.__setattr__(self, "fundamental_period", period)
        object.__setattr__(self, "applied_degeneracy_cutoff", cutoff)


@final
class PeriodicHarmonicMaximizer:
    """Select a canonical closed-form maximum of one periodic harmonic."""

    __slots__ = (
        "_absolute_degeneracy_tolerance",
        "_relative_degeneracy_tolerance",
    )

    def __init__(
        self,
        *,
        absolute_degeneracy_tolerance: float = (
            DEFAULT_PERIODIC_HARMONIC_ABSOLUTE_DEGENERACY_TOLERANCE
        ),
        relative_degeneracy_tolerance: float = (
            DEFAULT_PERIODIC_HARMONIC_RELATIVE_DEGENERACY_TOLERANCE
        ),
    ) -> None:
        absolute = finite_float(
            absolute_degeneracy_tolerance,
            "absolute_degeneracy_tolerance",
        )
        relative = finite_float(
            relative_degeneracy_tolerance,
            "relative_degeneracy_tolerance",
        )
        if absolute < 0.0:
            raise ValueError("absolute_degeneracy_tolerance must be nonnegative")
        if relative < 0.0:
            raise ValueError("relative_degeneracy_tolerance must be nonnegative")
        if absolute == 0.0 and relative == 0.0:
            raise ValueError("at least one degeneracy tolerance must be positive")
        self._absolute_degeneracy_tolerance = absolute
        self._relative_degeneracy_tolerance = relative

    @property
    def absolute_degeneracy_tolerance(self) -> float:
        """Return the absolute amplitude-degeneracy tolerance."""
        return self._absolute_degeneracy_tolerance

    @property
    def relative_degeneracy_tolerance(self) -> float:
        """Return the relative amplitude-degeneracy tolerance."""
        return self._relative_degeneracy_tolerance

    def maximize(
        self,
        harmonic: PeriodicHarmonic,
        /,
    ) -> PeriodicHarmonicMaximum:
        """Return the mathematical maximum and resolved canonical argument, if any."""
        if type(harmonic) is not PeriodicHarmonic:
            raise TypeError("harmonic must be PeriodicHarmonic")
        amplitude = math.hypot(
            harmonic.cosine_coefficient,
            harmonic.sine_coefficient,
        )
        if not math.isfinite(amplitude):
            raise FloatingPointError("harmonic amplitude must be finite")
        maximum = harmonic.offset + amplitude
        if not math.isfinite(maximum):
            raise FloatingPointError("harmonic maximum must be finite")
        scale = max(1.0, abs(harmonic.offset), amplitude)
        cutoff = self.absolute_degeneracy_tolerance + (
            self.relative_degeneracy_tolerance * scale
        )
        if not math.isfinite(cutoff):
            raise FloatingPointError("harmonic degeneracy cutoff must be finite")
        if amplitude <= cutoff:
            argument = None
            status = PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT
        else:
            argument = _canonical_maximizing_argument(harmonic)
            status = PeriodicHarmonicMaximumStatus.UNIQUE
        return PeriodicHarmonicMaximum(
            harmonic=harmonic,
            maximum_value=maximum,
            amplitude=amplitude,
            canonical_maximizing_argument=argument,
            fundamental_period=harmonic.fundamental_period,
            applied_degeneracy_cutoff=cutoff,
            status=status,
        )


def _canonical_maximizing_argument(harmonic: PeriodicHarmonic, /) -> float:
    phase = math.atan2(
        harmonic.sine_coefficient,
        harmonic.cosine_coefficient,
    )
    raw_argument = phase / float(harmonic.harmonic_order)
    period = harmonic.fundamental_period
    start = harmonic.fundamental_interval_start
    relative = (raw_argument - start) % period
    argument = start + relative
    stop = harmonic.fundamental_interval_stop
    if argument >= stop:
        argument = start
    if not math.isfinite(argument) or argument < start or argument >= stop:
        raise FloatingPointError(
            "canonical maximizing argument must lie in the fundamental interval"
        )
    return argument


__all__ = [
    "DEFAULT_PERIODIC_HARMONIC_ABSOLUTE_DEGENERACY_TOLERANCE",
    "DEFAULT_PERIODIC_HARMONIC_RELATIVE_DEGENERACY_TOLERANCE",
    "PeriodicHarmonic",
    "PeriodicHarmonicMaximum",
    "PeriodicHarmonicMaximumStatus",
    "PeriodicHarmonicMaximizer",
]
