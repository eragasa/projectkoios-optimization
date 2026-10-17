"""Named analytical periodic-harmonic fixtures."""

from __future__ import annotations

import math
from dataclasses import dataclass

from projectkoios.optimization.selection.harmonic import (
    PeriodicHarmonic,
    PeriodicHarmonicMaximum,
    PeriodicHarmonicMaximumStatus,
)


@dataclass(frozen=True, slots=True)
class ResolvedHarmonicCase:
    """Retain one analytical 3-4-5 harmonic maximum case."""

    result: PeriodicHarmonicMaximum
    offset: float
    amplitude: float
    maximum: float
    maximizing_argument: float
    cutoff: float


def resolved_harmonic_case() -> ResolvedHarmonicCase:
    """Construct a resolved maximum from an independent 3-4-5 oracle."""
    offset = 2.0
    cosine_coefficient = 3.0
    sine_coefficient = 4.0
    harmonic_order = 2
    interval_start = 0.0
    harmonic = PeriodicHarmonic(
        offset,
        cosine_coefficient,
        sine_coefficient,
        harmonic_order,
        interval_start,
    )
    amplitude = 5.0
    maximum = 7.0
    maximizing_argument = math.atan2(sine_coefficient, cosine_coefficient) / float(
        harmonic_order
    )
    cutoff = math.ulp(amplitude)
    result = PeriodicHarmonicMaximum(
        harmonic=harmonic,
        maximum_value=maximum,
        amplitude=amplitude,
        canonical_maximizing_argument=maximizing_argument,
        fundamental_period=harmonic.fundamental_period,
        applied_degeneracy_cutoff=cutoff,
        status=PeriodicHarmonicMaximumStatus.UNIQUE,
    )
    return ResolvedHarmonicCase(
        result=result,
        offset=offset,
        amplitude=amplitude,
        maximum=maximum,
        maximizing_argument=maximizing_argument,
        cutoff=cutoff,
    )
