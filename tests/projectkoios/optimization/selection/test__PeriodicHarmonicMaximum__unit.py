"""Unit tests for periodic harmonic maximum evidence records."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import pytest

from projectkoios.optimization.selection.harmonic import (
    PeriodicHarmonic,
    PeriodicHarmonicMaximum,
    PeriodicHarmonicMaximumStatus,
)

pytestmark = pytest.mark.unit


def _unique_maximum() -> PeriodicHarmonicMaximum:
    harmonic = PeriodicHarmonic(2.0, 3.0, 4.0, 2, -math.pi / 2.0)
    raw_argument = 0.5 * math.atan2(4.0, 3.0)
    canonical_argument = harmonic.fundamental_interval_start + (
        (raw_argument - harmonic.fundamental_interval_start)
        % harmonic.fundamental_period
    )
    return PeriodicHarmonicMaximum(
        harmonic=harmonic,
        maximum_value=7.0,
        amplitude=5.0,
        canonical_maximizing_argument=canonical_argument,
        fundamental_period=math.pi,
        applied_degeneracy_cutoff=1.0e-6,
        status=PeriodicHarmonicMaximumStatus.UNIQUE,
    )


def test_unique_result_retains_correlated_evidence() -> None:
    maximum = _unique_maximum()

    assert maximum.maximum_value == 7.0
    assert maximum.amplitude == 5.0
    assert maximum.canonical_maximizing_argument == pytest.approx(
        0.5 * math.atan2(4.0, 3.0),
        rel=0.0,
        abs=1.0e-16,
    )
    assert maximum.fundamental_period == math.pi
    assert maximum.status is PeriodicHarmonicMaximumStatus.UNIQUE


def test_numerically_constant_result_retains_no_selected_argument() -> None:
    harmonic = PeriodicHarmonic(2.0, 0.0, 0.0, 2, -math.pi / 2.0)
    maximum = PeriodicHarmonicMaximum(
        harmonic=harmonic,
        maximum_value=2.0,
        amplitude=0.0,
        canonical_maximizing_argument=None,
        fundamental_period=math.pi,
        applied_degeneracy_cutoff=1.0e-12,
        status=PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT,
    )

    assert maximum.canonical_maximizing_argument is None
    assert maximum.status is PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT


def test_result_rejects_inconsistent_derived_values() -> None:
    maximum = _unique_maximum()

    with pytest.raises(ValueError):
        replace(maximum, maximum_value=6.0)
    with pytest.raises(ValueError):
        replace(maximum, amplitude=6.0)
    with pytest.raises(ValueError):
        replace(maximum, fundamental_period=6.0)
    with pytest.raises(ValueError):
        replace(maximum, canonical_maximizing_argument=6.0)


def test_result_rejects_status_and_argument_inconsistency() -> None:
    maximum = _unique_maximum()
    with pytest.raises(ValueError, match="status"):
        replace(
            maximum,
            status=PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT,
        )

    constant = PeriodicHarmonicMaximum(
        PeriodicHarmonic(1.0, 0.0, 0.0, 1, 0.0),
        1.0,
        0.0,
        None,
        math.tau,
        1.0e-12,
        PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT,
    )
    with pytest.raises(ValueError, match="must not select"):
        replace(constant, canonical_maximizing_argument=0.0)


def test_result_rejects_invalid_field_types_and_ranges() -> None:
    maximum = _unique_maximum()
    with pytest.raises(TypeError, match="harmonic"):
        replace(maximum, harmonic=cast(PeriodicHarmonic, object()))
    with pytest.raises(TypeError, match="status"):
        replace(
            maximum,
            status=cast(PeriodicHarmonicMaximumStatus, "unique"),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        replace(maximum, applied_degeneracy_cutoff=-1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(maximum, amplitude=-1.0)
