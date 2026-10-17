"""Unit tests for periodic-harmonic maximum evidence records."""

from __future__ import annotations

import math

import pytest

from projectkoios.optimization.selection.harmonic import (
    PeriodicHarmonic,
    PeriodicHarmonicMaximum,
    PeriodicHarmonicMaximumStatus,
)

pytestmark = pytest.mark.unit


def _pythagorean_harmonic() -> PeriodicHarmonic:
    return PeriodicHarmonic(2.0, 3.0, 4.0, 2, -math.pi / 2.0)


def _pythagorean_argument(harmonic: PeriodicHarmonic) -> float:
    raw_argument = 0.5 * math.atan2(4.0, 3.0)
    return harmonic.fundamental_interval_start + (
        (raw_argument - harmonic.fundamental_interval_start)
        % harmonic.fundamental_period
    )


def test_unique_pythagorean_harmonic_retains_derived_evidence() -> None:
    harmonic = _pythagorean_harmonic()
    argument = _pythagorean_argument(harmonic)

    maximum = PeriodicHarmonicMaximum(
        harmonic=harmonic,
        maximum_value=7.0,
        amplitude=5.0,
        canonical_maximizing_argument=argument,
        fundamental_period=math.pi,
        applied_degeneracy_cutoff=1.0e-6,
        status=PeriodicHarmonicMaximumStatus.UNIQUE,
    )

    assert maximum.maximum_value == 7.0
    assert maximum.amplitude == 5.0
    assert maximum.canonical_maximizing_argument == argument
    assert maximum.fundamental_period == math.pi
    assert maximum.status is PeriodicHarmonicMaximumStatus.UNIQUE


def test_constant_harmonic_retains_no_selected_argument() -> None:
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


def test_constructor_reconstructs_pythagorean_amplitude() -> None:
    harmonic = _pythagorean_harmonic()

    with pytest.raises(ValueError):
        PeriodicHarmonicMaximum(
            harmonic=harmonic,
            maximum_value=7.0,
            amplitude=4.0,
            canonical_maximizing_argument=_pythagorean_argument(harmonic),
            fundamental_period=math.pi,
            applied_degeneracy_cutoff=1.0e-6,
            status=PeriodicHarmonicMaximumStatus.UNIQUE,
        )


def test_constructor_reconstructs_pythagorean_maximum_value() -> None:
    harmonic = _pythagorean_harmonic()

    with pytest.raises(ValueError):
        PeriodicHarmonicMaximum(
            harmonic=harmonic,
            maximum_value=6.0,
            amplitude=5.0,
            canonical_maximizing_argument=_pythagorean_argument(harmonic),
            fundamental_period=math.pi,
            applied_degeneracy_cutoff=1.0e-6,
            status=PeriodicHarmonicMaximumStatus.UNIQUE,
        )


def test_constructor_requires_the_canonical_unique_argument() -> None:
    harmonic = _pythagorean_harmonic()

    with pytest.raises(ValueError):
        PeriodicHarmonicMaximum(
            harmonic=harmonic,
            maximum_value=7.0,
            amplitude=5.0,
            canonical_maximizing_argument=harmonic.fundamental_interval_stop,
            fundamental_period=math.pi,
            applied_degeneracy_cutoff=1.0e-6,
            status=PeriodicHarmonicMaximumStatus.UNIQUE,
        )


def test_constructor_forbids_argument_selection_for_constant_harmonic() -> None:
    harmonic = PeriodicHarmonic(1.0, 0.0, 0.0, 1, 0.0)

    with pytest.raises(ValueError):
        PeriodicHarmonicMaximum(
            harmonic=harmonic,
            maximum_value=1.0,
            amplitude=0.0,
            canonical_maximizing_argument=0.0,
            fundamental_period=math.tau,
            applied_degeneracy_cutoff=1.0e-12,
            status=PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT,
        )
