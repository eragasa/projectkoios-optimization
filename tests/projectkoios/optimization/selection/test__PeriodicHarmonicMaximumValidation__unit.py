"""Direct tests for periodic-harmonic maximum validation owners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import pytest

from projectkoios.optimization.selection.harmonic import (
    PeriodicHarmonic,
    PeriodicHarmonicMaximum,
    PeriodicHarmonicMaximumStatus,
    _HarmonicMaximumValues,
    _validate_harmonic_maximum_argument,
    _validate_harmonic_maximum_relations,
    _validate_harmonic_maximum_values,
)
from tests.projectkoios.optimization.selection.fixtures import ResolvedHarmonicCase

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _HarmonicMaximumProbe:
    """Mirror maximum attributes while permitting one deliberately invalid scalar."""

    harmonic: PeriodicHarmonic
    maximum_value: float | int
    amplitude: float
    canonical_maximizing_argument: float | None
    fundamental_period: float
    applied_degeneracy_cutoff: float
    status: PeriodicHarmonicMaximumStatus

    @classmethod
    def from_result(
        cls,
        result: PeriodicHarmonicMaximum,
        /,
    ) -> _HarmonicMaximumProbe:
        """Copy one valid result into an immutable validation probe."""
        return cls(
            harmonic=result.harmonic,
            maximum_value=result.maximum_value,
            amplitude=result.amplitude,
            canonical_maximizing_argument=result.canonical_maximizing_argument,
            fundamental_period=result.fundamental_period,
            applied_degeneracy_cutoff=result.applied_degeneracy_cutoff,
            status=result.status,
        )

    def as_result(self) -> PeriodicHarmonicMaximum:
        """Expose the structural probe to a single validation owner."""
        return cast(PeriodicHarmonicMaximum, self)


def test_value_validation_requires_canonical_scalar_evidence(
    resolved_harmonic: ResolvedHarmonicCase,
) -> None:
    case = resolved_harmonic
    probe = _HarmonicMaximumProbe.from_result(case.result)

    values = _validate_harmonic_maximum_values(case.result)

    assert values == _HarmonicMaximumValues(
        maximum=case.maximum,
        amplitude=case.amplitude,
        period=case.result.harmonic.fundamental_period,
        cutoff=case.cutoff,
    )
    with pytest.raises(TypeError, match="exact built-in float"):
        _validate_harmonic_maximum_values(
            replace(probe, maximum_value=int(case.maximum)).as_result()
        )
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_harmonic_maximum_values(
            replace(probe, amplitude=-case.amplitude).as_result()
        )


def test_value_factory_normalizes_before_frozen_construction(
    resolved_harmonic: ResolvedHarmonicCase,
) -> None:
    case = resolved_harmonic

    result = PeriodicHarmonicMaximum.from_values(
        harmonic=case.result.harmonic,
        maximum_value=int(case.maximum),
        amplitude=int(case.amplitude),
        canonical_maximizing_argument=case.maximizing_argument,
        fundamental_period=case.result.harmonic.fundamental_period,
        applied_degeneracy_cutoff=case.cutoff,
        status=PeriodicHarmonicMaximumStatus.UNIQUE,
    )

    assert result.maximum_value == case.maximum
    assert result.amplitude == case.amplitude
    assert type(result.maximum_value) is float
    assert type(result.amplitude) is float


def test_relational_validation_reconstructs_unique_status(
    resolved_harmonic: ResolvedHarmonicCase,
) -> None:
    case = resolved_harmonic
    probe = _HarmonicMaximumProbe.from_result(case.result)
    values = _validate_harmonic_maximum_values(case.result)
    inconsistent = replace(probe, maximum_value=case.offset).as_result()
    inconsistent_values = _validate_harmonic_maximum_values(inconsistent)

    assert _validate_harmonic_maximum_relations(case.result, values) is False

    with pytest.raises(ValueError, match="maximum_value"):
        _validate_harmonic_maximum_relations(
            inconsistent,
            inconsistent_values,
        )
    with pytest.raises(ValueError, match="status"):
        _validate_harmonic_maximum_relations(
            replace(
                probe,
                status=PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT,
            ).as_result(),
            values,
        )


def test_argument_validation_separates_unique_and_degenerate_contracts(
    resolved_harmonic: ResolvedHarmonicCase,
) -> None:
    case = resolved_harmonic
    probe = _HarmonicMaximumProbe.from_result(case.result)
    degenerate_probe = replace(
        probe,
        maximum_value=case.offset,
        amplitude=0.0,
        canonical_maximizing_argument=None,
        status=PeriodicHarmonicMaximumStatus.NUMERICALLY_CONSTANT,
    )
    degenerate = degenerate_probe.as_result()

    assert (
        _validate_harmonic_maximum_argument(case.result, False)
        == case.maximizing_argument
    )
    assert _validate_harmonic_maximum_argument(degenerate, True) is None

    with pytest.raises(ValueError, match="must not select"):
        _validate_harmonic_maximum_argument(
            replace(
                degenerate_probe,
                canonical_maximizing_argument=(
                    case.result.harmonic.fundamental_interval_start
                ),
            ).as_result(),
            True,
        )
    with pytest.raises(ValueError, match="inconsistent with the harmonic"):
        _validate_harmonic_maximum_argument(
            replace(
                probe,
                canonical_maximizing_argument=(
                    case.result.harmonic.fundamental_interval_stop
                ),
            ).as_result(),
            False,
        )
