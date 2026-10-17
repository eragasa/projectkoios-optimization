"""Unit verification for dependency-free finite-transform diagnostics."""

from __future__ import annotations

import pytest

from projectkoios.numerics.finite_periodic.transforms import (
    FinitePeriodicHermiticityDiagnostics,
    FinitePeriodicParsevalDiagnostics,
)

pytestmark = pytest.mark.unit


def test_parseval_diagnostics_retain_exact_represented_energy_defect() -> None:
    result = FinitePeriodicParsevalDiagnostics(3.25, 3.0, 0.25)

    assert result.sample_mean_squared_frobenius_norm == 3.25
    assert result.coefficient_sum_squared_frobenius_norm == 3.0
    assert result.absolute_defect == 0.25


def test_parseval_diagnostics_reject_inconsistent_defect() -> None:
    with pytest.raises(ValueError):
        FinitePeriodicParsevalDiagnostics(3.25, 3.0, 0.125)


def test_hermiticity_diagnostics_retain_separate_cyclic_and_ordinary_defects() -> None:
    result = FinitePeriodicHermiticityDiagnostics(2.0e-12, 3.0e-15)

    assert result.twisted_cyclic_maximum_frobenius_defect == 2.0e-12
    assert result.ordinary_maximum_frobenius_defect == 3.0e-15


def test_transform_diagnostics_reject_negative_values() -> None:
    with pytest.raises(ValueError):
        FinitePeriodicHermiticityDiagnostics(-1.0, 0.0)
