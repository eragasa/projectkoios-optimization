"""Unit tests for explicit SciPy CSR spectral-norm controls."""

from __future__ import annotations

import pytest

from projectkoios.numerics.linalg.scipy.spectral_norm import (
    ScipyCsrSpectralNormControls,
)

pytestmark = pytest.mark.unit


def test_controls_retain_every_explicit_valid_value() -> None:
    """Verify the immutable record retains its complete caller-owned policy."""
    controls = ScipyCsrSpectralNormControls(1.0e-9, 800, 12, 37)

    assert controls.tolerance == 1.0e-9
    assert controls.maximum_iterations == 800
    assert controls.krylov_dimension == 12
    assert controls.seed == 37
    assert not hasattr(controls, "ncv")


def test_controls_accept_machine_precision_and_scipy_krylov_policy() -> None:
    """Verify zero tolerance and explicit ``None`` encode backend-owned choices."""
    controls = ScipyCsrSpectralNormControls(0.0, 1, None, 0)

    assert controls.tolerance == 0.0
    assert controls.maximum_iterations == 1
    assert controls.krylov_dimension is None
    assert controls.seed == 0


def test_controls_reject_non_exact_scalar_types() -> None:
    """Verify coercible Boolean and integer/float substitutes fail by kind."""
    with pytest.raises(TypeError, match="tolerance must be an exact built-in float"):
        ScipyCsrSpectralNormControls(False, 100, None, 0)
    with pytest.raises(TypeError, match="maximum_iterations"):
        ScipyCsrSpectralNormControls(0.0, True, None, 0)
    with pytest.raises(TypeError, match="krylov_dimension"):
        ScipyCsrSpectralNormControls(0.0, 100, True, 0)
    with pytest.raises(TypeError, match="seed"):
        ScipyCsrSpectralNormControls(0.0, 100, None, False)


def test_controls_reject_invalid_numeric_boundaries() -> None:
    """Verify every convergence and deterministic-seed boundary fails closed."""
    for tolerance in (-1.0, 1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="tolerance"):
            ScipyCsrSpectralNormControls(tolerance, 100, None, 0)
    with pytest.raises(ValueError, match="maximum_iterations must be positive"):
        ScipyCsrSpectralNormControls(0.0, 0, None, 0)
    with pytest.raises(ValueError, match="krylov_dimension must be positive"):
        ScipyCsrSpectralNormControls(0.0, 100, 0, 0)
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        ScipyCsrSpectralNormControls(0.0, 100, None, -1)
