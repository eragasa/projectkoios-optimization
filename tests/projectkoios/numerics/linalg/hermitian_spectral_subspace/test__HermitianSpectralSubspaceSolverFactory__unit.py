"""Unit tests for the lazy Hermitian spectral-subspace solver factory."""

from __future__ import annotations

import sys
from typing import Any, cast

import pytest

from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceMethod,
    HermitianSpectralSubspaceSolver,
    HermitianSpectralSubspaceSolverFactory,
)

pytestmark = pytest.mark.unit


def test_factory_constructs_nominal_numpy_solver_lazily() -> None:
    module_name = "projectkoios.numerics.linalg.numpy.hermitian_spectral_subspace"
    sys.modules.pop(module_name, None)

    solver = HermitianSpectralSubspaceSolverFactory.create(
        HermitianSpectralSubspaceMethod.NUMPY_DENSE
    )

    assert isinstance(solver, HermitianSpectralSubspaceSolver)
    assert type(solver).__name__ == "NumpyHermitianSpectralSubspaceSolver"
    assert module_name in sys.modules


def test_factory_rejects_raw_string_and_wrong_enum_types() -> None:
    with pytest.raises(TypeError, match="HermitianSpectralSubspaceMethod"):
        HermitianSpectralSubspaceSolverFactory.create(
            cast(Any, HermitianSpectralSubspaceMethod.NUMPY_DENSE.value)
        )


def test_method_is_closed_to_reviewed_builtin() -> None:
    assert tuple(HermitianSpectralSubspaceMethod) == (
        HermitianSpectralSubspaceMethod.NUMPY_DENSE,
    )
