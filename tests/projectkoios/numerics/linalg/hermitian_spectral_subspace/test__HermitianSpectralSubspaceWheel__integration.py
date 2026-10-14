"""Isolated-wheel smoke test for Hermitian spectral-subspace decomposition."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration


def _load_build_backend() -> ModuleType:
    root = Path(__file__).resolve().parents[5]
    spec = spec_from_file_location(
        "_projectkoios_optimization_build_backend",
        root / "build_backend" / "backend.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the repository build backend")
    backend = module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def test_isolated_wheel_exposes_lazy_hermitian_spectral_subspace(
    tmp_path: Path,
) -> None:
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    backend = _load_build_backend()
    wheel_path = wheel_directory / backend.build_wheel(str(wheel_directory))
    installed = tmp_path / "installed"
    installed.mkdir()
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(installed)

    dependency_free_script = """
import importlib.util
import os
from pathlib import Path
import sys

import projectkoios.numerics.linalg.hermitian_spectral_subspace as spectral
from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceDefinition,
    HermitianSpectralSubspaceTolerance,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(spectral.__file__).resolve().is_relative_to(installed)
assert importlib.util.find_spec("numpy") is None
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
definition = HermitianSpectralSubspaceDefinition(
    ((1.0 + 0.0j,),),
    0,
    1,
    HermitianSpectralSubspaceTolerance(*(1.0e-12,) * 7),
)
assert definition.dimension == 1
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
"""
    numpy_script = """
import os
from pathlib import Path
import sys

import projectkoios.numerics.linalg.hermitian_spectral_subspace as spectral
from projectkoios.numerics.linalg.hermitian_spectral_subspace import (
    HermitianSpectralSubspaceDefinition,
    HermitianSpectralSubspaceMethod,
    HermitianSpectralSubspaceSolverFactory,
    HermitianSpectralSubspaceTolerance,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(spectral.__file__).resolve().is_relative_to(installed)
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)

tolerance = HermitianSpectralSubspaceTolerance(*(1.0e-12,) * 7)
solver = HermitianSpectralSubspaceSolverFactory.create(
    HermitianSpectralSubspaceMethod.NUMPY_DENSE
)
result = solver.solve(
    HermitianSpectralSubspaceDefinition(
        ((2.0 + 0.0j, 0.0 + 1.0j), (0.0 - 1.0j, 2.0 + 0.0j)),
        0,
        1,
        tolerance,
    )
)
assert abs(result.eigenvalues[0] - 1.0) < 1.0e-14
assert abs(result.eigenvalues[1] - 3.0) < 1.0e-14
assert result.backend_identifier == "numpy.linalg.eigh"
assert "numpy" in sys.modules
"""
    environment = os.environ.copy()
    environment.update(
        {
            "EXPECTED_WHEEL_ROOT": str(installed),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(installed),
        }
    )
    dependency_free_completed = subprocess.run(
        [sys.executable, "-S", "-c", dependency_free_script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    numpy_completed = subprocess.run(
        [sys.executable, "-c", numpy_script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert dependency_free_completed.returncode == 0, dependency_free_completed.stderr
    assert numpy_completed.returncode == 0, numpy_completed.stderr
