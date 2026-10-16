"""Isolated-wheel smoke test for batched complete Hermitian spectra."""

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


def test_built_wheel_exposes_dependency_free_contract_and_numpy_backend(
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

import projectkoios.numerics.linalg.hermitian_spectrum as spectrum
from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
)
installed = Path(os.environ['EXPECTED_WHEEL_ROOT']).resolve()
assert Path(spectrum.__file__).resolve().is_relative_to(installed)
assert importlib.util.find_spec('numpy') is None
assert 'numpy' not in sys.modules
empty = BatchedHermitianSpectrumDefinition.empty(
    2,
    BatchedHermitianSpectrumTolerance(0.0, 0.0),
)
assert empty.dimension == 2 and empty.batch_size == 0
assert 'numpy' not in sys.modules
"""
    numpy_script = """
import os
from pathlib import Path

import projectkoios.numerics.linalg.hermitian_spectrum as spectrum
from projectkoios.numerics.linalg.hermitian_spectrum import (
    BatchedHermitianSpectrumDefinition,
    BatchedHermitianSpectrumTolerance,
)
from projectkoios.numerics.linalg.numpy.hermitian_spectrum import (
    NumpyBatchedHermitianSpectrumSolver,
)
installed = Path(os.environ['EXPECTED_WHEEL_ROOT']).resolve()
assert Path(spectrum.__file__).resolve().is_relative_to(installed)
solver = NumpyBatchedHermitianSpectrumSolver()
tolerance = BatchedHermitianSpectrumTolerance(0.0, 0.0)
empty = solver.solve(BatchedHermitianSpectrumDefinition.empty(2, tolerance))
assert empty.eigenvalues == () and empty.diagnostics is None
definition = BatchedHermitianSpectrumDefinition(
    [[[1, 0+1j], [0-1j, 1]]],
    tolerance,
)
result = solver.solve(definition)
assert abs(result.eigenvalues[0][0]) < 1e-15
assert abs(result.eigenvalues[0][1] - 2.0) < 1e-15
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
