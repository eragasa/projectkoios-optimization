"""Isolated-wheel smoke test for finite box quadratic minimization."""

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


def test_isolated_wheel_keeps_contract_lazy_and_executes_numpy_backend(
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

import projectkoios.optimization.quadratic.box as box
from projectkoios.optimization.quadratic.factory import (
    BoxQuadraticMinimizerFactory,
    BoxQuadraticMinimizerMethod,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(box.__file__).resolve().is_relative_to(installed)
assert BoxQuadraticMinimizerFactory is not None
assert BoxQuadraticMinimizerMethod.NUMPY_DENSE.value == "numpy-dense"
assert importlib.util.find_spec("numpy") is None
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
"""
    numpy_script = """
import os
from pathlib import Path
import sys

import projectkoios.optimization.quadratic.box as box
from projectkoios.optimization.quadratic.factory import (
    BoxQuadraticMinimizerFactory,
    BoxQuadraticMinimizerMethod,
)
from projectkoios.optimization.quadratic.quadratic import PositiveDefiniteQuadraticLoss
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(box.__file__).resolve().is_relative_to(installed)
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
solver = BoxQuadraticMinimizerFactory.create(BoxQuadraticMinimizerMethod.NUMPY_DENSE)
result = solver.solve(
    PositiveDefiniteQuadraticLoss((2.0, -2.0), ((2.0, 0.5), (0.5, 1.0)), 0.0),
    ParameterSpace((ParameterBound("x", 0.0, 1.0), ParameterBound("y", -1.0, 1.0))),
)
assert result.coordinates == (1.0, -1.0)
assert result.backend_identifier == "projectkoios.active-set.numpy"
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
