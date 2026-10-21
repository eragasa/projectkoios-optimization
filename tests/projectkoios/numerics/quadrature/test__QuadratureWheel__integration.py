"""Isolated-wheel verification for uniform one-dimensional quadrature."""

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
    root = Path(__file__).resolve().parents[4]
    spec = spec_from_file_location(
        "_projectkoios_optimization_build_backend",
        root / "build_backend" / "backend.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the repository build backend")
    backend = module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def test_isolated_wheel_exposes_lazy_uniform_quadrature(tmp_path: Path) -> None:
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    backend = _load_build_backend()
    wheel_path = wheel_directory / backend.build_wheel(str(wheel_directory))
    installed = tmp_path / "installed"
    installed.mkdir()
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(installed)

    script = """
import os
from pathlib import Path
import sys

from projectkoios.numerics.quadrature.uniform_1d import (
    UniformClosedIntervalTrapezoidalRuleDefinition,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
definition = UniformClosedIntervalTrapezoidalRuleDefinition(0.0, 1.0, 5)
assert definition.spacing == 0.25
assert definition.interior_point_count == 3

import numpy as np
from projectkoios.numerics.quadrature.numpy import uniform_1d as capability
from projectkoios.numerics.quadrature.numpy.uniform_1d import (
    NumpyUniformClosedIntervalTrapezoidalRule,
)

assert Path(capability.__file__).resolve().is_relative_to(installed)
result = NumpyUniformClosedIntervalTrapezoidalRule().calculate(definition)
assert np.array_equal(
    result.full_weights,
    np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=np.float64),
)
interior = result.restrict_homogeneous_dirichlet()
assert interior.full_grid_indices.tolist() == [1, 2, 3]
assert np.array_equal(interior.mass_matrix.to_dense(), 0.25 * np.eye(3))
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
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
