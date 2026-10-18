"""Isolated-wheel smoke test for complex-unitary Procrustes alignment."""

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

import projectkoios.optimization.alignment.procrustes as procrustes
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesDefinition,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(procrustes.__file__).resolve().is_relative_to(installed)
assert importlib.util.find_spec("numpy") is None
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
definition = ComplexUnitaryProcrustesDefinition(
    ((1.0 + 0.0j,),),
    ((0.0 + 1.0j,),),
)
assert definition.row_count == definition.column_count == 1
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
"""
    numpy_script = """
import os
from pathlib import Path
import sys

import projectkoios.optimization.alignment.procrustes as procrustes
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesDefinition,
    ComplexUnitaryProcrustesMethod,
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesSolverFactory,
    ComplexUnitaryProcrustesStatus,
    ComplexUnitaryProcrustesTolerance,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(procrustes.__file__).resolve().is_relative_to(installed)
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
solver = ComplexUnitaryProcrustesSolverFactory.create(
    ComplexUnitaryProcrustesMethod.NUMPY_DENSE
)
result = solver.solve(
    ComplexUnitaryProcrustesDefinition(
        ((0.0 + 1.0j,),),
        ((1.0 + 0.0j,),),
        ComplexUnitaryProcrustesTolerance(
            rank_absolute=1.0e-12,
            rank_relative=1.0e-10,
            rank_cutoff_rule=ComplexUnitaryProcrustesRankCutoffRule.MAXIMUM,
        ),
    )
)
assert result.status is ComplexUnitaryProcrustesStatus.UNIQUE
assert result.transformation == ((0.0 - 1.0j,),)
assert result.optimal_squared_residual == 0.0
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
