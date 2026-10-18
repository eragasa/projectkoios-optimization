"""Isolated-wheel smoke test for NumPy batched Procrustes alignment."""

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
    root = Path(__file__).resolve().parents[6]
    spec = spec_from_file_location(
        "_projectkoios_optimization_build_backend",
        root / "build_backend" / "backend.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the repository build backend")
    backend = module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def test_isolated_wheel_exposes_batched_right_alignment(tmp_path: Path) -> None:
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
import numpy as np
import projectkoios.optimization.alignment.numpy.batched_procrustes as capability
from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesRankCutoffRule,
    ComplexUnitaryProcrustesTolerance,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(capability.__file__).resolve().is_relative_to(installed)
source = np.broadcast_to(np.eye(2, dtype=np.complex128), (2, 3, 2, 2)).copy()
rotation = np.asarray(((0.0, 1.0), (-1.0, 0.0)), dtype=np.complex128)
target = source @ rotation
tolerance = ComplexUnitaryProcrustesTolerance(
    rank_absolute=1.0e-12,
    rank_relative=1.0e-10,
    rank_cutoff_rule=ComplexUnitaryProcrustesRankCutoffRule.MAXIMUM,
)
result = NumpyBatchedComplexUnitaryProcrustesSolver().solve(
    NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(
        source,
        target,
        tolerance,
    )
)
assert result.batch_shape == (2, 3)
assert np.all(result.full_rank)
assert result.all_full_rank
assert result.rank_deficient_count == 0
transformations = result.require_unique_right_unitary_transformations()
assert np.max(np.abs(transformations - rotation)) < 1.0e-15
assert result.backend_identifier == "numpy.linalg.svd"
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
