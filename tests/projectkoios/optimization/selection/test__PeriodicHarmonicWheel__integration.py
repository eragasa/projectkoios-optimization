"""Isolated-wheel smoke test for periodic harmonic maximization."""

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


def test_isolated_wheel_executes_without_optional_backend(tmp_path: Path) -> None:
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    backend = _load_build_backend()
    wheel_path = wheel_directory / backend.build_wheel(str(wheel_directory))
    installed = tmp_path / "installed"
    installed.mkdir()
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(installed)

    script = """
import math
import os
from pathlib import Path
import sys

from projectkoios.optimization.selection import harmonic as module
from projectkoios.optimization.selection.harmonic import (
    PeriodicHarmonic,
    PeriodicHarmonicMaximumStatus,
    PeriodicHarmonicMaximizer,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(module.__file__).resolve().is_relative_to(installed)
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
result = PeriodicHarmonicMaximizer().maximize(
    PeriodicHarmonic(0.25, 0.6, 0.8, 2, -math.pi / 2.0)
)
assert result.status is PeriodicHarmonicMaximumStatus.UNIQUE
assert result.maximum_value == 1.25
assert math.isclose(
    result.canonical_maximizing_argument,
    0.5 * math.atan2(0.8, 0.6),
    rel_tol=0.0,
    abs_tol=1.0e-16,
)
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
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
