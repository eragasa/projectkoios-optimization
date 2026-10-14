"""Isolated-wheel smoke test for finite Fourier-series evaluation."""

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


def test_isolated_wheel_exposes_lazy_finite_fourier_series(
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

    script = """
import os
from pathlib import Path
import sys

import projectkoios.numerics.finite_periodic.finite_series as finite_series
from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluationDefinition,
    FiniteFourierSeriesEvaluatorFactory,
    FiniteFourierSeriesMethod,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(finite_series.__file__).resolve().is_relative_to(installed)
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)

evaluator = FiniteFourierSeriesEvaluatorFactory.create(
    FiniteFourierSeriesMethod.NUMPY_DENSE
)
result = evaluator.evaluate(
    FiniteFourierSeriesEvaluationDefinition(
        1,
        ((0.25,),),
        ((1,),),
        (1.0 + 0.0j,),
    )
)
assert abs(result.values[0] - 1.0j) < 1.0e-15
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
