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


def test_isolated_wheel_keeps_optional_backends_lazy(
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

import projectkoios.optimization.quadratic.constrained as constrained
from projectkoios.optimization.quadratic.factory import (
    RectangleQuadraticMinimizerFactory,
    RectangleQuadraticMinimizerMethod,
)
from projectkoios.optimization.selection.finite import first_minimum

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert Path(constrained.__file__).resolve().is_relative_to(installed)
assert first_minimum((2.0, 1.0, 1.0)).index == 1
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)

RectangleQuadraticMinimizerFactory.create(
    RectangleQuadraticMinimizerMethod.NUMPY_DENSE
)
assert "numpy" in sys.modules
assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
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
