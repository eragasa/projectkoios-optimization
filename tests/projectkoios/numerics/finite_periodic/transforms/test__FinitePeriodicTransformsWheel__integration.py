"""Isolated-wheel verification for centered finite-periodic matrix transforms."""

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


def test_isolated_wheel_exposes_lazy_centered_matrix_transforms(tmp_path: Path) -> None:
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

from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh
from projectkoios.numerics.finite_periodic.transforms import (
    FinitePeriodicParsevalDiagnostics,
)

installed = Path(os.environ["EXPECTED_WHEEL_ROOT"]).resolve()
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
mesh = CenteredUniformMesh(1, 3, (0.5,))
assert mesh.coordinate_origin == (-0.5 + 0.5 / 3.0,)
assert FinitePeriodicParsevalDiagnostics(1.0, 1.0, 0.0).absolute_defect == 0.0

import numpy as np
from projectkoios.numerics.finite_periodic.numpy import matrix_evaluation
from projectkoios.numerics.finite_periodic.numpy import tensor_product_evaluation
from projectkoios.numerics.finite_periodic.numpy import transforms as capability
from projectkoios.numerics.finite_periodic.numpy.matrix_evaluation import (
    NumpyFiniteFourierMatrixEvaluationDefinition,
    NumpyFiniteFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.tensor_product_evaluation import (
    NumpyTensorProductFourierMatrixEvaluationDefinition,
    NumpyTensorProductFourierMatrixSeriesEvaluator,
)
from projectkoios.numerics.finite_periodic.numpy.transform_diagnostics import (
    NumpyFinitePeriodicParsevalAnalyzer,
)
from projectkoios.numerics.finite_periodic.numpy.transforms import (
    NumpyCenteredUniformMeshMatrixFourierTransformDefinition,
    NumpyCenteredUniformMeshMatrixFourierTransformer,
)

assert Path(capability.__file__).resolve().is_relative_to(installed)
assert Path(matrix_evaluation.__file__).resolve().is_relative_to(installed)
assert Path(tensor_product_evaluation.__file__).resolve().is_relative_to(installed)
samples = np.full((3, 1, 1), 2.0 + 0.0j, dtype=np.complex128)
definition = NumpyCenteredUniformMeshMatrixFourierTransformDefinition.from_array(
    mesh,
    samples,
)
transformed = NumpyCenteredUniformMeshMatrixFourierTransformer(
    definition=definition
).result()
coordinates = (-0.5 + (np.arange(3, dtype=np.float64) + 0.5) / 3.0)[:, None]
evaluated = NumpyFiniteFourierMatrixSeriesEvaluator().evaluate(
    NumpyFiniteFourierMatrixEvaluationDefinition.from_array(
        transformed.series,
        coordinates,
        maximum_phase_elements=1,
    )
)
assert np.allclose(evaluated.matrix_values, samples, rtol=0.0, atol=1.0e-15)
tensor_evaluated = NumpyTensorProductFourierMatrixSeriesEvaluator(
    definition=NumpyTensorProductFourierMatrixEvaluationDefinition.from_arrays(
        mesh,
        transformed.series,
        (coordinates[:, 0],),
        maximum_working_elements=4,
    )
).result()
assert np.allclose(tensor_evaluated.matrix_values, samples, rtol=0.0, atol=1.0e-15)
parseval = NumpyFinitePeriodicParsevalAnalyzer(
        definition=definition,
        transform_result=transformed,
    ).result()
assert parseval.absolute_defect == 0.0
assert transformed.backend_identifier == "numpy.fft.fftn"
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
