"""Factory tests for complex-observation affine quadratic construction."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from projectkoios.optimization.affine.complex_quadratic import (
    ComplexAffineResidualQuadraticTolerance,
)
from projectkoios.optimization.affine.factory import (
    ComplexAffineResidualQuadraticConstructorFactory,
    ComplexAffineResidualQuadraticMethod,
)
from projectkoios.optimization.affine.numpy.complex_quadratic import (
    NumpyComplexAffineResidualQuadraticConstructor,
)

pytestmark = pytest.mark.unit


def test_factory_constructs_exact_numpy_backend_with_tolerance() -> None:
    tolerance = ComplexAffineResidualQuadraticTolerance(1.0e-9, 0.0)

    constructor = ComplexAffineResidualQuadraticConstructorFactory.create(
        ComplexAffineResidualQuadraticMethod.NUMPY_DENSE,
        tolerance=tolerance,
    )

    assert type(constructor) is NumpyComplexAffineResidualQuadraticConstructor
    assert constructor.tolerance is tolerance


def test_factory_rejects_non_enum_method() -> None:
    with pytest.raises(TypeError, match="ComplexAffineResidualQuadraticMethod"):
        ComplexAffineResidualQuadraticConstructorFactory.create(  # type: ignore[arg-type]
            "numpy-dense"
        )


def test_factory_module_import_does_not_load_numpy() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = """
import json
import sys
import projectkoios.optimization.affine.factory
print(json.dumps({"numpy": "numpy" in sys.modules}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == {"numpy": False}
