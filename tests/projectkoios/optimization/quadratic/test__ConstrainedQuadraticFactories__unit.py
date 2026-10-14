import os
import subprocess
import sys

import pytest

from projectkoios.optimization.quadratic.constrained import (
    ConstrainedQuadraticTolerance,
)
from projectkoios.optimization.quadratic.factory import (
    QuadraticSublevelSupportMethod,
    QuadraticSublevelSupportSolverFactory,
    RectangleQuadraticMinimizerFactory,
    RectangleQuadraticMinimizerMethod,
)

pytestmark = pytest.mark.unit


def test_factory_import_keeps_optional_backends_lazy() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src/python"
    program = """
import sys
from projectkoios.optimization.quadratic.factory import (
    QuadraticSublevelSupportSolverFactory,
    RectangleQuadraticMinimizerFactory,
)
assert QuadraticSublevelSupportSolverFactory is not None
assert RectangleQuadraticMinimizerFactory is not None
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_factories_require_exact_closed_methods_and_tolerance() -> None:
    with pytest.raises(TypeError, match="RectangleQuadraticMinimizerMethod"):
        RectangleQuadraticMinimizerFactory.create("numpy-dense")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="QuadraticSublevelSupportMethod"):
        QuadraticSublevelSupportSolverFactory.create("numpy-dense")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ConstrainedQuadraticTolerance"):
        RectangleQuadraticMinimizerFactory.create(
            RectangleQuadraticMinimizerMethod.NUMPY_DENSE,
            tolerance=(1.0e-12, 1.0e-10),  # type: ignore[arg-type]
        )

    tolerance = ConstrainedQuadraticTolerance(absolute=1.0e-11)
    assert tolerance.absolute == 1.0e-11
    assert QuadraticSublevelSupportMethod.SCIPY_SPARSE.value == "scipy-sparse"
