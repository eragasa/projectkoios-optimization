"""Unit tests for the lazy box-quadratic minimizer factory."""

import os
import subprocess
import sys

import pytest

from projectkoios.optimization.quadratic.box import BoxQuadraticMinimizer
from projectkoios.optimization.quadratic.constrained import (
    ConstrainedQuadraticTolerance,
)
from projectkoios.optimization.quadratic.factory import (
    BoxQuadraticMinimizerFactory,
    BoxQuadraticMinimizerMethod,
)

pytestmark = pytest.mark.unit


def test_factory_contract_is_closed_and_lazy() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src/python"
    program = """
import sys
from projectkoios.optimization.quadratic.factory import (
    BoxQuadraticMinimizerFactory,
    BoxQuadraticMinimizerMethod,
)
assert BoxQuadraticMinimizerFactory is not None
assert BoxQuadraticMinimizerMethod.NUMPY_DENSE.value == "numpy-dense"
assert "numpy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_factory_validates_nominal_inputs_and_constructs_backend() -> None:
    with pytest.raises(TypeError, match="BoxQuadraticMinimizerMethod"):
        BoxQuadraticMinimizerFactory.create("numpy-dense")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ConstrainedQuadraticTolerance"):
        BoxQuadraticMinimizerFactory.create(
            BoxQuadraticMinimizerMethod.NUMPY_DENSE,
            tolerance=(1.0e-12, 1.0e-10),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="maximum_iterations"):
        BoxQuadraticMinimizerFactory.create(
            BoxQuadraticMinimizerMethod.NUMPY_DENSE,
            maximum_iterations=True,
        )
    with pytest.raises(ValueError, match="positive u32"):
        BoxQuadraticMinimizerFactory.create(
            BoxQuadraticMinimizerMethod.NUMPY_DENSE,
            maximum_iterations=0,
        )

    tolerance = ConstrainedQuadraticTolerance(absolute=1.0e-11)
    solver = BoxQuadraticMinimizerFactory.create(
        BoxQuadraticMinimizerMethod.NUMPY_DENSE,
        tolerance=tolerance,
    )
    assert isinstance(solver, BoxQuadraticMinimizer)
    assert solver.tolerance is tolerance
    assert solver.backend_identifier == "projectkoios.active-set.numpy"
    assert solver.maximum_iterations == 10_000
