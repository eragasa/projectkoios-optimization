import os
import subprocess
import sys

import pytest

from projectkoios.optimization.least_squares.factory import (
    LinearLeastSquaresMethod,
    LinearLeastSquaresSolverFactory,
)
from projectkoios.optimization.least_squares.linear import (
    LinearLeastSquaresTolerance,
)

pytestmark = pytest.mark.unit


def test_factory_import_keeps_numpy_lazy() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src/python"
    program = """
import sys
from projectkoios.optimization.least_squares.factory import (
    LinearLeastSquaresSolverFactory,
)
assert LinearLeastSquaresSolverFactory is not None
assert "numpy" not in sys.modules
assert "scipy" not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_factory_requires_exact_closed_method_and_tolerance() -> None:
    with pytest.raises(TypeError, match="LinearLeastSquaresMethod"):
        LinearLeastSquaresSolverFactory.create("numpy-dense")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LinearLeastSquaresTolerance"):
        LinearLeastSquaresSolverFactory.create(
            LinearLeastSquaresMethod.NUMPY_DENSE,
            tolerance=(1.0e-12, 1.0e-10),  # type: ignore[arg-type]
        )

    solver = LinearLeastSquaresSolverFactory.create(
        LinearLeastSquaresMethod.NUMPY_DENSE,
        tolerance=LinearLeastSquaresTolerance(absolute=1.0e-11),
    )

    assert type(solver).__name__ == "NumpyLinearLeastSquaresSolver"
    assert solver.tolerance.absolute == 1.0e-11
