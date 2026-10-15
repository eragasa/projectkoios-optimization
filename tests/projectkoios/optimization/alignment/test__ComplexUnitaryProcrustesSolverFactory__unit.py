"""Unit tests for the closed lazy complex-unitary Procrustes factory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesMethod,
    ComplexUnitaryProcrustesSolverFactory,
)

pytestmark = pytest.mark.unit


def test_method_vocabulary_is_closed() -> None:
    assert tuple(ComplexUnitaryProcrustesMethod) == (
        ComplexUnitaryProcrustesMethod.NUMPY_DENSE,
    )
    assert ComplexUnitaryProcrustesMethod.NUMPY_DENSE.value == "numpy-dense"


def test_factory_rejects_strings_and_foreign_values() -> None:
    for value in ("numpy-dense", object(), True):
        with pytest.raises(TypeError, match="ComplexUnitaryProcrustesMethod"):
            ComplexUnitaryProcrustesSolverFactory.create(cast(Any, value))


def test_contract_import_does_not_eagerly_import_numpy() -> None:
    root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src" / "python")
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import projectkoios.optimization.alignment.procrustes; "
            "assert 'numpy' not in sys.modules; "
            "assert 'projectkoios.optimization.alignment.numpy.procrustes' "
            "not in sys.modules",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_factory_lazily_constructs_nominal_numpy_solver() -> None:
    from projectkoios.optimization.alignment.numpy.procrustes import (
        NumpyComplexUnitaryProcrustesSolver,
    )

    solver = ComplexUnitaryProcrustesSolverFactory.create(
        ComplexUnitaryProcrustesMethod.NUMPY_DENSE
    )

    assert type(solver) is NumpyComplexUnitaryProcrustesSolver
