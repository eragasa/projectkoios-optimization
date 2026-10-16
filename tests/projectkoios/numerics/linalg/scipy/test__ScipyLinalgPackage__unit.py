"""Unit tests for the import-empty linear-algebra SciPy namespace."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_scipy_linalg_namespace_does_not_import_optional_dependencies() -> None:
    root = Path(__file__).resolve().parents[5]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src" / "python")
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import projectkoios.numerics.linalg.scipy; "
            "assert 'numpy' not in sys.modules; assert 'scipy' not in sys.modules",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
