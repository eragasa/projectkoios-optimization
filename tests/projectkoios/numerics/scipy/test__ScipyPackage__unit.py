"""Unit tests for the import-empty optional SciPy support namespace."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_scipy_support_namespace_imports_no_optional_backend() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = """
import json
import sys
import projectkoios.numerics.scipy as scipy_support
print(json.dumps({
    "exports": scipy_support.__all__,
    "numpy": "numpy" in sys.modules,
    "scipy": "scipy" in sys.modules,
    "typing": "projectkoios.numerics.scipy.typing" in sys.modules,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == {
        "exports": [],
        "numpy": False,
        "scipy": False,
        "typing": False,
    }
