"""Unit tests for the import-empty optional NumPy support namespace."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_numpy_support_namespace_does_not_import_numpy() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = """
import json
import sys
import projectkoios.numerics.numpy as numpy_support
print(json.dumps({
    "exports": numpy_support.__all__,
    "numpy": "numpy" in sys.modules,
    "typing": "projectkoios.numerics.numpy.typing" in sys.modules,
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
        "typing": False,
    }
