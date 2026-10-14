"""Unit tests for the import-empty Numerics namespace."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_numerics_namespace_imports_no_capability_or_backend() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = """
import json
import sys
import projectkoios.numerics as numerics
print(json.dumps({
    "exports": numerics.__all__,
    "finite_series": (
        "projectkoios.numerics.finite_periodic.finite_series" in sys.modules
    ),
    "numpy": "numpy" in sys.modules,
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
        "finite_series": False,
        "numpy": False,
    }
