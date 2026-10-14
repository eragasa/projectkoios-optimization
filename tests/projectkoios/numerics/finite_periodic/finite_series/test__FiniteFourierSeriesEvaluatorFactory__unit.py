"""Factory tests for finite Fourier-series evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, cast

import pytest

from projectkoios.numerics.finite_periodic.finite_series import (
    FiniteFourierSeriesEvaluatorFactory,
    FiniteFourierSeriesMethod,
)
from projectkoios.numerics.finite_periodic.numpy.finite_series import (
    NumpyFiniteFourierSeriesEvaluator,
)

pytestmark = pytest.mark.unit


def test_factory_constructs_exact_numpy_backend() -> None:
    evaluator = FiniteFourierSeriesEvaluatorFactory.create(
        FiniteFourierSeriesMethod.NUMPY_DENSE
    )

    assert type(evaluator) is NumpyFiniteFourierSeriesEvaluator


def test_factory_rejects_non_enum_method() -> None:
    with pytest.raises(TypeError, match="FiniteFourierSeriesMethod"):
        FiniteFourierSeriesEvaluatorFactory.create(cast(Any, "numpy-dense"))


def test_contract_module_import_does_not_load_numpy() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = """
import json
import sys
import projectkoios.numerics.finite_periodic.finite_series
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
