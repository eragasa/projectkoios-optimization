import os
import subprocess
import sys

import pytest

from projectkoios.optimization.probability.factory import ProbabilitySamplerFactory
from projectkoios.optimization.sampling.uniform import UniformRandomSampler
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.unit


def test_factory_constructs_uniform_parameter_sampler() -> None:
    space = ParameterSpace((ParameterBound("x", 0.0, 1.0),))

    sampler = ProbabilitySamplerFactory().create_uniform_random(space)

    assert type(sampler) is UniformRandomSampler
    assert sampler.space is space


def test_factory_import_keeps_optional_numerical_backends_lazy() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src/python"
    program = """
import sys
from projectkoios.optimization.probability.factory import ProbabilitySamplerFactory
assert ProbabilitySamplerFactory is not None
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
assert 'sklearn' not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
