import pytest

from projectkoios.optimization.sampling.uniform import UniformRandomSampler
from projectkoios.optimization.spaces.parameters import ParameterBound, ParameterSpace

pytestmark = pytest.mark.numerical


def test_splitmix64_reference_box_sequence() -> None:
    space = ParameterSpace(
        (ParameterBound("x", -1.0, 1.0), ParameterBound("y", 10.0, 20.0))
    )

    observed = UniformRandomSampler(space).sample(3, random_seed=17)

    assert observed == (
        (0.0040423796433759485, 13.914336791976073),
        (-0.3433403712645877, 12.238961929824475),
        (-0.9654819495224691, 10.814753253696637),
    )
