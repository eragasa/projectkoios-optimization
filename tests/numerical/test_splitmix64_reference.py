import pytest

from projectkoios.optimization import (
    BoundedUniformOptimizer,
    ParameterBound,
    ParameterSpace,
)

pytestmark = pytest.mark.numerical


def test_splitmix64_reference_box_sequence() -> None:
    space = ParameterSpace(
        (ParameterBound("x", -1.0, 1.0), ParameterBound("y", 10.0, 20.0))
    )
    observed = tuple(
        BoundedUniformOptimizer.sample_values(space, 17, index) for index in range(3)
    )
    assert observed == (
        (0.0040423796433759485, 13.914336791976073),
        (-0.3433403712645877, 12.238961929824475),
        (-0.9654819495224691, 10.814753253696637),
    )
