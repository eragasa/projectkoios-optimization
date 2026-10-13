from fractions import Fraction

import pytest

from projectkoios.optimization.engines.records import Candidate

pytestmark = pytest.mark.unit


def test_candidate_normalizes_finite_decision_values() -> None:
    candidate = Candidate("c", "run", 0, (Fraction(1, 2),))  # type: ignore[arg-type]

    assert candidate.values == (0.5,)
