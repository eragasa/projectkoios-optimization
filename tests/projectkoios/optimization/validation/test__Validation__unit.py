from fractions import Fraction

import pytest

from projectkoios.optimization.validation import validation

pytestmark = pytest.mark.unit


def test_numeric_boundaries_normalize_only_supported_exact_kinds() -> None:
    assert validation.finite_float(Fraction(1, 2), "value") == 0.5
    assert validation.integer(3, "count") == 3
    assert validation.u32(1, "count", allow_zero=False) == 1
    assert validation.u64(0, "seed") == 0

    with pytest.raises(TypeError):
        validation.finite_float(True, "value")
    with pytest.raises(TypeError):
        validation.integer(True, "count")
    with pytest.raises(ValueError):
        validation.u32(0, "count", allow_zero=False)
    with pytest.raises(ValueError):
        validation.u64(-1, "seed")


def test_identifier_boundaries_reject_untrimmed_or_overlong_text() -> None:
    assert validation.text("run", "run_id") == "run"
    assert validation.run_identifier("run", 99) == "run"
    assert validation.candidate_identifier("run", 3) == "run.candidate.00000003"

    with pytest.raises(ValueError):
        validation.text(" run", "run_id")
    with pytest.raises(ValueError):
        validation.run_identifier("x" * 256, 99)
