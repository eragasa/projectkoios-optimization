import pytest

from projectkoios.optimization.validation import validation

pytestmark = pytest.mark.unit

NUMERICAL_VALIDATOR_NAMES = ("finite_float", "integer", "u32", "u64")


def test_numerical_validators_are_owned_by_numerics() -> None:
    for name in NUMERICAL_VALIDATOR_NAMES:
        assert not hasattr(validation, name)


def test_identifier_boundaries_reject_untrimmed_or_overlong_text() -> None:
    assert validation.text("run", "run_id") == "run"
    assert validation.run_identifier("run", 99) == "run"
    assert validation.candidate_identifier("run", 3) == "run.candidate.00000003"

    with pytest.raises(ValueError):
        validation.text(" run", "run_id")
    with pytest.raises(ValueError):
        validation.run_identifier("x" * 256, 99)
