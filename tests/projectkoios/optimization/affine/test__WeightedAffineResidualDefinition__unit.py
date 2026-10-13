import pytest

pytestmark = pytest.mark.unit

from projectkoios.optimization.affine.affine import (  # noqa: E402
    WeightedAffineResidualDefinition,
    WeightedAffineResidualTerm,
)


def definition() -> WeightedAffineResidualDefinition:
    return WeightedAffineResidualDefinition(
        (
            WeightedAffineResidualTerm((1.0, 0.0), 1.0, 2.0),
            WeightedAffineResidualTerm((0.0, 1.0), -2.0, 3.0),
            WeightedAffineResidualTerm((1.0, 1.0), -1.0, 0.5),
        )
    )


def test_definition_is_immutable_and_evaluates_weighted_residuals() -> None:
    residuals = definition()

    assert residuals.dimension == 2
    assert residuals.squared_loss((1.0, -2.0)) == 0.0
    assert residuals.squared_loss((3.0, -1.0)) == 15.5
    with pytest.raises(AttributeError):
        residuals.terms = ()  # type: ignore[misc]


def test_definition_rejects_invalid_scalar_and_dimension_inputs() -> None:
    with pytest.raises(TypeError):
        WeightedAffineResidualTerm((True,), 0.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WeightedAffineResidualTerm((1.0,), 0.0, 0.0)
    with pytest.raises(ValueError, match="common dimension"):
        WeightedAffineResidualDefinition(
            (
                WeightedAffineResidualTerm((1.0, 2.0), 0.0, 1.0),
                WeightedAffineResidualTerm((1.0,), 0.0, 1.0),
            )
        )


def test_direct_evaluation_rejects_container_dimension_and_overflow() -> None:
    residuals = definition()

    with pytest.raises(TypeError):
        residuals.squared_loss([1.0, -2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        residuals.squared_loss((1.0,))
    with pytest.raises(ValueError, match="nonfinite"):
        WeightedAffineResidualDefinition(
            (WeightedAffineResidualTerm((1.0,), 0.0, 1.0),)
        ).squared_loss((1.0e308,))
