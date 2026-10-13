import pytest

from projectkoios.optimization.quadratic import validation

pytestmark = pytest.mark.unit


def test_quadratic_matrix_checks_preserve_exact_shape_and_symmetry() -> None:
    matrix = validation.finite_square_matrix(((2.0, 0.5), (0.5, 3.0)), 2)

    assert matrix == ((2.0, 0.5), (0.5, 3.0))
    assert validation.symmetric(matrix) is True
    assert validation.symmetric(((2.0, 0.5), (0.25, 3.0))) is False


def test_quadratic_vector_and_matrix_checks_reject_wrong_structure() -> None:
    with pytest.raises(TypeError):
        validation.finite_vector([1.0], "center")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validation.finite_square_matrix(((1.0, 0.0),), 2)
