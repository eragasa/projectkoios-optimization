import pytest

from projectkoios.optimization.multiobjective.pareto.shells import pareto_shells

pytestmark = pytest.mark.unit


def test_shells_recursively_peel_and_pull_back_original_indices() -> None:
    decisions = tuple((float(index),) for index in range(8))
    objectives = (
        (1.0, 4.0),
        (2.0, 3.0),
        (3.0, 2.0),
        (4.0, 1.0),
        (2.0, 4.0),
        (3.0, 3.0),
        (4.0, 2.0),
        (4.0, 4.0),
    )

    shells = pareto_shells(decisions, objectives, shell_count=3, chunk_size=2)

    assert tuple(shell.rank for shell in shells) == (0, 1, 2)
    assert tuple(shell.indices for shell in shells) == (
        (0, 1, 2, 3),
        (4, 5, 6),
        (7,),
    )
    assert shells[1].decision_vectors == ((4.0,), (5.0,), (6.0,))
    assert shells[1].objective_vectors == (
        (2.0, 4.0),
        (3.0, 3.0),
        (4.0, 2.0),
    )


def test_equal_objective_vectors_receive_the_same_shell_rank() -> None:
    decisions = ((0.0,), (1.0,), (2.0,), (3.0,))
    objectives = ((1.0, 2.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0))

    shells = pareto_shells(decisions, objectives)

    assert tuple(shell.indices for shell in shells) == ((0, 1, 2), (3,))


def test_shell_count_bounds_recursive_peeling() -> None:
    decisions = ((0.0,), (1.0,), (2.0,))
    objectives = ((0.0,), (1.0,), (2.0,))

    shells = pareto_shells(decisions, objectives, shell_count=2)

    assert tuple(shell.indices for shell in shells) == ((0,), (1,))


def test_shells_reject_misaligned_or_inconsistent_observations() -> None:
    with pytest.raises(ValueError, match="counts must match"):
        pareto_shells(((0.0,),), ())
    with pytest.raises(ValueError, match="objective-vector dimension"):
        pareto_shells(((0.0,), (1.0,)), ((1.0,), (2.0, 3.0)))
    with pytest.raises(ValueError, match="shell_count"):
        pareto_shells((), (), shell_count=0)
