import random

import pytest

from projectkoios.optimization.multiobjective.pareto.algorithm import (
    dominates,
    pareto,
    pareto_bruteforce,
)

pytestmark = pytest.mark.unit


def historical_pareto_bruteforce(
    points: tuple[tuple[float, ...], ...],
    indices: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Execute the exact PyPosPack brute-force comparison structure."""
    selected = list(range(len(points))) if indices is None else indices
    result: list[int] = []
    for first in selected:
        for second in selected:
            if first == second:
                continue
            if all(
                left <= right
                for left, right in zip(points[second], points[first], strict=False)
            ):
                break
        else:
            result.append(first)
    return result


def historical_pareto(
    points: tuple[tuple[float, ...], ...],
    chunk_size: int,
) -> list[int]:
    """Execute the historical full-batch-only PyPosPack chunk loop."""
    indices = list(range(len(points)))
    size = chunk_size
    old_length = len(indices)
    while True:
        if len(indices) < size:
            break
        chunks = zip(*[iter(indices)] * size, strict=False)
        indices = []
        for chunk in chunks:
            indices += historical_pareto_bruteforce(points, chunk)
        new_length = len(indices)
        if old_length == new_length:
            break
        if old_length - new_length < size:
            size = 2 * size
        old_length = new_length
    return historical_pareto_bruteforce(points, indices)


def test_port_matches_donor_when_no_incomplete_batch_occurs() -> None:
    tradeoff = tuple((float(index), float(999 - index)) for index in range(1_000))
    correlated = tuple((float(index), float(index)) for index in range(1_000))

    assert pareto(tradeoff, chunk_sz=500) == historical_pareto(tradeoff, 500)
    assert pareto(correlated, chunk_sz=500) == historical_pareto(correlated, 500)


def test_bruteforce_matches_donor_across_generated_inputs_and_subsets() -> None:
    generator = random.Random(201612)
    for size in range(101):
        dimension = 1 + size % 8
        points = tuple(
            tuple(float(generator.randrange(7)) for _ in range(dimension))
            for _ in range(size)
        )
        subset = tuple(index for index in range(size) if index % 3 != 1)

        assert pareto_bruteforce(points) == historical_pareto_bruteforce(points)
        assert pareto_bruteforce(points, subset) == historical_pareto_bruteforce(
            points,
            subset,
        )


def test_port_preserves_historical_equal_vector_semantics() -> None:
    points = ((1.0, 1.0), (1.0, 1.0), (2.0, 2.0))

    assert dominates(points[0], points[1]) is True
    assert pareto_bruteforce(points) == historical_pareto_bruteforce(points) == []


def test_corrected_port_retains_the_historical_incomplete_tail() -> None:
    points = tuple((float(index), float(500 - index)) for index in range(501))

    assert historical_pareto(points, 500) == list(range(500))
    assert pareto(points, chunk_sz=500) == list(range(501))
