import random
import time
from itertools import batched

import pytest

from projectkoios.optimization.multiobjective.pareto.algorithm import (
    pareto,
    pareto_bruteforce,
)

pytestmark = pytest.mark.numerical


def _donor_dominates(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    for left, right in zip(first, second):  # noqa: B905, SIM110
        if right < left:
            return False
    return True


def _donor_bruteforce(
    points: tuple[tuple[float, ...], ...],
    indices: tuple[int, ...] | list[int] | None = None,
) -> list[int]:
    selected = list(range(len(points))) if indices is None else indices
    result: list[int] = []
    for first in selected:
        for second in selected:
            if first == second:
                continue
            if _donor_dominates(points[second], points[first]):
                break
        else:
            result.append(first)
    return result


def _corrected_donor_pareto(
    points: tuple[tuple[float, ...], ...],
    chunk_size: int,
) -> list[int]:
    indices = list(range(len(points)))
    size = chunk_size
    old_length = len(indices)
    while True:
        if len(indices) < size:
            break
        chunks = batched(indices, size)  # noqa: B911
        indices = []
        for chunk in chunks:
            indices += _donor_bruteforce(points, chunk)
        new_length = len(indices)
        if old_length == new_length:
            break
        if old_length - new_length < size:
            size = 2 * size
        old_length = new_length
    return _donor_bruteforce(points, indices)


def test_chunked_selection_is_exact_and_preserves_donor_performance() -> None:
    generator = random.Random(20261012)
    points = tuple(
        tuple(quality + 0.1 * generator.random() for _ in range(10))
        for quality in (generator.random() for _ in range(20_000))
    )

    port_times: list[float] = []
    donor_times: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        selected = pareto(points, chunk_sz=500)
        port_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        donor_selected = _corrected_donor_pareto(points, 500)
        donor_times.append(time.perf_counter() - started)

    port_seconds = sorted(port_times)[2]
    donor_seconds = sorted(donor_times)[2]

    started = time.perf_counter()
    expected = pareto_bruteforce(points)
    brute_seconds = time.perf_counter() - started

    assert selected == donor_selected == expected
    assert len(selected) == 387
    assert port_seconds <= 1.25 * donor_seconds, (
        f"port={port_seconds:.6f}s corrected-donor={donor_seconds:.6f}s"
    )
    assert 2.0 * port_seconds < brute_seconds, (
        f"chunked={port_seconds:.6f}s brute={brute_seconds:.6f}s"
    )
