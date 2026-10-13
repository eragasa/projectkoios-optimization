"""Performance-oriented Pareto selection adapted from PyPosPack.

This provenance-bound algorithm originates in Eugene J. Ragasa's PhD work. The source
header credits Dmitriy Morozov with December 2016 performance improvements. Do not move,
split, refactor, or modify it without explicit human authorization and a passing speed
regression against the minimally corrected donor implementation in
``test__ParetoAlgorithm__numerical.py``.

The comparison and adaptive chunk-pruning structure are preserved. The historical
full-batch-only ``zip`` idiom is replaced by lazy :func:`itertools.batched` so the final
short batch is retained.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import batched


def dominates(p1: Sequence[float], p2: Sequence[float], /) -> bool:
    """Return whether ``p1`` is no larger than ``p2`` in every objective."""
    for x, y in zip(p1, p2):  # noqa: B905, SIM110 - preserve donor hot loop
        if y < x:
            return False
    return True


def pareto_bruteforce(
    pts: Sequence[Sequence[float]],
    indices: Sequence[int] | None = None,
    /,
) -> list[int]:
    """Return nondominated indices from one caller-declared subset."""
    # PERFORMANCE GUARD: this is the donor hot loop. Do not change it without running
    # the corrected-donor speed regression in test__ParetoAlgorithm__numerical.py.
    if indices is None:
        indices = list(range(len(pts)))
    result: list[int] = []
    for i in indices:
        for j in indices:
            if i == j:
                continue
            if dominates(pts[j], pts[i]):
                break
        else:
            result.append(i)
    return result


def pareto(
    pts: Sequence[Sequence[float]],
    chunk_sz: int = 500,
    is_debug: bool = False,
) -> list[int]:
    """Return Pareto indices using the PyPosPack adaptive chunk-pruning algorithm."""
    if type(chunk_sz) is not int:
        raise TypeError("chunk_sz must be a built-in integer")
    if chunk_sz < 1:
        raise ValueError("chunk_sz must be positive")
    if type(is_debug) is not bool:
        raise TypeError("is_debug must be a built-in bool")

    idx = list(range(len(pts)))
    sz = chunk_sz

    old_len_idx = len(idx)
    while True:
        if len(idx) < sz:
            break
        if is_debug:
            print(f"total_idx {len(idx)} > chunk_size {sz}")
        chunks = batched(idx, sz)  # noqa: B911 - final short batch is required
        idx = []
        for i, c in enumerate(chunks):
            if is_debug:
                print(f"\tworking on chunk {i}")
            idx += pareto_bruteforce(pts, c)

        new_len_idx = len(idx)
        if old_len_idx == new_len_idx:
            break
        if (old_len_idx - new_len_idx) < sz:
            sz = 2 * sz
        old_len_idx = new_len_idx
    if is_debug:
        print(f"running final pareto calculation on {len(idx)} points.")
    return pareto_bruteforce(pts, idx)


__all__ = ["dominates", "pareto", "pareto_bruteforce"]
