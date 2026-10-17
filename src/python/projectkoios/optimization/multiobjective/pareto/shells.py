"""Recursive Pareto shells with exact decision-vector pullback.

This owner-authored capability is separate from the provenance-bound PyPosPack kernel in
:mod:`projectkoios.optimization.multiobjective.pareto.algorithm`. It repeatedly invokes
that kernel without modifying it, groups equal objective vectors so equal observations
receive the same shell rank, and pulls every selected objective shell back to the
corresponding decision vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

from projectkoios.numerics.validation import finite_float, integer

from .algorithm import pareto


@dataclass(frozen=True, slots=True)
class ParetoShell:
    """One zero-based nondominated shell and its exact decision-space pullback."""

    rank: int
    indices: tuple[int, ...]
    decision_vectors: tuple[tuple[float, ...], ...]
    objective_vectors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        rank = integer(self.rank, "rank")
        if rank < 0:
            raise ValueError("rank must be nonnegative")
        if type(self.indices) is not tuple or not self.indices:
            raise TypeError("indices must be a nonempty tuple")
        indices = tuple(integer(index, "observation index") for index in self.indices)
        if any(index < 0 for index in indices):
            raise ValueError("observation indices must be nonnegative")
        if len(set(indices)) != len(indices):
            raise ValueError("observation indices must be unique")
        decisions, objectives = _validated_observations(
            self.decision_vectors,
            self.objective_vectors,
        )
        if len(indices) != len(decisions):
            raise ValueError("indices and shell vector counts must match")
        object.__setattr__(self, "decision_vectors", decisions)
        object.__setattr__(self, "objective_vectors", objectives)


def pareto_shells(
    decision_vectors: tuple[tuple[float, ...], ...],
    objective_vectors: tuple[tuple[float, ...], ...],
    /,
    *,
    shell_count: int | None = None,
    chunk_size: int = 500,
) -> tuple[ParetoShell, ...]:
    """Peel ordered Pareto shells and pull each shell back to decision space.

    Shell zero is the nondominated set. Shell one is the nondominated set after shell
    zero is removed, and so on. Equal objective vectors are deliberately grouped before
    invoking the donor kernel because its historical weak-dominance semantics otherwise
    remove every member of an equal-vector group.
    """
    decisions, objectives = _validated_observations(
        decision_vectors,
        objective_vectors,
    )
    if shell_count is not None:
        maximum_shells = integer(shell_count, "shell_count")
        if maximum_shells < 1:
            raise ValueError("shell_count must be positive when provided")
    else:
        maximum_shells = len(objectives)
    size = integer(chunk_size, "chunk_size")
    if size < 1:
        raise ValueError("chunk_size must be positive")

    remaining = list(range(len(objectives)))
    result: list[ParetoShell] = []
    while remaining and len(result) < maximum_shells:
        unique_objectives = tuple(
            dict.fromkeys(objectives[index] for index in remaining)
        )
        selected_unique_indices = pareto(unique_objectives, chunk_sz=size)
        selected_objectives = {
            unique_objectives[index] for index in selected_unique_indices
        }
        selected = tuple(
            index for index in remaining if objectives[index] in selected_objectives
        )
        if not selected:
            raise RuntimeError("Pareto shell peeling made no progress")
        selected_set = set(selected)
        result.append(
            ParetoShell(
                rank=len(result),
                indices=selected,
                decision_vectors=tuple(decisions[index] for index in selected),
                objective_vectors=tuple(objectives[index] for index in selected),
            )
        )
        remaining = [index for index in remaining if index not in selected_set]
    return tuple(result)


def _validated_observations(
    decision_vectors: tuple[tuple[float, ...], ...],
    objective_vectors: tuple[tuple[float, ...], ...],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    if type(decision_vectors) is not tuple:
        raise TypeError("decision_vectors must be a tuple")
    if type(objective_vectors) is not tuple:
        raise TypeError("objective_vectors must be a tuple")
    if len(decision_vectors) != len(objective_vectors):
        raise ValueError("decision and objective vector counts must match")

    decisions: list[tuple[float, ...]] = []
    objectives: list[tuple[float, ...]] = []
    decision_dimension: int | None = None
    objective_dimension: int | None = None
    for decision, objective in zip(
        decision_vectors,
        objective_vectors,
        strict=True,
    ):
        if type(decision) is not tuple:
            raise TypeError("each decision vector must be a tuple")
        if type(objective) is not tuple or not objective:
            raise TypeError("each objective vector must be a nonempty tuple")
        if decision_dimension is None:
            decision_dimension = len(decision)
            objective_dimension = len(objective)
        elif len(decision) != decision_dimension:
            raise ValueError("decision-vector dimension must remain constant")
        elif len(objective) != objective_dimension:
            raise ValueError("objective-vector dimension must remain constant")
        decisions.append(
            tuple(finite_float(value, "decision value") for value in decision)
        )
        objectives.append(
            tuple(finite_float(value, "objective value") for value in objective)
        )
    return tuple(decisions), tuple(objectives)


__all__ = ["ParetoShell", "pareto_shells"]
