"""Explicit construction of maintained multi-objective optimizers."""

from __future__ import annotations

from ..spaces.parameters import ParameterSpace
from .pareto.optimizer import ParetoOptimizer, ParetoSamplingStage


class MultiObjectiveOptimizerFactory:
    """Construct concrete multi-objective optimizers without string dispatch."""

    __slots__ = ()

    def create_pareto(
        self,
        space: ParameterSpace,
        stages: tuple[ParetoSamplingStage, ...],
        *,
        seed: int,
        run_id: str,
        chunk_size: int = 500,
    ) -> ParetoOptimizer:
        """Construct iterative Pareto optimization over explicit sampling stages."""
        return ParetoOptimizer(
            space,
            stages,
            seed=seed,
            run_id=run_id,
            chunk_size=chunk_size,
        )


__all__ = ["MultiObjectiveOptimizerFactory"]
