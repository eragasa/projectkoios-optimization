# Evaluation and result-handling boundaries

`optimization/engines/` contains narrow injected evaluation/result-handling behavior and
explicit optimizer construction. It does not own parameter sampling, Pareto selection,
calculator execution, persistence, or workflow lifecycle.

Scalar weighted-loss handling is separate from multi-objective Pareto optimization. It
may feed scalar algorithms such as `CoordinateSearch`; `ParetoOptimizer` consumes ordered
objective vectors directly.

The PhD-work lineage and exact behavior-specific mappings are maintained in
[`docs/provenance/source-mapping.md`](../../../../provenance/source-mapping.md).
