# Results and reports

`ScoredResult` correlates one candidate and feedback record.
`OptimizationReport` stores canonical `(loss, candidate_id)` ranking and sorted
Pareto identities. Pareto dominance uses the ordered objective vector: another
result dominates only if all objectives are no worse and at least one is
strictly better. Objective names and order must match across a report.

Reports are pure in-memory records. They do not write files, render plots,
select scientific acceptance criteria, or claim scientific validation.
