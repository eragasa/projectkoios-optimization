# Technical debt

This page tracks only unresolved repository-wide maintenance work. Durable rules live in
[Python tests](../../code/python/tests.md) and
[Python code smells](../../code/python/code_smells.md).

## Open

- Obtain an independent review of the bounded cleanup.

## Protected or deferred

- [Repeated finite-periodic tensor evaluation](finite-periodic-repeated-evaluation/index.md)
  remains deferred until production-representative multi-shell profiling demonstrates a
  material runtime or memory cost.
- Provenance-protected GMM and engine code remains unchanged without explicit
  authorization. The protected GMM result constructor retains its complexity finding.
- The protected Pareto algorithm retains its complexity finding and historical operation
  structure.
- Numerical algorithms retain deterministic operation order and exception
  classification during maintenance.
- Scientific validation and consumer policy remain outside this repository.
