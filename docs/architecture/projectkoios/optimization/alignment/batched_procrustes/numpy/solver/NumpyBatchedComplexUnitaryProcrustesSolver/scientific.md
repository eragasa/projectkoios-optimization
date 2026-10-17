# Solver consumer needs

Consumers need deterministic shape-preserving execution and explicit failure rather than
partial results. They remain responsible for interpreting ranks and residuals, choosing
scientific thresholds, and deciding whether downstream work is applicable.

The solver receives no domain objects and authorizes no scientific conclusion.
