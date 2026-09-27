# Project Koios optimization

This repository owns reusable optimization contracts and deterministic optimization algorithms for Project Koios. The Python distribution is `projectkoios-optimization`; owned modules live under `projectkoios.optimization`.

## Ownership

- Own behaviorful optimizer contracts, candidate proposal and observation/update records, explicit serializable optimizer state/checkpoints, bounded deterministic sampling and optimization algorithms, loss aggregation, and optimization result/report records.
- Keep forward evaluators and results handlers behind narrow injected ports. This repository may define the information those ports exchange; it does not own simulation execution or scientific-domain models.
- `projectkoios-workflow` owns execution authorization, dispatch, retries, persistence, and lifecycle orchestration.
- Simulation and atomistic implementations belong in `projectkoios-simulations` under their neutral, integration, or adapter namespace as appropriate.
- Material-specific campaigns remain application/research-owner artifacts.

## Extraction policy

- Recovery commits and source drops are evidence, not canonical or release-ready implementations.
- Preserve source repository, exact commit, original path, tree/blob identity, namespace rewrite, license, notices, adaptations, and test mapping.
- Extract the smallest reusable behavior. Do not copy unrelated campaigns, calculators, filesystem launchers, workflow state, or legacy package structure.
- Do not vendor PyPosPack, PyMatMC2, or other material with unresolved or incompatible provenance/licensing.
- Do not introduce a behaviorless universal base solely to unify types.
- Prefer immutable records and deterministic, bounded algorithms. Make randomness explicit and seeded.
- Distinguish software verification and numerical verification from scientific validation and human acceptance.

## Development

- Mirror package/module/class ownership under `docs/architecture` using indexed current-state documentation.
- Use strict typing, deterministic sanitized fixtures, failure and awkward cases, reproducible offline builds, wheel import checks, and exact provenance tests.
- Keep one writer per worktree. Use exact-path staging, disable hooks/signing for controlled commits, and verify the staged tree before commit.
- Do not fetch, push, publish, merge, or modify another repository without explicit authorization.
