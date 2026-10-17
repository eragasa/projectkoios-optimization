# Project Koios optimization

This repository owns reusable mathematical numerics and optimization for Project Koios. The Python distribution is `projectkoios-optimization`; owned modules live under `projectkoios.numerics` and `projectkoios.optimization`.

## Ownership

- `projectkoios.numerics` owns reusable transformations, decompositions, geometry, numerical analysis, immutable records, diagnostics, and capability-local backends.
- `projectkoios.optimization` owns reusable function and multi-function optimization contracts, candidate proposal and observation behavior, fitting, constrained minimization, alignment, assignment, selection, clustering, sampling, search, immutable algorithm state, loss aggregation, and results.
- Numerics may support Optimization; Numerics must not import Optimization or downstream scientific packages.
- Consumers own units, physical and material meaning, study thresholds, evidence interpretation, and scientific acceptance.
- Keep forward evaluators and result handlers behind narrow injected ports. This repository may define exchanged data, but simulation execution and scientific-domain models remain external.
- `projectkoios-workflow` owns authorization, dispatch, retries, persistence, and lifecycle orchestration.
- Atomistic implementations belong in the appropriate neutral, integration, or adapter namespace of `projectkoios-simulations`; material-specific campaigns remain application or research artifacts.

## Protected Pareto block

- `projectkoios.optimization.multiobjective.pareto` reconstructs behavior from Eugene J. Ragasa's PhD work. PyPosPack and PyMatMC2 are the authoritative historical mechanism repositories; KSD recovery copies are evidence only.
- Keep its code, architecture, provenance, adaptations, and protection rules under `multiobjective/pareto`. Do not move, split, rename, or refactor it without explicit human authorization; preserve its source, commit, blob, contributor, license, adaptation, and test mappings.

## Extraction policy

- Treat recovery commits and source drops as evidence, not release-ready implementations.
- For external or historically independent transfers, preserve the source repository, commit, path, tree/blob identity, namespace rewrite, license, notices, adaptations, and test mapping.
- ProjectKoios cross-repository movement, including KSD-to-owner extraction, is internal ownership work: retain design and verification context, but do not add cross-project provenance or `TRANSFER.toml` entries.
- Extract the smallest reusable behavior; exclude campaigns, calculators, filesystem launchers, workflow state, and legacy package structure.
- Do not vendor PyPosPack, PyMatMC2, or other material with unresolved or incompatible provenance or licensing.
- Do not introduce behaviorless universal bases or structural `Protocol` substitution. Use capability-local nominal abstract classes, closed method enums, and factories.
- Prefer immutable records and deterministic, bounded algorithms with explicit seeded randomness.
- Keep software and numerical verification distinct from scientific validation and human acceptance.

## Development

- Follow [`docs/code/python/tests.md`](docs/code/python/tests.md) and [`docs/code/python/code_smells.md`](docs/code/python/code_smells.md).
- Mirror package, module, and class ownership under indexed current-state documentation in `docs/architecture`.
- Use strict typing, deterministic sanitized fixtures, failure and awkward cases, reproducible offline builds, and wheel import checks. Require exact provenance tests only for external or historically independent transfers.
- Keep one writer per worktree. Stage exact paths, disable hooks and signing for controlled commits, and verify the staged tree before committing.
- Do not fetch, push, publish, merge, or modify another repository without explicit authorization.
