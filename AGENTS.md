# Project Koios optimization

This repository owns reusable mathematical numerics and optimization for Project Koios. The Python distribution is `projectkoios-optimization`; mathematical numerical modules live under `projectkoios.numerics`, and optimization policy lives under `projectkoios.optimization`.

## Ownership

- Own reusable transformations, decompositions, mathematical geometry, numerical analysis, immutable records, diagnostics, and capability-local backend implementations under `projectkoios.numerics`.
- Own every reusable problem whose public contract optimizes a function or set of functions under `projectkoios.optimization`, including fitting, constrained minimization, Procrustes, assignment, selection, clustering, sampling policy, and search.
- Own behaviorful optimizer contracts, candidate proposal and observation behavior, explicit immutable algorithm state, loss aggregation, and optimization results under `projectkoios.optimization`.
- Physical and scientific interpretation is the responsibility of each consumer. Keep units, material meaning, study-specific thresholds, evidence interpretation, and scientific acceptance out of both owned numerical namespaces.
- Keep forward evaluators and results handlers behind narrow injected ports. This repository may define the information those ports exchange; it does not own simulation execution or scientific-domain models.
- `projectkoios-workflow` owns execution authorization, dispatch, retries, persistence, and lifecycle orchestration.
- Simulation and atomistic implementations belong in `projectkoios-simulations` under their neutral, integration, or adapter namespace as appropriate.
- Material-specific campaigns remain application/research-owner artifacts.

## Protected Pareto block

- `projectkoios.optimization.multiobjective.pareto` reconstructs optimization behavior
  originating in Eugene J. Ragasa's PhD work. PyPosPack and PyMatMC2 are the more
  authoritative historical mechanism repositories; KSD recovery copies are evidence,
  not mechanism origin or release authority.
- Keep Pareto behavior, provenance, adaptations, and protection rules under the
  `multiobjective/pareto` source and architecture namespaces.
- Do not move, split, rename, or refactor this protected block without explicit human
  authorization. Preserve exact behavior-specific source, commit, blob, contributor,
  license, adaptation, and test mappings.

## Extraction policy

- Recovery commits and source drops are evidence, not canonical or release-ready implementations.
- Preserve source repository, exact commit, original path, tree/blob identity, namespace rewrite, license, notices, adaptations, and test mapping for external or historically independent source transfers.
- Movement among ProjectKoios repositories, including KSD-to-owner extraction, is internal ownership work: retain design and verification context, but do not add cross-project provenance or `TRANSFER.toml` entries.
- Extract the smallest reusable behavior. Do not copy unrelated campaigns, calculators, filesystem launchers, workflow state, or legacy package structure.
- Do not vendor PyPosPack, PyMatMC2, or other material with unresolved or incompatible provenance/licensing.
- Do not introduce a behaviorless universal base solely to unify types.
- Do not use structural `Protocol` types for solver, fitter, optimizer, or backend substitution. Use capability-local nominal abstract classes, closed method enums, and factories.
- Prefer immutable records and deterministic, bounded algorithms. Make randomness explicit and seeded.
- Distinguish software verification and numerical verification from scientific validation and human acceptance.

## Development

- Mirror package/module/class ownership under `docs/architecture` using indexed current-state documentation.
- Use strict typing, deterministic sanitized fixtures, failure and awkward cases, reproducible offline builds, and wheel import checks. Require exact provenance tests only for external or historically independent source transfers.
- Keep one writer per worktree. Use exact-path staging, disable hooks/signing for controlled commits, and verify the staged tree before commit.
- Do not fetch, push, publish, merge, or modify another repository without explicit authorization.
