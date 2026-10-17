# PyMatMC2-to-Optimization migration

This task covers a dedicated reusable Monte Carlo module sourced from PyMatMC2.

## Status

- State: source audit not started.
- Source repository: `https://github.com/eragasa/pymatmc2`.
- Reference commit: `9d31d7fd4f8902f17864fbf391059101a3f5afda`.
- Reference tree: `7773f886eaecfe919abf68d9e4f990fb398a82be`.
- Transfer state: no PyMatMC2 source transferred.

## Target ownership

- Reusable Monte Carlo contracts.
- Explicit immutable Monte Carlo state.
- Deterministic seeded transition behavior.
- Proposal and acceptance records.
- Bounded stepping behavior.
- State restoration required for exact continuation.

## Source-audit inventory

- `src/pymatmc2/mc2_base.py`.
- `src/pymatmc2/configuration.py`.
- `src/pymatmc2/data.py`.
- `src/pymatmc2/concentration_matrix.py`.
- `src/pymatmc2/chemical_potential.py`.
- `src/pymatmc2/mc2_mpi_serial.py`: behavior evidence only until execution concerns are
  separated.
- `src/pymatmc2/mc2_mpi_spawn.py`: execution code excluded.
- `src/pymatmc2/mc2_mpi_subpoll.py`: execution code excluded.

## Exclusions

- MPI launch and process management.
- CLI entry points.
- Filesystem and database integration.
- Scheduler integration.
- Atomistic simulation execution.
- Material models and scientific acceptance policy.

## Required evidence

- [ ] Identify the exact Monte Carlo mechanism commit.
- [ ] Pin every transferred source path, blob, and SHA-256.
- [ ] Resolve license and notice requirements.
- [ ] Separate numerical state transitions from MPI and file orchestration.
- [ ] Define deterministic random-state ownership.
- [ ] Define checkpoint-free intrinsic state and restoration.
- [ ] Add source-mirrored unit and numerical tests.
- [ ] Add the accepted mapping to `TRANSFER.toml` only after implementation approval.
