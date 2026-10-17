# Pareto optimization

`projectkoios.optimization.multiobjective.pareto` owns Pareto selection, the
iterative Pareto optimizer, and exact Pareto-shell pullback. This is the canonical
location for the capability's behavior, historical authority, adaptations, and
protection rules.

## Numerical boundary

All objectives are finite ordered vectors interpreted as minimization objectives.
The package uses the following terms consistently:

- **decision vector**: one proposed point in parameter space;
- **objective vector**: the evaluated numeric image of that decision vector;
- **Pareto set**: nondominated decision vectors;
- **Pareto front**: the corresponding nondominated objective vectors; and
- **Pareto shell**: one zero-based nondominated layer peeled from an observed
  population.

Scientific names, units, direction conversion, thresholds, evaluator execution, and
acceptance remain consumer-owned. `ParetoOptimizer` receives evaluated objective
vectors through `tell`; it does not run a simulator or own workflow lifecycle.

## Historical mechanism authority

Historical authority is determined by proximity to the original mechanism, not by the
location of the most convenient recovered copy.

1. **PyPosPack and PyMatMC2 (`pymatmc2`, also called PyMC2) are the more
   authoritative historical mechanism repositories.** PyPosPack is the exact source
   authority for the selector and iterative sampling policy transferred here. PyMatMC2
   is authoritative mechanism and design context for the related historical
   multi-objective optimization lineage. No PyMatMC2 source is currently transferred
   into this package.
2. **KSD is recovery evidence, not mechanism origin or release authority.** An
   identical PyPosPack selector blob survives in KSD, but that later copy does not
   supersede PyPosPack or PyMatMC2.
3. This repository is the current authority for the adapted Project Koios Optimization
   API and its explicitly documented corrections. That current ownership does not
   rewrite historical origin.

The PyMatMC2 reference pin is repository
`https://github.com/eragasa/pymatmc2`, commit
`9d31d7fd4f8902f17864fbf391059101a3f5afda`, tree
`7773f886eaecfe919abf68d9e4f990fb398a82be`. Its provenance record identifies the
historical `pymatmc2` source as commit
`3063f031301c04e395cb9d7bc5064f2a52e1110c`, tree
`fd9788560365e93faae596ac91f2530aa6bd92a8`, and the bundled historical `mexm`
source as commit `3963a1a30ba1595fba69cd0f791dc31fe2e08f7c`, tree
`a19cd86088aad5ecb0b3bbf8a4c73e0159a22fd3`. These identities establish context;
they do not claim that PyMatMC2 contains the byte-identical selector transferred from
PyPosPack.

## Transferred PyPosPack behavior

The iterative stage policy comes from PyPosPack commit
`be453fa7191e55a0426f66e8b5b5b0b103c8b29d`, path
`pypospack/pyposmat/engines/mc_sampler_iterate.py`, blob
`a39c7ba9956d7b6fac6399d3fdf915bb3418bed8`, SHA-256
`bdaaca875f52e82fa17e440dd30a3e7dd5db8df56db2698f4f263bf723838c6e`.
The adaptation preserves interchangeable sampling stages and preceding-archive
feedback while removing filesystem, YAML, MPI, logging, calculator execution, and
string-selected sampling modes.

The performance-sensitive selector comes from the same PyPosPack commit, path
`pypospack/pareto.py`, blob `217ea0a75e60c79529f566aace35c1e14d4c627b`,
SHA-256
`dd0a541d16b34ccc803967e37471ea0f26f9068725a5c69c83454400bb0b0198`.
The source header credits Dmitriy Morozov of Lawrence Berkeley National Laboratory
with December 2016 performance improvements. The donor is BSD-2-Clause; its retained
license is `licenses/PyPosPack-BSD-2-Clause.txt` and attribution is in `NOTICE`.

The adaptive chunk-pruning loop, comparison order, early breaks, and historical
weak-dominance duplicate behavior are preserved. The sole algorithmic correction
replaces the full-batch-only `zip(*[iter(indices)] * size)` idiom with Python 3.14
`itertools.batched`, retaining the final short batch. Tests compare the port with the
minimally corrected donor and brute force, including a 501-point tail regression and a
20,000-point, ten-objective timing case.

## Owner-authored extension

Recursive shell peeling and shell-contrastive KDE sampling are owner-authored research
extensions, not recovered PyPosPack or PyMatMC2 behavior. Equal objective vectors are
grouped before invoking the protected selector so duplicates receive the same shell
rank. The dated design record is
[`docs/research/pareto-shell-contrastive-sampling.md`](../../../../../research/pareto-shell-contrastive-sampling.md).

## Protection rule

Do not move, split, rename, refactor, or alter the protected Pareto selector or optimizer
without explicit human authorization. Any performance-sensitive change requires
correctness comparison with the minimally corrected donor and timing evidence. Keep
PyPosPack/PyMatMC2 mechanism authority distinct from KSD recovery evidence and from
owner-authored shell research.

Exact machine-readable file mappings remain in
[`TRANSFER.toml`](../../../../../../TRANSFER.toml); the broader source inventory is in
[`docs/provenance/source-mapping.md`](../../../../../provenance/source-mapping.md).
