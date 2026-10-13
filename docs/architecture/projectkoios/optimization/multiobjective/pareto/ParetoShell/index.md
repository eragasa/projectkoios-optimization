# ParetoShell

`projectkoios.optimization.multiobjective.pareto.shells.ParetoShell` is an immutable
zero-based nondominated shell record. It retains original observation indices and the
corresponding ordered decision and objective vectors.

`pareto_shells(...)` recursively removes each selected objective shell before selecting
the next. Its exact index pullback avoids fitting an inverse objective-to-decision model.
Equal objective vectors are grouped and assigned to the same shell before the preserved
PyPosPack selection kernel is called.

This capability is owner-authored research behavior documented in
[`docs/research/pareto-shell-contrastive-sampling.md`](../../../../../../research/pareto-shell-contrastive-sampling.md),
not behavior attributed to the PyPosPack donor.
