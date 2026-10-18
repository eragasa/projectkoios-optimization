# Batched nominal solver numerics

The nominal base broadcasts the shared absolute and scaled-relative components over
per-item largest singular values, combines them by the declared additive or maximum rule,
produces exact float64 cutoffs, and fails closed if cutoff arithmetic becomes nonfinite. Family solvers use strict greater-than comparison for rank.
