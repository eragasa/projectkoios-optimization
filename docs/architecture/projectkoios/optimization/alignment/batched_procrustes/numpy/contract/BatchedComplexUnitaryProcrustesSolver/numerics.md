# Batched nominal solver numerics

The nominal base broadcasts the shared additive coefficients over per-item largest
singular values, produces exact float64 cutoffs, and fails closed if cutoff arithmetic
becomes nonfinite. Family solvers use strict greater-than comparison for rank.
