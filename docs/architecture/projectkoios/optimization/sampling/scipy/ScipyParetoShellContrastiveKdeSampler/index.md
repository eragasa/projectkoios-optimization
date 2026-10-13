# ScipyParetoShellContrastiveKdeSampler

`projectkoios.optimization.sampling.scipy.pareto_shells.ScipyParetoShellContrastiveKdeSampler`
is an optional SciPy implementation of owner-authored Pareto-shell contrastive sampling.

For every adaptive stage it:

1. recursively obtains shell zero and the requested dominated shells from the complete
   ordered evaluation history;
2. pulls each shell back to its exact decision vectors;
3. fits one Gaussian KDE per available shell;
4. draws a bounded candidate pool from the shell-zero KDE using an explicit seed;
5. ranks candidates by shell-zero log density minus the weighted log-sum density of
   dominated shells; and
6. returns the highest-ranked exact candidate count.

The current default uses Silverman's bandwidth, negative shell weights `(1.0, 0.5)`, and
a candidate-pool multiplier of eight. These are algorithm parameters, not scientific
acceptance thresholds. Singular shell covariance remains an explicit failure until the
historical covariance-repair behavior has been reconstructed and verified separately.

This sampler is not the historical global KDE sampler and is not represented as donor
behavior. Its dated design record is
[`docs/research/pareto-shell-contrastive-sampling.md`](../../../../../../research/pareto-shell-contrastive-sampling.md).
