# Probability

`probability/base.py` owns finite scalar CDF, inverse-CDF, and composed distribution
contracts. `probability/distributions.py` owns the finite continuous uniform distribution.
`UniformProbabilityDistributionParameters` has one strict canonical constructor that
requires exact finite built-in float endpoints and validates without field replacement.
`UniformProbabilityDistribution` stores only that canonical parameter record. Its
properties construct immutable CDF and inverse-CDF evaluators on demand rather than
mutating frozen storage to install private caches.

`ProbabilityDistributionSampler` composes a unit-uniform source with an inverse CDF.
`SplitMix64UnitUniformSampler` is the dependency-free deterministic source;
capability-local NumPy and SciPy implementations remain optional.

`ProbabilitySamplerFactory` exposes concrete named construction methods without string
dispatch. It constructs `UniformRandomSampler`, global SciPy Gaussian KDE,
PyPosPack-lineage GMM-conditioned local KDE, and owner-authored Pareto-shell contrastive
KDE through lazy backend imports.

Probability sampling assigns no scientific meaning, units, priors, or acceptance policy.
