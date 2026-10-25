# `ScipyCsrSpectralNormControls` numerics

`tolerance` controls SciPy ARPACK convergence and the owner residual cutoff. Zero selects
SciPy's machine-precision convergence mode. Values greater than or equal to one are
rejected because they do not define a meaningful relative qualification boundary.

`maximum_iterations` is passed directly to `svds(maxiter=...)` on the iterative branch.
It remains recorded for direct small-shape problems even though no iteration occurs.

`krylov_dimension=None` delegates selection to pinned SciPy. Explicit integers use
shape- and dtype-dependent validation in the evaluator:

- direct `min(shape) <= 2`: only `None`;
- real iterative input: `2 <= krylov_dimension < min(shape)`;
- complex iterative input: `3 <= krylov_dimension < min(shape)`.

At the private SciPy call boundary, the evaluator maps
`krylov_dimension -> ncv`. `ncv` is not a public control field or alias.

`seed` initializes one local NumPy generator. It does not read or modify process-global
random state. Real and complex deterministic starts derive from that generator.

These values govern software computation. They are not scientific tolerances, uncertainty
parameters, or acceptance decisions.
