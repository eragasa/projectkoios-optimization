# Scientific and ownership boundary

## Reusable mathematical statement

The capability solves one strictly convex, unit-free mathematical problem: minimize an
already constructed positive-definite completed-square quadratic over an already declared
finite closed box. Positive definiteness gives a unique mathematical minimizer. Compactness
of the box gives existence independently of whether the unconstrained center lies inside.

The returned point and KKT diagnostics are reusable numerical evidence. They establish
neither a scientific model claim nor the acceptability of any consumer-selected box.

## Ownership

Optimization owns:

- deterministic selection of the minimizing coordinates;
- deterministic bounded active-set iteration for the supported dense representation;
- equality-constrained free-subspace solves and feasible blocking steps;
- active-bound classification;
- gradient and KKT-residual reconstruction;
- numerical tolerance and failure semantics; and
- backend, iteration, active-set-change, and linear-solve metadata.

A scientific consumer owns:

- coordinate meaning and units;
- construction and normalization of the quadratic;
- selection and justification of lower and upper bounds;
- scientific thresholds and resolutions;
- experiment sample roles and retained evidence;
- dispositions and interpretation; and
- execution, persistence, retries, and publication decisions.

PhysKit continues to own physical quantities and unit-bearing adapters. KSD may consume
the operation directly when its unit-free boundary fits, but this repository does not own
a KSD facade or evidence hierarchy.

## Verification boundary

Software tests verify nominal records, deterministic ordering, validation, packaging, and
failure translation. Bounded numerical tests compare with independent SciPy minimization
and high-precision calculations. These checks are numerical verification on sanitized
fixtures, not scientific validation of a hopping model, admissible region, or model-
adequacy conclusion.
