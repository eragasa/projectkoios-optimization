# Slice 02 independent implementation review

## Scope

The independent read-only review covered the standing-authorized generic linear least-
squares support required to keep reusable mathematics out of KSD:

- dependency-free real/complex input, result, tolerance, and nominal solver contracts;
- dense NumPy ordinary and positive-weight multiple-RHS solving;
- rank, singular-value, condition-number, residual, and failure semantics;
- final relational backend-result validation;
- lazy factory, package import, and wheel behavior;
- source-mirrored software and bounded numerical tests; and
- the boundary retaining Fourier, hopping, units, diagnostics, and scientific policy in
  KSD.

It did not assess KSD scientific validation or authorize a dependency switch, release,
commit, merge, or push.

## Resolved findings

### Truncated-system stationarity

Disposition: `NO_ACTION_REQUIRED`

Full-gradient stationarity originally rejected valid truncated-SVD solutions. Final
validation now applies that check only when no positive singular direction was discarded.
Exact-zero rank deficiency still receives stationarity validation. An order-one discarded-
direction regression fixes the intended behavior.

### Stable residual diagnostics

Disposition: `NO_ACTION_REQUIRED`

The backend originally used `numpy.linalg.norm`, which could overflow or underflow by
squaring finite residuals. Raw and weighted diagnostics now use scale-stable
`math.hypot`-based contract helpers. Regressions retain finite, nonzero norms at `1e200`
and `1e-200`.

### Rank and spectrum trust boundary

Disposition: `NO_ACTION_REQUIRED`

The owner now defines the default relative rank cutoff as
`sys.float_info.epsilon * max(m, n)` and constrains explicit cutoffs to `[0, 1)`. Final
validation checks strict-cutoff rank, singular-value shape/order/finite state, the scale-
homogeneous weighted-design Frobenius/singular-value identity, and the derived condition
number. A forged zero spectrum is rejected.

Individual singular values remain numerical decomposition-backend evidence; the final
template does not duplicate an SVD. Independent normal-equation and QR comparisons bound
the built-in backend verification.

### Overflow-safe stationarity

Disposition: `NO_ACTION_REQUIRED`

Stationarity originally formed weighted design-residual products before cancellation, so
terms near `1e310` overflowed. Validation now normalizes weighted design and residual
vectors independently, accumulates bounded inner-product terms, and translates the
absolute tolerance into scaled units. The previously failing `1e155` canceling-gradient
system now passes with a finite full-rank result.

### Smallest-subnormal tolerance boundary

Disposition: `NO_ACTION_REQUIRED`

The final re-review identified an optional one-ULP hardening at the smallest subnormal.
The lower comparison is now strict, and a focused test retains `math.ulp(0.0)` rather than
rounding that exact boundary to zero.

### Migration status

Disposition: `NO_ACTION_REQUIRED`

The completed-work inventory now identifies the owner least-squares implementation and
separates acceptance review from later consumer version binding.

## Acceptance evidence

- Full staged-isolation test suite: **149 passed**.
- Least-squares scoped suite: **28 passed**.
- Ruff checks: passed.
- Ruff formatting check: passed for **149 files**.
- Strict mypy: passed for **62 source files**.
- `git diff --check`: passed.
- Relative Markdown targets validated for **30 documents**.
- Fixed-epoch wheel and source-distribution pairs were byte-identical.
- Isolated wheel import and lazy NumPy loading passed.
- Final independent re-review found no blocking implementation issue.

Reproducible artifact digests are reported from the final external build rather than
embedded here, because this review document is itself part of the source distribution.
Those digests are validation evidence, not a published or durable dependency locator.

## Consumer caveats

- `weights` are positive objective multipliers, not pre-square-rooted values.
- Raw residuals are `fitted_values - target_matrix`.
- Raw diagnostics are distinct from weighted objective diagnostics.
- Rank and singular values describe the square-root-weighted design.
- The condition number is present only at numerical full column rank.
- `rank_tolerance=None` uses the owner-defined binary64 cutoff above.
- Full-gradient stationarity does not apply after a positive singular direction is
  intentionally discarded.
- KSD retains design construction, units, hopping reconstruction, domain diagnostics,
  route comparison, fit adequacy, independent verification, and scientific validation.

## Outcome

`ReviewOutcome: NO_BLOCKING_FINDINGS`

`OperatorRequest: NONE`
