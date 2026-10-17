# Implementation and failure contract

## Source organization

```text
src/python/projectkoios/optimization/quadratic/
├── box.py
│   ├── BoxQuadraticBoundStatus
│   ├── BoxQuadraticMinimum
│   ├── BoxQuadraticBackendResult
│   └── BoxQuadraticMinimizer
├── factory.py
│   ├── BoxQuadraticMinimizerMethod
│   └── BoxQuadraticMinimizerFactory
└── numpy/
    └── box.py
        ├── NumpyBoxQuadraticMinimizer
        ├── _ActiveSetKktEvidence
        ├── _active_set_kkt_evidence
        ├── _solve_free_direction
        ├── _blocking_step
        └── _advance_free_coordinates
```

Package initializers remain import-empty. `box.py` and `factory.py` import no optional
backend. The closed factory lazily imports the NumPy implementation. External nominal
subclasses may be injected directly and do not require factory registration, but the
nominal base is dense-only and imposes the same final validation on every subclass.

The NumPy backend keeps the public iteration template small by assigning one coherent
responsibility to each named internal operation. `_active_set_kkt_evidence()` reconstructs
immutable ordered decision evidence, `_solve_free_direction()` owns the principal-system
solve and its exception translation, `_blocking_step()` owns the largest feasible step,
and `_advance_free_coordinates()` owns the in-place feasible update and deterministic
endpoint activation. These operations preserve the documented coordinate order and do
not introduce an alternative objective or tolerance policy.

`BoxQuadraticMinimizer` accepts an exact `ConstrainedQuadraticTolerance` and a positive
built-in unsigned-32-bit `maximum_iterations` at construction and exposes both through
read-only properties. The default limit is 10,000. Concrete subclasses implement
read-only `backend_identifier` and `backend_version` properties plus
`_solve_validated()`. The protected method returns the public immutable backend-extension contract
`BoxQuadraticBackendResult`, containing only finite candidate coordinates, positive
iteration count, nonnegative active-set-change count, and nonnegative linear-solve count.
The final template validates backend metadata as nonempty trimmed built-in strings and
treats it as trusted API-level declaration, not authenticated identity.

`BoxQuadraticMinimizerMethod` has the single closed member
`NUMPY_DENSE = "numpy-dense"`. `BoxQuadraticMinimizerFactory.create(method, *,
tolerance=DEFAULT_CONSTRAINED_QUADRATIC_TOLERANCE, maximum_iterations=10000)` validates
through the nominal constructor and returns `NumpyBoxQuadraticMinimizer` lazily.

## Result record

`BoxQuadraticMinimum` is immutable and contains:

- `coordinates`: the finite minimizing tuple;
- `squared_loss`: fresh evaluation at those coordinates;
- `bound_statuses`: one `free`, `lower`, `upper`, or `fixed` enum per coordinate;
- `gradient`: the fresh full gradient $2Q(x-c)$;
- `kkt_tolerance_bounds`: one row-local applied acceptance bound per coordinate;
- `maximum_kkt_violation`: maximum reconstructed coordinate violation;
- `maximum_kkt_ratio`: maximum coordinate violation divided by its own bound;
- `iteration_count`: active-set states checked through the accepted result;
- `active_set_change_count`: bound activations and releases after initialization;
- `linear_solve_count`: free-subspace Newton systems solved;
- `maximum_iterations`: the applied positive execution limit;
- `backend_identifier`; and
- `backend_version`.

The record does not retain coordinate names, units, consumer thresholds, or scientific
dispositions. Its sole constructor accepts canonical exact tuples and built-in floats and
validates them without `object.__setattr__` or any other postconstruction replacement.
Its local structural, KKT, execution-count, and backend-identity invariants remain in the
constructor rather than being spread across one-call private validators and private
transfer records. The constructor has a narrow C901 exemption so the complexity metric
does not force artificial validation abstractions.

## Final-template validation

`BoxQuadraticMinimizer.solve()` is final. It validates exact nominal input types,
positive matching dimensions, dense nonsparse loss state with an exact tuple matrix,
concrete public backend-handoff type, iteration/count relations, coordinate feasibility,
and backend metadata. It then independently:

1. evaluates the quadratic;
2. classifies bounds from exact returned coordinates and endpoints;
3. reconstructs the full gradient from the original dense matrix;
4. computes every coordinate's KKT violation and row-local tolerance bound; and
5. rejects any coordinate outside its declared optimality tolerance.

The backend result supplies only candidate coordinates and execution counts. It does
not supply public loss, active-bound classification, gradient, KKT, or backend-identity evidence for the
final template to trust. Backend identity comes from the concrete subclass's abstract
property implementation and remains a trusted declaration.

## Dense backend

`NumpyBoxQuadraticMinimizer` requires `loss.sparse is False` and an immutable tuple
matrix. It converts the already validated quadratic and bounds to binary64 arrays, starts
at the clipped center, and applies the bounded primal active-set transitions defined in
`numerics.md`. Each nonstationary free subspace uses `numpy.linalg.solve` on one positive-
definite principal matrix. The final template evaluates the accepted point through the
existing loss facade; the backend does not maintain a second objective formula.

The backend identifier is `projectkoios.active-set.numpy`; version strings have the form
`1/numpy-<version>`. These are trusted API-level declarations, not authenticated package
or BLAS/LAPACK provenance.

## Failure classification

- `TypeError`: wrong nominal records, wrong exact container types, Boolean/string numeric
  coercion at existing record boundaries, malformed backend return types, or malformed
  public result fields.
- `ValueError`: zero or mismatched dimension, sparse/dense representation mismatch,
  invalid relational state, infeasible backend output, or KKT rejection.
- `FloatingPointError`: nonfinite or overflowing backend/shared arithmetic and nonfinite
  quadratic evaluation during the solve. After validated candidate shape, finiteness,
  dimension, and feasibility, arithmetic `ValueError` from the reused loss evaluation is
  translated narrowly to a chained `FloatingPointError` during final-template
  reevaluation.
- `RuntimeError`: `numpy.linalg.LinAlgError` from an admitted free-subspace solve
  (chained), pre-optimality stagnation, or exhaustion of `maximum_iterations`.
- undeclared exceptions propagate unchanged.

No failure or lack of a returned point is an incompatibility, infeasibility, or scientific
model-adequacy disposition.

## Qualification

Mapped contract, result, factory, wheel, integration, and numerical tests live under:

- `tests/projectkoios/optimization/quadratic/box/`; and
- `tests/projectkoios/optimization/quadratic/numpy/box/`.

The capability-local `tests/projectkoios/optimization/quadratic/box/fixtures/` package
owns named analytical case records instead of repeating numeric setup literals. These
fixtures span all bound statuses and directly exercise the public-result vector, KKT,
and execution validators, the immutable value
factory, and the backend KKT-evidence, free-direction, blocking-step,
and coordinate-advance operations in addition to covering 1-D, interior, faces, edges,
corners, fixed coordinates, parity with the
existing exact 2-D rectangle minimizer, coordinate permutation, ill-conditioned admitted
matrices, deterministic repetition, simultaneous-hit sets, smallest-index release,
exact-endpoint retention, interior snapping precedence, invalid inputs, iteration
exhaustion, stagnation/failure behavior, subclass postconditions, arithmetic failures,
lazy factory import, isolated wheel behavior, independent SciPy minimization, and a
100-decimal-digit high-precision reference. The disclosed consumer inventory 1, 2, 3, 4,
6, 12, 27, and 36 is exercised with its symmetric bound. Tests specifically exercise
lower, upper, free, and fixed KKT rules; prevent a huge correctly signed active gradient
from masking a free stationarity defect; reject sparse state at the nominal boundary;
reject malformed private backend results and metadata; and cover documented exception
translations. Strict mypy includes both mapped test directories.

Retained source-bound numerical evidence is documented in
`benchmarks/box_quadratic_minimization.md` and
`benchmarks/results/box_quadratic_minimization_calibration.json`. Independent architecture,
implementation, and numerical-evidence reviews found no remaining blocking issue after
the snapping and retained-conditioning follow-ups. These gates establish software and
bounded numerical verification only, not scientific validation or KSD threshold meaning.
