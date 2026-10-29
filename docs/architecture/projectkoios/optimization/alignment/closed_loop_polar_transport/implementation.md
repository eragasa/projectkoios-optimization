# Closed-loop polar transport implementation

## Source ownership

```text
projectkoios/optimization/alignment/
├── closed_loop_polar_transport.py
├── numpy/
│   └── closed_loop_polar_transport.py
└── scipy/
    └── closed_loop_polar_transport.py
```

The dependency-free module owns enums, including the explicit principal-phase branch,
ordered link diagnostics, and the bounded typed backend failure. The NumPy module owns
exact complex128 inventory, matrix, right-action, definition, and self-validating result
records. It uses capability-specific aliases from
`projectkoios.numerics.numpy.typing`. The SciPy module owns only execution and complex
Schur decomposition.

`ScipyClosedLoopPolarFrameTransportSolver` requires an explicit
`NumpyComplexUnitaryProcrustesSolver` at construction. All decomposition, progression,
and distribution behavior is owned by that stateful solver instance. The implementation
contains no empty action constructor, static method, or module-level numerical helper.
Parent package initializers remain import-empty.

## Verification map

Dependency-free unit tests cover link evidence and typed backend failures. NumPy unit
tests cover exact dtype and shape, defensive immutable ownership, feasible rank,
orthonormality, cutoff correlation, progression order, both phase branches,
right-action, endpoint-mismatch orientation, correction-power relations, and rejection of
inconsistent public results.

SciPy integration tests cover scalar and composite transport, noncommuting local gauges,
nondiagonal closure, cutoff equality, neighbor and closure deficiency, explicit
unattempted links, exact `-1` endpoint mismatch under both branches, direct correction
identities, metric discrimination between maximum-entry and normalized Frobenius values,
and typed SVD and Schur nonconvergence or invalid-evidence mapping.

Repository-wide lint, format, strict typing, software tests, package build, and wheel
import remain release-independent verification gates.
