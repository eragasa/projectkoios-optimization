# Finite box quadratic-minimization numerical calibration

The mapped numerical test
`test__NumpyBoxQuadraticMinimizer__numerical.py` exercises 64 deterministic dense
positive-definite quadratics. Forty-eight seeded cases cover dimensions one through six;
sixteen consumer-shaped full-rank and diagonal cases cover active dimensions 1, 2, 3, 4,
6, 12, 27, and 36 with the disclosed symmetric bound. It compares the bounded active-set
result with an independently invoked SciPy L-BFGS-B minimization using an analytical
gradient. The same retained calibration converts an ill-conditioned four-dimensional
binary64 fixture exactly through `float.as_integer_ratio()` and compares it with a
100-decimal-digit `mpmath` face calculation; a separate mapped test repeats that check in
the ordinary focused and full suites.

Regenerate the retained environment and maximum-error evidence with:

```bash
PROJECTKOIOS_BOX_QUADRATIC_CALIBRATION_OUTPUT=\
benchmarks/results/box_quadratic_minimization_calibration.json \
PYTHONPATH=src/python .venv/bin/python -m pytest -q \
tests/projectkoios/optimization/quadratic/numpy/box/\
test__NumpyBoxQuadraticMinimizer__numerical.py::\
test_seeded_and_consumer_dimensions_agree_with_scipy_reference
```

The JSON binds SHA-256 identities for the dependency-free final template, concrete NumPy
backend, existing loss facade and NumPy loss backend, constrained-quadratic tolerance,
parameter-space bounds, and numerical-test file. It records deterministic dimensions,
disclosed consumer dimension and bound inventories, and seed; Python, NumPy, SciPy, and mpmath
versions; NumPy linked-library configuration; the requested decimal bound and its
normalized binary64 value; maximum generated matrix condition number; maximum coordinate
and loss differences from the independent SciPy result; maximum accepted KKT ratio; and
high-precision condition number, execution diagnostics, KKT ratio, and coordinate/loss
differences. The high-precision check is source-bound
by the same numerical-test hash and is also run explicitly by the focused and full suites.

The retained record is one-platform bounded numerical evidence. It is not a performance
claim, cross-platform stability guarantee, exact arithmetic certificate, scientific
validation, or evidence that the default iteration limit is suitable for every admitted
binary64 quadratic.
