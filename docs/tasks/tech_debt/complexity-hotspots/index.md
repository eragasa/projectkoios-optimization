# Reduce concentrated cyclomatic complexity

## Status

- State: open technical debt.
- Baseline owner commit: `335411f4a7ce3180062f8b14d29641b16bc468fa`.
- Classification: maintainability debt, not a demonstrated numerical defect.

## Problem

The repository passes its configured Ruff checks, but an explicit McCabe scan with a maximum complexity of ten reports 18 functions above that threshold:

```text
ruff check --select C90 src/python
```

The current findings are:

| Complexity | Callable |
|---:|---|
| 20 | `optimization/quadratic/numpy/box.py::_solve_validated` |
| 17 | `numerics/finite_periodic/batched_finite_series.py::evaluate` |
| 16 | `optimization/clustering/gmm.py::__post_init__` |
| 15 | `optimization/alignment/procrustes.py::ComplexUnitaryProcrustesResult.__post_init__` |
| 14 | `optimization/quadratic/box.py::BoxQuadraticMinimum.__post_init__` |
| 14 | `optimization/selection/harmonic.py::PeriodicHarmonicMaximum.__post_init__` |
| 13 | `numerics/linalg/hermitian_spectrum.py::__post_init__` |
| 13 | `optimization/quadratic/box.py::solve` |
| 13 | `optimization/quadratic/constrained.py::solve` |
| 12 | `numerics/linalg/sparse_hermitian_spectrum.py::__init__` |
| 12 | `optimization/algorithms/coordinate.py::_prepare_iteration` |
| 12 | `optimization/least_squares/linear.py::__post_init__` |
| 12 | `optimization/least_squares/linear.py::solve` |
| 12 | `optimization/multiobjective/pareto/algorithm.py::pareto` |
| 12 | `optimization/quadratic/scipy/quadratic.py::__init__` |
| 11 | `numerics/linalg/hermitian_spectral_subspace.py::_matrix` |
| 11 | `numerics/linalg/sparse_hermitian_spectrum.py::__post_init__` |
| 11 | `optimization/sampling/scipy/gmm_kde.py::__init__` |

The findings combine algorithmic branching with validation-heavy relational checks. The metric alone does not establish incorrect behavior, but concentrated branches make review and failure-policy verification harder.

## Recommended sequence

1. Split the complexity-20 and complexity-17 numerical kernels into named preparation, decision, and evidence-construction operations.
2. Split validation-heavy `__post_init__` methods into owner-specific structural and relational validators.
3. Refactor solver branches only where the extracted operation has a coherent mathematical responsibility.
4. Re-run the complexity scan and document any deliberately retained exceptions.

## Constraints

- Preserve exception classes and documented inducing conditions.
- Preserve deterministic operation order where it affects floating-point results.
- Do not move scientific thresholds, dispositions, or interpretation into Numerics or Optimization.
- Do not introduce behaviorless universal helper classes merely to lower a metric.
- Refactoring provenance-protected PyPosPack/PyMatMC2-lineage code requires separate explicit authorization.

## Definition of done

- [ ] Every listed callable either satisfies the accepted complexity bound or has a documented, reviewed exception.
- [ ] Extracted helpers have precise ownership, complete NumPy-style documentation, and direct tests.
- [ ] Numerical regression tests preserve existing values and failure classifications.
- [ ] Full Ruff, formatting, strict mypy, test, compile, lock, reproducible-build, and wheel-smoke qualification passes.
- [ ] Independent review finds no blocking behavior or ownership regression.
