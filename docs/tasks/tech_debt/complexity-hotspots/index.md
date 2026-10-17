# Reduce concentrated cyclomatic complexity

## Status

- State: in progress.
- Baseline owner commit: `335411f4a7ce3180062f8b14d29641b16bc468fa`.
- Current local owner commit: `d77dd89dc58adf441d212226061dfd71b8707ce9`.
- Current remote owner commit: `f3a295fde4aeb905414da6375ec4f2794480628d`.
- Classification: maintainability debt, not a demonstrated numerical defect.

## Problem

The repository passes its configured Ruff checks, but the baseline explicit McCabe scan with a maximum complexity of ten reported 18 functions above that threshold:

```text
ruff check --select C90 src/python
```

The completed refactors split concentrated owners into named mathematical and validation operations with direct tests. Their original findings are now eliminated:

| Baseline complexity | Callable | Extracted ownership |
|---:|---|---|
| 20 | `optimization/quadratic/numpy/box.py::_solve_validated` | KKT evidence, free-face solve, blocking step, and coordinate advance |
| 17 | `numerics/finite_periodic/batched_finite_series.py::evaluate` | trivial values, metadata correlation, value validation, and backend evidence |
| 15 | `optimization/alignment/procrustes.py::ComplexUnitaryProcrustesResult.__post_init__` | structure, rank, residual, transformation, and backend evidence |
| 14 | `optimization/quadratic/box.py::BoxQuadraticMinimum.__post_init__` | immutable value factory plus vector, KKT, and execution evidence |
| 14 | `optimization/selection/harmonic.py::PeriodicHarmonicMaximum.__post_init__` | immutable value factory plus scalar, relational, and argument evidence |

The current worktree scan reports the following 13 remaining findings:

| Complexity | Callable |
|---:|---|
| 16 | `optimization/clustering/gmm.py::__post_init__` |
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

## Current slice evidence

- Dense box helper and mapped owner tests: 44 passed.
- Batched finite-series mapped owner tests: 65 passed.
- Direct extracted-operation tests use named analytical cases and cover KKT decision evidence, principal-system direction, blocking steps, coordinate activation, trivial values, structural metadata, value surfaces, scalar Procrustes rank/residual/transformation relations, box-minimum vector/KKT/execution relations, periodic-harmonic scalar/status/argument relations, and backend evidence.
- Full repository qualification passes with 873 tests, Ruff check and format, strict mypy over 184 source files, compileall, and `uv lock --check`.
- The explicit complexity scan now reports 13 findings and no longer reports either numerical kernel or any of the three completed validation-heavy constructors.
- Reproducible artifacts and an isolated wheel smoke pass: wheel SHA-256 `af67b8bbef64b3ef231dc06d37af6281144324201600f3f68b09f25c5b3728d7`; sdist SHA-256 `c048ae20fcf802e9bafe72a5b56f25052300ed66ed1a2dacae4725c2864da042`.
- Independent review reported `NO_BLOCKING_FINDINGS` after the periodic-harmonic fixture was corrected to use an independent analytical oracle.

## Recommended sequence

1. **Completed in the current worktree:** split the complexity-20 and complexity-17 numerical kernels into named preparation, decision, and evidence-construction operations.
2. **Completed in the current worktree:** split scalar Procrustes, box-minimum, and periodic-harmonic validation-heavy `__post_init__` methods into owner-specific structural and relational validators.
3. Refactor the next unprotected finding only where the extracted operation has a coherent mathematical responsibility; the provenance-protected GMM validator remains excluded without separate authorization.
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
