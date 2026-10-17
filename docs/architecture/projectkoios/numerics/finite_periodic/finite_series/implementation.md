# Finite Fourier-series implementation organization

## Source layout

```text
src/python/projectkoios/numerics/
├── __init__.py
├── py.typed
└── finite_periodic/
    ├── __init__.py
    ├── finite_series.py
    └── numpy/
        ├── __init__.py
        └── finite_series.py
```

`finite_periodic/finite_series.py` defines the dependency-free records, nominal abstract
evaluator, closed built-in method enum, and lazy factory. The NumPy backend is isolated
in `finite_periodic/numpy/finite_series.py`. Package initializers remain import-empty,
and `py.typed` marks the distributed Numerics package as typed.

The broader planned `transforms.py` modules remain deferred. They retain the separate
future forward/inverse finite-mesh transform, truncation, and Parseval-diagnostic scope.
The implemented finite-series module contains only the accepted evaluation capability;
it is not a general transform utility collection.

## Dependency-free records

`FiniteFourierSeriesEvaluationDefinition` and
`FiniteFourierSeriesEvaluationResult` are frozen, slotted, immutable records. Their
constructors validate exact built-in tuple and scalar contracts, copy no backend arrays,
and import neither NumPy nor a consumer package. Both constructors validate their
already-canonical immutable fields without replacing them after construction.

The definition retains an explicit positive dimension so empty point and term collections
remain unambiguous. The result retains dimension, point count, and term count so an
evaluator can validate backend correlation without attaching consumer identities.

## Nominal evaluator

`FiniteFourierSeriesEvaluator` is a capability-local abstract class with one final public
template method:

```python
from abc import ABC, abstractmethod
from typing import final


class FiniteFourierSeriesEvaluator(ABC):
    @final
    def evaluate(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult: ...

    @abstractmethod
    def _evaluate_validated(
        self,
        definition: FiniteFourierSeriesEvaluationDefinition,
        /,
    ) -> FiniteFourierSeriesEvaluationResult:
        raise NotImplementedError
```

The final method must:

1. require the exact nominal definition type;
2. return exact empty-point and empty-term results without importing a backend;
3. call the protected backend operation for nontrivial definitions;
4. propagate every undeclared backend exception unchanged;
5. require the exact result type;
6. verify dimension, point count, term count, output length, and finite values; and
7. return no backend object or consumer metadata.

Built-in implementations are final subclasses. External implementations opt in through
nominal inheritance and must pass the owner conformance suite. Structural protocols,
duck-typed substitution, automatic discovery, and registry mutation are forbidden.

## NumPy implementation

`NumpyFiniteFourierSeriesEvaluator` is the initial concrete implementation. It imports
NumPy only from its backend module and follows the exact scalar accumulation order in
[numerics.md](numerics.md). It may use NumPy scalar types internally but must return the
backend-neutral result record containing built-in complex values.

The backend must not:

- allocate a full point-by-term phase matrix as an undocumented requirement;
- use a reduction whose summation order is unspecified;
- coalesce or reorder modes;
- mutate caller-owned data;
- attach units or lattice meaning; or
- claim cross-platform bitwise reproducibility.

A later vectorized or accelerated backend requires its own concrete class, numerical
parity policy, memory bound, and architecture review.

## Deferred backend triggers

The initial term-list representation is already sparse in the relevant sense: the
evaluator visits only the supplied modes and does not allocate a dense point-by-term
operator. A SciPy sparse backend would not make the generally dense complex-exponential
operator sparse and is not justified for this slice.

Reconsider a separate backend only when a concrete workload requires one of these
capabilities:

- FFT evaluation on a complete regular mesh;
- NUFFT evaluation for very large irregular point sets;
- chunked or vectorized evaluation for measured throughput needs; or
- a genuinely sparse reusable operator representation.

Each such backend requires a separate contract for accepted inputs, ordering and
numerical parity, memory bounds, failure behavior, and backend-specific dependencies.

## Factory

`FiniteFourierSeriesMethod` is a closed enum whose initial member is
`NUMPY_DENSE`. `FiniteFourierSeriesEvaluatorFactory.create()` accepts only the exact enum,
lazily imports `NumpyFiniteFourierSeriesEvaluator`, and returns the nominal evaluator
type.

The factory performs no numerical work, contains no consumer-specific default, and is
not an extension registry. External subclasses are injected directly by their consumers.

## Test mapping

Tests mirror core and backend source ownership:

```text
tests/projectkoios/numerics/finite_periodic/
├── finite_series/
│   ├── test__FiniteFourierSeriesEvaluationDefinition__unit.py
│   ├── test__FiniteFourierSeriesEvaluationResult__unit.py
│   ├── test__FiniteFourierSeriesEvaluator__contract.py
│   ├── test__FiniteFourierSeriesEvaluatorFactory__unit.py
│   └── test__FiniteFourierSeriesWheel__integration.py
└── numpy/
    └── finite_series/
        ├── test__NumpyFiniteFourierSeriesEvaluator__integration.py
        └── test__NumpyFiniteFourierSeriesEvaluator__numerical.py
```

The owner contract suite must be reusable by nominal external subclasses. Numerical tests
must include independent bounded references rather than reproducing backend control flow.
Consumer-shaped regression fixtures remain owner-native and must not import PhysKit or
KSD.

## Packaging and dependency requirements

- The core wheel imports `projectkoios.numerics.finite_periodic.finite_series` without
  importing NumPy.
- Constructing the `NUMPY_DENSE` method is the first action that requires NumPy.
- `projectkoios.numerics` never imports Optimization, PhysKit, or KSD.
- PhysKit and KSD consume an exact reviewed release or commit and use narrow adapters.
- The wheel must pass isolated core-import and NumPy-backend smoke tests.
- Source distributions and wheels must be reproducible under the repository build
  contract.

## Acceptance gate

An owner revision is not ready for downstream binding until the scoped capability
passes:

1. contract, unit, numerical, adversarial-subclass, and integration tests;
2. Ruff lint and formatting checks;
3. strict mypy over the complete owned source and mapped tests;
4. documentation-link validation;
5. reproducible wheel and source-distribution builds;
6. isolated wheel import before and after NumPy backend construction;
7. an independent implementation review; and
8. a downstream PhysKit adapter review against an exact owner revision.

These gates establish bounded software and numerical behavior only. Commit, push,
release, publication, and downstream dependency changes remain separately authorized.
