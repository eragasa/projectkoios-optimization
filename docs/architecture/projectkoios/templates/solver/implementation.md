# Solver implementation template

## Source layout

```text
<namespace>/<capability>/
├── __init__.py
├── <problem>.py
├── factory.py
└── <backend>/
    ├── __init__.py
    └── <problem>.py
```

## Abstract class

The capability module defines one nominal abstract class:

```python
from abc import ABC, abstractmethod
from typing import final


class ProblemSolver(ABC):
    @final
    def solve(self, inputs: ProblemInputs, /) -> ProblemResult:
        normalized = self._validate(inputs)
        result = self._solve_validated(normalized)
        return self._validate_result(result)

    @abstractmethod
    def _solve_validated(self, inputs: NormalizedInputs, /) -> ProblemResult:
        raise NotImplementedError
```

The exact signature remains capability-specific. The final shared public method enforces
invariants; the protected abstract method isolates backend numerical work. Built-in
concrete implementations are also marked `final`; external implementations extend only
the capability abstract class.

## Factory

```python
class ProblemMethod(Enum):
    NUMPY = "numpy"


class ProblemSolverFactory:
    @staticmethod
    def create(method: ProblemMethod, /) -> ProblemSolver: ...
```

Factory behavior:

- accepts the exact method enum;
- lazily imports the selected optional backend;
- returns the nominal abstract type;
- constructs only supported owner implementations; and
- contains no numerical algorithm.

## External SDK implementations

An external implementation:

- explicitly subclasses the capability abstract class;
- implements the protected backend operation;
- returns the exact owner result type;
- passes the owner conformance suite; and
- is injected directly into higher-level APIs or constructed by the external package.

No structural duck typing, automatic discovery, or unbounded global registry is used.

## Conformance tests

Every solver capability supplies reusable tests for:

- nominal subclass identity;
- shared validation;
- result invariants;
- deterministic ordering;
- failure semantics;
- optional-backend isolation; and
- agreement with independent numerical references.

Backend-specific tests extend the conformance suite with numerical acceptance and
performance evidence.

## Restrictions

- No universal `projectkoios.Solver` base class.
- No `solvers` or `fitters` junk-drawer package.
- No structural `Protocol` substitution.
- No algorithm in the factory.
- No backend object in public immutable results.
