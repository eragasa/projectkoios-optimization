# Finite Fourier-series schematics

## Status

These diagrams describe the current implemented finite-series capability and its
permitted downstream composition. Prospective backends are not shown.

## Evaluation flow

```mermaid
flowchart LR
    D["FiniteFourierSeriesEvaluationDefinition<br/>dimension, points, modes, coefficients"]
    V["shared contract validation"]
    E["nominal evaluator"]
    N["NumPy backend<br/>ordered complex128 sum"]
    C["shared result validation"]
    R["FiniteFourierSeriesEvaluationResult<br/>ordered complex values"]

    D --> V --> E --> N --> C --> R
```

The backend evaluates

$$
y_i=\sum_j c_j\exp\!\left(+2\pi i\,x_i\cdot m_j\right)
$$

without normalization, inference, sorting, or coefficient coalescence.

## Nominal SDK boundary

```mermaid
classDiagram
    class FiniteFourierSeriesEvaluator {
        <<abstract>>
        +evaluate(definition) FiniteFourierSeriesEvaluationResult
        #_evaluate_validated(definition) FiniteFourierSeriesEvaluationResult
    }
    class NumpyFiniteFourierSeriesEvaluator {
        #_evaluate_validated(definition) FiniteFourierSeriesEvaluationResult
    }
    class ExternalFiniteFourierSeriesEvaluator {
        #_evaluate_validated(definition) FiniteFourierSeriesEvaluationResult
    }
    class FiniteFourierSeriesEvaluatorFactory {
        +create(method) FiniteFourierSeriesEvaluator
    }
    class FiniteFourierSeriesMethod {
        <<enumeration>>
        NUMPY_DENSE
    }

    FiniteFourierSeriesEvaluator <|-- NumpyFiniteFourierSeriesEvaluator
    FiniteFourierSeriesEvaluator <|-- ExternalFiniteFourierSeriesEvaluator
    FiniteFourierSeriesEvaluatorFactory --> FiniteFourierSeriesMethod
    FiniteFourierSeriesEvaluatorFactory --> NumpyFiniteFourierSeriesEvaluator
```

The factory controls only built-in construction. External implementations use nominal
inheritance and direct injection; they do not mutate the factory.

## Dependency graph

```mermaid
flowchart TB
    NUM["projectkoios.numerics<br/>unit-free finite Fourier series"]
    OPT["projectkoios.optimization<br/>objective construction and selection"]
    PHY["projectkoios.physkit<br/>hopping models, units, semantic adapters"]
    KSD["ksdft2effmass<br/>experiment composition and interpretation"]

    NUM --> OPT
    NUM --> PHY
    NUM --> KSD
    OPT --> KSD
    PHY --> KSD
```

PhysKit is not a required facade. KSD may call the Numerics capability directly when it
already owns the scientific mapping. Numerics imports none of the downstream packages.

## Physics-semantic adaptation

```mermaid
flowchart LR
    M["PhysKit ScalarHoppingModel"]
    A["PhysKit adapter<br/>extract unit-free points, modes, coefficients"]
    F["Numerics finite Fourier-series evaluator"]
    W["PhysKit result<br/>model identity and energy unit"]

    M --> A --> F --> W
```

The adapter, not Numerics, owns model identity, basis, energy reference, units,
Hermiticity diagnostics, and physical interpretation.

## KSD composition

```mermaid
flowchart LR
    C["KSD frozen coordinates and coefficient meaning"]
    F["Numerics evaluator"]
    Y["ordered unit-free values"]
    Q["Optimization quadratic construction"]
    E["KSD evidence and interpretation"]

    C --> F --> Y --> Q --> E
```

The arrows do not authorize KSD to duplicate the Fourier kernel or Optimization to infer
scientific meaning from its numeric inputs.

## Boundary behavior

```mermaid
flowchart TD
    I["validated definition"] --> P{"point count = 0?"}
    P -- yes --> ER["empty result"]
    P -- no --> T{"term count = 0?"}
    T -- yes --> Z["one exact complex zero per point"]
    T -- no --> B["ordered backend evaluation"]
    B --> F{"finite correlated result?"}
    F -- yes --> R["return immutable result"]
    F -- no --> X["fail closed"]
```

Empty behavior is part of the public contract. Nonfinite arithmetic and malformed backend
results fail closed rather than returning partial or sanitized values.
