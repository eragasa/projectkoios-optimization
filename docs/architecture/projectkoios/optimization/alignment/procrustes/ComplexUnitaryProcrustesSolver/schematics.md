# Generic solver schematics

```mermaid
classDiagram
    ComplexUnitaryProcrustesSolver <|-- ScalarComplexUnitaryProcrustesSolver
    ComplexUnitaryProcrustesSolver <|-- BatchedComplexUnitaryProcrustesSolver
    ScalarComplexUnitaryProcrustesSolver <|-- NumpyComplexUnitaryProcrustesSolver
    BatchedComplexUnitaryProcrustesSolver <|-- NumpyBatchedComplexUnitaryProcrustesSolver
```
