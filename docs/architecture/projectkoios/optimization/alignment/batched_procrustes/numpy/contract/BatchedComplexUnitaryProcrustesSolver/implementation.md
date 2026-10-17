# Batched nominal solver implementation

The nominal batch solver is slotted and abstract. It inherits final shared tolerance,
roundoff, and backend-evidence policy and adds a final vectorized `rank_cutoffs` method.
`NumpyBatchedComplexUnitaryProcrustesSolver` supplies the concrete solve operation.
