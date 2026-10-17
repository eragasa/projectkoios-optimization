# Scalar solver implementation

The scalar nominal base inherits shared policy and requires concrete backends to provide
backend identity, version, and one validated complete square SVD. The public solve method
is final. `NumpyComplexUnitaryProcrustesSolver` is the reviewed implementation selected
by the scalar factory.
