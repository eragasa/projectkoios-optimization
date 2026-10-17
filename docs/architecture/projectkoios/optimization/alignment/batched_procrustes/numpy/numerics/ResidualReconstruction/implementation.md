# ResidualReconstruction implementation

`ResidualReconstruction` is a frozen, slotted evidence record with a classmethod
constructor. The constructor uses vectorized float64 arrays, normalized scaling, and
C-order failure indexing. The class remains inside `numerics.py` and is not a façade
export.
