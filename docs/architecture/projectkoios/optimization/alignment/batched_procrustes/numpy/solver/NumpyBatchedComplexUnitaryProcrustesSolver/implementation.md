# Solver implementation

The stateless slotted solver in `solver.py` consumes only the exact definition. Nonempty
execution constructs all cross matrices vectorially, makes one public batched NumPy SVD
call, validates backend evidence, and assembles the public result. Empty execution avoids
SVD entirely.
