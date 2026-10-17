# Result implementation

The result is a keyword-only frozen, slotted dataclass in `contract.py`. Its raw constructor accepts only canonical C-order arrays with immutable bytes backing and validates structural and relational evidence without replacing fields.

`from_arrays(...)` defensively copies ordinary exact arrays before canonical construction. `from_immutable_arrays(...)` is the explicit zero-copy path and retains every already-canonical array by identity. Neither path uses `object.__setattr__`, a mutable sealing phase, or a postconstruction installer.

Rank summaries are derived properties. The fail-closed accessor returns the existing completion array without copying only when all items are full rank.
