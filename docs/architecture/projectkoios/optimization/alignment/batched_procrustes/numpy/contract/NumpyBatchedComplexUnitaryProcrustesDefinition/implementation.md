# Definition implementation

The definition is a frozen, slotted dataclass in `contract.py`. Its raw constructor accepts only canonical C-order arrays with immutable bytes backing and performs validation without replacing fields.

`from_arrays(...)` is the normal factory constructor: it validates exact NumPy arrays and exact dtypes, creates independent immutable defensive copies, and invokes the canonical constructor. `from_immutable_arrays(...)` is the explicit zero-copy factory and retains already-canonical arrays by identity.

No backend is imported or invoked. Derived shape properties read the stored immutable arrays and do not cache duplicate state. No `object.__setattr__` or mutable-then-sealed construction is used.
