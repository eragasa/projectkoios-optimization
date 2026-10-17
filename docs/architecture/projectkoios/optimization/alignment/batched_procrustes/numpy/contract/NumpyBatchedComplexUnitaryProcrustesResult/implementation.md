# Result implementation

The result is a keyword-only frozen, slotted dataclass in `contract.py`. It normalizes
and validates all evidence before construction-scoped `object.__setattr__` installs
immutable byte-backed arrays. No callable postconstruction installer exists.

Rank summaries are derived properties. The fail-closed accessor returns the existing
completion array without copying only when all items are full rank.
