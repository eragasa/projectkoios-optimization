# Definition implementation

The definition is a frozen, slotted dataclass in `contract.py`. Construction requires
exact NumPy arrays and exact tolerance ownership, makes C-order defensive copies backed
by immutable bytes, and installs them only within `__post_init__` after validation.

No backend is imported or invoked. Derived shape properties read the stored immutable
arrays and do not cache duplicate state.
