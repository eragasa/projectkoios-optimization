# Generic solver implementation

The generic abstract base is dependency-free and slotted. Final shared policy methods
cannot be overridden by scalar or batched subclasses. Family bases specialize definition
and result type parameters and own their execution templates.

There is no compatibility alias retaining the former scalar meaning of this name.
