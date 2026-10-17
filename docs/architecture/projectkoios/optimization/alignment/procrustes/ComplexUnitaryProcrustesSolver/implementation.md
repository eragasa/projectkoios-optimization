# Shared solver-policy implementation

The shared policy owner is dependency-free and slotted. Final tolerance, rank, residual,
and backend-evidence methods cannot be overridden by scalar or batched subclasses. It
does not expose a generic execution method or generic return type. Family-specific
nominal bases own their concrete definition and result contracts and execution templates.

There is no compatibility alias retaining the former scalar meaning of this name.
