# Generic solver consumer needs

Consumers need scalar and batched solvers to share mathematical policy without sharing
consumer semantics. The generic base receives no units, frames, lattice topology,
scientific thresholds, persistence rules, or dispositions. Those remain in PhysKit or
KSD around the family-specific solve.
