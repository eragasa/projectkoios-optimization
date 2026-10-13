# Shared legacy validation primitives

`validation/validation.py` retains the established portable scalar and identifier
boundaries used by pre-existing Optimization contracts. It normalizes supported real
values to finite binary64, enforces exact built-in integer ranges, and preserves the
existing identifier limits and candidate-ID construction.

This module is public and directly tested after the source-tree reorganization; it is
not a generic semantic-validation service and must not be extended to interpret
scientific meaning. New numerical capabilities should keep their invariant checks with
the owning mathematical object rather than route inputs through this module.

Scientific units, names, channel alignment, schema meaning, thresholds, and acceptance
remain consumer-owned.
