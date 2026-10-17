# Optimization identifier validation

`validation/validation.py` owns portable text, measurement-name, run-identifier, and candidate-identifier boundaries used by Optimization contracts.

Dependency-free numerical representation and range checks live in [`projectkoios.numerics.validation`](../../numerics/validation/index.md). Optimization imports those primitives in the permitted Numerics-to-Optimization dependency direction.

This module does not interpret scientific meaning. Scientific units, channel alignment, thresholds, dispositions, and acceptance remain consumer-owned.
