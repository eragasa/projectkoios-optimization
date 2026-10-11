"""Optional third-party adapter namespace.

Adapter modules are never imported by the dependency-free package root.
"""

from .quadratic import PositiveDefiniteQuadraticLoss

__all__ = ["PositiveDefiniteQuadraticLoss"]
