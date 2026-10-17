# Bounded parameter spaces

`ParameterBound` and `ParameterSpace` define finite ordered closed boxes used by bounded
numerical algorithms. Equal endpoints represent fixed coordinates, and an empty tuple is
a valid zero-dimensional space.

The space records define numerical admissibility only. Scientific parameter meaning,
units, normalization, and conversion into the ordered coordinates remain caller-owned.

`ParameterBound` uses canonical frozen construction. Its dataclass constructor accepts
exact finite built-in floats and validates without field replacement. No alternate
normalization constructor is exposed. Neither record uses reflective attribute access or
postconstruction mutation.
