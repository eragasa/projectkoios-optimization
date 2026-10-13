# Bounded parameter spaces

`ParameterBound` and `ParameterSpace` define finite ordered closed boxes used by bounded
numerical algorithms. Equal endpoints represent fixed coordinates, and an empty tuple is
a valid zero-dimensional space.

The space records define numerical admissibility only. Scientific parameter meaning,
units, normalization, and conversion into the ordered coordinates remain caller-owned.
