# Implementation

The frozen record groups the complete singular spectrum, square left singular vectors, and adjoint right singular vectors. Backend subclasses return this record instead of an incidental positional tuple. The final scalar solver validates every field and relation before constructing a result.
